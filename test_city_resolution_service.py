"""City preflight rejects before quota/memory; late edit failures preserve data."""

import asyncio
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import Mock, patch

with patch("dotenv.load_dotenv", return_value=False):
    import db
    import web_app
    import web_service

from generation_feedback import generation_failure_message
from schemas import Itinerary, ItineraryDay, TripRequest
from test_bot_generation_outcome import _job
from weather import CityResolutionError
import weather


class CityResolutionServiceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="city-resolution-service-"))
        self.stack.enter_context(patch.object(db, "DB_PATH", str(Path(directory) / "synthetic.db")))
        db.init_db()
        self.owner = 123
        db.save_user(self.owner, "Synthetic", 2, [9], "EUR", pace="спокойный", wishes="Original preference")
        self.request = TripRequest(
            destination="Кемер, Турция", start_date="2030-11-03", end_date="2030-11-03",
            adults=2, children_ages=[], budget=45000.0, currency="RUB",
            trip_goal="Познакомиться с городом", interest_categories=["main_sights"],
        )
        self.current = Itinerary(title="Synthetic", summary="", days=[
            ItineraryDay(day=1, date=self.request.start_date, theme="Synthetic", items=[]),
        ])
        self.trip_id = db.save_trip(self.owner, self.request.model_dump(), self.current.model_dump())
        db.save_checklist(self.trip_id, [("Synthetic", "Unchanged")])
        self.before = self.saved_data()
        self.generate = self.mock(web_service, "generate_itinerary")
        self.edit = self.mock(web_service, "edit_itinerary")
        with patch.object(weather, "find_city", return_value={"name": "Кемер", "lat": 36.6, "lon": 30.56,
                          "country": "Турция", "country_code": "TR", "admin1": "Анталья", "geoname_id": 308220}):
            self.resolved = weather.resolve_city_for_planning(self.request.destination)
        self.resolve = self.mock(web_service, "resolve_city_for_planning", return_value=self.resolved)
        self.mock(web_service, "_daily_ai_limit", return_value=30)
        self.recall = self.mock(web_service.vector_memory, "recall_family_memories", return_value=[])
        self.mock(web_service.vector_memory, "format_memory_context", return_value="")
        self.memory_write = self.mock(web_service, "_save_trip_memory_safely")

    def mock(self, owner, name, **kwargs):
        return self.stack.enter_context(patch.object(owner, name, **kwargs))

    def saved_data(self):
        with db.connect() as connection:
            return {table: [tuple(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1")]
                    for table in ("users", "trips", "checklist_items", "memory_items")}

    def assert_failed_without_changes(self, count=1):
        self.assertEqual(self.saved_data(), self.before)
        self.memory_write.assert_not_called()
        with db.connect() as connection:
            rows = connection.execute("SELECT status, finished_at FROM ai_requests").fetchall()
        self.assertEqual(len(rows), count)
        self.assertTrue(all(row["status"] == "failed" and row["finished_at"] for row in rows))

    def test_web_creation_returns_clarification_for_all_city_reasons(self):
        codes = {"invalid_context": "city_invalid", "invalid_response": "city_unavailable",
                 "context_mismatch": "city_not_found", "ambiguous": "city_ambiguous",
                 "search_limit": "city_invalid", "not_found": "city_not_found", "unavailable": "city_unavailable"}
        for reason, code in codes.items():
            with self.subTest(reason=reason):
                error = CityResolutionError(self.request.destination, reason=reason)
                self.resolve.side_effect = error
                with self.assertRaises(web_service.TripCreateValidationError) as raised:
                    web_service.create_trip(self.owner, self.request)
                self.assertIn("Уточните город", str(raised.exception))
                self.assertIn("Новая поездка не сохранена", str(raised.exception))
                self.assertIn("попытка не списана", str(raised.exception))
                self.assertEqual(raised.exception.code, code)
        self.assertEqual(self.resolve.call_count, len(codes))
        self.generate.assert_not_called()
        self.recall.assert_not_called()
        self.assert_failed_without_changes(0)

    def test_moscow_provider_ambiguity_preserves_all_data_and_zero_attempts(self):
        request = self.request.model_copy(update={"destination": "Москва, Россия"})
        self.resolve.side_effect = weather.resolve_city_for_planning
        rows = [
            {"id": 524901, "name": "Москва", "country": "Россия", "country_code": "RU",
             "admin1": "Москва", "feature_code": "PPLC", "latitude": 55.75, "longitude": 37.62},
            {"id": 524900, "name": "Москва", "country": "Россия", "country_code": "RU",
             "admin1": "Тверская область", "feature_code": "PPL", "latitude": 56.90, "longitude": 32.20},
        ]
        with patch.object(weather, "_get_json", return_value={"results": rows}) as provider, \
                patch.object(db, "claim_ai_request", side_effect=AssertionError("Preflight must precede quota")) as claim:
            with self.assertRaises(web_service.TripCreateValidationError) as raised:
                web_service.create_trip(self.owner, request)
        self.assertEqual(raised.exception.code, "city_ambiguous")
        self.assertEqual(raised.exception.destination_options,
                         ("Москва, Москва, Россия", "Москва, Тверская область, Россия"))
        self.resolve.assert_called_once_with(request.destination)
        provider.assert_called_once()
        claim.assert_not_called()
        self.recall.assert_not_called()
        self.generate.assert_not_called()
        self.assert_failed_without_changes(0)

    def test_successful_preflight_context_is_reused_once_after_quota(self):
        self.generate.return_value = SimpleNamespace(problems=["Synthetic route rejection"],
                                                     generation_status="insufficient_data")
        with self.assertRaises(web_service.TripCreateValidationError) as raised:
            web_service.create_trip(self.owner, self.request)
        self.assertEqual(raised.exception.code, "insufficient_data")
        self.resolve.assert_called_once_with(self.request.destination)
        self.recall.assert_called_once()
        self.generate.assert_called_once()
        context = self.generate.call_args.kwargs["resolved_city"]
        self.assertIs(context, self.resolved)
        self.assertEqual(weather.validated_planning_city(context, self.request.destination)["geoname_id"], 308220)
        self.assert_failed_without_changes(1)

    def test_regional_creation_never_calls_city_resolver(self):
        request = self.request.model_copy(update={"destination": "Байкал, Россия", "destination_kind": "region"})
        self.generate.return_value = SimpleNamespace(problems=["Synthetic route rejection"],
                                                     generation_status="insufficient_data")
        with self.assertRaises(web_service.TripCreateValidationError):
            web_service.create_trip(self.owner, request)
        self.resolve.assert_not_called()
        self.assertNotIn("resolved_city", self.generate.call_args.kwargs)
        self.assertTrue(self.generate.call_args.kwargs["ground_candidates"])
        self.assert_failed_without_changes(1)

    def test_web_edit_returns_clarification_and_keeps_old_trip(self):
        self.edit.side_effect = CityResolutionError(self.request.destination, reason="ambiguous")
        with self.assertRaises(ValueError) as raised:
            web_service.edit_trip_day(self.owner, self.trip_id, 1, "Перенеси обед")
        self.assertIn("Уточните город", str(raised.exception))
        self.assertIn("Старый маршрут сохранён без изменений", str(raised.exception))
        self.assert_failed_without_changes()

    def inline_api(self):
        async def call(function, *args):
            return function(*args)
        self.mock(web_app.anyio.to_thread, "run_sync", side_effect=call)

    def test_creation_endpoint_is_422_not_generic_502(self):
        self.inline_api()
        self.resolve.side_effect = CityResolutionError(self.request.destination, reason="context_mismatch")
        payload = web_app.TripCreateRequest(**self.request.model_dump())
        context = SimpleNamespace(state=SimpleNamespace(account_key=db.get_account_key(self.owner)))
        with self.assertRaises(web_app.HTTPException) as raised:
            asyncio.run(web_app.create_trip(payload, context, tg_id=self.owner))
        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail["code"], "city_not_found")
        self.assertEqual(raised.exception.detail["clarification"],
                         {"kind": "city", "options": [], "attempt_used": False})
        self.generate.assert_not_called()
        self.recall.assert_not_called()
        self.assert_failed_without_changes(0)

    def test_legacy_creation_endpoint_returns_only_safe_clarification_options_without_charge(self):
        self.inline_api()
        self.resolve.side_effect = CityResolutionError("private-input", reason="ambiguous", suggestions=[
            "Москва, Москва, Россия", "<script>, Province, Country", "Москва, Тверская область, Россия"])
        payload = web_app.TripCreateRequest(**self.request.model_dump())
        context = SimpleNamespace(state=SimpleNamespace(account_key=db.get_account_key(self.owner)))
        with self.assertRaises(web_app.HTTPException) as raised:
            asyncio.run(web_app.create_trip(payload, context, tg_id=self.owner))
        detail = raised.exception.detail
        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(set(detail), {"message", "code", "clarification"})
        self.assertEqual(detail["code"], "city_ambiguous")
        self.assertEqual(detail["clarification"], {"kind": "city", "attempt_used": False, "options": [
            "Москва, Москва, Россия", "Москва, Тверская область, Россия"]})
        self.assertNotIn("private-input", str(detail))
        self.assertNotIn("<script>", str(detail))
        self.generate.assert_not_called()
        self.recall.assert_not_called()
        self.assert_failed_without_changes(0)

    def test_edit_endpoint_is_422_with_original_route_preserved(self):
        self.inline_api()
        self.edit.side_effect = CityResolutionError(self.request.destination, reason="ambiguous")
        with self.assertRaises(web_app.HTTPException) as raised:
            asyncio.run(web_app.edit_day(self.trip_id, 1,
                web_app.DayEditRequest(instruction="Перенеси обед"), tg_id=self.owner))
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("Уточните город", raised.exception.detail)
        self.assertIn("Старый маршрут сохранён без изменений", raised.exception.detail)
        self.assert_failed_without_changes()

    def bot_namespace(self):
        error = CityResolutionError(self.request.destination, reason="ambiguous")
        memory = Mock()
        memory.recall_family_memories.return_value = []
        memory.format_memory_context.return_value = ""
        return {
            "Optional": Optional, "TripRequest": TripRequest, "Itinerary": Itinerary,
            "CityResolutionError": CityResolutionError,
            "bot": Mock(), "db": db, "logger": Mock(), "vector_memory": memory,
            "_memory_query": Mock(return_value="Synthetic"),
            "city_candidates_enabled": Mock(return_value=False),
            "generate_itinerary": Mock(side_effect=error), "edit_itinerary": Mock(side_effect=error),
            "generation_failure_message": generation_failure_message,
            "_remember_trip_safely": Mock(), "format_itinerary": Mock(), "send_parts": Mock(),
            "_finish_ai_claim": Mock(side_effect=lambda claim, success: db.finish_ai_request(claim, succeeded=success)),
            "_finish_generation_job": Mock(),
        }

    def test_bot_creation_clarifies_city_and_finishes_claim_without_writes(self):
        namespace = self.bot_namespace()
        claim = db.claim_ai_request(self.owner, "2030-11-01", "plan", "telegram", daily_limit=30)
        _job(namespace)(self.owner, self.owner, self.request, claim.claim_id)
        message = namespace["bot"].send_message.call_args.args[1]
        self.assertIn("Уточните город", message)
        self.assertIn("Новая поездка не сохранена", message)
        namespace["generate_itinerary"].assert_called_once()
        namespace["city_candidates_enabled"].assert_called_once_with()
        namespace["logger"].exception.assert_not_called()
        namespace["_remember_trip_safely"].assert_not_called()
        namespace["_finish_generation_job"].assert_called_once_with(self.owner)
        namespace["_finish_ai_claim"].assert_called_once_with(claim.claim_id, False)
        self.assert_failed_without_changes()

    def test_bot_edit_clarifies_city_and_finishes_claim_without_writes(self):
        namespace = self.bot_namespace()
        claim = db.claim_ai_request(self.owner, "2030-11-01", "edit", "telegram", trip_id=self.trip_id, daily_limit=30)
        _job(namespace, "edit_generation_job")(
            self.owner, self.trip_id, 1, "Перенеси обед", claim.claim_id,
        )
        message = namespace["bot"].send_message.call_args.args[1]
        self.assertIn("Уточните город", message)
        self.assertIn("Старый маршрут сохранён без изменений", message)
        namespace["logger"].exception.assert_not_called()
        namespace["_finish_ai_claim"].assert_called_once_with(claim.claim_id, False)
        namespace["_finish_generation_job"].assert_called_once_with(self.owner)
        self.assert_failed_without_changes()


if __name__ == "__main__":
    unittest.main()
