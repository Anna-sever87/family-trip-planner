"""Actual bot bodies, real city snapshots, no token/working DB/HTTP/AI."""

import ast
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

from city_context import (CITY_EDIT_NOTE, CONTEXT_ERROR, CONTEXT_KEY,
                          check_context_places, load_city_context, prepare_stored_city_context,
                          validate_city_edit_context, attach_city_presentation, city_presentation_tips)
from generation_feedback import generation_failure_message
from schemas import Itinerary, ItineraryDay, TripRequest
from weather import CityResolutionError
from test_bot_generation_outcome import city_fixture


def function(name, namespace):
    source = Path(__file__).with_name("bot.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    body = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    body.decorator_list = []
    exec(compile(ast.Module(body=[body], type_ignores=[]), "bot.py", "exec"), namespace)
    return namespace[name]


def itinerary(*, city=False, source=False):
    return Itinerary(title="Synthetic route", summary="", city_schedule=city, source_schedule=source,
                     days=[ItineraryDay(day=1, date="2099-06-01", theme="", items=[]),
                           ItineraryDay(day=2, date="2099-06-02", theme="", items=[], day_mode="free")])


class BotCityEditGateTests(unittest.TestCase):
    def setUp(self):
        self.request, self.route, self.context = city_fixture()
        self.trip = {"request": {**self.request.model_dump(), CONTEXT_KEY: deepcopy(self.context)},
                     "itinerary": self.route.model_dump(), "revision": 1}
        self.events = []
        self.ns = {
            "Optional": Optional, "TripRequest": TripRequest, "Itinerary": Itinerary,
            "deepcopy": deepcopy, "CityResolutionError": CityResolutionError,
            "CITY_EDIT_NOTE": CITY_EDIT_NOTE, "CONTEXT_KEY": CONTEXT_KEY, "CONTEXT_ERROR": CONTEXT_ERROR,
            "load_city_context": load_city_context, "check_context_places": check_context_places,
            "validate_city_edit_context": validate_city_edit_context,
            "prepare_stored_city_context": Mock(wraps=prepare_stored_city_context),
            "bot": Mock(), "db": Mock(), "logger": Mock(), "threading": Mock(),
            "_clean_edit_request": Mock(side_effect=lambda text: text),
            "_reserve_generation_job": Mock(return_value=True),
            "_claim_route_ai_request": Mock(side_effect=lambda *args: self.events.append("claim") or
                                              SimpleNamespace(status="ok", claim_id="fake")),
            "_finish_generation_job": Mock(), "_finish_ai_claim": Mock(),
            "edit_generation_job": Mock(), "edit_itinerary": Mock(),
            "generation_failure_message": generation_failure_message,
            "format_itinerary": Mock(return_value="Updated route"), "send_parts": Mock(),
        }
        self.ns["bot"].send_message.side_effect = lambda _chat, text: self.events.append(text)
        self.ns["db"].get_trip.return_value = self.trip
        self.ns["db"].update_trip_plan.return_value = True

    def handle(self, day=1):
        function("handle_edit", self.ns)(SimpleNamespace(
            chat=SimpleNamespace(id=100), text=f"/edit 7 {day} Перенеси обед"))

    def worker(self, *, context=True, revision=1):
        kwargs = {"city_map_context": deepcopy(self.context), "expected_revision": revision} if context else {}
        function("edit_generation_job", self.ns)(100, 7, 1, "Перенеси обед", "fake", **kwargs)

    def successful_result(self):
        result = SimpleNamespace(itinerary=self.route.model_copy(deep=True), city_map_context=deepcopy(self.context),
                                 weather=None, weather_text="", problems=[], attempts=1,
                                 prompt_tokens=1, completion_tokens=1)
        result.itinerary.days[0].items[-2].time = "13:30"
        result.itinerary.days[0].items[-1].time = "14:30"
        self.ns["edit_itinerary"].return_value = result
        return result

    def test_program_and_free_city_edits_preflight_before_quota_and_pass_bound_revision(self):
        for day in (1, 2):
            with self.subTest(day=day):
                self.setUp()
                before = deepcopy(self.trip)
                self.handle(day)
                self.ns["prepare_stored_city_context"].assert_called_once()
                self.assertLess(self.events.index(CITY_EDIT_NOTE), self.events.index("claim"))
                thread = self.ns["threading"].Thread.call_args.kwargs
                self.assertEqual(thread["args"], (100, 7, day, "Перенеси обед", "fake"))
                self.assertEqual(thread["kwargs"], {"city_map_context": self.context, "expected_revision": 1})
                self.ns["threading"].Thread.return_value.start.assert_called_once()
                self.assertEqual(self.trip, before)

    def test_invalid_target_and_instruction_do_not_lookup_or_claim(self):
        self.handle(99)
        self.ns["prepare_stored_city_context"].assert_not_called()
        self.ns["_claim_route_ai_request"].assert_not_called()
        self.ns["_clean_edit_request"].side_effect = ValueError("Некорректный текст")
        self.handle()
        self.ns["prepare_stored_city_context"].assert_not_called()
        self.ns["_claim_route_ai_request"].assert_not_called()

    def test_failed_recovery_does_not_claim_quota_or_leak_exception(self):
        for error in (ValueError(CONTEXT_ERROR), RuntimeError("token=DO_NOT_LEAK https://private.invalid"),
                      CityResolutionError("Кемер", reason="ambiguous")):
            with self.subTest(error=type(error).__name__):
                self.setUp()
                self.ns["prepare_stored_city_context"].side_effect = error
                self.handle()
                self.ns["bot"].send_message.assert_called_with(100, CONTEXT_ERROR)
                self.ns["_reserve_generation_job"].assert_not_called()
                self.ns["_claim_route_ai_request"].assert_not_called()
                self.ns["threading"].Thread.assert_not_called()
                self.ns["logger"].exception.assert_not_called()
                self.assertNotIn("DO_NOT_LEAK", str(self.ns["logger"].mock_calls))

    def test_foreign_or_missing_trip_does_not_disclose_metadata_or_claim_quota(self):
        self.ns["db"].get_trip.return_value = None
        self.handle()
        self.ns["db"].get_trip.assert_called_once_with(7, 100)
        self.assertIn("Такой поездки у вас нет", self.ns["bot"].send_message.call_args.args[1])
        self.ns["prepare_stored_city_context"].assert_not_called()
        self.ns["_claim_route_ai_request"].assert_not_called()

    def test_legacy_and_source_trips_keep_existing_edit_flow_and_thread_tuple(self):
        for source in (False, True):
            with self.subTest(source=source):
                self.setUp()
                stored = itinerary(source=source).model_dump()
                stored.pop("city_schedule")
                self.trip["itinerary"] = stored
                self.handle()
                self.ns["prepare_stored_city_context"].assert_not_called()
                self.ns["_claim_route_ai_request"].assert_called_once_with(100, "edit", 7)
                self.assertNotIn("kwargs", self.ns["threading"].Thread.call_args.kwargs)

    def test_worker_direct_city_call_missing_preflight_fails_before_editor_or_write(self):
        self.worker(context=False)
        self.ns["bot"].send_message.assert_called_with(100, CONTEXT_ERROR)
        self.ns["edit_itinerary"].assert_not_called()
        self.ns["db"].update_trip_plan.assert_not_called()
        self.ns["_finish_ai_claim"].assert_called_once_with("fake", False)
        self.ns["_finish_generation_job"].assert_called_once_with(100)

    def test_worker_revision_change_fails_before_editor_without_recovery(self):
        self.trip["revision"] = 2
        self.worker()
        self.assertIn("AI не запускался", self.ns["bot"].send_message.call_args.args[1])
        self.ns["edit_itinerary"].assert_not_called()
        self.ns["prepare_stored_city_context"].assert_not_called()
        self.ns["db"].update_trip_plan.assert_not_called()

    def test_worker_rebinds_loaded_request_and_places_not_just_revision(self):
        for mutation in ("request", "coordinates", "context"):
            with self.subTest(mutation=mutation):
                self.setUp()
                if mutation == "request":
                    self.trip["request"]["budget"] += 1
                elif mutation == "coordinates":
                    self.trip["itinerary"]["days"][0]["items"][0]["lat"] += .001
                else:
                    self.context["candidates"][0]["candidate_id"] = "osm-node-999"
                self.worker()
                self.ns["edit_itinerary"].assert_not_called()
                self.ns["db"].update_trip_plan.assert_not_called()
                self.ns["prepare_stored_city_context"].assert_not_called()

    def test_successful_city_edit_persists_context_and_cas_without_mutating_old_trip(self):
        result = self.successful_result()
        before = deepcopy(self.trip)
        self.worker()
        self.ns["edit_itinerary"].assert_called_once()
        self.assertEqual(self.ns["edit_itinerary"].call_args.kwargs,
                         {"city_map_context": self.context, "prior_weather": None})
        call = self.ns["db"].update_trip_plan.call_args
        self.assertEqual(call.kwargs, {"expected_revision": 1})
        self.assertEqual(call.args[2][CONTEXT_KEY], result.city_map_context)
        self.assertIsNot(call.args[2][CONTEXT_KEY], result.city_map_context)
        self.assertEqual(call.args[3]["days"][1], before["itinerary"]["days"][1])
        self.assertEqual(self.trip, before)
        self.ns["prepare_stored_city_context"].assert_not_called()
        self.ns["_finish_ai_claim"].assert_called_once_with("fake", True)

    def test_worker_forwards_copy_of_weather_from_fresh_owner_bound_trip(self):
        weather = {"source": "forecast", "city": "Тестгород", "days": [
            {"date": self.request.start_date, "tmin": 10, "tmax": 15, "rain_mm": 8}]}
        self.trip["request"]["weather"] = deepcopy(weather)
        result = self.successful_result()
        def edit(*args, **kwargs):
            self.assertEqual(kwargs["prior_weather"], weather)
            self.assertIsNot(kwargs["prior_weather"], self.trip["request"]["weather"])
            kwargs["prior_weather"]["days"][0]["rain_mm"] = 0
            return result
        self.ns["edit_itinerary"].side_effect = edit
        self.worker()
        self.ns["db"].get_trip.assert_called_once_with(7, 100)
        self.ns["edit_itinerary"].assert_called_once()
        self.assertEqual(self.trip["request"]["weather"], weather)

    def test_worker_persists_refreshed_presentation_but_rejects_changed_map_evidence(self):
        from city_evidence import city_managed_tips
        pool, _ = load_city_context(self.context, self.request)
        self.route.tips = city_managed_tips(self.route, pool, self.request)
        self.context = attach_city_presentation(self.context, self.route, pool, self.request)
        self.trip["request"][CONTEXT_KEY] = deepcopy(self.context)
        self.trip["itinerary"] = self.route.model_dump()
        result = self.successful_result()
        result.itinerary.days[0].items[0].duration_min = 100
        result.itinerary.tips = city_managed_tips(result.itinerary, pool, self.request)
        result.city_map_context = attach_city_presentation(self.context, result.itinerary, pool, self.request)
        self.assertNotEqual(result.city_map_context["presentation"], self.context["presentation"])
        self.worker()
        saved = self.ns["db"].update_trip_plan.call_args.args[2][CONTEXT_KEY]
        self.assertEqual(saved, result.city_map_context)
        self.assertTrue(city_presentation_tips(saved, result.itinerary, pool, self.request))

    def test_worker_rejects_valid_replacement_evidence_or_malformed_manifest_without_save(self):
        for change in ("center", "captured_at", "manifest", "mutate_argument"):
            with self.subTest(change=change):
                self.setUp()
                result = self.successful_result()
                before = deepcopy(self.trip)
                if change == "center":
                    result.city_map_context["resolved_city"]["lat"] += .001
                elif change == "captured_at":
                    result.city_map_context["captured_at"] = "2099-01-01T00:00:00+00:00"
                elif change == "manifest":
                    result.city_map_context["presentation"] = {"version": 1, "managed_tip_sha256": ["0" * 64]}
                else:
                    def edit(*args, **kwargs):
                        kwargs["city_map_context"]["resolved_city"]["lat"] += .001
                        result.city_map_context = kwargs["city_map_context"]
                        return result
                    self.ns["edit_itinerary"].side_effect = edit
                self.worker()
                self.ns["db"].update_trip_plan.assert_not_called()
                self.ns["_finish_ai_claim"].assert_called_once_with("fake", False)
                self.assertEqual(self.trip, before)

    def test_result_missing_context_or_downgraded_city_flag_cannot_overwrite(self):
        for mutation in ("context", "flag", "place"):
            with self.subTest(mutation=mutation):
                self.setUp()
                result = self.successful_result()
                if mutation == "context":
                    result.city_map_context = None
                elif mutation == "flag":
                    result.itinerary.city_schedule = False
                else:
                    result.itinerary.days[0].items[0].lat += .001
                self.worker()
                self.ns["db"].update_trip_plan.assert_not_called()
                self.ns["_finish_ai_claim"].assert_called_once_with("fake", False)

    def test_city_worker_roundtrip_uses_only_temporary_sqlite_and_preserves_other_trip(self):
        import db
        with tempfile.TemporaryDirectory(prefix="bot-city-context-") as directory, ExitStack() as stack:
            path = (Path(directory) / "isolated.db").resolve()
            connect = sqlite3.connect
            def isolated_connect(database, *args, **kwargs):
                if kwargs.get("uri") or Path(database).resolve() != path:
                    raise AssertionError("SQLite outside disposable test database")
                return connect(database, *args, **kwargs)
            stack.enter_context(patch.object(db, "DB_PATH", str(path)))
            stack.enter_context(patch.object(sqlite3, "connect", side_effect=isolated_connect))
            for owner, name in ((socket.socket, "connect"), (socket, "getaddrinfo")):
                stack.enter_context(patch.object(owner, name, side_effect=AssertionError("No network")))
            for target in ("places.geocode", "places._load_cache", "city_context.recover_city_context"):
                stack.enter_context(patch(target, side_effect=AssertionError("No lookup or recovery in worker")))
            db.init_db()
            trip_id = db.save_trip(100, self.trip["request"], self.trip["itinerary"])
            other_id = db.save_trip(200, self.trip["request"], self.trip["itinerary"])
            db.save_checklist(trip_id, [("Документы", "Паспорта")])
            old = db.get_trip(trip_id, 100)
            other = db.get_trip(other_id, 200)
            self.ns["db"] = db
            result = self.successful_result()
            function("edit_generation_job", self.ns)(100, trip_id, 1, "Перенеси обед", "fake",
                city_map_context=deepcopy(self.context), expected_revision=old["revision"])
            saved = db.get_trip(trip_id, 100)
            self.assertEqual(saved["revision"], old["revision"] + 1)
            self.assertEqual(saved["request"][CONTEXT_KEY], result.city_map_context)
            self.assertEqual(saved["itinerary"]["days"][1], old["itinerary"]["days"][1])
            self.assertEqual(db.get_trip(trip_id, 100), saved)
            self.assertIsNone(db.get_trip(trip_id, 200))
            self.assertEqual(db.get_trip(other_id, 200), other)
            self.assertEqual(len(db.get_checklist(trip_id)), 1)
            self.ns["edit_itinerary"].assert_called_once()
            self.ns["_finish_ai_claim"].assert_called_once_with("fake", True)

    def test_creation_footer_exposes_edit_with_pool_limitation_and_preserves_other_links(self):
        import test_bot_generation_outcome as fixtures
        for city, source in ((True, False), (False, False), (False, True)):
            with self.subTest(city=city, source=source):
                fixture = fixtures.BotGenerationOutcomeTests()
                fixture.setUp()
                if city:
                    fixture.request, fixture.result.itinerary, fixture.result.city_map_context = city_fixture()
                else:
                    fixture.result.itinerary = itinerary(source=source)
                fixture.run_job()
                text = fixture.namespace["bot"].send_message.call_args.args[1]
                self.assertIn("/edit 99 1 текст", text)
                self.assertEqual(CITY_EDIT_NOTE in text, city)
                for action in ("/checklist 99", "/pdf 99", "/reminders 99", "/trips", "/memory"):
                    self.assertIn(action, text)


if __name__ == "__main__":
    unittest.main()
