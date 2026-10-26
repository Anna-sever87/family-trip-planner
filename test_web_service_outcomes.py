"""Offline regressions for rejected, committed and unrenderable web operations."""

import asyncio
import json
from contextlib import ExitStack
from types import SimpleNamespace
import socket
import unittest
from unittest.mock import patch

import web_app
import web_service
from schemas import TripRequest


class WebServiceOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(
            web_service.db, "connect", side_effect=AssertionError("Real DB access forbidden")
        ))
        original_connect = socket.socket.connect

        def offline_connect(sock, address):
            # Windows asyncio uses a loopback socket pair internally.
            if isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}:
                return original_connect(sock, address)
            raise AssertionError("External network forbidden")

        self.stack.enter_context(patch.object(socket.socket, "connect", offline_connect))
        self.request = TripRequest(
            destination="Воркута, Россия", start_date="2030-11-03", end_date="2030-11-03",
            budget=45000, travel_cost=15000, lodging_cost=20000, currency="RUB",
        )
        self.trip = {"request": self.request.model_dump(), "revision": 1}
        self.current = SimpleNamespace(days=[SimpleNamespace(day=1)])
        self.result = SimpleNamespace(
            problems=[], itinerary=SimpleNamespace(model_dump=lambda: {"days": []}),
            weather_text="", weather={}, attempts=1, prompt_tokens=2,
            completion_tokens=3, total_tokens=5,
        )
        self.mock(web_service, "_validated_trip", return_value=(self.trip, self.request, self.current))
        self.mock(web_service, "_normalized_trip_request", return_value=self.request)
        self.mock(web_service, "_daily_ai_limit", return_value=5)
        self.mock(web_service.reminders, "get_days_before", return_value=3)
        self.mock(web_service.db, "get_account_key", return_value="test-account")
        self.mock(web_service.db, "get_user", return_value={"name": "Test"})
        self.claim = self.mock(web_service.db, "claim_ai_request", return_value=SimpleNamespace(
            status="ok", claim_id="test-claim"
        ))
        self.finish = self.mock(web_service, "_finish_ai_request_safely")
        self.edit = self.mock(web_service, "edit_itinerary", return_value=self.result)
        self.generate = self.mock(web_service, "generate_itinerary", return_value=self.result)
        # This suite tests post-planner outcomes, not free geocoding; the latter
        # has real-helper regressions in test_city_resolution_service.
        self.mock(web_service, "resolve_city_for_planning", return_value=object())
        self.save = self.mock(web_service.db, "save_trip_bundle", return_value=9)
        self.update = self.mock(web_service.db, "update_trip_plan", return_value=True)
        self.delete = self.mock(web_service.db, "delete_trip")
        self.detail = self.mock(web_service, "get_trip_detail", return_value={"id": 9})
        self.mock(web_service, "build_checklist", return_value=[("test", "item")])
        self.mock(web_service.vector_memory, "recall_family_memories", return_value=[])
        self.mock(web_service.vector_memory, "format_memory_context", return_value="")
        self.mock(web_service, "_save_trip_memory_safely", return_value=0)
        self.mock(web_app.logger, "exception")

    def mock(self, target, name, **kwargs):
        return self.stack.enter_context(patch.object(target, name, **kwargs))

    def edit_day(self):
        return web_service.edit_trip_day(123, 9, 1, "Добавь парк")

    def assert_finished(self, succeeded):
        self.finish.assert_called_once_with("test-claim", succeeded=succeeded)
        self.claim.assert_called_once()

    def test_rejected_edit_is_failed_and_does_not_save(self):
        self.result.problems = [f"День {day}: base_area не подтверждена; исправь place_query" for day in range(1, 15)]
        original = list(self.result.problems)
        with self.assertRaises(web_service.EditRejectedError) as raised:
            self.edit_day()
        self.assertEqual(raised.exception.diagnostic_problems, original)
        self.assertIsNot(raised.exception.diagnostic_problems, self.result.problems)
        self.assertNotIn("place_query", str(raised.exception))
        self.assertNotIn("base_area", " ".join(raised.exception.problems))
        self.assertEqual(len(raised.exception.problems), 1)
        self.update.assert_not_called()
        self.assert_finished(False)

    def test_conflicting_edit_is_failed_and_does_not_render(self):
        self.update.return_value = False
        self.mock(web_service.db, "get_trip", return_value=self.trip)
        with self.assertRaises(web_service.EditConflictError):
            self.edit_day()
        self.detail.assert_not_called()
        self.assert_finished(False)

    def test_edit_storage_error_is_failed(self):
        self.update.side_effect = RuntimeError("Storage unavailable")
        with self.assertRaisesRegex(RuntimeError, "Storage unavailable"):
            self.edit_day()
        self.detail.assert_not_called()
        self.assert_finished(False)

    def test_edit_detail_failure_keeps_committed_result_and_reports_saved(self):
        self.detail.side_effect = RuntimeError("Detail unavailable")
        with self.assertRaises(web_service.TripSavedResponseError) as raised:
            self.edit_day()
        self.assertEqual(raised.exception.trip_id, 9)
        self.assertIn("сохранены", str(raised.exception))
        self.assertNotIn("Старый", str(raised.exception))
        self.update.assert_called_once()
        self.delete.assert_not_called()
        self.assert_finished(True)

    def test_successful_edit_finishes_only_after_commit(self):
        self.update.side_effect = lambda *args, **kwargs: not self.finish.called
        detail = self.edit_day()
        self.assertEqual(detail["edit"]["day"], 1)
        self.assert_finished(True)

    def test_rejected_creation_is_failed_without_saving(self):
        self.result.problems = [f"День {day}: place_query пуст; Укажи официальное название" for day in range(1, 15)]
        original = list(self.result.problems)
        with self.assertRaises(web_service.TripCreateValidationError) as raised:
            web_service.create_trip(123, self.request)
        self.assertEqual(raised.exception.diagnostic_problems, original)
        self.assertIsNot(raised.exception.diagnostic_problems, self.result.problems)
        self.assertNotIn("place_query", str(raised.exception))
        self.assertNotIn("Укажи", str(raised.exception))
        self.assertIn("Новая поездка не сохранена", str(raised.exception))
        self.save.assert_not_called()
        self.assert_finished(False)

    def test_creation_storage_error_is_failed(self):
        self.save.side_effect = RuntimeError("Storage unavailable")
        with self.assertRaisesRegex(RuntimeError, "Storage unavailable"):
            web_service.create_trip(123, self.request)
        self.detail.assert_not_called()
        self.assert_finished(False)

    def test_provider_format_exhaustion_is_validation_without_storage_or_raw_exception(self):
        self.generate.side_effect=web_service.PlannerFormatError('REJECTED_PRIVATE_MODEL_STRING')
        with self.assertRaises(web_service.TripCreateValidationError) as raised:
            web_service.create_trip(123,self.request)
        self.assertEqual(raised.exception.code,'provider_format')
        self.assertEqual(raised.exception.diagnostic_problems,['provider_response_schema_invalid'])
        self.assertNotIn('REJECTED_PRIVATE',str(raised.exception))
        self.save.assert_not_called();self.detail.assert_not_called()
        self.assert_finished(False)

    def test_unrelated_planner_runtime_error_is_not_relabelled_as_provider_format(self):
        self.generate.side_effect=RuntimeError('ordinary internal error')
        with self.assertRaisesRegex(RuntimeError,'ordinary internal error'):
            web_service.create_trip(123,self.request)
        self.save.assert_not_called()
        self.assert_finished(False)

    def test_two_real_planner_enum_failures_become_web_validation_without_saving(self):
        import planner
        import test_grounded_planner as grounded
        import test_regional_generation_repair as regional
        request=regional._request()
        self.mock(web_service,'_normalized_trip_request',return_value=request)
        source=grounded._source()
        bad=json.loads(grounded._id_wire_response_json(source))
        bad['day_4']['attractions'][0]['candidate_id']='REJECTED_PRIVATE_MODEL_STRING'
        completion,_,map_verify=grounded.GroundedPlannerTests()._context(self.stack,[
            json.dumps(grounded._batch(source)),json.dumps(bad),json.dumps(bad)])
        self.generate.side_effect=lambda normalized,**kwargs:planner.generate_itinerary(
            normalized,max_attempts=3,ground_candidates=True)
        with self.assertRaises(web_service.TripCreateValidationError) as raised:
            web_service.create_trip(123,request)
        self.assertEqual(raised.exception.code,'provider_format')
        self.assertNotIn('REJECTED_PRIVATE',str(raised.exception))
        self.assertEqual(completion.call_count,3)
        map_verify.assert_not_called()
        self.save.assert_not_called();self.detail.assert_not_called()
        self.assert_finished(False)

    def test_creation_detail_failure_does_not_delete_saved_trip(self):
        self.detail.side_effect = RuntimeError("Detail unavailable")
        with self.assertRaises(web_service.TripSavedResponseError) as raised:
            web_service.create_trip(123, self.request)
        self.assertEqual(raised.exception.trip_id, 9)
        self.assertIn("создан и сохранён", str(raised.exception))
        self.save.assert_called_once()
        self.delete.assert_not_called()
        self.assert_finished(True)

    def test_successful_creation_is_saved_and_succeeded(self):
        detail = web_service.create_trip(123, self.request)
        self.assertEqual(detail["id"], 9)
        self.assertEqual(detail["generation"]["total_tokens"], 5)
        self.save.assert_called_once()
        self.assert_finished(True)

    def test_creation_persists_safe_quality_without_raw_diagnostics(self):
        self.result.generation_coverage = {"quality_status": "partial", "discovery": {
            "confirmed_candidates": 3, "raw_response": "private-secret"}}
        web_service.create_trip(123, self.request)
        quality = self.save.call_args.args[1]["generation_coverage"]
        self.assertEqual(quality["quality_status"], "partial")
        self.assertNotIn("raw_response", quality["discovery"])

    def test_edit_does_not_retain_old_quality_audit(self):
        self.trip["request"]["generation_coverage"] = {"quality_status": "complete", "program_days": 7}
        self.result.generation_coverage = {"quality_status": "partial", "program_days": 1}
        self.edit_day()
        self.assertEqual(self.update.call_args.args[2]["generation_coverage"]["quality_status"], "partial")

    def test_edit_endpoint_distinguishes_postcommit_failure(self):
        self.detail.side_effect = RuntimeError("Detail unavailable")

        async def run_inline(function, *args):
            return function(*args)

        self.mock(web_app.anyio.to_thread, "run_sync", side_effect=run_inline)
        with self.assertRaises(web_app.HTTPException) as raised:
            asyncio.run(web_app.edit_day(
                9, 1, web_app.DayEditRequest(instruction="Добавь парк"), tg_id=123
            ))
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail["trip_id"], 9)
        self.assertTrue(raised.exception.detail["saved"])
        self.assertIn("сохранены", raised.exception.detail["message"])
        self.assert_finished(True)

    def test_edit_endpoint_keeps_failure_but_exposes_only_safe_reason(self):
        self.result.problems = ["День 1: base_area не подтверждена. Укажи другое place_query."]

        async def run_inline(function, *args):
            return function(*args)

        self.mock(web_app.anyio.to_thread, "run_sync", side_effect=run_inline)
        with self.assertRaises(web_app.HTTPException) as raised:
            asyncio.run(web_app.edit_day(
                9, 1, web_app.DayEditRequest(instruction="Добавь парк"), tg_id=123
            ))
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("Изменения не сохранены", raised.exception.detail["message"])
        self.assertIn("подтверждённых картой", " ".join(raised.exception.detail["problems"]))
        for internal in ("place_query", "base_area", "Укажи", "diagnostic_problems"):
            self.assertNotIn(internal, str(raised.exception.detail))
        self.update.assert_not_called()
        self.assert_finished(False)

    def test_creation_endpoint_keeps_failure_but_exposes_only_safe_reason(self):
        self.result.problems = ["place_query пуст. Укажи официальное название"] * 7

        async def run_inline(function, *args):
            return function(*args)

        self.mock(web_app.anyio.to_thread, "run_sync", side_effect=run_inline)
        payload = web_app.TripCreateRequest(
            destination="Воркута, Россия", start_date="2030-11-03", end_date="2030-11-03",
            adults=2, budget=45000.0, currency="RUB",
        )
        request = SimpleNamespace(state=SimpleNamespace(account_key="test-account"))
        with self.assertRaises(web_app.HTTPException) as raised:
            asyncio.run(web_app.create_trip(payload, request, tg_id=123))
        self.assertEqual(raised.exception.status_code, 422)
        self.assertIn("Новая поездка не сохранена", raised.exception.detail)
        self.assertLess(len(raised.exception.detail), 300)
        for internal in ("place_query", "Укажи", "diagnostic_problems"):
            self.assertNotIn(internal, raised.exception.detail)
        self.save.assert_not_called()
        self.assert_finished(False)

    def test_creation_endpoint_distinguishes_postcommit_failure(self):
        self.detail.side_effect = RuntimeError("Detail unavailable")

        async def run_inline(function, *args):
            return function(*args)

        self.mock(web_app.anyio.to_thread, "run_sync", side_effect=run_inline)
        payload = web_app.TripCreateRequest(
            destination="Воркута, Россия", start_date="2030-11-03", end_date="2030-11-03",
            adults=2, budget=45000.0, currency="RUB",
        )
        request = SimpleNamespace(state=SimpleNamespace(account_key="test-account"))
        with self.assertRaises(web_app.HTTPException) as raised:
            asyncio.run(web_app.create_trip(payload, request, tg_id=123))
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail["trip_id"], 9)
        self.assertTrue(raised.exception.detail["saved"])
        self.assertIn("создан и сохранён", raised.exception.detail["message"])
        self.delete.assert_not_called()
        self.assert_finished(True)


if __name__ == "__main__":
    unittest.main()
