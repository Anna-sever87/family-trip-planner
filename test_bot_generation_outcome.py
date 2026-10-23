"""Execute the real bot job body without importing bot or starting its services."""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import Mock

from generation_feedback import generation_failure_message
from city_context import CITY_EDIT_NOTE, CONTEXT_ERROR, CONTEXT_KEY, check_context_places, load_city_context
from schemas import Itinerary, TripRequest
from weather import CityResolutionError


def city_fixture():
    """Real map-bound pool: one program day and one free day, no I/O."""
    from city_context import build_city_context
    from city_coverage import build_city_free_day
    from places import verify_itinerary
    from program_coverage import rebuild_program_report
    from schemas import ItineraryDay, ItineraryItem
    from test_city_candidate_planner import default_candidates, request

    req = request().model_copy(update={"end_date": "2099-10-29"})
    pool = default_candidates()
    visits = [ItineraryItem(time=time, kind="attraction", title=c.title, area=c.area,
                            place_query=c.place_query, description="Осмотр экспозиции",
                            duration_min=90, cost=1000, indoor=True)
              for time, c in zip(("09:00", "10:45"), pool)]
    pauses = [ItineraryItem(time="13:00", kind="meal", title="Обед", description="Перерыв", duration_min=60, cost=1400),
              ItineraryItem(time="14:00", kind="rest", title="Отдых", description="Перерыв", duration_min=60, cost=0)]
    route = Itinerary(title="Test city", summary="", city_schedule=True, days=[
        ItineraryDay(day=1, date=req.start_date, theme="Музеи", items=visits + pauses),
        build_city_free_day(req, 2)])
    errors = verify_itinerary(route, verified_records={c.place_query: c.record for c in pool})
    if errors:
        raise AssertionError(errors)
    rebuild_program_report(route, req, eligible_unique_visits=2, allow_city_limited_program=True)
    context = build_city_context(req, pool, {"name": "Тестгород", "lat": 55.0, "lon": 83.0, "country_code": "RU"})
    return req, route, context


def _job(namespace, name="generation_job"):
    source = Path(__file__).with_name("bot.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    # Only the unchanged production function body is executed; module imports,
    # token loading, db.init_db, scheduler and Telegram startup never run.
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, "bot.py", "exec"), namespace)
    return namespace[name]


class BotGenerationOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.request = TripRequest(
            destination="Тестовый край, Россия", destination_kind="region",
            start_date="2099-06-01", end_date="2099-06-01", budget=50_000,
        )
        self.result = SimpleNamespace(
            problems=[], itinerary=Mock(), weather_text="", weather=None,
            attempts=2, prompt_tokens=11, completion_tokens=13,
        )
        self.result.itinerary.model_dump.return_value = {"days": []}
        self.namespace = {
            "Optional": Optional, "TripRequest": TripRequest, "Itinerary": Itinerary,
            "deepcopy": deepcopy, "CONTEXT_KEY": CONTEXT_KEY, "CONTEXT_ERROR": CONTEXT_ERROR,
            "CITY_EDIT_NOTE": CITY_EDIT_NOTE, "load_city_context": load_city_context,
            "check_context_places": check_context_places,
            "CityResolutionError": CityResolutionError,
            "city_candidates_enabled": Mock(return_value=False),
            "bot": Mock(), "db": Mock(), "logger": Mock(),
            "vector_memory": Mock(), "_memory_query": Mock(return_value="test"),
            "generate_itinerary": Mock(return_value=self.result),
            "generation_failure_message": generation_failure_message,
            "build_checklist": Mock(return_value=[]),
            "_remember_trip_safely": Mock(return_value=0),
            "format_itinerary": Mock(return_value="Test route"), "send_parts": Mock(),
            "_finish_ai_claim": Mock(), "_finish_generation_job": Mock(),
        }
        self.namespace["vector_memory"].recall_family_memories.return_value = []
        self.namespace["vector_memory"].format_memory_context.return_value = "test memory"
        self.namespace["db"].save_trip.return_value = 99
        self.job = _job(self.namespace)

    def run_job(self):
        self.job(100, 100, self.request, "test-claim")
        self.namespace["_finish_generation_job"].assert_called_once_with(100)

    def assert_no_trip_writes(self):
        self.namespace["db"].save_trip.assert_not_called()
        self.namespace["db"].save_checklist.assert_not_called()
        self.namespace["_remember_trip_safely"].assert_not_called()
        self.namespace["format_itinerary"].assert_not_called()
        self.namespace["send_parts"].assert_not_called()

    def test_rejected_result_is_not_saved_or_counted_as_success(self):
        self.result.problems = ["Недостаточно подтверждённых мест"]
        self.run_job()
        self.assert_no_trip_writes()
        self.result.itinerary.model_dump.assert_not_called()
        self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", False)
        message = self.namespace["bot"].send_message.call_args.args[1]
        self.assertIn("Новая поездка не сохранена", message)
        self.assertIn("подтверждённых картой", message)

    def test_creation_does_not_send_internal_repair_instructions(self):
        self.result.problems = [
            f"День {day}: place_query пуст. Укажи официальное название; исправь base_area."
            for day in range(1, 8)
        ]
        self.run_job()
        message = self.namespace["bot"].send_message.call_args.args[1]
        self.assertLess(len(message), 300)
        for internal in ("place_query", "base_area", "Укажи", "исправь"):
            self.assertNotIn(internal, message)
        self.assert_no_trip_writes()
        self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", False)

    def test_region_enables_grounding_and_success_is_saved(self):
        self.run_job()
        self.namespace["generate_itinerary"].assert_called_once_with(
            self.request, memory_context="test memory", ground_candidates=True,
        )
        self.namespace["db"].save_trip.assert_called_once()
        self.namespace["db"].save_checklist.assert_called_once()
        self.namespace["_remember_trip_safely"].assert_called_once()
        self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", True)

    def test_city_keeps_legacy_generation_path(self):
        self.request.destination_kind = "city"
        self.run_job()
        self.assertFalse(self.namespace["generate_itinerary"].call_args.kwargs["ground_candidates"])
        self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", True)

    def test_new_city_context_is_saved_privately_without_mutating_result(self):
        self.request, self.result.itinerary, context = city_fixture()
        self.result.city_map_context = context
        before = deepcopy(context)
        self.run_job()
        saved_request = self.namespace["db"].save_trip.call_args.args[1]
        self.assertEqual(saved_request[CONTEXT_KEY], before)
        self.assertIsNot(saved_request[CONTEXT_KEY], context)
        self.assertEqual(context, before)
        self.assertNotIn(CONTEXT_KEY, self.namespace["db"].save_trip.call_args.args[2])
        footer = self.namespace["bot"].send_message.call_args.args[1]
        self.assertIn("/edit 99 1 текст", footer)
        self.assertIn(CITY_EDIT_NOTE, footer)
        self.assertNotIn("request_sha256", footer)
        self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", True)

    def test_new_city_cannot_be_saved_without_exact_valid_context_dictionary(self):
        for invalid in (None, [], "snapshot", {"version": 1}):
            with self.subTest(invalid=invalid):
                self.setUp()
                self.request, self.result.itinerary, _ = city_fixture()
                self.result.city_map_context = invalid
                self.run_job()
                self.assert_no_trip_writes()
                self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", False)
                self.assertIn("Новая поездка не сохранена", self.namespace["bot"].send_message.call_args.args[1])

    def test_new_city_context_rejects_request_or_place_mismatch_before_save(self):
        for mutation in ("request", "place"):
            with self.subTest(mutation=mutation):
                self.setUp()
                self.request, self.result.itinerary, context = city_fixture()
                self.result.city_map_context = context
                if mutation == "request":
                    self.request.budget += 1
                else:
                    self.result.itinerary.days[0].items[0].lat += .001
                self.run_job()
                self.assert_no_trip_writes()
                self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", False)

    def test_generation_error_releases_job_without_saving(self):
        self.namespace["generate_itinerary"].side_effect = RuntimeError("offline failure")
        self.run_job()
        self.assert_no_trip_writes()
        self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", False)

    def test_storage_error_does_not_mark_unsaved_result_successful(self):
        self.namespace["db"].save_trip.side_effect = RuntimeError("offline storage failure")
        self.run_job()
        self.namespace["db"].save_trip.assert_called_once()
        self.namespace["db"].save_checklist.assert_not_called()
        self.namespace["_remember_trip_safely"].assert_not_called()
        self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", False)

    def test_saved_id_survives_checklist_memory_format_and_send_failures_without_retry_advice(self):
        for stage in ("checklist", "memory", "format", "parts", "footer"):
            with self.subTest(stage=stage):
                self.setUp()
                failure = RuntimeError("synthetic failure after persisted trip")
                if stage == "checklist":
                    self.namespace["db"].save_checklist.side_effect = failure
                elif stage == "memory":
                    self.namespace["_remember_trip_safely"].side_effect = failure
                elif stage == "format":
                    self.namespace["format_itinerary"].side_effect = failure
                elif stage == "parts":
                    self.namespace["send_parts"].side_effect = failure
                else:
                    self.namespace["bot"].send_message.side_effect = [failure, None]
                self.run_job()
                message = self.namespace["bot"].send_message.call_args.args[1]
                self.assertIn("Поездка №99 сохранена", message)
                self.assertIn("/trip 99", message)
                self.assertIn("Не запускайте создание повторно", message)
                self.assertNotIn("/plan", message)
                self.namespace["db"].save_trip.assert_called_once()
                self.namespace["generate_itinerary"].assert_called_once()
                self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", True)

    def test_saved_id_also_survives_late_city_resolution_error(self):
        self.namespace["format_itinerary"].side_effect = CityResolutionError("Тестовый город", reason="ambiguous")
        self.run_job()
        message = self.namespace["bot"].send_message.call_args.args[1]
        self.assertIn("Поездка №99 сохранена", message)
        self.assertIn("/trip 99", message)
        self.assertNotIn("/plan", message)
        self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", True)

    def run_edit(self, *, saved=True):
        current = Itinerary(title="Test route", summary="", days=[])
        self.namespace["db"].get_trip.return_value = {
            "request": self.request.model_dump(), "itinerary": current.model_dump(), "revision": 1,
        }
        self.namespace["edit_itinerary"] = Mock(return_value=self.result)
        self.namespace["db"].update_trip_plan.return_value = saved
        _job(self.namespace, "edit_generation_job")(100, 99, 1, "Test edit", "test-claim")
        self.namespace["_finish_generation_job"].assert_called_once_with(100)

    def test_rejected_edit_keeps_old_trip_and_is_not_successful(self):
        self.result.problems = ["Недостаточно подтверждённых мест"]
        self.run_edit()
        self.namespace["db"].update_trip_plan.assert_not_called()
        self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", False)

    def test_edit_does_not_send_internal_repair_instructions(self):
        self.result.problems = ["День 1: base_area вне территории. Укажи другое place_query."]
        self.run_edit()
        message = self.namespace["bot"].send_message.call_args.args[1]
        self.assertIn("Изменения не сохранены", message)
        self.assertIn("Старый маршрут сохранён без изменений", message)
        for internal in ("place_query", "base_area", "Укажи"):
            self.assertNotIn(internal, message)
        self.namespace["db"].update_trip_plan.assert_not_called()
        self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", False)

    def test_conflicting_edit_is_not_counted_as_success(self):
        self.run_edit(saved=False)
        self.namespace["db"].update_trip_plan.assert_called_once()
        self.namespace["format_itinerary"].assert_not_called()
        self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", False)

    def test_saved_edit_is_counted_as_success(self):
        self.run_edit()
        self.namespace["db"].update_trip_plan.assert_called_once()
        self.assertEqual(self.namespace["edit_itinerary"].call_args.kwargs, {})
        self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", True)


if __name__ == "__main__":
    unittest.main()
