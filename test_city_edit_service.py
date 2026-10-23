"""Bounded real city orchestration with synthetic completions and no services."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import city_edit_service
import planner
import test_city_candidate_planner as city_fixture
from test_city_candidate_planner import request, payload
from yandex_ai import YandexAIError


class CityEditServiceTests(unittest.TestCase):
    def setUp(self):
        self.request = request()
        generated, _, _ = city_fixture.CityCandidatePlannerTests().run_plan([payload()])
        self.assertEqual(generated.problems, [])
        self.original = generated.itinerary
        self.snapshot = generated.city_map_context
        self.changed = {"day_1": payload()["day_1"]}
        self.changed["day_1"]["meal"]["time"] = "13:30"
        self.changed["day_1"]["rest"]["time"] = "14:30"

    def run_edit(self, values, **options):
        latest_weather = options.pop("latest_weather", None)
        responses = [value if isinstance(value, Exception) else
                     SimpleNamespace(text=value if isinstance(value, str) else json.dumps(value, ensure_ascii=False),
                                     input_tokens=11, output_tokens=7)
                     for value in values]
        complete = Mock(side_effect=responses)
        with patch.object(city_edit_service, "get_weather", return_value=latest_weather) as weather, \
                patch.object(planner, "_load_blocked_places", return_value=[]) as blocked, \
                patch("places.geocode", side_effect=AssertionError("No geocoder")), \
                patch("httpx.get", side_effect=AssertionError("No HTTP")):
            result = city_edit_service.edit_city_itinerary(
                self.request, self.original, 1, "Перенеси обед на 13:30, а отдых на 14:30",
                city_map_context=self.snapshot, completion=complete, **options)
        return result, complete, weather

    def test_success_requires_generation_and_semantic_audit_and_keeps_snapshot(self):
        before = deepcopy(self.original.model_dump())
        result, complete, weather = self.run_edit([self.changed, {"fulfilled": True, "problems": []}])
        self.assertEqual(result.problems, [])
        self.assertEqual((result.attempts, complete.call_count, result.total_tokens), (2, 2, 36))
        self.assertEqual(self.original.model_dump(), before)
        self.assertEqual(result.city_map_context, self.snapshot)
        self.assertIsNot(result.city_map_context, self.snapshot)
        self.assertEqual([item.time for item in result.itinerary.days[0].items if item.kind == "meal"], ["13:30"])
        self.assertEqual(weather.call_args.kwargs["resolved_city"]["name"], "Тестгород")
        self.assertEqual(complete.call_args_list[1].kwargs["json_schema"], planner.EDIT_AUDIT_SCHEMA)

    def test_two_rejected_audits_cannot_exceed_four_calls_or_change_original(self):
        rejected = {"fulfilled": False, "problems": ["Не выполнена часть пожелания"]}
        result, complete, _ = self.run_edit([self.changed, rejected, self.changed, rejected])
        self.assertEqual((result.attempts, complete.call_count), (4, 4))
        self.assertTrue(result.problems)
        self.assertEqual(result.itinerary.model_dump(), self.original.model_dump())

    def test_structural_repair_then_audit_uses_three_calls(self):
        invalid = deepcopy(self.changed)
        invalid["day_1"]["attractions"][0]["candidate_id"] = "osm-node-999"
        result, complete, _ = self.run_edit([invalid, self.changed, {"fulfilled": True, "problems": []}])
        self.assertEqual(result.problems, [])
        self.assertEqual((result.attempts, complete.call_count), (3, 3))

    def test_invalid_audit_does_not_authorize_save(self):
        for audit in ({"fulfilled": "true", "problems": []},
                      {"fulfilled": True, "problems": ["Не выполнено"]},
                      {"fulfilled": True, "problems": [], "extra": "not_allowed"}):
            with self.subTest(audit=audit):
                result, complete, _ = self.run_edit([self.changed, audit], max_attempts=1)
                self.assertTrue(result.problems)
                self.assertEqual(complete.call_count, 2)
                self.assertEqual(result.itinerary.model_dump(), self.original.model_dump())

    def test_invalid_snapshot_rejected_before_weather_and_completion(self):
        self.snapshot["request_sha256"] = "wrong"
        with patch.object(city_edit_service, "get_weather") as weather, \
                patch.object(planner, "generate_json") as complete, self.assertRaises(ValueError):
            city_edit_service.edit_city_itinerary(self.request, self.original, 1, "Измени обед",
                                                  city_map_context=self.snapshot, completion=complete)
        weather.assert_not_called()
        complete.assert_not_called()

    def test_duplicate_audit_keys_and_nonfinite_constants_never_authorize_save(self):
        for raw in ('{"fulfilled":false,"fulfilled":true,"problems":[]}',
                    '{"fulfilled":true,"problems":["no"],"problems":[]}',
                    '{"fulfilled":true,"problems":[NaN]}'):
            with self.subTest(raw=raw):
                result, complete, _ = self.run_edit([self.changed, raw], max_attempts=1)
                self.assertTrue(result.problems)
                self.assertEqual(result.itinerary.model_dump(), self.original.model_dump())
                self.assertEqual(complete.call_count, 2)

    def weather(self, rain=5, source="forecast"):
        return {"city": "Тестгород", "source": source,
                "days": [{"date": self.request.start_date, "tmin": 10, "tmax": 15, "rain_mm": rain}]}

    def test_failed_refresh_preserves_weather_and_labels_saved_evidence(self):
        previous = self.weather()
        result, _, _ = self.run_edit([self.changed, {"fulfilled": True, "problems": []}],
                                    prior_weather=previous)
        self.assertEqual(result.problems, [])
        self.assertEqual(result.weather, previous)
        self.assertIsNot(result.weather, previous)
        self.assertIn("ранее сохранённые", result.weather_text)

    def test_failed_refresh_or_history_cannot_remove_previous_rain_rules(self):
        outdoor = deepcopy(self.changed)
        for item in outdoor["day_1"]["attractions"]:
            item["indoor"] = False
        for latest in (None, self.weather(0, "history")):
            with self.subTest(latest=latest):
                result, complete, _ = self.run_edit([outdoor], max_attempts=1,
                                                    prior_weather=self.weather(), latest_weather=latest)
                self.assertTrue(result.problems)
                self.assertEqual(complete.call_count, 1)
                self.assertEqual(result.itinerary.model_dump(), self.original.model_dump())
                self.assertEqual(result.weather, self.weather())

    def test_current_forecast_can_replace_old_forecast(self):
        fresh = self.weather(0)
        result, _, _ = self.run_edit([self.changed, {"fulfilled": True, "problems": []}],
                                    prior_weather=self.weather(), latest_weather=fresh)
        self.assertFalse(result.problems)
        self.assertEqual(result.weather, fresh)
        self.assertNotIn("ранее сохранённые", result.weather_text)

    def test_corrupt_saved_weather_fails_closed_before_completion(self):
        for mutation in ({"date": "2099-10-29"}, {"rain_mm": float("nan")}, {"rain_mm": True}):
            previous = self.weather()
            previous["days"][0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "данные погоды"):
                self.run_edit([], prior_weather=previous)

    def test_missing_snapshot_still_fails_closed_at_planner_entry(self):
        with patch.object(city_edit_service, "edit_city_itinerary") as edit, self.assertRaises(ValueError):
            planner.edit_itinerary(self.request, self.original, 1, "Измени обед")
        edit.assert_not_called()

    def test_planner_routes_only_city_with_server_context_to_city_editor(self):
        with patch.object(city_edit_service, "edit_city_itinerary", return_value="sentinel") as edit:
            value = planner.edit_itinerary(self.request, self.original, 1, "Измени обед",
                                           city_map_context=self.snapshot, prior_weather=self.weather())
        self.assertEqual(value, "sentinel")
        self.assertIs(edit.call_args.kwargs["completion"], planner.generate_json)
        self.assertEqual(edit.call_args.kwargs["prior_weather"], self.weather())


if __name__ == "__main__":
    unittest.main()
