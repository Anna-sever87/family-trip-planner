"""The new pipeline never pays an AI to repair a server-owned timetable."""

from contextlib import ExitStack
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import city_planner
import city_rollout
import planner
from city_candidates import CityCandidateResult
from schemas import Itinerary
from test_city_candidate_planner import default_candidates, request


class DeterministicIntegrationTests(unittest.TestCase):
    def run_plan(self, *, builder=None, validators=None, trip_request=None, selection_response=None,
                 candidate_result=None):
        candidates = default_candidates()
        center = {"lat": 55.0, "lon": 83.0, "name": "Тестгород", "country_code": "RU"}

        def ground(req, resolved, *, completion, **kwargs):
            completion([{"role": "user", "content": "Synthetic names only"}])
            return candidate_result if candidate_result is not None else CityCandidateResult(candidates=candidates)

        with ExitStack() as stack:
            stack.enter_context(patch("httpx.get", side_effect=AssertionError("No network")))
            stack.enter_context(patch("places.geocode", side_effect=AssertionError("No map query")))
            stack.enter_context(patch.object(city_planner, "city_deterministic_enabled", return_value=True))
            stack.enter_context(patch.object(city_planner, "find_city", return_value=center))
            stack.enter_context(patch.object(city_planner, "get_weather", return_value=None))
            stack.enter_context(patch.object(city_planner, "ground_city_candidates", side_effect=ground))
            stack.enter_context(patch.object(planner, "_load_blocked_places", return_value=[]))
            if builder is not None:
                stack.enter_context(patch("city_deterministic.build_deterministic_city_itinerary", return_value=builder))
            if validators:
                for name, value in validators.items():
                    stack.enter_context(patch.object(city_planner, name, return_value=value))
            completion = stack.enter_context(patch.object(planner, "generate_json", side_effect=[
                SimpleNamespace(text='{"titles":[]}', input_tokens=11, output_tokens=7),
                SimpleNamespace(text=selection_response, input_tokens=11, output_tokens=7)
                if selection_response is not None else AssertionError("No second AI call allowed"),
                AssertionError("No third AI call allowed"),
            ]))
            result = planner.generate_itinerary(trip_request or request(), city_candidates_first=True)
        return result, completion

    def test_schedule_and_budget_are_server_owned_with_only_discovery_completion(self):
        result, completion = self.run_plan()
        self.assertEqual(result.problems, [])
        self.assertEqual(result.generation_status, "success")
        self.assertEqual((completion.call_count, result.attempts, result.total_tokens), (1, 1, 18))
        self.assertTrue(result.itinerary.city_schedule)
        self.assertFalse(result.itinerary.source_schedule)
        self.assertIsNotNone(result.city_map_context)
        visits = [item for day in result.itinerary.days for item in day.items if item.kind == "attraction"]
        self.assertEqual([item.duration_min for item in visits], [90, 90])
        self.assertTrue(all(item.verified for item in visits))
        self.assertTrue(all(not item.indoor for item in visits))

    def test_insufficient_data_never_falls_back_to_model_schedule(self):
        built = SimpleNamespace(itinerary=None, problems=["Недостаточно подтверждённых данных"],
                                status="insufficient_data", coverage={"eligible": 1})
        result, completion = self.run_plan(builder=built)
        self.assertEqual(result.generation_status, "insufficient_data")
        self.assertEqual(result.generation_coverage["eligible"], 1)
        self.assertEqual(result.itinerary.days, [])
        completion.assert_called_once()

    def test_final_boundary_still_rejects_invalid_builder_output(self):
        built = SimpleNamespace(itinerary=Itinerary(title="Invalid test", summary="", days=[], city_schedule=True),
                                problems=[], status="success", coverage={})
        result, completion = self.run_plan(builder=built)
        self.assertTrue(result.problems)
        self.assertEqual(result.generation_status, "validation_error")
        self.assertIsNone(result.city_map_context)
        completion.assert_called_once()

    def test_all_final_validator_findings_reject_without_paid_repair(self):
        names = ("check_rules", "verify_itinerary", "check_city_radius", "check_verified_place_duplicates",
                 "check_program_depth", "check_city_quality", "check_city_visit_estimates",
                 "check_city_visit_conditions")
        valid, _ = self.run_plan()
        self.assertFalse(valid.problems)
        built = SimpleNamespace(itinerary=valid.itinerary, problems=[], status="success", coverage={})
        for name in names:
            with self.subTest(validator=name):
                result, completion = self.run_plan(builder=built, validators={name: ["Synthetic guard finding"]})
                self.assertEqual(result.generation_status, "validation_error")
                self.assertIsNone(result.city_map_context)
                completion.assert_called_once()

    def test_new_switch_is_separate_strict_and_off_without_configuration(self):
        for value, expected in ((None, False), ("", False), ("1", True), ("true", True),
                                ("TRUE", False), ("yes", False), (" true ", False)):
            with self.subTest(value=value), patch.object(city_rollout.os, "getenv", return_value=value):
                self.assertIs(city_rollout.city_deterministic_enabled(), expected)

    def test_semantic_selection_uses_second_call_but_never_model_timetable(self):
        value = request()
        value.interest_categories = ["museums"]
        result, completion = self.run_plan(trip_request=value, selection_response=
                                          '{"status":"selected","candidate_ids":["osm-node-102","osm-node-101"]}')
        self.assertEqual(result.problems, [])
        self.assertEqual(result.generation_status, "success")
        self.assertEqual((completion.call_count, result.attempts, result.total_tokens), (2, 2, 36))
        self.assertEqual(set(completion.call_args.kwargs["json_schema"]["properties"]), {"status", "candidate_ids"})
        self.assertEqual(result.generation_coverage["preference_candidates"], 2)

    def test_selection_refusal_does_not_silently_drop_preferences(self):
        value = request()
        value.wishes = "Посещения только с подтверждённой бесплатностью"
        result, completion = self.run_plan(trip_request=value, selection_response=
                                          '{"status":"unsupported_preferences","candidate_ids":[]}')
        self.assertEqual(result.generation_status, "unsupported_preferences")
        self.assertTrue(result.problems)
        self.assertFalse(result.itinerary.days)
        self.assertEqual(completion.call_count, 2)

    def test_proximity_preflight_does_not_pay_for_known_missing_anchor(self):
        value = request()
        value.location_preference, value.stay_area = "nearby", "Северный район"
        result, completion = self.run_plan(trip_request=value)
        self.assertEqual(result.generation_status, "unsupported_preferences")
        self.assertEqual(result.attempts, 0)
        completion.assert_not_called()

    def test_requested_shopping_reaches_final_calendar_not_only_ai_selection(self):
        from test_city_deterministic import DeterministicCityTests
        pool = DeterministicCityTests.shopping_pool()
        value = request()
        value.interest_categories = ["museums", "shopping_malls"]
        result, completion = self.run_plan(trip_request=value,
            candidate_result=CityCandidateResult(candidates=pool), selection_response=json.dumps({
                "status": "selected", "candidate_ids": [item.candidate_id for item in pool]}))
        self.assertEqual(result.problems, [])
        self.assertEqual(result.generation_status, "success")
        self.assertTrue(result.generation_coverage["requested_category_satisfied"])
        self.assertTrue(any(item.place_query == pool[-1].place_query
                            for day in result.itinerary.days for item in day.items))
        self.assertEqual(completion.call_count, 2)

    def test_explicit_shopping_exclusion_beats_checkbox_in_final_schedule(self):
        from test_city_deterministic import DeterministicCityTests
        pool = DeterministicCityTests.shopping_pool()
        value = request()
        value.interest_categories = ["museums", "shopping_malls"]
        value.wishes = "Без торговых центров"
        result, completion = self.run_plan(trip_request=value,
            candidate_result=CityCandidateResult(candidates=pool), selection_response=json.dumps({
                "status": "selected", "candidate_ids": [item.candidate_id for item in pool[:-1]]}))
        self.assertEqual(result.problems, [])
        self.assertEqual(result.generation_coverage["requested_category_candidates"], 0)
        self.assertFalse(any(item.place_query == pool[-1].place_query
                             for day in result.itinerary.days for item in day.items))
        self.assertEqual(completion.call_count, 2)

    def test_missing_confirmed_mall_is_disclosed_without_rejecting_valid_trip(self):
        value = request()
        value.interest_categories = ["museums", "shopping_malls"]
        result, completion = self.run_plan(trip_request=value, selection_response=json.dumps({
            "status": "selected", "candidate_ids": [item.candidate_id for item in default_candidates()]}))
        self.assertEqual(result.problems, [])
        self.assertEqual(result.generation_status, "success")
        self.assertTrue(any("Не удалось подтвердить подходящий торговый центр" in tip
                            for tip in result.itinerary.tips))
        self.assertEqual(completion.call_count, 2)

    def test_discovery_diagnostics_keep_only_safe_counters_and_static_user_warning(self):
        marker = "synthetic-private-content-not-for-report"
        pool = CityCandidateResult(candidates=default_candidates(), map_requests=12,
            rejected=[{"title": marker, "reason": "name_mismatch", "detail": marker},
                      {"reason": marker}, {"reason": {"text": marker}}],
            warnings=[marker], diagnostics={"attempted_names": 14, "name_limit": 28,
                "usable_limit": 18, "limits_reached": ["name_limit", marker], "wishes": marker})
        result, _ = self.run_plan(candidate_result=pool)
        self.assertFalse(result.problems)
        self.assertEqual(result.generation_coverage["discovery"], {
            "confirmed_candidates": 2, "rejected_count": 3,
            "rejection_reasons": {"name_mismatch": 1, "other": 2}, "warning_count": 1,
            "map_requests": 12, "attempted_names": 14, "name_limit": 28, "usable_limit": 18,
            "limits_reached": ["name_limit"]})
        self.assertIn(city_planner.DISCOVERY_WARNING, result.itinerary.tips)
        self.assertNotIn(marker, json.dumps(result.generation_coverage, ensure_ascii=False))
        self.assertNotIn(marker, result.itinerary.model_dump_json())

    def test_invalid_diagnostic_types_cannot_leak_details_or_invent_counts(self):
        pool = SimpleNamespace(candidates=[], rejected="not a list", warnings={"private": "text"},
                               map_requests=True, diagnostics={"attempted_names": True,
                                   "name_limit": -1, "usable_limit": "secret", "limits_reached": [{}, True]})
        self.assertEqual(city_planner._discovery_diagnostics(pool), {
            "confirmed_candidates": 0, "rejected_count": 0, "rejection_reasons": {},
            "warning_count": 0, "map_requests": 0, "limits_reached": []})


if __name__ == "__main__":
    unittest.main()
