"""Offline city-day quality: real scheduler, no network, AI, cache or DB."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import city_deterministic as engine
from city_quality import (_family_minimum, apply_city_day_coverage, city_break_minutes,
                          city_day_coverage, desired_city_program_minutes)
from rules import check_rules, program_depth_target
from test_city_deterministic import CENTER, candidates, options, request, start, visits


class CityDayBalanceTests(unittest.TestCase):
    def setUp(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.blocked = [stack.enter_context(patch(name, side_effect=AssertionError("External I/O forbidden")))
                        for name in ("httpx.get", "httpx.post", "httpx.Client.send", "sqlite3.connect",
                                     "places.geocode", "places._load_cache", "places._save_cache",
                                     "planner.generate_json")]

    def tearDown(self):
        for mocked in self.blocked:
            mocked.assert_not_called()

    def build(self, req, pool, chosen=None, **kwargs):
        result = engine.build_deterministic_city_itinerary(
            req, pool, resolved_city=CENTER, options=chosen, **kwargs)
        self.assertEqual((result.status, result.problems), ("success", []), result.coverage)
        self.assertEqual(check_rules(result.itinerary, req, allow_city_limited_program=True), [])
        chosen_visits = visits(result.itinerary)
        self.assertEqual(len({item.place_query for item in chosen_visits}), len(chosen_visits))
        self.assertTrue({item.place_query for item in chosen_visits} <= {item.place_query for item in pool})
        self.assertLessEqual(sum(item.cost for day in result.itinerary.days for item in day.items), req.program_budget)
        self.assertLessEqual(result.coverage["states_explored"], engine.MAX_SEARCH_STATES)
        for day in result.itinerary.days:
            for item in day.items:
                self.assertGreaterEqual(start(item), int(req.start_time[:2]) * 60 + int(req.start_time[3:]))
                self.assertLessEqual(start(item) + item.duration_min, int(req.end_time[:2]) * 60 + int(req.end_time[3:]))
        return result

    def test_teen_desired_volume_matches_adults_without_raising_hard_minimum(self):
        adult = request(4, end_time="18:00")
        teen = request(4, end_time="18:00", children_ages=[14])
        self.assertEqual(desired_city_program_minutes(adult), 270)
        self.assertEqual(desired_city_program_minutes(teen), 270)
        self.assertEqual(program_depth_target(adult), 180)
        self.assertEqual(_family_minimum(teen), 120)
        self.assertEqual(city_break_minutes(teen), 20)

    def test_medium_teen_four_days_use_reserve_evenly_and_have_afternoon_visits(self):
        req, pool = request(4, end_time="18:00", children_ages=[14]), candidates(12, separation=.002)
        original = deepcopy(req.model_dump())
        result = self.build(req, pool)
        self.assertEqual(req.model_dump(), original)
        self.assertEqual(len(visits(result.itinerary)), 12)
        self.assertEqual(result.coverage["quality_status"], "complete", result.coverage)
        self.assertEqual(result.coverage["light_program_days"], 0)
        for day in result.itinerary.days:
            day_visits = [item for item in day.items if item.kind == "attraction"]
            self.assertEqual([item.duration_min for item in day_visits], [90] * 3)
            self.assertTrue(any(start(item) < 13 * 60 for item in day_visits))
            self.assertTrue(any(start(item) >= 14 * 60 for item in day_visits))
            self.assertGreaterEqual(max(start(item) + item.duration_min for item in day_visits), 16 * 60 + 30)
            self.assertEqual(next(item.duration_min for item in day.items if item.kind == "rest"), 20)

    def test_adult_day_keeps_short_pause_and_fixed_visit_durations(self):
        req, pool = request(1, end_time="18:00"), candidates(3, separation=.002)
        result = self.build(req, pool)
        self.assertEqual(result.coverage["quality_status"], "complete", result.coverage)
        self.assertEqual([item.duration_min for item in visits(result.itinerary)], [90, 90, 90])
        self.assertEqual(next(item.duration_min for item in result.itinerary.days[0].items if item.kind == "rest"), 20)

    def test_under_seven_keeps_hour_rest_and_age_visit_cap(self):
        req, pool = request(2, end_time="18:00", children_ages=[5]), candidates(6, separation=.002)
        self.assertEqual(desired_city_program_minutes(req), 165)
        result = self.build(req, pool)
        for day in result.itinerary.days:
            self.assertLessEqual(sum(item.kind == "attraction" for item in day.items), 3)
            self.assertEqual(next(item.duration_min for item in day.items if item.kind == "rest"), 60)
        broken = result.itinerary.model_copy(deep=True)
        next(item for item in broken.days[0].items if item.kind == "rest").duration_min = 20
        self.assertTrue(any("60 минут" in error for error in check_rules(broken, req)))

    def test_short_morning_window_does_not_demand_afternoon_or_exceed_bounds(self):
        req, pool = request(1, end_time="13:00", children_ages=[14]), candidates(1)
        result = self.build(req, pool)
        self.assertEqual(result.coverage["unbalanced_program_days"], 0)
        self.assertEqual(len(visits(result.itinerary)), 1)
        self.assertEqual(visits(result.itinerary)[0].duration_min, 90)

    def test_london_shaped_sparse_reserve_is_honest_partial_not_invented_visits(self):
        req, pool = request(4, end_time="18:00", children_ages=[14]), candidates(9, separation=.002)
        durations = [90, 90, 90, 45, 45, 45, 45, 90, 45]
        chosen = [replace(option, duration_min=duration) for option, duration in zip(options(pool, cost=None), durations)]
        result = self.build(req, pool, chosen)
        self.assertEqual(result.coverage["program_days"], 4)
        self.assertEqual(result.coverage["quality_status"], "partial")
        self.assertIn("day_program_light", result.coverage["quality_reasons"])
        self.assertEqual(sum(item.duration_min for item in visits(result.itinerary)), 585)
        self.assertEqual(sorted(item.duration_min for item in visits(result.itinerary)), sorted(durations))
        self.assertEqual(result.coverage["desired_program_minutes"], 270)

    def test_source_morning_only_windows_are_not_overridden_for_balance(self):
        req, pool = request(1, end_time="18:00"), candidates(3, separation=.002)
        chosen = [replace(value, allowed_dates=(req.start_date,), opening_windows=(
            {"date": req.start_date, "start": "09:00", "end": "13:00", "basis": "source"},))
                  for value in options(pool, cost=None)]
        result = self.build(req, pool, chosen)
        self.assertEqual(result.coverage["quality_status"], "partial")
        self.assertIn("day_window_unbalanced", result.coverage["quality_reasons"])
        self.assertTrue(all(start(item) + item.duration_min <= 13 * 60 for item in visits(result.itinerary)))

    def test_exhausted_long_calendar_search_keeps_capacity_for_enrichment_and_afternoon(self):
        # Structural reproduction of the acceptance failure, not real personal
        # data or a claim that these synthetic objects are the actual city.
        req, pool = request(14, end_time="18:00", children_ages=[14]), candidates(14, separation=.002)
        durations = [120] + [90] * 10 + [45] * 3
        chosen = [replace(option, duration_min=duration)
                  for option, duration in zip(options(pool, cost=None), durations)]
        before = [item.model_dump() if hasattr(item, 'model_dump') else deepcopy(item) for item in chosen]
        result = self.build(req, pool, chosen)
        self.assertEqual(result.coverage["program_days"], 7)
        self.assertFalse(result.coverage["search_complete"])
        self.assertIn("search_limit", result.coverage["reason_codes"])
        self.assertEqual(result.coverage["selected_unique_visits"], 14)
        self.assertEqual(result.coverage["enrichment_visits"], 1)
        self.assertEqual(sorted(item.duration_min for item in visits(result.itinerary)), sorted(durations))
        self.assertEqual(chosen, before)
        self.assertEqual(result.coverage["quality_status"], "partial")
        self.assertIn("day_program_light", result.coverage["quality_reasons"])
        program = [day for day in result.itinerary.days if any(item.kind == "attraction" for item in day.items)]
        self.assertEqual(len(program), 7)
        for day in program:
            points = [item for item in day.items if item.kind == "attraction"]
            self.assertTrue(any(start(item) < 13 * 60 for item in points))
            self.assertTrue(any(start(item) >= 14 * 60 for item in points))

    def test_reserved_balance_never_overrides_closed_morning_windows(self):
        req, pool = request(14, end_time="18:00", children_ages=[14]), candidates(14, separation=.002)
        durations = [120] + [90] * 10 + [45] * 3
        from datetime import date, timedelta
        dates = [(date.fromisoformat(req.start_date) + timedelta(days=index)).isoformat() for index in range(14)]
        windows = tuple({"date": value, "start": "09:00", "end": "13:00", "basis": "source"}
                        for value in dates)
        chosen = [replace(option, duration_min=duration, opening_windows=windows)
                  for option, duration in zip(options(pool, cost=None), durations)]
        result = self.build(req, pool, chosen)
        self.assertGreaterEqual(result.coverage["program_days"], 7)
        self.assertEqual(result.coverage["quality_status"], "partial")
        self.assertIn("day_window_unbalanced", result.coverage["quality_reasons"])
        self.assertTrue(all(start(item) + item.duration_min <= 13 * 60 for item in visits(result.itinerary)))

    def test_colocated_pair_is_not_separated_to_fake_a_full_day(self):
        req, pool = request(1, end_time="18:00"), candidates(2, separation=0)
        result = self.build(req, pool)
        first, second = visits(result.itinerary)
        self.assertEqual(start(second), start(first) + first.duration_min)
        self.assertEqual(result.coverage["quality_status"], "partial")

    def test_observable_reassessment_clears_stale_soft_reasons_after_improvement(self):
        req, pool = request(1, end_time="18:00"), candidates(3, separation=.002)
        result = self.build(req, pool)
        report = {"quality_status": "partial", "quality_reasons": ["day_program_light", "day_window_unbalanced"]}
        apply_city_day_coverage(req, result.itinerary, report)
        self.assertEqual(report["quality_reasons"], [])
        self.assertEqual(report["quality_status"], "complete")
        self.assertEqual(report["light_program_days"], 0)
        self.assertEqual(report["unbalanced_program_days"], 0)
        regional = req.model_copy(update={"destination_kind": "region"})
        self.assertEqual(city_day_coverage(regional, result.itinerary), {})


class CityDesiredDemandTests(unittest.TestCase):
    """Actual bounded discovery with mocked provider transport, not live calls."""

    def setUp(self):
        from test_city_candidates import CityCandidateTests
        self.patched = lambda *args, **kwargs: CityCandidateTests.patched(self, *args, **kwargs)
        CityCandidateTests.setUp(self)

    def test_discovery_requests_desired_reserve_and_keeps_all_existing_caps(self):
        import city_candidates
        from test_city_candidates import CENTER as POOL_CENTER, request as pool_request
        req = pool_request(4, children_ages=[14], start_time="09:00", end_time="18:00", pace="средний")
        result = city_candidates.ground_city_candidates(req, POOL_CENTER, completion=self.completion,
                                                        allow_partial_program=True)
        self.assertEqual(_family_minimum(req) * req.days_count, 480)
        self.assertEqual(result.diagnostics["required_planning_minutes"], 1080)
        self.assertEqual(result.diagnostics["pool_target"], 12)
        self.assertLessEqual(result.diagnostics["usable_limit"], 24)
        self.assertLessEqual(result.map_requests, city_candidates.MAX_MAP_REQUESTS)
        self.assertEqual(result.diagnostics["pool_quality"], "limited")
        self.assertTrue(result.diagnostics["replenishment_attempted"])

    def test_standard_preferences_preserve_thin_reserve_but_expose_desired_demand(self):
        from city_preferences import select_city_preferences
        req, pool = request(4, end_time="18:00", children_ages=[14], trip_goal="Познакомиться с городом"), candidates(9)
        result = select_city_preferences(req, pool, completion=self.completion)
        self.assertEqual(result.problems, [])
        self.assertEqual(len(result.candidates), 9)
        self.assertEqual(result.diagnostics["required_planning_minutes"], 1080)
        self.assertEqual(result.diagnostics["preference_planning_minutes"], 810)
        self.completion.assert_not_called()

    def test_reserve_below_desired_but_above_minimum_still_gets_one_useful_depth_review(self):
        import json
        from types import SimpleNamespace
        from city_preferences import select_city_preferences
        req = request(4, end_time="18:00", children_ages=[14], trip_goal="История и культура города")
        pool = candidates(9)
        all_ids = [item.candidate_id for item in pool]
        self.completion.side_effect = [
            SimpleNamespace(text=json.dumps({"status": "selected", "candidate_ids": all_ids[:4]})),
            SimpleNamespace(text=json.dumps({"status": "selected",
                "candidate_decisions": {identifier: "include" for identifier in all_ids}})),
        ]
        result = select_city_preferences(req, pool, completion=self.completion, allow_depth_repair=True)
        self.assertEqual(result.problems, [])
        self.assertEqual(self.completion.call_count, 2)
        self.assertEqual(result.diagnostics["selection_repair_status"], "improved")
        self.assertEqual(result.diagnostics["required_planning_minutes"], 1080)
        self.assertEqual(result.diagnostics["preference_planning_minutes"], 810)
        self.assertFalse(result.diagnostics["selection_depth_shortfall"])
        self.assertEqual(result.candidate_priority, all_ids)


if __name__ == "__main__":
    unittest.main()
