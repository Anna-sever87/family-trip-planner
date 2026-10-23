"""Offline, read-only regressions for explicit complete calendars with free days."""

from copy import deepcopy
from dataclasses import replace
import socket
import sqlite3
import unittest
from unittest.mock import patch

from pricing import minimum_meal_cost
from regional_skeleton import TravelFact, build_regional_skeleton
from rules import program_depth_target
from test_regional_skeleton import fixture, request_for, windows_for


class RegionalFreeDayTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "create_connection"),
                            (socket, "getaddrinfo"), (sqlite3, "connect")):
            blocker = patch.object(owner, name, side_effect=AssertionError("external I/O forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)
        for target in ("places.geocode", "places._load_cache", "places._save_cache"):
            blocker = patch(target, side_effect=AssertionError("map/cache I/O forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)

    def build(self, request, grounding, facts, graph, **kwargs):
        return build_regional_skeleton(request, grounding, facts, travel_facts=graph,
                                       allow_free_days=True, **kwargs)

    def assert_complete_calendar(self, result, request, active_count):
        self.assertEqual(result.status, "limited_program", result)
        self.assertEqual(len(result.days), request.days_count)
        self.assertEqual([d.day for d in result.days], list(range(1, request.days_count + 1)))
        self.assertEqual([d.date for d in result.days], [w.date for w in windows_for(request)])
        active = [d for d in result.days if not d.is_free_day]
        self.assertEqual(len(active), active_count)
        self.assertGreater(active_count, 0)
        self.assertLess(active_count, request.days_count)
        ids = [visit for day in active for visit in day.visit_ids]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertLessEqual(result.known_cost_total + result.planning_reserve_total, request.program_budget)
        for day in result.days:
            kinds = [s.kind for s in day.slots]
            self.assertEqual(kinds.count("meal"), 1)
            self.assertEqual(kinds.count("rest"), 1)
            for a, b in zip(day.slots, day.slots[1:]):
                self.assertLessEqual(a.end_time, b.start_time)
            self.assertTrue(all(request.start_time <= s.start_time < s.end_time <= request.end_time
                                for s in day.slots))
            if day.is_free_day:
                self.assertEqual(set(kinds), {"meal", "rest"})
                self.assertEqual(day.visit_ids, ())
                self.assertEqual((day.visit_minutes, day.travel_minutes, day.depth_credit_min), (0, 0, 0))
                self.assertEqual(day.known_cost_total, 0)
                self.assertEqual(day.planning_reserve_total, minimum_meal_cost(request))
                self.assertEqual(sum(s.duration_min for s in day.slots), 75)
            else:
                self.assertGreaterEqual(day.visit_minutes + day.depth_credit_min, program_depth_target(request) or 0)

    def test_seven_dates_with_two_or_three_full_days_and_no_input_mutation(self):
        for count in (2, 3):
            with self.subTest(active_days=count):
                request = request_for(7)
                grounding, facts, graph = fixture(request, count, 240)
                before = deepcopy((request.model_dump(), vars(grounding), facts, graph))
                result = self.build(request, grounding, facts, graph)
                self.assert_complete_calendar(result, request, count)
                self.assertTrue(result.search_complete)
                self.assertEqual(result.known_cost_total, count * 200)
                self.assertEqual(result.planning_reserve_total, 7 * 1400)
                self.assertEqual(before, (request.model_dump(), vars(grounding), facts, graph))

    def test_default_strict_still_refuses_partial_and_full_result_is_unchanged(self):
        request = request_for(7)
        grounding, facts, graph = fixture(request, 2, 240)
        strict = build_regional_skeleton(request, grounding, facts, travel_facts=graph)
        self.assertEqual(strict.status, "insufficient_content")
        self.assertEqual(strict.days, ())
        request = request_for(2)
        grounding, facts, graph = fixture(request, 2, 240)
        strict = build_regional_skeleton(request, grounding, facts, travel_facts=graph)
        opted_in = self.build(request, grounding, facts, graph)
        self.assertEqual(strict, opted_in)
        self.assertEqual(opted_in.status, "ready")
        self.assertTrue(all(not d.is_free_day for d in opted_in.days))

    def test_original_calendar_leading_free_days_and_real_active_transfer(self):
        request = request_for(7)
        grounding, facts, graph = fixture(request, 2, 240)
        facts = [replace(facts[0], date_windows=(windows_for(request)[2],)),
                 replace(facts[1], date_windows=(windows_for(request)[4],))]
        grounding.base_facts.append(dict(base_key="base_2", base_query="Город Б, Россия", lat=51, lon=100))
        grounding.confirmed_hints[1].update(base_key="base_2", allowed_base_keys=["base_2"])
        graph = [edge for edge in graph if "visit_2" not in (edge.from_key, edge.to_key)]
        graph.extend(TravelFact(a, b, 10, "walking", "routing", source="fixture-router",
                                group_cost=0, price_source="walking-no-fare")
                     for a, b in (("base_2", "visit_2"), ("visit_2", "base_2"), ("base_1", "base_2")))
        result = self.build(request, grounding, facts, graph)
        self.assert_complete_calendar(result, request, 2)
        self.assertEqual([d.day for d in result.days if not d.is_free_day], [3, 5])
        self.assertEqual([d.base_key for d in result.days], ["base_1"] * 4 + ["base_2"] * 3)
        transfer = result.days[4].slots[0].travel_fact
        self.assertEqual((transfer.from_key, transfer.to_key), ("base_1", "base_2"))
        for day in result.days:
            if day.is_free_day:
                self.assertTrue(all(s.travel_fact is None and s.visit_fact is None for s in day.slots))

    def test_free_days_never_hide_an_unconfirmed_base_change(self):
        request = request_for(7)
        grounding, facts, graph = fixture(request, 2, 240)
        facts = [replace(facts[0], date_windows=(windows_for(request)[0],)),
                 replace(facts[1], date_windows=(windows_for(request)[6],))]
        grounding.base_facts.append(dict(base_key="base_2", base_query="Город Б, Россия", lat=51, lon=100))
        grounding.confirmed_hints[1].update(base_key="base_2", allowed_base_keys=["base_2"])
        graph = [edge for edge in graph if "visit_2" not in (edge.from_key, edge.to_key)]
        graph.extend(TravelFact(a, b, 10, "walking", "routing", source="fixture-router",
                                group_cost=0, price_source="walking-no-fare")
                     for a, b in (("base_2", "visit_2"), ("visit_2", "base_2")))
        result = self.build(request, grounding, facts, graph)
        self.assert_complete_calendar(result, request, 1)
        self.assertEqual(len({d.base_key for d in result.days}), 1)

    def test_food_for_every_day_and_fixed_costs_are_reserved(self):
        request = request_for(7, budget=35200, travel_cost=10000, lodging_cost=15000)
        grounding, facts, graph = fixture(request, 2, 240)
        result = self.build(request, grounding, facts, graph)
        self.assert_complete_calendar(result, request, 2)
        self.assertEqual(result.known_cost_total + result.planning_reserve_total, 10200)
        tighter = request.model_copy(update={"budget": 35199})
        self.assert_complete_calendar(self.build(tighter, grounding, facts, graph), tighter, 1)
        impossible = request.model_copy(update={"budget": 34999})
        refused = self.build(impossible, grounding, facts, graph)
        self.assertNotIn(refused.status, ("ready", "limited_program"))
        self.assertEqual(refused.days, ())
        self.assertEqual((request.days_count, request.travel_cost, request.lodging_cost), (7, 10000, 15000))

    def test_no_all_free_success_and_no_lowered_active_depth(self):
        request = request_for(7)
        grounding, facts, graph = fixture(request, 1, 45)
        for supplied in ([], facts):
            with self.subTest(cards=len(supplied)):
                result = self.build(request, grounding, supplied, graph)
                self.assertNotIn(result.status, ("ready", "limited_program"))
                self.assertEqual(result.days, ())
        one_day = request_for(1)
        grounding, facts, graph = fixture(one_day, 1, 45)
        self.assertNotEqual(self.build(one_day, grounding, facts, graph).status, "limited_program")

    def test_active_days_still_require_map_windows_return_edges_and_safe_credit(self):
        request = request_for(7)
        grounding, facts, graph = fixture(request, 1, 240)
        variants = [
            (grounding, [replace(facts[0], date_windows=windows_for(request, "13:00", "14:00"))], graph),
            (grounding, facts, [edge for edge in graph if edge.from_key == "base_1"]),
            (grounding, facts, [replace(edge, minutes=5, depth_credit_min=6) for edge in graph]),
            (grounding, [replace(facts[0], place_query="Неподтверждённое место")], graph),
        ]
        for index, (ground, supplied, edges) in enumerate(variants):
            with self.subTest(variant=index):
                result = self.build(request, ground, supplied, edges)
                self.assertNotIn(result.status, ("ready", "limited_program"))
                self.assertEqual(result.days, ())

    def test_rain_children_interests_and_duplicate_identity_are_not_relaxed(self):
        request = request_for(7)
        grounding, facts, graph = fixture(request, 1, 240)
        rainy = [replace(facts[0], indoor=False, date_windows=(windows_for(request)[2],))]
        result = self.build(request, grounding, rainy, graph, rainy_dates=[windows_for(request)[2].date])
        self.assertEqual(result.days, ())
        rejected = [replace(facts[0], interest_categories=("shopping",))]
        self.assertEqual(self.build(request, grounding, rejected, graph).days, ())
        child_request = request.model_copy(update={"children_ages": [2]})
        rejected = [replace(facts[0], children_allowed=False)]
        self.assertEqual(self.build(child_request, grounding, rejected, graph).days, ())
        accepted = self.build(child_request, grounding, facts, graph)
        self.assertEqual(accepted.status, "limited_program")
        self.assertTrue(all(next(s.duration_min for s in d.slots if s.kind == "rest") == 60
                            for d in accepted.days))
        grounding, facts, graph = fixture(request, 2, 240)
        grounding.confirmed_hints[1].update(lat=grounding.confirmed_hints[0]["lat"],
                                            lon=grounding.confirmed_hints[0]["lon"])
        self.assert_complete_calendar(self.build(request, grounding, facts, graph), request, 1)

    def test_single_global_state_budget_never_publishes_an_incomplete_calendar(self):
        request = request_for(7)
        grounding, facts, graph = fixture(request, 3, 240)
        full = self.build(request, grounding, facts, graph)
        bounded_success = False
        for limit in range(1, full.search_states + 1):
            result = self.build(request, grounding, facts, graph, max_search_states=limit)
            self.assertLessEqual(result.search_states, limit)
            if result.status == "limited_program":
                active_count = sum(not d.is_free_day for d in result.days)
                self.assert_complete_calendar(result, request, active_count)
                bounded_success |= not result.search_complete
            else:
                self.assertEqual(result.days, ())
                self.assertEqual(result.status, "search_limit")
                self.assertFalse(result.search_complete)
        self.assertTrue(bounded_success)

    def test_opt_in_argument_is_strictly_boolean(self):
        request = request_for(7)
        grounding, facts, graph = fixture(request, 1, 240)
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                build_regional_skeleton(request, grounding, facts, travel_facts=graph, allow_free_days=value)


if __name__ == "__main__":
    unittest.main()
