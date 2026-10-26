"""Adaptive verified reserves: real scheduling/identity checks, all external IO blocked."""

from contextlib import ExitStack
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import city_context
import city_coverage
import city_deterministic
import city_edit
import city_planner
import city_preferences
import city_schedule
from planner import build_itinerary_json_schema, build_itinerary_day_json_schema
from rules import city_candidate_limit, city_candidate_name_limit, check_rules, check_program_depth
from test_city_deterministic import CENTER, candidates, request, visits


class LongCityCapacityTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.external = [self.stack.enter_context(patch(name, side_effect=AssertionError("External IO forbidden")))
                         for name in ("httpx.get", "httpx.post", "places.geocode", "places._load_cache",
                                      "places._save_cache", "sqlite3.connect", "planner.generate_json")]

    def tearDown(self):
        for value in self.external: value.assert_not_called()

    def req(self, days=14, **changes):
        return request(days=days, children_ages=[14], end_time="18:00", budget=500000, **changes)

    def test_capacity_is_trip_bound_and_short_old_guard_is_unchanged(self):
        for days in range(1, 15):
            req = self.req(days)
            self.assertEqual(city_candidate_limit(req), 24 if days <= 8 else 3 * days)
            self.assertEqual(city_candidate_name_limit(req), min(36 if days <= 8 else 60, 4 * days + 8))
        with self.assertRaises(ValueError): city_candidate_limit(self.req().model_copy(update={"destination_kind": "region"}))

    def test_long_context_roundtrips_28_and_42_but_short25_and_long43_reject(self):
        for size in (28, 42):
            pool = candidates(size)
            original = deepcopy(pool)
            payload = city_context.build_city_context(self.req(), pool, CENTER)
            loaded, _ = city_context.load_city_context(payload, self.req())
            self.assertEqual(len(loaded), size)
            self.assertEqual(pool, original)
            loaded[0].record["lat"] = 0
            self.assertEqual(payload["candidates"][0]["record"]["lat"], 55.0)
            forged = deepcopy(payload); forged["candidates"][1] = deepcopy(forged["candidates"][0])
            with self.assertRaises(ValueError): city_context.load_city_context(forged, self.req())
            with self.assertRaises(ValueError): city_context.load_city_context(payload, self.req(8))
        for days, size in ((8, 25), (9, 28), (14, 43)):
            with self.subTest(days=days), self.assertRaises(ValueError):
                city_context.build_city_context(self.req(days), candidates(size), CENTER)

    def test_real_long_schedules_fill_all_dates_and_retain_exact_estimates(self):
        for size in (28, 42):
            with self.subTest(size=size):
                req, pool = self.req(), candidates(size, separation=0.001)
                original = deepcopy(pool)
                built = city_deterministic.build_deterministic_city_itinerary(req, pool, resolved_city=CENTER)
                self.assertEqual(built.problems, [], built.coverage)
                self.assertIsNotNone(built.itinerary)
                self.assertEqual(len(built.itinerary.days), 14)
                self.assertTrue(all(day.day_mode == "program" for day in built.itinerary.days), built.coverage)
                used = visits(built.itinerary)
                self.assertEqual(len({item.place_query for item in used}), len(used))
                self.assertTrue(all(item.duration_min == 90 for item in used))
                self.assertTrue(all(any(item.kind == "attraction" and item.time >= "14:00" for item in day.items)
                                    for day in built.itinerary.days), built.coverage)
                self.assertLessEqual(built.coverage["states_explored"], 20000)
                self.assertEqual(check_rules(built.itinerary, req, [], allow_city_limited_program=True), [])
                self.assertEqual(check_program_depth(built.itinerary, req, allow_city_limited_program=True), [])
                self.assertEqual(pool, original)
                self.assertEqual(len(used), size, built.coverage)

    def test_scheduler_rejects25_short43_long_without_using_large_search(self):
        for days, size in ((8, 25), (14, 43)):
            built = city_deterministic.build_deterministic_city_itinerary(self.req(days), candidates(size), resolved_city=CENTER)
            self.assertEqual(built.status, "invalid_request")
            self.assertIsNone(built.itinerary)

    def test_preferences_keep_long_server_reserve_and_short_rejects(self):
        no_ai = Mock(side_effect=AssertionError("Structured form needs no AI"))
        req = self.req(trip_goal="Познакомиться с городом", wishes="")
        for size in (28, 42):
            selected = city_preferences.select_city_preferences(req, candidates(size), completion=no_ai)
            self.assertEqual(selected.problems, [])
            self.assertEqual(len(selected.candidates), size)
        for days, size in ((8, 25), (14, 43)):
            selected = city_preferences.select_city_preferences(self.req(days), candidates(size), completion=no_ai)
            self.assertTrue(selected.problems)
        no_ai.assert_not_called()

    def test_schedule_schema_and_single_day_edit_accept_bound_long_pool(self):
        req, pool = self.req(), candidates(42, separation=0.001)
        canonical = build_itinerary_json_schema(req)
        wire = city_schedule.build_city_schedule_schema(canonical, req, pool)
        self.assertEqual(len(wire["properties"]["day_1"]["properties"]["attractions"]["items"]["properties"]["candidate_id"]["enum"]), 42)
        with self.assertRaises(ValueError):
            city_schedule.build_city_schedule_schema(build_itinerary_json_schema(self.req(8)), self.req(8), candidates(25))
        built = city_deterministic.build_deterministic_city_itinerary(req, pool, resolved_city=CENTER)
        saved_context = city_context.attach_city_presentation(
            city_context.build_city_context(req, pool, CENTER), built.itinerary, pool, req,
        )
        self.assertGreater(len(saved_context["presentation"]["managed_tip_sha256"]), 32)
        city_context.load_city_context(saved_context, req)
        city_context.validate_city_edit_context(saved_context, deepcopy(saved_context), req, built.itinerary)
        tampered = deepcopy(saved_context)
        tampered["candidates"][-1]["record"]["lon"] += 0.001
        with self.assertRaises(ValueError):
            city_context.validate_city_edit_context(saved_context, tampered, req, built.itinerary)
        excessive = deepcopy(saved_context)
        excessive["presentation"]["managed_tip_sha256"] = [f"{value:064x}" for value in range(45)]
        with self.assertRaises(ValueError): city_context.load_city_context(excessive, req)
        context = city_edit.prepare_city_edit(
            req, built.itinerary, 1, candidates=pool, resolved_city=CENTER,
            canonical_day_schema=build_itinerary_day_json_schema(req, 1),
        )
        self.assertEqual(len(context.candidates), 42)
        # Every non-target day remains reserved, and exact context integrity is
        # still enforced; a larger reserve is not edit authority over other days.
        elsewhere = {item.place_query for day in built.itinerary.days[1:] for item in day.items
                     if item.kind == "attraction"}
        self.assertTrue(all(candidate.place_query not in elsewhere for candidate in context.allowed_candidates))

    def test_witness_and_quality_diagnostics_have_the_same_bound(self):
        req, pool = self.req(), candidates(42)
        witness = city_coverage.build_city_program_witness(req, pool)
        self.assertLessEqual(witness.states_explored, 20000)
        with self.assertRaises(ValueError): city_coverage.build_city_program_witness(self.req(8), candidates(25))
        data = SimpleNamespace(candidates=pool, rejected=[], warnings=[], map_requests=36,
                               diagnostics={"pool_target": 42, "category_counts": {"museums": 42},
                                            "source_alternatives_confirmed": 42, "name_limit": 60})
        checked = city_planner._discovery_diagnostics(data)
        self.assertEqual(checked["pool_target"], 42)
        self.assertEqual(checked["category_counts"], {"museums": 42})


if __name__ == "__main__":
    unittest.main()
