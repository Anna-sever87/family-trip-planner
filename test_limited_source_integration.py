"""Explicit free-day source plans retain the whole calendar and all hard gates."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import socket
import sqlite3
import unittest
from unittest.mock import patch

import planner
import source_planning
from candidate_grounding import GroundingResult
from grounded_schedule import enforce_grounded_choices
from regional_skeleton import build_regional_skeleton
from rules import check_program_depth, check_rules
from schemas import Itinerary
from test_regional_skeleton import fixture, request_for
import test_regional_generation_repair as regional


class LimitedSourceIntegrationTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "create_connection"),
                            (socket, "getaddrinfo"), (sqlite3, "connect")):
            blocker = patch.object(owner, name, side_effect=AssertionError("real I/O forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)

    def scenario(self):
        request = request_for(7, budget=60000, travel_cost=10000, lodging_cost=15000)
        grounding, facts, graph = fixture(request, 4, 120)
        cards = [dict(id=fact.visit_id, place_queries=[fact.place_query]) for fact in facts]
        return request, grounding, facts, graph, cards

    def prepare(self, request, grounding, facts, graph, cards, *, route_errors=()):
        with patch.object(source_planning, "load_visit_cards", return_value=cards), \
                patch.object(source_planning, "visit_facts_for_grounding", return_value=facts), \
                patch.object(source_planning, "build_travel_graph", return_value=(graph, [])), \
                patch.object(source_planning, "verify_selected_routes", return_value=list(route_errors)):
            return source_planning.prepare_source_plan(request, grounding)

    def test_source_short_pool_yields_all_dates_explicit_free_days_and_full_food_budget(self):
        request, grounding, facts, graph, cards = self.scenario()
        before = deepcopy((request.model_dump(), vars(grounding), facts, graph))
        plan = self.prepare(request, grounding, facts, graph, cards)
        self.assertEqual((plan.status, plan.problems), ("limited_program", []))
        itinerary = plan.itinerary
        report = itinerary.program_report
        self.assertEqual((report.requested_days, report.program_days, report.free_days), (7, 2, 5))
        self.assertEqual((report.eligible_unique_visits, report.selected_unique_visits), (4, 4))
        self.assertEqual([day.day for day in itinerary.days], list(range(1, 8)))
        self.assertEqual((itinerary.days[0].date, itinerary.days[-1].date),
                         (request.start_date, request.end_date))
        self.assertEqual(sum(item.cost for day in itinerary.days for item in day.items), 10600)
        self.assertEqual(request.fixed_costs, 25000)
        for day in itinerary.days:
            if day.day_mode == "free":
                self.assertEqual([item.kind for item in day.items], ["meal", "rest"])
                self.assertEqual(day.transfer_note, "")
                self.assertEqual(sum(item.cost for item in day.items), 1400)
                self.assertTrue(all(not item.place_query for item in day.items))
        self.assertEqual(enforce_grounded_choices(itinerary, grounding), [])
        self.assertEqual(check_rules(itinerary, request, allow_limited_program=True), [])
        self.assertEqual(check_program_depth(itinerary, request, plan.depth_by_day,
                                            allow_limited_program=True), [])
        self.assertTrue(check_rules(itinerary, request), "Default/model validation stays strict")
        self.assertEqual(before, (request.model_dump(), vars(grounding), facts, graph))
        restored = Itinerary.model_validate_json(itinerary.model_dump_json())
        self.assertEqual(restored.model_dump(), itinerary.model_dump())

    def test_selected_road_failure_is_not_masked_by_a_free_day_report(self):
        request, grounding, facts, graph, cards = self.scenario()
        plan = self.prepare(request, grounding, facts, graph, cards, route_errors=["Unknown ferry"])
        self.assertEqual(plan.itinerary.days, [])
        self.assertTrue(plan.problems)
        self.assertIn("SOURCE_ROUTES_UNCONFIRMED", plan.problems[0])

    def test_adapter_cannot_publish_all_free_or_mismatched_status(self):
        request, grounding, facts, graph, _ = self.scenario()
        skeleton = build_regional_skeleton(request, grounding, facts, travel_facts=graph,
                                           allow_free_days=True)
        self.assertEqual(skeleton.status, "limited_program")
        for broken in (replace(skeleton, status="ready"),
                       replace(skeleton, days=tuple(replace(day, is_free_day=True) for day in skeleton.days)),
                       replace(skeleton, days=skeleton.days[:-1])):
            with self.subTest(broken=broken.status), self.assertRaises(ValueError):
                source_planning.itinerary_from_skeleton(request, grounding, broken)

    def test_main_generator_does_not_reject_short_source_pool_or_request_an_ai_schedule(self):
        request, grounding, facts, graph, cards = self.scenario()
        source = self.prepare(request, grounding, facts, graph, cards)
        grounded = GroundingResult(base_facts=grounding.base_facts, confirmed_hints=grounding.confirmed_hints,
                                   completion_calls=1, input_tokens=11, output_tokens=13)
        with ExitStack() as stack:
            for name, value in (("resolve_region", regional._scope()), ("get_weather", None),
                                ("find_city", None), ("cached_place_hints", []), ("curated_place_hints", []),
                                ("curated_blocked_place_terms", []), ("ground_region_candidates", grounded),
                                ("prepare_source_plan", source)):
                stack.enter_context(patch.object(planner, name, return_value=value))
            generate = stack.enter_context(patch.object(planner, "generate_json",
                                                       side_effect=AssertionError("No AI schedule")))
            verify = stack.enter_context(patch.object(planner, "_map_verification_problems", return_value=[]))
            result = planner.generate_itinerary(request, ground_candidates=True, max_attempts=3)
        self.assertEqual(result.problems, [])
        self.assertEqual(result.itinerary.program_report.free_days, 5)
        self.assertEqual((result.attempts, result.prompt_tokens, result.completion_tokens), (1, 11, 13))
        generate.assert_not_called()
        verify.assert_called_once()

    def test_source_edit_rechecks_free_days_without_fictitious_self_edges(self):
        request, grounding, facts, graph, cards = self.scenario()
        itinerary = self.prepare(request, grounding, facts, graph, cards).itinerary
        for day in itinerary.days:
            for item in day.items:
                if item.kind == "attraction":
                    hint = next(hint for hint in grounding.confirmed_hints if hint["place_query"] == item.place_query)
                    item.verified, item.lat, item.lon = True, hint["lat"], hint["lon"]
        with patch.object(source_planning, "visit_facts_for_grounding", return_value=facts), \
                patch.object(source_planning, "build_travel_graph", return_value=(graph, [])), \
                patch.object(source_planning, "verify_route_edges", return_value=[]) as verify, \
                patch("places.geocode", return_value={"lat": 50, "lon": 100}):
            problems, credit = source_planning.validate_source_edit(request, itinerary)
        self.assertEqual(problems, [])
        self.assertEqual(set(credit), set(range(1, 8)))
        self.assertTrue(all(credit[day.day] == 0 for day in itinerary.days if day.day_mode == "free"))
        self.assertTrue(all(edge.from_key != edge.to_key for edge in verify.call_args.args[0]))
        self.assertEqual(check_rules(itinerary, request, allow_limited_program=True), [])

    def test_model_only_generation_cannot_authorize_its_own_free_day_report(self):
        from schemas import LimitedProgramReport
        source = regional._valid_source()
        source.source_schedule = True
        source.days[-1].day_mode = "free"
        # Deliberately inconsistent model metadata is well-formed JSON, but
        # cannot survive the production model-response trust boundary.
        source.program_report = LimitedProgramReport(requested_days=7, program_days=6, free_days=1,
                                                     eligible_unique_visits=7, selected_unique_visits=7)
        result, generate, _, observed = regional.RegionalRepairGenerationTests()._generate(
            [regional._model_response_json(source)], [[]])
        self.assertEqual(result.problems, [])
        self.assertFalse(result.itinerary.source_schedule)
        self.assertIsNone(result.itinerary.program_report)
        self.assertTrue(all(day.day_mode == "program" for day in result.itinerary.days))
        self.assertEqual(generate.call_count, 1)
        self.assertIsNone(observed[0][0].program_report)


if __name__ == "__main__":
    unittest.main()
