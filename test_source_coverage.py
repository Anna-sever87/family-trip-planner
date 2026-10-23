"""Coverage explanations describe supplied evidence without relaxing planning."""

from dataclasses import replace
import socket
import sqlite3
import unittest
from unittest.mock import patch

from generation_feedback import generation_failure_reason
from regional_skeleton import assess_visit_coverage, build_regional_skeleton
import source_planning
from test_regional_skeleton import fixture, request_for, windows_for


class SourceCoverageTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "getaddrinfo"), (sqlite3, "connect")):
            blocker = patch.object(owner, name, side_effect=AssertionError("external I/O forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)

    def seven_days(self, count=7):
        request = request_for(7)
        grounding, facts, graph = fixture(request, count, 90)
        durations = (120, 150, 45, 45, 60, 90, 150, 90)
        facts = [replace(fact, duration_min=durations[i]) for i, fact in enumerate(facts)]
        graph = [replace(edge, depth_credit_min=5) for edge in graph]
        return request, grounding, facts, graph

    def prepare(self, request, grounding, facts, graph):
        cards = [dict(id=fact.visit_id, place_queries=[fact.place_query]) for fact in facts]
        with patch.object(source_planning, "load_visit_cards", return_value=cards), \
                patch.object(source_planning, "visit_facts_for_grounding", return_value=facts), \
                patch.object(source_planning, "build_travel_graph", return_value=(graph, [])) as routing, \
                patch.object(source_planning, "verify_selected_routes", return_value=[]):
            result = source_planning.prepare_source_plan(request, grounding)
        return result, routing

    def test_known_interest_rejection_is_not_missing_hours_evidence(self):
        request = request_for()
        grounding, facts, graph = fixture(request, 1, 240)
        facts = [replace(facts[0], interest_categories=("theatre",), date_windows=())]
        result = build_regional_skeleton(request, grounding, facts, travel_facts=graph)
        self.assertEqual(result.status, "insufficient_content")
        self.assertEqual(result.missing_facts, ())
        self.assertEqual(result.coverage.exclusions, (("visit_1", "interests_not_matched"),))
        self.assertEqual(result.excluded_visit_ids, ("visit_1",))

    def test_known_age_rejection_differs_from_unknown_child_permission(self):
        request = request_for(children_ages=[6])
        grounding, facts, graph = fixture(request, 1, 120)
        for change, status, rejection in (
            (dict(minimum_age=12), "insufficient_content", "minimum_age"),
            (dict(children_allowed=False), "insufficient_content", "children_not_eligible_or_unknown"),
            (dict(children_allowed=None), "needs_facts", None),
        ):
            with self.subTest(change=change):
                result = build_regional_skeleton(request, grounding, [replace(facts[0], **change)], travel_facts=graph)
                self.assertEqual(result.status, status)
                if rejection:
                    self.assertEqual(result.missing_facts, ())
                    self.assertEqual(result.coverage.exclusions, (("visit_1", rejection),))
                else:
                    self.assertEqual(result.coverage.exclusions, ())
                    self.assertTrue(result.missing_facts)

    def test_date_coverage_requires_whole_visit_inside_requested_window(self):
        request = request_for(2)
        grounding, facts, _ = fixture(request, 2, 120)
        facts = [replace(facts[0], date_windows=(windows_for(request)[0],)),
                 replace(facts[1], date_windows=(replace(windows_for(request)[1], start_time="20:00"),))]
        coverage = assess_visit_coverage(request, grounding, facts)
        self.assertEqual([row.unique_candidates for row in coverage.by_date], [1, 0])
        self.assertEqual((coverage.unique_candidates, coverage.available_visit_minutes), (1, 120))
        self.assertEqual((coverage.target_minutes_per_day, coverage.target_total_minutes), (240, 480))
        self.assertEqual(coverage.reason, "dates_without_visits")

    def test_seven_unique_visits_explain_first_day_capacity_not_region_impossibility(self):
        request, grounding, facts, graph = self.seven_days()
        result = build_regional_skeleton(request, grounding, facts, travel_facts=graph)
        self.assertEqual((result.status, result.days), ("insufficient_content", ()))
        self.assertEqual(result.coverage.reason, "first_day_depth_capacity")
        self.assertEqual((result.coverage.unique_candidates, result.coverage.available_visit_minutes), (7, 660))
        self.assertEqual(result.coverage.first_day_max_minutes, 160)
        # The strict solver still exposes this proof. The production adapter
        # now additionally permits an explicitly reported free-day fallback.
        plan = source_planning._source_failure(source_planning._skeleton_failure_marker(result),
                                              request, len(facts), result.status, coverage=result.coverage)
        reason = generation_failure_reason(plan.problems)
        self.assertIn("SOURCE_CANDIDATE_CAPACITY", plan.problems[0])
        self.assertIn("по одному посещению", reason)
        self.assertIn("160 мин при ориентире 240 мин", reason)
        self.assertIn("не доказывает невозможность поездки по региону", reason)
        self.assertEqual(plan.coverage, result.coverage)

    def test_irrelevant_missing_card_does_not_hide_proved_current_pool_capacity(self):
        request, grounding, facts, graph = self.seven_days(8)
        facts[-1] = replace(facts[-1], date_windows=())
        result = build_regional_skeleton(request, grounding, facts, travel_facts=graph)
        self.assertEqual(result.status, "insufficient_content")
        self.assertEqual(result.coverage.reason, "first_day_depth_capacity")
        self.assertIn("visit:visit_8:date_windows", result.missing_facts)

    def test_missing_potential_first_day_connection_prevents_capacity_claim(self):
        request, grounding, facts, graph = self.seven_days()
        graph = [edge for edge in graph if (edge.from_key, edge.to_key) != ("base_1", "visit_1")]
        result = build_regional_skeleton(request, grounding, facts, travel_facts=graph)
        self.assertEqual(result.status, "needs_facts")
        self.assertNotEqual(result.coverage.reason, "first_day_depth_capacity")
        self.assertIsNone(result.coverage.first_day_max_minutes)
        self.assertIn("travel:base_1->visit_1", result.missing_facts)

    def test_reserve_candidate_allows_multiple_visits_so_single_visit_proof_is_inapplicable(self):
        request, grounding, facts, graph = self.seven_days(8)
        result = build_regional_skeleton(request, grounding, facts, travel_facts=graph)
        self.assertNotEqual(result.coverage.reason, "first_day_depth_capacity")
        self.assertIsNone(result.coverage.first_day_max_minutes)

    def test_search_limit_keeps_coverage_without_claiming_infeasibility(self):
        request = request_for(2)
        grounding, facts, graph = fixture(request, 4, 120)
        result = build_regional_skeleton(request, grounding, facts, travel_facts=graph, max_search_states=1)
        self.assertEqual(result.status, "search_limit")
        self.assertEqual(result.coverage.unique_candidates, 4)
        plan = source_planning._source_failure(source_planning._skeleton_failure_marker(result), request, 4,
                                              result.status, coverage=result.coverage)
        reason = generation_failure_reason(plan.problems)
        self.assertIn("безопасного предела", reason)
        self.assertNotIn("по одному посещению", reason)

    def test_free_day_fallback_keeps_known_exclusions_out_of_the_program(self):
        request = request_for(2)
        grounding, facts, graph = fixture(request, 2, 240)
        facts[-1] = replace(facts[-1], interest_categories=("theatre",))
        plan, routing = self.prepare(request, grounding, facts, graph)
        self.assertEqual((plan.status, plan.problems), ("limited_program", []))
        self.assertEqual((plan.itinerary.program_report.program_days, plan.itinerary.program_report.free_days), (1, 1))
        self.assertEqual(plan.coverage.exclusions, (("visit_2", "interests_not_matched"),))
        self.assertEqual([item.place_query for day in plan.itinerary.days for item in day.items
                          if item.kind == "attraction"], [facts[0].place_query])
        routing.assert_called_once()

    def test_free_day_fallback_never_uses_the_candidate_with_missing_hours(self):
        request = request_for(2)
        grounding, facts, graph = fixture(request, 2, 240)
        facts[-1] = replace(facts[-1], date_windows=())
        plan, routing = self.prepare(request, grounding, facts, graph)
        self.assertEqual((plan.status, plan.problems), ("limited_program", []))
        self.assertEqual((plan.itinerary.program_report.program_days, plan.itinerary.program_report.free_days), (1, 1))
        self.assertEqual(plan.coverage.exclusions, ())
        self.assertIn("visit:visit_2:date_windows", plan.coverage.missing_facts)
        self.assertEqual([item.place_query for day in plan.itinerary.days for item in day.items
                          if item.kind == "attraction"], [facts[0].place_query])
        routing.assert_called_once()

    def test_two_day_schedule_and_costs_are_unchanged(self):
        request = request_for(2)
        grounding, facts, graph = fixture(request, 4, 120)
        result = build_regional_skeleton(request, grounding, facts, travel_facts=graph)
        self.assertEqual(result.status, "ready")
        self.assertEqual([day.visit_ids for day in result.days], [("visit_1", "visit_2"), ("visit_3", "visit_4")])
        self.assertEqual((result.known_cost_total, result.planning_reserve_total), (800, 2800))
        self.assertEqual(result.coverage.reason, "")

    def test_public_capacity_details_reject_unknown_duplicate_and_unbounded_values(self):
        prefix = ("SOURCE_PLAN_UNAVAILABLE: SOURCE_CANDIDATE_CAPACITY requestedDays=7 usableCards=7 "
                  "uniqueCandidates=7 targetMinutes=240 availableMinutes=660 ")
        for suffix in ("coverageReason=SECRET firstDayMaxMinutes=160",
                       "coverageReason=first_day_depth_capacity firstDayMaxMinutes=99999",
                       "coverageReason=first_day_depth_capacity firstDayMaxMinutes=True",
                       "coverageReason=first_day_depth_capacity firstDayMaxMinutes=160 firstDayMaxMinutes=160",
                       "coverageReason=first_day_depth_capacity coverageReason=SECRET firstDayMaxMinutes=160"):
            reason = generation_failure_reason([prefix + suffix])
            self.assertNotIn("SECRET", reason)
            self.assertNotIn("по одному посещению", reason)
            self.assertNotIn("99999", reason)
        limited = generation_failure_reason([prefix.replace("SOURCE_CANDIDATE_CAPACITY", "SOURCE_SEARCH_LIMIT")
                                             + "coverageReason=first_day_depth_capacity firstDayMaxMinutes=160"])
        self.assertNotIn("по одному посещению", limited)


if __name__ == "__main__":
    unittest.main()
