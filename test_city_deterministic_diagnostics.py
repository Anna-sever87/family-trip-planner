"""Finite, input-free diagnostics never relax timetable validation."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import city_deterministic as engine
from test_city_deterministic import CENTER, candidates, options, request


class DeterministicInputDiagnosticsTests(unittest.TestCase):
    def build(self, req=None, pool=None, **kwargs):
        return engine.build_deterministic_city_itinerary(
            request(1) if req is None else req,
            candidates(2) if pool is None else pool,
            resolved_city=CENTER, **kwargs)

    def assert_rejected_at(self, stage, **kwargs):
        with self.assertLogs("city_deterministic", level="WARNING") as captured:
            result = self.build(**kwargs)
        self.assertEqual(result.status, "invalid_request")
        self.assertIsNone(result.itinerary)
        self.assertEqual(result.coverage["invalid_input_stage"], stage)
        self.assertIn(stage, engine.INVALID_INPUT_STAGES)
        self.assertEqual(result.reasons, ["invalid_inputs"])
        self.assertEqual(result.coverage["quality_status"], "insufficient_data")
        self.assertEqual([record.getMessage() for record in captured.records],
                         [f"city_schedule_input_rejected stage={stage}"])
        self.assertTrue(all(record.exc_info is None for record in captured.records))
        return result

    def test_real_distinct_map_ids_with_same_canonical_query_are_context_failure(self):
        pool = candidates(2)
        duplicate_record = deepcopy(pool[0].record)
        duplicate_record["osm_id"] = pool[1].record["osm_id"]
        pool[1] = replace(pool[1], title=pool[0].title, place_query=pool[0].place_query,
                          record=duplicate_record)
        before = deepcopy(pool)
        self.assert_rejected_at("city_context", pool=pool)
        self.assertEqual(pool, before)

    def test_request_search_weather_and_pool_stages_remain_distinct(self):
        cases = [
            ("search_budget", {"max_states": 0}),
            ("request", {"req": request(1).model_copy(update={"budget": float("nan")})}),
            ("rain_dates", {"rainy_dates": ["2099-01-01"]}),
            ("candidate_pool", {"pool": candidates(25)}),
        ]
        for stage, kwargs in cases:
            with self.subTest(stage=stage):
                self.assert_rejected_at(stage, **kwargs)

    def test_priorities_and_interest_groups_never_accept_unknown_ids(self):
        self.assert_rejected_at("interest_groups", required_any_of_ids=["unknown"])
        self.assert_rejected_at("interest_groups", required_category_groups={"museums": ["unknown"]})
        self.assert_rejected_at("candidate_priority", candidate_priority=["unknown"])

    def test_option_structure_conditions_and_exclusion_stages_remain_distinct(self):
        self.assert_rejected_at("visit_options", options=[object()])
        pool = candidates(2)
        values = options(pool)
        values[0] = replace(values[0], duration_min=0)
        self.assert_rejected_at("visit_conditions", pool=pool, options=values)
        self.assert_rejected_at("blocked_places", blocked_places={"unexpected": "value"})

    def test_reserve_and_visit_limit_fail_closed(self):
        with patch.object(engine, "minimum_meal_cost", side_effect=[1400, ValueError("private marker")]):
            self.assert_rejected_at("meal_budget", options=options(candidates(2)))
        with patch("planner._max_attractions_for_request", return_value=6):
            self.assert_rejected_at("visit_limit")

    def test_untrusted_exception_text_never_reaches_logs_or_coverage(self):
        marker = "private-user-input-and-provider-body"
        with patch.object(engine, "build_city_context", side_effect=ValueError(marker)):
            result = self.assert_rejected_at("city_context")
        self.assertNotIn(marker, repr(result))

    def test_success_and_expected_insufficient_pool_have_no_invalid_input_diagnostic(self):
        with self.assertNoLogs("city_deterministic", level="WARNING"):
            successful = self.build()
            empty = self.build(pool=[])
        self.assertEqual(successful.status, "success")
        self.assertNotIn("invalid_input_stage", successful.coverage)
        self.assertEqual(empty.status, "insufficient_data")
        self.assertEqual(empty.reasons, ["empty_pool"])
        self.assertNotIn("invalid_input_stage", empty.coverage)

    def test_safe_discovery_preserves_only_bounded_duplicate_count(self):
        from generation_quality import safe_coverage

        for count in (0, 1, 24):
            raw = {"discovery": {"detail_rejection_reasons": {
                "duplicate_canonical_query": count, "private-provider-body": 1,
            }}}
            before = deepcopy(raw)
            self.assertEqual(safe_coverage(raw), {"discovery": {"detail_rejection_reasons": {
                "duplicate_canonical_query": count,
            }}})
            self.assertEqual(raw, before)
        for invalid in (True, -1, 100001, "private-provider-body", {"query": "private-user-text"}):
            with self.subTest(value=invalid):
                self.assertEqual(safe_coverage({"detail_rejection_reasons": {
                    "duplicate_canonical_query": invalid,
                }}), {"detail_rejection_reasons": {}})


if __name__ == "__main__":
    unittest.main()
