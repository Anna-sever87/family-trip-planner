"""Regional search logs finite aggregate diagnostics, never requests or names."""
from contextlib import ExitStack
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from candidate_grounding import GroundingResult
import planner
import test_grounded_planner as fixtures
from yandex_ai import YandexAIError


class RegionGroundingDiagnosticTests(unittest.TestCase):
    def test_projection_counts_reasons_and_cache_reservation_without_exposing_values(self):
        private = "private user query / secret token"
        result = GroundingResult(
            base_facts=[{"query": private}], confirmed_hints=[{"title": private}],
            rejected=[{"kind": "base", "reason": "outside_region", "query": private},
                      {"kind": "poi", "reason": "limited_search_incomplete", "query": private},
                      {"kind": "poi", "reason": "limited_search_incomplete", "base_key": private},
                      {"kind": private, "reason": private}, {"kind": [], "reason": []}],
            problems=[private], incomplete=True, completion_calls=1,
            verification_queries=5, geocode_calls=3, cache_only_checks=4, cache_only_added=1,
            source_roster_used=True, source_clusters_omitted=[private],
            reserved_http_attempts_upper_bound=6, http_attempt_limit=42, query_limit=21,
        )
        before = deepcopy(result)
        data = planner._regional_grounding_diagnostics(result, blocked_count=2)
        self.assertEqual(data["rejection_reasons"], {"outside_region": 1, "limited_search_incomplete": 2, "other": 2})
        self.assertEqual(data["rejection_kinds"], {"base": 1, "poi": 2, "other": 2})
        self.assertEqual(data["problem_codes"], {"other": 1})
        self.assertEqual((data["base_count"], data["poi_count"], data["blocked_poi_count"]), (1, 1, 2))
        self.assertEqual((data["geocode_calls"], data["reserved_http_attempts_upper_bound"]), (3, 6))
        self.assertEqual((data["cache_only_checks"], data["cache_only_added"], data["source_clusters_omitted_count"]), (4, 1, 1))
        self.assertNotIn("http_requests", data)  # Reservation is not an actual HTTP count.
        self.assertNotIn(private, json.dumps(data))
        self.assertEqual(result, before)

    def test_invalid_counter_values_never_become_log_text_or_booleans(self):
        result = GroundingResult(completion_calls="private", geocode_calls=True,
                                 verification_queries=-1, cache_only_checks=10**9)
        data = planner._regional_grounding_diagnostics(result, blocked_count="private")
        self.assertEqual((data["completion_calls"], data["geocode_calls"], data["verification_queries"], data["blocked_poi_count"]), (0, 0, 0, 0))
        self.assertEqual(data["cache_only_checks"], 100_000)

    def run_pre_map_failure(self, response):
        fixture = fixtures.GroundedPlannerTests()
        with ExitStack() as stack:
            generate, lookup, verify = fixture._context(stack, [response])
            with self.assertLogs("planner", level="INFO") as logs:
                result = planner.generate_itinerary(fixtures.regional._request(), ground_candidates=True)
            self.assertEqual((generate.call_count, result.attempts), (1, 1))
            lookup.assert_not_called()
            verify.assert_not_called()
        lines = [record.getMessage() for record in logs.records if record.getMessage().startswith("Региональный подбор: ")]
        self.assertEqual(len(lines), 1)
        return json.loads(lines[0].split(": ", 1)[1]), result

    def test_actual_service_error_has_distinct_safe_code_before_map_or_schedule(self):
        data, result = self.run_pre_map_failure(YandexAIError("private response body"))
        self.assertEqual(data["problem_codes"], {"candidate_service_unavailable": 1})
        self.assertEqual((data["verification_queries"], data["reserved_http_attempts_upper_bound"]), (0, 0))
        self.assertEqual(result.itinerary.days, [])
        self.assertNotIn("private", json.dumps(data))

    def test_actual_invalid_batch_has_distinct_safe_code_before_map_or_schedule(self):
        data, result = self.run_pre_map_failure('{"untrusted user text": "private"}')
        self.assertEqual(data["problem_codes"], {"invalid_candidate_batch": 1})
        self.assertTrue(data["incomplete"])
        self.assertEqual(data["rejected_count"], 0)
        self.assertEqual(result.itinerary.days, [])

    def test_real_search_rejections_are_logged_without_changing_validation(self):
        fixture = fixtures.GroundedPlannerTests()
        payload = fixtures._batch(fixtures._source())
        with ExitStack() as stack:
            generate, lookup, _ = fixture._context(stack, [json.dumps(payload)],
                payload=payload, unconfirmed={poi["title"] for poi in payload["pois"]})
            with self.assertLogs("planner", level="INFO") as logs:
                result = planner.generate_itinerary(fixtures.regional._request(), ground_candidates=True)
        line = next(record.getMessage() for record in logs.records if record.getMessage().startswith("Региональный подбор: "))
        data = json.loads(line.split(": ", 1)[1])
        self.assertEqual(data["rejection_reasons"], {"limited_search_incomplete": 14})
        self.assertEqual((data["base_count"], data["poi_count"], data["geocode_calls"]), (2, 0, 16))
        self.assertEqual((generate.call_count, lookup.call_count, result.attempts), (1, 14, 1))
        self.assertTrue(result.problems)
        self.assertEqual(result.itinerary.days, [])
        self.assertNotIn(payload["pois"][0]["title"], line)


if __name__ == "__main__":
    unittest.main()
