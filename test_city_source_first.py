"""Source-first discovery regression proofs; every external dependency is mocked."""
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest

import httpx
import city_candidates as candidates
import test_city_candidates as fixtures

TITLE, CENTER, AREA = fixtures.TITLE, fixtures.CENTER, fixtures.AREA
request, record, response = fixtures.request, fixtures.record, fixtures.response


class CitySourceFirstTests(unittest.TestCase):
    setUp = fixtures.CityCandidateTests.setUp
    patched = fixtures.CityCandidateTests.patched
    run_pool = fixtures.CityCandidateTests.run_pool

    def seed(self, title=TITLE):
        self.hints.return_value = [{"title": title, "source_url": "https://example.invalid/official-name"}]

    def test_source_is_grounded_before_completion_not_only_first_after_completion(self):
        self.seed()
        def complete(*args, **kwargs):
            self.assertEqual(self.get.call_count, 1)
            self.assertEqual(self.get.call_args.kwargs["params"]["q"], f"{TITLE}, Россия")
            return SimpleNamespace(text='{"titles": []}', input_tokens=1, output_tokens=2)
        self.completion.side_effect = complete
        result = self.run_pool()
        self.assertEqual((result.status, len(result.candidates), result.map_requests), ("success", 1, 1))

    def test_ai_unavailable_keeps_usable_source_pool_in_partial_mode(self):
        self.seed()
        self.completion.side_effect = RuntimeError("untrusted provider detail")
        result = candidates.ground_city_candidates(request(days=4), CENTER, completion=self.completion,
                                                   allow_partial_program=True)
        self.assertEqual((result.status, len(result.candidates), result.problems), ("success", 1, []))
        self.assertTrue(result.warnings)
        self.assertNotIn("untrusted provider detail", str(result))
        self.assertEqual((result.map_requests, result.usage["completion_calls"]), (1, 1))

    def test_bad_ai_json_cannot_inject_names_or_discard_source_records(self):
        self.seed()
        for text in ('{"titles": [], "titles": ["Injected"]}', '{"titles": [NaN]}',
                     '{"titles": [], "indoor": true}', '{broken'):
            with self.subTest(text=text):
                self.completion.return_value.text = text
                before = self.get.call_count
                result = self.run_pool()
                self.assertEqual([item.title for item in result.candidates], [TITLE])
                self.assertEqual((result.status, result.problems), ("success", []))
                self.assertTrue(result.warnings)
                self.assertEqual(self.get.call_count - before, 1)

    def test_strict_mode_does_not_claim_incomplete_pool_is_sufficient(self):
        self.seed()
        self.completion.side_effect = RuntimeError("offline")
        result = self.run_pool(days=2)
        self.assertEqual((result.status, len(result.candidates)), ("ai_unavailable", 1))
        self.assertTrue(result.problems)

    def test_no_source_keeps_strict_invalid_ai_response_failure_and_zero_map(self):
        self.completion.return_value.text = '{"titles": [], "extra": true}'
        result = self.run_pool()
        self.assertEqual((result.status, result.map_requests), ("invalid_ai_response", 0))
        self.assertTrue(result.problems)
        self.get.assert_not_called()

    def test_entire_name_budget_checked_as_sources_skips_unusable_ai_supplement(self):
        titles = [f"Музей ремёсел {index}" for index in range(6)]
        self.hints.return_value = [{"title": title, "source_url": "https://example.invalid/name"} for title in titles]
        self.get.side_effect = [response([record(title, identifier=index + 1)]) for index, title in enumerate(titles)]
        result = self.run_pool()
        self.assertEqual((len(result.candidates), result.map_requests, result.usage["completion_calls"]), (6, 6, 0))
        self.completion.assert_not_called()

    def test_seed_and_ai_alias_same_provider_identity_remains_one_visit(self):
        self.seed()
        other = "Музей иных ремёсел"
        self.get.side_effect = [response([record()]), response([record(other)])]
        result = self.run_pool([other])
        self.assertEqual(len(result.candidates), 1)

    def test_empty_response_is_distinct_from_country_name_type_and_locality_failure(self):
        cases = [([], "empty_result", ""),
                 ([record(title="Другой музей")], "record_mismatch", "name_mismatch"),
                 ([record(address={"country_code": "tr"})], "record_mismatch", "country_mismatch_or_missing"),
                 ([record(lat="55")], "record_mismatch", "outside_city_radius"),
                 ([record(type="hotel")], "record_mismatch", "unsupported_poi_type"),
                 ([record(address={"country_code": "ru", "country": "Россия"})], "record_mismatch", "missing_locality"),
                 ([record(osm_id=None)], "record_mismatch", "missing_provider_identity")]
        for values, detail, reason in cases:
            with self.subTest(reason=reason):
                self.get.return_value = response(values)
                result = self.run_pool()
                if reason == "name_mismatch":
                    # The requested alias still failed. A separate, genuinely
                    # named OSM observation may enter only under its own name.
                    self.assertEqual(result.status, "success")
                    self.assertEqual([item.title for item in result.candidates], ["Другой музей"])
                    self.assertEqual(result.candidates[0].candidate_id, "osm-node-123")
                    self.assertNotEqual(result.candidates[0].title, TITLE)
                    self.assertEqual(result.rejected[0]["title"], TITLE)
                    self.assertIsNone(candidates._canonical_record(TITLE,
                        candidates._from_provider(values[0]), dict(CENTER, country_code="ru")))
                    self.assertEqual(result.diagnostics["source_alternatives_confirmed"], 1)
                else:
                    self.assertEqual(result.status, "insufficient_data")
                    self.assertFalse(result.candidates)
                    self.assertEqual(result.diagnostics["source_alternatives_confirmed"], 0)
                self.assertEqual(result.rejected[0]["detail"], detail)
                self.assertEqual(result.rejected[0]["record_reasons"], reason)

    def test_network_exhaustion_has_typed_status_without_extra_attempts(self):
        self.get.side_effect = httpx.ConnectTimeout("offline")
        result = self.run_pool()
        self.assertEqual((result.status, result.map_requests), ("map_unavailable", 2))

    def test_source_records_and_names_are_not_mutated(self):
        self.seed()
        names = deepcopy(self.hints.return_value)
        raw = record()
        original = deepcopy(raw)
        self.get.return_value = response([raw])
        result = self.run_pool([])
        result.candidates[0].record["address"]["city"] = "Changed"
        self.assertEqual(raw, original)
        self.assertEqual(self.hints.return_value, names)


if __name__ == "__main__":
    unittest.main()
