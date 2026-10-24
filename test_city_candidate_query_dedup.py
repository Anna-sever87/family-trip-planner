"""Observed query collisions are filtered before semantic selection, offline."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import city_candidates
import city_context
import city_deterministic
import city_planner
import planner
from test_city_candidates import CENTER, request
import test_city_map_discovery as discovery_fixture


class CanonicalQueryDedupTests(unittest.TestCase):
    def setUp(self):
        self.provider = discovery_fixture.MapFirstDiscoveryTests(
            "test_balanced_map_first_pool_has_all_requested_types_without_ai_names")
        self.provider.setUp()
        self.addCleanup(self.provider.doCleanups)

    def pool(self):
        return self.provider.pool(days=1, interest_categories=["museums"])

    def add_colliding_records(self):
        self.provider.add("museums", "Музей Северный", 111)
        self.provider.add("museums", "Музей Северный", 112)
        self.provider.add("museums", "Музей Южный", 113)

    def test_true_grounding_then_full_planner_sees_only_unique_canonical_queries(self):
        self.add_colliding_records()
        value = request(1, children_ages=[], start_time="11:00", end_time="20:00",
                        trip_goal="Подробно изучить музеи", wishes="Музеи без повторов",
                        interest_categories=["museums"])
        payloads = []

        def completion(messages, **kwargs):
            data = json.loads(messages[1]["content"])
            payloads.append(data)
            if "titles" in kwargs["json_schema"]["properties"]:
                answer = {"titles": []}
            else:
                seen = data["verified_candidates"]
                self.assertEqual(len(seen), 2)
                self.assertEqual({item["candidate_id"] for item in seen}, {"osm-node-111", "osm-node-113"})
                answer = {"status": "selected", "candidate_ids": [item["candidate_id"] for item in seen]}
            return SimpleNamespace(text=json.dumps(answer), input_tokens=0, output_tokens=0)

        with patch.object(city_planner, "find_city", return_value=CENTER), \
                patch.object(city_planner, "get_weather", return_value=None), \
                patch.object(city_planner, "city_deterministic_enabled", return_value=True), \
                patch.object(planner, "_load_blocked_places", return_value=[]), \
                patch.object(planner, "generate_json", side_effect=completion) as ai:
            result = planner.generate_itinerary(value, max_attempts=3, city_candidates_first=True)
        self.assertEqual((result.generation_status, result.problems), ("success", []))
        self.assertEqual((ai.call_count, result.attempts), (2, 2))
        self.assertEqual(len(payloads), 2)
        visits = [item for day in result.itinerary.days for item in day.items if item.kind == "attraction"]
        self.assertEqual(len(visits), 2)
        self.assertEqual(len({item.place_query for item in visits}), 2)
        candidates, _ = city_context.load_city_context(result.city_map_context, value)
        self.assertEqual({item.candidate_id for item in candidates}, {"osm-node-111", "osm-node-113"})
        diagnostics = result.generation_coverage["discovery"]
        self.assertGreaterEqual(diagnostics["detail_rejection_reasons"]["duplicate_canonical_query"], 1)

    def test_category_fallback_deduplicates_without_merging_provider_facts(self):
        self.add_colliding_records()
        self.provider.overpass_failure = True
        self.provider.records[1]["namedetails"]["name:en"] = "Second record only"
        self.provider.records[1]["lon"] = "105.04"
        original = deepcopy(self.provider.records)
        pool = self.pool()
        self.assertEqual(pool.status, "success")
        self.assertEqual([item.candidate_id for item in pool.candidates], ["osm-node-111", "osm-node-113"])
        first = pool.candidates[0]
        self.assertEqual(first.record["lon"], float(original[0]["lon"]))
        self.assertNotIn("name:en", first.record["namedetails"])
        self.assertEqual(self.provider.records, original)
        self.assertGreaterEqual(pool.diagnostics["detail_rejection_reasons"]["duplicate_canonical_query"], 1)
        self.assertTrue(all(set(item) == {"reason"} for item in pool.rejected
                            if item["reason"] == "duplicate_canonical_query"))
        self.assertLessEqual(pool.map_requests, city_candidates.MAX_MAP_REQUESTS)

    def test_distant_same_named_branch_keeps_first_without_claiming_same_identity(self):
        self.add_colliding_records()
        self.provider.records[1]["lon"] = "105.10"  # Different branch, still inside the city radius.
        pool = self.pool()
        self.assertEqual([item.candidate_id for item in pool.candidates], ["osm-node-111", "osm-node-113"])
        self.assertEqual(pool.candidates[0].record["lon"], 105.03)
        self.assertIn({"reason": "duplicate_canonical_query"}, pool.rejected)

    def test_different_canonical_localities_with_same_name_are_not_collapsed(self):
        self.add_colliding_records()
        self.provider.records[1]["address"]["city"] = "Соседний город"
        self.provider.records[1]["display_name"] = "Музей Северный, Соседний город, Россия"
        pool = self.pool()
        self.assertEqual(len(pool.candidates), 3)
        self.assertEqual(len({item.place_query for item in pool.candidates}), 3)
        self.assertFalse(any(item["reason"] == "duplicate_canonical_query" for item in pool.rejected))

    def test_rejected_country_type_or_radius_cannot_reserve_a_query(self):
        for invalid in ("country", "type", "radius"):
            with self.subTest(invalid=invalid):
                self.provider.records.clear()
                self.add_colliding_records()
                first = self.provider.records[0]
                if invalid == "country":
                    first["address"].update(country="Китай", country_code="cn")
                elif invalid == "type":
                    first.update(category="highway", type="residential")
                else:
                    first.update(lat="1", lon="1")
                pool = self.pool()
                self.assertEqual({item.candidate_id for item in pool.candidates}, {"osm-node-112", "osm-node-113"})

    def test_hard_snapshot_duplicate_query_guard_remains_strict(self):
        self.add_colliding_records()
        pool = self.pool()
        first = pool.candidates[0]
        other = replace(first, candidate_id="osm-node-112", record=deepcopy(first.record))
        other.record["osm_id"] = 112
        value = request(1, children_ages=[], start_time="11:00", end_time="20:00",
                        interest_categories=["museums"])
        with self.assertRaises(ValueError):
            city_context.build_city_context(value, [first, other], CENTER)
        result = city_deterministic.build_deterministic_city_itinerary(value, [first, other], resolved_city=CENTER)
        self.assertEqual((result.status, result.itinerary), ("invalid_request", None))
        self.assertIn("invalid_inputs", result.reasons)


if __name__ == "__main__":
    unittest.main()
