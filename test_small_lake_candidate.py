"""Small named lakes are not a bypass for broad regional water bodies."""

import json
import unittest
from unittest.mock import patch

import candidate_grounding as grounding
import places
from test_candidate_grounding import batch, record, request, scope
from yandex_ai import GenerationResult


class SmallLakeCandidateTests(unittest.TestCase):
    def test_only_valid_small_box_containing_the_map_point_is_allowed(self):
        small = record("Сухое озеро, Долинный, Россия", poi_type="lake",
                       boundingbox=["50.999", "51.001", "100.999", "101.001"])
        self.assertTrue(grounding._small_named_lake(small))
        for bounds in (None, [], [50, 53, 100, 103], ["nan", 51.001, 100.999, 101.001],
                       [51.001, 50.999, 100.999, 101.001], [91, 92, 100, 101],
                       [51.1, 51.2, 101.1, 101.2], [False, 51, 100, 101],
                       [51, 51, 101, 101], [50.999, 51.001, 101.001, 100.999]):
            with self.subTest(bounds=bounds):
                self.assertFalse(grounding._small_named_lake({**small, "boundingbox": bounds}))
        self.assertFalse(grounding._small_named_lake({**small, "type": "island"}))

    def test_small_lake_passes_but_large_or_unbounded_lake_is_rejected(self):
        payload = batch(2)
        for index, poi in enumerate(payload["pois"]):
            poi["title"] = f"Озеро Лесное {index}"
            poi["place_query"] = f"Озеро Лесное {index}, Долинный, Россия"
        for shape, expected in (("small", 2), ("large", 0), ("missing", 0)):
            def base_lookup(query, **_):
                return record(query, poi_type="village")
            def nearby(title, country, *_, **_checks):
                index = int(title.rsplit(" ", 1)[-1])
                lat, lon = 51.0, 101.01 + index * 0.02
                result = record(f"{title}, Долинный, {country}", lat=lat, lon=lon,
                                poi_type="lake", category="water", locality="Долинный",
                                place_query=f"{title}, Долинный, {country}")
                if shape == "small":
                    result["boundingbox"] = [lat - .001, lat + .001, lon - .001, lon + .001]
                elif shape == "large":
                    result["boundingbox"] = [50, 53, 100, 103]
                return result
            with self.subTest(shape=shape), patch.object(places, "_cache", {}), patch.object(places, "_save_cache", side_effect=AssertionError("No cache writes")), patch.object(places.httpx, "get", side_effect=AssertionError("No network")), patch.object(places, "geocode", side_effect=base_lookup), patch.object(places, "geocode_nearby_named", side_effect=nearby):
                result = grounding.ground_region_candidates(request(1), scope(), completion_fn=lambda *_a, **_k: GenerationResult(json.dumps(payload)))
            self.assertEqual(len(result.confirmed_hints), expected)
            if not expected:
                self.assertEqual([item["reason"] for item in result.rejected], ["not_a_specific_poi"] * 2)

    def test_small_box_does_not_bypass_name_region_or_base(self):
        payload = batch(2)
        for failure in ("identity_mismatch", "outside_region", "outside_base_radius"):
            def base_lookup(query, **_):
                return record(query, poi_type="village")
            def nearby(title, country, *_, **_checks):
                lat, lon = (20.0, 20.0) if failure == "outside_region" else (51.0, 102.9) if failure == "outside_base_radius" else (51.0, 101.01)
                query = f"{title}, Долинный, {country}"
                value = record(query if failure != "identity_mismatch" else "Совсем другой объект, Долинный, Россия",
                               lat=lat, lon=lon, poi_type="lake", category="water", locality="Долинный", place_query=query,
                               boundingbox=[lat - .001, lat + .001, lon - .001, lon + .001])
                return value
            with self.subTest(failure=failure), patch.object(places, "_cache", {}), patch.object(places, "_save_cache", side_effect=AssertionError("No cache writes")), patch.object(places.httpx, "get", side_effect=AssertionError("No network")), patch.object(places, "geocode", side_effect=base_lookup), patch.object(places, "geocode_nearby_named", side_effect=nearby):
                result = grounding.ground_region_candidates(request(1), scope(), completion_fn=lambda *_a, **_k: GenerationResult(json.dumps(payload)))
            self.assertEqual(result.confirmed_hints, [])
            self.assertEqual([item["reason"] for item in result.rejected], [failure] * 2)


if __name__ == "__main__":
    unittest.main()
