"""Offline regressions for the water-body coastal planning convention.

All map facts are synthetic and held in memory. No live geocoder, AI, database,
or persistent cache is permitted; the convention is not an official boundary.
"""

from copy import deepcopy
from dataclasses import replace
import json
import socket
import sqlite3
import unittest
from unittest.mock import Mock, patch

import candidate_grounding as grounding
import places
from destination_scope import (
    RegionScope, _distance_from_bounds, build_region_context,
    check_region_itinerary, region_margin_km, resolve_region,
)
from grounded_schedule import build_grounded_messages
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest
from yandex_ai import GenerationResult


LOCALITY = "Береговой посёлок"
BASE_QUERY = f"{LOCALITY}, Россия"
NEAR_TITLE = "Музей прибрежной истории"
FAR_TITLE = "Музей лесных промыслов"


def lake_scope(area_type="lake"):
    return RegionScope("Тестовый водоём", "Россия", (50, 50.1, 100, 100.1), {
        "type": "Polygon", "coordinates": [[
            [100, 50], [100.1, 50], [100.1, 50.1], [100, 50.1], [100, 50],
        ]],
    }, area_type=area_type)


def address():
    return {"village": LOCALITY, "country": "Россия", "country_code": "ru"}


def base_record(lat=50.105):
    return {"name": BASE_QUERY, "lat": lat, "lon": 100.05,
            "type": "village", "category": "place", "address": address()}


def poi_record(title=NEAR_TITLE, lat=50.12):
    query = f"{title}, {LOCALITY}, Россия"
    return {"name": query, "place_query": query, "locality": LOCALITY,
            "lat": lat, "lon": 100.05, "type": "museum", "category": "tourism",
            "address": address()}


def base_fact(lat=50.105):
    return {"base_key": "base_1", "base_query": BASE_QUERY,
            "official_name": BASE_QUERY, "lat": lat, "lon": 100.05,
            "map_address": address()}


def request():
    return TripRequest(destination="Тестовый водоём, Россия", destination_kind="region",
                       start_date="2099-06-01", end_date="2099-06-01", budget=100000)


def batch():
    return {"bases": [{"base_key": "base_1", "base_query": BASE_QUERY}],
            "pois": [{"title": title, "place_query": poi_record(title)["place_query"],
                      "locality": LOCALITY, "base_key": "base_1"}
                     for title in (NEAR_TITLE, FAR_TITLE)]}


def itinerary(poi_lat=50.12):
    record = poi_record(lat=poi_lat)
    return Itinerary(title="Тестовый маршрут", summary="Проверка географии", days=[
        ItineraryDay(day=1, date="2099-06-01", theme="Музей", base_area=BASE_QUERY, items=[
            ItineraryItem(time="12:00", kind="attraction", title=NEAR_TITLE,
                          description="Посещение", duration_min=60, verified=True,
                          place_query=record["place_query"], lat=record["lat"], lon=record["lon"],
                          official_name=record["name"]),
        ]),
    ])


class LakeScopeMarginTests(unittest.TestCase):
    def setUp(self):
        self.cache = {}
        boundaries = [
            patch.object(places, "_cache", self.cache),
            patch.object(places, "_remember_nearby_alias", return_value=None),
            patch.object(grounding, "destination_name_hints", return_value=[]),
        ]
        for owner, name in (
            (places, "_save_cache"), (places, "_search"), (places, "geocode"),
            (places, "geocode_nearby_named"), (grounding, "generate_json"),
            (socket.socket, "connect"), (socket, "getaddrinfo"), (sqlite3, "connect"),
            (places.httpx, "get"), (places.httpx, "post"),
        ):
            boundaries.append(patch.object(owner, name, side_effect=AssertionError(f"{name} forbidden")))
        for boundary in boundaries:
            boundary.start()
            self.addCleanup(boundary.stop)

    def cache_pois(self):
        for record in (poi_record(), poi_record(FAR_TITLE, 50.25)):
            self.cache[record["place_query"].lower()] = record

    def run_grounding(self, scope, *, base_lat=50.105):
        self.cache[BASE_QUERY.lower()] = base_record(base_lat)
        self.cache_pois()
        complete = Mock(return_value=GenerationResult(json.dumps(batch(), ensure_ascii=False)))
        result = grounding.ground_region_candidates(request(), scope, max_http_attempts=0,
                                                    completion_fn=complete)
        self.assertEqual(result.geocode_calls, 0)
        self.assertEqual(result.reserved_http_attempts_upper_bound, 0)
        return result, json.loads(complete.call_args.args[0][1]["content"])

    def test_policy_uses_map_type_and_preserves_four_argument_legacy_scopes(self):
        for kind in ("lake", "water", "administrative", "national_park", "island", ""):
            with self.subTest(kind=kind):
                self.assertEqual(region_margin_km(lake_scope(kind)), 5 if kind in ("lake", "water") else 40)
        scope = lake_scope()
        legacy = RegionScope(scope.name, scope.country, scope.bounds, scope.geometry)
        self.assertEqual(legacy.area_type, "")
        self.assertEqual(region_margin_km(legacy), 40)
        self.assertEqual(region_margin_km(replace(legacy, name="Озеро по названию")), 40)

    def test_resolver_retains_validated_map_feature_type(self):
        area = lake_scope()
        for kind in ("lake", "water", "administrative", "national_park"):
            with self.subTest(kind=kind), patch.object(places, "geocode", return_value={
                "name": area.name, "boundingbox": list(area.bounds),
                "geojson": area.geometry, "type": kind,
            }) as geocode:
                resolved = resolve_region("Тестовый водоём, Россия")
                self.assertEqual(resolved.area_type, kind)
                self.assertEqual(resolved.geometry, area.geometry)
                geocode.assert_called_once_with("Тестовый водоём, Россия", include_geometry=True)

    def test_small_lake_margin_is_measured_from_edge_not_centroid_or_size(self):
        area = RegionScope("Малый водоём", "Россия", (50, 50.002, 100, 100.002), {
            "type": "Polygon", "coordinates": [[
                [100, 50], [100.002, 50], [100.002, 50.002], [100, 50.002], [100, 50],
            ]],
        }, "water")
        self.assertLess(_distance_from_bounds(50.02, 100.001, area), region_margin_km(area))
        self.assertGreater(_distance_from_bounds(50.06, 100.001, area), region_margin_km(area))
        self.assertEqual(_distance_from_bounds(50.001, 100.001, area), 0)

    def test_full_polygon_not_bounding_box_controls_coastal_admission(self):
        area = RegionScope("Треугольный водоём", "Россия", (50, 51, 100, 101), {
            "type": "Polygon", "coordinates": [[[100, 50], [101, 50], [100, 51], [100, 50]]],
        }, "lake")
        record = poi_record(lat=50.65)
        record["lon"] = 100.65  # Inside the bbox but far outside the triangular body.
        self.cache[record["place_query"].lower()] = record
        distance = _distance_from_bounds(record["lat"], record["lon"], area)
        self.assertGreater(distance, 5)
        self.assertLess(distance, 40)
        self.assertEqual(places.cached_place_hints(request().destination, None, region_scope=area), [])
        self.assertEqual(len(places.cached_place_hints(request().destination, None,
                                                      region_scope=replace(area, area_type="administrative"))), 1)

    def test_full_polygon_hole_is_not_treated_as_water(self):
        area = RegionScope("Водоём с островом", "Россия", (49, 52, 99, 102), {
            "type": "Polygon", "coordinates": [
                [[99, 49], [102, 49], [102, 52], [99, 52], [99, 49]],
                [[100, 50], [101, 50], [101, 51], [100, 51], [100, 50]],
            ],
        }, "lake")
        self.assertGreater(_distance_from_bounds(50.5, 100.5, area), 5)
        self.assertEqual(_distance_from_bounds(49.5, 100.5, area), 0)

    def test_final_validator_rejects_inland_base_but_other_regions_keep_40km(self):
        with patch.object(places, "geocode", return_value=base_record(50.25)):
            errors = check_region_itinerary(itinerary(), lake_scope(), start_time="08:00", end_time="21:00")
            self.assertEqual(len(errors), 1, errors)
            self.assertIn("база", errors[0])
            self.assertIn("за пределами", errors[0])
            self.assertEqual(check_region_itinerary(itinerary(), lake_scope("administrative"),
                                                    start_time="08:00", end_time="21:00"), [])

    def test_final_validator_rejects_inland_poi_but_preserves_coastal_poi(self):
        with patch.object(places, "geocode", return_value=base_record()):
            errors = check_region_itinerary(itinerary(50.25), lake_scope(), start_time="08:00", end_time="21:00")
            self.assertEqual(len(errors), 1, errors)
            self.assertIn(NEAR_TITLE, errors[0])
            self.assertIn("за пределами", errors[0])
            self.assertEqual(check_region_itinerary(itinerary(), lake_scope(),
                                                    start_time="08:00", end_time="21:00"), [])
            self.assertEqual(check_region_itinerary(itinerary(50.25), lake_scope("administrative"),
                                                    start_time="08:00", end_time="21:00"), [])

    def test_newly_confirmed_poi_uses_the_same_scope_gate(self):
        bases = {"base_1": base_fact()}
        for kind, accepted in (("lake", False), ("water", False), ("administrative", True)):
            with self.subTest(kind=kind):
                fact, reason = grounding._confirmed_fact(FAR_TITLE, poi_record(FAR_TITLE, 50.25),
                                                         bases, lake_scope(kind), grounding.GroundingResult())
                self.assertEqual(fact is not None, accepted)
                self.assertEqual(reason, "" if accepted else "outside_region")
        fact, reason = grounding._confirmed_fact(NEAR_TITLE, poi_record(), bases,
                                                 lake_scope(), grounding.GroundingResult())
        self.assertIsNotNone(fact)
        self.assertEqual(reason, "")

    def test_cache_hint_extraction_is_narrow_and_read_only(self):
        self.cache_pois()
        before = deepcopy(self.cache)
        hints = places.cached_place_hints(request().destination, None, region_scope=lake_scope())
        self.assertEqual([item["title"] for item in hints], [NEAR_TITLE.lower()])
        self.assertEqual(len(places.cached_place_hints(request().destination, None,
                                                      region_scope=lake_scope("administrative"))), 2)
        self.assertEqual(self.cache, before)

    def test_warm_cache_reuse_cannot_restore_inland_poi(self):
        self.cache_pois()
        hints = [{"title": title, "place_query": poi_record(title)["place_query"]}
                 for title in (NEAR_TITLE, FAR_TITLE)]
        for kind, count in (("lake", 1), ("administrative", 2)):
            with self.subTest(kind=kind):
                result = grounding.GroundingResult()
                grounding._reuse_cached_hints(hints, {"base_1": base_fact()}, lake_scope(kind), result, 18)
                self.assertEqual(len(result.confirmed_hints), count)
                self.assertEqual(result.geocode_calls, 0)
                self.assertEqual(result.reserved_http_attempts_upper_bound, 0)
                if count == 1:
                    self.assertEqual(result.confirmed_hints[0]["title"], NEAR_TITLE)

    def test_candidate_pipeline_filters_cached_poi_and_inland_base_without_http(self):
        result, _ = self.run_grounding(lake_scope())
        self.assertEqual(len(result.base_facts), 1)
        self.assertEqual([item["title"] for item in result.confirmed_hints], [NEAR_TITLE])
        result, _ = self.run_grounding(lake_scope(), base_lat=50.25)
        self.assertEqual(result.base_facts, [])
        self.assertEqual(result.confirmed_hints, [])
        self.assertTrue(any(item["reason"] == "outside_region" for item in result.rejected), result.rejected)
        result, _ = self.run_grounding(lake_scope("administrative"), base_lat=50.25)
        self.assertEqual(len(result.base_facts), 1)
        self.assertEqual(len(result.confirmed_hints), 2)

    def test_all_three_prompt_contexts_share_the_policy(self):
        for kind, margin in (("lake", 5), ("water", 5), ("administrative", 40), ("", 40)):
            with self.subTest(kind=kind):
                area = lake_scope(kind)
                initial = build_region_context(area)
                result, candidate = self.run_grounding(area)
                schedule = json.loads(build_grounded_messages(request(), area, result, 5)[1]["content"])
                self.assertEqual(initial["area_type"], kind)
                self.assertEqual(candidate["region"]["area_type"], kind)
                self.assertEqual(schedule["region"]["area_type"], kind)
                self.assertEqual(initial["limits_km"]["outside_territory_margin"], margin)
                self.assertEqual(candidate["region"]["outside_territory_margin_km"], margin)
                self.assertEqual(schedule["region"]["maximum_outside_boundary_km"], margin)
                self.assertTrue(any("не официальная" in rule for rule in initial["planning_rules"]))


if __name__ == "__main__":
    unittest.main()
