"""Region cache hints, with an in-memory cache and all network/write paths blocked."""

import unittest
from unittest.mock import patch

import places
from destination_scope import RegionScope


def scope(geometry=None):
    return RegionScope(
        "Тестовая территория", "Россия", (0, 10, 0, 10),
        geometry or {"type": "Polygon", "coordinates": [
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
        ]},
    )


def record(title="Музей Северного края", locality="Посёлок Северный", *, lat=2, lon=2, **overrides):
    result = {
        "lat": lat, "lon": lon,
        "name": f"{title}, {locality}, Россия",
        "type": "museum", "category": "tourism",
        "address": {"village": locality, "country": "Россия", "country_code": "ru"},
    }
    result.update(overrides)
    return result


class RegionPlaceHintsTests(unittest.TestCase):
    def setUp(self):
        self.cache = {}
        for target, replacement in (
            ("_cache", self.cache),
            ("geocode", None),
            ("_search", None),
            ("_save_cache", None),
        ):
            patcher = patch.object(places, target, replacement) if target == "_cache" else patch.object(
                places, target, side_effect=AssertionError(f"{target} must not run"),
            )
            patcher.start()
            self.addCleanup(patcher.stop)
        network = patch.object(places.httpx, "get", side_effect=AssertionError("network must not run"))
        network.start()
        self.addCleanup(network.stop)

    def hints(self, **kwargs):
        return places.cached_place_hints("Тестовая территория, Россия", None, region_scope=scope(), **kwargs)

    def test_preserves_matched_cache_query_and_locality_without_rewriting_to_region(self):
        query = "Музей Северного края, Посёлок Северный, Россия"
        self.cache[query] = record()
        result = self.hints()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["place_query"], query)
        self.assertEqual(result[0]["title"], "Музей Северного края")
        self.assertNotIn("Тестовая территория", result[0]["place_query"])
        self.assertEqual(self.cache, {query: record()}, "hint extraction must not mutate the cache")

    def test_lowercase_stored_query_is_kept_verbatim(self):
        query = "музей северного края, посёлок северный, россия"
        self.cache[query] = record()
        self.assertEqual(self.hints()[0]["place_query"], query)

    def test_rejects_mismatched_name_locality_country_and_undocumented_poi_type(self):
        query = "Музей Северного края, Посёлок Северный, Россия"
        invalid = [
            record(title="Другой музей"),
            record(locality="Другой посёлок"),
            record(address={"village": "Посёлок Северный", "country_code": "fr"}),
            record(type=""), record(type="village", category="place"),
            record(category="boundary"),
        ]
        for value in invalid:
            with self.subTest(record=value):
                self.cache.clear()
                self.cache[query] = value
                self.assertEqual(self.hints(), [])

    def test_generic_names_and_settlements_are_not_attractions(self):
        for title, poi_type in (("Набережная", "attraction"), ("Музей", "museum"), ("Посёлок Северный", "village")):
            with self.subTest(title=title):
                self.cache.clear()
                self.cache[f"{title}, Северный район, Россия"] = record(title, "Северный район", type=poi_type)
                self.assertEqual(self.hints(), [])

    def test_rejects_invalid_coordinates(self):
        query = "Музей Северного края, Посёлок Северный, Россия"
        for coordinate in (float("nan"), float("inf"), 91, "bad"):
            with self.subTest(coordinate=coordinate):
                self.cache[query] = record(lat=coordinate)
                self.assertEqual(self.hints(), [])

    def test_full_polygon_not_bbox_controls_admission(self):
        triangle = {"type": "Polygon", "coordinates": [
            [[0, 0], [10, 0], [0, 10], [0, 0]],
        ]}
        query = "Музей Северного края, Посёлок Северный, Россия"
        self.cache[query] = record(lat=8, lon=8)
        self.assertEqual(places.cached_place_hints("Регион, Россия", None, region_scope=scope(triangle)), [])

    def test_polygon_holes_are_respected(self):
        geometry = {"type": "Polygon", "coordinates": [
            [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]],
            [[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]],
        ]}
        self.cache["Музей Северного края, Посёлок Северный, Россия"] = record(lat=5, lon=5)
        self.assertEqual(places.cached_place_hints("Регион, Россия", None, region_scope=scope(geometry)), [])

    def test_margin_is_40km_even_if_city_max_distance_is_larger(self):
        query = "Музей Северного края, Посёлок Северный, Россия"
        self.cache[query] = record(lat=-0.2, lon=2)
        self.assertEqual(len(self.hints()), 1)
        self.cache[query] = record(lat=-0.5, lon=2)
        self.assertEqual(self.hints(max_distance_km=1000), [])

    def test_missing_or_invalid_geometry_does_not_fall_back_to_bbox(self):
        self.cache["Музей Северного края, Посёлок Северный, Россия"] = record()
        for geometry in (None, {"type": "Point", "coordinates": [2, 2]}):
            with self.subTest(geometry=geometry):
                area = RegionScope("Регион", "Россия", (0, 10, 0, 10), geometry)
                self.assertEqual(places.cached_place_hints("Регион, Россия", None, region_scope=area), [])

    def test_negative_geometry_and_legacy_miss_records_are_never_hints(self):
        query = "Музей Северного края, Посёлок Северный, Россия"
        for suffix, value in (
            ("|miss", {"checked_at": 100, "reason": "not_found"}),
            ("|miss", {"checked_at": 100, "reason": "mismatch"}),
            ("|miss", True), ("|geometry", record()),
            ("", {"checked_at": 100, "reason": "mismatch"}),
        ):
            with self.subTest(suffix=suffix, value=value):
                self.cache.clear()
                self.cache[query + suffix] = value
                self.assertEqual(self.hints(), [])

    def test_fresh_negative_blocks_conflicting_positive_but_expired_negative_is_not_extended(self):
        query = "Музей Северного края, Посёлок Северный, Россия"
        self.cache[query] = record()
        negative = {"checked_at": 100, "reason": "mismatch"}
        self.cache[query + "|miss"] = negative
        with patch.object(places.time, "time", return_value=101):
            self.assertEqual(self.hints(), [])
        with patch.object(places.time, "time", return_value=100 + places.NEGATIVE_CACHE_TTL_SECONDS):
            self.assertEqual(len(self.hints()), 1)
        self.assertEqual(self.cache[query + "|miss"], negative)

    def test_same_name_in_different_localities_survives_but_physical_aliases_are_deduplicated(self):
        self.cache["Музей Северного края, Первый посёлок, Россия"] = record(locality="Первый посёлок")
        self.cache["Музей Северного края, Второй посёлок, Россия"] = record(locality="Второй посёлок", lon=3)
        self.cache["Музей Северного края, Район первого посёлка, Россия"] = record(locality="Район первого посёлка")
        self.assertEqual(len(self.hints()), 2)
        self.assertEqual(len(self.hints(limit=1)), 1)
        self.assertEqual(self.hints(limit=0), [])

    def test_city_branch_remains_unchanged_and_does_not_require_poi_type(self):
        query = "Музей Северного края, Посёлок Северный, Россия"
        value = record()
        value.pop("type")
        self.cache[query] = value
        self.assertEqual(places.cached_place_hints("Посёлок Северный, Россия", None), [])
        result = places.cached_place_hints("Посёлок Северный, Россия", (2, 2))
        self.assertEqual(result[0]["place_query"], query)
        self.assertEqual(result[0]["title"], "Музей Северного края")


if __name__ == "__main__":
    unittest.main()
