"""Exact nearby-name geocoding: real matcher, mocked HTTP and in-memory cache."""

from copy import deepcopy
import unittest
from unittest.mock import Mock, patch

import httpx

import places


class NearbyNamedGeocodeTests(unittest.TestCase):
    title = "Музей северных промыслов"
    country = "Россия"
    locality = "Долинный"
    lat = 51.0
    lon = 101.0

    def setUp(self):
        self.cache = {}
        self._patch("_cache", self.cache)
        self.save = self._patch("_save_cache")
        self._patch("_wait_for_turn")
        self._patch("time.sleep")
        self.get = self._patch("httpx.get", side_effect=AssertionError("HTTP must be mocked"))

    def _patch(self, name, *args, **kwargs):
        patcher = patch("places." + name, *args, **kwargs)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    @property
    def canonical(self):
        return f"{self.title}, {self.locality}, {self.country}"

    def raw_record(self, **overrides):
        record = {
            "lat": "51.02", "lon": "101.03", "display_name": self.canonical,
            "type": "museum", "category": "tourism",
            "address": {"village": self.locality, "country": self.country, "country_code": "ru"},
        }
        record.update(overrides)
        return record

    def cached_record(self, **overrides):
        record = self.raw_record(**overrides)
        record["name"] = record.pop("display_name")
        record["lat"], record["lon"] = float(record["lat"]), float(record["lon"])
        return record

    def response(self, records):
        response = Mock()
        response.json.return_value = records
        return response

    def lookup(self, **kwargs):
        return places.geocode_nearby_named(self.title, self.country, self.lat, self.lon, **kwargs)

    def mock_records(self, records):
        self.get.side_effect = None
        self.get.return_value = self.response(records)

    def test_exact_name_country_bounded_viewbox_and_canonical_positive_alias(self):
        self.mock_records([self.raw_record()])
        result = self.lookup()
        self.get.assert_called_once()
        params = self.get.call_args.kwargs["params"]
        self.assertEqual(params["q"], f"{self.title}, {self.country}")
        self.assertEqual((params["bounded"], params["addressdetails"], params["namedetails"]), (1, 1, 1))
        west, north, east, south = map(float, params["viewbox"].split(","))
        self.assertLess(west, self.lon)
        self.assertGreater(east, self.lon)
        self.assertLess(south, self.lat)
        self.assertGreater(north, self.lat)
        self.assertAlmostEqual(places.distance_km(self.lat, self.lon, north, self.lon), 80, places=6)
        self.assertEqual((result["place_query"], result["locality"]), (self.canonical, self.locality))
        self.assertIn(self.canonical.lower(), self.cache)
        self.assertNotIn(f"{self.title}, {self.country}".lower(), self.cache)
        self.assertNotIn("place_query", self.cache[self.canonical.lower()])
        self.assertNotIn("locality", self.cache[self.canonical.lower()])
        self.save.assert_called_once_with()

    def test_first_named_result_outside_circle_is_skipped_for_nearby_match(self):
        far = self.raw_record(lat="52.5")
        near = self.raw_record()
        self.mock_records([far, near])
        result = self.lookup()
        self.assertEqual((result["lat"], result["lon"]), (51.02, 101.03))
        self.get.assert_called_once()

    def test_county_from_actual_address_is_accepted_without_inventing_village(self):
        county = "Тестовый район"
        name = f"{self.title}, {county}, {self.country}"
        self.mock_records([self.raw_record(display_name=name, address={
            "county": county, "country": self.country, "country_code": "ru",
        })])
        result = self.lookup()
        self.assertEqual(result["locality"], county)
        self.assertEqual(result["place_query"], name)
        self.assertNotIn("village", result["address"])

    def test_missing_actual_locality_cannot_create_canonical_alias(self):
        for address in ({}, {"country": self.country, "country_code": "ru"}, None):
            with self.subTest(address=address):
                self.mock_records([self.raw_record(address=address)])
                with self.assertRaises(places.GeocodeMismatch):
                    self.lookup()
        self.assertEqual(self.cache, {})
        self.save.assert_not_called()

    def test_unusable_type_or_locality_does_not_hide_later_valid_result(self):
        for invalid in (self.raw_record(type=""), self.raw_record(address={})):
            with self.subTest(invalid=invalid):
                self.cache.clear()
                self.mock_records([invalid, self.raw_record(lat="51.04")])
                result = self.lookup()
                self.assertEqual(result["lat"], 51.04)
                self.assertEqual(result["place_query"], self.canonical)

    def test_wrong_name_country_and_invalid_coordinates_are_rejected(self):
        records = [
            self.raw_record(display_name=f"Другой музей, {self.locality}, {self.country}"),
            self.raw_record(address={"village": self.locality, "country": "Canada", "country_code": "ca"}),
            self.raw_record(lat="nan"), self.raw_record(lon="inf"), self.raw_record(lat="91"),
        ]
        for record in records:
            with self.subTest(record=record):
                self.mock_records([record])
                with self.assertRaises(places.GeocodeMismatch):
                    self.lookup()
        self.assertEqual(self.cache, {})
        self.save.assert_not_called()

    def test_matching_name_outside_custom_radius_is_not_confirmed(self):
        self.mock_records([self.raw_record(lat="51.2")])
        with self.assertRaises(places.GeocodeMismatch):
            self.lookup(max_distance_km=10)
        self.assertEqual(self.cache, {})
        self.save.assert_not_called()

    def test_positive_canonical_cache_reused_without_network_or_mutation(self):
        self.mock_records([self.raw_record()])
        first = self.lookup()
        before = deepcopy(self.cache)
        self.get.reset_mock()
        self.get.side_effect = AssertionError("Must reuse positive canonical cache")
        self.save.reset_mock()
        second = self.lookup()
        self.assertEqual(second, first)
        self.assertEqual(self.cache, before)
        second["address"]["village"] = "Changed return value"
        self.assertEqual(self.cache, before)
        self.get.assert_not_called()
        self.save.assert_not_called()

    def test_cached_record_without_type_does_not_short_circuit_search(self):
        stale = self.cached_record(type="")
        self.cache[self.canonical.lower()] = stale
        self.mock_records([self.raw_record()])
        result = self.lookup()
        self.get.assert_called_once()
        self.assertEqual(result["type"], "museum")
        self.assertEqual(self.cache[self.canonical.lower()]["type"], "museum")

    def test_cache_under_wrong_key_cannot_supply_named_match(self):
        self.cache["другое место, россия"] = self.cached_record()
        self.mock_records([])
        before = deepcopy(self.cache)
        self.assertIsNone(self.lookup())
        self.get.assert_called_once()
        self.assertEqual(self.cache, before)
        self.save.assert_not_called()

    def test_success_removes_only_its_canonical_negative(self):
        canonical_miss = self.canonical.lower() + "|miss"
        broad_miss = f"{self.title}, {self.country}".lower() + "|miss"
        self.cache[canonical_miss] = {"checked_at": 100, "reason": "not_found"}
        self.cache[broad_miss] = {"checked_at": 100, "reason": "not_found"}
        self.cache["other|miss"] = {"checked_at": 100, "reason": "mismatch"}
        self.mock_records([self.raw_record()])
        self.lookup()
        self.assertNotIn(canonical_miss, self.cache)
        self.assertIn(broad_miss, self.cache)
        self.assertIn("other|miss", self.cache)
        self.save.assert_called_once_with()

    def test_empty_or_mismatched_search_does_not_poison_existing_cache(self):
        self.cache[self.canonical.lower() + "|miss"] = {"checked_at": 100, "reason": "not_found"}
        self.cache["unrelated, россия"] = {"name": "Unrelated"}
        before = deepcopy(self.cache)
        self.mock_records([])
        self.assertIsNone(self.lookup())
        self.mock_records([self.raw_record(lat="52.5")])
        with self.assertRaises(places.GeocodeMismatch):
            self.lookup()
        self.assertEqual(self.cache, before)
        self.save.assert_not_called()

    def test_transient_failure_retries_only_once_and_can_recover(self):
        self.get.side_effect = [httpx.ReadTimeout("offline test"), self.response([self.raw_record()])]
        result = self.lookup()
        self.assertEqual(self.get.call_count, 2)
        self.assertEqual(result["place_query"], self.canonical)
        self.save.assert_called_once_with()

    def test_repeated_transport_failure_stops_at_two_without_negative_cache(self):
        self.get.side_effect = httpx.ReadTimeout("offline test")
        with self.assertRaises(places.GeocodeUnavailable):
            self.lookup()
        self.assertEqual(self.get.call_count, 2)
        self.assertEqual(self.cache, {})
        self.save.assert_not_called()

    def test_invalid_name_coordinates_and_radius_fail_before_search(self):
        invalid = [
            ("Музей", self.country, self.lat, self.lon, 80),
            (self.title + ", другое место", self.country, self.lat, self.lon, 80),
            (self.title, "", self.lat, self.lon, 80),
            (self.title, self.country, float("nan"), self.lon, 80),
            (self.title, self.country, self.lat, 181, 80),
            (self.title, self.country, True, self.lon, 80),
            (self.title, self.country, self.lat, self.lon, 0),
            (self.title, self.country, self.lat, self.lon, 81),
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                places.geocode_nearby_named(*arguments)
        self.get.assert_not_called()
        self.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
