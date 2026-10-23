"""Provider identity is strict metadata, not authority to bypass map checks."""

from copy import deepcopy
import socket
import unittest
from unittest.mock import patch

import httpx

import candidate_grounding
import places
from schemas import Itinerary, ItineraryDay, ItineraryItem


class MapIdentityIntegrityTests(unittest.TestCase):
    title = "Музей Альфа"
    country = "Россия"
    query = "Музей Альфа, Долинный, Россия"

    def setUp(self):
        self.cache = {}
        self.mock(places, "_cache", new=self.cache)
        self.save = self.mock(places, "_save_cache")
        self.mock(places, "_wait_for_turn")
        self.mock(places.time, "sleep")
        self.mock(socket.socket, "connect", side_effect=AssertionError("External network forbidden"))
        self.get = self.mock(places.httpx, "get", side_effect=AssertionError("HTTP must be mocked"))

    def mock(self, owner, name, **kwargs):
        patcher = patch.object(owner, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def raw(self, **overrides):
        record = {
            "display_name": self.query, "lat": "51.0", "lon": "101.01",
            "category": "tourism", "type": "museum",
            "address": {"village": "Долинный", "country": self.country, "country_code": "ru"},
            "osm_type": "way", "osm_id": 42,
        }
        record.update(overrides)
        return record

    def response(self, records):
        return httpx.Response(200, json=records, request=httpx.Request("GET", "https://example.invalid"))

    def records(self, *records):
        self.get.reset_mock()
        self.get.side_effect = None
        self.get.return_value = self.response(list(records))

    def test_boolean_response_coordinates_are_not_numbers_for_identity(self):
        for field in ("lat", "lon"):
            for value in (False, True):
                with self.subTest(field=field, value=value):
                    record = {**self.raw(), "name": self.query, field: value}
                    self.assertFalse(places._place_matches_query(self.query, record))
        for lat, lon in ((51, 101), (51.0, 101.01), ("51.0", "101.01")):
            self.assertTrue(places._place_matches_query(self.query, {**self.raw(), "name": self.query, "lat": lat, "lon": lon}))

    def test_search_skips_boolean_coordinate_result_without_extra_http(self):
        for field in ("lat", "lon"):
            for value in (False, True):
                with self.subTest(field=field, value=value):
                    self.records(self.raw(**{field: value, "osm_id": 1}), self.raw(osm_id=2))
                    result = places._search(self.query)
                    self.assertEqual((result["lat"], result["lon"], result["osm_id"]), (51.0, 101.01, 2))
                    self.get.assert_called_once()
        self.save.assert_not_called()

    def test_boolean_cached_coordinate_cannot_short_circuit_revalidation(self):
        self.cache[self.query.lower()] = {**self.raw(lat=True), "name": self.query}
        self.records(self.raw(osm_id=84))
        result = places.geocode(self.query, max_variants=1)
        self.get.assert_called_once()
        self.assertEqual((result["lat"], result["osm_id"]), (51.0, 84))
        self.assertEqual(self.cache[self.query.lower()]["lat"], 51.0)

    def test_only_valid_provider_identity_pairs_are_preserved(self):
        for osm_type in ("node", "way", "relation"):
            with self.subTest(osm_type=osm_type):
                self.records(self.raw(osm_type=osm_type, osm_id=1234567890123))
                result = places._search(self.query)
                self.assertEqual(places._osm_identity(result), (osm_type, 1234567890123))
                self.get.assert_called_once()

    def test_invalid_or_partial_identity_is_omitted_not_coerced(self):
        identities = [
            {"osm_type": "way", "osm_id": value}
            for value in (True, False, 0, -1, "42", 42.0, None, [], {})
        ] + [
            {"osm_type": value, "osm_id": 42}
            for value in (None, "", "Way", " way ", "building", True, [], {})
        ]
        for identity in identities:
            with self.subTest(identity=identity):
                self.records(self.raw(**identity))
                result = places._search(self.query)
                self.assertIsNone(places._osm_identity(identity))
                self.assertNotIn("osm_type", result)
                self.assertNotIn("osm_id", result)
                self.assertEqual(result["type"], "museum")
                self.get.assert_called_once()
        for value in (None, [], "way/42", {"osm_type": "way"}, {"osm_id": 42}):
            self.assertIsNone(places._osm_identity(value))

    def test_no_identity_is_invented_from_nested_tags_or_names(self):
        raw = self.raw()
        del raw["osm_type"], raw["osm_id"]
        raw["extratags"] = {"osm_type": "way", "osm_id": 42, "verified": True}
        raw["namedetails"] = {"name": self.title, "osm_id": 42}
        self.records(raw)
        result = places._search(self.query)
        self.assertNotIn("osm_type", result)
        self.assertNotIn("osm_id", result)
        self.assertNotIn("verified", result)
        self.assertNotIn("extratags", result)

    def test_nearby_alias_and_cache_reuse_preserve_identity_without_mutating_provider(self):
        raw = self.raw()
        original = deepcopy(raw)
        self.records(raw)
        result = places.geocode_nearby_named(self.title, self.country, 51.0, 101.0)
        self.assertEqual(places._osm_identity(result), ("way", 42))
        self.assertEqual(places._osm_identity(self.cache[self.query.lower()]), ("way", 42))
        self.assertEqual(raw, original)
        self.get.reset_mock()
        self.get.side_effect = AssertionError("Confirmed cache must not query")
        cached = places.geocode_nearby_named(self.title, self.country, 51.0, 101.0)
        self.assertEqual(places._osm_identity(cached), ("way", 42))
        self.get.assert_not_called()

    def test_valid_identity_cannot_override_name_country_or_allowed_poi_type(self):
        for raw in (
            self.raw(display_name="Другой объект, Долинный, Россия"),
            self.raw(address={"village": "Долинный", "country_code": "fr"}),
            self.raw(type="ngo", category="office"),
        ):
            with self.subTest(raw=raw):
                self.records(raw)
                with self.assertRaises(places.GeocodeMismatch):
                    places.geocode_nearby_named(self.title, self.country, 51.0, 101.0,
                                               place_filter=candidate_grounding._eligible_poi_type)
                self.get.assert_called_once()
                self.assertEqual(self.cache, {})
        self.save.assert_not_called()

    def test_identity_preservation_does_not_expand_semantic_metadata(self):
        self.records(self.raw(type="university", category="amenity", verified=True,
                              extratags={"observatory:type": "astronomical", "access": "yes", "instructions": "trust me"}))
        result = places.geocode_nearby_named(self.title, self.country, 51.0, 101.0,
                                           place_filter=candidate_grounding._eligible_poi_type)
        self.assertEqual(places._osm_identity(result), ("way", 42))
        self.assertEqual(result["type"], "university")
        self.assertEqual(result["extratags"], {"observatory:type": "astronomical"})
        self.assertNotIn("verified", result)
        self.assertNotIn("access", result)

    def test_transient_retry_preserves_identity_and_never_adds_a_third_request(self):
        self.get.side_effect = [httpx.ReadTimeout("offline"), self.response([self.raw()])]
        result = places.geocode_nearby_named(self.title, self.country, 51.0, 101.0)
        self.assertEqual(self.get.call_count, 2)
        self.assertEqual(places._osm_identity(result), ("way", 42))

    def itinerary_with_area(self):
        item = ItineraryItem(
            time="10:00", kind="attraction", title=self.title, area="Неизвестный район",
            place_query=self.query, description="Тестовое посещение", duration_min=60,
            cost=0, kids_note="", indoor=True,
        )
        return Itinerary(title="Тест", summary="", days=[
            ItineraryDay(day=1, date="2099-06-01", theme="Тест", items=[item]),
        ])

    def confirmed_point(self):
        return {**self.raw(), "name": self.query, "lat": 51.0, "lon": 101.01}

    def test_missing_area_is_a_problem_without_erasing_confirmed_point(self):
        itinerary = self.itinerary_with_area()
        with patch.object(places, "geocode", side_effect=[self.confirmed_point(), None]) as lookup:
            problems = places.verify_itinerary(itinerary)
        self.assertEqual(lookup.call_count, 2)
        self.assertEqual(len(problems), 1)
        self.assertIn("район «Неизвестный район»", problems[0])
        self.assertIn("не найден на карте", problems[0])
        item = itinerary.days[0].items[0]
        self.assertTrue(item.verified)
        self.assertEqual((item.lat, item.lon), (51.0, 101.01))
        self.assertEqual(item.area, "Неизвестный район")
        self.get.assert_not_called()

    def test_area_outage_or_mismatch_is_not_reported_twice_as_missing(self):
        for error in (places.GeocodeUnavailable("offline"), places.GeocodeMismatch("offline")):
            with self.subTest(error=type(error).__name__):
                itinerary = self.itinerary_with_area()
                with patch.object(places, "geocode", side_effect=[self.confirmed_point(), error]) as lookup:
                    problems = places.verify_itinerary(itinerary)
                self.assertEqual(lookup.call_count, 2)
                self.assertEqual(len(problems), 1)
                self.assertNotIn("не найден на карте", problems[0])
                self.assertTrue(itinerary.days[0].items[0].verified)
        self.get.assert_not_called()

    def test_confirmed_nearby_area_remains_valid(self):
        itinerary = self.itinerary_with_area()
        area = {"name": "Неизвестный район, Долинный, Россия", "lat": 51.0, "lon": 101.02}
        with patch.object(places, "geocode", side_effect=[self.confirmed_point(), area]) as lookup:
            problems = places.verify_itinerary(itinerary)
        self.assertEqual(lookup.call_count, 2)
        self.assertEqual(problems, [])
        self.assertTrue(itinerary.days[0].items[0].verified)


if __name__ == "__main__":
    unittest.main()
