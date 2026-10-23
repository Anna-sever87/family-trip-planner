"""Exact map-address locality is already known; unrelated areas still fail."""

import socket
import unittest
from unittest.mock import patch

import places
from schemas import Itinerary, ItineraryDay, ItineraryItem


TITLE = "Музей северных ремёсел"
LOCALITY = "Реальный район"
QUERY = f"{TITLE}, {LOCALITY}, Россия"


def record(**updates):
    value = {
        "name": f"{TITLE}, Россия", "lat": 52.0, "lon": 105.0,
        "type": "museum", "category": "tourism",
        "address": {"county": LOCALITY, "country": "Россия", "country_code": "ru"},
    }
    value.update(updates)
    return value


def itinerary(area=LOCALITY):
    return Itinerary(title="Поездка", summary="Тест", days=[ItineraryDay(
        day=1, date="2099-06-01", theme="Музей", items=[ItineraryItem(
            time="10:00", title=TITLE, area=area, place_query=QUERY,
            description="Тестовое посещение", duration_min=60,
        )],
    )])


class VerifiedAreaContextTests(unittest.TestCase):
    def setUp(self):
        for owner, name, options in (
            (socket.socket, "connect", {"side_effect": AssertionError("No network")}),
            (places, "_cache", {"new": {}}),
            (places, "_save_cache", {"side_effect": AssertionError("No cache writes")}),
            (places, "_search", {"side_effect": AssertionError("No external lookup")}),
        ):
            guard = patch.object(owner, name, **options)
            guard.start()
            self.addCleanup(guard.stop)

    def test_verified_address_locality_missing_from_display_name_needs_no_area_lookup(self):
        places._cache[QUERY.lower()] = record()
        route = itinerary()
        with patch.object(places, "geocode", wraps=places.geocode) as geocode:
            self.assertEqual(places.verify_itinerary(route), [])
        geocode.assert_called_once_with(QUERY)
        self.assertTrue(route.days[0].items[0].verified)
        self.assertEqual(route.days[0].items[0].area, LOCALITY)

    def test_whitelisted_administrative_and_settlement_fields_match_exactly(self):
        for field in ("city", "town", "village", "hamlet", "suburb", "neighbourhood", "quarter", "locality",
                      "municipality", "county", "state", "region"):
            with self.subTest(field=field):
                place = record(address={field: LOCALITY})
                self.assertTrue(places._area_matches_map_context(LOCALITY, place))
                self.assertFalse(places._area_matches_map_context("Реальный", place))
                self.assertFalse(places._area_matches_map_context("Другой район", place))

    def test_normalization_is_exact_without_changing_model_area(self):
        route = itinerary("  РЕАЛЬНЫЙ   РАЙОН  ")
        with patch.object(places, "geocode", return_value=record()) as geocode:
            self.assertEqual(places.verify_itinerary(route), [])
        self.assertEqual(geocode.call_count, 1)
        self.assertEqual(route.days[0].items[0].area, "  РЕАЛЬНЫЙ   РАЙОН  ")

    def test_wrong_area_or_substring_still_requires_lookup_and_missing_area_fails(self):
        for area in ("Другой район", "Реальный", TITLE, "Россия"):
            with self.subTest(area=area):
                route = itinerary(area)
                with patch.object(places, "geocode", side_effect=[record(), None]) as geocode:
                    problems = places.verify_itinerary(route)
                self.assertEqual(geocode.call_count, 2)
                self.assertEqual(len(problems), 1, problems)
                self.assertIn("не найден на карте", problems[0])
                self.assertTrue(route.days[0].items[0].verified)

    def test_legacy_display_address_uses_whole_components_not_poi_or_country(self):
        place = record(name=f"{TITLE}, {LOCALITY}, Россия", address={})
        self.assertTrue(places._area_matches_map_context(LOCALITY, place))
        for area in ("Реальный", TITLE, "Россия"):
            with self.subTest(area=area):
                self.assertFalse(places._area_matches_map_context(area, place))

    def test_unrelated_address_fields_and_malformed_values_are_not_area_evidence(self):
        for field in ("country", "country_code", "road", "postcode", "house_number", "tourism"):
            with self.subTest(field=field):
                self.assertFalse(places._area_matches_map_context(LOCALITY, record(address={field: LOCALITY})))
        for address in (None, [], LOCALITY, {"county": [LOCALITY]}, {"county": True}):
            with self.subTest(address=address):
                self.assertFalse(places._area_matches_map_context(LOCALITY, record(address=address)))

    def test_matching_locality_cannot_bypass_wrong_country_or_poi_identity(self):
        for wrong in (
            record(name=f"{TITLE}, США", address={"county": LOCALITY, "country_code": "us"}),
            record(name="Другой музей, Россия"),
        ):
            with self.subTest(wrong=wrong):
                places._cache[QUERY.lower()] = wrong
                route = itinerary()
                with patch.object(places, "_search", side_effect=places.GeocodeUnavailable("offline")), \
                        patch.object(places, "geocode", wraps=places.geocode) as geocode:
                    problems = places.verify_itinerary(route)
                geocode.assert_called_once_with(QUERY)
                self.assertEqual(len(problems), 1)
                self.assertFalse(route.days[0].items[0].verified)
                self.assertIsNone(route.days[0].items[0].lat)
                self.assertFalse(places._place_matches_query(QUERY, wrong))


if __name__ == "__main__":
    unittest.main()
