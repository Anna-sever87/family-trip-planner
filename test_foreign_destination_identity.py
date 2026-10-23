"""Synthetic foreign-country identity regressions; never live map evidence.

The coordinates and IDs below are test fixtures, not a catalogue of visits.
Every network, shared-clock/DB and persistent-cache write is replaced/blocked.
These assertions require the correct alias behavior, not the current defect.
"""

from contextlib import nullcontext
from copy import deepcopy
import socket
import sqlite3
import unittest
from unittest.mock import Mock, patch

import places
from rules import check_city_radius
from schemas import Itinerary, ItineraryDay, ItineraryItem


TITLE = "Fixture Museum"
NHA_TRANG = (12.24, 109.19)
KEMER = (36.60, 30.56)


def record(city="Nha Trang", country="Việt Nam", code="vn", *, point=NHA_TRANG):
    return {
        "lat": point[0], "lon": point[1],
        "name": f"{TITLE}, {city}, {country}",
        "namedetails": {"name:en": TITLE},
        "address": {"city": city, "country": country, "country_code": code},
        "category": "tourism", "type": "museum", "osm_type": "node", "osm_id": 123,
    }


def nominatim_record(value):
    result = deepcopy(value)
    result["display_name"] = result.pop("name")
    result["lat"], result["lon"] = str(result["lat"]), str(result["lon"])
    return result


def itinerary(query, *, point=None):
    return Itinerary(title="Synthetic foreign fixture", summary="Offline only", days=[
        ItineraryDay(day=1, date="2026-10-28", theme="Museum", items=[
            ItineraryItem(time="10:00", title=TITLE, kind="attraction", description="Synthetic visit",
                          place_query=query, duration_min=60, verified=point is not None,
                          lat=point[0] if point else None, lon=point[1] if point else None),
        ]),
    ])


class ForeignDestinationIdentityTests(unittest.TestCase):
    def setUp(self):
        self.cache = {}
        boundaries = [
            patch.object(places, "_cache", self.cache),
            patch.object(places, "_save_cache"),  # In-memory results only.
            patch.object(places, "_wait_for_turn", side_effect=lambda: nullcontext()),
        ]
        for owner, name in (
            (places.httpx, "get"), (places.httpx, "post"),
            (socket.socket, "connect"), (socket, "getaddrinfo"), (sqlite3, "connect"),
        ):
            boundaries.append(patch.object(owner, name, side_effect=AssertionError(f"{name} forbidden")))
        for boundary in boundaries:
            boundary.start()
            self.addCleanup(boundary.stop)

    def fake_response(self, *records):
        response = Mock()
        response.json.return_value = [nominatim_record(value) for value in records]
        return response

    def test_vietnam_country_aliases_have_one_iso_identity(self):
        for label in ("vn", "Vietnam", "Viet Nam", "Việt Nam", "Вьетнам"):
            with self.subTest(label=label):
                self.assertEqual(places._country_identity(label), ("vn",))

    def test_turkey_country_aliases_have_one_iso_identity(self):
        for label in ("tr", "Turkey", "Türkiye", "Turkiye", "Турция"):
            with self.subTest(label=label):
                self.assertEqual(places._country_identity(label), ("tr",))

    def test_legitimate_vietnam_aliases_match_same_named_place(self):
        for country in ("Vietnam", "Viet Nam", "Việt Nam", "Вьетнам"):
            with self.subTest(country=country):
                self.assertTrue(places._place_matches_query(f"{TITLE}, Nha Trang, {country}", record()))

    def test_legitimate_turkey_aliases_match_same_named_place(self):
        for country in ("Turkey", "Türkiye", "Turkiye", "Турция"):
            with self.subTest(country=country):
                self.assertTrue(places._place_matches_query(
                    f"{TITLE}, Kemer, {country}", record("Kemer", "Türkiye", "tr", point=KEMER)))

    def test_country_label_fallback_without_iso_code_still_accepts_known_alias(self):
        self.assertTrue(places._place_matches_query(f"{TITLE}, Nha Trang, Vietnam", record(code="")))
        self.assertTrue(places._place_matches_query(
            f"{TITLE}, Kemer, Turkey", record("Kemer", "Türkiye", "", point=KEMER)))

    def test_wrong_iso_country_cannot_be_hidden_by_matching_display_label(self):
        # An authoritative contradictory country code must win over a label.
        for city, country, point in (("Nha Trang", "Vietnam", NHA_TRANG), ("Kemer", "Turkey", KEMER)):
            with self.subTest(city=city):
                self.assertFalse(places._place_matches_query(
                    f"{TITLE}, {city}, {country}", record(city, country, "ca", point=point)))

    def test_other_country_and_other_city_are_not_added_as_name_aliases(self):
        self.assertFalse(places._place_matches_query(
            f"{TITLE}, Nha Trang, Vietnam", record(country="Canada", code="ca")))
        self.assertFalse(places._place_matches_query(
            f"{TITLE}, Kemer, Türkiye", record("Antalya", "Türkiye", "tr", point=KEMER)))
        self.assertFalse(places._place_matches_query(
            f"{TITLE}, Nha Trang, Việt Nam", record(city="Da Nang")))

    def test_search_skips_wrong_country_homonym_then_accepts_correct_alias(self):
        query = f"{TITLE}, Kemer, Turkey"
        wrong = record("Kemer", "Turkey", "ca", point=(48.0, -123.0))
        correct = record("Kemer", "Türkiye", "tr", point=KEMER)
        with patch.object(places.httpx, "get", return_value=self.fake_response(wrong, correct)) as get:
            actual = places._search(query)
        self.assertEqual((actual["lat"], actual["lon"]), KEMER)
        self.assertEqual(actual["address"]["country_code"], "tr")
        self.assertEqual(get.call_count, 1)

    def test_geocode_stage_accepts_correct_vietnam_alias_without_retry(self):
        query = f"{TITLE}, Nha Trang, Vietnam"
        with patch.object(places.httpx, "get", return_value=self.fake_response(record())) as get:
            actual = places.geocode(query, max_variants=1)
        self.assertEqual((actual["lat"], actual["lon"]), NHA_TRANG)
        self.assertEqual(get.call_count, 1)
        self.assertNotIn(query.lower() + "|miss", self.cache)

    def test_itinerary_map_stage_accepts_correct_turkey_alias(self):
        query = f"{TITLE}, Kemer, Turkey"
        route = itinerary(query)
        with patch.object(places.httpx, "get", return_value=self.fake_response(
            record("Kemer", "Türkiye", "tr", point=KEMER))):
            problems = places.verify_itinerary(route)
        self.assertEqual(problems, [])
        item = route.days[0].items[0]
        self.assertTrue(item.verified)
        self.assertEqual((item.lat, item.lon), KEMER)
        self.assertEqual(item.provider, "osm")

    def test_nearby_search_rejects_remote_same_country_homonym(self):
        # Country aliases must not bypass a separately supplied geographic bound.
        query = f"{TITLE}, Türkiye"
        remote = record("Kemer", "Türkiye", "tr", point=(40.0, 30.56))
        with patch.object(places.httpx, "get", return_value=self.fake_response(remote)):
            with self.assertRaises(places.GeocodeMismatch):
                places._search(query, within_radius=(*KEMER, 40.0))

    def test_final_city_radius_still_rejects_remote_same_named_place(self):
        route = itinerary(f"{TITLE}, Kemer, Türkiye", point=(40.0, 30.56))
        self.assertEqual(len(check_city_radius(route, KEMER)), 1)
        self.assertEqual(check_city_radius(itinerary(f"{TITLE}, Kemer, Türkiye", point=KEMER), KEMER), [])

    def test_full_city_province_context_accepts_only_both_components(self):
        query = f"{TITLE}, Kemer, Antalya, Turkey"
        good = record("Kemer", "Türkiye", "tr", point=KEMER)
        good["address"]["state"] = "Antalya"
        self.assertTrue(places._place_matches_query(query, good))
        wrong_city = deepcopy(good)
        wrong_city["name"] = f"{TITLE}, Other Town, Antalya, Türkiye"
        wrong_city["address"]["city"] = "Other Town"
        wrong_city["lat"] = 36.75  # Still within the ordinary 40 km city radius.
        self.assertFalse(places._place_matches_query(query, wrong_city))
        wrong_province = deepcopy(good)
        wrong_province["address"]["state"] = "Burdur"
        self.assertFalse(places._place_matches_query(query, wrong_province))

    def test_search_does_not_choose_same_province_wrong_city_first(self):
        query = f"{TITLE}, Kemer, Antalya, Turkey"
        wrong = record("Other Town", "Türkiye", "tr", point=(36.75, 30.56))
        good = record("Kemer", "Türkiye", "tr", point=KEMER)
        for value in (wrong, good):
            value["address"]["state"] = "Antalya"
        with patch.object(places.httpx, "get", return_value=self.fake_response(wrong, good)) as get:
            actual = places._search(query)
        self.assertEqual(actual["address"]["city"], "Kemer")
        self.assertEqual(get.call_count, 1)

    def test_wrong_cached_city_cannot_bypass_full_context_reverification(self):
        query = f"{TITLE}, Kemer, Antalya, Turkey"
        wrong = record("Other Town", "Türkiye", "tr", point=(36.75, 30.56))
        good = record("Kemer", "Türkiye", "tr", point=KEMER)
        for value in (wrong, good):
            value["address"]["state"] = "Antalya"
        self.cache[query.lower()] = wrong
        with patch.object(places.httpx, "get", return_value=self.fake_response(good)) as get:
            actual = places.geocode(query, max_variants=1)
        self.assertEqual(actual["address"]["city"], "Kemer")
        self.assertEqual(get.call_count, 1)

    def test_itinerary_rejects_wrong_city_even_inside_city_radius(self):
        query = f"{TITLE}, Kemer, Antalya, Turkey"
        wrong = record("Other Town", "Türkiye", "tr", point=(36.75, 30.56))
        wrong["address"]["state"] = "Antalya"
        route = itinerary(query)
        with patch.object(places.httpx, "get", return_value=self.fake_response(wrong)):
            problems = places.verify_itinerary(route)
        self.assertTrue(problems)
        self.assertFalse(route.days[0].items[0].verified)
        self.assertIsNone(route.days[0].items[0].lat)

    def test_house_number_and_all_localities_are_checked_in_five_part_address(self):
        query = "Example Street, 12, Kemer, Antalya, Turkey"
        good = record("Kemer", "Türkiye", "tr", point=KEMER)
        good["name"] = "12, Example Street, Kemer, Antalya, Türkiye"
        good["address"].update(road="Example Street", house_number="12", state="Antalya")
        self.assertTrue(places._place_matches_query(query, good))
        for field, wrong_value in (("house_number", "15"), ("city", "Other Town"), ("state", "Burdur")):
            wrong = deepcopy(good)
            wrong["address"][field] = wrong_value
            # Do not leave the original matching address in display_name.
            wrong["name"] = f"{wrong['address']['house_number']}, Example Street, {wrong['address']['city']}, {wrong['address']['state']}, Türkiye"
            with self.subTest(field=field):
                self.assertFalse(places._place_matches_query(query, wrong))

    def test_existing_four_part_house_address_stays_supported(self):
        query = "Example Street, 12, Kemer, Turkey"
        good = record("Kemer", "Türkiye", "tr", point=KEMER)
        good["name"] = "12, Example Street, Kemer, Türkiye"
        good["address"].update(road="Example Street", house_number="12")
        self.assertTrue(places._place_matches_query(query, good))
        good["address"].pop("house_number")
        good["name"] = "Example Street, Kemer, Türkiye"
        self.assertFalse(places._place_matches_query(query, good))

    def test_explicit_neighbourhood_is_checked_without_requiring_city_type_field(self):
        query = f"{TITLE}, Example Quarter, Kemer, Turkey"
        good = record("Kemer", "Türkiye", "tr", point=KEMER)
        good["address"]["neighbourhood"] = "Example Quarter"
        self.assertTrue(places._place_matches_query(query, good))
        good["address"]["neighbourhood"] = "Other Quarter"
        self.assertFalse(places._place_matches_query(query, good))

    def test_query_variants_keep_both_city_and_province(self):
        variants = places._variants(f"{TITLE}, Kemer, Antalya, Turkey")
        self.assertTrue(variants)
        self.assertTrue(all("Kemer" in query and "Antalya" in query for query in variants))


if __name__ == "__main__":
    unittest.main()
