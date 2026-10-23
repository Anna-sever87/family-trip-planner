"""Offline bounded city resolution; fixture responses are not live evidence.

The Kemer regression reflects the provider's observed same-name ordering, but
all coordinates/responses here are supplied only by mocks. No city catalogue,
network call, real weather request, database or persistent cache is used.
"""

from copy import deepcopy
from datetime import date, timedelta
import socket
import sqlite3
import unittest
from unittest.mock import patch

import weather
from weather import CityResolutionError, find_city, get_weather


def city(name="Кемер", *, lat=36.59778, lon=30.56056, country="Турция", code="TR",
         admin="Анталья", identity=308220, population=17300):
    return dict(id=identity, name=name, latitude=lat, longitude=lon, country=country,
                country_code=code, admin1=admin, feature_code="PPL", population=population)


def burdur():
    return city(lat=37.35222, lon=30.06306, admin="Бурдур", identity=308217, population=2217)


class CityResolutionTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((weather.httpx, "get"), (socket.socket, "connect"),
                            (socket, "getaddrinfo"), (sqlite3, "connect")):
            boundary = patch.object(owner, name, side_effect=AssertionError(f"{name} forbidden"))
            boundary.start()
            self.addCleanup(boundary.stop)

    def resolve(self, destination, rows):
        with patch.object(weather, "_get_json", return_value={"results": rows}) as get:
            result = find_city(destination)
        return result, get.call_args.kwargs

    def assert_rejected(self, destination, rows, reason):
        with patch.object(weather, "_get_json", return_value={"results": rows}):
            with self.assertRaises(CityResolutionError) as caught:
                find_city(destination)
        self.assertEqual(caught.exception.reason, reason)
        self.assertIn("Уточните город", str(caught.exception))

    def test_unqualified_kemer_does_not_choose_first_burdur_homonym(self):
        self.assert_rejected("Кемер, Турция", [burdur(), city()], "ambiguous")

    def test_bare_city_still_rejects_homonyms(self):
        self.assert_rejected("Кемер", [burdur(), city()], "ambiguous")

    def test_explicit_admin_selects_coastal_kemer_without_population_rule(self):
        # Even if the provider ignores filtering, inspect country and admin1.
        wrong = burdur(); wrong["population"] = 1_000_000
        actual, params = self.resolve("Кемер, Анталья, Турция", [wrong, city()])
        self.assertEqual((actual["lat"], actual["lon"]), (36.59778, 30.56056))
        self.assertEqual(actual["geoname_id"], 308220)
        self.assertEqual(params, {"name": "Кемер, Анталья", "countryCode": "TR",
                                  "count": weather.MAX_CITY_RESULTS, "language": "ru"})

    def test_same_admin_exact_homonyms_remain_ambiguous(self):
        first = city("Kemer", admin="Antalya")
        second = city("Kemer", lat=38.10618, lon=31.67249, admin="Antalya", identity=308213, population=None)
        self.assert_rejected("Kemer, Antalya, Turkey", [first, second], "ambiguous")

    def test_country_alias_applies_iso_filter_before_result_limit(self):
        aliases = (("Турция", "TR", "Кемер", city()),
                   ("Türkiye", "TR", "Кемер", city()),
                   ("Vietnam", "VN", "Nha Trang", city("Nha Trang", lat=12.24507, lon=109.19432,
                                                         code="VN", country="Việt Nam", admin="Khánh Hòa")),
                   ("Россия", "RU", "Магадан", city("Магадан", lat=59.56, lon=150.80,
                                                     code="RU", country="Россия", admin="Магаданская область")))
        for country, code, name, row in aliases:
            with self.subTest(country=country):
                actual, params = self.resolve(f"{name}, {country}", [row])
                self.assertEqual(actual["country_code"], code)
                self.assertEqual(params["countryCode"], code)
                self.assertEqual(params["name"], name)
                self.assertEqual(params["count"], weather.MAX_CITY_RESULTS)

    def test_nha_trang_cyrillic_unique_response_preserves_vietnam(self):
        row = city("Нячанг", lat=12.24507, lon=109.19432, country="Вьетнам", code="VN", admin="Khanh Hoa", identity=1572151)
        actual, params = self.resolve("Нячанг, Вьетнам", [row])
        self.assertEqual(actual["geoname_id"], 1572151)
        self.assertEqual(params["countryCode"], "VN")
        self.assertEqual(params["language"], "ru")

    def test_unique_bare_city_backwards_compatible_minimum_response(self):
        actual, params = self.resolve("Магадан", [{"name": "Магадан", "latitude": 59.56, "longitude": 150.80}])
        self.assertEqual({key: actual[key] for key in ("name", "lat", "lon")},
                         {"name": "Магадан", "lat": 59.56, "lon": 150.80})
        self.assertNotIn("countryCode", params)

    def test_wrong_country_code_is_rejected_despite_matching_labels(self):
        row = city(code="CA")
        self.assert_rejected("Кемер, Турция", [row], "context_mismatch")

    def test_missing_country_code_requires_matching_country_name(self):
        row = city(country="Türkiye", code="")
        actual, _ = self.resolve("Кемер, Турция", [row])
        self.assertEqual(actual["name"], "Кемер")
        self.assert_rejected("Кемер, Турция", [city(country="Canada", code="")], "context_mismatch")

    def test_explicit_admin_is_not_silently_discarded(self):
        self.assert_rejected("Кемер, Анталья, Турция", [burdur()], "context_mismatch")

    def test_unknown_country_retains_provider_qualifier_and_local_check(self):
        row = city("Example City", country="Example Country", code="XY", admin="Example Province")
        actual, params = self.resolve("Example City, Example Country", [row])
        self.assertEqual(actual["name"], "Example City")
        self.assertEqual(params["name"], "Example City, Example Country")
        self.assertNotIn("countryCode", params)
        self.assert_rejected("Example City, Other Country", [row], "context_mismatch")

    def test_prefix_matches_do_not_create_a_different_city(self):
        rows = [city("Kemerköy", country="Türkiye", admin="Antalya", identity=1),
                city("Kemer", country="Türkiye", admin="Antalya")]
        actual, _ = self.resolve("Kemer, Antalya, Turkey", rows)
        self.assertEqual(actual["name"], "Kemer")
        self.assert_rejected("Kemer, Turkey", rows[:1], "context_mismatch")

    def test_same_provider_result_duplicate_is_not_ambiguous(self):
        row = city()
        actual, _ = self.resolve("Кемер, Турция", [row, deepcopy(row)])
        self.assertEqual(actual["geoname_id"], row["id"])

    def test_conflicting_coordinates_for_one_id_remain_ambiguous(self):
        self.assert_rejected("Кемер, Турция", [city(), city(lat=38.0)], "ambiguous")

    def test_bounded_response_does_not_claim_unique_when_cut_off(self):
        rows = [city()] + [city(f"Кемер{i}", identity=i) for i in range(weather.MAX_CITY_RESULTS - 1)]
        self.assert_rejected("Кемер, Турция", rows, "search_limit")

    def test_invalid_coordinates_and_non_settlement_results_are_rejected(self):
        for invalid in (city(lat=True), city(lat=float("nan")), city(lon=181), {**city(), "feature_code": "MT"}):
            with self.subTest(invalid=invalid):
                self.assert_rejected("Кемер, Турция", [invalid], "context_mismatch")

    def test_empty_response_remains_none_without_retry(self):
        actual, _ = self.resolve("Нячанг, Вьетнам", [])
        self.assertIsNone(actual)

    def test_invalid_context_rejected_before_provider_call(self):
        for value in ("", "Кемер,,Турция", "Кемер, Анталья, Район, Турция", "Кемер\nТурция"):
            with self.subTest(value=value), patch.object(weather, "_get_json") as get:
                with self.assertRaises(CityResolutionError) as caught:
                    find_city(value)
                self.assertEqual(caught.exception.reason, "invalid_context")
                get.assert_not_called()

    def test_weather_passes_full_destination_to_resolver(self):
        tomorrow = date.today() + timedelta(days=1)
        raw_weather = {"daily": {"time": [tomorrow.isoformat()], "temperature_2m_min": [17],
                                  "temperature_2m_max": [25], "precipitation_sum": [0]}}
        resolved = {"name": "Кемер", "lat": 36.59778, "lon": 30.56056}
        with patch.object(weather, "find_city", return_value=resolved) as find, \
                patch.object(weather, "_get_json", return_value=raw_weather) as get:
            result = get_weather("Кемер, Анталья, Турция", tomorrow.isoformat(), tomorrow.isoformat())
        find.assert_called_once_with("Кемер, Анталья, Турция")
        self.assertEqual(get.call_args.kwargs["latitude"], resolved["lat"])
        self.assertEqual(get.call_args.kwargs["longitude"], resolved["lon"])
        self.assertEqual(result["city"], "Кемер")

    def test_weather_does_not_swallow_geographic_ambiguity(self):
        error = CityResolutionError("Кемер, Турция", reason="ambiguous")
        with patch.object(weather, "find_city", side_effect=error), patch.object(weather, "_get_json") as get:
            with self.assertRaises(CityResolutionError):
                get_weather("Кемер, Турция", "2026-10-28", "2026-10-31")
        get.assert_not_called()

    def test_weather_keeps_network_failure_as_optional_missing_weather(self):
        with patch.object(weather, "find_city", side_effect=weather.httpx.ConnectError("offline")):
            self.assertIsNone(get_weather("Нячанг, Вьетнам", "2026-10-28", "2026-11-01"))


if __name__ == "__main__":
    unittest.main()
