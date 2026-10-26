"""Offline country-label regressions, not a catalogue of real places.

Synthetic IDs/POIs below exercise the unchanged source, country, name and
radius guards. No HTTP, AI, database or persistent-cache access is permitted.
"""

from contextlib import ExitStack
from copy import deepcopy
from datetime import date, timedelta
import unittest
from unittest.mock import patch

import city_candidates
import city_context
import city_deterministic
import places
import weather
from city_visit_options import check_city_visit_conditions, visit_option
from schemas import TripRequest


def request(destination="Сингапур, Сингапур", days=1):
    return TripRequest(destination=destination, start_date="2099-06-01",
                       end_date=(date(2099, 6, 1) + timedelta(days=days - 1)).isoformat(),
                       adults=2, children_ages=[], budget=240000, travel_cost=110000,
                       lodging_cost=45000, currency="RUB", pace="средний",
                       start_time="09:00", end_time="18:00", interest_categories=["museums"])


def fixture(*, city="Сингапур", locality="Singapore", country="Singapore", code="SG", count=2):
    center = dict(name=city, country=country, country_code=code, admin1="", lat=1.29, lon=103.85)
    values = []
    for index in range(count):
        title = f"Synthetic Museum {index + 1}"
        raw = dict(name=f"{title}, {locality}, {country}", namedetails={"name": title},
                   address={"city": locality, "country": country, "country_code": code.lower()},
                   lat=1.29, lon=103.85 + .001 * index, osm_type="node", osm_id=1001 + index,
                   category="tourism", type="museum")
        canonical = city_candidates._canonical_record(title, raw, dict(center, country_code=code.lower()))
        if canonical is None:
            raise AssertionError("Synthetic provider record did not pass ordinary map guards")
        values.append(city_candidates.CityCandidate(f"osm-node-{1001 + index}", title,
                      canonical["place_query"], canonical["locality"], canonical))
    return center, values


class CountryAliasPipelineTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("httpx.get", "httpx.post", "socket.socket.connect", "socket.getaddrinfo",
                     "sqlite3.connect", "places.geocode", "places._load_cache", "places._save_cache",
                     "planner.generate_json"):
            self.stack.enter_context(patch(name, side_effect=AssertionError("External I/O forbidden")))

    def test_reviewed_country_names_share_only_their_own_iso_identity(self):
        cases = {
            "sg": ("Сингапур", "Singapore", "Republic of Singapore", "Singapura", "新加坡"),
            "cn": ("Китай", "China", "中国", "中华人民共和国", "Китайская Народная Республика"),
            "va": ("Ватикан", "Vatican City", "Vatican City State", "Holy See", "Città del Vaticano"),
        }
        for code, labels in cases.items():
            for label in (*labels, code, code.upper()):
                with self.subTest(country=label):
                    self.assertEqual(places._country_identity(label), (code,))
        for label in ("Taiwan", "Hong Kong", "Macao", "Malaysia", "Italy", "Singapore District"):
            self.assertNotIn(places._country_identity(label), {(code,) for code in cases})

    def test_singapore_same_city_and_country_are_not_duplicate_input(self):
        from web_service import destination_with_country
        self.assertEqual(destination_with_country("Сингапур", "Сингапур"), "Сингапур, Сингапур")
        self.assertEqual(destination_with_country("Сингапур, Сингапур", "Singapore"), "Сингапур, Сингапур")

    def test_foreign_provider_country_labels_keep_verified_context_and_options(self):
        for city, native_city, target_country, provider_country, code in (
            ("Сингапур", "Singapore", "Сингапур", "Singapore", "SG"),
            ("Пекин", "北京市", "Китай", "中国", "CN"),
            ("Ватикан", "Città del Vaticano", "Ватикан", "Vatican City", "VA"),
        ):
            with self.subTest(code=code):
                req = request(f"{city}, {target_country}")
                center, pool = fixture(city=city, locality=native_city, country=provider_country, code=code)
                before = deepcopy((center, pool))
                snapshot = city_context.build_city_context(req, pool, center)
                loaded, saved_center = city_context.load_city_context(snapshot, req)
                self.assertEqual(saved_center["country_code"], code.lower())
                self.assertEqual([value.candidate_id for value in loaded], [value.candidate_id for value in pool])
                for item in pool:
                    option = visit_option(item, req)
                    self.assertIsNotNone(option)
                    self.assertEqual(option.duration_min, 90)
                    self.assertIsNone(option.estimated_cost)
                    self.assertIsNone(option.indoor)
                    self.assertEqual(option.basis, "planning_estimate")
                self.assertEqual((center, pool), before)

    def test_four_day_singapore_schedule_uses_real_guards_not_a_mocked_acceptance(self):
        req = request(days=4)
        center, pool = fixture(count=8)
        result = city_deterministic.build_deterministic_city_itinerary(req, pool, resolved_city=center)
        self.assertEqual((result.status, result.problems), ("success", []), result.coverage)
        self.assertEqual(len(result.itinerary.days), 4)
        self.assertTrue(all(day.day_mode == "program" for day in result.itinerary.days))
        self.assertEqual(check_city_visit_conditions(result.itinerary, pool, req), [])
        self.assertEqual(places.verify_itinerary(result.itinerary, verified_records={
            value.place_query: value.record for value in pool}), [])

    def test_pre_fix_shape_failed_at_context_and_removed_visit_options(self):
        # Reproduce the exact code boundary, not an assertion about the old
        # live job whose input snapshot was not retained.
        req = request()
        center, pool = fixture()
        without_new_aliases = {key: value for key, value in places._COUNTRY_NAME_CODES.items()
                               if key not in {"sg", "cn", "va"}}
        with patch.object(places, "_COUNTRY_NAME_CODES", without_new_aliases):
            self.assertIsNone(visit_option(pool[0], req))
            with self.assertLogs("city_deterministic", level="WARNING"):
                result = city_deterministic.build_deterministic_city_itinerary(req, pool, resolved_city=center)
            self.assertEqual(result.status, "invalid_request")
            self.assertEqual(result.coverage["invalid_input_stage"], "city_context")

    def test_wrong_country_cannot_hide_behind_matching_city_or_label(self):
        req = request()
        center, pool = fixture()
        for changes in ({"country_code": "MY"}, {"country_code": "CN"}, {"name": "Other City"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                city_context.build_city_context(req, pool, {**center, **changes})
        for code in ("my", "cn", "va"):
            forged = deepcopy(pool[0])
            forged.record["address"]["country_code"] = code
            self.assertIsNone(visit_option(forged, req))
            with self.assertRaises(ValueError):
                city_context.build_city_context(req, [forged], center)
        foreign = request("Сингапур, Россия")
        self.assertIsNone(visit_option(pool[0], foreign))
        with self.assertRaises(ValueError):
            city_context.build_city_context(foreign, pool, center)

    def test_alias_does_not_bypass_radius_name_or_provider_identity(self):
        req = request()
        center, pool = fixture()
        for changes in ({"lat": 10.0}, {"osm_id": None}, {"name": "Another Museum, Singapore, Singapore"}):
            forged = deepcopy(pool[0])
            forged.record.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                city_context.build_city_context(req, [forged], center)

    def test_geocoder_uses_iso_country_filter_and_rejects_wrong_country_homonym(self):
        found = dict(id=100, name="Сингапур", latitude=1.29, longitude=103.85,
                     country_code="SG", country="Singapore", feature_code="PPLC")
        with patch.object(weather, "_get_json", return_value={"results": [found]}) as get:
            center = weather.find_city("Сингапур, Сингапур")
        self.assertEqual(center["country_code"], "SG")
        self.assertEqual(get.call_args.kwargs["name"], "Сингапур")
        self.assertEqual(get.call_args.kwargs["countryCode"], "SG")
        for code in ("MY", "CN"):
            with patch.object(weather, "_get_json", return_value={"results": [{**found, "country_code": code}]}):
                with self.assertRaises(weather.CityResolutionError):
                    weather.find_city("Сингапур, Сингапур")


if __name__ == "__main__":
    unittest.main()
