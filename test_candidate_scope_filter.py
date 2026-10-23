"""A nearby namesake outside the region must not hide a valid local result."""

import unittest
from unittest.mock import patch

import httpx
import candidate_grounding as grounding
from destination_scope import RegionScope
import places


TITLE = "Нерпинарий Тестовый"
COUNTRY = "Россия"
BASE = {"lat": 50.5, "lon": 100.95, "map_address": {"town": "База", "county": "Тестовый район", "country_code": "ru"}}
SCOPE = RegionScope("Тестовый край", COUNTRY, (50, 51, 100, 101), {
    "type": "Polygon", "coordinates": [[[100, 50], [101, 50], [101, 51], [100, 51], [100, 50]]],
})


def record(locality, lon):
    return {"name": f"{TITLE}, {locality}, {COUNTRY}", "lat": 50.5, "lon": lon,
            "type": "aquarium", "category": "tourism",
            "address": {"town": locality, "county": "Тестовый район", "country": COUNTRY, "country_code": "ru"}}


def api_record(value):
    return {**value, "display_name": value["name"]}


class CandidateScopeFilterTests(unittest.TestCase):
    def setUp(self):
        self.outside = record("Внешний", 101.7)
        self.inside = record("Внутренний", 100.9)
        self.cache = {}
        for owner, name, kwargs in (
            (places, "_cache", {"new": self.cache}),
            (places, "_save_cache", {"return_value": None}),
            (places, "_wait_for_turn", {}),
            (places.time, "sleep", {"return_value": None}),
        ):
            patcher = patch.object(owner, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    def lookup(self, responses):
        budget = grounding.GroundingResult(query_limit=1)
        response = httpx.Response(200, json=[api_record(value) for value in responses],
                                  request=httpx.Request("GET", "https://example.invalid"))
        with patch.object(places.httpx, "get", return_value=response) as http:
            value, reason = grounding._lookup_nearby_poi(TITLE, COUNTRY, BASE, budget, scope=SCOPE)
        return value, reason, budget, http

    def test_outside_region_cache_does_not_stop_the_only_bounded_search(self):
        self.cache[self.outside["name"].lower()] = self.outside
        value, reason, budget, http = self.lookup([self.inside])
        self.assertEqual(reason, "")
        self.assertEqual(value["place_query"], self.inside["name"])
        self.assertEqual(http.call_count, 1)
        self.assertEqual(budget.reserved_http_attempts_upper_bound, 2)
        self.assertEqual(self.cache[self.outside["name"].lower()], self.outside)
        self.assertIn(self.inside["name"].lower(), self.cache)

    def test_first_api_namesake_outside_region_is_skipped_for_later_valid_result(self):
        value, reason, budget, http = self.lookup([self.outside, self.inside])
        self.assertEqual(reason, "")
        self.assertEqual(value["place_query"], self.inside["name"])
        self.assertEqual((http.call_count, budget.geocode_calls, budget.verification_queries), (1, 1, 1))
        self.assertEqual(http.call_args.kwargs["params"]["bounded"], 1)
        self.assertEqual(http.call_args.kwargs["params"]["q"], f"{TITLE}, {COUNTRY}")

    def test_outside_cache_is_skipped_for_later_valid_cache_without_network(self):
        self.cache[self.outside["name"].lower()] = self.outside
        self.cache[self.inside["name"].lower()] = self.inside
        value, reason, budget, http = self.lookup([])
        self.assertEqual(reason, "")
        self.assertEqual(value["place_query"], self.inside["name"])
        http.assert_not_called()
        self.assertEqual(budget.reserved_http_attempts_upper_bound, 0)

    def test_same_name_hamlet_is_skipped_for_museum_in_one_api_response(self):
        hamlet = {**self.inside, "type": "hamlet", "category": "place"}
        museum = {**self.inside, "type": "museum", "category": "tourism"}
        value, reason, budget, http = self.lookup([hamlet, museum])
        self.assertEqual(reason, "")
        self.assertEqual((value["type"], value["category"]), ("museum", "tourism"))
        self.assertEqual((http.call_count, budget.geocode_calls, budget.verification_queries), (1, 1, 1))
        self.assertEqual(budget.reserved_http_attempts_upper_bound, 2)

    def test_cached_hamlet_does_not_block_named_museum_search(self):
        hamlet = {**self.inside, "type": "hamlet", "category": "place"}
        museum = {**self.inside, "type": "museum", "category": "tourism"}
        self.cache[hamlet["name"].lower()] = hamlet
        value, reason, budget, http = self.lookup([museum])
        self.assertEqual(reason, "")
        self.assertEqual(value["type"], "museum")
        self.assertEqual(http.call_count, 1)
        self.assertEqual(budget.reserved_http_attempts_upper_bound, 2)

    def test_namesake_filter_does_not_admit_unapproved_types(self):
        for kind, category in (("ngo", "office"), ("university", "amenity"),
                               ("locality", "place"), ("building", "building"),
                               ("commercial", "landuse")):
            with self.subTest(kind=kind):
                value, reason, _, http = self.lookup([{**self.inside, "type": kind, "category": category}])
                self.assertIsNone(value)
                self.assertEqual(reason, "identity_mismatch")
                self.assertEqual(http.call_count, 1)
                self.assertEqual(self.cache, {})

    def test_scope_filter_never_admits_wrong_identity_country_or_radius(self):
        wrong_name = {**self.inside, "name": f"Другой объект, Внутренний, {COUNTRY}"}
        wrong_country = {**self.inside, "address": {**self.inside["address"], "country_code": "fr"}}
        far_away = {**self.inside, "lon": 99.0}
        for value in (wrong_name, wrong_country, far_away, self.outside):
            with self.subTest(value=value):
                accepted, reason, _, http = self.lookup([value])
                self.assertIsNone(accepted)
                self.assertEqual(reason, "identity_mismatch")
                self.assertEqual(http.call_count, 1)
                self.assertEqual(self.cache, {})


if __name__ == "__main__":
    unittest.main()
