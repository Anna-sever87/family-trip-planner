"""Positive map cache reuse is bounded, independent of AI proposal omissions."""

from copy import deepcopy
import json
import socket
import unittest
from unittest.mock import Mock, patch

import httpx
import candidate_grounding as grounding
import places
from destination_scope import RegionScope
from schemas import TripRequest
from yandex_ai import GenerationResult


COUNTY = "Иркутский муниципальный округ"
BASE_QUERY = "Листвянка, Россия"
SCOPE = RegionScope("Байкал", "Россия", (50, 56, 103, 110), {
    "type": "Polygon", "coordinates": [[[103, 50], [110, 50], [110, 56], [103, 56], [103, 50]]],
})


def map_record(title, *, locality=COUNTY, lat=51.88, lon=104.86, kind="museum", **updates):
    value = {
        "name": f"{title}, {locality}, Россия", "lat": lat, "lon": lon,
        "type": kind, "category": "tourism",
        "address": {"county": COUNTY, "country": "Россия", "country_code": "ru"},
    }
    if locality != COUNTY:
        value["address"]["village"] = locality
    value.update(updates)
    return value


def hint(title):
    return {"title": title, "place_query": f"{title}, Листвянка, Россия"}


class GroundingPoolReuseTests(unittest.TestCase):
    def setUp(self):
        self.cache = {BASE_QUERY.lower(): {
            "name": BASE_QUERY, "lat": 51.86, "lon": 104.85, "type": "village", "category": "place",
            "address": {"village": "Листвянка", "county": COUNTY, "country_code": "ru", "postcode": "not needed"},
        }}
        for owner, name, options in (
            (places, "_cache", {"new": self.cache}),
            (places, "_save_cache", {"return_value": None}),
            (places.httpx, "get", {"side_effect": AssertionError("No HTTP")}),
            (places, "geocode", {"side_effect": AssertionError("No base network")}),
            (places, "geocode_nearby_named", {"side_effect": AssertionError("No POI network")}),
            (socket.socket, "connect", {"side_effect": AssertionError("No network")}),
        ):
            patcher = patch.object(owner, name, **options)
            patcher.start()
            self.addCleanup(patcher.stop)

    def put(self, title, **updates):
        record = map_record(title, **updates)
        self.cache[record["name"].lower()] = record
        return record

    def run_grounding(self, hints=(), sources=(), *, days=1, payload=None):
        request = TripRequest(destination="Байкал, Россия", destination_kind="region",
                              start_date="2099-06-01", end_date=f"2099-06-{days:02d}", budget=100000)
        payload = payload or {
            "bases": [{"base_key": "base_1", "base_query": BASE_QUERY}],
            "pois": [{"title": f"Неизвестный музей {index}", "place_query": f"Неизвестный музей {index}, Листвянка, Россия",
                      "locality": "Листвянка", "base_key": "base_1"} for index in range(min(18, days * 2))],
        }
        completion = Mock(return_value=GenerationResult(json.dumps(payload, ensure_ascii=False), 11, 13))
        with patch.object(grounding, "destination_name_hints", return_value=list(sources)):
            result = grounding.ground_region_candidates(request, SCOPE, max_http_attempts=0,
                                                        known_hints=list(hints), completion_fn=completion)
        completion.assert_called_once()
        self.assertEqual(result.completion_calls, 1)
        self.assertEqual((result.geocode_calls, result.reserved_http_attempts_upper_bound), (0, 0))
        return result

    def test_omitted_chersky_is_reused_from_positive_map_record_not_hint_coordinates(self):
        record = self.put("Камень Черского", kind="peak", osm_type="node", osm_id=123)
        proposed = {**hint("Камень Черского"), "lat": 0, "lon": 0, "osm_type": "way", "osm_id": 999,
                    "allowed_base_keys": ["invented"], "verified": True}
        before = deepcopy(proposed)
        result = self.run_grounding([proposed])
        self.assertEqual((result.cache_only_checks, result.cache_only_added), (1, 1))
        accepted = result.confirmed_hints[0]
        self.assertEqual((accepted["lat"], accepted["lon"], accepted["osm_type"], accepted["osm_id"]),
                         (record["lat"], record["lon"], "node", 123))
        self.assertEqual(accepted["allowed_base_keys"], ["base_1"])
        self.assertEqual(accepted["place_query"], f"Камень Черского, {COUNTY}, Россия")
        self.assertEqual(proposed, before)
        self.assertNotIn("postcode", result.base_facts[0]["map_address"])
        self.assertNotIn("verified", accepted)

    def test_source_hint_with_positive_cache_is_also_eligible(self):
        self.put("Байкальский музей", locality="Листвянка")
        result = self.run_grounding(sources=[{**hint("Байкальский музей"), "source_url": "https://example.org/museum"}])
        self.assertEqual([fact["title"] for fact in result.confirmed_hints], ["Байкальский музей"])
        self.assertEqual(result.cache_only_added, 1)

    def test_cache_hint_lowercase_uses_map_spelling_of_exact_same_name(self):
        self.put("Камень Черского", kind="peak")
        result = self.run_grounding([hint("камень черского")])
        self.assertEqual(result.confirmed_hints[0]["title"], "Камень Черского")
        self.assertEqual(result.confirmed_hints[0]["place_query"], f"Камень Черского, {COUNTY}, Россия")

    def test_map_display_name_does_not_replace_a_different_verified_alias(self):
        self.put("Музей деревянного зодчества", namedetails={"alt_name": "Музей старинных домов"})
        result = self.run_grounding([hint("Музей старинных домов")])
        self.assertEqual(result.confirmed_hints[0]["title"], "Музей старинных домов")

    def test_cache_seed_prefers_nearest_compatible_base_without_removing_other_pairs(self):
        second = "Большое Голоустное, Россия"
        self.cache[second.lower()] = {
            "name": second, "lat": 52.03, "lon": 105.41, "type": "village", "category": "place",
            "address": {"village": "Большое Голоустное", "county": COUNTY, "country_code": "ru"},
        }
        self.put("Никольская церковь", locality="Большое Голоустное", lat=52.04, lon=105.42, kind="place_of_worship")
        payload = {"bases": [{"base_key": "base_1", "base_query": BASE_QUERY}, {"base_key": "base_2", "base_query": second}],
                   "pois": [{**hint(f"Музей {number}"), "locality": "Листвянка", "base_key": "base_1"} for number in (1, 2)]}
        result = self.run_grounding([hint("Никольская церковь")], payload=payload)
        fact = result.confirmed_hints[0]
        self.assertEqual(fact["base_key"], "base_2")
        self.assertEqual(fact["allowed_base_keys"], ["base_1", "base_2"])

    def test_hint_source_url_or_forged_verified_flag_is_not_map_evidence(self):
        result = self.run_grounding([hint("Музей без карты")], [{**hint("Музей без карты"), "verified": True}])
        self.assertEqual(result.confirmed_hints, [])
        self.assertEqual((result.cache_only_checks, result.cache_only_added), (1, 0))

    def test_seed_checks_do_not_spend_query_or_http_budget(self):
        baseline = self.run_grounding()
        self.put("Камень Черского", kind="peak")
        result = self.run_grounding([hint("Камень Черского"), hint("Нет в кеше")])
        self.assertEqual(result.verification_queries, baseline.verification_queries)
        self.assertEqual((result.query_limit, result.http_attempt_limit), (baseline.query_limit, 0))
        self.assertEqual(result.cache_only_checks, 2)

    def test_cache_only_cap_is_eighteen_and_stops_before_model_poi_lookups(self):
        names = [f"Именованный музей {index}" for index in range(24)]
        for index, title in enumerate(names):
            self.put(title, lon=104.86 + index * 0.002)
        result = self.run_grounding([hint(title) for title in names], days=7)
        self.assertEqual((len(result.confirmed_hints), result.cache_only_added, result.cache_only_checks), (18, 18, 18))
        self.assertEqual(result.verification_queries, 1)  # Only the base.

    def test_cache_records_retain_all_identity_country_type_scope_and_context_checks(self):
        variants = [
            {"name": "Совсем другой музей, Листвянка, Россия"},
            {"address": {"county": COUNTY, "country_code": "us"}},
            {"type": "village", "category": "place"},
            {"lon": 120},
            {"address": {"county": "Слюдянский район", "country_code": "ru"}},
            {"address": {"country_code": "ru"}},
            {"lat": True},
        ]
        for update in variants:
            with self.subTest(update=update):
                title = "Особый музей"
                key = f"{title}, {COUNTY}, Россия".lower()
                self.cache[key] = map_record(title, **update)
                result = self.run_grounding([hint(title)])
                self.assertEqual(result.confirmed_hints, [])
                self.cache.pop(key)

    def test_cache_reuse_checks_full_geometry_not_just_its_bounding_box(self):
        title = "Музей у границы"
        self.cache[BASE_QUERY.lower()]["lon"] = 104.75
        self.put(title, lat=52.3, lon=105.5)
        # A real hole in the region, with the POI more than 40 km from its edge.
        scope_with_hole = RegionScope(SCOPE.name, SCOPE.country, SCOPE.bounds, {
            "type": "Polygon", "coordinates": [SCOPE.geometry["coordinates"][0],
                [[104.8, 51.6], [106.3, 51.6], [106.3, 53.0], [104.8, 53.0], [104.8, 51.6]]],
        })
        with patch(__name__ + ".SCOPE", scope_with_hole):
            result = self.run_grounding([hint(title)])
        self.assertEqual(result.confirmed_hints, [])

    def test_same_osm_object_with_distant_alias_coordinates_is_one_place(self):
        self.put("Первый музей", osm_type="way", osm_id=42)
        self.put("Второе имя музея", lon=104.88, osm_type="way", osm_id=42)
        result = self.run_grounding([hint("Первый музей"), hint("Второе имя музея")])
        self.assertEqual(len(result.confirmed_hints), 1)

    def test_same_numeric_id_different_osm_types_are_not_duplicates(self):
        self.put("Первый музей", osm_type="node", osm_id=42)
        self.put("Второй музей", lon=104.88, osm_type="way", osm_id=42)
        result = self.run_grounding([hint("Первый музей"), hint("Второй музей")])
        self.assertEqual(len(result.confirmed_hints), 2)

    def test_missing_or_invalid_osm_identity_still_uses_proximity(self):
        self.put("Первый музей", osm_type="way", osm_id=True)
        self.put("Второй музей", lon=104.86001, osm_type="way", osm_id="42")
        result = self.run_grounding([hint("Первый музей"), hint("Второй музей")])
        self.assertEqual(len(result.confirmed_hints), 1)
        self.assertNotIn("osm_id", result.confirmed_hints[0])

    def test_model_proposal_and_cache_seed_share_osm_deduplication(self):
        self.put("Первый музей", osm_type="relation", osm_id=42)
        self.put("Второе имя музея", lon=104.88, osm_type="relation", osm_id=42)
        payload = {"bases": [{"base_key": "base_1", "base_query": BASE_QUERY}], "pois": [
            {**hint(title), "locality": "Листвянка", "base_key": "base_1"}
            for title in ("Второе имя музея", "Первый музей")
        ]}
        result = self.run_grounding([hint("Первый музей")], payload=payload)
        self.assertEqual(len(result.confirmed_hints), 1)
        self.assertEqual([entry["reason"] for entry in result.rejected], ["duplicate_place", "duplicate_place"])

    def test_actual_county_canonical_alias_survives_normal_query_recheck(self):
        title = "Камень Черского"
        record = self.put(title, kind="peak")
        # The positive entry originally had a different valid country spelling.
        self.cache.pop(record["name"].lower())
        self.cache[f"{title}, {COUNTY}, Russia".lower()] = record
        result = self.run_grounding([hint(title)])
        canonical = result.confirmed_hints[0]["place_query"]
        self.assertIn(canonical.lower(), self.cache)
        self.assertTrue(places._place_matches_query(canonical, self.cache[canonical.lower()]))
        self.assertEqual(self.cache[canonical.lower()]["address"]["county"], COUNTY)


class GroundingBaseEnrichmentTests(unittest.TestCase):
    def setUp(self):
        self.old = {"name": BASE_QUERY, "lat": 51.86, "lon": 104.85,
                    "type": "village", "category": "place", "address": {"country_code": "ru"}}
        self.full = {**self.old, "address": {"village": "Листвянка", "county": COUNTY, "country_code": "ru"}}
        self.cache = {BASE_QUERY.lower(): deepcopy(self.old)}
        self.budget = grounding.GroundingResult(query_limit=21)
        for owner, name, options in (
            (places, "_cache", {"new": self.cache}),
            (places, "_save_cache", {"return_value": None}),
            (socket.socket, "connect", {"side_effect": AssertionError("No network")}),
        ):
            guard = patch.object(owner, name, **options)
            guard.start()
            self.addCleanup(guard.stop)

    def test_incomplete_positive_base_gets_one_exact_lookup_and_persists_complete_fact(self):
        with patch.object(places, "_search", return_value=self.full) as lookup, patch.object(places, "geocode", side_effect=AssertionError("Would return old cache again")):
            found, reason = grounding._lookup(BASE_QUERY, self.budget)
            lookup.assert_called_once()
            self.assertEqual(lookup.call_args.args, (BASE_QUERY,))
            self.assertEqual(lookup.call_args.kwargs["validation_query"], BASE_QUERY)
            self.assertTrue(lookup.call_args.kwargs["place_filter"](self.full))
            self.assertFalse(lookup.call_args.kwargs["place_filter"](self.old))
        self.assertEqual((found, reason), (self.full, ""))
        self.assertEqual((self.budget.verification_queries, self.budget.geocode_calls, self.budget.reserved_http_attempts_upper_bound), (1, 1, 2))
        self.assertEqual(self.cache[BASE_QUERY.lower()], self.full)
        self.assertEqual(self.old["address"], {"country_code": "ru"})

    def test_enriched_cache_does_not_repeat_lookup_on_next_use(self):
        with patch.object(places, "_search", return_value=self.full) as lookup:
            grounding._lookup(BASE_QUERY, self.budget)
            grounding._lookup(BASE_QUERY, self.budget)
        self.assertEqual(lookup.call_count, 1)
        self.assertEqual(self.budget.reserved_http_attempts_upper_bound, 2)

    def test_missing_type_metadata_is_enriched_even_when_address_is_present(self):
        self.cache[BASE_QUERY.lower()] = {key: value for key, value in self.full.items() if key not in {"type", "category"}}
        with patch.object(places, "_search", return_value=self.full) as lookup:
            found, reason = grounding._lookup(BASE_QUERY, self.budget)
        lookup.assert_called_once()
        self.assertEqual((found["type"], reason), ("village", ""))

    def test_complete_positive_base_remains_zero_http(self):
        self.cache[BASE_QUERY.lower()] = deepcopy(self.full)
        with patch.object(places, "_search", side_effect=AssertionError("Complete cache")), patch.object(places, "geocode", side_effect=AssertionError("Complete cache")):
            found, reason = grounding._lookup(BASE_QUERY, self.budget)
        self.assertEqual((found, reason), (self.full, ""))
        self.assertEqual((self.budget.geocode_calls, self.budget.reserved_http_attempts_upper_bound), (0, 0))

    def test_zero_budget_does_not_refresh_or_admit_incomplete_base(self):
        self.budget.http_attempt_limit = 0
        before = deepcopy(self.cache)
        with patch.object(places, "_search", side_effect=AssertionError("No budget")):
            found, reason = grounding._lookup(BASE_QUERY, self.budget)
        self.assertEqual((found, reason), (None, "http_budget_exhausted"))
        self.assertEqual(self.cache, before)
        self.assertEqual(self.budget.reserved_http_attempts_upper_bound, 0)

    def test_refresh_uses_remaining_shared_budget_without_going_above_42(self):
        self.budget.reserved_http_attempts_upper_bound = 40
        second = "Тальцы, Россия"
        self.cache[second.lower()] = {**self.old, "name": second}
        with patch.object(places, "_search", return_value=self.full) as lookup:
            self.assertEqual(grounding._lookup(BASE_QUERY, self.budget)[1], "")
            self.assertEqual(grounding._lookup(second, self.budget), (None, "http_budget_exhausted"))
        lookup.assert_called_once()
        self.assertEqual(self.budget.reserved_http_attempts_upper_bound, 42)

    def test_incomplete_or_wrong_type_refresh_is_not_saved_or_accepted(self):
        for refreshed in (self.old, {**self.full, "type": "lake", "category": "natural"}, None):
            with self.subTest(refreshed=refreshed), patch.object(places, "_search", return_value=refreshed) as lookup:
                found, reason = grounding._lookup(BASE_QUERY, self.budget)
                lookup.assert_called_once()
            self.assertEqual((found, reason), (None, "base_address_unconfirmed"))
            self.assertEqual(self.cache[BASE_QUERY.lower()], self.old)
            self.assertNotIn(BASE_QUERY.lower() + "|miss", self.cache)

    def test_mismatched_country_or_identity_refresh_cannot_replace_positive_cache(self):
        for refreshed in ({**self.full, "address": {"village": "Листвянка", "country_code": "us"}},
                          {**self.full, "name": "Другой населённый пункт, Россия"}):
            with self.subTest(refreshed=refreshed), patch.object(places, "_search", return_value=refreshed):
                self.assertEqual(grounding._lookup(BASE_QUERY, self.budget), (None, "identity_mismatch"))
            self.assertEqual(self.cache[BASE_QUERY.lower()], self.old)

    def test_real_search_retry_ceiling_is_reserved_and_failure_preserves_cache(self):
        with patch.object(places.httpx, "get", side_effect=httpx.ReadTimeout("offline timeout")) as http, patch.object(places, "_wait_for_turn"), patch.object(places.time, "sleep"):
            found, reason = grounding._lookup(BASE_QUERY, self.budget)
        self.assertEqual((found, reason), (None, "map_unavailable"))
        self.assertEqual(http.call_count, 2)
        self.assertEqual(self.budget.reserved_http_attempts_upper_bound, 2)
        self.assertEqual(self.cache[BASE_QUERY.lower()], self.old)

    def test_fresh_incomplete_map_response_does_not_trigger_second_enrichment_search(self):
        self.cache.clear()
        with patch.object(places, "geocode", return_value=self.old) as lookup, patch.object(places, "_search", side_effect=AssertionError("No second lookup")):
            found, reason = grounding._lookup(BASE_QUERY, self.budget)
        lookup.assert_called_once_with(BASE_QUERY, max_variants=1)
        self.assertEqual((found, reason), (None, "base_address_unconfirmed"))
        self.assertEqual(self.budget.reserved_http_attempts_upper_bound, 2)


if __name__ == "__main__":
    unittest.main()
