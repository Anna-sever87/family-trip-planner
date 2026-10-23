"""Narrow provider metadata proves an observatory type, never visitability."""

import unittest
from unittest.mock import patch

import httpx
import candidate_grounding as grounding
import places


TITLE = "Астрономическая обсерватория Тестовая"
QUERY = f"{TITLE}, Долинный, Россия"


def provider_record(kind="university", category="amenity", *, tags=None):
    return {"display_name": QUERY, "lat": "51.0", "lon": "101.01",
            "type": kind, "class": category, "addresstype": category,
            "address": {"village": "Долинный", "country": "Россия", "country_code": "ru"},
            "namedetails": {"name": TITLE},
            "extratags": {"observatory:type": "astronomical"} if tags is None else tags}


class SemanticPoiMetadataTests(unittest.TestCase):
    def setUp(self):
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

    def search(self, records, *, place_filter=grounding._eligible_poi_type):
        response = httpx.Response(200, json=records, request=httpx.Request("GET", "https://example.invalid"))
        with patch.object(places.httpx, "get", return_value=response) as http:
            result = places.geocode_nearby_named(TITLE, "Россия", 51.0, 101.0, place_filter=place_filter)
        return result, http

    def test_provider_tag_is_requested_whitelisted_and_persisted_as_positive_evidence(self):
        tags = {"observatory:type": "astronomical", "operator": "Not copied",
                "website": "https://example.org", "instructions": "Not copied", "opening_hours": "24/7"}
        result, http = self.search([provider_record(tags=tags)])
        self.assertEqual(http.call_count, 1)
        self.assertEqual(http.call_args.kwargs["params"]["extratags"], 1)
        self.assertEqual(http.call_args.kwargs["params"]["bounded"], 1)
        self.assertEqual(result["extratags"], {"observatory:type": "astronomical"})
        self.assertEqual(result["type"], "university")  # Never rewrite the provider's primary type.
        self.assertEqual(self.cache[QUERY.lower()]["extratags"], {"observatory:type": "astronomical"})
        self.assertNotIn("opening_hours", result)
        with patch.object(places.httpx, "get", side_effect=AssertionError("Cache reuse must not query")):
            cached = places.geocode_nearby_named(TITLE, "Россия", 51.0, 101.0, place_filter=grounding._eligible_poi_type)
        self.assertEqual(cached["place_query"], QUERY)

    def test_only_compatible_primary_types_with_exact_tag_are_eligible(self):
        for category, kind in (("amenity", "university"), ("office", "research")):
            for tags, expected in (({"observatory:type": "astronomical"}, True), ({}, False),
                                   ({"observatory:type": "meteorological"}, False),
                                   ({"observatory:type": "Astronomical"}, False),
                                   ({"observatory:type": True}, False),
                                   ({"website": "https://official.example"}, False), (None, False)):
                with self.subTest(category=category, kind=kind, tags=tags):
                    self.assertEqual(grounding._eligible_poi_type({"category": category, "type": kind, "extratags": tags}), expected)
        for category, kind in (("office", "ngo"), ("place", "locality"), ("building", "building"),
                               ("amenity", "research"), ("office", "university")):
            self.assertFalse(grounding._eligible_poi_type({"category": category, "type": kind,
                                                        "extratags": {"observatory:type": "astronomical"}}))

    def test_ordinary_university_is_skipped_for_same_response_tagged_observatory(self):
        result, http = self.search([provider_record(tags={}), provider_record("research", "office")])
        self.assertEqual(result["type"], "research")
        self.assertEqual(http.call_count, 1)

    def test_tag_cannot_bypass_identity_country_radius_or_caller_scope(self):
        wrong_name = {**provider_record(), "display_name": "Другая организация, Долинный, Россия", "namedetails": {}}
        wrong_country = {**provider_record(), "address": {"village": "Долинный", "country_code": "fr"}}
        too_far = {**provider_record(), "lat": "20.0", "lon": "20.0"}
        for record in (wrong_name, wrong_country, too_far):
            with self.subTest(record=record), self.assertRaises(places.GeocodeMismatch):
                self.search([record])
            self.assertEqual(self.cache, {})
        with self.assertRaises(places.GeocodeMismatch):
            self.search([provider_record()], place_filter=lambda record: grounding._eligible_poi_type(record) and record["lon"] < 101.0)
        self.assertEqual(self.cache, {})

    def test_malformed_extra_metadata_and_general_city_lookup_are_unchanged(self):
        for value in (None, [], "astronomical", {"observatory:type": ["astronomical"]}, {"observatory:type": {"value": "astronomical"}}):
            self.assertEqual(places._semantic_place_metadata(value), {})
        response = httpx.Response(200, json=[provider_record(tags={})], request=httpx.Request("GET", "https://example.invalid"))
        with patch.object(places.httpx, "get", return_value=response) as http:
            result = places._search(QUERY)
        self.assertEqual(result["type"], "university")
        self.assertNotIn("extratags", http.call_args.kwargs["params"])
        self.assertNotIn("extratags", result)


if __name__ == "__main__":
    unittest.main()
