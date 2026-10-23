"""Synthetic fixtures and a documented public provider-name regression."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
import socket
import sqlite3
import unittest
from unittest.mock import patch

import httpx
import places
from city_visit_estimates import (
    candidate_planning_data, check_city_visit_estimates, get_city_planning_cap,
    lookup_city_visit_estimate,
)
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def yersin_candidate():
    """Not a real provider observation; no actual coordinates or IDs claimed."""
    title = "Bảo tàng Alexandre Yersin"
    query = f"{title}, Nha Trang, Việt Nam"
    return SimpleNamespace(
        candidate_id="osm-node-101", title=title, place_query=query, area="Nha Trang",
        record={"name": query, "type": "museum", "provider": "osm", "osm_type": "node", "osm_id": 101,
                "lat": 0.0, "lon": 0.0,
                "address": {"city": "Nha Trang", "country": "Việt Nam", "country_code": "vn"}},
    )


def itinerary_for(candidate, duration=45, *, kind="attraction"):
    return Itinerary(title="Синтетическая проверка", summary="", days=[ItineraryDay(
        day=1, date="2099-10-28", theme="", items=[ItineraryItem(
            kind=kind, title=candidate.title, place_query=candidate.place_query, area=candidate.area,
            time="09:00", duration_min=duration, cost=123, description="Не изменять описание",
            indoor=False, kids_note="", verified=True, provider="osm", lat=0.0, lon=0.0,
        )],
    )])


def observed_yersin_candidate():
    """Public provider fields from nha-trang-children4-10-days5-policy4.

    The observed grave-accent spelling is not the municipality's official
    spelling. Coordinates/ID reproduce that saved observation, not a fixture
    claim about current visit access or conditions.
    """
    title = "Bào Tàng Alexandre Yersin"
    query = f"{title}, Nha Trang, Việt Nam"
    return SimpleNamespace(
        candidate_id="osm-way-1193057654", title=title, place_query=query, area="Nha Trang",
        record={
            "name": "Bào Tàng Alexandre Yersin, Trần Phú, Phước Hòa, Phường Nha Trang, Nha Trang, Khánh Hòa, 57100, Việt Nam",
            "type": "museum", "category": "tourism", "osm_type": "way", "osm_id": 1193057654,
            "lat": 12.2509232, "lon": 109.1961308,
            "namedetails": {"name": title, "name:vi": title, "name:en": "Museum of Yersin"},
            "address": {"road": "Trần Phú", "suburb": "Phước Hòa", "city": "Nha Trang",
                        "state": "Khánh Hòa", "country": "Việt Nam", "country_code": "vn"},
        },
    )


class CityVisitEstimateTests(unittest.TestCase):
    def test_observed_provider_spelling_binds_45_and_rejects_120(self):
        candidate = observed_yersin_candidate()
        self.assertEqual(get_city_planning_cap(candidate), 45)
        self.assertEqual(check_city_visit_estimates(itinerary_for(candidate, 45), [candidate]), [])
        self.assertEqual(len(check_city_visit_estimates(itinerary_for(candidate, 120), [candidate])), 1)

    def test_observed_spelling_still_requires_exact_country_locality_and_id(self):
        for changed in ("country", "locality", "identity"):
            candidate = observed_yersin_candidate()
            if changed == "country":
                candidate.record["address"]["country_code"] = "fr"
            elif changed == "locality":
                candidate.area = "Dalat"
                candidate.place_query = f"{candidate.title}, Dalat, Việt Nam"
                candidate.record["name"] = candidate.place_query
                candidate.record["address"]["city"] = "Dalat"
            else:
                candidate.candidate_id = "osm-way-1193057655"
            with self.subTest(changed=changed):
                self.assertIsNone(lookup_city_visit_estimate(candidate))

    def test_other_unreviewed_accents_are_not_fuzzy_aliases(self):
        candidate = observed_yersin_candidate()
        candidate.title = "Báo Tàng Alexandre Yersin"
        candidate.place_query = f"{candidate.title}, Nha Trang, Việt Nam"
        candidate.record["name"] = candidate.place_query
        candidate.record["namedetails"] = {"name": candidate.title, "name:vi": candidate.title}
        self.assertIsNone(lookup_city_visit_estimate(candidate))

    def test_one_observed_yersin_cannot_fill_one_family_program_day(self):
        from city_coverage import city_program_day_capacity
        request = TripRequest(destination="Нячанг, Вьетнам", start_date="2026-10-28", end_date="2026-11-01",
                              adults=2, children_ages=[4, 10], budget=3000, currency="USD",
                              start_time="08:00", end_time="21:00", pace="средний")
        self.assertEqual(city_program_day_capacity(request, [observed_yersin_candidate()]), 0)

    def test_native_exact_name_city_country_returns_reviewed_estimate(self):
        value = lookup_city_visit_estimate(yersin_candidate())
        self.assertEqual((value.typical_min_minutes, value.typical_max_minutes, value.planning_cap_minutes),
                         (30, 45, 45))
        self.assertEqual(value.checked_at, "2026-10-21")
        self.assertTrue(value.source_url.startswith("https://nhatrang.khanhhoa.gov.vn/"))

    def test_country_code_country_name_and_query_country_must_agree(self):
        for changed in ("code", "country", "query", "missing"):
            candidate = yersin_candidate()
            if changed == "code":
                candidate.record["address"]["country_code"] = "fr"
            elif changed == "country":
                candidate.record["address"]["country"] = "France"
            elif changed == "query":
                candidate.place_query = candidate.place_query.replace("Việt Nam", "France")
            else:
                del candidate.record["address"]["country_code"]
            with self.subTest(changed=changed):
                self.assertIsNone(lookup_city_visit_estimate(candidate))

    def test_same_name_elsewhere_and_substring_localities_do_not_match(self):
        for locality in ("Dalat", "Nha Trang district annex", "Trang", ""):
            candidate = yersin_candidate()
            candidate.area = locality
            candidate.place_query = f"{candidate.title}, {locality}, Vietnam"
            candidate.record["name"] = candidate.place_query
            candidate.record["address"]["city"] = locality
            with self.subTest(locality=locality):
                self.assertIsNone(lookup_city_visit_estimate(candidate))

    def test_documented_ward_and_provider_owned_sub_locality_keep_city_binding(self):
        candidate = yersin_candidate()
        candidate.area = "Phường Nha Trang"
        candidate.place_query = f"{candidate.title}, Phường Nha Trang, Vietnam"
        candidate.record["name"] = candidate.place_query
        candidate.record["address"]["city"] = "Phường Nha Trang"
        self.assertIsNotNone(lookup_city_visit_estimate(candidate))
        candidate.area = "Xương Huân"
        candidate.place_query = f"{candidate.title}, Xương Huân, Vietnam"
        candidate.record["name"] = candidate.place_query
        candidate.record["address"]["suburb"] = "Xương Huân"
        self.assertIsNotNone(lookup_city_visit_estimate(candidate))
        del candidate.record["address"]["city"]
        self.assertIsNone(lookup_city_visit_estimate(candidate))

    def test_canonical_title_query_area_record_and_provider_id_must_agree(self):
        mutations = (
            lambda c: setattr(c, "title", "Other museum"),
            lambda c: setattr(c, "area", "Other area"),
            lambda c: setattr(c, "candidate_id", "osm-node-999"),
            lambda c: c.record.update(name="Other museum, Nha Trang, Vietnam"),
            lambda c: c.record.update(type="artwork"),
            lambda c: c.record.update(osm_id=True),
            lambda c: c.record.update(address={}),
        )
        for mutate in mutations:
            candidate = yersin_candidate()
            mutate(candidate)
            with self.subTest(mutation=mutate):
                self.assertIsNone(lookup_city_visit_estimate(candidate))

    def test_no_partial_reordered_or_unreviewed_translated_name_matching(self):
        for title in ("Bảo tàng Alexandre Yersin annex", "Yersin Alexandre Bảo tàng", "Музей Йерсена",
                      "Bao tang Alexandre Yersin", "Another museum"):
            candidate = yersin_candidate()
            candidate.title = title
            candidate.place_query = f"{title}, Nha Trang, Vietnam"
            candidate.record["name"] = candidate.place_query
            with self.subTest(title=title):
                self.assertIsNone(lookup_city_visit_estimate(candidate))

    def test_provider_native_name_can_bind_international_canonical_name(self):
        candidate = yersin_candidate()
        candidate.title = "Yersin Museum"
        candidate.place_query = "Yersin Museum, Nha Trang, Vietnam"
        candidate.record["name"] = candidate.place_query
        self.assertIsNone(lookup_city_visit_estimate(candidate))
        candidate.record["namedetails"] = {"name:vi": "Bảo tàng Alexandre Yersin"}
        self.assertIsNotNone(lookup_city_visit_estimate(candidate))

    def test_model_or_map_duration_and_source_fields_never_raise_or_create_cap(self):
        candidate = yersin_candidate()
        candidate.duration_min = 180
        candidate.planning_cap_minutes = 180
        candidate.record.update(duration_min=180, planning_cap_minutes=180,
                                published_estimate={"planning_cap_minutes": 180},
                                source_url="https://untrusted.invalid/estimate")
        self.assertEqual(get_city_planning_cap(candidate), 45)
        candidate.title = "Other museum"
        candidate.place_query = "Other museum, Nha Trang, Vietnam"
        candidate.record["name"] = candidate.place_query
        self.assertIsNone(lookup_city_visit_estimate(candidate))
        self.assertEqual(get_city_planning_cap(candidate), 120)

    def test_policy_caps_are_conservative_defaults_not_map_evidence(self):
        for kind, cap in (("museum", 120), ("park", 120), ("garden", 120), ("archaeological_site", 120),
                          ("place_of_worship", 60), ("artwork", 60), ("monument", 60), ("memorial", 60),
                          ("aquarium", 120), ("zoo", 120), ("theme_park", 120), ("unknown", 120)):
            candidate = SimpleNamespace(record={"type": kind})
            with self.subTest(kind=kind):
                self.assertEqual(get_city_planning_cap(candidate), cap)
                data = candidate_planning_data(candidate)
                self.assertEqual((data["basis"], data["planning_cap_minutes"]), ("product_policy", cap))
                self.assertNotIn("published_estimate", data)

    def test_prompt_data_distinguishes_published_range_from_product_cap(self):
        candidate = yersin_candidate()
        data = candidate_planning_data(candidate)
        self.assertEqual((data["basis"], data["policy_cap_minutes"], data["planning_cap_minutes"]),
                         ("product_policy", 120, 45))
        self.assertEqual(data["published_estimate"]["basis"], "published_typical_duration")
        self.assertIn("не физический максимум", data["note"])
        self.assertNotIn("indoor", data)
        self.assertNotIn("opening_hours", data)
        data["published_estimate"]["planning_cap_minutes"] = 180
        self.assertEqual(candidate_planning_data(candidate)["planning_cap_minutes"], 45)
        with self.assertRaises(FrozenInstanceError):
            lookup_city_visit_estimate(candidate).planning_cap_minutes = 180

    def test_published_boundary_and_observed_ninety_and_one_eighty_are_rejected(self):
        candidate = yersin_candidate()
        for duration, errors in ((30, 0), (45, 0), (46, 1), (90, 1), (180, 1)):
            with self.subTest(duration=duration):
                problems = check_city_visit_estimates(itinerary_for(candidate, duration), [candidate])
                self.assertEqual(len(problems), errors)
                if errors:
                    self.assertIn("30–45", problems[0])
                    self.assertIn("не физический максимум", problems[0])

    def test_generic_policy_is_enforced_without_fabricated_published_source(self):
        candidate = yersin_candidate()
        candidate.title = "Other museum"
        candidate.place_query = "Other museum, Nha Trang, Vietnam"
        candidate.record["name"] = candidate.place_query
        for duration, errors in ((120, 0), (121, 1), (180, 1)):
            problems = check_city_visit_estimates(itinerary_for(candidate, duration), [candidate])
            self.assertEqual(len(problems), errors)
            if errors:
                self.assertIn("правило продукта", problems[0])
                self.assertNotIn("Источник:", problems[0])

    def test_invalid_bypassed_duration_is_not_accepted_for_a_bound_visit(self):
        candidate = yersin_candidate()
        for duration in (True, 0, -1, 45.0, float("nan"), "45"):
            itinerary = itinerary_for(candidate)
            itinerary.days[0].items[0] = itinerary.days[0].items[0].model_copy(update={"duration_min": duration})
            with self.subTest(duration=duration):
                self.assertEqual(len(check_city_visit_estimates(itinerary, [candidate])), 1)

    def test_pauses_are_not_visits_and_unknown_queries_still_need_map_validator(self):
        candidate = yersin_candidate()
        for kind in ("meal", "rest"):
            self.assertEqual(check_city_visit_estimates(itinerary_for(candidate, 180, kind=kind), [candidate]), [])
        itinerary = itinerary_for(candidate, 180)
        itinerary.days[0].items[0].place_query += " unknown"
        self.assertEqual(check_city_visit_estimates(itinerary, [candidate]), [])

    def test_pure_helpers_preserve_all_fields_and_never_read_files_network_or_database(self):
        candidate = yersin_candidate()
        itinerary = itinerary_for(candidate, 180)
        before = deepcopy((candidate, itinerary.model_dump()))
        with patch.object(httpx, "get", side_effect=AssertionError("No HTTP")), \
                patch.object(places, "geocode", side_effect=AssertionError("No map")), \
                patch.object(places, "_load_cache", side_effect=AssertionError("No cache")), \
                patch.object(Path, "read_text", side_effect=AssertionError("No files")), \
                patch.object(sqlite3, "connect", side_effect=AssertionError("No database")), \
                patch.object(socket.socket, "connect", side_effect=AssertionError("No socket")):
            self.assertEqual(get_city_planning_cap(candidate), 45)
            self.assertEqual(candidate_planning_data(candidate)["planning_cap_minutes"], 45)
            self.assertEqual(len(check_city_visit_estimates(itinerary, [candidate])), 1)
        self.assertEqual((candidate, itinerary.model_dump()), before)


if __name__ == "__main__":
    unittest.main()
