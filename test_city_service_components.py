"""Observed service labels must not become standalone museum/mall visits.

Names reproduce observed labels; all coordinates, IDs and provider calls below
are synthetic. No real place, address, price or opening time is asserted.
"""
from copy import deepcopy
import unittest

import city_candidates
import test_city_candidates as fixtures


PARKING = "Nhà đậu xe Aeon Mall Tân Phú Celadon;Commercial_AEONMallTanPhu Storie 4"
ENTRANCE = "アオザイ博物館入口"
CENTER = {**fixtures.CENTER, "country_code": "ru"}


class CityServiceComponentTests(unittest.TestCase):
    def test_actual_observed_component_labels_are_not_canonical_visits(self):
        for title, category, kind in ((PARKING, "shop", "mall"), (ENTRANCE, "tourism", "museum")):
            with self.subTest(title=title):
                record = fixtures.cached(fixtures.record(title, category=category, type=kind))
                self.assertIsNone(city_candidates._canonical_record(title, record, CENTER))
                self.assertEqual(city_candidates._rejection_detail(title, record, CENTER),
                                 "service_component_not_visit")

    def test_museum_and_mall_parents_remain_distinct_valid_records(self):
        for title, category, kind in (("Aeon Mall Tân Phú Celadon", "shop", "mall"),
                                      ("アオザイ博物館", "tourism", "museum")):
            with self.subTest(title=title):
                record = fixtures.cached(fixtures.record(title, category=category, type=kind))
                canonical = city_candidates._canonical_record(title, record, CENTER)
                self.assertIsNotNone(canonical)
                self.assertEqual(canonical["name"], record["name"])

    def test_service_names_in_multiple_languages_and_unicode_forms_are_rejected(self):
        for title in ("Parking garage at Example Mall", "Car park Example Museum",
                      "Парковка музея", "Вход в музей", "Example Museum entrance",
                      "Entrance to Example Museum", "Касса театра", "Nhà đậu xe Example",
                      "NHA DAU XE Example", "Bãi đỗ xe Example", "Example Museum入り口"):
            with self.subTest(title=title):
                self.assertTrue(city_candidates._service_component({"name": title}))

    def test_service_words_in_addresses_or_business_descriptions_do_not_reject_poi(self):
        record = {"name": "Example Mall, Parking garage street, Test City",
                  "address": {"road": "Parking garage street"},
                  "description": "Entrance to the underground parking is nearby"}
        self.assertFalse(city_candidates._service_component(record))

    def test_legitimate_names_are_not_rejected_by_substrings(self):
        for title in ("Parking Museum", "Музей истории парковки", "Entrance Gallery",
                      "Park Museum", "Parker Gallery", "Бранденбургские ворота", "天安门"):
            with self.subTest(title=title):
                self.assertFalse(city_candidates._service_component({"name": title}))

    def test_observed_language_alias_reveals_component_without_renaming(self):
        record = {"name": "Example Mall", "namedetails": {"name:vi": "Nhà đậu xe Example Mall"}}
        before = deepcopy(record)
        self.assertTrue(city_candidates._service_component(record))
        self.assertEqual(record, before)

    def _provider_case(self, *, warm):
        fixture = fixtures.CityCandidateTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        raw = fixtures.record(PARKING, category="shop", type="mall")
        fixture.get.return_value = fixtures.response([raw])
        if warm:
            fixture.cache[f"{PARKING}, {fixtures.AREA}, Россия".lower()] = fixtures.cached(raw)
        result = fixture.run_pool([PARKING], interest_categories=["shopping_malls"])
        self.assertFalse(result.candidates)
        self.assertGreater(result.diagnostics["detail_rejection_reasons"].get("service_component_not_visit", 0), 0)
        self.assertLessEqual(result.map_requests, city_candidates.MAX_MAP_REQUESTS)

    def test_provider_mall_tag_does_not_override_parking_name(self):
        self._provider_case(warm=False)

    def test_cached_parent_mall_tag_does_not_restore_rejected_parking(self):
        self._provider_case(warm=True)


if __name__ == "__main__":
    unittest.main()
