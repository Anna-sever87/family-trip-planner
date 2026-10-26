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
    def test_observed_transport_services_and_unnamed_bridge_cannot_fill_days(self):
        labels = {
            "Cambodia Bus Companies (many)": "service_component_not_visit",
            "Example Taxi Operator": "service_component_not_visit",
            "Bãi chờ ra ca đi nhậu": "service_component_not_visit",
            "Cầu": "generic_name_or_non_visit",
            "Street Art Street (partially demolished)": "explicit_structure_warning",
        }
        for title, reason in labels.items():
            with self.subTest(title=title):
                raw = fixtures.record(title, category="tourism", type="attraction")
                record = fixtures.cached(raw)
                before = deepcopy(record)
                self.assertIsNone(city_candidates._canonical_record(title, record, CENTER))
                self.assertEqual(city_candidates._rejection_detail(title, raw, CENTER), reason)
                self.assertEqual(record, before)

    def test_transport_museums_named_bridges_tours_and_historic_ruins_remain_eligible(self):
        for title, category, kind in (
                ("National Bus Company Museum", "tourism", "museum"),
                ("Taxi Company Heritage Museum", "tourism", "museum"),
                ("Museum of Bus Companies", "tourism", "attraction"),
                ("Музей транспорта Cambodia Bus Company", "tourism", "museum"),
                ("City Bus Tours", "tourism", "attraction"),
                ("Cầu Ánh Sao", "tourism", "attraction"),
                ("Ancient Castle Ruins", "historic", "ruins"),
                ("Old Temple (partially demolished)", "historic", "ruins"),
                ("Demolition History Museum", "tourism", "museum")):
            with self.subTest(title=title):
                record = fixtures.cached(fixtures.record(title, category=category, type=kind))
                self.assertIsNotNone(city_candidates._canonical_record(title, record, CENTER))

    def test_source_warnings_are_not_inferred_from_address_or_description(self):
        record = fixtures.cached(fixtures.record("Example City Gallery", category="tourism", type="attraction"))
        record['address']['road'] = "Bus Companies Street"
        record['description'] = "Nearby street is partially demolished; taxi company at corner."
        self.assertFalse(city_candidates._service_component(record))
        self.assertEqual(city_candidates._explicit_visit_label_rejection(record), "")
        self.assertIsNotNone(city_candidates._canonical_record("Example City Gallery", record, CENTER))

    def test_explicit_warning_in_observed_name_alias_is_preserved_as_rejection_not_renamed(self):
        record = fixtures.cached(fixtures.record("Street Art Street", category="tourism", type="attraction"))
        record['namedetails']['name:en'] = "Street Art Street (partially demolished)"
        self.assertEqual(city_candidates._explicit_visit_label_rejection(record), "explicit_structure_warning")
        self.assertIsNone(city_candidates._canonical_record("Street Art Street", record, CENTER))

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

    def _provider_case(self, *, warm, title=PARKING, category="shop", kind="mall",
                       reason="service_component_not_visit"):
        fixture = fixtures.CityCandidateTests(methodName="runTest")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        raw = fixtures.record(title, category=category, type=kind)
        fixture.get.return_value = fixtures.response([raw])
        if warm:
            fixture.cache[f"{title}, {fixtures.AREA}, Россия".lower()] = fixtures.cached(raw)
        result = fixture.run_pool([title], interest_categories=["shopping_malls" if category == "shop" else "main_sights"])
        self.assertFalse(result.candidates)
        self.assertGreater(result.diagnostics["detail_rejection_reasons"].get(reason, 0), 0)
        self.assertLessEqual(result.map_requests, city_candidates.MAX_MAP_REQUESTS)
        fixture.completion.assert_called_once()

    def test_live_and_cached_broad_tags_do_not_restore_service_or_damaged_structure(self):
        for warm in (False, True):
            for title, reason in (("Cambodia Bus Companies (many)", "service_component_not_visit"),
                                  ("Cầu", "generic_name_or_non_visit"),
                                  ("Street Art Street (partially demolished)", "explicit_structure_warning")):
                with self.subTest(warm=warm, title=title):
                    self._provider_case(warm=warm, title=title, category="tourism", kind="attraction", reason=reason)

    def test_provider_mall_tag_does_not_override_parking_name(self):
        self._provider_case(warm=False)

    def test_cached_parent_mall_tag_does_not_restore_rejected_parking(self):
        self._provider_case(warm=True)


if __name__ == "__main__":
    unittest.main()
