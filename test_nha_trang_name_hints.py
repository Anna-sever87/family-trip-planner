"""Reviewed map-name hint, distinct from an institution's official naming.

The museum website uses the shorter Bảo tàng Hải dương học. On 2026-10-21,
Nominatim returned relation/15662995 with the longer name below. The catalogue
links to that map identity, not to an institution page claiming this exact
variation. This is only a discovery seed, never opening/duration/indoor facts.
"""

import unittest

from city_candidates import _canonical_record
from destination_name_hints import destination_name_hints


class NhaTrangNameHintTests(unittest.TestCase):
    def record(self):
        return {
            "lat": "12.2077321", "lon": "109.2153621",
            "name": "Bảo tàng Hải dương học Nha Trang, Trần Phú, Phường Nha Trang, Nha Trang, Khánh Hòa, 57131, Việt Nam",
            "namedetails": {"name": "Bảo tàng Hải dương học Nha Trang"},
            "address": {"road": "Trần Phú", "suburb": "Phường Nha Trang", "city": "Nha Trang",
                        "state": "Khánh Hòa", "country": "Việt Nam", "country_code": "vn"},
            "category": "tourism", "type": "museum", "osm_type": "relation", "osm_id": 15662995,
        }

    def test_single_full_map_name_replaces_short_seed_and_keeps_other_institutions(self):
        hints = destination_name_hints("Нячанг, Вьетнам")
        self.assertEqual([hint["title"] for hint in hints], [
            "Bảo tàng Hải dương học Nha Trang", "Bảo tàng Alexandre Yersin", "Bảo tàng Khánh Hòa",
        ])
        self.assertEqual(hints[0]["source_url"], "https://www.openstreetmap.org/relation/15662995")
        self.assertEqual(set(hints[0]), {"title", "place_query", "locality", "source_url"})

    def test_observed_exact_name_passes_without_relaxing_global_matcher(self):
        center = {"lat": 12.24507, "lon": 109.19432, "country_code": "vn"}
        record = self.record()
        self.assertIsNone(_canonical_record("Bảo tàng Hải dương học", record, center))
        full = _canonical_record("Bảo tàng Hải dương học Nha Trang", record, center)
        self.assertIsNotNone(full)
        self.assertEqual(full["place_query"], "Bảo tàng Hải dương học Nha Trang, Nha Trang, Việt Nam")
        self.assertIsNone(_canonical_record("Bảo tàng Khánh Hòa", record, center))

    def test_name_does_not_override_country_type_or_radius_guards(self):
        center = {"lat": 12.24507, "lon": 109.19432, "country_code": "vn"}
        for mutation in ("country", "type", "radius"):
            record = self.record()
            if mutation == "country":
                record["address"]["country_code"] = "tr"
            elif mutation == "type":
                record.update(category="building", type="yes")
            else:
                record["lat"] = 13.24507
            with self.subTest(mutation=mutation):
                self.assertIsNone(_canonical_record("Bảo tàng Hải dương học Nha Trang", record, center))


if __name__ == "__main__":
    unittest.main()
