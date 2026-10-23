"""Read-only regressions for qualified Kemer discovery names, not map facts."""

import unittest
from urllib.parse import urlsplit

from destination_name_hints import destination_name_hints


class KemerNameHintsTests(unittest.TestCase):
    def test_qualified_aliases_preserve_official_names_and_real_localities(self):
        expected = [
            ("Phaselis", "Tekirova"),
            ("Folklorik Yörük Parkı", "Kemer"),
            ("Olbia", "Kemer"),
            ("Ayışığı Parkı", "Kemer"),
        ]
        for alias in ("Кемер, Анталья, Турция", "Kemer, Antalya, Türkiye",
                      "Kemer, Antalya, Turkey"):
            with self.subTest(alias=alias):
                records = destination_name_hints(alias)
                self.assertEqual([(r["title"], r["locality"]) for r in records], expected)
                for record in records:
                    self.assertEqual(record["place_query"],
                                     f'{record["title"]}, {record["locality"]}, Türkiye')

    def test_ambiguous_or_wrong_province_destination_gets_no_kemer_hints(self):
        for alias in ("Кемер", "Кемер, Турция", "Kemer", "Kemer, Turkey",
                      "Kemer, Burdur, Turkey", "Кемер, Бурдур, Турция"):
            with self.subTest(alias=alias):
                self.assertEqual(destination_name_hints(alias), [])

    def test_names_are_primary_source_discovery_only_and_other_catalogues_survive(self):
        records = destination_name_hints("Кемер, Анталья, Турция")
        self.assertEqual(len(records), 4)
        for record in records:
            self.assertEqual(set(record), {"title", "place_query", "locality", "source_url"})
            self.assertIn(urlsplit(record["source_url"]).hostname,
                          {"www.kemer.bel.tr", "www.muze.gov.tr"})
        self.assertFalse(any("Tahtalı" in r["title"] or "Archaeology" in r["title"]
                             for r in records))
        self.assertEqual(len(destination_name_hints("Нячанг, Вьетнам")), 3)
        self.assertEqual(len(destination_name_hints("Магадан, Россия")), 6)
        self.assertTrue(destination_name_hints("Байкал"))

    def test_full_official_park_name_matches_observed_map_without_alias_weakening(self):
        # Public provider fields observed on 2026-10-21, not invented map facts.
        # The nearby museum node is deliberately not a second catalogue visit.
        from city_candidates import _canonical_record
        record = {
            "lat": "36.6004167", "lon": "30.5764295",
            "name": "Folklorik Yörük Parkı, Merkez, Kemer, Antalya, Akdeniz Bölgesi, Türkiye",
            "namedetails": {"name": "Folklorik Yörük Parkı"},
            "address": {"park": "Folklorik Yörük Parkı", "suburb": "Merkez", "town": "Kemer",
                        "province": "Antalya", "region": "Akdeniz Bölgesi",
                        "country": "Türkiye", "country_code": "tr"},
            "category": "leisure", "type": "park", "osm_type": "way", "osm_id": 176919554,
        }
        center = {"lat": 36.59778, "lon": 30.56056, "country_code": "tr"}
        self.assertIsNone(_canonical_record("Yörük Parkı", record, center))
        full = _canonical_record("Folklorik Yörük Parkı", record, center)
        self.assertIsNotNone(full)
        self.assertEqual(full["place_query"], "Folklorik Yörük Parkı, Kemer, Türkiye")
        hints = destination_name_hints("Кемер, Анталья, Турция")
        self.assertEqual(sum("Yörük" in hint["title"] for hint in hints), 1)


if __name__ == "__main__":
    unittest.main()
