"""Independent map evidence survives grounding without acquiring an OSM ID."""
from copy import deepcopy
from datetime import date
import unittest
from unittest.mock import patch

import candidate_grounding as grounding
from destination_scope import RegionScope
import places
import regional_skeleton
import rules
from schemas import Itinerary, ItineraryDay, ItineraryItem
import verified_map_places as catalog


class CandidateMapProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.records = catalog.load_verified_map_places(today=date(2026, 10, 21))
        self.assertEqual(len(self.records), 2)
        loader = patch.object(catalog, "load_verified_map_places", return_value=self.records)
        loader.start()
        self.addCleanup(loader.stop)
        self.record = deepcopy(self.records[1])
        self.query = self.record["catalog_queries"][0]
        self.title = self.query.split(",")[0]
        self.record.update(place_query=self.query, locality="Слюдянка")
        self.bases = {"base_1": {"base_key": "base_1", "lat": 51.656769,
                      "lon": 103.719292, "map_address": self.record["address"]}}
        self.scope = RegionScope("Байкал", "Россия", (51, 53, 103, 106), {
            "type": "Polygon", "coordinates": [[[103, 51], [106, 51], [106, 53],
                                                 [103, 53], [103, 51]]],
        })
        self.result = grounding.GroundingResult()

    def confirm(self, record=None):
        return grounding._confirmed_fact(self.title, record or self.record, self.bases,
                                         self.scope, self.result, preferred_base_key="base_1")

    def test_preserves_verified_provider_and_source_without_osm_identity(self):
        fact, problem = self.confirm()
        self.assertEqual(problem, "")
        for key in catalog.PROVENANCE_FIELDS:
            self.assertEqual(fact[key], self.record[key])
        self.assertNotIn("osm_id", fact)
        self.assertNotIn("osm_type", fact)
        self.assertEqual((fact["lat"], fact["lon"]), (51.654642, 103.726667))

    def test_deduplicates_provider_identity_even_if_previous_coordinates_differ(self):
        fact, _ = self.confirm()
        self.result.confirmed_hints.append({**fact, "lat": fact["lat"] + .01})
        self.assertEqual(self.confirm(), (None, "duplicate_place"))

    def test_distinct_provider_ids_can_share_one_building(self):
        # The in-memory reviewed catalogue represents two co-located venues.
        self.records[0].update(lat=self.record["lat"], lon=self.record["lon"])
        other = deepcopy(self.records[0])
        query = other["catalog_queries"][0]
        other.update(place_query=query, locality="Слюдянка")
        fact, problem = grounding._confirmed_fact(query.split(",")[0], other, self.bases,
                                                  self.scope, self.result, preferred_base_key="base_1")
        self.assertEqual(problem, "")
        self.result.confirmed_hints.append(fact)
        second, problem = self.confirm()
        self.assertEqual(problem, "")
        self.assertIsNotNone(second)
        self.assertNotEqual(fact["provider_id"], second["provider_id"])
        self.result.confirmed_hints.append(second)
        self.result.base_facts = [{**self.bases["base_1"], "base_query": "Слюдянка, Россия"}]
        _, hints = regional_skeleton._grounded_pools(self.result)
        self.assertEqual(len({hint["_identity"] for hint in hints.values()}), 2)
        itinerary = Itinerary(title="Проверка", summary="", days=[ItineraryDay(
            day=1, date="2026-10-28", theme="", items=[ItineraryItem(
                time="10:00", title=hint["title"], description="", place_query=hint["place_query"],
                lat=hint["lat"], lon=hint["lon"], provider=hint["provider"],
                source_url=hint["source_url"], verified=True) for hint in self.result.confirmed_hints])])
        with patch.object(places, "_cache", {}), patch.object(places.httpx, "get", side_effect=AssertionError("No network")):
            self.assertEqual(rules.check_verified_place_duplicates(itinerary), [])
            # An expired catalogue, wrong source or moved point cannot grant an
            # authoritative identity exception to conservative proximity checks.
            item = itinerary.days[0].items[0]
            self.assertIsNotNone(places.resolved_map_identity(item))
            with patch.object(catalog, "load_verified_map_places", return_value=[]):
                self.assertIsNone(places.resolved_map_identity(item))
                self.assertEqual(len(rules.check_verified_place_duplicates(itinerary)), 1)
            for changes in ({"source_url": "https://example.invalid/forged"}, {"lat": item.lat + .000001}, {"place_query": "Чужой музей, Слюдянка, Россия"}):
                self.assertIsNone(places.resolved_map_identity(item.model_copy(update=changes)))

    def test_rejects_forged_or_expired_provenance(self):
        for changes in ({"source_url": "https://example.invalid/fake"},
                        {"provider_id": "12345"}, {"checked_at": "2020-01-01"},
                        {"lat": self.record["lat"] + .001}, {"osm_id": 123}):
            with self.subTest(changes=changes):
                self.assertEqual(self.confirm({**self.record, **changes}), (None, "identity_mismatch"))


if __name__ == "__main__":
    unittest.main()
