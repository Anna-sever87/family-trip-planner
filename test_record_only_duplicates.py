"""Fresh trusted map IDs prevent false proximity duplicates without cache writes."""

from copy import deepcopy
from datetime import date
import socket
import sqlite3
import unittest
from unittest.mock import patch

import places
import rules
from schemas import Itinerary, ItineraryDay, ItineraryItem
import verified_map_places as catalog


def osm_record(title, identifier, *, lat=52.0, lon=105.0, osm_type="node"):
    return {
        "name": f"{title}, Тестовый город, Россия", "namedetails": {"name": title},
        "address": {"city": "Тестовый город", "country": "Россия", "country_code": "ru"},
        "lat": lat, "lon": lon, "category": "tourism", "type": "museum",
        "osm_type": osm_type, "osm_id": identifier,
    }


def record_query(record):
    return record.get("catalog_queries", [record["name"]])[0]


def route_for(records, *, separate_days=False):
    items = [ItineraryItem(time="10:00", title=record["namedetails"]["name"], description="",
                           area=record["address"]["city"], place_query=record_query(record))
             for record in records]
    days = ([ItineraryDay(day=index + 1, date=f"2099-06-{index + 1:02d}", theme="", items=[item])
             for index, item in enumerate(items)] if separate_days else
            [ItineraryDay(day=1, date="2099-06-01", theme="", items=items)])
    return Itinerary(title="Проверка дублей", summary="", days=days)


class RecordOnlyDuplicateTests(unittest.TestCase):
    def setUp(self):
        self.cache = {}
        self.patched(places, "_cache", new=self.cache)
        for owner, name in ((places, "_save_cache"), (places, "geocode"),
                            (places.httpx, "get"), (socket.socket, "connect"),
                            (socket, "getaddrinfo"), (sqlite3, "connect")):
            self.patched(owner, name, side_effect=AssertionError("No external I/O or writes"))
        self.catalogue = []
        self.patched(catalog, "load_verified_map_places", side_effect=lambda: deepcopy(self.catalogue))

    def patched(self, owner, name, **kwargs):
        handle = patch.object(owner, name, **kwargs)
        value = handle.start()
        self.addCleanup(handle.stop)
        return value

    def verified(self, records, *, separate_days=False):
        itinerary = route_for(records, separate_days=separate_days)
        mapping = {record_query(record): deepcopy(record) for record in records}
        self.assertEqual(places.verify_itinerary(itinerary, verified_records=mapping), [])
        return itinerary, mapping

    def test_distinct_fresh_ids_at_same_point_pass_only_explicit_record_path(self):
        records = [osm_record("Музей ремёсел", 123), osm_record("Музей керамики", 124)]
        itinerary, mapping = self.verified(records)
        before = deepcopy((itinerary.model_dump(), mapping, self.cache))
        self.assertEqual(len(rules.check_verified_place_duplicates(itinerary)), 1)
        self.assertEqual(rules.check_verified_place_duplicates(itinerary, verified_records=mapping), [])
        self.assertEqual(len(rules.check_verified_place_duplicates(itinerary, verified_records={})), 1)
        self.assertEqual((itinerary.model_dump(), mapping, self.cache), before)

    def test_same_id_aliases_remain_duplicates_across_days_and_distant_coordinates(self):
        records = [osm_record("Музей ремёсел", 123), osm_record("Галерея ремёсел", 123, lat=52.02)]
        itinerary, mapping = self.verified(records, separate_days=True)
        problems = rules.check_verified_place_duplicates(itinerary, verified_records=mapping)
        self.assertEqual(len(problems), 1)
        self.assertIn("Дни 1 и 2", problems[0])

    def test_osm_type_is_part_of_identity(self):
        records = [osm_record("Музей ремёсел", 123), osm_record("Музей керамики", 123, osm_type="way")]
        itinerary, mapping = self.verified(records)
        self.assertEqual(rules.check_verified_place_duplicates(itinerary, verified_records=mapping), [])

    def test_missing_wrong_key_or_foreign_record_cannot_exempt_proximity(self):
        itinerary, mapping = self.verified([osm_record("Музей ремёсел", 123), osm_record("Музей керамики", 124)])
        query = itinerary.days[0].items[1].place_query
        for label in ("missing", "case_changed_key", "foreign_name", "foreign_country", "foreign_city"):
            records = deepcopy(mapping)
            target = records.pop(query)
            if label == "case_changed_key": records[query.lower()] = target
            elif label != "missing":
                if label == "foreign_name": target.update(name="Другой объект, Тестовый город, Россия", namedetails={})
                elif label == "foreign_country": target["address"]["country_code"] = "vn"
                else:
                    target["name"] = target["name"].replace("Тестовый город", "Другой город")
                    target["address"]["city"] = "Другой город"
                records[query] = target
            with self.subTest(label=label):
                self.assertEqual(len(rules.check_verified_place_duplicates(itinerary, verified_records=records)), 1)

    def test_record_coordinates_provider_or_identity_mismatch_cannot_exempt_proximity(self):
        itinerary, mapping = self.verified([osm_record("Музей ремёсел", 123), osm_record("Музей керамики", 124)])
        query = itinerary.days[0].items[1].place_query
        mutations = ({"lat": 52.001}, {"lon": 105.001}, {"lat": float("nan")}, {"lat": True},
                     {"provider": "unknown"}, {"provider": "2gis"}, {"osm_id": True}, {"osm_id": 0},
                     {"osm_type": "unknown"}, {"source_url": "https://example.invalid/forged"})
        for mutation in mutations:
            records = deepcopy(mapping)
            records[query].update(mutation)
            with self.subTest(mutation=mutation):
                self.assertEqual(len(rules.check_verified_place_duplicates(itinerary, verified_records=records)), 1)

    def test_item_coordinate_query_and_provider_must_match_record(self):
        itinerary, mapping = self.verified([osm_record("Музей ремёсел", 123), osm_record("Музей керамики", 124)])
        for changes in ({"lat": 52.000001}, {"lon": 105.000001}, {"provider": "2gis"},
                        {"provider": ""}, {"place_query": "Чужое место, Тестовый город, Россия"}):
            candidate = itinerary.model_copy(deep=True)
            candidate.days[0].items[1] = candidate.days[0].items[1].model_copy(update=changes)
            with self.subTest(changes=changes):
                self.assertEqual(len(rules.check_verified_place_duplicates(candidate, verified_records=mapping)), 1)

    def test_model_item_ids_are_never_identity_evidence(self):
        itinerary, mapping = self.verified([osm_record("Музей ремёсел", 123), osm_record("Музей керамики", 124)])
        for item in itinerary.days[0].items:
            item.__dict__.update(osm_type="node", osm_id=999, provider_id="888")
        self.assertEqual(rules.check_verified_place_duplicates(itinerary, verified_records=mapping), [])
        self.assertEqual(len(rules.check_verified_place_duplicates(itinerary)), 1)

    def test_non_attraction_or_unverified_item_never_uses_supplied_record(self):
        itinerary, mapping = self.verified([osm_record("Музей ремёсел", 123), osm_record("Музей керамики", 124)])
        item = itinerary.days[0].items[1]
        for changes in ({"verified": False}, {"verified": 1}, {"kind": "rest"}, {"lat": True}):
            with self.subTest(changes=changes):
                changed = item.model_copy(update=changes)
                self.assertIsNone(rules._record_only_duplicate_identity(changed, mapping))

    def independent_record(self, title, identifier):
        query = f"{title}, Тестовый город, Россия"
        return catalog._record({
            "title": title, "place_queries": [query], "lat": 52.0, "lon": 105.0, "type": "museum",
            "address": {"road": "Улица тестов", "house_number": "1", "city": "Тестовый город",
                        "county": "Тестовый район", "state": "Тестовая область", "country": "Россия", "country_code": "ru"},
            "provider": "2gis", "provider_id": identifier,
            "source_url": f"https://2gis.ru/test/firm/{identifier}", "official_url": "https://example.invalid/museum",
            "official_link_url": f"https://go.2gis.com/test{identifier}", "checked_at": date.today().isoformat(),
        }, date.today())

    def test_independent_provider_current_evidence_source_link_and_expiry(self):
        self.catalogue = [self.independent_record("Музей ремёсел", "123"), self.independent_record("Музей керамики", "124")]
        itinerary, mapping = self.verified(self.catalogue)
        self.assertEqual(rules.check_verified_place_duplicates(itinerary, verified_records=mapping), [])
        item = itinerary.days[0].items[1]
        self.assertIsNotNone(rules._record_only_duplicate_identity(item, mapping))
        self.assertIsNone(rules._record_only_duplicate_identity(item.model_copy(update={"source_url": "https://example.invalid/forged"}), mapping))
        self.catalogue = []
        self.assertEqual(len(rules.check_verified_place_duplicates(itinerary, verified_records=mapping)), 1)

    def test_different_providers_do_not_prove_distinct_places_at_same_point(self):
        independent = self.independent_record("Музей керамики", "124")
        self.catalogue = [independent]
        itinerary, mapping = self.verified([osm_record("Музей ремёсел", 123), independent])
        self.assertEqual(len(rules.check_verified_place_duplicates(itinerary, verified_records=mapping)), 1)


if __name__ == "__main__":
    unittest.main()
