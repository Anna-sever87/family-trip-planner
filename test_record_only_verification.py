"""Server-owned map records verify routes without any lookup or cache writes."""

from copy import deepcopy
from datetime import date
import socket
import sqlite3
from types import MappingProxyType
import unittest
from unittest.mock import patch

import places
import verified_map_places as catalog
from schemas import Itinerary, ItineraryDay, ItineraryItem


TITLE = "Музей тестовых ремёсел"
AREA = "Тестовый город"
QUERY = f"{TITLE}, {AREA}, Россия"


def record(**changes):
    result = {
        "lat": 52.0, "lon": 105.0, "name": QUERY,
        "type": "museum", "category": "tourism",
        "address": {"city": AREA, "country": "Россия", "country_code": "ru"},
        "osm_type": "node", "osm_id": 123,
    }
    result.update(changes)
    return result


def route(*, query=QUERY, area=AREA):
    return Itinerary(title="Проверка", summary="Синтетические данные", days=[
        ItineraryDay(day=1, date="2099-06-01", theme="Музей", items=[
            ItineraryItem(time="10:00", title=TITLE, place_query=query, area=area,
                          description="Посещение", verified=True, lat=1, lon=2,
                          official_name="Поддельное имя", provider="2gis",
                          source_url="https://example.invalid/model-url"),
        ]),
    ])


def independent_record():
    return catalog._record({
        "title": TITLE, "place_queries": [QUERY], "lat": 52.0, "lon": 105.0,
        "address": {"city": AREA, "county": "Тестовый округ", "state": "Тестовая область",
                    "country": "Россия", "country_code": "ru", "road": "Тестовая улица",
                    "house_number": "1"},
        "type": "museum", "provider": "2gis", "provider_id": "123",
        "source_url": "https://2gis.ru/test/firm/123",
        "official_url": "https://museum.example.test/contacts",
        "official_link_url": "https://go.2gis.com/testlink",
        "checked_at": "2026-10-21",
    }, date(2026, 10, 21))


class RecordOnlyVerificationTests(unittest.TestCase):
    def setUp(self):
        self.blockers = []
        for owner, name in (
            (places, "geocode"), (places, "geocode_nearby_named"), (places, "_search"),
            (places, "_load_cache"), (places, "_save_cache"),
            (places.httpx, "get"), (places.httpx, "post"),
            (socket.socket, "connect"), (socket, "getaddrinfo"), (sqlite3, "connect"),
        ):
            blocker = patch.object(owner, name, side_effect=AssertionError("External I/O forbidden"))
            self.blockers.append(blocker.start())
            self.addCleanup(blocker.stop)
        self.current_records = []
        loader = patch.object(catalog, "load_verified_map_places",
                              side_effect=lambda: deepcopy(self.current_records))
        loader.start()
        self.addCleanup(loader.stop)

    def tearDown(self):
        for blocker in self.blockers:
            blocker.assert_not_called()

    def assert_cleared(self, itinerary):
        item = itinerary.days[0].items[0]
        self.assertFalse(item.verified)
        self.assertEqual((item.lat, item.lon, item.official_name, item.provider, item.source_url),
                         (None, None, "", "", ""))

    def test_record_only_success_replaces_model_metadata_without_mutating_pool(self):
        itinerary = route()
        facts = {QUERY: record()}
        before = deepcopy(facts)
        self.assertEqual(places.verify_itinerary(itinerary, verified_records=MappingProxyType(facts)), [])
        item = itinerary.days[0].items[0]
        self.assertEqual((item.lat, item.lon, item.official_name, item.provider),
                         (52.0, 105.0, QUERY, "osm"))
        self.assertTrue(item.verified)
        self.assertTrue(item.source_url.startswith("https://www.openstreetmap.org/"))
        self.assertNotIn("model-url", item.source_url)
        self.assertEqual(facts, before)

    def test_empty_pool_and_nonexact_query_never_fall_back_to_geocode(self):
        for facts, query in (({}, QUERY), ({QUERY.lower(): record()}, QUERY),
                             ({QUERY: record()}, QUERY + " "), ({QUERY: record()}, "")):
            with self.subTest(query=query):
                itinerary = route(query=query)
                self.assertTrue(places.verify_itinerary(itinerary, verified_records=facts))
                self.assert_cleared(itinerary)

    def test_invalid_coordinates_fail_closed(self):
        for field, values in (("lat", (True, "52.0", None, float("nan"), float("inf"), 91)),
                              ("lon", (False, "105.0", float("-inf"), 181))):
            for value in values:
                with self.subTest(field=field, value=value):
                    itinerary = route()
                    self.assertTrue(places.verify_itinerary(
                        itinerary, verified_records={QUERY: record(**{field: value})}))
                    self.assert_cleared(itinerary)

    def test_wrong_name_country_or_locality_is_rejected(self):
        for wrong in (
            record(name=f"Другой музей, {AREA}, Россия"),
            record(address={"city": AREA, "country": "США", "country_code": "us"}),
            record(name=f"{TITLE}, Другой город, Россия",
                   address={"city": "Другой город", "country_code": "ru"}),
            record(name=None),
            None,
            [],
        ):
            with self.subTest(record=wrong):
                itinerary = route()
                self.assertTrue(places.verify_itinerary(itinerary, verified_records={QUERY: wrong}))
                self.assert_cleared(itinerary)

    def test_every_explicit_context_component_is_checked(self):
        query = f"{TITLE}, {AREA}, Тестовая область, Россия"
        for place, accepted in (
            (record(address={"city": AREA, "state": "Тестовая область", "country_code": "ru"}), True),
            (record(address={"city": AREA, "state": "Другая область", "country_code": "ru"}), False),
        ):
            with self.subTest(accepted=accepted):
                itinerary = route(query=query)
                problems = places.verify_itinerary(itinerary, verified_records={query: place})
                self.assertEqual(not problems, accepted)
                if not accepted:
                    self.assert_cleared(itinerary)

    def test_incomplete_query_cannot_drop_country_or_locality(self):
        for query in (TITLE, f"{TITLE}, Россия", f"{TITLE}, , Россия"):
            with self.subTest(query=query):
                itinerary = route(query=query)
                self.assertTrue(places.verify_itinerary(itinerary, verified_records={query: record()}))
                self.assert_cleared(itinerary)

    def test_invalid_provider_or_identity_cannot_become_osm(self):
        for changes in (
            {"provider": "unknown"}, {"provider": None}, {"provider": ["osm"]},
            {"provenance": "model"}, {"source_url": "https://example.invalid/model-url"},
            {"osm_type": "node", "osm_id": True}, {"osm_type": "node", "osm_id": "123"},
            {"osm_type": None}, {"osm_id": None},
            {"provider": "2gis", "provider_id": "123"},
        ):
            with self.subTest(changes=changes):
                itinerary = route()
                self.assertTrue(places.verify_itinerary(itinerary, verified_records={QUERY: record(**changes)}))
                self.assert_cleared(itinerary)

    def test_area_requires_record_address_even_for_generic_center_or_empty_area(self):
        for area in ("", "Центр", "Тестовый", "Другой район", TITLE, "Россия"):
            with self.subTest(area=area):
                itinerary = route(area=area)
                self.assertTrue(places.verify_itinerary(itinerary, verified_records={QUERY: record()}))
                self.assert_cleared(itinerary)

    def test_address_locality_does_not_need_display_name_or_an_area_lookup(self):
        itinerary = route()
        self.assertEqual(places.verify_itinerary(
            itinerary, verified_records={QUERY: record(name=f"{TITLE}, Россия")}), [])
        self.assertTrue(itinerary.days[0].items[0].verified)

    def test_display_only_locality_is_not_a_provider_owned_area(self):
        itinerary = route()
        self.assertTrue(places.verify_itinerary(
            itinerary, verified_records={QUERY: record(address={"country_code": "ru"})}))
        self.assert_cleared(itinerary)

    def test_independent_record_requires_current_exact_catalog_attribution(self):
        place = independent_record()
        self.current_records = [deepcopy(place)]
        itinerary = route()
        self.assertEqual(places.verify_itinerary(itinerary, verified_records={QUERY: place}), [])
        item = itinerary.days[0].items[0]
        self.assertEqual(catalog.item_map_attribution(item),
                         {"provider": "2gis", "source_url": place["source_url"]})
        # A matching but unreviewed alias must not produce an unrenderable point.
        alias = f"{TITLE}, Тестовая область, Россия"
        itinerary = route(query=alias)
        self.assertTrue(places.verify_itinerary(itinerary, verified_records={alias: place}))
        self.assert_cleared(itinerary)

    def test_expired_or_modified_independent_evidence_is_rejected(self):
        place = independent_record()
        for trusted, changed in (
            ([], place),
            ([place], {**place, "source_url": "https://example.invalid/forged"}),
            ([place], {**place, "lat": 52.001}),
            ([place], {**place, "osm_type": "node", "osm_id": 123}),
        ):
            with self.subTest(trusted=bool(trusted), changed=changed):
                self.current_records = trusted
                itinerary = route()
                self.assertTrue(places.verify_itinerary(itinerary, verified_records={QUERY: changed}))
                self.assert_cleared(itinerary)

    def test_non_attractions_are_cleared_without_resolving_their_model_query(self):
        itinerary = route()
        item = itinerary.days[0].items[0]
        item.kind = "meal"
        self.assertEqual(places.verify_itinerary(itinerary, verified_records={}), [])
        self.assert_cleared(itinerary)

    def test_pairwise_day_distance_uses_the_same_default_and_explicit_limit(self):
        second_query = f"Галерея тестового города, {AREA}, Россия"
        itinerary = route()
        itinerary.days[0].items.append(ItineraryItem(
            time="15:00", title="Галерея тестового города", area=AREA,
            place_query=second_query, description="Осмотр"))
        facts = {QUERY: record(), second_query: record(name=second_query, lat=52.1, osm_id=124)}
        problems = places.verify_itinerary(itinerary, verified_records=facts)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("далеко друг от друга", problems[0])
        self.assertTrue(all(item.verified for item in itinerary.days[0].items))
        self.assertEqual(places.verify_itinerary(itinerary, 12, verified_records=facts), [])


if __name__ == "__main__":
    unittest.main()
