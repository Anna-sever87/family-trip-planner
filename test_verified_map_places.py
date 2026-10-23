"""Independent reviewed map evidence stays fresh, exact and outside OSM cache."""
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import places
import verified_map_places as catalog
from schemas import Itinerary, ItineraryDay, ItineraryItem


TODAY = date(2026, 10, 21)


class VerifiedMapPlacesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(catalog.CATALOG_FILE.read_text(encoding="utf-8"))

    def setUp(self):
        self.cache = {}
        self.patch(places, "_cache", new=self.cache)
        self.save = self.patch(places, "_save_cache", side_effect=AssertionError("OSM writes forbidden"))
        self.get = self.patch(places.httpx, "get", side_effect=AssertionError("HTTP forbidden"))
        self.patch(socket.socket, "connect", side_effect=AssertionError("Network forbidden"))
        self.patch(socket, "getaddrinfo", side_effect=AssertionError("DNS forbidden"))
        self.patch(sqlite3, "connect", side_effect=AssertionError("Database forbidden"))
        self.patch(catalog, "date", wraps=date).today.return_value = TODAY

    def patch(self, owner, name, **kwargs):
        patcher = patch.object(owner, name, **kwargs)
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def load(self, payload=None, *, today=TODAY, text=None):
        with tempfile.TemporaryDirectory(prefix="verified-map-test-") as folder:
            path = Path(folder) / "catalog.json"
            path.write_text(text if text is not None else json.dumps(
                self.payload if payload is None else payload, ensure_ascii=False
            ), encoding="utf-8")
            return catalog.load_verified_map_places(path=path, today=today)

    def one(self, **changes):
        row = deepcopy(self.payload["places"][0])
        row.update(changes)
        return {"version": 1, "places": [row]}

    def test_real_two_records_preserve_exact_names_addresses_and_provider(self):
        records = self.load()
        self.assertEqual(len(records), 2)
        self.assertEqual((records[0]["lat"], records[0]["lon"]), (51.653934, 103.726181))
        self.assertEqual((records[1]["lat"], records[1]["lon"]), (51.654642, 103.726667))
        self.assertEqual([row["address"]["house_number"] for row in records], ["20", "22"])
        for record in records:
            self.assertEqual((record["provider"], record["provenance"]), ("2gis", "official_linked_map"))
            self.assertEqual(record["type"], "museum")
            self.assertIsNone(places._osm_identity(record))
            self.assertTrue(places._place_matches_query(record["catalog_queries"][0], record))

    def test_freshness_is_inclusive_90_days_and_rejects_future(self):
        self.assertEqual(len(self.load(today=TODAY + timedelta(days=90))), 2)
        self.assertEqual(self.load(today=TODAY + timedelta(days=91)), [])
        self.assertEqual(self.load(today=TODAY - timedelta(days=1)), [])

    def test_deeply_nested_corrupt_catalogue_fails_closed(self):
        self.assertEqual(self.load(text="[" * 3000 + "0" + "]" * 3000), [])

    def test_rejects_invalid_checked_at_without_coercion(self):
        for value in (None, True, 20261021, "2026-02-30", "2026-10-21T00:00:00", "20261021"):
            with self.subTest(value=value):
                self.assertEqual(self.load(self.one(checked_at=value)), [])

    def test_rejects_boolean_nonfinite_string_and_out_of_range_coordinates(self):
        for field, values in (("lat", (True, False, "51.653934", None, float("nan"), float("inf"), 91, -91)),
                              ("lon", (True, "103.726181", float("-inf"), 181, -181))):
            for value in values:
                with self.subTest(field=field, value=value):
                    self.assertEqual(self.load(self.one(**{field: value})), [])

    def test_requires_complete_country_and_administrative_address(self):
        for field in ("city", "county", "state", "country", "country_code", "road", "house_number"):
            with self.subTest(field=field):
                payload = self.one()
                del payload["places"][0]["address"][field]
                self.assertEqual(self.load(payload), [])
        for code, country in (("us", "Россия"), ("ru", "США"), ("RU", "Россия")):
            payload = self.one()
            payload["places"][0]["address"].update(country=country, country_code=code)
            self.assertEqual(self.load(payload), [])

    def test_query_cannot_change_title_city_country_or_use_address_only(self):
        title = self.payload["places"][0]["title"]
        for query in (f"{title}, Иркутск, Россия", f"{title}, Слюдянка, США",
                      "Древности Байкала, Слюдянка, Россия", "улица Бабушкина, 20, Слюдянка, Россия",
                      f"{title}, Россия", f"{title}, , Россия"):
            with self.subTest(query=query):
                self.assertEqual(self.load(self.one(place_queries=[query])), [])

    def test_https_sources_and_independent_publisher_are_required(self):
        for field in ("source_url", "official_url", "official_link_url"):
            for url in ("http://example.org/", "https://", "https://user:secret@example.org/", "https://example.org/#x"):
                with self.subTest(field=field, url=url):
                    self.assertEqual(self.load(self.one(**{field: url})), [])
        self.assertEqual(self.load(self.one(official_url="https://2gis.ru/")), [])
        self.assertEqual(self.load(self.one(official_link_url="https://example.org/x")), [])

    def test_provider_id_matches_exact_organisation_url_and_is_not_osm(self):
        for changes in ({"provider": "osm"}, {"provider_id": True}, {"provider_id": 42},
                        {"provider_id": "0"}, {"provider_id": "42"}, {"type": "city"},
                        {"source_url": "https://2gis.ru/slyudyanka/search/museum"},
                        {"source_url": "https://2gis.ru.evil.example/slyudyanka/firm/70000001044533663"},
                        {"osm_id": 123}):
            with self.subTest(changes=changes):
                self.assertEqual(self.load(self.one(**changes)), [])

    def test_catalogue_limit_and_duplicate_id_query_fail_closed(self):
        self.assertEqual(self.load({"version": 1, "places": self.payload["places"] * 17}), [])
        self.assertEqual(self.load({"version": 1, "places": [self.payload["places"][0]] * 2}), [])
        self.assertEqual(self.load({"version": True, "places": []}), [])
        self.assertEqual(self.load({"version": 1, "places": {}, "unexpected": "field"}), [])

    def test_duplicate_json_keys_nonfinite_and_invalid_json_fail_closed(self):
        for raw in ('{"version":1,"version":1,"places":[]}', '{"version":1,"places":NaN}',
                    '[1,2]', '{broken', ' ' * (catalog.MAX_BYTES + 1)):
            with self.subTest(raw=raw[:40]):
                self.assertEqual(self.load(text=raw), [])

    def test_missing_file_is_empty_and_no_in_memory_expiry_bypass(self):
        with tempfile.TemporaryDirectory(prefix="verified-map-test-") as folder:
            self.assertEqual(catalog.load_verified_map_places(path=Path(folder) / "missing"), [])
        self.assertEqual(len(catalog.load_verified_map_places()), 2)
        with patch.object(catalog, "date", wraps=date) as clock:
            clock.today.return_value = TODAY + timedelta(days=91)
            self.assertEqual(catalog.load_verified_map_places(), [])

    def test_exact_geocode_ignores_osm_negative_without_http_or_cache_write(self):
        for raw in self.payload["places"]:
            query = raw["place_queries"][0]
            self.cache[query.lower() + "|miss"] = {"checked_at": places.time.time(), "reason": "mismatch"}
            before = deepcopy(self.cache)
            result = places.geocode(query)
            self.assertEqual(result["provider_id"], raw["provider_id"])
            self.assertEqual(self.cache, before)
        self.get.assert_not_called()
        self.save.assert_not_called()

    def test_exact_query_not_a_country_wide_or_address_lookup(self):
        title = self.payload["places"][0]["title"]
        for query in (title, f"{title}, Россия", f"{title}, Иркутск, Россия",
                      "улица Бабушкина, 20, Слюдянка, Россия", "Древности Байкала, Слюдянка, Россия"):
            self.assertIsNone(catalog.exact_verified_map_place(query))

    def test_returned_records_are_detached_from_catalogue(self):
        query = self.payload["places"][0]["place_queries"][0]
        first = places.geocode(query)
        first["address"]["city"] = "Иркутск"
        first["catalog_queries"].clear()
        self.assertEqual(places.geocode(query)["address"]["city"], "Слюдянка")

    def test_forged_provider_marker_or_changed_coordinates_not_trusted(self):
        trusted = catalog.load_verified_map_places()[0]
        query = trusted["catalog_queries"][0]
        for change in ({"lat": 52.0}, {"provider_id": "123"}, {"checked_at": "2026-10-22"},
                       {"osm_type": "node", "osm_id": 123}, {"geojson": None},
                       {"source_url": "https://example.org/"}):
            record = {**trusted, **change}
            self.assertFalse(places._place_matches_query(query, record))
        self.assertFalse(places._place_matches_query(query, {
            "lat": 51.653934, "lon": 103.726181, "name": query, "type": "museum",
            "provider": "2gis", "provenance": "official_linked_map",
        }))

    def test_matcher_enforces_country_city_and_name_for_independent_record(self):
        record = catalog.load_verified_map_places()[0]
        title = self.payload["places"][0]["title"]
        for query in (f"{title}, Иркутск, Россия", f"{title}, Слюдянка, США",
                      "Другой музей, Слюдянка, Россия"):
            self.assertFalse(places._place_matches_query(query, record))

    def test_nearby_uses_same_title_country_radius_and_caller_filter(self):
        raw = self.payload["places"][0]
        title, query = raw["title"], raw["place_queries"][0]
        result = places.geocode_nearby_named(title, "Россия", 51.65, 103.72)
        self.assertEqual(result["place_query"], query)
        self.assertEqual(result["locality"], "Слюдянка")
        self.assertEqual(result["provider"], "2gis")
        self.assertEqual(self.cache, {})
        self.get.assert_not_called()
        self.save.assert_not_called()
        for args in ((title, "США", 51.65, 103.72), (title, "Россия", 55, 100),
                     ("Древности Байкала", "Россия", 51.65, 103.72)):
            self.assertIsNone(places._cached_nearby_named(*args))
        self.assertIsNone(places._cached_nearby_named(title, "Россия", 51.65, 103.72,
                                                    place_filter=lambda _: False))

    def test_cached_nearby_does_not_allow_invalid_base_or_unbounded_radius(self):
        title = self.payload["places"][0]["title"]
        for lat, lon, radius in ((True, 103.72, 80), (float("nan"), 103.72, 80),
                                 (51.65, 181, 80), (51.65, 103.72, float("inf")),
                                 (51.65, 103.72, 81), (51.65, 103.72, 0)):
            self.assertIsNone(places._cached_nearby_named(title, "Россия", lat, lon, radius))

    def test_nearby_cannot_consume_2gis_copied_into_osm_cache(self):
        record = catalog.load_verified_map_places()[0]
        self.cache[record["catalog_queries"][0].lower()] = record
        with patch.object(catalog, "load_verified_map_places", return_value=[]):
            self.assertIsNone(places._cached_nearby_named(self.payload["places"][0]["title"], "Россия", 51.65, 103.72))

    def test_remember_alias_is_noop_even_for_independent_marker_only(self):
        places._remember_nearby_alias({"provider": "2gis"})
        places._remember_nearby_alias({"provenance": "official_linked_map"})
        self.assertEqual(self.cache, {})
        self.save.assert_not_called()

    def test_geometry_request_never_uses_point_catalogue_or_copied_osm_entry(self):
        query = self.payload["places"][0]["place_queries"][0]
        self.cache[query.lower() + "|geometry"] = catalog.load_verified_map_places()[0]
        with patch.object(places, "_search", return_value=None) as search:
            with self.assertRaises(places.GeocodeIncomplete):
                places.geocode(query, include_geometry=True, max_variants=1)
        self.assertTrue(search.call_args.kwargs["include_geometry"])
        self.save.assert_not_called()

    def test_expired_exact_evidence_cannot_survive_in_osm_cache(self):
        record = catalog.load_verified_map_places()[0]
        query = record["catalog_queries"][0]
        self.cache[query.lower()] = record
        with patch.object(catalog, "date", wraps=date) as clock, patch.object(places, "_search", return_value=None):
            clock.today.return_value = TODAY + timedelta(days=91)
            with self.assertRaises(places.GeocodeIncomplete):
                places.geocode(query, max_variants=1)
        self.save.assert_not_called()

    def test_final_itinerary_verification_reuses_independent_exact_place_offline(self):
        raw = self.payload["places"][0]
        item = ItineraryItem(time="10:00", title=raw["title"], kind="attraction", duration_min=60,
                             description="Знакомство с музейной экспозицией.",
                             cost=100, place_query=raw["place_queries"][0], area="Слюдянка")
        itinerary = Itinerary(title="Музей", summary="Проверка", days=[ItineraryDay(
            day=1, date="2026-10-28", theme="Музей", items=[item], total_cost=100,
        )], tips=[])
        self.assertEqual(places.verify_itinerary(itinerary), [])
        self.assertTrue(itinerary.days[0].items[0].verified)
        self.assertEqual(itinerary.days[0].items[0].lat, raw["lat"])
        self.get.assert_not_called()
        self.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
