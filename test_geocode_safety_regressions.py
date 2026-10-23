"""Offline map safety regressions; shared clocks use temporary files only."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx

import candidate_grounding as grounding
from destination_scope import RegionScope
import places
import regional_skeleton
import rules
from schemas import Itinerary, ItineraryDay, ItineraryItem


class SharedGeocoderClockTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="geocoder-clock-offline-")
        self.addCleanup(temporary.cleanup)
        self.clock_file = Path(temporary.name) / "clock.db"
        for owner, name, options in (
            (places, "NOMINATIM_CLOCK_FILE", {"new": self.clock_file}),
            (places.httpx, "get", {"side_effect": AssertionError("No network")}),
            (socket.socket, "connect", {"side_effect": AssertionError("No network")}),
        ):
            patcher = patch.object(owner, name, **options)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_concurrent_threads_share_one_request_interval(self):
        barrier = threading.Barrier(3)

        def attempt(_):
            barrier.wait(timeout=5)
            with places._wait_for_turn():
                started = time.perf_counter()
                time.sleep(0.02)  # A synthetic HTTP attempt, while holding the lock.
                return started

        with ThreadPoolExecutor(max_workers=3) as pool:
            starts = sorted(pool.map(attempt, range(3)))
        self.assertTrue(all(b - a >= 1.0 for a, b in zip(starts, starts[1:])), starts)

    def test_two_processes_share_the_same_clock(self):
        child = """
from pathlib import Path
import socket, sys, time
from unittest.mock import patch
import places
places.NOMINATIM_CLOCK_FILE = Path(sys.argv[1])
with patch.object(socket.socket, 'connect', side_effect=AssertionError('No network')):
    print('ready', flush=True)
    sys.stdin.readline()
    with places._wait_for_turn():
        print(time.perf_counter(), flush=True)
"""
        processes = [subprocess.Popen(
            [sys.executable, "-X", "utf8", "-B", "-c", child, str(self.clock_file)],
            cwd=Path(places.__file__).resolve().parent,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8",
        ) for _ in range(2)]
        try:
            for process in processes:
                self.assertEqual(process.stdout.readline().strip(), "ready")
            for process in processes:
                process.stdin.write("go\n")
                process.stdin.flush()
            starts = []
            for process in processes:
                output, error = process.communicate(timeout=15)
                self.assertEqual(process.returncode, 0, error)
                starts.append(float(output.strip()))
            self.assertGreaterEqual(abs(starts[0] - starts[1]), places.MIN_INTERVAL - 0.02)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close()

    def test_failed_attempt_keeps_its_reservation(self):
        now = [10000.0]
        pauses = []

        def sleep(seconds):
            pauses.append(seconds)
            now[0] += seconds

        with patch.object(places.time, "time", side_effect=lambda: now[0]), patch.object(places.time, "sleep", side_effect=sleep):
            with self.assertRaises(httpx.ReadTimeout):
                with places._wait_for_turn():
                    raise httpx.ReadTimeout("Synthetic failure")
            with places._wait_for_turn():
                pass
        self.assertEqual(len(pauses), 1)
        self.assertAlmostEqual(pauses[0], places.MIN_INTERVAL)
        with closing(sqlite3.connect(self.clock_file)) as connection:
            self.assertAlmostEqual(connection.execute("SELECT requested FROM request_clock").fetchone()[0], now[0])

    def test_unavailable_clock_never_sends_an_unthrottled_request(self):
        with patch.object(places, "NOMINATIM_CLOCK_FILE", self.clock_file.parent), patch.object(places.httpx, "get") as get:
            with self.assertRaises(places.GeocodeUnavailable):
                places._search("Музей камня, Тестовый, Россия")
        get.assert_not_called()

    def test_search_uses_the_shared_clock_before_http(self):
        now, starts = [10000.0], []

        def sleep(seconds):
            now[0] += seconds

        def get(*args, **kwargs):
            starts.append(now[0])
            return httpx.Response(200, json=[], request=httpx.Request("GET", "https://example.invalid"))

        with patch.object(places.time, "time", side_effect=lambda: now[0]), patch.object(places.time, "sleep", side_effect=sleep), patch.object(places.httpx, "get", side_effect=get):
            self.assertIsNone(places._search("Музей камня, Тестовый, Россия"))
            self.assertIsNone(places._search("Музей живописи, Тестовый, Россия"))
        self.assertEqual(len(starts), 2)
        self.assertAlmostEqual(starts[1] - starts[0], places.MIN_INTERVAL)


class MapIdentityRegressionTests(unittest.TestCase):
    def setUp(self):
        self.address = {"village": "Тестовый", "county": "Тестовый район", "country": "Россия", "country_code": "ru"}
        self.scope = RegionScope("Тестовый край", "Россия", (50, 52, 100, 102), {
            "type": "Polygon", "coordinates": [[[100, 50], [102, 50], [102, 52], [100, 52], [100, 50]]],
        })
        self.bases = {"base_1": {"lat": 51.0, "lon": 101.0, "map_address": self.address}}
        self.cache = {}
        for owner, name, options in (
            (places, "_cache", {"new": self.cache}),
            (places, "_save_cache", {}),
            (places, "_wait_for_turn", {}),
            (places.httpx, "get", {"side_effect": AssertionError("HTTP must be mocked")}),
            (socket.socket, "connect", {"side_effect": AssertionError("No network")}),
        ):
            patcher = patch.object(owner, name, **options)
            patcher.start()
            self.addCleanup(patcher.stop)

    def record(self, title, osm_id=None, *, osm_type="node", lon=101.0):
        query = f"{title}, Тестовый, Россия"
        record = {"name": query, "namedetails": {"name": title}, "place_query": query,
                  "locality": "Тестовый", "address": dict(self.address), "lat": 51.01, "lon": lon,
                  "type": "museum", "category": "tourism"}
        if osm_id is not None:
            record.update(osm_id=osm_id, osm_type=osm_type)
        return record

    def confirm_pair(self, first, second):
        result = grounding.GroundingResult()
        first_fact, reason = grounding._confirmed_fact(first["namedetails"]["name"], first, self.bases, self.scope, result)
        self.assertEqual(reason, "")
        result.confirmed_hints.append(first_fact)
        return grounding._confirmed_fact(second["namedetails"]["name"], second, self.bases, self.scope, result)

    def test_distinct_osm_venues_can_share_coordinates(self):
        for osm_type, osm_id in (("node", 202), ("way", 101)):
            with self.subTest(osm_type=osm_type, osm_id=osm_id):
                fact, reason = self.confirm_pair(self.record("Музей камня", 101), self.record("Музей живописи", osm_id, osm_type=osm_type))
                self.assertIsNotNone(fact)
                self.assertEqual(reason, "")

    def test_same_osm_identity_is_still_a_duplicate_even_when_coordinates_differ(self):
        fact, reason = self.confirm_pair(self.record("Музей камня", 101), self.record("Другое имя музея", 101, lon=101.05))
        self.assertIsNone(fact)
        self.assertEqual(reason, "duplicate_place")

    def test_missing_identity_still_uses_proximity(self):
        fact, reason = self.confirm_pair(self.record("Музей камня"), self.record("Другое имя музея", 101))
        self.assertIsNone(fact)
        self.assertEqual(reason, "duplicate_place")

    def itinerary(self, records):
        return Itinerary(title="Проверка", summary="", days=[ItineraryDay(
            day=1, date="2099-06-01", theme="", items=[ItineraryItem(
                time="10:00", title=record["namedetails"]["name"], description="",
                place_query=record["place_query"], verified=True, lat=record["lat"], lon=record["lon"],
                provider="osm") for record in records])])

    def pool(self, records):
        result = grounding.GroundingResult()
        result.base_facts = [{"base_key": "base_1", "base_query": "Тестовый, Россия", "lat": 51.0, "lon": 101.0}]
        for record in records:
            fact, reason = grounding._confirmed_fact(record["namedetails"]["name"], record, self.bases, self.scope, grounding.GroundingResult())
            self.assertEqual(reason, "")
            result.confirmed_hints.append(fact)
        return result

    def test_distinct_ids_survive_candidate_skeleton_and_final_rule(self):
        records = [self.record("Музей камня", 101), self.record("Музей живописи", 202)]
        self.assertEqual(self.confirm_pair(*records)[1], "")
        self.cache.update({record["place_query"].lower(): record for record in records})
        _, hints = regional_skeleton._grounded_pools(self.pool(records))
        self.assertEqual(len({hint["_identity"] for hint in hints.values()}), 2)
        itinerary = self.itinerary(records)
        self.assertEqual(places.verify_itinerary(itinerary), [])
        self.assertTrue(all(item.verified for item in itinerary.days[0].items))
        self.assertEqual(rules.check_verified_place_duplicates(itinerary), [])
        self.assertEqual(len(places.cached_place_hints("", None, region_scope=self.scope)), 2)

    def test_same_id_aliases_remain_duplicates_in_every_stage_even_far_apart(self):
        records = [self.record("Музей камня", 101), self.record("Другое имя музея", 101, lon=101.05)]
        self.assertEqual(self.confirm_pair(*records)[1], "duplicate_place")
        self.cache.update({record["place_query"].lower(): record for record in records})
        _, hints = regional_skeleton._grounded_pools(self.pool(records))
        self.assertEqual(len({hint["_identity"] for hint in hints.values()}), 1)
        self.assertEqual(len(rules.check_verified_place_duplicates(self.itinerary(records))), 1)
        self.assertEqual(len(places.cached_place_hints("", None, region_scope=self.scope)), 1)

    def test_unknown_point_cannot_bridge_two_distinct_ids_in_skeleton(self):
        records = [self.record("Музей камня", 101), self.record("Старый музей"), self.record("Музей живописи", 202)]
        _, hints = regional_skeleton._grounded_pools(self.pool(records))
        self.assertNotEqual(hints[records[0]["place_query"]]["_identity"], hints[records[2]["place_query"]]["_identity"])

    def test_planner_history_does_not_confuse_distinct_ids_or_miss_distant_alias(self):
        import planner
        first = self.record("Музей камня", 101)
        second = self.record("Музей живописи", 202)
        alias = self.record("Другое имя музея", 101, lon=101.05)
        self.cache.update({record["place_query"].lower(): record for record in (first, second, alias)})
        items = self.itinerary([first, second, alias]).days[0].items
        self.assertFalse(planner._verified_record_matches_item(items[0].model_dump(), items[1]))
        self.assertTrue(planner._verified_record_matches_item(items[0].model_dump(), items[2]))

    def test_duplicate_repair_can_use_a_distinct_nearby_cached_place(self):
        import planner
        first = self.record("Музей камня", 101)
        second = self.record("Музей живописи", 202)
        self.cache.update({record["place_query"].lower(): record for record in (first, second)})
        itinerary = self.itinerary([first])
        itinerary.days.append(itinerary.days[0].model_copy(deep=True, update={"day": 2, "date": "2099-06-02"}))
        hints = [{"title": record["namedetails"]["name"], "place_query": record["place_query"],
                  "official_name": record["name"], "lat": record["lat"], "lon": record["lon"]}
                 for record in (first, second)]
        count = planner._prune_safe_exact_place_duplicates(itinerary, {}, map_hints=hints)
        self.assertEqual(count, 1)
        self.assertEqual({day.items[0].place_query for day in itinerary.days}, {first["place_query"], second["place_query"]})

    def test_untrusted_or_mismatching_cache_cannot_supply_distinct_identity(self):
        records = [self.record("Музей камня", 101), self.record("Музей живописи", 202)]
        itinerary = self.itinerary(records)
        for mutation in ("absent", "wrong_coordinates", "wrong_name", "unknown_provider", "independent_cache", "negative_cache"):
            with self.subTest(mutation=mutation):
                self.cache.clear()
                for record in records:
                    self.cache[record["place_query"].lower()] = dict(record)
                second = self.cache[records[1]["place_query"].lower()]
                if mutation == "absent": self.cache.clear()
                elif mutation == "wrong_coordinates": second["lon"] = 101.01
                elif mutation == "wrong_name": second.update(name="Другой музей, Тестовый, Россия", namedetails={})
                elif mutation == "unknown_provider": second["provider"] = "untrusted"
                elif mutation == "independent_cache": second["provider"] = "2gis"
                else: self.cache[records[1]["place_query"].lower() + "|miss"] = {"checked_at": time.time(), "reason": "not_found"}
                self.assertEqual(len(rules.check_verified_place_duplicates(itinerary)), 1)
        forged = {**records[0], "verified": True, "provider": "osm", "source_url": "https://www.openstreetmap.org/node/101"}
        self.cache.clear()
        self.assertIsNone(places.resolved_map_identity(forged))

    def test_explicit_house_needs_a_matching_house_number(self):
        query = "улица Ленина, 12, Тестовый, Россия"
        street = {"name": "улица Ленина, Тестовый, Россия", "lat": 51.0, "lon": 101.0,
                  "type": "residential", "address": {**self.address, "road": "улица Ленина"}}
        self.assertFalse(places._place_matches_query(query, street))
        for house, expected in (("12", True), ("14", False)):
            self.assertEqual(places._place_matches_query(query, {**street, "address": {**street["address"], "house_number": house}}), expected)

    def test_street_only_cached_and_api_results_do_not_mark_a_house_verified(self):
        query = "улица Ленина, 12, Тестовый, Россия"
        street = {"name": "улица Ленина, Тестовый, Россия", "lat": 51.0, "lon": 101.0,
                  "type": "residential", "address": {**self.address, "road": "улица Ленина"}}
        self.cache[query.lower()] = street
        item = ItineraryItem(time="10:00", title="Историческое здание", description="Посещение", place_query=query)
        itinerary = Itinerary(title="Проверка", summary="", days=[ItineraryDay(day=1, date="2099-06-01", theme="", items=[item])])
        response = httpx.Response(200, json=[{**street, "display_name": street["name"]}], request=httpx.Request("GET", "https://example.invalid"))
        with patch.object(places.httpx, "get", return_value=response):
            problems = places.verify_itinerary(itinerary)
        self.assertTrue(problems)
        self.assertFalse(item.verified)
        self.assertIsNone(item.lat)
        self.assertEqual(item.official_name, "")

    def test_non_attractions_lose_all_server_map_fields_without_lookup(self):
        for kind in ("meal", "rest", "travel"):
            with self.subTest(kind=kind):
                item = ItineraryItem(time="12:00", kind=kind, title="Перерыв", description="", verified=True,
                                     lat=51.0, lon=101.0, official_name="Старый музей", provider="2gis", source_url="https://example.org/old")
                itinerary = Itinerary(title="Проверка", summary="", days=[ItineraryDay(day=1, date="2099-06-01", theme="", items=[item])])
                with patch.object(places, "geocode", side_effect=AssertionError("No lookup for non-attractions")):
                    self.assertEqual(places.verify_itinerary(itinerary), [])
                self.assertEqual((item.verified, item.lat, item.lon, item.official_name, item.provider, item.source_url), (False, None, None, "", "", ""))


if __name__ == "__main__":
    unittest.main()
