"""Saved city pools remain request-bound and day edits preserve identities offline."""

from contextlib import ExitStack, nullcontext
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

import httpx
import city_candidates

from city_context import build_city_context, load_city_context, check_context_places
from city_deterministic import build_deterministic_city_itinerary
from city_edit import decode_city_edit, validate_city_edit
from test_city_deterministic import CENTER, candidates, request
from test_city_edit import context_for, wire_for


class LongCityContextRegressionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.blockers = [
            self.stack.enter_context(patch(name, side_effect=AssertionError("External IO forbidden")))
            for name in ("httpx.get", "httpx.post", "socket.socket.connect", "places.geocode",
                         "places._load_cache", "places._save_cache", "sqlite3.connect",
                         "planner.generate_json")
        ]

    def tearDown(self):
        for blocker in self.blockers:
            blocker.assert_not_called()

    def fixture(self, pool_size=42):
        req, pool = request(14, budget=500000), candidates(pool_size)
        # A previously saved partial route need not use all the verified reserve.
        built = build_deterministic_city_itinerary(req, pool[:4], resolved_city=CENTER)
        self.assertEqual(built.problems, [], built.coverage)
        self.assertIsNotNone(built.itinerary)
        return req, built.itinerary, pool, CENTER

    def test_category_provider_can_fill42_with_same_http_budget_and_strict_identity(self):
        from test_city_candidates import CENTER as map_center, record, request as map_request, response

        kinds = {"museums": ("tourism", "museum", "Музей"),
                 "zoos": ("tourism", "zoo", "Зоопарк"),
                 "theatre": ("amenity", "theatre", "Театр"),
                 "shopping_malls": ("shop", "mall", "Пассаж"),
                 "parks": ("leisure", "park", "Парк")}
        buckets = {}
        for group, (category, kind, label) in enumerate(kinds.values()):
            buckets[list(kinds)[group]] = [record(f"{label} Северный номер {i}", 100 * group + i + 1,
                category=category, type=kind) for i in range(10)]
        # Same title/location but another actual OSM identity is not a second
        # visit; a conflicting country must also be rejected in a rich pool.
        duplicate = deepcopy(buckets["museums"][0])
        duplicate["osm_id"] = 9999
        buckets["museums"][1] = duplicate
        buckets["museums"][2]["address"] = dict(city=map_center["name"], country="Китай", country_code="cn")
        calls = []

        def provider(url, **kwargs):
            calls.append((url, deepcopy(kwargs["params"])))
            if url == city_candidates._SHOPPING_DISCOVERY_URL:
                raise httpx.ConnectTimeout("Synthetic discovery outage")
            params = kwargs["params"]
            self.assertEqual(params["bounded"], 1)
            self.assertEqual(params["countrycodes"], "ru")
            self.assertIn("viewbox", params)
            self.assertEqual(params["limit"], 10)
            key = next(key for key, term in city_candidates._CATEGORY_TERMS.items()
                       if params["q"] == f"[{term}]")
            return response(deepcopy(buckets[key]))

        ai = Mock(return_value=SimpleNamespace(text='{"titles":[]}', input_tokens=0, output_tokens=0))
        req = map_request(14, children_ages=[], budget=500000, start_time="09:00", end_time="20:00",
                          interest_categories=list(kinds), wishes="", trip_goal="Познакомиться с городом")
        with patch("httpx.get", side_effect=provider), \
                patch("city_candidates.destination_name_hints", return_value=[]), \
                patch("places._cached_nearby_named", return_value=None), \
                patch("places._wait_for_turn", side_effect=lambda: nullcontext()):
            result = city_candidates.ground_city_candidates(req, map_center, completion=ai,
                                                           allow_partial_program=True)
        self.assertEqual(result.problems, [])
        self.assertEqual(len(result.candidates), 42)
        self.assertEqual(len({item.candidate_id for item in result.candidates}), 42)
        self.assertEqual(len({item.place_query for item in result.candidates}), 42)
        self.assertEqual((result.map_requests, len(calls)), (6, 6))
        self.assertLessEqual(result.map_requests, city_candidates.MAX_MAP_REQUESTS)
        self.assertEqual(result.diagnostics["name_limit"], 60)
        self.assertIn("duplicate_canonical_query", {item["reason"] for item in result.rejected})
        self.assertIn("country_mismatch_or_missing", result.diagnostics["detail_rejection_reasons"])
        self.assertTrue(all(item.record["address"]["country_code"] == "ru" for item in result.candidates))
        ai.assert_called_once()
        saved = build_city_context(req, result.candidates, map_center)
        loaded, _ = load_city_context(saved, req)
        self.assertEqual(len(loaded), 42)

    def test_version_one_24_record_long_snapshot_stays_loadable_without_migration(self):
        values = self.fixture(24)
        req, original, pool, center = values
        payload = build_city_context(req, pool, center)
        frozen = deepcopy(payload)
        loaded, resolved = load_city_context(payload, req)
        check_context_places(original, loaded)
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload, frozen)
        self.assertEqual([item.place_query for item in loaded], [item.place_query for item in pool])
        self.assertEqual(resolved["country_code"], "ru")
        context = context_for((req, original, loaded, resolved))
        wire = wire_for(context)
        wire["day_1"]["attractions"][0]["candidate_id"] = loaded[-1].candidate_id
        wire["day_1"]["attractions"][1]["time"] = "11:00"
        result = decode_city_edit(json.dumps(wire, ensure_ascii=False), context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])

    def test_last_of_42_can_replace_visit_when_edit_decoder_receives_over24_candidates(self):
        values = self.fixture()
        req, original, pool, center = values
        frozen = original.model_dump()
        context = context_for(values)
        self.assertGreater(len(context.allowed_candidates), 24)
        wire = wire_for(context)
        wire["day_1"]["attractions"][0]["candidate_id"] = pool[-1].candidate_id
        wire["day_1"]["attractions"][1]["time"] = "11:00"
        result = decode_city_edit(json.dumps(wire, ensure_ascii=False), context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        self.assertIn(pool[-1].place_query, {item.place_query for item in result.days[0].items})
        self.assertEqual([day.model_dump() for day in result.days[1:]], frozen["days"][1:])
        self.assertEqual(original.model_dump(), frozen)

    def test_long_edit_still_rejects_other_day_id_duplicate_id_and_forged_day(self):
        values = self.fixture()
        context = context_for(values)
        elsewhere = next(item.place_query for item in values[1].days[1].items if item.kind == "attraction")
        used_id = next(item.candidate_id for item in values[2] if item.place_query == elsewhere)
        for mode in ("other_day", "duplicate", "extra_day"):
            with self.subTest(mode=mode):
                wire = wire_for(context)
                if mode == "other_day":
                    wire["day_1"]["attractions"][0]["candidate_id"] = used_id
                elif mode == "duplicate":
                    wire["day_1"]["attractions"][1]["candidate_id"] = wire["day_1"]["attractions"][0]["candidate_id"]
                else:
                    wire["day_2"] = deepcopy(wire["day_1"])
                with self.assertRaises(ValueError):
                    decode_city_edit(json.dumps(wire, ensure_ascii=False), context)

    def test_long_snapshot_still_rejects_same_query_other_id_and_wrong_country(self):
        req, _, pool, center = self.fixture()
        for mode in ("query", "country"):
            with self.subTest(mode=mode):
                changed = deepcopy(pool)
                if mode == "query":
                    rec = deepcopy(pool[0].record)
                    rec["osm_id"] = 99999
                    changed[-1] = replace(pool[0], candidate_id="osm-node-99999", record=rec)
                else:
                    changed[-1].record["address"]["country_code"] = "cn"
                    changed[-1].record["address"]["country"] = "Китай"
                with self.assertRaises(ValueError):
                    build_city_context(req, changed, center)


if __name__ == "__main__":
    unittest.main()
