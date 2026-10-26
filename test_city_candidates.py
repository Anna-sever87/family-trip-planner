"""Offline city-first pool: real identity rules, mocked completion/HTTP only."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import date
import json
import re
import socket
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

import city_candidates as candidates
_REAL_MAP_HINTS = candidates._map_name_hints
import destination_name_hints as name_hints
import places
import verified_map_places
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


TITLE = "Музей тестовых ремёсел"
AREA = "Тестовый город"
CENTER = {"lat": 52.0, "lon": 105.0, "name": AREA, "country_code": "RU", "admin1": "Тестовая область"}


def request(days=1, **changes):
    values = dict(destination=f"{AREA}, Россия", start_date="2099-10-01", end_date=f"2099-10-{days:02d}",
                  adults=2, children_ages=[5], budget=60_000, currency="RUB", interest_categories=["museums"])
    values.update(changes)
    return TripRequest(**values)


def record(title=TITLE, identifier=123, **changes):
    value = {
        "lat": "52.02", "lon": "105.03", "display_name": f"{title}, {AREA}, Россия",
        "namedetails": {"name": title}, "address": {"city": AREA, "country": "Россия", "country_code": "ru"},
        "category": "tourism", "type": "museum", "osm_type": "node", "osm_id": identifier,
    }
    value.update(changes)
    return value


def cached(raw):
    value = deepcopy(raw)
    value["name"] = value.pop("display_name")
    value["lat"], value["lon"] = float(value["lat"]), float(value["lon"])
    return value


def response(records):
    value = Mock()
    value.json.return_value = records
    value.content = b"[]"
    return value


class CityCandidateTests(unittest.TestCase):
    def setUp(self):
        self.cache, self.independent = {}, []
        self.patched(places, "_cache", new=self.cache)
        self.patched(verified_map_places, "load_verified_map_places", side_effect=lambda: deepcopy(self.independent))
        self.patched(places, "_save_cache", side_effect=AssertionError("No cache writes"))
        self.patched(places, "geocode", side_effect=AssertionError("No legacy lookup"))
        self.patched(places, "geocode_nearby_named", side_effect=AssertionError("No mutating nearby lookup"))
        self.patched(places, "_search", side_effect=AssertionError("No uncounted lookup"))
        self.patched(socket.socket, "connect", side_effect=AssertionError("No network"))
        self.patched(socket, "getaddrinfo", side_effect=AssertionError("No DNS"))
        self.patched(sqlite3, "connect", side_effect=AssertionError("No DB"))
        self.gate = self.patched(places, "_wait_for_turn", side_effect=lambda: nullcontext())
        self.get = self.patched(httpx, "get", return_value=response([record()]))
        self.hints = self.patched(candidates, "destination_name_hints", return_value=[])
        # These tests isolate exact-name lookup. Category discovery has its own
        # provider fixtures; enable it explicitly in the source-identity cases.
        self.map_hints = self.patched(candidates, "_map_name_hints", return_value=[])
        self.category_search = self.patched(candidates, "_category_search", return_value=[])
        self.completion = Mock(return_value=SimpleNamespace(text=json.dumps({"titles": [TITLE]}), input_tokens=11, output_tokens=17))

    def patched(self, owner, name, **kwargs):
        handle = patch.object(owner, name, **kwargs)
        self.addCleanup(handle.stop)
        return handle.start()

    def run_pool(self, titles=None, *, days=1, center=None, **changes):
        if titles is not None:
            self.completion.return_value.text = json.dumps({"titles": titles}, ensure_ascii=False)
        return candidates.ground_city_candidates(request(days, **changes), CENTER if center is None else center,
                                                 completion=self.completion)

    def foreign_pool(self, titles, *, code="VN", days=1, provider_title=None, locality="Test City"):
        center = {**CENTER, "name": "Test City", "country_code": code}
        title = provider_title or titles[0]
        self.get.return_value = response([record(title=title, display_name=f"{title}, {locality}, {code}",
            address={"city": locality, "country": code, "country_code": code.lower()})])
        return self.run_pool(titles, days=days, center=center, destination=f"Test City, {code}")

    def test_vn_tr_native_names_share_general_long_trip_budgets(self):
        for code, language in (("VN", "vi,en"), ("TR", "tr,en")):
            with self.subTest(code=code):
                self.foreign_pool([], code=code, days=14, provider_title="Test Museum")
                call = self.completion.call_args
                item_schema = call.kwargs["json_schema"]["properties"]["titles"]
                self.assertEqual(item_schema["maxItems"], 60)
                self.assertEqual(item_schema["items"]["pattern"], candidates._NON_CYRILLIC_PATTERN)
                system = call.args[0][0]["content"]
                context = json.loads(call.args[0][1]["content"])
                self.assertIn("не больше 60 названий", system)
                self.assertIn("не делай самодельную транслитерацию", system)
                self.assertIn("не предполагай", system.casefold())
                self.assertIn("пустой список", system)
                self.assertEqual((context["max_titles"], context["map_languages"]), (60, language))
                self.assertEqual(context["usable_candidate_limit"], 42)
        self.get.assert_not_called()

    def test_fourteen_day_vn_tr_can_confirm_thirty_six_native_places_with_one_ai_call(self):
        for code, prefix in (("VN", "Bảo tàng Thử Nghiệm"), ("TR", "İlçe Şehir Müzesi")):
            with self.subTest(country=code):
                self.get.reset_mock()
                self.completion.reset_mock()
                titles = [f"{prefix} {number}" for number in range(36)]
                def lookup(_url, **kwargs):
                    title = kwargs["params"]["q"].split(",", 1)[0]
                    return response([record(title=title, identifier=titles.index(title) + 1000,
                        display_name=f"{title}, Test City, {code}",
                        address={"city": "Test City", "country": code, "country_code": code.lower()})])
                self.get.side_effect = lookup
                result = self.run_pool(titles, days=14,
                    center={**CENTER, "name": "Test City", "country_code": code}, destination=f"Test City, {code}",
                    children_ages=[14], start_time="09:00", end_time="18:00", pace="средний")
                self.assertEqual([item.title for item in result.candidates], titles)
                self.assertEqual(result.status, "success")
                self.assertEqual((result.diagnostics["name_limit"], result.diagnostics["usable_limit"]), (60, 42))
                self.assertEqual(result.diagnostics["attempted_names"], 36)
                self.assertEqual(result.diagnostics["required_planning_minutes"], 3780)
                self.assertEqual(result.diagnostics["pool_quality"], "limited")
                self.assertEqual((result.map_requests, self.get.call_count), (36, 36))
                self.assertEqual(result.usage["completion_calls"], 1)
                self.completion.assert_called_once()

    def test_unicode_local_and_other_original_names_are_not_ascii_restricted(self):
        for code, title in (("VN", "Bảo tàng Thử Nghiệm"), ("VN", "Đường Ánh Sáng"),
                            ("TR", "İlçe Şehir Müzesi"), ("TR", "Işık Doğa Müzesi"),
                            ("VN", "展館數字 12")):
            with self.subTest(code=code, title=title):
                result = self.foreign_pool([title], code=code)
                self.assertEqual(result.problems, [])
                self.assertEqual(result.candidates[0].title, title)
                pattern = self.completion.call_args.kwargs["json_schema"]["properties"]["titles"]["items"]["pattern"]
                self.assertIsNotNone(re.fullmatch(pattern, title))
                params = self.get.call_args.kwargs["params"]
                self.assertEqual(params["accept-language"], "vi,en" if code == "VN" else "tr,en")
                self.assertEqual(params["countrycodes"], code.lower())

    def test_translated_or_mixed_russian_proposals_are_rejected_without_losing_local_name(self):
        title = "Bảo tàng Thử Nghiệm"
        for code in ("VN", "TR"):
            self.get.reset_mock()
            result = self.foreign_pool(["Музей океанографии", title, "Парк Testland"], code=code,
                                       provider_title=title)
            with self.subTest(code=code):
                self.assertEqual(result.problems, [])
                self.assertEqual([item.title for item in result.candidates], [title])
                self.assertEqual((result.map_requests, self.get.call_count), (1, 1))
                self.assertEqual([item["reason"] for item in result.rejected],
                                 ["nonlocal_cyrillic_proposal", "nonlocal_cyrillic_proposal"])
                pattern = self.completion.call_args.kwargs["json_schema"]["properties"]["titles"]["items"]["pattern"]
                self.assertIsNone(re.fullmatch(pattern, "Парк Testland"))

    def test_vn_russian_only_does_not_trigger_transliteration_or_alternative_search(self):
        result = self.foreign_pool(["Музей океанографии Нячанга", "Зоопарк в Нячанге"])
        self.assertTrue(result.problems)
        self.assertEqual(result.candidates, [])
        self.assertEqual((result.usage["completion_calls"], result.map_requests), (1, 0))
        self.get.assert_not_called()

    def test_local_policy_does_not_change_other_countries_or_force_all_foreign_names_to_latin(self):
        for code in ("RU", "BY", "KZ", "KG"):
            with self.subTest(code=code):
                result = self.foreign_pool([TITLE], code=code, days=7)
                self.assertEqual(len(result.candidates), 1)
                schema = self.completion.call_args.kwargs["json_schema"]["properties"]["titles"]
                self.assertEqual(schema["maxItems"], 36)
                self.assertNotIn("pattern", schema["items"])
                self.assertEqual(self.get.call_args.kwargs["params"]["accept-language"], "ru")

    def test_local_names_reject_excess_before_map_and_share_36_send_limit(self):
        center = {**CENTER, "name": "Test City", "country_code": "VN"}
        self.get.side_effect = httpx.ConnectTimeout("offline")
        result = self.run_pool([f"Test Museum {number}" for number in range(37)], days=7,
                               center=center, destination="Test City, VN")
        self.assertTrue(result.problems)
        self.assertEqual(result.map_requests, 0)
        self.get.assert_not_called()
        result = self.run_pool([f"Test Museum {number}" for number in range(36)], days=7,
                               center=center, destination="Test City, VN")
        self.assertEqual((result.map_requests, self.get.call_count, self.gate.call_count), (36, 36, 36))

    def test_single_leading_period_is_proposal_only_normalized_and_deduplicated(self):
        title = "Pagoda Testland"
        result = self.foreign_pool([f".{title}", title], provider_title=title)
        self.assertEqual(result.problems, [])
        self.assertEqual(result.map_requests, 1)
        self.assertEqual(self.get.call_args.kwargs["params"]["q"], f"{title}, VN")
        self.assertEqual(result.normalized_proposals, [
            {"original_title": f".{title}", "title": title, "reason": "single_leading_period"}])
        diagnostic = asdict(result)
        self.assertEqual(diagnostic["normalized_proposals"], result.normalized_proposals)
        json.dumps(diagnostic, ensure_ascii=False, allow_nan=False)

    def test_single_period_with_space_is_allowed_but_ellipsis_is_not_repeatedly_stripped(self):
        result = self.foreign_pool([". Test Museum"], provider_title="Test Museum")
        self.assertEqual(result.candidates[0].title, "Test Museum")
        self.get.reset_mock()
        result = self.foreign_pool(["..Test Museum", ". .Test Museum", "."], provider_title="Test Museum")
        self.assertEqual(result.map_requests, 0)
        self.assertEqual(result.normalized_proposals, [])
        self.assertTrue(all(item["reason"] == "unsupported_leading_punctuation" for item in result.rejected))
        self.get.assert_not_called()

    def test_provider_canonical_name_and_actual_locality_are_never_rewritten(self):
        title = ".Test Museum"
        locality = "Phường Thử Nghiệm"
        result = self.foreign_pool([title], provider_title=title, locality=locality)
        self.assertEqual(result.problems, [])
        item = result.candidates[0]
        self.assertEqual(item.title, title)
        self.assertEqual(item.area, locality)
        self.assertEqual(item.place_query, f"{title}, {locality}, VN")
        self.assertEqual(item.record["name"], f"{title}, {locality}, VN")
        self.assertEqual(self.get.call_args.kwargs["params"]["q"], "Test Museum, VN")

    def test_local_spelling_does_not_allow_mismatching_other_entity_or_country(self):
        center = {**CENTER, "name": "Test City", "country_code": "vn"}
        for provider_name, country in (("Entirely Different Museum", "vn"), ("Test Museum", "tr")):
            self.get.reset_mock()
            self.get.return_value = response([record(title=provider_name,
                display_name=f"{provider_name}, Test City, {country.upper()}",
                address={"city": "Test City", "country": country.upper(), "country_code": country})])
            result = candidates.CityCandidateResult()
            confirmed = candidates._lookup("Test Museum", "VN", center, result)
            with self.subTest(provider_name=provider_name, country=country):
                # An unrelated provider name never proves the requested alias.
                self.assertIsNone(confirmed)
                self.assertEqual((result.map_requests, self.get.call_count), (1, 1))

    def test_descriptive_zoo_is_not_a_named_candidate_but_specific_names_survive(self):
        result = self.foreign_pool(["Zoo in Test City"], provider_title="Zoo in Test City")
        self.assertEqual(result.map_requests, 0)
        self.assertEqual(result.rejected[0]["reason"], "generic_name_or_non_visit")
        self.get.assert_not_called()
        for title in ("Testland Zoo", "Museum X", "St. Test Museum"):
            with self.subTest(title=title):
                result = self.foreign_pool([title])
                self.assertEqual(result.problems, [])
                self.assertEqual(result.candidates[0].title, title)
                self.assertEqual(result.normalized_proposals, [])

    def test_source_names_use_exact_alias_only_and_do_not_match_unknown_city(self):
        self.hints.side_effect = name_hints.destination_name_hints
        catalogue = {"version": 1, "regions": [{"aliases": [f"{AREA}, Россия"], "places": [{
            "title": TITLE, "place_query": f"{TITLE}, {AREA}, Россия", "locality": AREA,
            "source_url": "https://example.invalid/official-name",
        }]}]}
        source_file = Mock()
        source_file.read_text.return_value = json.dumps(catalogue)
        with patch.object(name_hints, "NAME_HINTS_FILE", source_file):
            for destination, city, expected in ((f"{AREA}, Россия", AREA, 1),
                                               (f"{AREA.lower()}, россия", AREA, 1),
                                               (f"Другой {AREA}, Россия", f"Другой {AREA}", 0)):
                self.get.reset_mock()
                result = self.run_pool([], destination=destination, center={**CENTER, "name": city})
                context = json.loads(self.completion.call_args.args[0][1]["content"])
                with self.subTest(destination=destination):
                    self.assertEqual(len(context["source_name_hints"]), expected)
                    self.assertEqual(self.get.call_count, expected)
                    self.assertEqual(len(result.candidates), expected)
                    self.hints.assert_called_with(destination, limit=12)

    def test_source_names_are_seed_first_deduplicated_and_still_require_map(self):
        seed = {"title": TITLE, "source_url": "https://example.invalid/official-name",
                "lat": 99, "indoor": True, "cost": 0, "place_query": "Do not copy this query"}
        self.hints.return_value = [seed, dict(seed)]
        other = "Музей иных ремёсел"
        self.get.side_effect = [response([record()]), response([record(other, identifier=456)])]
        result = self.run_pool([f".{TITLE}", other])
        self.assertEqual([item.title for item in result.candidates], [TITLE, other])
        self.assertEqual([call.kwargs["params"]["q"] for call in self.get.call_args_list],
                         [f"{TITLE}, Россия", f"{other}, Россия"])
        context = json.loads(self.completion.call_args.args[0][1]["content"])
        self.assertTrue(all(set(hint) == {"title", "source_url"} for hint in context["source_name_hints"]))
        self.assertIn("не VisitFacts", self.completion.call_args.args[0][0]["content"])
        self.assertIn("не копируй их в titles", self.completion.call_args.args[0][0]["content"])
        self.assertIn("дополнительные", self.completion.call_args.args[0][0]["content"])
        for item in result.candidates:
            self.assertNotIn("indoor", item.record)
            self.assertNotIn("cost", item.record)
            self.assertNotIn("source_url", item.record)
        self.get.side_effect = None
        self.get.return_value = response([])
        self.assertFalse(self.run_pool([]).candidates)

    def test_source_names_do_not_bypass_country_radius_type_or_identity(self):
        self.hints.return_value = [{"title": TITLE, "source_url": "https://example.invalid/name"}]
        for raw in (record(title="Другой музей"), record(lat="55"), record(type="hotel"),
                    record(address={"city": AREA, "country": "TR", "country_code": "tr"})):
            with self.subTest(raw=raw):
                self.get.return_value = response([raw])
                self.assertFalse(self.run_pool([]).candidates)
        self.hints.return_value = [{"title": "Городской парк", "source_url": "https://example.invalid/name"}]
        self.get.reset_mock()
        self.assertFalse(self.run_pool([]).candidates)
        self.assertEqual(self.get.call_count, 1)  # No source-name bypass of the ordinary map check.

    def test_seed_plus_ai_raw_names_share_unchanged_http_cap(self):
        for days, code, limit in ((1, "RU", 12), (1, "VN", 12), (1, "TR", 12),
                                  (7, "RU", 36), (7, "VN", 36), (7, "TR", 36)):
            with self.subTest(days=days, code=code):
                self.get.reset_mock()
                self.gate.reset_mock()
                self.get.side_effect = httpx.ConnectTimeout("offline")
                seeds = [f"Seed Museum {number}" for number in range(3)]
                self.hints.return_value = [{"title": title, "source_url": "https://example.invalid/name"}
                                          for title in seeds]
                ai_names = [seeds[0]] + [f"Model Museum {number}" for number in range(limit - 1)]
                center = {**CENTER, "name": "Test City", "country_code": code}
                result = self.run_pool(ai_names, days=days, center=center, destination=f"Test City, {code}")
                sends = min(36, 2 * limit)
                self.assertEqual(result.map_requests, sends)
                self.assertEqual(self.get.call_count, sends)
                self.assertEqual(self.gate.call_count, sends)
                queries = [call.kwargs["params"]["q"] for call in self.get.call_args_list[::2]]
                self.assertEqual(queries[:3], [f"{title}, {code}" for title in seeds])
                self.assertEqual(len(set(queries)), sends // 2)
                self.hints.assert_called_with(f"Test City, {code}", limit=limit)

    def test_peak_record_cannot_become_city_visit_from_fresh_or_cached_map(self):
        title = "Test Mountain"
        peak = record(title, category="natural", type="peak")
        self.assertTrue(candidates._eligible_poi_type(cached(peak)))  # Regional policy is unchanged.
        for use_cache in (False, True):
            with self.subTest(use_cache=use_cache):
                self.cache.clear()
                if use_cache:
                    self.cache[f"{title}, {AREA}, Россия".lower()] = cached(peak)
                    self.cache[next(iter(self.cache))].update(cost=0, indoor=False, duration_min=180,
                                                              access="yes", cableway="invented")
                before = deepcopy(self.cache)
                self.get.return_value = response([peak])
                result = self.run_pool([title])
                self.assertFalse(result.candidates)
                self.assertTrue(result.problems)
                self.assertEqual(result.map_requests, 1)
                self.assertEqual(self.cache, before)
        system = self.completion.call_args.args[0][0]["content"]
        self.assertIn("Координата вершины/острова не доказывает доступ", system)

    def test_peak_guard_does_not_infer_type_from_name_or_relabel_museum(self):
        title = "Test Mountain Museum"
        self.get.return_value = response([record(title, type="museum")])
        result = self.run_pool([title])
        self.assertEqual(result.problems, [])
        self.assertEqual(result.candidates[0].record["type"], "museum")
        self.assertEqual(result.candidates[0].title, title)

    def test_one_completion_named_bounded_search_and_canonical_record(self):
        result = self.run_pool()
        self.assertEqual(result.problems, [])
        self.assertEqual(result.usage, {"completion_calls": 1, "input_tokens": 11, "output_tokens": 17})
        self.completion.assert_called_once()
        self.assertEqual(result.map_requests, 1)
        self.assertEqual(self.gate.call_count, 1)
        params = self.get.call_args.kwargs["params"]
        self.assertEqual((params["q"], params["countrycodes"], params["bounded"]), (f"{TITLE}, Россия", "ru", 1))
        west, north, east, south = map(float, params["viewbox"].split(","))
        self.assertLess(west, CENTER["lon"])
        self.assertGreater(east, CENTER["lon"])
        self.assertAlmostEqual(places.distance_km(CENTER["lat"], CENTER["lon"], north, CENTER["lon"]), 40, places=6)
        item = result.candidates[0]
        self.assertEqual((item.candidate_id, item.title, item.area), ("osm-node-123", TITLE, AREA))
        self.assertEqual(item.place_query, f"{TITLE}, {AREA}, Россия")
        self.assertIs(type(item.record["lat"]), float)
        self.assertNotIn("indoor", item.record)
        self.assertEqual(self.cache, {})

    def shopping_response(self, elements, **changes):
        payload = {"elements": elements, **changes}
        return SimpleNamespace(raise_for_status=lambda: None,
                               content=json.dumps(payload).encode(), json=lambda: deepcopy(payload))

    def mall_element(self, title="Северный пассаж", identifier=123, **changes):
        return {"type": "node", "id": identifier, "lat": 52.02, "lon": 105.03,
                "tags": {"shop": "mall", "name": title, "opening_hours": "24/7", "fee": "no"}, **changes}

    def test_shopping_checkbox_discovers_source_names_then_independently_checks_same_map_id(self):
        self.map_hints.side_effect = _REAL_MAP_HINTS
        title = "Северный пассаж"
        observed = record(title, category="shop", type="mall")
        self.get.side_effect = [self.shopping_response([self.mall_element(title)]), response([observed])]
        result = self.run_pool([], interest_categories=["main_sights", "museums", "theatre", "shopping_malls"], wishes="")
        self.assertEqual([item.title for item in result.candidates], [title])
        self.assertEqual((result.map_requests, self.get.call_count, self.gate.call_count), (2, 2, 1))
        discovery = self.get.call_args_list[0]
        self.assertEqual(discovery.args[0], candidates._SHOPPING_DISCOVERY_URL)
        query = discovery.kwargs["params"]["data"]
        self.assertIn('["shop"~"^(mall|shopping_centre)$"]', query)
        self.assertIn("around:40000,52.0000000,105.0000000", query)
        self.assertIn("out body center 24", query)
        self.assertEqual(self.get.call_args_list[1].kwargs["params"]["osm_ids"], "N123")
        self.assertEqual(self.get.call_args_list[1].args[0], candidates._MAP_LOOKUP_URL)
        self.assertNotIn("opening_hours", str(result.candidates[0].record))
        self.assertNotIn("fee", str(result.candidates[0].record))
        context = json.loads(self.completion.call_args.args[0][1]["content"])
        self.assertEqual(context["shopping_preference"], "requested")
        self.assertEqual(context["shopping_names_already_checked"], [title])

    def test_shopping_discovery_is_not_used_without_interest_or_with_explicit_exclusion(self):
        for interests, wishes in ((["museums"], ""), (["shopping_malls"], "Без торговых центров"),
                                  (["shopping_malls"], "No malls")):
            with self.subTest(wishes=wishes):
                self.get.reset_mock()
                result = self.run_pool([TITLE], interest_categories=interests, wishes=wishes)
                self.assertEqual(result.map_requests, 1)
                self.assertEqual(self.get.call_args.args[0], places.NOMINATIM_URL)

    def test_shopping_discovery_does_not_accept_other_identity_type_country_or_remote_object(self):
        self.map_hints.side_effect = _REAL_MAP_HINTS
        title = "Северный пассаж"
        base = record(title, category="shop", type="mall")
        bad = [dict(base, osm_id=999), dict(base, category="tourism", type="museum"),
               dict(base, address={"city": AREA, "country_code": "fr", "country": "France"}),
               dict(base, lat="53.5")]
        for observation in bad:
            with self.subTest(observation=observation):
                self.get.side_effect = [self.shopping_response([self.mall_element(title)]), response([observation])]
                result = self.run_pool([], interest_categories=["shopping_malls"])
                self.assertFalse(result.candidates)
                self.assertEqual(result.map_requests, 2)

    def test_shopping_discovery_accepts_way_center_and_keeps_shopping_centre_type(self):
        self.map_hints.side_effect = _REAL_MAP_HINTS
        title = "Северный пассаж"
        element = self.mall_element(title, type="way", center={"lat": 52.02, "lon": 105.03},
                                    tags={"shop": "shopping_centre", "name": title})
        self.get.side_effect = [self.shopping_response([element]), response([
            record(title, category="shop", type="shopping_centre", osm_type="way")])]
        result = self.run_pool([], interest_categories=["shopping_malls"])
        self.assertEqual(result.candidates[0].record["type"], "shopping_centre")
        self.assertEqual(result.candidates[0].candidate_id, "osm-way-123")

    def test_shopping_discovery_failures_are_bounded_and_do_not_claim_no_malls(self):
        self.map_hints.side_effect = _REAL_MAP_HINTS
        for discovery in (httpx.ConnectTimeout("offline"), self.shopping_response([], remark="timeout"),
                          self.shopping_response([self.mall_element()] * 65)):
            with self.subTest(discovery=type(discovery).__name__):
                self.get.side_effect = [discovery, response([record()])]
                result = self.run_pool([TITLE], interest_categories=["shopping_malls"])
                self.assertEqual((result.map_requests, len(result.candidates)), (2, 1))
                self.assertTrue(result.warnings)
                self.assertEqual(result.usage["completion_calls"], 1)

    def test_shopping_discovery_skips_generic_remote_and_malformed_source_items(self):
        self.map_hints.side_effect = _REAL_MAP_HINTS
        bad = [self.mall_element(title="Торговый центр"), self.mall_element(lat=True),
               self.mall_element(identifier=True), self.mall_element(lat=60.0),
               self.mall_element(tags={"shop": "supermarket", "name": "Северный магазин"}),
               self.mall_element(type="way", center={"lat": float("nan"), "lon": 105.03}),
               {"tags": "mall"}, None]
        self.get.side_effect = [self.shopping_response(bad)]
        result = self.run_pool([], interest_categories=["shopping_malls"])
        self.assertFalse(result.candidates)
        self.assertEqual(result.map_requests, 1)
        self.assertTrue(result.warnings)

    def test_leading_period_in_source_name_is_not_ai_normalized_or_used_as_dictionary_key(self):
        self.map_hints.side_effect = _REAL_MAP_HINTS
        title = ". Северный пассаж"
        self.get.side_effect = [self.shopping_response([self.mall_element(title)])]
        result = self.run_pool([], interest_categories=["shopping_malls"])
        self.assertEqual(result.map_requests, 1)
        self.assertFalse(result.candidates)
        self.assertFalse(result.normalized_proposals)
        self.assertIn({"title": title, "reason": "unsupported_leading_punctuation"}, result.rejected)

    def test_generic_source_name_with_period_is_not_silently_rewritten(self):
        self.hints.return_value = [{"title": ". " + TITLE, "source_url": "https://example.invalid/name"}]
        result = self.run_pool([])
        self.assertEqual(result.map_requests, 0)
        self.assertFalse(result.normalized_proposals)
        self.get.assert_not_called()

    def test_shopping_lookup_does_not_reuse_unrelated_cached_object(self):
        self.map_hints.side_effect = _REAL_MAP_HINTS
        title = "Северный пассаж"
        self.cache[title] = cached(record(title, identifier=999, category="shop", type="mall"))
        self.get.side_effect = [self.shopping_response([self.mall_element(title)]),
                               response([record(title, category="shop", type="mall")])]
        result = self.run_pool([], interest_categories=["shopping_malls"])
        self.assertEqual(result.candidates[0].candidate_id, "osm-node-123")
        self.assertEqual(result.map_requests, 2)

    def test_all_mall_candidates_do_not_expand_usable_pool_or_call_ai_when_full(self):
        self.map_hints.side_effect = _REAL_MAP_HINTS
        elements = [self.mall_element(f"Северный пассаж {i}", i + 1) for i in range(4)]
        seeds = [f"Test Museum {i}" for i in range(10)]
        self.hints.return_value = [{"title": title, "source_url": "https://example.invalid/name"} for title in seeds]
        def get(url, **kwargs):
            if url == candidates._SHOPPING_DISCOVERY_URL:
                return self.shopping_response(elements)
            if url == candidates._MAP_LOOKUP_URL:
                identifier = int(kwargs["params"]["osm_ids"][1:])
                return response([record(f"Северный пассаж {identifier - 1}", identifier, category="shop", type="mall")])
            title = kwargs["params"]["q"].split(",", 1)[0]
            if title.startswith("Северный пассаж"):
                return response([record(title, int(title.rsplit(" ", 1)[1]) + 1, category="shop", type="mall")])
            return response([record(title, seeds.index(title) + 10)])
        self.get.side_effect = get
        result = self.run_pool([], interest_categories=["shopping_malls"])
        self.assertEqual((len(result.candidates), result.map_requests), (6, 7))
        self.assertEqual(result.usage["completion_calls"], 0)
        self.completion.assert_not_called()

    def test_shopping_discovery_limits_source_names_and_counts_send_in_total_36(self):
        self.map_hints.side_effect = _REAL_MAP_HINTS
        elements = [self.mall_element(f"Северный пассаж {i}", i + 1) for i in range(8)]
        calls = 0
        def get(url, **kwargs):
            nonlocal calls
            calls += 1
            if url == candidates._SHOPPING_DISCOVERY_URL:
                return self.shopping_response(elements)
            raise httpx.ConnectTimeout("offline")
        self.get.side_effect = get
        result = self.run_pool([f"Museum proposal {i}" for i in range(36)], days=7,
                               interest_categories=["shopping_malls"])
        self.assertEqual((calls, result.map_requests), (36, 36))
        self.assertEqual(self.gate.call_count, 35)
        mall_queries = {call.kwargs["params"]["osm_ids"] for call in self.get.call_args_list[1:]
                        if "osm_ids" in call.kwargs["params"]}
        # Only one source per category precedes AI. Remaining exact-ID work
        # shares the same budget rather than exhausting it before AI runs.
        self.assertEqual(len(mall_queries), 4)
        checked = json.loads(self.completion.call_args.args[0][1]["content"])
        self.assertEqual(checked["remaining_name_slots"], 35)
        self.completion.assert_called_once()

    def test_rejected_raw_names_do_not_consume_usable_pool_capacity(self):
        titles = [f"Test Museum {number}" for number in range(28)]
        def get(url, **kwargs):
            title = kwargs["params"]["q"].split(",", 1)[0]
            index = titles.index(title)
            return response([] if index < 10 else [record(title, index + 1)])
        self.get.side_effect = get
        result = self.run_pool(titles, days=5)
        self.assertEqual((len(result.candidates), result.map_requests), (18, 28))
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.rejected), 10)
        self.completion.assert_called_once()

    def test_usable_pool_cap_is_separate_from_larger_raw_budget(self):
        titles = [f"Test Museum {number}" for number in range(28)]
        def get(url, **kwargs):
            title = kwargs["params"]["q"].split(",", 1)[0]
            return response([record(title, titles.index(title) + 1)])
        self.get.side_effect = get
        result = self.run_pool(titles, days=5)
        self.assertEqual((len(result.candidates), result.map_requests), (18, 18))
        self.assertEqual(self.completion.call_args.kwargs["json_schema"]["properties"]["titles"]["maxItems"], 28)

    def test_large_unconfirmed_source_list_reserves_one_successful_ai_supplement(self):
        seeds = [f"Source Museum {index}" for index in range(28)]
        self.hints.return_value = [{"title": title, "source_url": "https://example.invalid/name"} for title in seeds]
        supplement = "Supplement Museum"
        self.completion.return_value.text = json.dumps({"titles": [supplement]})
        def get(url, **kwargs):
            title = kwargs["params"]["q"].split(",", 1)[0]
            return response([record(title)] if title == supplement else [])
        self.get.side_effect = get
        result = candidates.ground_city_candidates(request(5), CENTER, completion=self.completion,
                                                   allow_partial_program=True)
        self.assertEqual((result.status, [item.title for item in result.candidates]), ("success", [supplement]))
        self.completion.assert_called_once()
        self.assertEqual(result.map_requests, 28)
        context = json.loads(self.completion.call_args.args[0][1]["content"])
        self.assertEqual([hint["title"] for hint in context["source_name_hints"]], seeds[:9])
        self.assertEqual(context["remaining_name_slots"], 19)
        self.assertEqual({key: result.diagnostics[key] for key in
                          ("attempted_names", "name_limit", "usable_limit", "limits_reached")},
                         {"attempted_names": 28, "name_limit": 28,
                          "usable_limit": 18, "limits_reached": ["name_limit"]})

    def test_source_network_failures_cannot_exhaust_http_before_ai_supplement(self):
        seeds = [f"Source Museum {index}" for index in range(36)]
        self.hints.return_value = [{"title": title, "source_url": "https://example.invalid/name"} for title in seeds]
        supplement = "Supplement Museum"
        self.completion.return_value.text = json.dumps({"titles": [supplement]})
        queried = []
        def get(url, **kwargs):
            title = kwargs["params"]["q"].split(",", 1)[0]
            queried.append(title)
            if title == supplement:
                return response([record(title)])
            raise httpx.ConnectTimeout("offline")
        self.get.side_effect = get
        result = candidates.ground_city_candidates(request(7), CENTER, completion=self.completion,
                                                   allow_partial_program=True)
        self.assertEqual((result.map_requests, len(result.candidates)), (36, 1))
        self.assertEqual(queried.index(supplement), 18)
        self.assertEqual(result.diagnostics["limits_reached"], ["http_limit"])
        self.completion.assert_called_once()

    def test_ai_failure_still_checks_deferred_source_names_within_shared_budget(self):
        seeds = [f"Source Museum {index}" for index in range(12)]
        self.hints.return_value = [{"title": title, "source_url": "https://example.invalid/name"} for title in seeds]
        self.completion.side_effect = RuntimeError("offline synthetic failure")
        def get(url, **kwargs):
            title = kwargs["params"]["q"].split(",", 1)[0]
            return response([record(title)] if title == seeds[-1] else [])
        self.get.side_effect = get
        result = candidates.ground_city_candidates(request(), CENTER, completion=self.completion,
                                                   allow_partial_program=True)
        self.assertEqual((result.status, [item.title for item in result.candidates]), ("success", [seeds[-1]]))
        self.assertEqual(result.map_requests, 12)
        self.assertTrue(result.warnings)
        self.completion.assert_called_once()

    def test_ekaterinburg_public_name_hints_survive_overpass_502_and_leave_room_for_museum(self):
        self.map_hints.side_effect = _REAL_MAP_HINTS
        city = "Екатеринбург"
        titles = ["Гринвич", "Пассаж"]
        museum = "Тестовый художественный музей"
        self.hints.side_effect = name_hints.destination_name_hints
        self.completion.return_value.text = json.dumps({"titles": [museum]}, ensure_ascii=False)
        def get(url, **kwargs):
            if url == candidates._SHOPPING_DISCOVERY_URL:
                raise httpx.HTTPStatusError("Synthetic 502", request=httpx.Request("GET", url),
                                            response=httpx.Response(502))
            title = kwargs["params"]["q"].split(",", 1)[0]
            self.assertIn(title, titles + [museum])
            # Deliberately synthetic coordinates; the real sources supply names only.
            return response([record(title, (titles + [museum]).index(title) + 1,
                                    category="tourism" if title == museum else "shop",
                                    type="museum" if title == museum else "mall",
                                    display_name=f"{title}, {city}, Россия",
                                    address={"city": city, "country": "Россия", "country_code": "ru"})])
        self.get.side_effect = get
        result = candidates.ground_city_candidates(
            request(5, destination=f"{city}, Россия", interest_categories=["museums", "shopping_malls"]),
            {**CENTER, "name": city}, completion=self.completion, allow_partial_program=True)
        self.assertEqual((result.status, [item.title for item in result.candidates]), ("success", titles + [museum]))
        self.assertEqual(result.map_requests, 4)
        self.assertTrue(result.warnings)
        self.completion.assert_called_once()
        context = json.loads(self.completion.call_args.args[0][1]["content"])
        self.assertEqual([hint["title"] for hint in context["source_name_hints"]], titles)
        self.assertEqual(context["remaining_name_slots"], 26)

    def test_names_only_schema_has_request_specific_cap(self):
        for days, expected in ((1, 12), (4, 24), (7, 36), (8, 36), (9, 44), (14, 60)):
            with self.subTest(days=days):
                self.run_pool([], days=days)
                schema = self.completion.call_args.kwargs["json_schema"]
                self.assertEqual(set(schema["properties"]), {"titles"})
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(schema["properties"]["titles"]["maxItems"], expected)
                self.assertEqual(schema["properties"]["titles"]["items"]["type"], "string")

    def test_rainy_dates_reach_name_only_prompt_without_new_calls_or_facts(self):
        result = candidates.ground_city_candidates(request(days=2), CENTER, completion=self.completion,
                                                   rainy_dates=["2099-10-02", "2099-10-01"])
        messages = self.completion.call_args.args[0]
        context = json.loads(messages[1]["content"])
        self.assertEqual(context["rainy_dates"], ["2099-10-01", "2099-10-02"])
        self.assertIn("разных дождливых дат", messages[0]["content"])
        self.assertIn("не подтверждают indoor", messages[0]["content"])
        self.assertEqual((self.completion.call_count, result.map_requests), (1, 1))
        self.assertEqual(set(self.completion.call_args.kwargs["json_schema"]["properties"]), {"titles"})
        self.assertNotIn("indoor", result.candidates[0].record)

    def test_invalid_rainy_dates_stop_before_completion_and_map(self):
        for rainy_dates in (None, "2099-10-01", {"2099-10-01"}, [True], ["20991001"],
                            ["2099-10-01T00:00:00"], ["2099-09-30"], ["2099-10-03"],
                            ["2099-10-01", "2099-10-01"], ["2099-10-01"] * 3):
            with self.subTest(rainy_dates=rainy_dates):
                result = candidates.ground_city_candidates(request(days=2), CENTER,
                    completion=self.completion, rainy_dates=rainy_dates)
                self.assertTrue(result.problems)
                self.assertEqual((result.usage["completion_calls"], result.map_requests), (0, 0))
        self.completion.assert_not_called()
        self.get.assert_not_called()

    def test_default_empty_rainy_dates_keep_single_names_completion(self):
        result = self.run_pool()
        self.assertEqual(json.loads(self.completion.call_args.args[0][1]["content"])["rainy_dates"], [])
        self.assertEqual((self.completion.call_count, result.map_requests), (1, 1))

    def test_invalid_json_unknown_fields_nonfinite_and_excess_names_fail_before_map(self):
        for text in ("bad", '{"titles": [], "titles": []}', '{"titles": [NaN]}',
                     '{"titles": [], "indoor": true}', '{"titles": [true]}',
                     json.dumps({"titles": ["Музей " + str(i) for i in range(13)]}),
                     json.dumps({"titles": ["Музей, город"]}), json.dumps({"titles": ["Музей\nтест"]})):
            with self.subTest(text=text):
                self.completion.return_value.text = text
                result = self.run_pool()
                self.assertTrue(result.problems)
                self.assertEqual((result.map_requests, result.usage["completion_calls"]), (0, 1))
        self.get.assert_not_called()

    def test_completion_failure_is_not_retried(self):
        self.completion.side_effect = RuntimeError("provider failure")
        result = self.run_pool()
        self.assertTrue(result.problems)
        self.assertEqual(result.usage["completion_calls"], 1)
        self.completion.assert_called_once()
        self.get.assert_not_called()

    def test_invalid_center_wrong_country_admin_or_region_stops_before_completion(self):
        for center, changes in ((dict(CENTER, lat=True), {}), (dict(CENTER, lon=float("nan")), {}),
                                (dict(CENTER, lat="52"), {}), (dict(CENTER, country_code=""), {}),
                                (dict(CENTER, country_code="TR"), {}), (dict(CENTER, name="Другой город"), {}),
                                (CENTER, {"destination_kind": "region"}),
                                (CENTER, {"destination": f"{AREA}, Другая область, Россия"})):
            with self.subTest(center=center, changes=changes):
                result = self.run_pool(center=center, **changes)
                self.assertTrue(result.problems)
                self.assertEqual(result.usage["completion_calls"], 0)
        self.completion.assert_not_called()
        self.get.assert_not_called()

    def test_country_alias_and_three_part_context_keep_confirmed_center(self):
        result = self.run_pool(destination=f"{AREA}, Тестовая область, Russia")
        self.assertFalse(result.problems)
        self.assertEqual(result.candidates[0].place_query, f"{TITLE}, {AREA}, Россия")

    def test_provider_alias_canonical_name_and_neighbour_locality_not_model_city(self):
        official = "Музей северных ремёсел"
        raw = record(title=official, namedetails={"name": official, "alt_name": TITLE},
                     address={"town": "Соседний посёлок", "country": "Россия", "country_code": "ru"},
                     display_name=f"{official}, Соседний посёлок, Россия")
        self.get.return_value = response([raw])
        item = self.run_pool().candidates[0]
        self.assertEqual((item.title, item.area), (official, "Соседний посёлок"))
        self.assertEqual(item.place_query, f"{official}, Соседний посёлок, Россия")

    def test_explicit_short_and_full_park_forms_keep_provider_canonical_name(self):
        for short, official in (("Парк Ривьера", "Парк культуры и отдыха Ривьера"),
                                ("парк «Северный свет»", "Парк культуры и отдыха Северный свет"),
                                ("Парк имени Гагарина", "Парк культуры и отдыха имени Гагарина"),
                                ("Парк культуры и отдыха Лунный сад", "Парк Лунный сад")):
            with self.subTest(short=short, official=official):
                self.get.return_value = response([record(official, type="park")])
                result = self.run_pool([short])
                self.assertEqual(len(result.candidates), 1)
                item = result.candidates[0]
                self.assertEqual(item.title, official)
                self.assertEqual(item.place_query, f"{official}, {AREA}, Россия")
                self.assertEqual(item.record["namedetails"], {"name": official})
                self.assertTrue(places._valid_record_only_place(item.place_query, item.record))
                self.assertFalse(places._place_matches_query(f"{short}, Россия", item.record))
                self.assertEqual(result.map_requests, 1)

    def test_park_prefix_normalization_cannot_drop_identity_or_change_type(self):
        cases = (
            ("Парк Ривьера", "Парк культуры и отдыха Южная Ривьера", "park"),
            ("Ривьера", "Парк культуры и отдыха Ривьера", "park"),
            ("Парк Ривьера", "Парк культуры и отдыха Ривьера", "museum"),
            ("Парк Ривьера", "Парк культуры и отдыха Ривьера", "artwork"),
            ("Парк имени Гагарина", "Парк культуры и отдыха Гагарина", "park"),
            ("Парк Ривьера", "Природный парк Ривьера", "park"),
            ("Парк Северный свет", "Парк культуры и отдыха Свет Северный", "park"),
            ("Парк", "Парк культуры и отдыха", "park"),
            ("Парк имени", "Парк культуры и отдыха имени", "park"),
        )
        for proposal, official, kind in cases:
            with self.subTest(proposal=proposal, official=official, kind=kind):
                self.get.return_value = response([record(official, type=kind)])
                result = candidates.CityCandidateResult()
                self.assertIsNone(candidates._lookup(proposal, "Россия", dict(CENTER, country_code="ru"), result))
                self.assertLessEqual(result.map_requests, 1)

    def test_park_name_normalization_keeps_country_and_radius_guards(self):
        official = "Парк культуры и отдыха Ривьера"
        for changes in ({"address": {"city": AREA, "country": "Турция", "country_code": "tr"}},
                        {"lat": "55"}, {"address": {"country": "Россия", "country_code": "ru"}}):
            with self.subTest(changes=changes):
                self.get.return_value = response([record(official, type="park", **changes)])
                self.assertFalse(self.run_pool(["Парк Ривьера"]).candidates)

    def test_wrong_name_country_type_identity_and_coordinates_are_not_confirmed(self):
        bad_records = [
            record(title="Другой музей"), record(address={"city": AREA, "country": "Россия", "country_code": "tr"}),
            record(type="locality", category="place"), record(type="hotel"), record(osm_id=True),
            record(osm_id="123"), record(osm_type="bad"), record(osm_id=None),
            record(lat=True), record(lat="nan"), record(lon="inf"), record(lat="91"),
            record(address={"state": "Тестовая область", "country": "Россия", "country_code": "ru"}),
        ]
        for raw in bad_records:
            with self.subTest(raw=raw):
                self.get.return_value = response([raw])
                result = candidates.CityCandidateResult()
                self.assertIsNone(candidates._lookup(TITLE, "Россия", dict(CENTER, country_code="ru"), result))
                self.assertEqual(result.map_requests, 1)
                self.assertEqual(result.rejected[0]["reason"], "limited_search_unconfirmed")

    def test_failed_ai_alias_can_add_separate_provider_named_osm_observation(self):
        official = "Музей истории Северного края имени Н. А. Белова"
        self.get.return_value = response([record(official, 987)])
        result = self.run_pool(["Музей Северного края"])
        self.assertEqual([item.title for item in result.candidates], [official])
        self.assertEqual(result.candidates[0].candidate_id, "osm-node-987")
        self.assertEqual(result.candidates[0].record["namedetails"], {"name": official})
        self.assertEqual(result.rejected[0]["title"], "Музей Северного края")
        self.assertIn("name_mismatch", result.rejected[0]["record_reasons"])
        self.assertEqual(result.diagnostics["source_alternatives_confirmed"], 1)
        self.assertEqual(result.diagnostics["attempted_names"], 2)
        self.assertEqual(result.map_requests, 1)
        self.assertFalse(places._place_matches_query("Музей Северного края, Россия", result.candidates[0].record))

    def test_provider_alternative_keeps_country_radius_type_name_and_identity_guards(self):
        other = "Музей истории Северного края"
        cases = ({"address": {"city": AREA, "country": "Турция", "country_code": "tr"}},
                 {"lat": "60"}, {"osm_id": None}, {"type": "hotel"},
                 {"namedetails": None}, {"category": "amenity", "type": "theatre"})
        for changes in cases:
            with self.subTest(changes=changes):
                self.get.return_value = response([record(other, **changes)])
                result = self.run_pool([TITLE])
                self.assertFalse(result.candidates)
                self.assertEqual(result.diagnostics["source_alternatives_confirmed"], 0)
                self.assertEqual(result.map_requests, 1)

    def test_provider_alternative_does_not_substitute_source_hint_or_exact_identity(self):
        self.get.return_value = response([record("Другой подтверждённый музей", 987)])
        self.hints.return_value = [{"title": TITLE, "source_url": "https://example.invalid/name"}]
        self.assertFalse(self.run_pool([]).candidates)
        alternatives = []
        result = candidates.CityCandidateResult()
        self.assertIsNone(candidates._lookup(TITLE, "Россия", dict(CENTER, country_code="ru"), result,
            source_identity=("osm", "node", 123), source_alternatives=alternatives,
            alternative_categories=["museums"]))
        self.assertEqual(alternatives, [])

    def test_provider_alternatives_share_name_capacity_and_deduplicate_actual_identity(self):
        self.get.return_value = response([record(f"Музей Северный {i}", i + 1) for i in range(5)])
        result = self.run_pool([f"Музей Южный {i}" for i in range(36)], days=7)
        self.assertEqual(len(result.candidates), 5)
        self.assertEqual(result.diagnostics["source_alternatives_confirmed"], 5)
        self.assertEqual(result.diagnostics["attempted_names"], 36)
        self.assertEqual(result.map_requests, 31)
        self.assertLessEqual(result.map_requests, 36)
        self.completion.assert_called_once()

    def test_strict_circle_rejects_outside_40_even_inside_viewbox(self):
        self.get.return_value = response([record(lat="52.3", lon="105.5")])
        self.assertGreater(places.distance_km(52, 105, 52.3, 105.5), 40)
        self.assertFalse(self.run_pool().candidates)
        self.get.return_value = response([record(lat="52.35", lon="105")])
        self.assertTrue(self.run_pool().candidates)

    def test_unusable_first_result_does_not_hide_later_valid_identity(self):
        self.get.return_value = response([record(osm_id=None), record(lat="55"), record(identifier=456)])
        result = self.run_pool()
        self.assertEqual(result.candidates[0].candidate_id, "osm-node-456")
        self.assertEqual(result.map_requests, 1)

    def test_cache_hit_has_zero_http_no_mutation_and_no_schedule_fact_injection(self):
        query = f"{TITLE}, {AREA}, Россия"
        self.cache[query.lower()] = cached(record())
        self.cache[query.lower()].update(indoor=True, cost=100, duration_min=999, opening_hours="24/7")
        before = deepcopy(self.cache)
        result = self.run_pool()
        self.assertEqual(result.map_requests, 0)
        self.assertEqual(self.cache, before)
        self.get.assert_not_called()
        for key in ("indoor", "cost", "duration_min", "opening_hours"):
            self.assertNotIn(key, result.candidates[0].record)
        result.candidates[0].record["address"]["city"] = "Изменено"
        self.assertEqual(self.cache, before)

    def test_invalid_cache_key_or_missing_id_requires_real_lookup(self):
        for key, raw in (("Другой музей, Россия", record()), (f"{TITLE}, {AREA}, Россия", record(osm_id=None))):
            with self.subTest(key=key):
                self.cache.clear()
                self.cache[key.lower()] = cached(raw)
                result = self.run_pool()
                self.assertEqual(result.map_requests, 1)

    def test_cached_provenance_must_pass_record_only_kernel_before_selection(self):
        query = f"{TITLE}, {AREA}, Россия"
        for changes in ({"source_url": "https://example.invalid/forged"}, {"provenance": "model"}):
            with self.subTest(changes=changes):
                self.cache.clear()
                self.cache[query.lower()] = cached(record(**changes))
                before = deepcopy(self.cache)
                self.get.return_value = response([])
                result = self.run_pool()
                self.assertFalse(result.candidates)
                self.assertEqual(result.map_requests, 1)
                self.assertEqual(self.cache, before)

    def test_actual_waterfall_artwork_record_is_not_a_natural_waterfall(self):
        # Identity/type/address from the observed map record, no live lookup.
        title = "Агурские водопады"
        raw = record(title, 7342982234, lat="43.5475075", lon="39.811327", type="artwork",
                     address={"suburb": "Малый Ахун", "county": "городской округ Сочи",
                              "country": "Россия", "country_code": "ru"},
                     display_name=f"{title}, Малый Ахун, городской округ Сочи, Россия")
        self.get.return_value = response([raw])
        result = self.run_pool([title], center={"lat": 43.6, "lon": 39.73, "name": "Сочи", "country_code": "RU"},
                               destination="Сочи, Россия")
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.map_requests, 1)
        self.assertEqual(result.rejected[0]["reason"], "limited_search_unconfirmed")

    def test_unlisted_waterfall_names_require_waterfall_type(self):
        for title in ("Северные водопады", "Водопад Серебряный", "Каскад Звёздных водопадов",
                      "Silver Waterfall", "Hidden Waterfalls"):
            for kind in ("waterfall", "artwork", "locality", "museum", "attraction", "viewpoint", "park"):
                with self.subTest(title=title, kind=kind):
                    self.get.return_value = response([record(title, type=kind, category="waterway" if kind == "waterfall" else "tourism")])
                    result = self.run_pool([title])
                    self.assertEqual(bool(result.candidates), kind == "waterfall")
                    self.assertEqual(result.map_requests, 1)

    def test_explicit_museum_and_sculpture_prefixes_do_not_mean_waterfall(self):
        for title, expected_type in (("Музей «Водопад»", "museum"), ("Museum Waterfalls", "museum"),
                                     ("Скульптура «Водопад»", "artwork"), ("Sculpture Waterfall", "artwork")):
            for kind in ("museum", "artwork", "waterfall"):
                with self.subTest(title=title, kind=kind):
                    self.get.return_value = response([record(title, type=kind)])
                    result = self.run_pool([title])
                    self.assertEqual(bool(result.candidates), kind == expected_type)

    def test_non_waterfall_artwork_remains_eligible(self):
        title = "Скульптура Северный ветер"
        self.get.return_value = response([record(title, type="artwork")])
        result = self.run_pool([title])
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].record["type"], "artwork")

    def test_natural_waterfall_cache_reuses_valid_record_without_map_requests(self):
        title = "Серебряный водопад"
        query = f"{title}, {AREA}, Россия"
        for category in ("waterway", "natural"):
            with self.subTest(category=category):
                self.cache.clear()
                self.cache[query.lower()] = cached(record(title, type="waterfall", category=category))
                before = deepcopy(self.cache)
                result = self.run_pool([title])
                self.assertEqual(len(result.candidates), 1)
                self.assertEqual(result.map_requests, 0)
                item = result.candidates[0]
                self.assertTrue(places._valid_record_only_place(item.place_query, item.record))
                self.assertEqual(self.cache, before)
        self.get.assert_not_called()

    def test_provider_alias_cannot_hide_explicit_natural_type_mismatch(self):
        for proposal, official in (("Северный каскад", "Северные водопады"),
                                   ("Северные водопады", "Северный каскад")):
            with self.subTest(proposal=proposal, official=official):
                self.get.return_value = response([record(official, type="artwork",
                    namedetails={"name": official, "alt_name": proposal})])
                self.assertFalse(self.run_pool([proposal]).candidates)

    def test_same_id_alias_deduplicates_but_distinct_ids_at_same_coordinates_survive(self):
        second, third = "Музей городского быта", "Музей современной истории"
        first = record(namedetails={"name": TITLE, "alt_name": second})
        self.get.side_effect = [response([first]), response([first]), response([record(third, 124)])]
        result = self.run_pool([TITLE, second, third])
        self.assertEqual([item.candidate_id for item in result.candidates], ["osm-node-123", "osm-node-124"])
        self.assertEqual(result.map_requests, 3)

    def test_duplicate_proposal_names_and_generic_labels_do_not_spend_queries(self):
        result = self.run_pool([TITLE, TITLE.upper(), "Обед", "Отдых", "Музей", "Парк"])
        self.assertEqual(result.map_requests, 1)
        self.assertEqual(len(result.rejected), 4)
        self.assertEqual(result.problems, [])

    def test_network_retry_counts_actual_attempts_and_uses_shared_gate_each_time(self):
        self.get.side_effect = [httpx.ConnectTimeout("offline"), response([record()])]
        result = self.run_pool()
        self.assertEqual((result.map_requests, self.get.call_count, self.gate.call_count), (2, 2, 2))
        self.assertFalse(result.problems)
        self.completion.assert_called_once()

    def test_18_names_with_two_failures_each_are_bounded_to_36_actual_requests(self):
        self.get.side_effect = httpx.ConnectTimeout("offline")
        result = self.run_pool([f"Музей истории номер {index}" for index in range(18)], days=7)
        self.assertEqual((result.map_requests, self.get.call_count, self.gate.call_count), (36, 36, 36))
        self.assertEqual(len(result.rejected), 18)
        self.assertTrue(result.problems)
        self.completion.assert_called_once()

    def test_clock_failure_has_zero_sends_no_retry(self):
        self.gate.side_effect = places.GeocodeUnavailable("clock failed")
        result = self.run_pool()
        self.assertEqual(result.map_requests, 0)
        self.get.assert_not_called()
        self.assertEqual(result.rejected[0]["reason"], "shared_map_clock_unavailable")

    def test_partial_rejection_does_not_make_sufficient_pool_fatal(self):
        self.get.side_effect = [response([]), response([record()])]
        result = self.run_pool(["Музей ненайденного объекта", TITLE])
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(result.rejected), 1)
        self.assertFalse(result.problems)

    def test_fewer_candidates_than_days_is_explicitly_insufficient(self):
        result = self.run_pool(days=2)
        self.assertEqual(len(result.candidates), 1)
        self.assertTrue(result.problems)
        self.assertIn("1 уникальных мест на 2 дней", result.problems[0])

    def test_authorized_partial_pool_retains_positive_candidates_without_claiming_calendar(self):
        result = candidates.ground_city_candidates(request(days=2), CENTER, completion=self.completion,
                                                   allow_partial_program=True)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.problems, [])
        self.assertEqual((result.usage["completion_calls"], result.map_requests), (1, 1))
        self.assertFalse(hasattr(result, "itinerary"))
        self.assertFalse(hasattr(result, "program_report"))

    def test_authorized_partial_pool_still_rejects_zero_candidates(self):
        self.get.return_value = response([])
        result = candidates.ground_city_candidates(request(days=2), CENTER, completion=self.completion,
                                                   allow_partial_program=True)
        self.assertEqual(result.candidates, [])
        self.assertTrue(result.problems)
        self.assertEqual((result.usage["completion_calls"], result.map_requests), (1, 1))

    def test_partial_pool_permission_requires_exact_boolean(self):
        for value in (1, 0, "true", None):
            with self.subTest(value=value):
                result = candidates.ground_city_candidates(request(days=2), CENTER, completion=self.completion,
                                                           allow_partial_program=value)
                self.assertTrue(result.problems)
                self.assertEqual((result.usage["completion_calls"], result.map_requests), (0, 0))
        self.completion.assert_not_called()
        self.get.assert_not_called()

    def test_current_independent_provider_uses_exact_catalogue_query(self):
        raw = verified_map_places._record({
            "title": TITLE, "place_queries": [f"{TITLE}, {AREA}, Россия"], "lat": 52.02, "lon": 105.03,
            "address": {"city": AREA, "county": "Тестовый округ", "state": "Тестовая область",
                        "country": "Россия", "country_code": "ru", "road": "Тестовая улица", "house_number": "1"},
            "type": "museum", "provider": "2gis", "provider_id": "123",
            "source_url": "https://2gis.ru/test/firm/123", "official_url": "https://museum.example.test/contacts",
            "official_link_url": "https://go.2gis.com/testlink", "checked_at": "2026-10-21",
        }, date(2026, 10, 21))
        self.independent.append(raw)
        result = self.run_pool()
        self.assertEqual(result.map_requests, 0)
        self.assertEqual(result.candidates[0].candidate_id, "2gis-123")
        self.assertIn(result.candidates[0].place_query, raw["catalog_queries"])
        self.assertTrue(verified_map_places.is_current_verified_map_record(result.candidates[0].record))

    def test_forged_independent_provenance_in_cache_is_not_evidence(self):
        self.cache[f"{TITLE}, {AREA}, Россия".lower()] = cached(record(provider="2gis", provider_id="123"))
        self.get.return_value = response([])
        result = self.run_pool()
        self.assertEqual(result.map_requests, 1)
        self.assertFalse(result.candidates)

    def test_candidate_records_work_with_record_only_verification_without_http(self):
        result = self.run_pool()
        item = result.candidates[0]
        route = Itinerary(title="Проверка", summary="Карта", days=[ItineraryDay(day=1, date="2099-10-01", theme="Музей", items=[
            ItineraryItem(title=item.title, place_query=item.place_query, area=item.area,
                          time="10:00", description="Посещение", duration_min=90),
        ])])
        self.get.reset_mock()
        self.assertEqual(places.verify_itinerary(route, verified_records={item.place_query: item.record}), [])
        self.assertTrue(route.days[0].items[0].verified)
        self.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
