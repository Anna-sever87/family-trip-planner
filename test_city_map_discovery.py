"""Offline provider-shaped fixtures for category discovery, not AI name mocks."""
from contextlib import nullcontext
from copy import deepcopy
import json
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

import city_candidates as city
import places
from test_city_candidates import CENTER, AREA, record, request, response


KINDS = {"museums": ("tourism", "museum"), "zoos": ("tourism", "zoo"),
         "theatre": ("amenity", "theatre"), "shopping_malls": ("shop", "mall")}


class MapFirstDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        self.query_log = []
        self.overpass_failure = False
        self.ai = Mock(return_value=SimpleNamespace(text='{"titles":[]}', input_tokens=0, output_tokens=0))
        self.patch(socket.socket, "connect", side_effect=AssertionError("No real network"))
        self.patch(socket, "getaddrinfo", side_effect=AssertionError("No DNS"))
        self.patch(places, "_cached_nearby_named", return_value=None)
        self.gate = self.patch(places, "_wait_for_turn", side_effect=lambda: nullcontext())
        self.patch(city, "destination_name_hints", return_value=[])
        self.get = self.patch(httpx, "get", side_effect=self.provider)

    def patch(self, owner, name, **kwargs):
        handle = patch.object(owner, name, **kwargs)
        self.addCleanup(handle.stop)
        return handle.start()

    def add(self, category, title, identifier):
        key, kind = KINDS[category]
        self.records.append(record(title, identifier, category=key, type=kind))

    def overpass_response(self, query):
        elements = [
            {"type": raw["osm_type"], "id": raw["osm_id"], "lat": float(raw["lat"]),
             "lon": float(raw["lon"]), "tags": {raw["category"]: raw["type"],
             "name": raw["namedetails"]["name"], "opening_hours": "invented must not propagate", "fee": "no",
             **raw.get("extratags", {})}}
            for raw in self.records]
        # Protocol-shaped fixture: `out tags center` omits node coordinates;
        # center supplies geometry only for ways/relations. Never mask this.
        if "out body center" not in query:
            for element in elements:
                element.pop("lat", None)
                element.pop("lon", None)
        return response({"elements": elements})

    def provider(self, url, **kwargs):
        params = kwargs["params"]
        self.query_log.append((url, deepcopy(params)))
        if url == city._SHOPPING_DISCOVERY_URL:
            if self.overpass_failure:
                raise httpx.HTTPStatusError("synthetic 502", request=httpx.Request("GET", url), response=httpx.Response(502))
            return self.overpass_response(params["data"])
        if url == city._MAP_LOOKUP_URL:
            identifier = int(params["osm_ids"][1:])
            return response([raw for raw in self.records if raw["osm_id"] == identifier])
        term = params["q"][1:-1] if params["q"].startswith("[") and params["q"].endswith("]") else ""
        category = next((key for key, value in city._CATEGORY_TERMS.items() if value == term), None)
        if category:
            return response([raw for raw in self.records
                             if category in city.candidate_categories(city._from_provider(raw))][:10])
        return response([])

    def pool(self, days=7, **changes):
        changes.setdefault("children_ages", [])
        changes.setdefault("start_time", "11:00")
        changes.setdefault("end_time", "20:00")
        changes.setdefault("interest_categories", list(KINDS))
        return city.ground_city_candidates(request(days, **changes), CENTER, completion=self.ai,
                                           allow_partial_program=True)

    def fill(self, count=6):
        for number in range(count):
            for category, label in (("museums", "Музей"), ("zoos", "Зоопарк"),
                                    ("theatre", "Театр"), ("shopping_malls", "Пассаж")):
                self.add(category, f"{label} Северный {number}", len(self.records) + 1)

    def test_balanced_map_first_pool_has_all_requested_types_without_ai_names(self):
        self.fill()
        result = self.pool()
        self.assertEqual(len(result.candidates), 24)
        self.assertTrue(all(result.diagnostics["category_counts"].values()))
        self.assertEqual(result.diagnostics["missing_categories"], [])
        # All categories can be present while their short visit estimates still
        # fall below the requested pace; category coverage alone is not fullness.
        self.assertEqual(result.diagnostics["pool_quality"], "limited")
        self.assertLess(result.diagnostics["planning_minutes"], result.diagnostics["required_planning_minutes"])
        self.assertTrue(result.diagnostics["replenishment_attempted"])
        self.assertLessEqual(result.map_requests, 36)
        self.assertEqual(result.usage["completion_calls"], 1)
        query = self.query_log[0][1]["data"]
        self.assertIn("out body center 24", query)
        self.assertIn("[maxsize:67108864]", query)
        for category in KINDS:
            for key, values in city._CATEGORY_FILTERS[category]:
                self.assertIn(f'["{key}"~"^({values})$"]', query)
        self.assertEqual(sum("q" in params for _, params in self.query_log), 0)
        for candidate in result.candidates:
            self.assertNotIn("opening_hours", candidate.record)
            self.assertNotIn("fee", candidate.record)
            self.assertNotIn("indoor", candidate.record)

    def test_overpass_502_has_typed_provider_search_reserve_not_three_ai_museums(self):
        self.fill()
        self.overpass_failure = True
        result = self.pool()
        self.assertEqual(len(result.candidates), 24)
        self.assertEqual(result.map_requests, 5)  # one outage + four category queries
        self.assertTrue(all(result.diagnostics["category_counts"].values()))
        self.assertTrue(result.warnings)
        self.assertEqual(result.diagnostics["map_discovery_status"], "unavailable")
        self.assertEqual(result.diagnostics["map_discovery_error"], "http_error")
        self.assertEqual(result.diagnostics["category_fallback_status"], "success")
        self.assertEqual(self.gate.call_count, 4)
        self.ai.assert_called_once()

    def test_provider_full_zoo_name_does_not_depend_on_ai_short_name_alias(self):
        self.add("zoos", "Новосибирский зоопарк имени Р. А. Шило", 501)
        result = self.pool(1, interest_categories=["zoos"])
        self.assertEqual([item.title for item in result.candidates], [self.records[0]["namedetails"]["name"]])
        self.assertEqual(self.query_log[1][1]["osm_ids"], "N501")
        self.assertEqual(result.diagnostics["category_counts"], {"zoos": 1})

    def test_live_observed_theatre_name_ambiguity_uses_bounded_special_phrase(self):
        # Live Nominatim: `theatre, Екатеринбург, Россия` and structured
        # amenity=theatre returned bus stops ТЮЗ/Драмтеатр. `[theatre]` returned
        # amenity/theatre. The fake models that protocol distinction explicitly.
        theatre = record("Театр Волхонка", 477653860, category="amenity", type="theatre")
        stop = record("Драмтеатр", 671868210, category="highway", type="bus_stop")
        def observed(url, **kwargs):
            if url == city._SHOPPING_DISCOVERY_URL:
                raise httpx.ConnectTimeout("offline")
            params = kwargs["params"]
            self.assertEqual(params["bounded"], 1)
            self.assertEqual(params["countrycodes"], "ru")
            self.assertIn("viewbox", params)
            return response([theatre] if params.get("q") == "[theatre]" else [stop])
        self.get.side_effect = observed
        result = self.pool(1, interest_categories=["theatre"])
        self.assertEqual([item.title for item in result.candidates], ["Театр Волхонка"])
        self.assertEqual(result.diagnostics["map_discovery_error"], "timeout")
        self.assertEqual(result.diagnostics["category_fallback_status"], "success")
        self.assertEqual(result.map_requests, 2)

    def test_fallback_terms_are_documented_amenity_keywords_not_descriptive_phrases(self):
        self.assertEqual(city._CATEGORY_TERMS["shopping_malls"], "mall")
        self.assertEqual(city._CATEGORY_TERMS["main_sights"], "attraction")
        self.get.return_value = response([])
        self.get.side_effect = None
        for category, term in city._CATEGORY_TERMS.items():
            result = city.CityCandidateResult()
            city._category_search(category, "Россия", CENTER, result)
            self.assertEqual(self.get.call_args.kwargs["params"]["q"], f"[{term}]")
            self.assertEqual(result.map_requests, 1)

    def test_category_discovery_does_not_promote_unnamed_theatre_address_to_poi_name(self):
        unnamed = record("улица Владимира Высоцкого", 827786173, category="amenity", type="theatre")
        unnamed["namedetails"] = None  # observed on a real unnamed theatre way
        self.get.side_effect = [httpx.ConnectTimeout("offline"), response([unnamed])]
        result = self.pool(1, interest_categories=["theatre"])
        self.assertFalse(result.candidates)
        self.assertEqual(result.diagnostics["detail_rejection_reasons"], {"invalid_record": 1})
        self.assertEqual(result.diagnostics["category_fallback_status"], "empty")

    def test_category_search_still_rejects_bus_stop_even_if_provider_misinterprets_special_phrase(self):
        stop = record("ТЮЗ", 1564976399, category="highway", type="bus_stop")
        self.get.side_effect = [httpx.ConnectTimeout("offline"), response([stop])]
        result = self.pool(1, interest_categories=["theatre"])
        self.assertFalse(result.candidates)
        self.assertEqual(result.diagnostics["detail_rejection_reasons"], {"category_mismatch": 1})

    def test_primary_provider_failure_reason_is_preserved_without_raw_server_text(self):
        for answer, expected in ((response({"elements": [], "remark": "server internal details"}), "incomplete_response"),
                                 (response([]), "invalid_response")):
            with self.subTest(expected=expected):
                result = city.CityCandidateResult()
                self.get.side_effect = [answer]
                self.assertEqual(city._map_name_hints(["theatre"], CENTER, result), [])
                self.assertEqual(result.diagnostics["map_discovery_error"], expected)
                self.assertNotIn("server internal details", json.dumps(result.diagnostics))

    def test_exact_source_id_must_not_be_substituted_by_same_named_object(self):
        self.add("shopping_malls", "Северный пассаж", 1)
        original = self.provider
        def other_id(url, **kwargs):
            if url == city._MAP_LOOKUP_URL:
                return response([dict(self.records[0], osm_id=2)])
            if url == places.NOMINATIM_URL:
                return response([])
            return original(url, **kwargs)
        self.get.side_effect = other_id
        result = self.pool(1, interest_categories=["shopping_malls"])
        self.assertFalse(result.candidates)
        self.assertEqual(result.diagnostics["detail_rejection_reasons"]["source_identity_mismatch"], 1)

    def test_fallback_rejects_wrong_country_radius_type_missing_id_and_generic_names(self):
        self.overpass_failure = True
        valid = record("Северный зоопарк", 1, type="zoo")
        bad = [dict(valid, address={"city": AREA, "country_code": "fr", "country": "France"}),
               dict(valid, lat="60"), dict(valid, type="museum"), dict(valid, osm_id=0),
               dict(valid, display_name=f"Зоопарк, {AREA}, Россия", namedetails={"name": "Зоопарк"})]
        def malformed(url, **kwargs):
            if url == city._SHOPPING_DISCOVERY_URL:
                raise httpx.ConnectTimeout("offline")
            return response(bad)
        self.get.side_effect = malformed
        result = self.pool(1, interest_categories=["zoos"])
        self.assertFalse(result.candidates)
        details = result.diagnostics["detail_rejection_reasons"]
        for reason in ("country_mismatch_or_missing", "outside_city_radius", "category_mismatch", "missing_provider_identity"):
            self.assertIn(reason, details)
        self.assertLessEqual(set(details), city.DETAIL_REJECTION_REASONS)

    def test_all_requests_share_global_budget_and_single_completion_on_outages(self):
        self.get.side_effect = httpx.ConnectTimeout("offline")
        self.ai.return_value.text = json.dumps({"titles": [f"Музей Северный {i}" for i in range(36)]})
        result = self.pool()
        self.assertEqual(result.map_requests, 36)
        self.assertEqual(self.get.call_count, 36)
        self.ai.assert_called_once()
        self.assertEqual(result.diagnostics["pool_quality"], "empty")

    def test_negative_shopping_wish_excludes_mall_discovery_filter(self):
        result = self.pool(1, wishes="Без торговых центров")
        self.assertNotIn('"shop"', self.query_log[0][1]["data"])
        self.assertNotIn("shopping_malls", result.diagnostics["requested_categories"])
        self.assertFalse(any(params.get("q") == "[mall]" for _, params in self.query_log))

    def test_goal_and_missing_categories_reach_only_name_proposal_context(self):
        result = self.pool(1, trip_goal="История и животные")
        context = json.loads(self.ai.call_args.args[0][1]["content"])
        self.assertEqual(context["trip_goal"], "История и животные")
        self.assertEqual(set(context["missing_categories"]), set(KINDS))
        self.assertNotIn("История и животные", json.dumps(result.diagnostics, ensure_ascii=False))

    def test_small_pool_reports_inadequate_volume_despite_successful_partial_result(self):
        for i in range(3):
            self.add("museums", f"Музей Северный {i}", i + 1)
        result = self.pool(7, interest_categories=["museums"])
        self.assertEqual(result.status, "success")
        self.assertEqual(result.diagnostics["pool_quality"], "limited")
        self.assertEqual(result.diagnostics["planning_minutes"], 270)
        self.assertEqual(result.diagnostics["required_planning_minutes"], 1890)
        self.assertTrue(result.diagnostics["replenishment_attempted"])

    def test_longer_museum_pool_can_meet_volume_without_claiming_actual_schedule(self):
        for i in range(21):
            self.add("museums", f"Музей Северный {i}", i + 1)
        result = self.pool(7, interest_categories=["museums"])
        self.assertEqual(result.diagnostics["pool_quality"], "adequate")
        self.assertGreaterEqual(result.diagnostics["planning_minutes"], 1890)
        self.assertNotIn("program_days", result.diagnostics)

    def test_usable_budget_scales_with_days_and_volume_without_expanding_http_cap(self):
        for days, start, end, expected in ((1, "11:00", "20:00", 6), (5, "10:00", "18:00", 20),
                                            (7, "11:00", "20:00", 24), (4, "08:00", "22:00", 20)):
            with self.subTest(days=days, start=start):
                result = self.pool(days, start_time=start, end_time=end, interest_categories=["museums"])
                self.assertEqual(result.diagnostics["usable_limit"], expected)
                self.assertLessEqual(result.map_requests, 36)

    def test_same_object_from_fallback_is_deduplicated_and_source_names_are_not_rewritten(self):
        self.add("museums", "Музей имени Н. К. Северного", 1)
        result = self.pool(7, interest_categories=["museums"])
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].title, "Музей имени Н. К. Северного")
        self.assertFalse(result.normalized_proposals)

    def test_water_trips_are_not_inferred_from_ferry_or_river_map_types(self):
        self.assertEqual(city.candidate_categories({"category": "amenity", "type": "ferry_terminal"}), frozenset())
        result = self.pool(1, interest_categories=["water_trips"])
        self.assertEqual(result.diagnostics["requested_categories"], [])
        self.assertEqual(result.map_requests, 0)

    def test_category_response_size_and_oversized_array_are_bounded(self):
        for payload, content in (([record()] * 11, b"[]"), ([], b"x" * (city._DISCOVERY_BYTE_LIMIT + 1))):
            with self.subTest(size=len(content)):
                answer = response(payload)
                answer.content = content
                self.get.side_effect = [httpx.ConnectTimeout("offline"), answer]
                result = self.pool(1, interest_categories=["museums"])
                self.assertFalse(result.candidates)
                self.assertEqual(result.map_requests, 2)
                self.assertIn("map_unavailable_or_invalid_response", result.diagnostics["detail_rejection_reasons"])

    def test_only_observed_small_ride_tags_remove_main_sight_classification(self):
        ride = record("Северная карусель", 81, type="attraction", extratags={"attraction": "carousel"})
        observed = city._from_provider(ride)
        self.assertEqual(city.candidate_categories(observed), frozenset({"amusement_parks"}))
        unclassified = city._from_provider(record("Воздушные шары", 82, type="attraction"))
        self.assertEqual(city.candidate_categories(unclassified), frozenset({"main_sights"}))
        # A label alone must not invent an observed ride subtype.
        self.assertFalse(city._small_ride(unclassified))
        self.records = [ride]
        result = self.pool(1, interest_categories=["main_sights"])
        self.assertFalse(result.candidates)
        self.assertEqual(result.diagnostics["category_counts"], {"main_sights": 0})

    def test_ride_ai_proposal_requires_explicit_amusement_interest(self):
        ride = record("Северная карусель", 81, type="attraction", extratags={"attraction": "carousel"})
        self.patch(city, "_map_name_hints", return_value=[])
        self.patch(city, "_category_search", return_value=[])
        self.get.side_effect = None
        self.get.return_value = response([ride])
        self.ai.return_value.text = json.dumps({"titles": ["Северная карусель"]})
        ordinary = self.pool(1, interest_categories=["main_sights"])
        self.assertFalse(ordinary.candidates)
        self.assertIn("amusement_interest_not_requested", ordinary.diagnostics["detail_rejection_reasons"])
        requested = self.pool(1, interest_categories=["amusement_parks"])
        self.assertEqual([item.candidate_id for item in requested.candidates], ["osm-node-81"])
        self.assertEqual(requested.candidates[0].record["type"], "attraction")

    def test_overview_reserve_limits_specialized_animal_places_without_claiming_duplicates(self):
        for number, title in enumerate(("Пони-ферма Северная", "Пони‑ферма Южная", "Контактный зоопарк Лесной"), 1):
            self.add("zoos", title, number)
            self.records[-1]["lat"] = str(52 + number * .02)
        self.add("zoos", "Северный городской зоопарк", 10)
        result = self.pool(7, interest_categories=["zoos"], trip_goal="Познакомиться с городом")
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(sum(city._specialized_animal_place(item.record) for item in result.candidates), 1)
        self.assertIn("osm-node-10", [item.candidate_id for item in result.candidates])
        self.assertEqual(result.diagnostics["detail_rejection_reasons"]["specialized_animal_reserve_limit"], 2)
        self.assertFalse(any("duplicate" in item.get("reason", "") for item in result.rejected))
        themed = self.pool(7, interest_categories=["zoos"], wishes="Хочу посетить пони-фермы")
        self.assertEqual(len(themed.candidates), 4)
        self.assertEqual(len({item.candidate_id for item in themed.candidates}), 4)

    def test_specialized_subtype_requires_zoo_and_never_infers_visitor_facts(self):
        ordinary = city._from_provider(record("Северный зоопарк", type="zoo", extratags={"zoo": "petting_zoo"}))
        self.assertTrue(city._specialized_animal_place(ordinary))
        self.assertFalse(city._specialized_animal_place(dict(ordinary, type="museum")))
        self.assertFalse(city._specialized_animal_place(city._from_provider(record("Северный зоопарк", type="zoo"))))

    def test_observed_subtype_roundtrips_city_context_without_schedule_facts(self):
        import city_context
        for kind, tags in (("attraction", {"attraction": "carousel"}), ("zoo", {"zoo": "petting_zoo"})):
            with self.subTest(kind=kind):
                supplied = dict(tags, opening_hours="09:00-18:00", fee="no", duration="180", operator="Someone")
                raw = record("Северная площадка", 81, type=kind, extratags=supplied)
                canonical = city._canonical_record("Северная площадка", city._from_provider(raw), dict(CENTER, country_code="ru"))
                candidate = city.CityCandidate("osm-node-81", "Северная площадка", canonical["place_query"], AREA, canonical)
                trip = request(1, children_ages=[], interest_categories=["amusement_parks", "zoos"])
                payload = city_context.build_city_context(trip, [candidate], CENTER)
                loaded, _ = city_context.load_city_context(payload, trip)
                self.assertEqual(loaded[0].record["extratags"], tags)
                self.assertEqual(city.candidate_categories(loaded[0]), city.candidate_categories(candidate))
                for forbidden in ("opening_hours", "fee", "duration", "operator", "indoor", "cost"):
                    self.assertNotIn(forbidden, loaded[0].record)
                    self.assertNotIn(forbidden, loaded[0].record["extratags"])

    def test_first_map_tranche_is_one_per_category_before_ai_named_museum(self):
        self.fill()
        actual_ai_museum = record("Музей северного искусства", 100)
        original_provider = self.provider
        def provider(url, **kwargs):
            if kwargs["params"].get("q", "").startswith("Музей северного искусства,"):
                return response([actual_ai_museum])
            return original_provider(url, **kwargs)
        self.get.side_effect = provider
        def propose(messages, **kwargs):
            checked = json.loads(messages[1]["content"])["map_names_already_checked"]
            self.assertEqual(len(checked), len(KINDS))
            for label in ("Музей", "Зоопарк", "Театр", "Пассаж"):
                self.assertEqual(sum(name.startswith(label) for name in checked), 1)
            return SimpleNamespace(text=json.dumps({"titles": ["Музей северного искусства"]}), input_tokens=0, output_tokens=0)
        self.ai.side_effect = propose
        result = self.pool()
        self.assertIn("osm-node-100", [item.candidate_id for item in result.candidates])
        self.assertLessEqual(len(result.candidates), 24)
        self.assertLessEqual(result.map_requests, 36)
        self.ai.assert_called_once()

    def test_high_quality_reserve_precedes_unclassified_attraction_fill(self):
        for number in range(20):
            self.add("museums", f"Музей Северный {number}", number + 1)
        self.records.extend(record(f"Развлекательная площадка Северная {i}", 100 + i, type="attraction") for i in range(5))
        result = self.pool(7, interest_categories=["museums", "main_sights"])
        counts = result.diagnostics["category_counts"]
        self.assertEqual(counts["museums"], 20)
        self.assertEqual(counts["main_sights"], 2)  # reserve remains museum-led, not equal category quotas
        self.assertEqual(result.diagnostics["pool_quality"], "adequate")
        self.assertEqual(result.usage["completion_calls"], 1)


if __name__ == "__main__":
    unittest.main()
