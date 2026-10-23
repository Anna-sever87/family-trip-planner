"""Candidate proposals are untrusted; all AI/map/cache boundaries are offline."""

import json
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import httpx
import candidate_grounding as grounding
import places
from destination_scope import RegionScope
from schemas import TripRequest
from yandex_ai import GenerationResult, YandexAIError


def request(days=2):
    return TripRequest(destination="Тестовый край, Россия", destination_kind="region",
                       start_date="2099-06-01", end_date=f"2099-06-{days:02d}", budget=100000)


def scope():
    return RegionScope("Тестовый край", "Россия", (50, 53, 100, 103), {
        "type": "Polygon", "coordinates": [[[100, 50], [103, 50], [103, 53], [100, 53], [100, 50]]],
    })


def batch(count=2):
    return {
        "bases": [
            {"base_key": "base_1", "base_query": "Северный, Тестовый край, Россия"},
            {"base_key": "base_2", "base_query": "Южный, Тестовый край, Россия"},
        ],
        "pois": [{"title": f"Музей промыслов номер {index}",
                  "place_query": f"Музей промыслов номер {index}, Долинный, Россия",
                  "locality": "Долинный", "base_key": "base_1"} for index in range(count)],
    }


def record(query, *, lat=51.0, lon=101.0, poi_type="museum", **overrides):
    parts = query.split(", ")
    result = {"name": query, "lat": lat, "lon": lon, "type": poi_type,
              "category": "place" if poi_type == "village" else "tourism",
              "address": {"village": parts[-2], "county": "Тестовый муниципальный округ", "country": "Россия", "country_code": "ru"}}
    result.update(overrides)
    return result


class CandidateGroundingTests(unittest.TestCase):
    def setUp(self):
        self.cache = {}
        for name, value in (("_cache", self.cache),):
            patcher = patch.object(places, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for owner, name in ((places.httpx, "get"), (places, "_save_cache")):
            patcher = patch.object(owner, name, side_effect=AssertionError("No network or cache writes"))
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_batch(self, payload=None, *, geocoder=None, days=2, max_http_attempts=42, raw=None, enforce_count=False):
        payload = batch() if payload is None else payload
        if geocoder is None:
            def geocoder(query, **kwargs):
                self.assertEqual(kwargs, {"max_variants": 1})
                if query.startswith(("Северный,", "Южный,")):
                    return record(query, poi_type="village", lon=101 if query.startswith("Северный") else 101.3)
                index = int(query.split(",")[0].split()[-1])
                return record(query, lon=101.05 + index / 100)
        lookup = Mock(side_effect=geocoder)
        def nearby(title, country, _lat, _lon, _radius, **_checks):
            proposal = next(poi for poi in payload["pois"] if poi["title"] == title)
            raw_record = lookup(f"{title}, {proposal['locality']}, {country}", max_variants=1)
            if raw_record is None:
                return None
            address = raw_record.get("address", {})
            locality = next((address[field] for field in ("village", "town", "city", "municipality", "county", "state", "region") if address.get(field)), "")
            return {**raw_record, "place_query": f"{title}, {locality}, {country}", "locality": locality}
        # The identity/cache/radius unit cases use tiny batches to isolate their
        # specific rule. Separate count tests below exercise the real min/max;
        # test_grounded_planner also uses the real minimum end to end.
        count_context = nullcontext() if enforce_count else patch.object(grounding, "candidate_minimum", return_value=0)
        with count_context, patch.object(grounding, "generate_json", return_value=GenerationResult(
            raw if raw is not None else json.dumps(payload, ensure_ascii=False), 11, 13,
        )) as generate, patch.object(places, "geocode", side_effect=lookup), patch.object(places, "geocode_nearby_named", side_effect=nearby):
            result = grounding.ground_region_candidates(request(days), scope(), max_http_attempts=max_http_attempts)
        return result, generate, lookup

    def test_schema_caps_and_independent_base_keys(self):
        for days, maximum in ((1, 6), (2, 8), (7, 18), (15, 18)):
            schema = grounding.build_candidate_schema(request(days))
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(schema["properties"]["pois"]["maxItems"], maximum)
            self.assertEqual(schema["properties"]["pois"]["minItems"], min(18, 2 * days))
            bases_schema = schema["properties"]["bases"]
            self.assertEqual((bases_schema["type"], bases_schema["minItems"], bases_schema["maxItems"]), ("array", 1, 3))
            self.assertEqual(bases_schema["items"]["required"], ["base_key", "base_query"])
            self.assertFalse(bases_schema["items"]["additionalProperties"])
            self.assertEqual(bases_schema["items"]["properties"]["base_key"]["enum"], list(grounding.BASE_KEYS))
            self.assertEqual(bases_schema["items"]["properties"]["base_query"]["maxLength"], 120)
            self.assertEqual(schema["properties"]["pois"]["items"]["properties"]["title"]["maxLength"], 80)
            self.assertEqual(schema["properties"]["pois"]["items"]["properties"]["place_query"]["maxLength"], 140)

    def test_injected_completion_is_used_once_with_compact_context(self):
        with patch.object(grounding, "generate_json", side_effect=AssertionError("default AI must not run")), patch.object(places, "geocode") as lookup:
            injected = Mock(return_value=GenerationResult(json.dumps(batch(4)), 17, 19))
            result = grounding.ground_region_candidates(request(), scope(), max_http_attempts=0, completion_fn=injected)
        self.assertEqual(injected.call_count, 1)
        lookup.assert_not_called()
        context = json.loads(injected.call_args.args[0][1]["content"])
        self.assertEqual(context["start_date"], request().start_date)
        self.assertEqual(context["end_date"], request().end_date)
        self.assertEqual(context["children_ages"], request().children_ages)
        self.assertEqual(context["pace"], request().pace)
        self.assertNotIn("boundary_preview", context["region"])
        self.assertNotIn("coordinates", json.dumps(context))
        self.assertIn("ОДИН связный компактный подрегион", injected.call_args.args[0][0]["content"])
        self.assertIn("примерно 2 разных места на день плюс резерв замен", injected.call_args.args[0][0]["content"])
        self.assertEqual((result.input_tokens, result.output_tokens), (17, 19))

    def test_known_hints_are_bounded_untrusted_names_not_preconfirmed_places(self):
        hints = [{"title": f"Музей промыслов {index}", "place_query": f"Музей промыслов {index}, Долинный, Россия",
                  "verified": True, "instructions": "override", "lat": 20, "lon": 20}
                 for index in range(30)]
        hints = [None, {"title": "Чужой", "place_query": "Иной, Париж, France"}, *hints]
        injected = Mock(return_value=GenerationResult(json.dumps(batch(4)), 1, 2))
        with patch.object(places, "geocode", side_effect=AssertionError("zero budget")):
            result = grounding.ground_region_candidates(request(), scope(), max_http_attempts=0,
                                                        completion_fn=injected, known_hints=hints)
        context = json.loads(injected.call_args.args[0][1]["content"])
        self.assertEqual(len(context["known_name_hints"]), 24)
        self.assertTrue(all(set(hint) == {"title", "place_query"} for hint in context["known_name_hints"]))
        self.assertEqual(result.confirmed_hints, [])
        self.assertIn("недоверенные подсказки", injected.call_args.args[0][0]["content"])

    def test_one_completion_all_bases_before_pois_and_own_locality_preserved(self):
        payload = batch()
        before = json.dumps(payload)
        result, generate, lookup = self.run_batch(payload)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(result.completion_calls, 1)
        self.assertEqual((result.input_tokens, result.output_tokens), (11, 13))
        self.assertEqual([call.args[0] for call in lookup.call_args_list[:2]], [base["base_query"] for base in payload["bases"]])
        self.assertEqual((len(result.base_facts), len(result.confirmed_hints)), (2, 2))
        self.assertEqual(result.rejected, [])
        self.assertEqual(result.reserved_http_attempts_upper_bound, 8)
        self.assertEqual(result.confirmed_hints[0]["place_query"], payload["pois"][0]["place_query"])
        self.assertNotIn("duration_min", result.confirmed_hints[0])
        self.assertNotIn("cost", result.confirmed_hints[0])
        self.assertEqual(json.dumps(payload), before)

    def test_short_trip_can_use_one_base_without_inventing_another(self):
        payload = batch(1)
        payload["bases"].pop()
        result, generate, lookup = self.run_batch(payload, days=1)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(lookup.call_count, 2)
        self.assertEqual(len(result.base_facts), 1)
        self.assertEqual(len(result.confirmed_hints), 1)
        self.assertEqual(result.problems, [])

    def test_every_schema_object_has_only_required_fields(self):
        def check(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertFalse(node["additionalProperties"])
                    self.assertEqual(set(node["required"]), set(node["properties"]))
                for child in node.values():
                    check(child)
            elif isinstance(node, list):
                for child in node:
                    check(child)
        check(grounding.build_candidate_schema(request()))

    def test_invalid_base_array_is_rejected_atomically_before_map(self):
        original = batch()
        first, second = original["bases"]
        cases = [
            [], [second], [first, first],
            [first, second, {"base_key": "base_3", "base_query": "Третий, Россия"}, first],
            [{"base_key": "base_4", "base_query": "Четвёртый, Россия"}],
            [{"base_query": first["base_query"]}],
            [{**first, "verified": True}],
            {"base_1": {"base_query": first["base_query"]}},
        ]
        for bases in cases:
            with self.subTest(bases=bases):
                result, generate, lookup = self.run_batch({**original, "bases": bases})
                self.assertEqual(generate.call_count, 1)
                lookup.assert_not_called()
                self.assertTrue(result.problems)
                self.assertEqual(result.base_facts, [])
        unknown_ref = batch()
        unknown_ref["pois"][0]["base_key"] = "base_3"
        result, _, lookup = self.run_batch(unknown_ref)
        lookup.assert_not_called()
        self.assertTrue(result.problems)

    def test_array_order_does_not_change_base_first_verification(self):
        payload = batch(1)
        payload["bases"].reverse()
        result, _, lookup = self.run_batch(payload)
        self.assertEqual([fact["base_key"] for fact in result.base_facts], ["base_1", "base_2"])
        self.assertTrue(lookup.call_args_list[0].args[0].startswith("Северный,"))
        self.assertTrue(lookup.call_args_list[1].args[0].startswith("Южный,"))

    def test_invalid_base_is_not_replaced_by_neighbor_city_and_its_pois_are_not_queried(self):
        def lookup(query, **_):
            return record(query, poi_type="city", lat=20, lon=20)
        result, _, calls = self.run_batch(geocoder=lookup)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual(result.base_facts, [])
        self.assertEqual(result.confirmed_hints, [])
        self.assertEqual([entry["reason"] for entry in result.rejected], ["outside_region", "outside_region", "base_unconfirmed", "base_unconfirmed"])

    def test_another_regions_cached_base_cannot_ground_local_pois(self):
        outside = record(batch()["bases"][0]["base_query"], lat=20, lon=20, poi_type="village")
        self.cache[outside["name"].lower()] = outside
        result, _, lookup = self.run_batch()
        self.assertEqual(lookup.call_count, 1)
        self.assertEqual(len(result.base_facts), 1)
        self.assertEqual(result.base_facts[0]["base_key"], "base_2")
        self.assertEqual(result.confirmed_hints, [])

    def test_poi_inside_bbox_but_outside_full_polygon_is_rejected(self):
        area = scope()
        area.geometry["coordinates"] = [[[100, 50], [103, 50], [100, 53], [100, 50]]]
        def lookup(query, **_):
            if query.startswith(("Северный,", "Южный,")):
                return record(query, poi_type="village")
            return record(query, lat=52.5, lon=102.5)
        with patch(__name__ + ".scope", return_value=area):
            result, _, _ = self.run_batch(batch(1), geocoder=lookup)
        self.assertEqual(len(result.base_facts), 2)
        self.assertEqual(result.confirmed_hints, [])
        self.assertEqual(result.rejected[-1]["reason"], "outside_region")

    def test_exact_32_http_reservation_ceiling_stops_without_failing_confirmed_subset(self):
        result, generate, lookup = self.run_batch(batch(18), days=7, max_http_attempts=32)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(lookup.call_count, 16)
        self.assertEqual(result.reserved_http_attempts_upper_bound, 32)
        self.assertEqual(len(result.confirmed_hints), 14)
        self.assertEqual([entry["reason"] for entry in result.rejected], ["http_budget_exhausted"] * 4)
        self.assertTrue(result.incomplete)
        self.assertLessEqual(result.verification_queries, result.query_limit)

    def test_14_day_cold_cache_pool_fits_three_bases_and_18_pois_in_42_http_reserve(self):
        payload = batch(18)
        payload["bases"].append({"base_key": "base_3", "base_query": "Восточный, Тестовый край, Россия"})
        base_queries = {base["base_query"] for base in payload["bases"]}
        def lookup(query, **kwargs):
            self.assertEqual(kwargs, {"max_variants": 1})
            if query in base_queries:
                return record(query, poi_type="village")
            index = int(query.split(",")[0].split()[-1])
            return record(query, lon=101.05 + index / 100)
        result, generate, calls = self.run_batch(payload, days=14, geocoder=lookup)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual((len(result.base_facts), len(result.confirmed_hints)), (3, 18))
        self.assertEqual((result.verification_queries, calls.call_count), (21, 21))
        self.assertEqual((result.reserved_http_attempts_upper_bound, result.http_attempt_limit), (42, 42))
        self.assertEqual(result.rejected, [])
        self.assertFalse(result.incomplete)

    def test_short_trip_reserves_only_request_specific_upper_bound(self):
        for days, expected in ((1, 18), (2, 22), (3, 26), (7, 42), (14, 42)):
            with self.subTest(days=days):
                result, _, _ = self.run_batch(batch(1), days=days)
                self.assertEqual(result.http_attempt_limit, expected)
                self.assertEqual(result.query_limit * 2, expected)

    def test_cached_bases_and_pois_need_no_http_even_with_zero_budget(self):
        payload = batch(1)
        for base in payload["bases"]:
            self.cache[base["base_query"].lower()] = record(base["base_query"], poi_type="village", lat="51.0", lon="101.0")
        query = payload["pois"][0]["place_query"]
        self.cache[query.lower()] = record(query, lon=101.1)
        result, _, lookup = self.run_batch(payload, max_http_attempts=0)
        lookup.assert_not_called()
        self.assertEqual(len(result.confirmed_hints), 1)
        self.assertEqual(result.reserved_http_attempts_upper_bound, 0)

    def test_limited_search_is_incomplete_not_a_negative_fact_and_does_not_retry(self):
        def lookup(query, **_):
            if query.startswith(("Северный,", "Южный,")):
                return record(query, poi_type="village")
            raise places.GeocodeIncomplete(query, reason="not_found")
        result, generate, calls = self.run_batch(geocoder=lookup)
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(calls.call_count, 4)
        self.assertEqual([entry["reason"] for entry in result.rejected], ["limited_search_incomplete"] * 2)
        self.assertTrue(result.incomplete)
        self.assertEqual(self.cache, {})

    def test_real_geocode_path_never_exceeds_reserved_http_attempts(self):
        http_calls = []
        def http_get(url, *, params, **kwargs):
            query = params["q"]
            http_calls.append(query)
            if len(http_calls) % 2:
                raise httpx.ReadTimeout("offline transient")
            data = record(query, poi_type="village" if query.startswith(("Северный,", "Южный,")) else "museum")
            data["display_name"] = data.pop("name")
            return httpx.Response(200, json=[data], request=httpx.Request("GET", "https://example.invalid"))
        with patch.object(grounding, "generate_json", return_value=GenerationResult(json.dumps(batch(4)))), patch.object(places.httpx, "get", side_effect=http_get), patch.object(places, "_wait_for_turn"), patch.object(places.time, "sleep"), patch.object(places, "_save_cache"):
            result = grounding.ground_region_candidates(request(), scope(), max_http_attempts=6)
        self.assertEqual(len(http_calls), 6)
        self.assertEqual(result.reserved_http_attempts_upper_bound, 6)
        self.assertEqual(result.geocode_calls, 3)
        self.assertEqual(len(result.confirmed_hints), 1)
        self.assertEqual(result.rejected[-1]["reason"], "http_budget_exhausted")

    def test_named_local_natural_types_pass_but_whole_island_or_lake_does_not(self):
        for poi_type in ("cape", "ridge", "rock", "stone", "observatory", "island", "lake"):
            def lookup(query, **_):
                return record(query, poi_type="village" if query.startswith(("Северный,", "Южный,")) else poi_type)
            with self.subTest(poi_type=poi_type):
                result, _, _ = self.run_batch(batch(1), geocoder=lookup)
                self.assertEqual(len(result.confirmed_hints), 0 if poi_type in {"island", "lake"} else 1)

    def test_oversized_strings_are_rejected_before_map_not_silently_truncated(self):
        for kind, field, value in (("base", "base_query", "Я" * 121), ("poi", "title", "Я" * 81), ("poi", "place_query", "Я" * 141)):
            payload = batch()
            target = payload["bases"][0] if kind == "base" else payload["pois"][0]
            target[field] = value
            with self.subTest(field=field):
                result, _, lookup = self.run_batch(payload)
                lookup.assert_not_called()
                self.assertTrue(result.problems)

    def test_generic_duplicate_locality_and_model_verified_metadata_are_not_trusted(self):
        payload = batch(1)
        payload["pois"] += [dict(payload["pois"][0]), {**payload["pois"][0], "locality": "Северный"},
                            {"title": "Музей", "place_query": "Музей, Долинный, Россия", "locality": "Долинный", "base_key": "base_1"}]
        result, _, lookup = self.run_batch(payload)
        self.assertEqual(lookup.call_count, 3)
        self.assertEqual([entry["reason"] for entry in result.rejected], ["duplicate_query", "duplicate_query", "generic_place"])
        payload["pois"][0]["verified"] = True
        result, _, lookup = self.run_batch(payload)
        lookup.assert_not_called()
        self.assertTrue(result.problems)

    def test_extra_administrative_address_parts_are_canonicalized_before_lookup(self):
        payload = batch(1)
        original = "Музей промыслов номер 0, Долинный, Тестовый район, Тестовый край, Россия"
        payload["pois"][0]["place_query"] = original
        canonical = "Музей промыслов номер 0, Долинный, Россия"
        result, _, lookup = self.run_batch(payload)
        self.assertEqual(lookup.call_args_list[-1].args[0], canonical)
        self.assertEqual(result.confirmed_hints[0]["place_query"], canonical)
        self.assertEqual(result.confirmed_hints[0]["original_proposal_query"], original)
        self.assertEqual(result.rejected, [])
        from grounded_schedule import build_grounded_messages
        schedule_context = build_grounded_messages(request(1), scope(), result, 3)[1]["content"]
        self.assertNotIn("original_proposal_query", schedule_context)
        self.assertNotIn(original, schedule_context)

    def test_canonicalization_requires_exact_title_and_country_components(self):
        for overrides in (
            {"title": "Другой музей"},
            {"place_query": "Музей промыслов номер 0, Долинный, Тестовый край, France"},
        ):
            payload = batch(1)
            payload["pois"][0]["place_query"] = "Музей промыслов номер 0, Долинный, Тестовый край, Россия"
            payload["pois"][0].update(overrides)
            with self.subTest(overrides=overrides):
                result, _, lookup = self.run_batch(payload)
                self.assertEqual(lookup.call_count, 2)  # Only bases, never a guessed POI query.
                self.assertEqual(result.confirmed_hints, [])
                self.assertEqual(result.rejected[-1]["reason"], "invalid_poi_context")

    def test_canonicalized_proposal_still_requires_identity_and_region_match(self):
        payload = batch(1)
        payload["pois"][0]["place_query"] = "Музей промыслов номер 0, Долинный, Тестовый край, Россия"
        for failure in ("identity_mismatch", "outside_region"):
            def lookup(query, **_):
                if query.startswith(("Северный,", "Южный,")):
                    return record(query, poi_type="village")
                if failure == "identity_mismatch":
                    return record("Совсем иной объект, Другой посёлок, Россия")
                return record(query, lat=20, lon=20)
            with self.subTest(failure=failure):
                result, _, _ = self.run_batch(payload, geocoder=lookup)
                self.assertEqual(result.confirmed_hints, [])
                self.assertEqual(result.rejected[-1]["reason"], failure)

    def test_canonical_query_deduplicates_long_and_short_proposals_before_map(self):
        payload = batch(1)
        payload["pois"].append({**payload["pois"][0], "place_query": "Музей промыслов номер 0, Долинный, Тестовый край, Россия"})
        result, _, lookup = self.run_batch(payload)
        self.assertEqual(lookup.call_count, 3)
        self.assertEqual(len(result.confirmed_hints), 1)
        self.assertEqual(result.rejected[-1]["reason"], "duplicate_query")

    def test_nearby_county_replaces_unproven_model_village_only_after_map_match(self):
        payload = batch(1)
        title = payload["pois"][0]["title"]
        actual = "Тестовый муниципальный округ"
        def lookup(query, **_):
            if query.startswith(("Северный,", "Южный,")):
                return record(query, poi_type="village")
            return record(f"{title}, {actual}, Россия", poi_type="peak",
                          address={"county": actual, "country": "Россия", "country_code": "ru"})
        result, _, _ = self.run_batch(payload, geocoder=lookup)
        self.assertEqual(result.rejected, [])
        self.assertEqual(result.confirmed_hints[0]["locality"], actual)
        self.assertEqual(result.confirmed_hints[0]["place_query"], f"{title}, {actual}, Россия")
        self.assertEqual(result.confirmed_hints[0]["original_proposal_query"], payload["pois"][0]["place_query"])

    def test_model_locality_mismatch_does_not_block_actual_map_county(self):
        payload = batch(2)
        payload["pois"][0].update(title="Камень Черского",
                                 place_query="Камень Черского, Иркутский муниципальный округ, Россия",
                                 locality="Листвянка")
        def lookup(query, **_):
            if query.startswith(("Северный,", "Южный,")):
                return record(query, poi_type="village", address={"village": query.split(",")[0], "county": "Иркутский муниципальный округ", "country": "Россия", "country_code": "ru"})
            if query.startswith("Камень Черского,"):
                return record("Камень Черского, Иркутский муниципальный округ, Россия", poi_type="peak",
                              address={"county": "Иркутский муниципальный округ", "country": "Россия", "country_code": "ru"})
            return record(query, lon=101.1, address={"village": "Долинный", "county": "Иркутский муниципальный округ", "country": "Россия", "country_code": "ru"})
        result, _, _ = self.run_batch(payload, days=1, geocoder=lookup, enforce_count=True)
        self.assertEqual(result.rejected, [])
        accepted = result.confirmed_hints[0]
        self.assertEqual(accepted["place_query"], "Камень Черского, Иркутский муниципальный округ, Россия")
        self.assertEqual(accepted["locality"], "Иркутский муниципальный округ")
        self.assertEqual(accepted["original_proposal_locality"], "Листвянка")

    def test_source_catalog_hints_are_prompt_only_not_confirmation(self):
        hint = {"title": "Музей промыслов", "place_query": "Музей промыслов, Долинный, Россия",
                "locality": "Долинный", "source_url": "https://example.org/official-name"}
        injected = Mock(return_value=GenerationResult(json.dumps(batch(4)), 1, 2))
        with (patch.object(grounding, "destination_name_hints", return_value=[hint]) as catalogue,
              patch.object(places, "geocode", side_effect=AssertionError("zero map budget"))):
            result = grounding.ground_region_candidates(request(), scope(), max_http_attempts=0, completion_fn=injected)
        catalogue.assert_called_once_with(request().destination)
        context = json.loads(injected.call_args.args[0][1]["content"])
        self.assertEqual(context["source_name_hints"], [hint])
        self.assertEqual(result.confirmed_hints, [])
        self.assertIn("не доказывает координаты", injected.call_args.args[0][0]["content"])

    def test_nearby_invalid_query_is_rejected_without_crashing_batch(self):
        result = grounding.GroundingResult(query_limit=1)
        with patch.object(places, "geocode_nearby_named", side_effect=ValueError("invalid named query")):
            record_value, reason = grounding._lookup_nearby_poi("Музей [категория]", "Россия", {"lat": 51, "lon": 101}, result)
        self.assertIsNone(record_value)
        self.assertEqual(reason, "invalid_query")

    def test_malformed_oversized_missing_base_and_duplicate_json_are_atomic_without_map(self):
        cases = [batch(9), {**batch(), "bases": []}, {**batch(), "foreign": True}]
        for payload in cases:
            result, generate, lookup = self.run_batch(payload)
            lookup.assert_not_called()
            self.assertEqual(generate.call_count, 1)
            self.assertTrue(result.problems)
            self.assertEqual(result.input_tokens + result.output_tokens, 24)
        raw = json.dumps(batch()).replace('"base_key": "base_1"', '"base_key": "base_2", "base_key": "base_1"', 1)
        result, _, lookup = self.run_batch(raw=raw)
        lookup.assert_not_called()
        self.assertTrue(result.problems)

    def test_network_failure_and_ai_failure_are_not_automatically_retried(self):
        result, _, lookup = self.run_batch(geocoder=lambda *_a, **_k: (_ for _ in ()).throw(places.GeocodeUnavailable("offline")))
        self.assertEqual(lookup.call_count, 2)
        self.assertTrue(result.incomplete)
        with patch.object(grounding, "generate_json", side_effect=YandexAIError("offline")) as generate, patch.object(places, "geocode") as lookup:
            result = grounding.ground_region_candidates(request(), scope())
        self.assertEqual(generate.call_count, 1)
        lookup.assert_not_called()
        self.assertTrue(result.problems)

    def test_wrong_country_unknown_base_type_and_poi_outside_base_radius_are_rejected(self):
        payload = batch(1)
        payload["bases"][1]["base_query"] = "Южный, France"
        def lookup(query, **_):
            if query.startswith("Северный,"):
                return record(query, poi_type="village")
            return record(query, lon=102.9)
        result, _, calls = self.run_batch(payload, geocoder=lookup)
        self.assertEqual(calls.call_count, 2)
        self.assertEqual([entry["reason"] for entry in result.rejected], ["invalid_base_context", "outside_base_radius"])
        result, _, calls = self.run_batch(geocoder=lambda query, **_: record(query, poi_type="island"))
        self.assertEqual(calls.call_count, 2)
        self.assertEqual(result.base_facts, [])

    def test_invalid_scope_budget_or_city_never_spends_ai_call(self):
        city = request().model_copy(update={"destination_kind": "city"})
        with patch.object(grounding, "generate_json") as generate:
            for candidate, area, budget in ((city, scope(), 42), (request(), RegionScope("Area", "Россия", (0, 1, 0, 1)), 42), (request(), scope(), 43), (request(), scope(), True)):
                with self.assertRaises(ValueError):
                    grounding.ground_region_candidates(candidate, area, max_http_attempts=budget)
            generate.assert_not_called()

    def test_real_candidate_minimum_and_maximum_batches_are_accepted(self):
        for days in (1, 2, 7, 14):
            for count in {min(18, 2 * days), min(18, 2 * days + 4)}:
                with self.subTest(days=days, count=count):
                    result, generate, lookup = self.run_batch(batch(count), days=days, enforce_count=True)
                    self.assertEqual(result.problems, [])
                    self.assertEqual(len(result.confirmed_hints), count)
                    self.assertEqual(lookup.call_count, count + 2)
                    generate.assert_called_once()

    def test_real_out_of_range_candidate_counts_stop_before_map(self):
        for days in (1, 2, 7, 14):
            for count in (min(18, 2 * days) - 1, min(18, 2 * days + 4) + 1):
                with self.subTest(days=days, count=count):
                    result, generate, lookup = self.run_batch(batch(count), days=days, enforce_count=True)
                    self.assertTrue(result.problems)
                    self.assertEqual((result.base_facts, result.confirmed_hints), ([], []))
                    self.assertEqual((result.completion_calls, result.input_tokens, result.output_tokens), (1, 11, 13))
                    generate.assert_called_once()
                    lookup.assert_not_called()


if __name__ == "__main__":
    unittest.main()
