"""Source-backed base selection, with synthetic records and no external IO."""

from contextlib import ExitStack
from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import Mock, patch

import candidate_grounding as candidates
import destination_name_hints as catalogue
import places
import visit_facts
from test_candidate_grounding import request, scope
from test_visit_facts import card_for
from yandex_ai import GenerationResult


def source(locality, number=1, **updates):
    title = f"Музей ремёсел {locality} {number}"
    return {"title": title, "place_query": f"{title}, {locality}, Россия",
            "locality": locality, "source_url": "https://example.invalid/museum", **updates}


def model_batch(locality="Чужой", count=4):
    return {"bases": [{"base_key": "base_1", "base_query": f"{locality}, Россия"}],
            "pois": [{"title": f"Музей модели {index}",
                      "place_query": f"Музей модели {index}, {locality}, Россия",
                      "locality": locality, "base_key": "base_1"} for index in range(count)]}


def roster(hints, payload=None, trip=None):
    return candidates._source_candidate_roster(
        trip or request(), scope(), hints, candidates._Batch.model_validate(payload or model_batch()),
    )


def card_request(days=2, **updates):
    return request(days).model_copy(update={"currency": "RUB", "interest_categories": ["main_sights"], **updates})


class SourceRosterTests(unittest.TestCase):
    def setUp(self):
        self.cards = []
        loader = patch.object(visit_facts, "load_visit_cards", side_effect=lambda *_args, **_kwargs: deepcopy(self.cards))
        loader.start()
        self.addCleanup(loader.stop)

    def add_card(self, hint, **updates):
        fields = dict(id=hint["title"], place_queries=[hint["place_query"]],
                      metadata_checked_at="2099-05-21", interest_categories=["main_sights"])
        fields.update(updates)
        card = card_for(**fields)
        self.cards.append(card)
        return card

    def run_grounding(self, hints, *, payload=None, days=2, stay_area="", cache=None,
                      miss=False, wrong_address=False, max_http_attempts=42, currency="EUR", poi_records=None):
        payload = payload or model_batch(count=min(18, 2 * days))
        addresses = {}
        points = {}
        for name in [*(hint["locality"] for hint in hints), "Чужой", "Настоящая база"]:
            if name not in points:
                points[name] = (51.0, 101.0 + len(points) * 0.05)
                addresses[name] = {"village": name, "county": f"Округ {name}", "country_code": "ru"}

        def base_lookup(query, **kwargs):
            self.assertEqual(kwargs, {"max_variants": 1})
            name = query.split(",")[0]
            lat, lon = points[name]
            return {"name": query, "lat": lat, "lon": lon, "type": "village", "category": "place",
                    "address": addresses[name]}

        def poi_lookup(title, country, base_lat, base_lon, radius, *, place_filter):
            self.assertEqual((country, radius), ("Россия", 80))
            if poi_records is not None and title in poi_records:
                return next((deepcopy(record) for record in poi_records[title] if place_filter(record)), None)
            if miss:
                return None
            hint = next((item for item in hints if item["title"] == title), None)
            locality = hint["locality"] if hint else payload["pois"][0]["locality"]
            number = int(title.split()[-1])
            lat, lon = points[locality]
            address = addresses[locality] if not wrong_address else addresses["Чужой"]
            record = {"name": f"{title}, {locality}, Россия", "lat": lat + .002 * number, "lon": lon + .01,
                      "type": "museum", "category": "tourism", "address": address,
                      "place_query": f"{title}, {locality}, Россия", "locality": locality}
            return record if place_filter(record) else None

        completion = Mock(return_value=GenerationResult(json.dumps(payload, ensure_ascii=False), 11, 13))
        with ExitStack() as stack:
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("No network")))
            stack.enter_context(patch.object(places.httpx, "get", side_effect=AssertionError("No HTTP")))
            stack.enter_context(patch.object(places, "_cache", deepcopy(cache or {})))
            stack.enter_context(patch.object(places, "_save_cache", return_value=None))
            stack.enter_context(patch.object(candidates, "destination_name_hints", return_value=deepcopy(hints)))
            bases = stack.enter_context(patch.object(places, "geocode", side_effect=base_lookup))
            pois = stack.enter_context(patch.object(places, "geocode_nearby_named", side_effect=poi_lookup))
            result = candidates.ground_region_candidates(
                request(days).model_copy(update={"stay_area": stay_area, "currency": currency}), scope(), completion_fn=completion,
                max_http_attempts=max_http_attempts,
            )
        return result, completion, bases, pois

    def test_source_known_title_cannot_be_rebound_by_ai(self):
        hints = [source("Слюдянка", 1), source("Слюдянка", 2)]
        payload = model_batch()
        payload["pois"][0] = {"title": hints[0]["title"], "place_query": f"{hints[0]['title']}, Чужой, Россия",
                              "locality": "Чужой", "base_key": "base_1"}
        before = deepcopy((hints, payload))
        result, completion, bases, pois = self.run_grounding(hints, payload=payload)
        self.assertTrue(result.source_roster_used)
        self.assertEqual([call.args[0] for call in bases.call_args_list], ["Слюдянка, Россия"])
        self.assertEqual([fact["place_query"] for fact in result.confirmed_hints], [hint["place_query"] for hint in hints])
        self.assertTrue(all(fact["base_key"] == "base_1" for fact in result.confirmed_hints))
        self.assertEqual((completion.call_count, result.completion_calls, pois.call_count), (1, 1, 2))
        self.assertEqual((result.input_tokens, result.output_tokens), (11, 13))
        self.assertEqual((hints, payload), before)

    def test_source_omitted_by_model_still_receives_lookup(self):
        hints = [source("Долинный", 1), source("Долинный", 2)]
        result, _, _, lookup = self.run_grounding(hints)
        self.assertEqual([call.args[0] for call in lookup.call_args_list], [item["title"] for item in hints])
        self.assertEqual(len(result.confirmed_hints), 2)

    def test_explicit_source_base_overrides_locality_but_not_map_checks(self):
        hints = [source("Долинный", 1, base_query="Настоящая база, Россия")]
        selected, _, _ = roster(hints)
        self.assertEqual(selected.bases[0].base_query, "Настоящая база, Россия")
        result, _, bases, _ = self.run_grounding(hints)
        self.assertEqual(bases.call_args.args[0], "Настоящая база, Россия")
        # The synthetic locality has another county: a source hint is not proof.
        self.assertEqual(result.confirmed_hints, [])

    def test_default_base_country_comes_from_scope_and_foreign_hint_is_ignored(self):
        hints = [source("Долинный"), source("Другой", place_query="Музей ремёсел Другой 1, Другой, France")]
        selected, _, _ = roster(hints)
        self.assertEqual([base.base_query for base in selected.bases], ["Долинный, Россия"])
        self.assertEqual(len(selected.pois), 1)

    def test_wrong_country_explicit_base_is_not_used_as_a_fallback(self):
        self.assertIsNone(roster([source("Долинный", base_query="Paris, France")]))

    def test_three_clusters_rank_by_unique_source_count_then_catalogue_order(self):
        hints = [source(name, number) for name, count in (("Альфа", 2), ("Бета", 4), ("Гамма", 3), ("Дельта", 1))
                 for number in range(1, count + 1)]
        hints += [deepcopy(hints[-1])] * 5
        selected, _, omitted = roster(hints, trip=request(7))
        self.assertEqual([base.base_query for base in selected.bases], ["Бета, Россия", "Гамма, Россия", "Альфа, Россия"])
        self.assertEqual(omitted, ["Дельта, Россия"])
        equal = roster([source(name) for name in ("Гамма", "Альфа", "Бета", "Дельта")])[0]
        self.assertEqual([base.base_query for base in equal.bases], ["Гамма, Россия", "Альфа, Россия", "Бета, Россия"])

    def test_date_coverage_keeps_the_only_last_day_cluster_with_same_three_base_cap(self):
        from types import SimpleNamespace
        from regional_skeleton import TravelFact, build_regional_skeleton

        trip = card_request(7)
        dates = [(date.fromisoformat(trip.start_date) + timedelta(days=i)).isoformat() for i in range(7)]
        hints = [source(name, number) for name, count in (("Альфа", 3), ("Бета", 3), ("Гамма", 3), ("Дельта", 1))
                 for number in range(1, count + 1)]
        for hint in hints:
            self.add_card(hint, duration_min=240, hours=[dict(
                dates=dates[-1:] if hint["locality"] == "Дельта" else dates[:6], start="08:00", end="21:00")])

        # Same budget, 240 minutes/day, no repeated POIs: an alternative subset
        # of three bases is genuinely schedulable in this synthetic road graph.
        alternative = [hint for hint in hints if hint["locality"] != "Гамма"]
        keys = {name: f"base_{i}" for i, name in enumerate(("Альфа", "Бета", "Дельта"), 1)}
        pool = SimpleNamespace(base_facts=[dict(base_key=key, base_query=f"{name}, Россия", lat=51., lon=101.)
                                          for name, key in keys.items()], confirmed_hints=[])
        facts = []
        for i, hint in enumerate(alternative):
            key = keys[hint["locality"]]
            pool.confirmed_hints.append({**hint, "base_key": key, "allowed_base_keys": [key],
                                         "lat": 51. + i * .001, "lon": 101., "osm_type": "node", "osm_id": i + 1})
            card = next(card for card in self.cards if hint["place_query"] in card["place_queries"])
            facts.append(visit_facts.visit_fact_from_card(trip, card, hint["place_query"]))
        nodes = [*keys.values(), *(fact.visit_id for fact in facts)]
        graph = [TravelFact(a, b, 5, "walking", "routing", source="offline-fixture",
                            group_cost=0, price_source="walking-no-fare") for a in nodes for b in nodes if a != b]
        feasible = build_regional_skeleton(trip, pool, facts, travel_facts=graph)
        self.assertEqual(feasible.status, "ready", feasible)
        self.assertEqual([day.visit_minutes for day in feasible.days], [240] * 7)
        self.assertEqual(len({visit for day in feasible.days for visit in day.visit_ids}), 7)

        selected, _, omitted = roster(hints, trip=trip)
        self.assertIn("Дельта, Россия", [base.base_query for base in selected.bases])
        self.assertEqual(omitted, ["Гамма, Россия"])
        self.assertLessEqual(len(selected.bases), 3)
        self.assertLessEqual(len(selected.pois), 18)

    def test_equal_date_coverage_keeps_previous_base_and_poi_order(self):
        hints = [source(name) for name in ("Гамма", "Альфа", "Бета", "Дельта")]
        for hint in hints:
            self.add_card(hint)
        selected, _, omitted = roster(hints, trip=card_request(2))
        self.assertEqual([base.base_query for base in selected.bases], ["Гамма, Россия", "Альфа, Россия", "Бета, Россия"])
        self.assertEqual([poi.place_query for poi in selected.pois], [hint["place_query"] for hint in hints[:3]])
        self.assertEqual(omitted, ["Дельта, Россия"])

    def test_eighteen_poi_cap_preserves_last_date_witness_before_trimming(self):
        trip = card_request(7)
        dates = [(date.fromisoformat(trip.start_date) + timedelta(days=i)).isoformat() for i in range(7)]
        hints = [source(name, number) for name in ("Альфа", "Бета", "Гамма") for number in range(1, 9)]
        rare = hints[7]
        for hint in hints:
            self.add_card(hint, duration_min=60 if hint is rare else 240, hours=[dict(
                dates=dates[-1:] if hint is rare else dates[:6], start="08:00", end="21:00")])
        selected, _, _ = roster(hints, trip=trip)
        self.assertEqual(len(selected.pois), 18)
        self.assertEqual(len(selected.bases), 3)
        self.assertIn(rare["place_query"], {poi.place_query for poi in selected.pois})
        options = {}
        candidates._source_visit_priorities(trip, hints, date_options=options)
        matched = candidates._source_date_matching([poi.model_dump() for poi in selected.pois], options)
        self.assertEqual(len(matched), 7)

    def test_date_coverage_counts_card_aliases_once(self):
        first, alias = source("Альфа"), source("Бета")
        options = {hint["place_query"]: ("one_card", frozenset(("2099-06-01", "2099-06-02")))
                   for hint in (first, alias)}
        self.assertEqual(len(candidates._source_date_matching([first, alias], options)), 1)

    def test_warm_usable_cache_cannot_displace_uncached_last_date_witness(self):
        trip = card_request(7)
        dates = [(date.fromisoformat(trip.start_date) + timedelta(days=i)).isoformat() for i in range(7)]
        hints = [source(name, number) for name in ("Альфа", "Бета", "Гамма") for number in range(1, 9)]
        rare = hints[7]
        cache = {}
        for i, hint in enumerate(hints):
            self.add_card(hint, duration_min=60 if hint is rare else 240, hours=[dict(
                dates=dates[-1:] if hint is rare else dates[:6], start="08:00", end="21:00")])
            if hint is not rare:
                cache[hint["place_query"].lower()] = {
                    "name": hint["place_query"], "lat": 51. + (i + 1) * .002, "lon": 101.01 + (i // 8) * .05,
                    "type": "museum", "category": "tourism", "osm_type": "node", "osm_id": i + 1,
                    "address": {"village": hint["locality"], "county": f"Округ {hint['locality']}", "country_code": "ru"},
                }
        self.assertEqual(len(cache), 23)
        result, completion, bases, lookup = self.run_grounding(hints, days=7, currency="RUB", cache=cache)
        self.assertIn(rare["place_query"], {hint["place_query"] for hint in result.confirmed_hints})
        options = {}
        candidates._source_visit_priorities(trip, hints, date_options=options)
        self.assertEqual(len(candidates._source_date_matching(result.confirmed_hints, options)), 7)
        self.assertEqual(len(result.confirmed_hints), 18)
        self.assertEqual((completion.call_count, bases.call_count, lookup.call_count), (1, 3, 1))
        self.assertLessEqual(result.verification_queries, 21)
        self.assertLessEqual(result.reserved_http_attempts_upper_bound, 42)

    def test_even_large_cardless_payload_is_bounded_to_twenty_four_groups(self):
        hints = [source(f"Кластер {i}") for i in range(40)]
        with patch.object(candidates, "_source_date_matching", wraps=candidates._source_date_matching) as matching:
            selected, _, omitted = roster(hints, trip=card_request(7))
        self.assertEqual(len(omitted), 21)
        self.assertEqual(len(selected.bases), 3)
        # Three matching calls per subset and two for final selection.
        self.assertLessEqual(matching.call_count, 3 * 2024 + 2)

    def test_round_robin_and_model_supplement_share_eighteen_poi_cap(self):
        hints = [source(name, number) for name in ("Альфа", "Бета", "Гамма") for number in range(1, 9)]
        selected, _, _ = roster(hints, model_batch(count=14), request(7))
        self.assertEqual(len(selected.pois), 18)
        self.assertEqual([sum(poi.base_key == key for poi in selected.pois) for key in candidates.BASE_KEYS], [6, 6, 6])
        small = roster([source("Альфа")], model_batch("Альфа", 14), request(7))[0]
        self.assertEqual(len(small.pois), 15)
        self.assertEqual({poi.base_key for poi in small.pois}, {"base_1"})

    def test_model_cannot_resurrect_omitted_source_title_with_changed_locality(self):
        hints = [source(name, number) for name in ("Альфа", "Бета", "Гамма") for number in (1, 2)] + [source("Дельта")]
        payload = model_batch("Альфа")
        payload["pois"][0].update(title=hints[-1]["title"], place_query=f"{hints[-1]['title']}, Альфа, Россия")
        selected, _, omitted = roster(hints, payload)
        self.assertEqual(omitted, ["Дельта, Россия"])
        self.assertNotIn(hints[-1]["title"], {poi.title for poi in selected.pois})

    def test_unknown_model_locality_does_not_get_a_source_base(self):
        selected, _, _ = roster([source("Альфа")], model_batch("Другой район"))
        self.assertEqual(len(selected.pois), 1)

    def test_no_catalogue_keeps_original_model_path(self):
        result, _, bases, pois = self.run_grounding([])
        self.assertFalse(result.source_roster_used)
        self.assertEqual(bases.call_args.args[0], "Чужой, Россия")
        self.assertEqual(pois.call_count, 4)

    def test_stay_area_keeps_original_model_path(self):
        result, _, bases, pois = self.run_grounding([source("Долинный")], stay_area="Чужой")
        self.assertFalse(result.source_roster_used)
        self.assertEqual(bases.call_args.args[0], "Чужой, Россия")
        self.assertEqual(pois.call_count, 4)

    def test_source_does_not_bypass_admin_or_identity_filters(self):
        result, _, _, pois = self.run_grounding([source("Долинный", i) for i in (1, 2)], wrong_address=True)
        self.assertEqual(pois.call_count, 2)
        self.assertEqual(result.confirmed_hints, [])

    def test_maximum_roster_uses_one_completion_and_at_most_42_reservation(self):
        hints = [source(name, number) for name in ("Альфа", "Бета", "Гамма") for number in range(1, 9)]
        result, completion, bases, pois = self.run_grounding(hints, days=7, miss=True)
        self.assertEqual((completion.call_count, bases.call_count, pois.call_count), (1, 3, 18))
        self.assertEqual((result.verification_queries, result.reserved_http_attempts_upper_bound), (21, 42))
        self.assertEqual(result.http_attempt_limit, 42)

    def test_small_map_budget_stops_without_extra_base_or_completion(self):
        result, completion, bases, pois = self.run_grounding([source(name) for name in ("Альфа", "Бета", "Гамма")], max_http_attempts=2)
        self.assertEqual((completion.call_count, bases.call_count, pois.call_count), (1, 1, 0))
        self.assertEqual(result.reserved_http_attempts_upper_bound, 2)

    def test_cached_poi_from_omitted_cluster_can_use_compatible_selected_base(self):
        hints = [source(name, number) for name in ("Альфа", "Бета", "Гамма") for number in (1, 2)] + [source("Дельта")]
        title = hints[-1]["title"]
        query = f"{title}, Дельта, Россия"
        cached = {"name": query, "lat": 51.01, "lon": 101.01, "type": "museum", "category": "tourism",
                  "address": {"village": "Дельта", "county": "Округ Альфа", "country_code": "ru"}}
        result, _, _, _ = self.run_grounding(hints, days=7, cache={query.lower(): cached}, miss=True)
        self.assertEqual(result.source_clusters_omitted, ["Дельта, Россия"])
        self.assertEqual([item["title"] for item in result.confirmed_hints], [title])
        self.assertEqual(result.confirmed_hints[0]["allowed_base_keys"], ["base_1"])

    def test_usable_clusters_outrank_larger_unknown_or_closed_clusters(self):
        hints = [source("Неизвестный", i) for i in range(1, 7)]
        hints += [source("Закрытый", i) for i in range(1, 6)]
        for hint in hints[6:]:
            self.add_card(hint, hours=[])
        useful = [source(name) for name in ("Альфа", "Бета", "Гамма")]
        hints += useful
        for hint in useful:
            self.add_card(hint)
        selected, _, omitted = roster(hints, trip=card_request())
        self.assertEqual([base.base_query for base in selected.bases], ["Альфа, Россия", "Бета, Россия", "Гамма, Россия"])
        self.assertEqual(set(omitted), {"Неизвестный, Россия", "Закрытый, Россия"})
        self.assertEqual({poi.place_query for poi in selected.pois}, {hint["place_query"] for hint in useful})

    def test_round_robin_never_spends_eighteen_slots_before_usable_cards(self):
        hints = [source(name, i) for name in ("Альфа", "Бета", "Гамма") for i in range(1, 9)]
        useful = [hint for hint in hints if int(hint["title"].split()[-1]) >= 3]
        for hint in useful:
            self.add_card(hint)
        selected, _, _ = roster(hints, trip=card_request(7))
        self.assertEqual(len(selected.pois), 18)
        self.assertEqual({poi.place_query for poi in selected.pois}, {hint["place_query"] for hint in useful})
        self.assertEqual([sum(poi.base_key == key for poi in selected.pois) for key in candidates.BASE_KEYS], [6, 6, 6])

    def test_shared_eligibility_rejects_stale_wrong_interest_family_and_impossible_windows(self):
        hints = [source("Альфа", i) for i in range(1, 8)]
        changes = [dict(metadata_checked_at="2098-01-01"), dict(interest_categories=["food"]),
                   dict(children_allowed=False), dict(hours=[dict(start="09:00", end="09:30")]),
                   dict(hours=[]), dict(adult_price=None, planning_reserve_per_person=None), {}]
        for hint, change in zip(hints, changes):
            defaults = dict(children_allowed=True, planning_reserve_per_person=500)
            defaults.update(change)
            self.add_card(hint, **defaults)
        trip = card_request(children_ages=[8])
        ranks = candidates._source_visit_priorities(trip, hints)
        self.assertEqual([ranks[hint["place_query"]][0] for hint in hints], [2, 2, 2, 2, 2, 2, 0])

    def test_content_duration_then_price_break_ties_without_claiming_total_budget(self):
        hints = [source(name) for name in ("Короткий", "Дорогой", "Дешёвый", "Недоступный")]
        for hint, duration, price in zip(hints, [60, 120, 120, 300], [100, 500, 100, 100000]):
            self.add_card(hint, duration_min=duration, adult_price=price)
        selected, _, omitted = roster(hints, trip=card_request())
        self.assertEqual([base.base_query for base in selected.bases], ["Дешёвый, Россия", "Дорогой, Россия", "Короткий, Россия"])
        self.assertEqual(omitted, ["Недоступный, Россия"])

    def test_unknown_cache_cannot_fill_pool_before_usable_new_names(self):
        hints = [source("Альфа", i) for i in range(1, 13)]
        for hint in hints[8:]:
            self.add_card(hint)
        cache = {hint["place_query"].lower(): {
            "name": hint["place_query"], "lat": 51.0 + i * .002, "lon": 101.01,
            "type": "museum", "category": "tourism",
            "address": {"village": "Альфа", "county": "Округ Альфа", "country_code": "ru"},
        } for i, hint in enumerate(hints[:8], 1)}
        result, completion, bases, _ = self.run_grounding(hints, cache=cache, currency="RUB")
        queries = {item["place_query"] for item in result.confirmed_hints}
        self.assertTrue({hint["place_query"] for hint in hints[8:]} <= queries)
        self.assertEqual(len(queries), 8)
        self.assertEqual((completion.call_count, bases.call_count), (1, 1))
        self.assertLessEqual(result.reserved_http_attempts_upper_bound, 42)

    def test_exact_source_query_and_card_projection_do_not_create_map_evidence(self):
        hint = source("Альфа")
        self.add_card(hint)
        different = {**hint, "place_query": hint["place_query"].replace("Альфа,", "Бета,")}
        ranks = candidates._source_visit_priorities(card_request(), [hint, different])
        self.assertEqual(ranks[hint["place_query"]][0], 0)
        self.assertEqual(ranks[different["place_query"]][0], 1)
        selected, selected_hints, _ = roster([hint], trip=card_request())
        self.assertNotIn("verified", selected.pois[0].model_dump())
        self.assertNotIn("lat", selected_hints[0])

    def test_same_name_other_locality_cache_cannot_replace_source_identity(self):
        hint = source("Листвянка")
        wrong = {"name": f"{hint['title']}, Большое Голоустное, Россия", "lat": 51.1, "lon": 101.1,
                 "type": "church", "category": "amenity",
                 "address": {"village": "Большое Голоустное", "county": "Округ Листвянка", "country_code": "ru"},
                 "place_query": f"{hint['title']}, Большое Голоустное, Россия", "locality": "Большое Голоустное"}
        self.assertFalse(places._place_matches_query(hint["place_query"], wrong))
        result, _, _, lookup = self.run_grounding(
            [hint], cache={wrong["place_query"].lower(): wrong}, poi_records={hint["title"]: [wrong]})
        self.assertEqual(result.confirmed_hints, [])
        self.assertEqual(lookup.call_count, 1)  # Wrong cache does not suppress a bounded new lookup.

    def test_source_identity_filter_skips_wrong_first_record_and_accepts_next(self):
        hint = source("Листвянка")
        wrong = {"name": f"{hint['title']}, Большое Голоустное, Россия", "lat": 51.1, "lon": 101.1,
                 "type": "museum", "category": "tourism",
                 "address": {"village": "Большое Голоустное", "county": "Округ Листвянка", "country_code": "ru"},
                 "place_query": f"{hint['title']}, Большое Голоустное, Россия", "locality": "Большое Голоустное"}
        correct = {**wrong, "name": hint["place_query"], "place_query": hint["place_query"], "locality": "Листвянка",
                   "address": {**wrong["address"], "village": "Листвянка"}}
        result, _, _, lookup = self.run_grounding([hint], poi_records={hint["title"]: [wrong, correct]})
        self.assertEqual([fact["place_query"] for fact in result.confirmed_hints], [hint["place_query"]])
        self.assertEqual(lookup.call_count, 1)

    def test_explicit_card_county_alias_allows_rural_poi_without_invented_village(self):
        hint = source("Листвянка")
        canonical = f"{hint['title']}, Округ Листвянка, Россия"
        record = {"name": canonical, "lat": 51.1, "lon": 101.1, "type": "museum", "category": "tourism",
                  "address": {"county": "Округ Листвянка", "country_code": "ru"},
                  "place_query": canonical, "locality": "Округ Листвянка"}
        self.assertFalse(places._place_matches_query(hint["place_query"], record))
        before, _, _, _ = self.run_grounding([hint], poi_records={hint["title"]: [record]})
        self.assertEqual(before.confirmed_hints, [])
        self.add_card(hint, place_queries=[hint["place_query"], canonical])
        after, _, _, _ = self.run_grounding([hint], poi_records={hint["title"]: [record]})
        self.assertEqual([fact["place_query"] for fact in after.confirmed_hints], [canonical])

    def test_independent_map_hint_keeps_own_locality_despite_homonymous_source(self):
        hint = source("Листвянка")
        actual = f"{hint['title']}, Большое Голоустное, Россия"
        cached = {"name": actual, "lat": 51.1, "lon": 101.1, "type": "museum", "category": "tourism",
                  "address": {"village": "Большое Голоустное", "county": "Округ Листвянка", "country_code": "ru"}}
        bases = {"base_1": {"lat": 51., "lon": 101., "map_address": {"county": "Округ Листвянка", "country_code": "ru"}}}
        result = candidates.GroundingResult()
        with patch.object(places, "_cache", {actual.lower(): cached}), patch.object(places, "_save_cache"):
            candidates._reuse_cached_hints([hint, {"title": hint["title"], "place_query": actual}], bases, scope(), result, 18,
                                           source_queries={hint["place_query"]: (hint["place_query"],)})
        self.assertEqual([fact["place_query"] for fact in result.confirmed_hints], [actual])
        self.assertNotIn("original_proposal_query", result.confirmed_hints[0])

    def test_short_model_batch_can_be_replaced_only_by_sufficient_source_roster(self):
        hints = [source("Альфа", i) for i in range(1, 15)]
        result, completion, bases, pois = self.run_grounding(hints, payload=model_batch(count=6), days=7)
        self.assertEqual(result.problems, [])
        self.assertTrue(result.source_roster_used)
        self.assertEqual((completion.call_count, bases.call_count, pois.call_count), (1, 1, 14))
        self.assertEqual(len(result.confirmed_hints), 14)
        self.assertLessEqual(result.reserved_http_attempts_upper_bound, 42)
        for source_hints, stay in (([], ""), (hints[:13], ""), (hints, "Чужой")):
            with self.subTest(count=len(source_hints), stay=stay):
                rejected, _, base_lookup, poi_lookup = self.run_grounding(source_hints, payload=model_batch(count=6), days=7, stay_area=stay)
                self.assertTrue(rejected.problems)
                base_lookup.assert_not_called()
                poi_lookup.assert_not_called()

    def test_source_roster_never_repairs_invalid_references_extras_or_oversized_batch(self):
        hints = [source("Альфа", i) for i in range(1, 15)]
        bad_ref = model_batch(count=6)
        bad_ref["pois"][0]["base_key"] = "base_3"
        extra = model_batch(count=6)
        extra["pois"][0]["verified"] = True
        duplicate_base = model_batch(count=6)
        duplicate_base["bases"].append(deepcopy(duplicate_base["bases"][0]))
        for payload in (bad_ref, extra, duplicate_base, model_batch(count=19)):
            with self.subTest(payload=payload):
                result, _, bases, pois = self.run_grounding(hints, payload=payload, days=7)
                self.assertTrue(result.problems)
                bases.assert_not_called()
                pois.assert_not_called()


class SourceBaseFieldTests(unittest.TestCase):
    def read(self, entry):
        payload = {"version": 1, "regions": [{"aliases": ["Тестовый край, Россия"], "places": [entry]}]}
        with patch.object(Path, "read_text", return_value=json.dumps(payload, ensure_ascii=False)):
            return catalogue.destination_name_hints("Тестовый край, Россия")

    def test_optional_base_query_is_whitelisted_without_other_metadata(self):
        entry = source("Долинный", base_query="Настоящая база, Россия")
        self.assertEqual(self.read({**entry, "lat": 0, "verified": True}), [entry])
        self.assertEqual(self.read(source("Долинный")), [source("Долинный")])

    def test_invalid_base_field_and_mismatched_locality_are_rejected(self):
        for base in (True, 123, "", " missing country", "Россия", "Город, ", "a" * 121):
            with self.subTest(base=base):
                self.assertEqual(self.read(source("Долинный", base_query=base)), [])
        self.assertEqual(self.read({**source("Долинный"), "locality": "Другой"}), [])


if __name__ == "__main__":
    unittest.main()
