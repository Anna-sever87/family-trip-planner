"""Visit-card loader regressions with memory-only catalogue and no network/DB."""

from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import socket
import unittest
from unittest.mock import patch

import visit_facts
from schemas import TripRequest


QUERY = "Музей природы, Город А, Россия"
ALIAS_QUERY = "Музей природы, Район А, Россия"


def request_for(**updates):
    fields = dict(destination="Область А, Россия", destination_kind="region",
                  start_date="2026-10-28", end_date="2026-11-03", adults=2,
                  children_ages=[], budget=100000, currency="RUB", pace="средний",
                  start_time="08:00", end_time="21:00", interest_categories=["museums"])
    fields.update(updates)
    return TripRequest(**fields)


def card_for(**updates):
    card = dict(id="nature_museum", place_queries=[QUERY], description="Содержательная экспозиция природы.",
                duration_min=90, duration_basis="planning_estimate", duration_source="https://example.org/museum",
                basis_note="Оценка на самостоятельное знакомство с экспозицией, не опубликованный сеанс.",
                indoor=True, hours=[dict(weekdays=list(range(7)), months=list(range(1, 13)), start="09:00", end="18:00")],
                hours_source="https://example.org/hours", hours_note="Опубликованный график, проверить перед визитом.",
                adult_price=500, child_price=200, group_min_price=None, price_source="https://example.org/prices",
                planning_reserve_per_person=None, price_note="Опубликованный тариф на самостоятельный осмотр.",
                currency="RUB", metadata_checked_at="2026-10-21", children_allowed=True, minimum_age=0,
                interest_categories=["museums"], restrictions=["На экскурсию нужна предварительная запись."], kids_note="")
    card.update(updates)
    return card


def catalogue(*cards, **updates):
    result = dict(version=1, metadata_checked_at="2026-10-21",
                  regions=[dict(aliases=["Область А", "Область А, Россия"], visits=list(cards or [card_for()]))])
    result.update(updates)
    return result


def grounding_for(*queries):
    return SimpleNamespace(confirmed_hints=[dict(place_query=query, title=query.split(",")[0],
                                                lat=50 + i * .1, lon=100, base_key="base_1")
                                            for i, query in enumerate(queries or (QUERY,))])


class VisitFactLoaderTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "create_connection"), (socket, "getaddrinfo")):
            blocker = patch.object(owner, name, side_effect=AssertionError("network forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)

    def load(self, payload, destination="Область А, Россия"):
        with patch.object(Path, "read_text", return_value=json.dumps(payload, ensure_ascii=False)):
            return visit_facts.load_visit_cards(destination, path=Path("unused-memory-catalogue.json"))

    def bind(self, payload, request=None, grounding=None):
        with patch.object(Path, "read_text", return_value=json.dumps(payload, ensure_ascii=False)):
            return visit_facts.visit_facts_for_grounding(request or request_for(), grounding or grounding_for(),
                                                        path=Path("unused-memory-catalogue.json"))

    def test_exact_normalized_region_alias_only(self):
        payload = catalogue()
        self.assertEqual(len(self.load(payload, "  ОБЛАСТЬ   А, РОССИЯ  ")), 1)
        for wrong in ("Область", "Область А, другая страна", "Город А", ""):
            with self.subTest(wrong=wrong):
                self.assertEqual(self.load(payload, wrong), [])

    def test_missing_bad_json_and_non_object_catalogue_fail_closed(self):
        for error in (OSError("missing"), UnicodeError("invalid text")):
            with self.subTest(error=error), patch.object(Path, "read_text", side_effect=error):
                self.assertEqual(visit_facts.load_visit_cards("Область А"), [])
        with patch.object(Path, "read_text", return_value="{broken-json"):
            self.assertEqual(visit_facts.load_visit_cards("Область А"), [])
        for payload in ([], None, {"version": 2, "regions": []}, {"version": 1, "regions": None}):
            with self.subTest(payload=payload):
                self.assertEqual(self.load(payload), [])

    def test_malformed_container_fields_do_not_crash_or_create_facts(self):
        for field, bad in (("aliases", None), ("aliases", 7), ("visits", None), ("visits", 7)):
            payload = catalogue()
            payload["regions"][0][field] = bad
            with self.subTest(field=field, bad=bad):
                self.assertEqual(self.bind(payload), [])
        for field in ("interest_categories", "restrictions"):
            with self.subTest(field=field):
                self.assertEqual(self.bind(catalogue(card_for(**{field: None}))), [])

    def test_unsafe_sources_invalid_numbers_and_no_basis_are_rejected(self):
        for change in (dict(duration_source="http://example.org/info"), dict(duration_source="https://user:pass@example.org/info"),
                       dict(duration_min=True), dict(duration_min=601), dict(adult_price=True),
                       dict(child_price=float("nan")), dict(group_min_price=-1),
                       dict(adult_price=10**400), dict(planning_reserve_per_person=10**400),
                       dict(planning_reserve_per_person=float("inf")), dict(basis_note=""), dict(indoor="yes")):
            with self.subTest(change=change):
                self.assertEqual(self.load(catalogue(card_for(**change))), [])

    def test_exact_map_query_required_and_ambiguous_aliases_rejected(self):
        payload = catalogue(card_for(place_queries=[QUERY, ALIAS_QUERY]))
        self.assertEqual(len(self.bind(payload, grounding=grounding_for(QUERY))), 1)
        self.assertEqual(self.bind(payload, grounding=grounding_for(QUERY, ALIAS_QUERY)), [])
        self.assertEqual(self.bind(payload, grounding=grounding_for("Музей природы, Другой город, Россия")), [])
        self.assertEqual(self.bind(payload, grounding=grounding_for("Музей природы")), [])

    def test_duplicate_map_query_with_conflicting_identities_is_not_silently_collapsed(self):
        ambiguous = grounding_for(QUERY, QUERY)
        self.assertEqual(self.bind(catalogue(), grounding=ambiguous), [])

    def test_freshness_includes_trip_end_and_rejects_future_or_stale_research(self):
        request = request_for()
        end = date.fromisoformat(request.end_date)
        for age, expected in ((0, 1), (90, 1), (91, 0), (-1, 0)):
            payload = catalogue(card_for(metadata_checked_at=(end - timedelta(days=age)).isoformat()))
            with self.subTest(age=age):
                self.assertEqual(len(self.bind(payload, request)), expected)
        self.assertEqual(self.bind(catalogue(card_for(metadata_checked_at="yesterday"))), [])

    def test_catalogue_research_date_applies_only_when_card_has_no_override(self):
        card = card_for()
        del card["metadata_checked_at"]
        self.assertEqual(len(self.bind(catalogue(card))), 1)
        card["metadata_checked_at"] = "2020-01-01"
        self.assertEqual(self.bind(catalogue(card)), [])

    def test_currency_is_never_silently_converted_or_relabelled(self):
        for request in (request_for(currency="EUR"), request_for(currency="USD"), request_for(destination_kind="city")):
            self.assertEqual(self.bind(catalogue(), request), [])
        self.assertEqual(self.bind(catalogue(card_for(currency="EUR"))), [])

    def test_published_group_minimum_and_group_composition(self):
        fact = self.bind(catalogue(card_for(adult_price=1000, child_price=500, group_min_price=5000)),
                         request_for())[0]
        self.assertEqual(fact.group_cost, 5000)
        self.assertIsNone(fact.planning_reserve)
        larger = self.bind(catalogue(card_for(adult_price=1000, child_price=500, group_min_price=5000)),
                           request_for(adults=6))[0]
        self.assertEqual(larger.group_cost, 6000)
        self.assertEqual(larger.price_source, "https://example.org/prices")

    def test_unknown_price_remains_unknown_with_explicit_group_reserve(self):
        payload = catalogue(card_for(adult_price=None, child_price=None, planning_reserve_per_person=800,
                                     price_note="800 рублей на человека — резерв, не опубликованная цена."))
        fact = self.bind(payload, request_for(children_ages=[8]))[0]
        self.assertIsNone(fact.group_cost)
        self.assertEqual(fact.planning_reserve, 2400)
        self.assertIn("не опубликованная цена", fact.price_note)
        no_reserve = self.bind(catalogue(card_for(adult_price=None, child_price=None)))[0]
        self.assertIsNone(no_reserve.group_cost)
        self.assertIsNone(no_reserve.planning_reserve)

    def test_unknown_reserve_without_note_never_claims_published_price(self):
        card = card_for(adult_price=None, child_price=None, planning_reserve_per_person=800)
        del card["price_note"]
        facts = self.bind(catalogue(card))
        if facts:
            self.assertIsNone(facts[0].group_cost)
            self.assertIn("резерв", facts[0].price_note.casefold())
            self.assertNotIn("по опубликованному источнику", facts[0].price_note.casefold())

    def test_unknown_child_tariff_is_not_free_or_replaced_by_adult_price(self):
        payload = catalogue(card_for(child_price=None, planning_reserve_per_person=700,
                                     price_note="Общий резерв на группу; детская категория не выяснена."))
        adults = self.bind(payload)[0]
        self.assertEqual(adults.group_cost, 1000)
        family = self.bind(payload, request_for(children_ages=[5, 12]))[0]
        self.assertIsNone(family.group_cost)
        self.assertEqual(family.planning_reserve, 2800)

    def test_single_child_price_without_age_bands_is_not_an_exact_family_tariff(self):
        payload = catalogue(card_for(child_price=200, planning_reserve_per_person=700,
                                     price_note="Резерв; возрастные границы тарифа не установлены."))
        for ages in ([1], [6], [12], [17]):
            with self.subTest(ages=ages):
                fact = self.bind(payload, request_for(children_ages=ages))[0]
                self.assertIsNone(fact.group_cost)
                self.assertEqual(fact.planning_reserve, 2100)

    def test_child_eligibility_unknown_is_preserved_not_assumed_safe(self):
        card = card_for()
        del card["children_allowed"]
        fact = self.bind(catalogue(card), request_for(children_ages=[5]))[0]
        self.assertIsNone(fact.children_allowed)
        self.assertEqual(fact.minimum_age, 0)

    def test_hours_respect_weekdays_months_exact_dates_and_lunch_closure(self):
        payload = catalogue(card_for(hours=[
            dict(weekdays=[2], months=[10], start="09:00", end="12:00"),
            dict(weekdays=[2], months=[10], start="13:00", end="18:00"),
            dict(dates=["2026-10-31"], start="10:00", end="16:00"),
        ]))
        fact = self.bind(payload)[0]
        self.assertEqual([(window.date, window.start_time, window.end_time) for window in fact.date_windows], [
            ("2026-10-28", "09:00", "12:00"), ("2026-10-28", "13:00", "18:00"),
            ("2026-10-31", "10:00", "16:00"),
        ])
        self.assertTrue(all(window.evidence == "source" for window in fact.date_windows))

    def test_no_hours_are_fabricated_from_unknown_hours_or_office_note(self):
        for change in (dict(hours=[]), dict(hours=[], hours_note="Офис открыт с 09:00 до 18:00, сеанс уточните."),
                       dict(hours_source="http://example.org/untrusted")):
            with self.subTest(change=change):
                fact = self.bind(catalogue(card_for(**change)))[0]
                self.assertEqual(fact.date_windows, ())

    def test_sessions_reserve_exact_start_not_arbitrary_time_inside_opening_hours(self):
        payload = catalogue(card_for(hours=[], duration_min=45, sessions=[
            dict(weekdays=[2], start="11:00"), dict(weekdays=[2], start="13:00"),
        ]))
        fact = self.bind(payload)[0]
        self.assertEqual([(window.date, window.start_time, window.end_time) for window in fact.date_windows], [
            ("2026-10-28", "11:00", "11:45"), ("2026-10-28", "13:00", "13:45"),
        ])
        from regional_skeleton import _fit
        self.assertEqual(_fit(11 * 60, 45, fact.date_windows, "2026-10-28"), 11 * 60)
        self.assertEqual(_fit(11 * 60 + 1, 45, fact.date_windows, "2026-10-28"), 13 * 60)
        self.assertIsNone(_fit(13 * 60 + 1, 45, fact.date_windows, "2026-10-28"))
        self.assertTrue(all(window.evidence == "estimate" and "конец" in window.note for window in fact.date_windows))

    def test_closed_occurrence_weekday_applies_to_hours_and_sessions(self):
        card = card_for(closed_weekdays=[dict(weekday=2, occurrence=1)],
                        sessions=[dict(weekdays=[2], start="11:00")])
        trip = request_for(start_date="2026-11-01", end_date="2026-11-11")
        fact = self.bind(catalogue(card), trip)[0]
        dates = {window.date for window in fact.date_windows}
        self.assertNotIn("2026-11-04", dates)  # First Wednesday.
        self.assertIn("2026-11-11", dates)  # Second Wednesday stays available.
        self.assertIn("2026-11-03", dates)

    def test_closed_weekdays_strict_shape_ranges_and_bool_rejection(self):
        for value in (None, {}, [None], [{"weekday": 2}], [{"weekday": 2, "occurrence": 0}],
                      [{"weekday": 2, "occurrence": 6}], [{"weekday": 7, "occurrence": 1}],
                      [{"weekday": -1, "occurrence": 1}], [{"weekday": True, "occurrence": 1}],
                      [{"weekday": 2, "occurrence": True}], [{"weekday": 2, "occurrence": 1, "extra": 1}]):
            with self.subTest(value=value):
                self.assertEqual(self.load(catalogue(card_for(closed_weekdays=value))), [])
        self.assertEqual(len(self.load(catalogue(card_for(closed_weekdays=[dict(weekday=6, occurrence=5)])))), 1)

    def test_source_card_projection_matches_binding_without_claiming_map_identity(self):
        raw = card_for()
        before = deepcopy(raw)
        projected = visit_facts.visit_fact_from_card(request_for(), raw, QUERY)
        self.assertEqual(projected, self.bind(catalogue(raw))[0])
        self.assertIsNone(visit_facts.visit_fact_from_card(request_for(), raw, ALIAS_QUERY))
        self.assertFalse(hasattr(projected, "lat"))
        self.assertFalse(hasattr(projected, "verified"))
        self.assertEqual(raw, before)

    def test_loop_access_point_is_separate_from_map_identity(self):
        raw = card_for(route_kind="loop", access_point=[51.123, 104.456],
                       access_point_source="https://example.org/official-route")
        ground = grounding_for()
        before = deepcopy(ground)
        fact = self.bind(catalogue(raw), grounding=ground)[0]
        self.assertEqual((fact.route_kind, fact.access_point, fact.access_point_source),
                         ("loop", (51.123, 104.456), "https://example.org/official-route"))
        self.assertEqual(ground, before)
        self.assertNotEqual(fact.access_point, (ground.confirmed_hints[0]["lat"], ground.confirmed_hints[0]["lon"]))
        point = self.bind(catalogue())[0]
        self.assertEqual((point.route_kind, point.access_point, point.access_point_source), ("point", None, ""))

    def test_loop_access_requires_finite_coordinates_https_and_explicit_loop(self):
        valid = dict(route_kind="loop", access_point=[51, 104], access_point_source="https://example.org/entry")
        for change in (dict(route_kind="linear"), dict(route_kind="point"), dict(route_kind=True),
                       dict(access_point=None), dict(access_point=[51]), dict(access_point=[True, 104]),
                       dict(access_point=[91, 104]), dict(access_point=[51, 181]),
                       dict(access_point=[float("nan"), 104]), dict(access_point=[51, float("inf")]),
                       dict(access_point_source=""), dict(access_point_source="http://example.org/entry"),
                       dict(access_point_source="https://user:pass@example.org/entry")):
            with self.subTest(change=change):
                self.assertEqual(self.load(catalogue(card_for(**{**valid, **change}))), [])
        self.assertEqual(self.load(catalogue(card_for(access_point_source="https://example.org/entry"))), [])

    def test_max_group_size_is_strict_and_filters_adults_plus_children(self):
        for value in (None, True, 0, 101, 5.0, "5"):
            with self.subTest(value=value):
                self.assertEqual(self.load(catalogue(card_for(max_group_size=value))), [])
        card = card_for(max_group_size=5, adult_price=0, group_min_price=3900,
                        planning_reserve_per_person=1000)
        for count in (1, 2, 5):
            with self.subTest(adults=count):
                fact = self.bind(catalogue(card), request_for(adults=count))[0]
                self.assertEqual(fact.group_cost, 3900)
        self.assertEqual(self.bind(catalogue(card), request_for(adults=6)), [])
        family = self.bind(catalogue(card), request_for(adults=2, children_ages=[8, 9, 10]))[0]
        self.assertIsNone(family.group_cost)  # A cap is not a children's tariff.
        self.assertEqual(family.planning_reserve, 5000)
        self.assertEqual(self.bind(catalogue(card), request_for(adults=2, children_ages=[8, 9, 10, 11])), [])

    def test_route_anchor_is_source_metadata_and_never_a_map_replacement(self):
        data = dict(route_kind="loop", access_point=[50, 100], map_anchor_point=[50.01, 100],
                    length_km=3.7, access_point_source="https://example.org/route.pdf")
        fact = self.bind(catalogue(card_for(**data)))[0]
        self.assertEqual(fact.map_anchor_point, (50.01, 100))
        self.assertFalse(hasattr(fact, "lat"))
        for change in (dict(length_km=None), dict(length_km=1), dict(length_km=True),
                       dict(length_km=51), dict(length_km=10**400), dict(map_anchor_point=[True, 100]),
                       dict(map_anchor_point=[91, 100]), dict(route_kind="point")):
            with self.subTest(change=change):
                self.assertEqual(self.load(catalogue(card_for(**{**data, **change}))), [])

    def test_real_catalog_taltsy_new_year_closure_and_nerpa_monday_are_not_open_windows(self):
        cards = {card["id"]: card for card in visit_facts.load_visit_cards("Байкал, Россия")}
        taltsy = cards["taltsy_museum"]
        request = request_for(destination="Байкал, Россия", start_date="2027-01-01", end_date="2027-01-02")
        facts = visit_facts.visit_facts_for_grounding(request, grounding_for(taltsy["place_queries"][0]))
        self.assertEqual(len(facts), 1)
        self.assertEqual([(window.date, window.start_time, window.end_time) for window in facts[0].date_windows],
                         [("2027-01-02", "10:00", "17:00")])
        nerpa = cards["listvyanka_nerpinarium"]
        request = request_for(destination="Байкал, Россия")
        fact = visit_facts.visit_facts_for_grounding(request, grounding_for(nerpa["place_queries"][0]))[0]
        self.assertFalse(any(window.date == "2026-11-02" for window in fact.date_windows))
        self.assertEqual({window.start_time for window in fact.date_windows},
                         {"11:00", "12:00", "13:00", "15:00", "16:00", "17:00", "18:00"})

    def test_bad_hour_filters_and_times_fail_closed_without_crashing(self):
        for hour in (dict(weekdays=None, start="09:00", end="18:00"),
                     dict(months=None, start="09:00", end="18:00"),
                     dict(weekdays="2", start="09:00", end="18:00"),
                     dict(start="09:-1", end="18:00"), dict(start="09:00", end="24:00"),
                     dict(start="9x:00", end="18:00"), dict(start="18:00", end="09:00")):
            with self.subTest(hour=hour):
                facts = self.bind(catalogue(card_for(hours=[hour])))
                self.assertTrue(not facts or all(not fact.date_windows for fact in facts))

    def test_reviewed_goloustnoye_trails_have_source_windows_and_separate_entries(self):
        request = request_for(destination="Байкал, Россия", interest_categories=["parks"])
        cards = {card["id"]: card for card in visit_facts.load_visit_cards(request.destination)}
        expected = {
            "sukhoye_lake_trail": (52.01338, 105.36101),
            "grebeshok_trail": (52.054339, 105.403944),
        }
        for key, entry in expected.items():
            with self.subTest(key=key):
                card = cards[key]
                fact = visit_facts.visit_fact_from_card(request, card, card["place_queries"][0])
                self.assertEqual(fact.duration_min, 120)
                self.assertEqual(fact.duration_basis, "published")
                self.assertEqual((fact.route_kind, fact.access_point), ("loop", entry))
                self.assertEqual(len(fact.date_windows), request.days_count)
                self.assertTrue(all((w.start_time, w.end_time) == ("10:00", "16:00")
                                    and w.source.endswith(".pdf") for w in fact.date_windows))
                self.assertIsNone(fact.group_cost, "Citizenship is unknown; no universal resident tariff")
                self.assertEqual(fact.planning_reserve, 2000)

        winter = request_for(destination="Байкал, Россия", start_date="2026-12-01", end_date="2026-12-02")
        card = cards["sukhoye_lake_trail"]
        fact = visit_facts.visit_fact_from_card(winter, card, card["place_queries"][0])
        self.assertEqual(fact.date_windows, (), "An autumn schedule is not proof of winter hours")

    def test_input_data_is_not_mutated_and_map_fields_are_never_sourced_from_cards(self):
        payload = catalogue(card_for(lat=1, lon=2, verified=True))
        grounding = grounding_for()
        before = deepcopy((payload, vars(grounding)))
        fact = self.bind(payload, grounding=grounding)[0]
        self.assertFalse(hasattr(fact, "lat"))
        self.assertFalse(hasattr(fact, "verified"))
        self.assertEqual(before, (payload, vars(grounding)))
        self.assertEqual(fact.duration_basis, "planning_estimate")
        self.assertIn("не опубликованный", fact.basis_note)


if __name__ == "__main__":
    unittest.main()
