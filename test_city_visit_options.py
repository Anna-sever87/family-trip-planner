"""Visit options: exact map binding, unknowns, primary conditions and policy estimates."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from datetime import date
import socket
import sqlite3
import unittest
from unittest.mock import patch

import httpx
from city_candidates import CityCandidate
import city_visit_options as options
import places
from schemas import Itinerary, TripRequest


class FixedDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 22)


def request(**changes):
    values = dict(destination="Нячанг, Вьетнам", start_date="2026-10-28", end_date="2026-11-01",
                  adults=2, children_ages=[4, 10], budget=3000, currency="USD", interest_categories=["museums"])
    values.update(changes)
    return TripRequest(**values)


def candidate(title="Test Museum", identifier=1, *, kind="museum", osm_type="node", city="Nha Trang", country="Việt Nam", code="vn"):
    query = f"{title}, {city}, {country}"
    record = {"name": query, "namedetails": {"name": title}, "lat": 12.25, "lon": 109.19,
              "address": {"city": city, "country": country, "country_code": code},
              "type": kind, "category": "tourism", "osm_type": osm_type, "osm_id": identifier}
    return CityCandidate(f"osm-{osm_type}-{identifier}", title, query, city, record)


def yersin():
    return candidate("Bào Tàng Alexandre Yersin", 1193057654, osm_type="way")


def ocean():
    return candidate("Bảo tàng Hải dương học Nha Trang", 15662995, osm_type="relation")


class CityVisitOptionsTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((httpx, "get"), (httpx, "post"), (socket.socket, "connect"),
                            (socket, "getaddrinfo"), (sqlite3, "connect"), (places, "geocode"),
                            (places, "_load_cache"), (places, "_save_cache")):
            handle = patch.object(owner, name, side_effect=AssertionError("No external I/O/cache"))
            handle.start(); self.addCleanup(handle.stop)
        handle = patch.object(options, "date", FixedDate)
        handle.start(); self.addCleanup(handle.stop)

    def test_unknown_indoor_and_price_remain_none_not_free_or_indoor(self):
        value = options.visit_option(candidate(), request())
        self.assertIsNone(value.indoor)
        self.assertIsNone(value.estimated_cost)
        self.assertIn("price", value.unknown_fields)
        self.assertIn("indoor", value.unknown_fields)
        self.assertIn("hours", value.unknown_fields)
        self.assertEqual((value.allowed_dates, value.opening_windows, value.sources), (None, (), ()))

    def test_shopping_centre_visit_is_estimated_with_no_invented_conditions(self):
        for kind in ("mall", "shopping_centre"):
            with self.subTest(kind=kind):
                item = candidate("Test Shopping Centre", kind=kind)
                item.record["category"] = "shop"
                value = options.visit_option(item, request(interest_categories=["shopping_malls"]))
                self.assertIsNotNone(value)
                self.assertEqual((value.duration_min, value.basis), (90, "planning_estimate"))
                self.assertIsNone(value.indoor)
                self.assertIsNone(value.estimated_cost)
                self.assertEqual((value.allowed_dates, value.opening_windows, value.sources), (None, (), ()))
                self.assertTrue({"price", "hours", "indoor"}.issubset(value.unknown_fields))
                self.assertIn("покупки по желанию", value.description)
                self.assertIn("Личные покупки не включены", value.description)
                self.assertIn("не подтверждённая длительность", value.description)

    def test_shopping_label_does_not_make_an_unrelated_map_type_a_visit(self):
        for category, kind in (("building", "mall"), ("tourism", "shopping_centre"), ("shop", "supermarket")):
            with self.subTest(category=category, kind=kind):
                item = candidate("Test Shopping Centre", kind=kind)
                item.record["category"] = category
                self.assertIsNone(options.visit_option(item, request()))

    def test_fixed_duration_is_selected_estimate_not_product_cap_or_family_minimum(self):
        for kind, duration in (("museum", 90), ("park", 90), ("garden", 90), ("gallery", 90),
                               ("artwork", 30), ("monument", 30), ("place_of_worship", 30),
                               ("archaeological_site", 60), ("theatre", 45)):
            with self.subTest(kind=kind):
                item = candidate(kind=kind)
                first = options.visit_option(item, request())
                other = options.visit_option(item, request(children_ages=[], pace="low"))
                self.assertEqual((first.duration_min, other.duration_min), (duration, duration))
                self.assertEqual(first.basis, "planning_estimate")
                self.assertIn("не подтверждённая длительность", first.description)
                self.assertLess(first.duration_min, options.get_city_planning_cap(item))

    def test_outdoor_or_explicit_open_air_negative_does_not_claim_shelter(self):
        for item in (candidate(kind="park"), candidate("Açık Hava Test Museum"), candidate("Open Air Test Museum")):
            self.assertIs(options.visit_option(item, request()).indoor, False)

    def test_model_record_fields_cannot_supply_conditions(self):
        item = candidate()
        item.record.update(indoor=True, duration_min=120, cost=15, opening_hours="24/7",
                           visit_conditions={"indoor": True, "price": 0})
        value = options.visit_option(item, request())
        self.assertEqual((value.duration_min, value.indoor, value.estimated_cost), (90, None, None))

    def test_bad_canonical_identity_query_area_or_country_fails_closed(self):
        for field, value in (("candidate_id", "osm-node-999"), ("title", "Another Museum"),
                             ("place_query", "Another Museum, Nha Trang, Việt Nam"), ("area", "Unknown")):
            with self.subTest(field=field):
                item = candidate(); setattr(item, field, value)
                self.assertIsNone(options.visit_option(item, request()))
        self.assertIsNone(options.visit_option(candidate(), request(destination="Нячанг, Россия")))
        forged = candidate(); forged.record["provenance"] = "model"
        self.assertIsNone(options.visit_option(forged, request()))

    def test_yersin_exact_observed_spelling_uses_published_thirty_and_room_evidence(self):
        value = options.visit_option(yersin(), request())
        self.assertEqual((value.duration_min, value.basis, value.indoor), (30, "published", True))
        self.assertIsNone(value.estimated_cost)  # No invented VND -> USD conversion.
        self.assertEqual(value.allowed_dates, ("2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31"))
        saturday = [window for window in value.opening_windows if window["date"] == "2026-10-31"]
        self.assertEqual([(window["start"], window["end"]) for window in saturday], [("07:30", "11:30")])
        self.assertTrue(all(source["checked_at"] == "2026-10-22" for source in value.sources))
        self.assertIn("праздник", " ".join(value.restrictions))

    def test_yersin_published_price_keeps_original_currency_only(self):
        value = options.visit_option(yersin(), request(currency="VND", budget=30_000_000))
        self.assertEqual(value.estimated_cost, 4 * 26000)
        self.assertNotIn("price", value.unknown_fields)
        for currency in ("USD", "EUR", "RUB"):
            self.assertIsNone(options.visit_option(yersin(), request(currency=currency)).estimated_cost)

    def test_known_sunday_closure_is_empty_allowed_dates_not_unknown_hours(self):
        value = options.visit_option(yersin(), request(start_date="2026-11-01", end_date="2026-11-01"))
        self.assertEqual((value.allowed_dates, value.opening_windows), ((), ()))
        self.assertNotIn("hours", value.unknown_fields)

    def test_ocean_conditions_are_exact_main_campus_and_room_program_only(self):
        value = options.visit_option(ocean(), request())
        self.assertEqual((value.duration_min, value.basis, value.indoor), (90, "planning_estimate", True))
        self.assertIsNone(value.estimated_cost)
        self.assertEqual(len(value.allowed_dates), 5)
        self.assertTrue(all((window["start"], window["end"]) == ("06:00", "18:00") for window in value.opening_windows))
        self.assertIn("не к филиалу Hòn Chồng", value.description)
        different = candidate(ocean().title, 123456, osm_type="relation")
        self.assertIsNone(options.visit_option(different, request()).indoor)

    def test_curated_indoor_never_transfers_to_wrong_locale_name_country_or_id(self):
        variants = [candidate(yersin().title, 111, osm_type="way"),
                    candidate(yersin().title, 1193057654, osm_type="way", city="Hanoi"),
                    candidate("Another Museum", 1193057654, osm_type="way"),
                    candidate(yersin().title, 1193057654, osm_type="way", country="Türkiye", code="tr")]
        for item in variants:
            with self.subTest(item=item):
                value = options.visit_option(item, request())
                self.assertTrue(value is None or value.indoor is None)

    def test_stale_conditions_become_unknown_not_future_booking_promises(self):
        future = request(start_date="2027-06-01", end_date="2027-06-05")
        value = options.visit_option(ocean(), future)
        self.assertEqual((value.indoor, value.estimated_cost, value.allowed_dates), (None, None, None))
        self.assertIn("hours", value.unknown_fields)
        # A stale published estimate cannot silently turn the policy cap into
        # a newly selected duration for an otherwise generic museum option.
        self.assertIsNone(options.visit_option(yersin(), future))

    def card(self, item, **changes):
        values = {"id": "test_card", "place_queries": [item.place_query], "metadata_checked_at": "2026-10-22",
                  "description": "Осмотр коллекции.", "duration_min": 45, "duration_basis": "published",
                  "duration_source": "https://example.invalid/duration", "indoor": True,
                  "indoor_source": "https://example.invalid/rooms", "hours_source": "https://example.invalid/hours",
                  "hours": [{"dates": ["2026-10-29"], "start": "10:00", "end": "12:00"}],
                  "currency": "RUB", "adult_price": 100, "price_source": "https://example.invalid/tickets",
                  "children_allowed": True, "minimum_age": 5}
        values.update(changes)
        return values

    def test_loaded_card_exact_query_date_age_cost_and_window_are_preserved(self):
        item = candidate()
        card = self.card(item)
        req = request(children_ages=[], currency="RUB")
        value = options.visit_option(item, req, visit_cards=[card])
        self.assertEqual((value.duration_min, value.estimated_cost, value.min_child_age), (45, 200, 5))
        self.assertEqual(value.allowed_dates, ("2026-10-29",))
        self.assertEqual((value.opening_windows[0]["start"], value.opening_windows[0]["end"]), ("10:00", "12:00"))
        self.assertIsNone(options.visit_option(item, request(), visit_cards=[card]))  # youngest 4 < 5
        child = options.visit_option(item, request(children_ages=[5]), visit_cards=[card])
        self.assertIsNone(child.estimated_cost)  # Adult rate cannot price an unknown child band.

    def test_unsupported_or_conflicting_source_card_is_excluded_not_silently_ignored(self):
        item = candidate()
        for changes in ({"route_kind": "loop"}, {"restrictions": ["Only with reservation"]},
                        {"closed_month_days": ["10-29"]}, {"duration_min": 180},
                        {"children_allowed": False}, {"metadata_checked_at": "2020-01-01"}):
            with self.subTest(changes=changes):
                card = self.card(item, minimum_age=0, **changes)
                self.assertIsNone(options.visit_option(item, request(), visit_cards=[card]))
        card = self.card(item, minimum_age=0)
        self.assertIsNone(options.visit_option(item, request(), visit_cards=[card, deepcopy(card)]))

    def test_results_and_nested_metadata_do_not_mutate_inputs_or_other_results(self):
        item, req = ocean(), request()
        before = deepcopy((item, req))
        first = options.visit_option(item, req)
        with self.assertRaises(FrozenInstanceError):
            first.duration_min = 999
        first.sources[0]["source_url"] = "changed"
        first.opening_windows[0]["start"] = "23:00"
        second = options.visit_option(item, req)
        self.assertNotEqual(second.sources[0]["source_url"], "changed")
        self.assertEqual(second.opening_windows[0]["start"], "06:00")
        self.assertEqual((item, req), before)

    def itinerary(self, item, *, on_date="2026-10-28", time="09:00", duration=30, indoor=True):
        return Itinerary.model_validate({"title": "Test", "summary": "Test", "city_schedule": True,
            "days": [{"day": 1, "date": on_date, "theme": "Test", "items": [{
                "time": time, "title": item.title, "area": item.area, "place_query": item.place_query,
                "description": "Модельная программа посещения; не утверждение источника.",
                "duration_min": duration, "cost": 0, "indoor": indoor}]}], "tips": []})

    def test_edit_recheck_enforces_closed_dates_and_whole_window_not_fixed_duration(self):
        item, req = yersin(), request()
        for time, duration, expected in (("11:00", 30, True), ("11:15", 30, False),
                                         ("12:00", 30, False), ("14:00", 45, True)):
            with self.subTest(time=time, duration=duration):
                plan = self.itinerary(item, time=time, duration=duration)
                original = plan.model_dump()
                self.assertEqual(not options.check_city_visit_conditions(plan, [item], req), expected)
                self.assertEqual(plan.model_dump(), original)
        sunday = self.itinerary(item, on_date="2026-11-01")
        self.assertTrue(options.check_city_visit_conditions(sunday, [item], req))

    def test_edit_recheck_does_not_require_policy_duration_or_upgrade_description(self):
        item = candidate()
        plan = self.itinerary(item, duration=45)
        before = plan.model_dump()
        self.assertEqual(options.check_city_visit_conditions(plan, [item], request()), [])
        self.assertEqual(plan.model_dump(), before)
        # A curated indoor source must not be used to rewrite arbitrary AI prose.
        item = ocean(); plan = self.itinerary(item, duration=60)
        before = plan.model_dump()
        self.assertEqual(options.check_city_visit_conditions(plan, [item], request()), [])
        self.assertEqual(plan.model_dump(), before)

    def test_edit_recheck_preserves_regional_and_legacy_boundary(self):
        item = yersin()
        for changes in ({"city_schedule": False}, {"source_schedule": True}):
            plan = self.itinerary(item, on_date="2026-11-01").model_copy(update=changes)
            self.assertEqual(options.check_city_visit_conditions(plan, [item], request()), [])

    def test_edit_recheck_rejects_unbound_identity_and_confirmed_age_or_outdoor_conflict(self):
        item = candidate()
        plan = self.itinerary(item)
        plan.days[0].items[0].area = "Another locality"
        self.assertTrue(options.check_city_visit_conditions(plan, [item], request()))
        plan = self.itinerary(item)
        self.assertTrue(options.check_city_visit_conditions(plan, [item], request(), visit_cards=[self.card(item)]))
        outside = candidate(kind="park")
        self.assertTrue(options.check_city_visit_conditions(self.itinerary(outside), [outside], request()))


if __name__ == "__main__":
    unittest.main()
