"""Map-only free calendars never impersonate source-validated programs."""

import unittest

from pydantic import ValidationError

from pricing import minimum_meal_cost
from program_coverage import (
    authorized_free_days, limited_program_problems, program_report_text, rebuild_program_report,
)
from schemas import Itinerary, LimitedProgramReport
from test_program_coverage import _request, _route


def request():
    return _request(destination="Тестгород, Россия", destination_kind="city")


def city_route(*, active=(1, 2, 3)):
    req = request()
    route = _route(req, active=active)
    route.source_schedule = False
    route.city_schedule = True
    route.program_report = None
    for day in route.days:
        day.base_area = ""
    rebuild_program_report(route, req, eligible_unique_visits=7, allow_city_limited_program=True)
    return route


class CityProgramSchemaTests(unittest.TestCase):
    def test_legacy_source_report_and_itinerary_keep_defaults(self):
        old = _route().model_dump()
        old.pop("city_schedule")
        old["program_report"].pop("coverage_basis")
        loaded = Itinerary.model_validate(old)
        self.assertFalse(loaded.city_schedule)
        self.assertEqual(loaded.program_report.coverage_basis, "source")
        self.assertEqual(limited_program_problems(loaded, _request(), allow_limited_program=True), [])

    def test_city_flag_is_strict_and_basis_has_no_arbitrary_claims(self):
        values = city_route().model_dump()
        for value in (1, 0, "true", None, []):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                Itinerary.model_validate({**values, "city_schedule": value})
        for value in (True, None, "VisitFacts", "fully_verified", {}):
            with self.subTest(basis=value), self.assertRaises(ValidationError):
                LimitedProgramReport.model_validate({**values["program_report"], "coverage_basis": value})


class CityProgramAuthorizationTests(unittest.TestCase):
    def test_map_free_days_require_distinct_explicit_city_context(self):
        route, req = city_route(), request()
        for options in ({}, {"allow_limited_program": True}, {"allow_city_limited_program": 1}):
            with self.subTest(options=options):
                self.assertTrue(limited_program_problems(route, req, **options))
                self.assertEqual(authorized_free_days(route, req, **options), frozenset())
        self.assertEqual(limited_program_problems(route, req, allow_city_limited_program=True), [])
        self.assertEqual(authorized_free_days(route, req, allow_city_limited_program=True), frozenset({4, 5, 6, 7}))
        self.assertFalse(route.source_schedule)
        self.assertEqual(route.program_report.coverage_basis, "map")

    def test_source_context_cannot_be_replaced_with_city_authorization(self):
        route = _route()
        self.assertTrue(limited_program_problems(route, _request(), allow_city_limited_program=True))
        self.assertEqual(limited_program_problems(route, _request(), allow_limited_program=True), [])
        route.program_report.coverage_basis = "map"
        self.assertTrue(limited_program_problems(route, _request(), allow_limited_program=True))

    def test_full_city_needs_no_free_day_permission_or_report(self):
        route = city_route(active=range(1, 8))
        self.assertIsNone(route.program_report)
        self.assertEqual(limited_program_problems(route, request()), [])
        self.assertEqual(program_report_text(route), "")

    def test_conflicting_or_nonboolean_flags_fail_even_without_free_days(self):
        for update in ({"source_schedule": True}, {"city_schedule": 1}, {"city_schedule": "true"},
                       {"source_schedule": 0}, {"source_schedule": "false"}):
            with self.subTest(update=update):
                route = city_route(active=range(1, 8)).model_copy(update=update)
                self.assertTrue(limited_program_problems(route, request(), allow_city_limited_program=True))

    def test_city_marker_cannot_authorize_region_or_invented_base(self):
        route = city_route()
        self.assertTrue(limited_program_problems(route, _request(), allow_city_limited_program=True))
        for day_index in (0, 3):
            with self.subTest(day_index=day_index):
                changed = route.model_copy(deep=True)
                changed.days[day_index].base_area = "Выдуманный отель"
                self.assertTrue(limited_program_problems(changed, request(), allow_city_limited_program=True))

    def test_report_basis_and_model_copy_bypasses_are_rechecked(self):
        for update in ({"coverage_basis": "source"}, {"coverage_basis": "verified"},
                       {"free_days": True}, {"selected_unique_visits": 9}):
            with self.subTest(update=update):
                route = city_route()
                route.program_report = route.program_report.model_copy(update=update)
                self.assertTrue(limited_program_problems(route, request(), allow_city_limited_program=True))
                self.assertEqual(program_report_text(route), "Сведения о свободных днях требуют перепроверки.")

    def test_first_and_middle_free_days_do_not_need_a_fictitious_base(self):
        route = city_route(active=(2, 4, 6))
        self.assertEqual(authorized_free_days(route, request(), allow_city_limited_program=True), frozenset({1, 3, 5, 7}))
        self.assertTrue(all(day.base_area == "" for day in route.days))

    def test_all_free_missing_visit_or_wrong_calendar_is_rejected(self):
        for change in ("all_free", "no_visit", "date", "order"):
            with self.subTest(change=change):
                route = city_route()
                if change == "all_free":
                    for day in route.days:
                        day.day_mode = "free"
                        day.items = [item for item in day.items if item.kind != "attraction"]
                elif change == "no_visit":
                    route.days[0].items = [item for item in route.days[0].items if item.kind != "attraction"]
                    route.program_report.selected_unique_visits -= 1
                elif change == "date":
                    route.days[3].date = route.days[2].date
                else:
                    route.days.reverse()
                self.assertTrue(limited_program_problems(route, request(), allow_city_limited_program=True))

    def test_free_days_cannot_hide_places_transfers_or_extra_visits(self):
        for mutation in ("query", "coords", "provider", "extra", "transfer", "alternative"):
            with self.subTest(mutation=mutation):
                route = city_route()
                day, item = route.days[3], route.days[3].items[0]
                if mutation == "query": item.place_query = "Музей, Тестгород, Россия"
                elif mutation == "coords": item.lat, item.lon = 55.0, 83.0
                elif mutation == "provider": item.provider = "osm"
                elif mutation == "extra": day.items.append(route.days[0].items[0].model_copy())
                elif mutation == "transfer": day.transfer_note = "Скрытый переезд"
                else: day.rainy_alternative = "Музей без проверки"
                self.assertTrue(limited_program_problems(route, request(), allow_city_limited_program=True))

    def test_default_budget_and_time_rules_are_not_replaced_by_report(self):
        from rules import check_program_depth, check_rules
        route, req = city_route(), request()
        self.assertEqual(check_rules(route, req, allow_city_limited_program=True), [])
        self.assertEqual(check_program_depth(route, req, allow_city_limited_program=True), [])
        for change in ("price", "time", "meal"):
            with self.subTest(change=change):
                changed = route.model_copy(deep=True)
                meal = changed.days[3].items[0]
                if change == "price": meal.cost = req.budget
                elif change == "time": meal.time = "23:59"
                else: meal.cost = minimum_meal_cost(req) - 1
                self.assertTrue(check_rules(changed, req, allow_city_limited_program=True))


class CityProgramLifecycleTests(unittest.TestCase):
    def test_rebuild_cannot_relabel_source_eligibility_as_map_coverage(self):
        route = city_route()
        route.program_report.coverage_basis = "source"
        with self.assertRaises(ValueError):
            rebuild_program_report(route, request(), allow_city_limited_program=True)
        self.assertEqual(route.program_report.coverage_basis, "source")

    def test_rebuild_requires_city_opt_in_preserves_counts_and_clears_full_report(self):
        route, req = city_route(), request()
        with self.assertRaises(ValueError):
            rebuild_program_report(route, req)
        rebuild_program_report(route, req, eligible_unique_visits=3, allow_city_limited_program=True)
        self.assertEqual(route.program_report.eligible_unique_visits, 7)
        self.assertEqual(route.program_report.coverage_basis, "map")
        full = city_route(active=range(1, 8))
        route.days = full.days
        rebuild_program_report(route, req, allow_city_limited_program=True)
        self.assertIsNone(route.program_report)

    def test_map_text_does_not_claim_source_verified_conditions(self):
        text = program_report_text(city_route())
        self.assertIn("Подтверждено мест на карте: 7", text)
        self.assertIn("но не условия посещения", text)
        self.assertIn("Резерв на обед учтён во все дни", text)
        self.assertNotIn("Проверено подходящих мест", text)
        self.assertIn("Проверено подходящих мест", program_report_text(_route()))


if __name__ == "__main__":
    unittest.main()
