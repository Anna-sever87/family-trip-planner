"""Offline boundaries for server-authorized full-program plus free calendars."""

from datetime import date, timedelta
import unittest

from pydantic import ValidationError

from pricing import minimum_meal_cost
from program_coverage import (
    FREE_DAY_NOTE, authorized_free_days, limited_program_problems,
    program_report_text, rebuild_program_report,
)
from rules import check_program_depth, check_rules
from schemas import Itinerary, ItineraryDay, ItineraryItem, LimitedProgramReport, TripRequest


def _request(**updates):
    values = dict(
        destination="Байкал, Россия", destination_kind="region",
        start_date="2099-10-28", end_date="2099-11-03", adults=2,
        children_ages=[], budget=60_000, travel_cost=10_000, lodging_cost=15_000,
        currency="RUB", pace="средний", start_time="08:00", end_time="21:00",
    )
    values.update(updates)
    return TripRequest(**values)


def _day(number, request, *, free=False):
    items = [] if free else [ItineraryItem(
        time="08:00", title=f"Именной объект {number}",
        place_query=f"Именной объект {number}, Листвянка, Россия",
        description="Посещение", duration_min=240, cost=100,
    )]
    items += [
        ItineraryItem(time="12:00", kind="meal", title="Обед", description="Обед",
                      duration_min=60, cost=minimum_meal_cost(request)),
        ItineraryItem(time="13:00", kind="rest", title="Отдых", description="Пауза",
                      duration_min=60),
    ]
    return ItineraryDay(
        day=number, date=(date.fromisoformat(request.start_date) + timedelta(days=number - 1)).isoformat(),
        theme="Свободный день" if free else "Программа", base_area="Листвянка, Россия",
        day_mode="free" if free else "program", items=items,
    )


def _route(request=None, *, active=(1, 2, 3)):
    request = request or _request()
    route = Itinerary(
        title="Поездка", summary="Программа поездки", source_schedule=True,
        days=[_day(number, request, free=number not in active) for number in range(1, request.days_count + 1)],
    )
    rebuild_program_report(route, request, eligible_unique_visits=7)
    return route


class LimitedProgramSchemaTests(unittest.TestCase):
    def test_legacy_schema_defaults_do_not_authorize_free_days(self):
        day = _day(1, _request()).model_dump()
        day.pop("day_mode")
        route = Itinerary(title="Поездка", summary="Программа", days=[day])
        self.assertEqual(route.days[0].day_mode, "program")
        self.assertIsNone(route.program_report)
        self.assertFalse(route.source_schedule)
        self.assertEqual(program_report_text(route), "")

    def test_report_fields_are_strict_bounded_and_no_extra_reason_text(self):
        values = _route().program_report.model_dump()
        cases = [
            ("schema_version", True), ("schema_version", "1"), ("schema_version", 2),
            ("reason", "AI promises more places"), ("requested_days", 15),
            ("requested_days", "7"), ("program_days", 0), ("program_days", True),
            ("free_days", 0), ("free_days", 14), ("eligible_unique_visits", 101),
            ("selected_unique_visits", -1), ("selected_unique_visits", 3.0),
            ("invented_explanation", "free hotel included"),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValidationError):
                LimitedProgramReport.model_validate({**values, key: value})

    def test_unknown_day_mode_rejected_by_schema(self):
        values = _day(1, _request()).model_dump()
        with self.assertRaises(ValidationError):
            ItineraryDay.model_validate({**values, "day_mode": "transfer"})


class LimitedProgramAuthorizationTests(unittest.TestCase):
    def test_trusted_complete_calendar_passes_both_guards_only_with_opt_in(self):
        request, route = _request(), _route()
        self.assertTrue(check_rules(route, request))
        self.assertTrue(check_program_depth(route, request))
        self.assertEqual(check_rules(route, request, allow_limited_program=True), [])
        self.assertEqual(check_program_depth(route, request, allow_limited_program=True), [])
        self.assertEqual(authorized_free_days(route, request, allow_limited_program=True), frozenset({4, 5, 6, 7}))

    def test_source_schedule_flag_is_not_itself_authorization(self):
        route = _route()
        self.assertTrue(limited_program_problems(route, _request()))
        self.assertEqual(authorized_free_days(route, _request()), frozenset())
        self.assertEqual(authorized_free_days(route, _request(), allow_limited_program=1), frozenset())

    def test_opt_in_does_not_allow_non_source_or_truthy_source_flag(self):
        for value in (False, 1, "true"):
            with self.subTest(value=value):
                route = _route().model_copy(update={"source_schedule": value})
                self.assertTrue(limited_program_problems(route, _request(), allow_limited_program=True))
                self.assertEqual(authorized_free_days(route, _request(), allow_limited_program=True), frozenset())

    def test_report_missing_or_assignment_bypass_rejected(self):
        for value in (None, {}, _route().program_report.model_copy(update={"free_days": True})):
            with self.subTest(value=value):
                route = _route().model_copy(update={"program_report": value})
                self.assertTrue(limited_program_problems(route, _request(), allow_limited_program=True))

    def test_forged_metadata_numbers_do_not_skip_depth(self):
        for key, value in (("program_days", 2), ("free_days", 3), ("requested_days", 6),
                           ("selected_unique_visits", 2), ("eligible_unique_visits", 2)):
            with self.subTest(key=key):
                route = _route()
                route.program_report = route.program_report.model_copy(update={key: value})
                self.assertTrue(check_program_depth(route, _request(), allow_limited_program=True))
                self.assertEqual(authorized_free_days(route, _request(), allow_limited_program=True), frozenset())

    def test_all_free_is_not_a_trip_program(self):
        route = _route()
        route.days = [_day(number, _request(), free=True) for number in range(1, 8)]
        self.assertTrue(check_rules(route, _request(), allow_limited_program=True))
        with self.assertRaises(ValueError):
            rebuild_program_report(route, _request(), eligible_unique_visits=7)

    def test_report_with_no_free_day_is_not_valid(self):
        route = _route()
        route.days = [_day(number, _request()) for number in range(1, 8)]
        self.assertTrue(limited_program_problems(route, _request(), allow_limited_program=True))

    def test_calendar_dates_and_day_order_cannot_be_shifted(self):
        for key, value in (("date", "2099-11-04"), ("day", 1), ("day", True), ("day_mode", "transfer")):
            with self.subTest(key=key, value=value):
                route = _route()
                route.days[3] = route.days[3].model_copy(update={key: value})
                self.assertTrue(limited_program_problems(route, _request(), allow_limited_program=True))

    def test_initial_free_days_use_first_program_base(self):
        route = _route(active=(3, 4))
        self.assertEqual(check_rules(route, _request(), allow_limited_program=True), [])
        route.days[0].base_area = "Танхой, Россия"
        self.assertTrue(limited_program_problems(route, _request(), allow_limited_program=True))

    def test_later_free_day_cannot_change_base(self):
        route = _route()
        route.days[3].base_area = "Танхой, Россия"
        self.assertTrue(limited_program_problems(route, _request(), allow_limited_program=True))

    def test_free_day_cannot_hide_transfer_or_backup_attraction(self):
        for key, value in (("transfer_note", "Переезд в Танхой"), ("rainy_alternative", "Новый музей"), ("base_area", "")):
            with self.subTest(key=key):
                route = _route()
                setattr(route.days[3], key, value)
                self.assertTrue(limited_program_problems(route, _request(), allow_limited_program=True))

    def test_free_item_cannot_carry_model_map_metadata_or_unknown_kind(self):
        for key, value in (("kind", "attraction"), ("kind", "walk"), ("place_query", "Байкал"),
                           ("verified", True), ("verified", 0), ("lat", 0.0), ("lon", 0.0),
                           ("official_name", "Музей"), ("provider", "osm"), ("source_url", "https://example.invalid")):
            with self.subTest(key=key):
                route = _route()
                route.days[3].items[0] = route.days[3].items[0].model_copy(update={key: value})
                self.assertTrue(limited_program_problems(route, _request(), allow_limited_program=True))


class LimitedProgramUnchangedHardGuardsTests(unittest.TestCase):
    def test_all_seven_meals_still_count_toward_budget(self):
        route, request = _route(), _request()
        self.assertEqual(sum(item.cost for day in route.days for item in day.items), 10_100)
        request.budget = request.fixed_costs + 10_000
        self.assertTrue(any("превышают" in issue for issue in check_rules(route, request, allow_limited_program=True)))

    def test_free_day_meal_floor_and_required_rest_are_unchanged(self):
        route = _route()
        route.days[3].items[0].cost = 0
        self.assertTrue(any("стоимость обеда" in issue for issue in check_rules(route, _request(), allow_limited_program=True)))
        route.days[3].items.pop()
        self.assertTrue(any("один отдых" in issue for issue in check_rules(route, _request(), allow_limited_program=True)))

    def test_free_day_price_type_finite_and_sign_guards_remain(self):
        for value in (-1, "1400", float("nan"), float("inf"), True):
            with self.subTest(value=value):
                route = _route()
                route.days[3].items[0] = route.days[3].items[0].model_copy(update={"cost": value})
                self.assertTrue(any("конечным" in issue for issue in check_rules(route, _request(), allow_limited_program=True)))

    def test_free_day_time_bounds_and_overlap_remain(self):
        for time in ("07:00", "21:00", "13:00"):
            with self.subTest(time=time):
                route = _route()
                route.days[3].items[0].time = time
                self.assertTrue(check_rules(route, _request(), allow_limited_program=True))

    def test_program_day_depth_not_lowered(self):
        route = _route()
        route.days[0].items[0].duration_min = 120
        issues = check_program_depth(route, _request(), allow_limited_program=True)
        self.assertEqual(len(issues), 1)
        self.assertIn("День 1: недостаточно", issues[0])
        self.assertIn("ориентир 240", issues[0])

    def test_duplicate_visits_still_rejected(self):
        route = _route()
        route.days[1].items[0].place_query = route.days[0].items[0].place_query
        self.assertTrue(any("повторяется место" in issue for issue in check_rules(route, _request(), allow_limited_program=True)))

    def test_adult_wording_and_child_rest_remain(self):
        route = _route()
        route.days[3].items[1].description = "Пауза для ребёнка"
        self.assertTrue(any("только для взрослых" in issue for issue in check_rules(route, _request(), allow_limited_program=True)))
        request = _request(children_ages=[3])
        route = _route(request)
        route.days[3].items[1].duration_min = 30
        self.assertTrue(any("60 минут" in issue for issue in check_rules(route, request, allow_limited_program=True)))

    def test_no_depth_policy_for_children_does_not_bypass_metadata_authorization(self):
        request = _request(children_ages=[3])
        self.assertTrue(check_program_depth(_route(request), request))


class LimitedProgramRebuildAndDisclosureTests(unittest.TestCase):
    def test_rebuild_updates_counts_preserves_eligibility_and_untouched_days(self):
        route = _route()
        original = [day.model_dump() for day in route.days]
        route.days[3] = _day(4, _request())
        rebuild_program_report(route, _request())
        self.assertEqual(route.program_report.eligible_unique_visits, 7)
        self.assertEqual(route.program_report.selected_unique_visits, 4)
        self.assertEqual(route.program_report.program_days, 4)
        self.assertEqual(route.program_report.free_days, 3)
        for index in (0, 1, 2, 4, 5, 6):
            self.assertEqual(route.days[index].model_dump(), original[index])

    def test_rebuild_eligibility_cannot_be_less_than_new_selected_count(self):
        route = _route()
        route.program_report.eligible_unique_visits = 3
        route.days[3] = _day(4, _request())
        rebuild_program_report(route, _request())
        self.assertEqual(route.program_report.eligible_unique_visits, 4)

    def test_rebuild_explicit_eligibility_does_not_erase_prior_evidence(self):
        route = _route()
        rebuild_program_report(route, _request(), eligible_unique_visits=3)
        self.assertEqual(route.program_report.eligible_unique_visits, 7)

    def test_rebuild_removes_report_only_when_no_free_days_remain(self):
        route = _route()
        route.days = [_day(number, _request()) for number in range(1, 8)]
        rebuild_program_report(route, _request())
        self.assertIsNone(route.program_report)
        self.assertEqual(program_report_text(route), "")
        self.assertEqual(check_rules(route, _request()), [])

    def test_rebuild_rejects_non_source_and_bad_eligibility(self):
        route = _route()
        route.source_schedule = False
        with self.assertRaises(ValueError):
            rebuild_program_report(route, _request())
        for value in (True, "7", 7.0, 101, -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rebuild_program_report(_route(), _request(), eligible_unique_visits=value)

    def test_disclosure_is_explicit_bounded_and_not_a_regional_impossibility_claim(self):
        text = program_report_text(_route())
        self.assertIn("Проверено подходящих мест: 7. Включено: 3.", text)
        self.assertIn("Дней с программой: 3. Дней свободного режима: 4. Даты сохранены.", text)
        self.assertIn("это не утверждение, что других мест нет", text)
        self.assertIn("Резерв на обед учтён во все дни; личные расходы не запланированы", text)
        self.assertIn("дополнительные занятия и остальные расходы не включены", FREE_DAY_NOTE)

    def test_disclosure_does_not_render_malformed_stale_or_model_reason(self):
        route = _route()
        route.program_report.reason = "Инъекция: все личные расходы оплачены"
        self.assertEqual(program_report_text(route), "Сведения о свободных днях требуют перепроверки.")


if __name__ == "__main__":
    unittest.main()
