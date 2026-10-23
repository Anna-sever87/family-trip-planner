"""Офлайн-проверки измеримой содержательности длинного взрослого дня."""

import unittest

from rules import check_program_depth, check_rules, program_depth_target
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def _request(**updates) -> TripRequest:
    data = {
        "destination": "Байкал, Россия",
        "destination_kind": "region",
        "start_date": "2099-10-28",
        "end_date": "2099-10-29",
        "adults": 2,
        "children_ages": [],
        "budget": 60_000,
        "currency": "RUB",
        "pace": "средний",
        "start_time": "08:00",
        "end_time": "21:00",
    }
    data.update(updates)
    return TripRequest(**data)


def _route(*day_durations: list[int]) -> Itinerary:
    days = []
    for day_number, durations in enumerate(day_durations, start=1):
        items = [
            ItineraryItem(
                time=f"{8 + index:02d}:00",
                kind="attraction",
                title=f"Объект {day_number}.{index + 1}",
                place_query=f"Объект {day_number}.{index + 1}, Листвянка, Россия",
                description="Посещение реального объекта",
                duration_min=duration,
            )
            for index, duration in enumerate(durations)
        ]
        items.extend([
            ItineraryItem(time="12:00", kind="meal", title="Обед", description="Обед", duration_min=60, cost=2000),
            ItineraryItem(time="13:00", kind="rest", title="Отдых", description="Пауза", duration_min=60),
        ])
        days.append(ItineraryDay(
            day=day_number,
            date=f"2099-10-{27 + day_number:02d}",
            theme="Окрестности базы",
            base_area="Листвянка",
            items=items,
        ))
    return Itinerary(title="Поездка", summary="Программа поездки", days=days)


class ProgramDepthTargetTests(unittest.TestCase):
    def test_target_scales_with_window_and_rounds_up_to_quarter_hour(self):
        for end, expected in (
            ("16:00", 180), ("16:01", 180), ("17:00", 180),
            ("18:00", 180), ("18:01", 195), ("19:00", 210),
            ("20:00", 225), ("21:00", 240), ("23:59", 240),
        ):
            with self.subTest(end=end):
                self.assertEqual(program_depth_target(_request(end_time=end)), expected)

    def test_short_windows_are_excluded(self):
        self.assertIsNone(program_depth_target(_request(end_time="15:59")))

    def test_family_and_other_paces_are_excluded(self):
        for updates in ({"children_ages": [15]}, {"children_ages": [2]}, {"pace": "спокойный"}, {"pace": "насыщенный"}):
            with self.subTest(updates=updates):
                self.assertIsNone(program_depth_target(_request(**updates)))
                self.assertEqual(check_program_depth(_route([15]), _request(**updates)), [])

    def test_medium_pace_is_case_and_whitespace_tolerant(self):
        self.assertEqual(program_depth_target(_request(pace=" Средний ")), 240)

    def test_both_time_fields_must_be_explicit(self):
        for explicit in ({}, {"start_time": "08:00"}, {"end_time": "21:00"}):
            request = TripRequest(destination="Казань", start_date="2099-10-28", end_date="2099-10-28", budget=60_000, **explicit)
            with self.subTest(explicit=explicit):
                self.assertIsNone(program_depth_target(request))
                self.assertEqual(check_program_depth(_route([15]), request), [])

    def test_invalid_or_reversed_times_do_not_add_a_quality_error(self):
        for start, end in (("bad", "21:00"), ("08:00", "24:00"), ("21:00", "08:00"), ("08:00", "08:00")):
            with self.subTest(start=start, end=end):
                request = _request(start_time=start, end_time=end)
                self.assertIsNone(program_depth_target(request))
                self.assertEqual(check_program_depth(_route([15]), request), [])


class ProgramDepthCheckTests(unittest.TestCase):
    def test_short_single_visit_is_rejected_with_actionable_numbers(self):
        problems = check_program_depth(_route([60]), _request())
        self.assertEqual(len(problems), 1)
        self.assertIn("День 1: недостаточно содержательная программа", problems[0])
        self.assertIn("посещения 60 мин.", problems[0])
        self.assertIn("доказанная нижняя оценка дороги 0 мин.", problems[0])
        self.assertIn("ориентир 240 мин.", problems[0])
        self.assertIn("не растягивай короткие остановки искусственно", problems[0])

    def test_two_short_stops_do_not_bypass_the_check(self):
        problems = check_program_depth(_route([20, 20]), _request())
        self.assertEqual(len(problems), 1)
        self.assertIn("посещения 40 мин.", problems[0])

    def test_three_hours_is_not_enough_for_thirteen_hour_window(self):
        self.assertEqual(len(check_program_depth(_route([180]), _request())), 1)

    def test_one_genuinely_long_visit_does_not_require_a_second_place(self):
        self.assertEqual(check_program_depth(_route([240]), _request()), [])

    def test_sum_equal_to_target_is_accepted(self):
        self.assertEqual(check_program_depth(_route([90, 90]), _request(end_time="16:00")), [])

    def test_meal_and_rest_do_not_count_as_program_depth(self):
        route = _route([30])
        for item in route.days[0].items:
            if item.kind != "attraction":
                item.duration_min = 240
        self.assertIn("посещения 30 мин.", check_program_depth(route, _request())[0])

    def test_base_change_and_transfer_text_are_not_evidence(self):
        route = _route([240], [30])
        route.days[1].base_area = "Слюдянка"
        route.days[1].transfer_note = "Весь день занимает переезд, 12 часов"
        problems = check_program_depth(route, _request())
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("День 2:"))
        self.assertIn("нижняя оценка дороги 0 мин.", problems[0])

    def test_numeric_proven_travel_can_complete_the_target(self):
        self.assertEqual(check_program_depth(_route([90]), _request(), {1: 150}), [])

    def test_fractional_travel_is_not_rounded_up_to_pass(self):
        problems = check_program_depth(_route([90]), _request(), {1: 149.5})
        self.assertEqual(len(problems), 1)
        self.assertIn("дороги 149.5 мин.", problems[0])

    def test_invalid_travel_values_are_ignored(self):
        for value in (-10, float("nan"), float("inf"), -float("inf"), True, "150", "переезд 5 часов", None, 10 ** 400):
            with self.subTest(value=value):
                problems = check_program_depth(_route([90]), _request(), {1: value})
                self.assertEqual(len(problems), 1)
                self.assertIn("нижняя оценка дороги 0 мин.", problems[0])

    def test_travel_credit_is_scoped_to_its_day(self):
        problems = check_program_depth(_route([90], [90]), _request(), {2: 150})
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("День 1:"))

    def test_only_day_does_not_revalidate_other_days_during_edit(self):
        route = _route([15], [240])
        self.assertEqual(check_program_depth(route, _request(), only_day=2), [])
        self.assertEqual(len(check_program_depth(route, _request(), only_day=1)), 1)
        self.assertEqual(check_program_depth(route, _request(), only_day=3), [])

    def test_quality_checker_does_not_mutate_inputs(self):
        route = _route([60], [90])
        request = _request()
        travel = {2: 150}
        before = route.model_dump(), request.model_dump(), dict(travel)
        check_program_depth(route, request, travel)
        self.assertEqual((route.model_dump(), request.model_dump(), travel), before)

    def test_existing_check_rules_does_not_gain_the_new_quality_gate(self):
        request = _request(end_date="2099-10-28", end_time="16:00")
        route = _route([60])
        self.assertEqual(check_rules(route, request), [])
        self.assertEqual(len(check_program_depth(route, request)), 1)


if __name__ == "__main__":
    unittest.main()
