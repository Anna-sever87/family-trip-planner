"""Условная переправа не является готовым переездом; без сети и реальной БД."""

import unittest
from unittest.mock import patch

import destination_scope
import planner
from rules import check_program_depth
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def _itinerary(note):
    return Itinerary(title="Две базы", summary="Тест", days=[
        ItineraryDay(
            day=number, date=f"2099-06-0{number}", theme="Местные объекты",
            base_area=f"База {number}", transfer_note=note if number == 2 else "",
            items=[ItineraryItem(
                time="12:00", title=f"Музей {number}", description="Посещение",
                duration_min=160, place_query=f"Музей {number}, База {number}, Россия",
                verified=True, lat=52, lon=105 if number == 1 else 106.1,
            )],
        ) for number in (1, 2)
    ])


def _base(query):
    return {"name": query, "lat": 52, "lon": 105 if query.startswith("База 1") else 106.1,
            "type": "village", "category": "place", "address": {"country_code": "ru"}}


SCOPE = destination_scope.RegionScope("Тестовая территория", "Россия", (51, 53, 104, 108))


class UnconfirmedWaterTransferTests(unittest.TestCase):
    def test_explicit_unknown_availability_and_conditions_are_rejected(self):
        for note in (
            "Сезонный паром: расписание и доступность не подтверждены",
            "Неподтверждённая переправа между базами.",
            "Условный переезд на пароме.",
            "Паром: расписание на эту дату ещё не подтверждено.",
            "Катер: уточните доступность рейса.",
            "Переправа, расписание необходимо проверить.",
            "Ледовая дорога: состояние льда не проверено.",
            "Переезд по льду, если разрешено движение.",
            "Паром при наличии рейса.",
            "На лодке при благоприятной погоде.",
            "Ледовая переправа зависит от ледовой обстановки.",
        ):
            with self.subTest(note=note):
                self.assertTrue(destination_scope._has_unconfirmed_water_transfer(note))

    def test_absent_crossings_and_unrelated_prices_are_not_flagged(self):
        for note in (
            "Паром не требуется; стоимость такси уточните.",
            "Без переправы, по автомобильной дороге; расписание такси не подтверждено.",
            "Без сезонной переправы, расписание такси неизвестно.",
            "Переправа не нужна, едем по суше.",
            "Не требуется паром, автомобильная дорога.",
            "Без парома и катера; цены уточните.",
            "Ледовая переправа не используется.",
            "На такси, стоимость и расписание необходимо уточнить.",
            "Паром: уточните цену билета.",
            "Паром по расписанию, стоимость такси не подтверждена.",
            "Сезонный паром между базами, ориентировочно 85 минут.",
            "Проверить дорогу и цены такси перед поездкой.",
            "",
        ):
            with self.subTest(note=note):
                self.assertFalse(destination_scope._has_unconfirmed_water_transfer(note))

    def test_negated_ferry_does_not_hide_a_different_conditional_crossing(self):
        note = "Паром не требуется, но катер зависит от погоды."
        self.assertTrue(destination_scope._has_unconfirmed_water_transfer(note))

    def test_error_excludes_day_from_travel_collector_without_extra_lookups(self):
        note = "Сезонный паром: расписание и доступность не подтверждены"
        minutes = {99: 900}
        with patch.object(destination_scope.places, "geocode", side_effect=_base) as geocode:
            problems = destination_scope.check_region_itinerary(
                _itinerary(note), SCOPE, travel_minutes_by_day=minutes,
            )
        self.assertEqual(geocode.call_count, 2)
        self.assertEqual(len(problems), 1, problems)
        self.assertTrue(problems[0].startswith("День 2:"))
        self.assertIn("неподтверждённого", problems[0])
        self.assertIn("Исключи этот переезд", problems[0])
        self.assertEqual(minutes, {1: 0})

    def test_sufficient_geometric_gap_is_not_evidence_of_water_availability(self):
        with patch.object(destination_scope.places, "geocode", side_effect=_base):
            normal_minutes = {}
            self.assertEqual(destination_scope.check_region_itinerary(
                _itinerary("Автомобильный переезд, ориентир 85 минут."), SCOPE,
                travel_minutes_by_day=normal_minutes,
            ), [])
            unsafe_minutes = {}
            problems = destination_scope.check_region_itinerary(
                _itinerary("Паром, если рейсы доступны."), SCOPE,
                travel_minutes_by_day=unsafe_minutes,
            )
        self.assertGreater(normal_minutes[2], 75)
        self.assertTrue(problems)
        self.assertNotIn(2, unsafe_minutes)

    def test_same_base_does_not_bypass_an_explicitly_conditional_transfer(self):
        itinerary = _itinerary("Катер при наличии рейса.")
        itinerary.days[1].base_area = itinerary.days[0].base_area
        with patch.object(destination_scope.places, "geocode", side_effect=_base):
            problems = destination_scope.check_region_itinerary(itinerary, SCOPE)
        self.assertTrue(any("неподтверждённого" in problem for problem in problems), problems)

    def test_planner_map_collector_and_program_depth_do_not_credit_unsafe_day(self):
        itinerary = _itinerary("Сезонный паром: расписание и доступность не подтверждены")
        minutes = {2: 85}
        with patch.object(destination_scope.places, "geocode", side_effect=_base), \
                patch.object(planner, "verify_itinerary", return_value=[]), \
                patch.object(planner, "check_verified_place_duplicates", return_value=[]):
            problems = planner._map_verification_problems(
                itinerary, None, SCOPE, travel_minutes_by_day=minutes,
            )
        self.assertTrue(any("неподтверждённого" in problem for problem in problems), problems)
        self.assertEqual(minutes, {1: 0})
        request = TripRequest(destination="Тестовая территория, Россия", destination_kind="region",
                              start_date="2099-06-01", end_date="2099-06-02", budget=10000,
                              start_time="09:00", end_time="20:00")
        depth_problems = check_program_depth(itinerary, request, minutes, only_day=2)
        self.assertTrue(depth_problems)
        self.assertIn("оценка дороги 0 мин.", depth_problems[0])


if __name__ == "__main__":
    unittest.main()
