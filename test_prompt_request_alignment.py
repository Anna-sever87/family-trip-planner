"""Офлайн-проверки согласованности цели и исключения для долгого переезда."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import planner
import web_service
from prompts import build_itinerary_system_prompt
from schemas import TripRequest


def request_for(**updates):
    fields = {
        "destination": "Байкал, Россия",
        "destination_kind": "region",
        "start_date": "2099-10-28",
        "end_date": "2099-11-03",
        "adults": 2,
        "children_ages": [],
        "budget": 60000,
        "travel_cost": 10000,
        "lodging_cost": 15000,
        "currency": "RUB",
        "pace": "средний",
        "start_time": "08:00",
        "end_time": "21:00",
        "interest_categories": ["main_sights", "parks"],
    }
    fields.update(updates)
    return TripRequest(**fields)


class TransferPromptAlignmentTests(unittest.TestCase):
    def test_system_and_user_both_allow_a_real_significant_transfer(self):
        request = request_for()
        system = build_itinerary_system_prompt(request)
        user = planner._generation_constraints(request, 5)
        for prompt in (system, user):
            with self.subTest(prompt="system" if prompt is system else "user"):
                self.assertIn("Один attraction допустим", prompt)
                self.assertIn("в день значительного переезда", prompt)
                self.assertIn("существенную часть доступного времени", prompt)
                self.assertIn("transfer_note", prompt)

    def test_transfer_does_not_waive_depth_or_count_a_note_as_road_time(self):
        prompt = planner._generation_constraints(request_for(), 5)
        self.assertIn("одной заметки без времени в расписании недостаточно", prompt)
        self.assertIn("240 минут посещений", prompt)
        self.assertIn("текст заметки о переезде в эту сумму не входят", prompt)
        self.assertIn("Не увеличивай duration_min искусственно", prompt)
        self.assertNotIn("не менее 2 attraction", prompt)


class NewRequestGoalAlignmentTests(unittest.TestCase):
    def test_empty_region_goal_uses_the_same_default_as_region_ui(self):
        for goal in ("", " \n\t "):
            with self.subTest(goal=repr(goal)):
                normalized = web_service._normalized_trip_request(request_for(trip_goal=goal))
                self.assertEqual(normalized.trip_goal, "Познакомиться с территорией")

    def test_empty_city_goal_keeps_legacy_default(self):
        normalized = web_service._normalized_trip_request(
            request_for(destination="Казань, Россия", destination_kind="city")
        )
        self.assertEqual(normalized.trip_goal, web_service.DEFAULT_TRIP_GOAL)

    def test_explicit_goal_is_not_replaced_for_either_destination_kind(self):
        for kind in ("region", "city"):
            for goal in ("Познакомиться с городом", "Увидеть природу и отдохнуть"):
                with self.subTest(kind=kind, goal=goal):
                    normalized = web_service._normalized_trip_request(
                        request_for(destination_kind=kind, trip_goal=goal)
                    )
                    self.assertEqual(normalized.trip_goal, goal)

    def test_normalization_does_not_mutate_the_source_request(self):
        request = request_for()
        before = request.model_dump()
        fields_set = set(request.model_fields_set)
        normalized = web_service._normalized_trip_request(request)
        self.assertIsNot(normalized, request)
        self.assertEqual(request.model_dump(), before)
        self.assertEqual(request.model_fields_set, fields_set)
        self.assertEqual(request.trip_goal, "")

    def test_saved_trip_read_does_not_apply_new_request_defaults(self):
        trip = {
            "request": request_for().model_dump(),
            "itinerary": {"title": "Сохранённая поездка", "summary": "Старые данные", "days": [], "tips": []},
        }
        before = deepcopy(trip)
        with patch.object(web_service.db, "get_trip", return_value=trip), patch.object(
            web_service, "_normalized_trip_request", side_effect=AssertionError("saved trip must not be normalized")
        ):
            _, request, _ = web_service._validated_trip(1, 2)
        self.assertEqual(request.trip_goal, "")
        self.assertEqual(trip, before)


if __name__ == "__main__":
    unittest.main()
