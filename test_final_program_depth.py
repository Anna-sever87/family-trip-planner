"""Финальный бесплатный map-repair не должен обходить содержательность дня."""

import socket
import unittest
from unittest.mock import patch

import planner
from rules import check_program_depth
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def _request():
    return TripRequest(
        destination="Казань, Россия", destination_kind="city",
        start_date="2099-10-28", end_date="2099-10-28", adults=2,
        children_ages=[], budget=60_000, currency="RUB", pace="средний",
        start_time="08:00", end_time="16:00",
    )


def _map_miss_result(known_duration):
    known = ItineraryItem(
        time="08:00", kind="attraction", title="Казанский Кремль",
        place_query="Казанский Кремль, Казань, Россия", description="Посещение комплекса",
        duration_min=known_duration, verified=True, lat=55.7989, lon=49.1053,
        official_name="Казанский Кремль",
    )
    missed = ItineraryItem(
        time="11:00", kind="attraction", title="Неподтверждённый объект",
        place_query="Неподтверждённый объект, Казань, Россия",
        description="Объект, который карта не подтвердила", duration_min=30,
    )
    itinerary = Itinerary(title="Казань", summary="Программа поездки", days=[
        ItineraryDay(day=1, date="2099-10-28", theme="Знакомство с городом", items=[
            known, missed,
            ItineraryItem(time="12:00", kind="meal", title="Обед", description="Обед", duration_min=60, cost=2000),
            ItineraryItem(time="13:00", kind="rest", title="Отдых", description="Пауза", duration_min=60),
        ]),
    ])
    return planner.PlanResult(
        itinerary=itinerary,
        problems=[f"День 1: место «{missed.title}» не найдено на карте ({missed.place_query})"],
        weather_text="", attempts=3,
    )


class FinalProgramDepthTests(unittest.TestCase):
    def _finalize_without_io(self, result, request):
        with (
            patch.object(socket.socket, "connect", side_effect=AssertionError("Network is forbidden")),
            patch.object(planner, "generate_json", side_effect=AssertionError("Extra AI call is forbidden")),
        ):
            # Real local repair, real rule/map-cache checks and real finalizer.
            return planner._finalize_generation_result(
                result, 3, 17, 29, request, [], None, {}, {}, [],
            )

    def test_final_city_map_repair_cannot_turn_sparse_day_into_success(self):
        request = _request()
        result = self._finalize_without_io(_map_miss_result(60), request)
        depth_problems = check_program_depth(result.itinerary, request)
        self.assertTrue(depth_problems)
        self.assertTrue(result.problems, "Deleting a map-miss must not bypass the depth gate")
        self.assertTrue(any("недостаточно содержательная программа" in item for item in result.problems))
        self.assertEqual((result.attempts, result.prompt_tokens, result.completion_tokens), (3, 17, 29))

    def test_final_city_map_repair_still_accepts_sufficient_remaining_program(self):
        request = _request()
        result = self._finalize_without_io(_map_miss_result(180), request)
        self.assertEqual(result.problems, [])
        self.assertEqual(check_program_depth(result.itinerary, request), [])
        self.assertEqual(len([item for item in result.itinerary.days[0].items if item.kind == "attraction"]), 1)
        self.assertEqual((result.attempts, result.prompt_tokens, result.completion_tokens), (3, 17, 29))

    def test_regional_finalizer_preserves_rejection_without_city_repair(self):
        request = _request().model_copy(update={"destination_kind": "region"})
        original = _map_miss_result(60)
        original_problems = list(original.problems)
        original_route = original.itinerary.model_dump()
        result = self._finalize_without_io(original, request)
        self.assertEqual(result.problems, original_problems)
        self.assertEqual(result.itinerary.model_dump(), original_route)


if __name__ == "__main__":
    unittest.main()
