"""City ambiguity stops generation and editing before any paid completion."""
import unittest
from unittest.mock import patch

import planner
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest
from weather import CityResolutionError


class PlannerCityIdentityTests(unittest.TestCase):
    def test_query_completion_preserves_explicit_province(self):
        for query in ("Тестовое место", "Тестовое место, Турция", "Тестовое место, Кемер",
                      "Тестовое место, Кемер, Турция", "Тестовое место, Кемер, Анталья, Турция"):
            with self.subTest(query=query):
                route = Itinerary(title="Тест", summary="", days=[
                    ItineraryDay(day=1, date="2099-10-28", theme="Тест", items=[
                        ItineraryItem(time="09:00", title="Тестовое место", description="Тест", place_query=query)
                    ])
                ])
                planner._fill_place_query_context(route, self.request())
                planner._fill_place_query_context(route, self.request())
                self.assertEqual(route.days[0].items[0].place_query,
                                 "Тестовое место, Кемер, Анталья, Турция")

    def test_explicit_other_province_is_not_overwritten(self):
        query = "Тестовое место, Кемер, Бурдур, Турция"
        route = Itinerary(title="Тест", summary="", days=[
            ItineraryDay(day=1, date="2099-10-28", theme="Тест", items=[
                ItineraryItem(time="09:00", title="Тестовое место", description="Тест", place_query=query)
            ])
        ])
        planner._fill_place_query_context(route, self.request())
        self.assertEqual(route.days[0].items[0].place_query, query)

    def request(self):
        return TripRequest(destination="Кемер, Анталья, Турция", destination_kind="city",
                           start_date="2099-10-28", end_date="2099-10-31", budget=2500,
                           currency="EUR", adults=2, start_time="09:00", end_time="19:00")

    def invoke(self, editing, request):
        if editing:
            return planner.edit_itinerary(request, Itinerary(title="Тест", summary="", days=[]),
                                          1, "Добавить отдых")
        return planner.generate_itinerary(request)

    def test_full_destination_is_passed_and_city_ambiguity_is_not_swallowed(self):
        for editing in (False, True):
            request = self.request()
            error = CityResolutionError(request.destination, reason="ambiguous")
            with self.subTest(editing=editing), \
                    patch.object(planner, "get_weather", return_value=None), \
                    patch.object(planner, "_load_blocked_places", return_value=[]), \
                    patch.object(planner, "find_city", side_effect=error) as find, \
                    patch.object(planner, "generate_json") as completion:
                with self.assertRaises(CityResolutionError) as caught:
                    self.invoke(editing, request)
                self.assertIs(caught.exception, error)
                find.assert_called_once_with(request.destination)
                completion.assert_not_called()

    def test_weather_city_ambiguity_stops_before_coordinate_or_completion_calls(self):
        for editing in (False, True):
            request = self.request()
            error = CityResolutionError(request.destination, reason="context_mismatch")
            with self.subTest(editing=editing), \
                    patch.object(planner, "get_weather", side_effect=error), \
                    patch.object(planner, "_load_blocked_places", return_value=[]), \
                    patch.object(planner, "find_city") as find, \
                    patch.object(planner, "generate_json") as completion:
                with self.assertRaises(CityResolutionError):
                    self.invoke(editing, request)
                find.assert_not_called()
                completion.assert_not_called()


if __name__ == "__main__":
    unittest.main()
