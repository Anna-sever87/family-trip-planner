"""Контекст карты и безопасная диагностика AI без внешних запросов."""
import json
import unittest
from unittest.mock import patch

import httpx
import planner
import yandex_ai
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def request(kind="city"):
    return TripRequest(destination="Владивосток, Россия", destination_kind=kind,
                       start_date="2099-11-09", end_date="2099-11-10", budget=300000,
                       currency="RUB", start_time="11:00", end_time="18:00")


def itinerary(query="Владивостокский драматический театр, Россия"):
    return Itinerary(title="Проверка", summary="Проверка карты", days=[
        ItineraryDay(day=number, date=f"2099-11-{number + 8:02d}", theme="Театр", items=[
            ItineraryItem(time="11:00", title="Конкретный театр", description="Посещение", place_query=query),
        ]) for number in (1, 2)
    ])


class PlannerContextTests(unittest.TestCase):
    def test_missing_city_filled_without_duplicate_context(self):
        for query in ("Владивостокский драматический театр, Россия", "Владивостокский драматический театр", "Владивостокский драматический театр, Владивосток"):
            route = itinerary(query)
            planner._fill_place_query_context(route, request())
            self.assertEqual(route.days[0].items[0].place_query, "Владивостокский драматический театр, Владивосток, Россия")
            planner._fill_place_query_context(route, request())
            self.assertEqual(route.days[0].items[0].place_query.count("Владивосток,"), 1)

    def test_other_city_and_address_are_not_rewritten(self):
        for query in ("Театр, Москва, Россия", "улица Ленина, 10, Владивосток, Россия", "Musée d'Orsay, Paris, France"):
            route = itinerary(query)
            planner._fill_place_query_context(route, request())
            self.assertEqual(route.days[0].items[0].place_query, query)

    def test_region_and_unchanged_day_not_touched(self):
        route = itinerary()
        original = route.model_dump()
        planner._fill_place_query_context(route, request("region"))
        self.assertEqual(route.model_dump(), original)
        planner._fill_place_query_context(route, request(), target_day=2)
        self.assertEqual(route.days[0].model_dump(), original["days"][0])
        self.assertIn(", Владивосток, Россия", route.days[1].items[0].place_query)

    def test_mismatch_is_rejected_but_not_a_destructive_map_miss(self):
        route = itinerary("Точное место, Владивосток, Россия")
        problem = "День 1: результат карты не соответствует месту «Точное место» (Точное место, Владивосток, Россия)"
        history = {}
        planner._remember_rejected_map_queries(route, [problem], history, {})
        self.assertTrue(history)
        self.assertIsNone(planner._repair_final_map_misses(route, [problem], request(), [], None, {}, history, []))

    def test_timeout_diagnostic_has_no_secrets_or_extra_request(self):
        failure = httpx.ReadTimeout("SECRET_PROXY_URL", request=httpx.Request("POST", "https://example.invalid"))
        with patch.object(yandex_ai.httpx, "post", side_effect=failure) as post, self.assertLogs("yandex_ai", level="WARNING") as logs:
            with self.assertRaisesRegex(yandex_ai.YandexAIError, "не ответил вовремя"):
                yandex_ai._send_request({"private": "SECRET_BODY"}, {"Authorization": "SECRET_KEY"})
        self.assertEqual(post.call_count, 1)
        text = "\n".join(logs.output)
        self.assertIn("ReadTimeout", text)
        self.assertNotIn("SECRET", text)


if __name__ == "__main__":
    unittest.main()
