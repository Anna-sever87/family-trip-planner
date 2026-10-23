"""Offline city-wire contract regressions; no sources or model calls."""

from copy import deepcopy
from datetime import date, timedelta
import json
import socket
import unittest
from unittest.mock import patch

from city_response import CITY_RESPONSE_FORMAT_RULE, build_city_response_schema
from grounded_response import build_grounded_response_schema, decode_grounded_response
import planner
from pricing import minimum_meal_cost
from schemas import Itinerary, TripRequest


def city_request(**updates):
    fields = dict(destination="Тестовый город, Россия", destination_kind="city",
                  start_date="2099-06-01", end_date="2099-06-02", adults=2,
                  children_ages=[5], budget=100000, currency="RUB", pace="средний",
                  start_time="09:00", end_time="19:00")
    fields.update(updates)
    return TripRequest(**fields)


def wire_item(kind, request):
    return dict(time={"attraction": "10:00", "meal": "13:00", "rest": "15:00"}[kind],
                kind=kind, title={"attraction": "Музей ремёсел", "meal": "Обед", "rest": "Отдых"}[kind],
                area="Центр", place_query="Музей ремёсел, Тестовый город, Россия" if kind == "attraction" else "",
                description="Исходное описание без новых фактов", duration_min=60,
                cost=minimum_meal_cost(request) if kind == "meal" else 0,
                kids_note="Учитывайте возраст ребёнка" if request.children_ages else "", indoor=False)


def city_wire(request):
    payload = dict(title="Маршрут", summary="Исходное описание", tips=["Сверяйте условия посещения"])
    for number in range(1, request.days_count + 1):
        payload[f"day_{number}"] = dict(
            day=number, date=(date.fromisoformat(request.start_date) + timedelta(days=number - 1)).isoformat(),
            theme="Городская программа", base_area="", transfer_note="", rainy_alternative="",
            attractions=[wire_item("attraction", request)], meal=wire_item("meal", request),
            rest=wire_item("rest", request))
    return payload


class CityResponseTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("No network"))
        network.start()
        self.addCleanup(network.stop)
        self.request = city_request()
        self.canonical = planner.build_itinerary_json_schema(self.request)
        self.schema = build_city_response_schema(self.canonical, self.request)
        self.wire = city_wire(self.request)

    def decode(self, payload, schema=None):
        return json.loads(decode_grounded_response(json.dumps(payload, ensure_ascii=False), schema or self.schema))

    def test_city_without_grounding_has_exact_days_and_does_not_mutate_canonical(self):
        original = deepcopy(self.canonical)
        schema = build_city_response_schema(self.canonical, self.request)
        self.assertEqual(self.canonical, original)
        self.assertEqual(set(schema["properties"]), {"title", "summary", "tips", "day_1", "day_2"})
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        for number in (1, 2):
            day = schema["properties"][f"day_{number}"]["properties"]
            self.assertEqual(day["day"]["const"], number)
            self.assertEqual(day["date"]["const"], f"2099-06-0{number}")
            self.assertNotIn("items", day)
            self.assertNotIn("enum", day["attractions"]["items"]["properties"]["place_query"])
        schema["properties"]["day_1"]["properties"]["rest"]["properties"]["duration_min"]["minimum"] = 80
        self.assertEqual(schema["properties"]["day_2"]["properties"]["rest"]["properties"]["duration_min"]["minimum"], 60)
        self.assertEqual(self.canonical, original)

    def test_only_query_pattern_and_young_child_rest_minimum_differ_from_adapter(self):
        original = build_grounded_response_schema(self.canonical, self.request)
        actual = deepcopy(self.schema)
        for key in ("day_1", "day_2"):
            props = actual["properties"][key]["properties"]
            self.assertEqual(props["attractions"]["items"]["properties"]["place_query"].pop("pattern"), r"\S")
            props["rest"]["properties"]["duration_min"]["minimum"] = original["properties"][key]["properties"]["rest"]["properties"]["duration_min"]["minimum"]
        self.assertEqual(actual, original)

    def test_existing_age_and_pace_attraction_caps_are_preserved_with_minimum_one(self):
        for ages, pace, expected in (([], "средний", 5), ([2], "средний", 2), ([5], "средний", 3),
                                     ([4, 10], "средний", 3), ([10], "средний", 4), ([5], "спокойный", 2)):
            with self.subTest(ages=ages, pace=pace):
                request = city_request(children_ages=ages, pace=pace)
                schema = build_city_response_schema(planner.build_itinerary_json_schema(request), request)
                attractions = schema["properties"]["day_1"]["properties"]["attractions"]
                self.assertEqual((attractions["minItems"], attractions["maxItems"]), (1, expected))
                payload = city_wire(request)
                self.assertEqual(len(self.decode(payload, schema)["days"][0]["items"]), 3)
                payload["day_1"]["attractions"] *= expected + 1
                with self.assertRaises(ValueError):
                    self.decode(payload, schema)

    def test_child_rest_boundary_and_older_or_adult_minimum(self):
        for ages, minimum in (([5], 60), ([10, 4], 60), ([6], 60), ([7], 15), ([], 15)):
            with self.subTest(ages=ages):
                request = city_request(children_ages=ages)
                schema = build_city_response_schema(planner.build_itinerary_json_schema(request), request)
                payload = city_wire(request)
                payload["day_1"]["rest"]["duration_min"] = minimum
                self.assertEqual(self.decode(payload, schema)["days"][0]["items"][-1]["duration_min"], minimum)
                payload["day_1"]["rest"]["duration_min"] = minimum - 1
                with self.assertRaises(ValueError):
                    self.decode(payload, schema)

    def test_meal_floor_uses_whole_group_and_currency_without_filling_costs(self):
        for currency, expected in (("RUB", 2200), ("EUR", 38), ("USD", 38)):
            with self.subTest(currency=currency):
                request = city_request(children_ages=[4, 10], currency=currency)
                schema = build_city_response_schema(planner.build_itinerary_json_schema(request), request)
                payload = city_wire(request)
                self.assertEqual(self.decode(payload, schema)["days"][0]["items"][1]["cost"], expected)
                payload["day_1"]["meal"]["cost"] = expected - 1
                with self.assertRaises(ValueError):
                    self.decode(payload, schema)
                self.assertEqual(payload["day_1"]["meal"]["cost"], expected - 1)

    def test_empty_or_whitespace_visit_queries_are_rejected(self):
        for query in ("", " ", "\t\n", "\u00a0\u2003"):
            with self.subTest(query=query):
                payload = deepcopy(self.wire)
                payload["day_1"]["attractions"][0]["place_query"] = query
                with self.assertRaises(ValueError):
                    self.decode(payload)
        payload = deepcopy(self.wire)
        payload["day_1"]["attractions"][0]["place_query"] = " Музей, Другой город, Другая страна "
        self.assertEqual(self.decode(payload)["days"][0]["items"][0]["place_query"], payload["day_1"]["attractions"][0]["place_query"])

    def test_zero_visits_multiple_rests_and_wrong_kinds_fail_closed(self):
        variants = []
        payload = deepcopy(self.wire)
        payload["day_1"]["attractions"] = []
        variants.append(payload)
        payload = deepcopy(self.wire)
        payload["day_1"]["rest"] = [deepcopy(payload["day_1"]["rest"]) for _ in range(3)]
        variants.append(payload)
        for field in ("attractions", "meal", "rest"):
            payload = deepcopy(self.wire)
            item = payload["day_1"][field][0] if field == "attractions" else payload["day_1"][field]
            item["kind"] = "rest" if field != "rest" else "meal"
            variants.append(payload)
        for payload in variants:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.decode(payload)

    def test_decode_preserves_content_and_rain_flags_without_trust_or_free_fields(self):
        before = deepcopy(self.wire)
        self.wire["day_1"]["attractions"][0]["time"] = "16:00"
        result = self.decode(self.wire)
        expected = {key: self.wire[key] for key in ("title", "summary", "tips")}
        expected["days"] = []
        for number in (1, 2):
            wire_day = self.wire[f"day_{number}"]
            day = {key: value for key, value in wire_day.items() if key not in {"attractions", "meal", "rest"}}
            day["items"] = sorted([*wire_day["attractions"], wire_day["meal"], wire_day["rest"]], key=lambda item: item["time"])
            expected["days"].append(day)
        self.assertEqual(result, expected)
        itinerary = Itinerary.model_validate(result)
        self.assertFalse(itinerary.source_schedule)
        self.assertIsNone(itinerary.program_report)
        self.assertTrue(all(day.day_mode == "program" for day in itinerary.days))
        self.assertTrue(all(not item.verified and item.lat is None and item.lon is None for day in itinerary.days for item in day.items))
        self.assertTrue(all(not item.indoor for day in itinerary.days for item in day.items))
        self.assertEqual(self.wire["day_2"], before["day_2"])

    def test_exact_dates_extra_fields_and_legacy_items_are_still_rejected(self):
        mutations = (("day", 2), ("date", "2099-06-02"), ("day_mode", "free"), ("items", []))
        for field, value in mutations:
            payload = deepcopy(self.wire)
            payload["day_1"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.decode(payload)
        for field in ("verified", "source_id", "lat"):
            payload = deepcopy(self.wire)
            payload["day_1"]["attractions"][0][field] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.decode(payload)
        with self.assertRaises(ValueError):
            self.decode(self.decode(self.wire))

    def test_original_bounds_and_adult_kids_note_constraint_remain_checked(self):
        for field, value in (("cost", 100001), ("description", "Я" * 181), ("duration_min", 241),
                             ("area", ""), ("time", "9:00"), ("indoor", 1)):
            payload = deepcopy(self.wire)
            payload["day_1"]["attractions"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.decode(payload)
        request = city_request(children_ages=[])
        schema = build_city_response_schema(planner.build_itinerary_json_schema(request), request)
        payload = city_wire(request)
        payload["day_1"]["attractions"][0]["kids_note"] = "Для детей"
        with self.assertRaises(ValueError):
            self.decode(payload, schema)

    def test_existing_stricter_duration_and_choices_are_preserved_or_fail_closed(self):
        item = self.canonical["properties"]["days"]["items"]["properties"]["items"]["items"]["properties"]
        item["duration_min"]["minimum"] = 90
        schema = build_city_response_schema(self.canonical, self.request)
        self.assertEqual(schema["properties"]["day_1"]["properties"]["rest"]["properties"]["duration_min"]["minimum"], 90)
        item["duration_min"]["minimum"] = 15
        item["duration_min"]["enum"] = [30, 60]
        schema = build_city_response_schema(self.canonical, self.request)
        self.assertEqual(schema["properties"]["day_1"]["properties"]["rest"]["properties"]["duration_min"]["enum"], [60])
        item["duration_min"]["enum"] = [30]
        with self.assertRaises(ValueError):
            build_city_response_schema(self.canonical, self.request)
        item["duration_min"].pop("enum")
        item["duration_min"]["maximum"] = 59
        with self.assertRaises(ValueError):
            build_city_response_schema(self.canonical, self.request)

    def test_existing_query_pattern_is_not_silently_weakened(self):
        query = self.canonical["properties"]["days"]["items"]["properties"]["items"]["items"]["properties"]["place_query"]
        query["pattern"] = "Музей"
        original = deepcopy(self.canonical)
        with self.assertRaises(ValueError):
            build_city_response_schema(self.canonical, self.request)
        self.assertEqual(self.canonical, original)

    def test_region_request_is_rejected_without_altering_legacy_adapter(self):
        request = city_request(destination_kind="region")
        canonical = planner.build_itinerary_json_schema(request)
        legacy = build_grounded_response_schema(canonical, request)
        with self.assertRaises(ValueError):
            build_city_response_schema(canonical, request)
        self.assertEqual(build_grounded_response_schema(canonical, request), legacy)

    def test_format_rule_is_neutral_and_retains_quality_obligations(self):
        self.assertIn("ФОРМАТ ОТВЕТА ДЛЯ ИСПРАВЛЕНИЯ ГОРОДСКОГО МАРШРУТА", CITY_RESPONSE_FORMAT_RULE)
        self.assertNotIn("ПОДТВЕРЖДЁННОГО", CITY_RESPONSE_FORMAT_RULE)
        self.assertNotIn("request_rules", CITY_RESPONSE_FORMAT_RULE)
        for text in ("attractions", "meal", "rest", "60 минут", "не подтверждает", "бюджета", "погодной пригодности", "содержательности"):
            self.assertIn(text, CITY_RESPONSE_FORMAT_RULE)


if __name__ == "__main__":
    unittest.main()
