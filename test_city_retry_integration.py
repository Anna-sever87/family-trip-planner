"""City retry changes its wire shape, not the downstream acceptance rules."""
from contextlib import ExitStack
from copy import deepcopy
import json
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import planner
from schemas import TripRequest


def request(children=(5,)):
    return TripRequest(destination="Тестгород, Россия", start_date="2099-10-28", end_date="2099-10-28",
                       adults=2, children_ages=list(children), budget=30000, currency="RUB",
                       pace="средний", start_time="09:00", end_time="19:00")


def payload(*, rest_duration=60, children=True, attraction_duration=90):
    def item(kind, time, title, duration, cost=0):
        return dict(kind=kind, time=time, title=title, area="Тестгород", duration_min=duration,
                    description="Описание посещения" if kind == "attraction" else "Перерыв",
                    cost=cost, kids_note="Пауза по необходимости" if children and kind == "attraction" else "",
                    indoor=True, place_query=f"{title}, Тестгород, Россия" if kind == "attraction" else "")
    return dict(title="Тестовая программа", summary="Синтетическая проверка", tips=["Уточните часы работы"], days=[dict(
        day=1, date="2099-10-28", theme="История города", base_area="", transfer_note="", rainy_alternative="",
        items=[item("attraction", "09:00", "Городской исторический музей", attraction_duration),
               item("meal", "13:00", "Обед", 60, 1800 if children else 1400),
               item("rest", "14:00", "Отдых", rest_duration)],
    )])


def wire(value):
    result = deepcopy(value)
    for day in result.pop("days"):
        items = day.pop("items")
        day["attractions"] = [item for item in items if item["kind"] == "attraction"]
        day["meal"] = next(item for item in items if item["kind"] == "meal")
        day["rest"] = next(item for item in items if item["kind"] == "rest")
        result[f"day_{day['day']}"] = day
    return result


class CityRetryIntegrationTests(unittest.TestCase):
    def run_plan(self, responses, *, children=(5,), map_problems=()):
        with ExitStack() as stack:
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("No network")))
            for name, result in (("get_weather", None), ("find_city", None), ("curated_place_hints", []),
                                 ("cached_place_hints", []), ("curated_blocked_place_terms", [])):
                stack.enter_context(patch.object(planner, name, return_value=result))
            verify = stack.enter_context(patch.object(planner, "_map_verification_problems", return_value=list(map_problems)))
            generate = stack.enter_context(patch.object(planner, "generate_json", side_effect=[
                SimpleNamespace(text=json.dumps(value, ensure_ascii=False), input_tokens=11, output_tokens=13)
                for value in responses
            ]))
            result = planner.generate_itinerary(request(children), max_attempts=len(responses))
        return result, generate, verify

    def test_accepted_first_city_answer_keeps_existing_contract(self):
        result, generate, verify = self.run_plan([payload()])
        self.assertEqual(result.problems, [])
        self.assertEqual(generate.call_count, 1)
        self.assertIn("days", generate.call_args.kwargs["json_schema"]["properties"])
        self.assertGreaterEqual(verify.call_count, 1)

    def test_invalid_city_structure_is_repaired_with_strict_wire_and_normal_checks(self):
        from city_response import CITY_RESPONSE_FORMAT_RULE
        result, generate, verify = self.run_plan([payload(rest_duration=30), wire(payload())])
        self.assertEqual(result.problems, [])
        self.assertEqual((generate.call_count, result.attempts, result.total_tokens), (2, 2, 48))
        self.assertIn("days", generate.call_args_list[0].kwargs["json_schema"]["properties"])
        second = generate.call_args_list[1]
        self.assertIn("day_1", second.kwargs["json_schema"]["properties"])
        self.assertNotIn("days", second.kwargs["json_schema"]["properties"])
        self.assertTrue(second.args[0][0]["content"].endswith(CITY_RESPONSE_FORMAT_RULE))
        self.assertEqual(sum(item.kind == "rest" for item in result.itinerary.days[0].items), 1)
        self.assertGreaterEqual(verify.call_count, 1)

    def test_wire_cannot_bypass_generic_place_rules(self):
        invalid = payload()
        invalid["days"][0]["items"][0].update(title="Музей", place_query="Музей, Тестгород, Россия")
        result, generate, verify = self.run_plan([payload(rest_duration=30), wire(invalid)])
        self.assertTrue(result.problems)
        self.assertEqual(generate.call_count, 2)
        verify.assert_not_called()

    def test_wire_cannot_bypass_map_rejection(self):
        result, generate, verify = self.run_plan([payload(rest_duration=30), wire(payload())],
                                               map_problems=["День 1: карта не подтвердила идентичность места"])
        self.assertTrue(result.problems)
        self.assertEqual(generate.call_count, 2)
        self.assertGreaterEqual(verify.call_count, 1)

    def test_wire_cannot_bypass_adult_program_depth(self):
        sparse = payload(children=False, attraction_duration=120)
        result, generate, verify = self.run_plan([sparse, wire(sparse)], children=())
        self.assertTrue(any("недостаточно содержательная программа" in problem for problem in result.problems))
        self.assertEqual(generate.call_count, 2)

    def test_legacy_retry_shape_is_not_silently_accepted(self):
        result, generate, _ = self.run_plan([payload(rest_duration=30), payload()])
        self.assertTrue(result.problems)
        self.assertEqual(generate.call_count, 2)


if __name__ == "__main__":
    unittest.main()
