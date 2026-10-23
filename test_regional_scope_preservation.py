"""A regional retry must repair its new duplicate, not mutate a preserved day.

All external AI/map/catalogue/weather boundaries are mocked. The generator,
merge, pruning, scope selection and validation pipeline run unchanged.
"""

import json
import socket
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import planner
from rules import check_program_depth, check_rules
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest
from test_baikal_duplicate_repair import _scope


def _attraction(title, time, duration):
    return ItineraryItem(
        time=time, kind="attraction", title=title,
        place_query=f"{title}, Листвянка, Россия", area="Листвянка",
        description="Синтетическое расписание для проверки границ исправления",
        duration_min=duration,
    )


def _day(number, attractions):
    return ItineraryDay(
        day=number, date=f"2099-10-{27 + number:02d}", theme=f"Программа дня {number}",
        base_area="Листвянка", items=attractions + [
            ItineraryItem(time="12:00", kind="meal", title="Обед", description="Обед", duration_min=60, cost=2000),
            ItineraryItem(time="13:00", kind="rest", title="Отдых", description="Пауза", duration_min=60),
        ],
    )


def _fixture(*, neighbor_has_second):
    request = TripRequest(
        destination="Байкал, Россия", destination_kind="region",
        start_date="2099-10-28", end_date="2099-10-29", adults=2,
        children_ages=[], budget=60_000, currency="RUB", pace="средний",
        start_time="08:00", end_time="21:00",
    )
    neighbor_attractions = [_attraction("Шаман-камень", "08:00", 60 if neighbor_has_second else 240)]
    if neighbor_has_second:
        neighbor_attractions.append(_attraction("Ретро-парк Осиповых", "14:00", 240))
    valid = Itinerary(title="Байкал", summary="Программа поездки", days=[
        _day(1, [_attraction("Камень Черского", "08:00", 60), _attraction("Байкальский музей ИНЦ СО РАН", "09:00", 180)]),
        _day(2, neighbor_attractions),
    ])
    planner.sanitize_rest_items(valid)
    planner._sort_itinerary_items(valid)

    first = valid.model_copy(deep=True)
    first.days[0].items = [item for item in first.days[0].items if item.kind != "meal"]
    second = valid.model_copy(deep=True)
    duplicate = valid.days[1].items[0].model_copy(deep=True)
    if neighbor_has_second:
        second.days[0].items[0] = duplicate
    else:
        second.days[0].items = [duplicate] + [item for item in second.days[0].items if item.kind != "attraction"]
    return request, first, second, valid


def _patch_json(route):
    return json.dumps({
        f"day_{day.day}": day.model_dump(mode="json") for day in route.days
    }, ensure_ascii=False)


def _retry_payload(messages):
    for message in reversed(messages):
        content = message["content"]
        if planner.MAP_RETRY_DATA_BEGIN in content:
            encoded = content.split(planner.MAP_RETRY_DATA_BEGIN, 1)[1].split(planner.MAP_RETRY_DATA_END, 1)[0]
            payload = json.loads(encoded.strip())
            if "replace_full_days" in payload:
                return payload
    raise AssertionError("Expected a regional scoped retry")


class RegionalScopePreservationTests(unittest.TestCase):
    def _assert_preserved_neighbor_after_new_duplicate(self, *, neighbor_has_second):
        request, first, second, final = _fixture(neighbor_has_second=neighbor_has_second)
        self.assertTrue(any("обед" in problem.casefold() for problem in check_rules(first, request)))
        self.assertEqual(check_rules(final, request), [])
        self.assertEqual(check_program_depth(final, request), [])
        expected_neighbor = final.days[1].model_dump()
        source_snapshot = first.model_dump()
        observed_map_routes = []

        def map_check(itinerary, *args, **kwargs):
            observed_map_routes.append(itinerary.model_dump())
            return []

        responses = [first.model_dump_json(), _patch_json(second), _patch_json(final)]
        with ExitStack() as stack:
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")))
            for name, value in (
                ("resolve_region", _scope()), ("get_weather", None), ("find_city", None),
                ("cached_place_hints", []), ("curated_place_hints", []), ("curated_blocked_place_terms", []),
            ):
                stack.enter_context(patch.object(planner, name, return_value=value))
            generate = stack.enter_context(patch.object(planner, "generate_json", side_effect=[
                SimpleNamespace(text=text, input_tokens=11, output_tokens=13) for text in responses
            ]))
            map_mock = stack.enter_context(patch.object(planner, "_map_verification_problems", side_effect=map_check))
            prune = stack.enter_context(patch.object(planner, "_prune_safe_exact_place_duplicates", wraps=planner._prune_safe_exact_place_duplicates))
            result = planner.generate_itinerary(request, max_attempts=3)

        self.assertEqual((generate.call_count, result.attempts, result.total_tokens), (3, 3, 72))
        self.assertEqual(result.problems, [])
        # Neither the initial missing-meal candidate nor its duplicate patch
        # may reach the map stage. In particular the duplicate cannot be
        # silently removed from day 2 and accepted on attempt 2.
        self.assertEqual(map_mock.call_count, 1)
        self.assertEqual(prune.call_count, 1, "General pruning must not run on either scoped patch")
        self.assertEqual(result.itinerary.days[1].model_dump(), expected_neighbor)
        self.assertEqual(observed_map_routes[0]["days"][1], expected_neighbor)
        self.assertEqual(first.model_dump(), source_snapshot)
        self.assertEqual((result.itinerary.title, result.itinerary.summary, result.itinerary.tips), (first.title, first.summary, first.tips))

        for call_index in (1, 2):
            call = generate.call_args_list[call_index]
            payload = _retry_payload(call.args[0])
            self.assertEqual(payload["replace_full_days"], [1])
            self.assertEqual(payload["transfer_note_only_days"], [2])
            self.assertEqual(payload["current_route"]["days"][1], expected_neighbor)
            schema = call.kwargs["json_schema"]
            self.assertEqual(set(schema["properties"]), {"day_1", "day_2"})
            self.assertEqual(set(schema["required"]), {"day_1", "day_2"})
            for day_number in (1, 2):
                fields = schema["properties"][f"day_{day_number}"]["properties"]
                self.assertEqual(fields["day"]["enum"], [day_number])
                self.assertEqual(fields["date"]["enum"], [first.days[day_number - 1].date])
        last_payload = _retry_payload(generate.call_args_list[2].args[0])
        self.assertTrue(any("повторяется место" in problem for problem in last_payload["problems_by_day"]["1"]))
        self.assertEqual(check_rules(result.itinerary, request), [])
        self.assertEqual(check_program_depth(result.itinerary, request), [])

    def test_new_duplicate_cannot_prune_attraction_from_multi_place_neighbor(self):
        self._assert_preserved_neighbor_after_new_duplicate(neighbor_has_second=True)

    def test_new_sole_duplicate_keeps_regional_scope_instead_of_legacy_shortcut(self):
        self._assert_preserved_neighbor_after_new_duplicate(neighbor_has_second=False)


if __name__ == "__main__":
    unittest.main()
