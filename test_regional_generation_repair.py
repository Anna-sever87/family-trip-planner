"""Offline regressions for atomic, bounded regional day repairs.

The first fixture reproduces the error pattern from the Baikal run, not its
unavailable full model response. All AI, map, catalogue and weather IO is mocked.
"""

from __future__ import annotations

import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import planner
from rules import check_rules
from test_baikal_duplicate_repair import (
    TARGETED_REPLACEMENTS,
    _attraction,
    _candidate_with_replacements,
    _itinerary,
    _model_response_json,
    _request,
    _scope,
)


def _valid_source():
    source = _candidate_with_replacements(
        _itinerary(), {7: TARGETED_REPLACEMENTS[7]},
    )
    return _long_visit_baseline(source)


def _long_visit_baseline(source):
    # Repair tests use a synthetic long-visit schedule meeting the separate
    # depth validator; they do not prescribe visit lengths for real POIs.
    for day in source.days:
        attraction = _attraction(day)
        attraction.time = "09:00"
        attraction.duration_min = 240
    planner.sanitize_rest_items(source)
    return source


def _map_failure(source, day=2):
    item = _attraction(source.days[day - 1])
    return f"День {day}: место «{item.title}» не найдено на карте ({item.place_query})"


def _replacement(source, day=2):
    return _candidate_with_replacements(source, {day: TARGETED_REPLACEMENTS[1]})


def _patch_json(source, selected):
    return json.dumps({
        f"day_{day.day}": day.model_dump(mode="json")
        for day in source.days if day.day in selected
    }, ensure_ascii=False)


def _retry_data(messages):
    for message in reversed(messages):
        content = message["content"]
        if planner.MAP_RETRY_DATA_BEGIN in content:
            encoded = content.split(planner.MAP_RETRY_DATA_BEGIN, 1)[1]
            encoded = encoded.split(planner.MAP_RETRY_DATA_END, 1)[0]
            data = json.loads(encoded.strip())
            if "replace_full_days" in data:
                return data
    raise AssertionError("Regional retry data was not sent")


class RegionalRepairScopeTests(unittest.TestCase):
    def test_observed_baikal_pattern_keeps_good_days_and_adds_transfer_dependency(self):
        source = _valid_source()
        problems = [
            _map_failure(source, 4),
            "День 5: результат карты не соответствует месту «Тальцы» (Тальцы, Иркутск, Россия)",
            "День 6: результат карты не соответствует месту «Музей истории» (Музей истории, Иркутск, Россия)",
            "День 7: результат карты не соответствует месту «Набережная Ангары» (Набережная Ангары, Иркутск, Россия)",
            "День 2: при смене базы «Листвянка» на «Иркутск» нужно описать переезд (transfer_note).",
            "День 2: база «Иркутск» находится за пределами выбранной территории и её ближайших окрестностей.",
            "День 2: «Музей» находится за пределами выбранной территории и её ближайших окрестностей. place_query «Музей, Иркутск, Россия»",
            "День 4: при смене базы «Ольхон» на «Иркутск» нужно описать переезд (transfer_note).",
        ] + [
            f"День {day}: база «Иркутск» находится за пределами выбранной территории и её ближайших окрестностей."
            for day in (4, 5, 6, 7)
        ]
        before = source.model_dump()
        self.assertEqual(planner._regional_repair_scope(source, problems), ([2, 4, 5, 6, 7], [3]))
        self.assertEqual(source.model_dump(), before)

    def test_mixed_rejected_query_and_duplicate_keep_first_occurrence(self):
        source = _itinerary()
        problems = check_rules(source, _request()) + [
            "День 3: place_query «Шаман-камень, Листвянка, Россия» уже отклонён картой в предыдущей попытке. Полностью замени это место другим реальным объектом",
        ]
        self.assertEqual(planner._regional_repair_scope(source, problems), ([3, 7], [4]))

    def test_duplicate_groups_and_adjacent_failures_have_disjoint_sorted_scope(self):
        problems = [
            "Дни 4 и 6: повторяется место «Музей» (ранее «Музей»)",
            "Дни 1 и 2: повторяется место «Маяк» (ранее «Маяк»)",
            "День 3: место «Монастырь» не найдено на карте (Монастырь, Листвянка, Россия)",
        ]
        self.assertEqual(planner._regional_repair_scope(_valid_source(), problems), ([2, 3, 6], [4, 7]))

    def test_empty_global_unknown_or_out_of_range_problems_disable_scoped_mode(self):
        source = _valid_source()
        for problems in (
            [],
            [_map_failure(source), "Общий бюджет поездки превышен"],
            [_map_failure(source), "День 3: неизвестное ограничение новой версии"],
            ["День 8: место «Музей» не найдено на карте (Музей)"],
        ):
            with self.subTest(problems=problems):
                self.assertEqual(planner._regional_repair_scope(source, problems), ([], []))


class RegionalRepairMergeTests(unittest.TestCase):
    def test_nonconsecutive_days_schema_requires_exact_independently_pinned_keys(self):
        source = _valid_source()
        schema = planner._regional_day_repair_schema(_request(), [2, 7], [3])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["properties"]), {"day_2", "day_3", "day_7"})
        self.assertEqual(schema["required"], ["day_2", "day_3", "day_7"])
        for number in (2, 3, 7):
            day_schema = schema["properties"][f"day_{number}"]
            self.assertFalse(day_schema["additionalProperties"])
            self.assertEqual(day_schema["properties"]["day"]["enum"], [number])
            self.assertEqual(day_schema["properties"]["date"]["enum"], [source.days[number - 1].date])
        schema["properties"]["day_2"]["properties"]["day"]["enum"].append(99)
        self.assertEqual(schema["properties"]["day_3"]["properties"]["day"]["enum"], [3])

    def test_full_day_changes_but_only_neighbor_transfer_and_no_trip_metadata(self):
        source = _valid_source()
        before = source.model_dump()
        candidate = _replacement(source)
        candidate.days[1].theme = "Исправленная программа"
        candidate.days[1].base_area = "Большие Коты"
        candidate.days[1].transfer_note = "Переезд из Листвянки утром"
        candidate.days[1].items[1].cost = 1700
        candidate.days[2].transfer_note = "Возвращение из Больших Котов в Листвянку"
        candidate.days[2].theme = "Нежелательная новая программа"
        candidate.days[2].base_area = "Иркутск"
        _attraction(candidate.days[2]).place_query = "Чужое место, Иркутск, Россия"
        merged = planner._merge_regional_day_repair(source, _patch_json(candidate, [2, 3]), [2], [3])
        self.assertEqual(merged.days[1].model_dump(), candidate.days[1].model_dump())
        expected_neighbor = source.days[2].model_dump()
        expected_neighbor["transfer_note"] = candidate.days[2].transfer_note
        self.assertEqual(merged.days[2].model_dump(), expected_neighbor)
        for day_number in (1, 4, 5, 6, 7):
            self.assertEqual(merged.days[day_number - 1].model_dump(), before["days"][day_number - 1])
        self.assertEqual((merged.title, merged.summary, merged.tips), (source.title, source.summary, source.tips))
        self.assertEqual(source.model_dump(), before)

    def test_model_verification_is_discarded_only_for_replaced_days(self):
        source = _valid_source()
        for day in source.days:
            for item in day.items:
                item.verified, item.lat, item.lon = True, 51.8, 104.8
                item.official_name = "Подтверждено исходной картой"
        candidate = _replacement(source)
        for day in candidate.days:
            for item in day.items:
                item.verified, item.lat, item.lon = True, 0.0, 0.0
                item.official_name = "Подмена моделью"
        merged = planner._merge_regional_day_repair(source, _patch_json(candidate, [2, 3]), [2], [3])
        for item in merged.days[1].items:
            self.assertEqual((item.verified, item.lat, item.lon, item.official_name), (False, None, None, ""))
        for day_number in (1, 3, 4, 5, 6, 7):
            self.assertEqual(merged.days[day_number - 1].model_dump(), source.days[day_number - 1].model_dump())

    def test_reapplying_same_patch_is_idempotent_and_independent_of_source(self):
        source = _valid_source()
        before = source.model_dump()
        raw = _patch_json(_replacement(source), [2, 3])
        first = planner._merge_regional_day_repair(source, raw, [2], [3])
        second = planner._merge_regional_day_repair(first, raw, [2], [3])
        self.assertEqual(first.model_dump(), second.model_dump())
        second.days[0].items[0].title = "Проверка отсутствия общих mutable объектов"
        second.tips.append("Дополнительный совет")
        self.assertEqual(source.model_dump(), before)
        self.assertNotEqual(first.model_dump(), second.model_dump())

    def test_malformed_or_full_route_envelopes_are_rejected(self):
        source = _valid_source()
        before = source.model_dump()
        for raw in ('{"days":', '[]', '{"days": null}', _model_response_json(source), '{"days": [], "title": "Замена"}'):
            with self.subTest(raw=raw[:30]), self.assertRaises(ValueError):
                planner._merge_regional_day_repair(source, raw, [2], [3])
            self.assertEqual(source.model_dump(), before)

    def test_selected_days_must_appear_exactly_once_with_no_unselected_day(self):
        source = _valid_source()
        before = source.model_dump()
        payload = json.loads(_patch_json(source, [2, 3, 7]))
        invalid = [
            {key: value for key, value in payload.items() if key != "day_3"},
            {**payload, "day_1": source.days[0].model_dump()},
            {**payload, "day_3": payload["day_2"]},
            {"days": list(payload.values())},
            {**payload, "days": []},
            {"day_02": payload["day_2"], "day_3": payload["day_3"], "day_7": payload["day_7"]},
        ]
        for candidate in invalid:
            with self.subTest(keys=list(candidate)), self.assertRaises(ValueError):
                planner._merge_regional_day_repair(source, json.dumps(candidate), [2, 7], [3])
            self.assertEqual(source.model_dump(), before)

    def test_duplicate_json_keys_are_rejected_before_any_merge(self):
        source = _valid_source()
        before = source.model_dump()
        payload = json.loads(_patch_json(_replacement(source), [2, 3, 7]))
        raw = json.dumps(payload, ensure_ascii=False)
        duplicate_day = json.dumps(payload["day_2"], ensure_ascii=False)
        for invalid in (
            raw[:-1] + ', "day_2": ' + duplicate_day + "}",
            raw[:-1] + r', "\u0064ay_2": ' + duplicate_day + "}",
            raw.replace('"day": 2', '"day": 2, "day": 2', 1),
        ):
            with self.subTest(raw=invalid[:80]), self.assertRaisesRegex(ValueError, "JSON-ключи"):
                planner._merge_regional_day_repair(source, invalid, [2, 7], [3])
            self.assertEqual(source.model_dump(), before)

    def test_wrong_date_or_coercible_day_number_is_not_normalized(self):
        source = _valid_source()
        for field, value in (("day", "2"), ("day", True), ("day", 2.0), ("day", 3), ("date", source.days[2].date), ("date", None)):
            payload = json.loads(_patch_json(source, [2, 3, 7]))
            payload["day_2"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                planner._merge_regional_day_repair(source, json.dumps(payload), [2, 7], [3])

    def test_nonconsecutive_keys_can_arrive_in_any_order_without_changing_route_order(self):
        source = _valid_source()
        payload = json.loads(_patch_json(source, [2, 3, 7]))
        reversed_payload = {key: payload[key] for key in reversed(payload)}
        merged = planner._merge_regional_day_repair(source, json.dumps(reversed_payload), [2, 7], [3])
        self.assertEqual(merged.model_dump(), source.model_dump())

    def test_invalid_scope_and_missing_required_item_field_are_rejected(self):
        source = _valid_source()
        for full, transfers in (([], []), ([8], [])):
            with self.subTest(full=full), self.assertRaises(ValueError):
                planner._merge_regional_day_repair(source, '{"days": []}', full, transfers)
        payload = json.loads(_patch_json(source, [2, 3]))
        del payload["day_2"]["items"][0]["title"]
        with self.assertRaises(ValueError):
            planner._merge_regional_day_repair(source, json.dumps(payload), [2], [3])

    def test_invalid_later_day_cannot_partially_apply_earlier_replacement(self):
        source = _valid_source()
        before = source.model_dump()
        payload = json.loads(_patch_json(_replacement(source), [2, 3]))
        del payload["day_3"]["theme"]
        with self.assertRaises(ValueError):
            planner._merge_regional_day_repair(source, json.dumps(payload), [2], [3])
        self.assertEqual(source.model_dump(), before)

    def test_changed_previous_base_rejects_stale_neighbor_transfer_atomically(self):
        for stale_note in ("Возвращение из прежней базы в Листвянку", ""):
            source = _valid_source()
            source.days[2].transfer_note = stale_note
            before = source.model_dump()
            candidate = _replacement(source)
            candidate.days[1].base_area = "Большие Коты"
            with self.subTest(stale_note=stale_note), self.assertRaisesRegex(ValueError, "transfer_note"):
                planner._merge_regional_day_repair(source, _patch_json(candidate, [2, 3]), [2], [3])
            self.assertEqual(source.model_dump(), before)

    def test_updated_neighbor_transfer_or_removed_base_change_preserves_program(self):
        for same_base in (False, True):
            source = _valid_source()
            if same_base:
                source.days[1].base_area = "Иркутск"
            before = source.model_dump()
            candidate = _replacement(source)
            candidate.days[1].base_area = "Листвянка" if same_base else "Большие Коты"
            new_note = "" if same_base else "Переезд из Больших Котов в Листвянку"
            candidate.days[2].transfer_note = new_note
            candidate.days[2].base_area = "Неавторизованная смена базы"
            candidate.days[2].theme = "Неавторизованная смена программы"
            candidate.days[2].items[1].cost = 99999
            with self.subTest(same_base=same_base):
                merged = planner._merge_regional_day_repair(source, _patch_json(candidate, [2, 3]), [2], [3])
                expected = source.days[2].model_dump()
                expected["transfer_note"] = new_note
                self.assertEqual(merged.days[2].model_dump(), expected)
                self.assertEqual(source.model_dump(), before)


class RegionalRepairGenerationTests(unittest.TestCase):
    def _generate(self, responses, map_results, *, blocked=None):
        generations = [
            text if isinstance(text, Exception) else SimpleNamespace(text=text, input_tokens=11, output_tokens=13)
            for text in responses
        ]
        observed = []
        def verify(itinerary, *args, **kwargs):
            observed.append((itinerary, itinerary.model_dump(), args, kwargs))
            return list(map_results[len(observed) - 1])
        with ExitStack() as stack:
            for name, value in (
                ("resolve_region", _scope()), ("get_weather", None), ("find_city", None),
                ("cached_place_hints", []), ("curated_place_hints", []),
                ("curated_blocked_place_terms", blocked or []),
            ):
                stack.enter_context(patch.object(planner, name, return_value=value))
            generate = stack.enter_context(patch.object(planner, "generate_json", side_effect=generations))
            map_check = stack.enter_context(patch.object(planner, "_map_verification_problems", side_effect=verify))
            result = planner.generate_itinerary(_request(), max_attempts=3)
        return result, generate, map_check, observed

    def _assert_scope(self, generate, index, full, transfers):
        call = generate.call_args_list[index]
        data = _retry_data(call.args[0])
        self.assertEqual(data["replace_full_days"], full)
        self.assertEqual(data["transfer_note_only_days"], transfers)
        schema = call.kwargs["json_schema"]
        selected = sorted(set(full + transfers))
        expected_keys = {f"day_{number}" for number in selected}
        self.assertEqual(set(schema["properties"]), expected_keys)
        self.assertEqual(set(schema["required"]), expected_keys)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(data["response_keys"]), expected_keys)
        for number in selected:
            properties = schema["properties"][f"day_{number}"]["properties"]
            self.assertEqual(properties["day"]["enum"], [number])
            self.assertEqual(properties["date"]["enum"], [data["current_route"]["days"][number - 1]["date"]])
            self.assertEqual(data["response_keys"][f"day_{number}"], {
                "day": number, "date": properties["date"]["enum"][0],
            })
        return data

    def test_first_map_failure_repairs_only_selected_days_and_rechecks_whole_route(self):
        first = _valid_source()
        repaired = _replacement(first)
        repaired.days[2].transfer_note = "Уточнён переезд после исправленного дня"
        repaired.days[2].theme = "Не должно измениться"
        result, generate, map_check, observed = self._generate(
            [_model_response_json(first), _patch_json(repaired, [2, 3])], [[_map_failure(first)], []],
        )
        self.assertEqual((generate.call_count, map_check.call_count, result.attempts, result.total_tokens), (2, 2, 2, 48))
        self.assertEqual(result.problems, [])
        self._assert_scope(generate, 1, [2], [3])
        self.assertEqual(len(observed[1][1]["days"]), 7)
        self.assertEqual(observed[1][3]["start_time"], "08:00")
        self.assertEqual(observed[1][3]["end_time"], "21:00")
        self.assertEqual(observed[0][0].model_dump(), observed[0][1], "Repair mutated first-attempt snapshot")
        for day_number in (1, 4, 5, 6, 7):
            self.assertEqual(result.itinerary.days[day_number - 1].model_dump(), first.days[day_number - 1].model_dump())
        expected_neighbor = first.days[2].model_dump()
        expected_neighbor["transfer_note"] = repaired.days[2].transfer_note
        self.assertEqual(result.itinerary.days[2].model_dump(), expected_neighbor)

    def test_blocked_place_and_duplicate_get_one_combined_regional_patch(self):
        first = _long_visit_baseline(_itinerary())
        blocked_item = _attraction(first.days[1])
        blocked = [{"aliases": [blocked_item.title], "place_queries": [blocked_item.place_query], "reason": "Объект закрыт"}]
        repaired = _candidate_with_replacements(first, {2: TARGETED_REPLACEMENTS[1], 7: TARGETED_REPLACEMENTS[7]})
        result, generate, map_check, _ = self._generate(
            [_model_response_json(first), _patch_json(repaired, [2, 3, 7])], [[]], blocked=blocked,
        )
        self.assertEqual(result.problems, [])
        self.assertEqual((generate.call_count, map_check.call_count), (2, 1))
        self._assert_scope(generate, 1, [2, 7], [3])
        self.assertEqual(result.itinerary.days[0].model_dump(), first.days[0].model_dump())

    def test_rejected_query_plus_new_duplicate_stays_scoped_on_third_call(self):
        first = _valid_source()
        second = first.model_copy(deep=True)
        duplicate = _attraction(second.days[0]).model_copy(deep=True)
        duplicate.time, duplicate.duration_min = "16:15", 45
        second.days[1].items.append(duplicate)
        repaired = _replacement(first)
        result, generate, map_check, _ = self._generate(
            [_model_response_json(first), _patch_json(second, [2, 3]), _patch_json(repaired, [2, 3])],
            [[_map_failure(first)], []],
        )
        self.assertEqual((generate.call_count, map_check.call_count, result.attempts), (3, 2, 3))
        self.assertEqual(result.problems, [])
        data = self._assert_scope(generate, 2, [2], [3])
        problems = data["problems_by_day"]["2"]
        self.assertTrue(any("уже отклонён картой" in problem for problem in problems), problems)
        self.assertTrue(any("повторяется место" in problem for problem in problems), problems)
        self.assertIn(_attraction(first.days[1]).place_query, data["rejected_place_queries"])
        self.assertEqual(result.itinerary.days[0].model_dump(), first.days[0].model_dump())

    def test_malformed_second_patch_preserves_scope_and_calendar_on_third_call(self):
        first = _valid_source()
        result, generate, map_check, _ = self._generate(
            [_model_response_json(first), '{"days":', _patch_json(_replacement(first), [2, 3])],
            [[_map_failure(first)], []],
        )
        self.assertEqual((generate.call_count, map_check.call_count, result.attempts, result.total_tokens), (3, 2, 3, 72))
        self.assertEqual(result.problems, [])
        self._assert_scope(generate, 2, [2], [3])
        self.assertEqual([day.date for day in result.itinerary.days], [day.date for day in first.days])
        guidance = generate.call_args_list[2].args[0][-1]["content"]
        self.assertIn("ключами day_N из response_keys", guidance)
        self.assertIn("Поле days, массив, пропуски и повторы ключей запрещены", guidance)
        self.assertNotIn("объект с полем days", guidance)

    def test_repeated_rejected_queries_stop_after_three_calls_without_success(self):
        first = _valid_source()
        result, generate, map_check, _ = self._generate(
            [_model_response_json(first), _patch_json(first, [2, 3]), _patch_json(first, [2, 3])],
            [[_map_failure(first)]],
        )
        self.assertEqual((generate.call_count, map_check.call_count, result.attempts, result.total_tokens), (3, 1, 3, 72))
        self.assertTrue(result.problems)
        self.assertEqual(result.itinerary.model_dump(), first.model_dump())
        self._assert_scope(generate, 2, [2], [3])

    def test_full_budget_validation_after_merge_falls_back_to_full_route_retry(self):
        first = _valid_source()
        expensive = _replacement(first)
        expensive.days[1].items[1].cost = _request().program_budget + 1
        repaired = _replacement(first)
        result, generate, map_check, _ = self._generate(
            [_model_response_json(first), _patch_json(expensive, [2, 3]), _model_response_json(repaired)],
            [[_map_failure(first)], []],
        )
        self.assertEqual((generate.call_count, map_check.call_count), (3, 2))
        self.assertEqual(result.problems, [])
        self.assertIn("title", generate.call_args_list[2].kwargs["json_schema"]["properties"])
        self.assertEqual(check_rules(result.itinerary, _request()), [])

    def test_neighbor_boundary_failure_is_revalidated_and_repaired_next(self):
        first = _valid_source()
        second = _replacement(first)
        final = second.model_copy(deep=True)
        final.days[2].transfer_note = "Уточнён межбазовый переезд с запасом времени"
        boundary = "День 3: при смене базы «Большие Коты» на «Листвянка» нужно описать переезд (transfer_note)."
        result, generate, map_check, _ = self._generate(
            [_model_response_json(first), _patch_json(second, [2, 3]), _patch_json(final, [3, 4])],
            [[_map_failure(first)], [boundary], []],
        )
        self.assertEqual((generate.call_count, map_check.call_count), (3, 3))
        self.assertEqual(result.problems, [])
        self._assert_scope(generate, 2, [3], [4])
        self.assertEqual(result.itinerary.days[1].model_dump(), second.days[1].model_dump())
        self.assertEqual(result.itinerary.days[2].transfer_note, final.days[2].transfer_note)

    def test_ai_error_during_patch_preserves_invalid_snapshot_without_extra_calls(self):
        first = _valid_source()
        failure = _map_failure(first)
        result, generate, map_check, _ = self._generate(
            [_model_response_json(first), planner.YandexAIError("offline timeout")], [[failure]],
        )
        self.assertEqual((generate.call_count, map_check.call_count, result.attempts, result.total_tokens), (2, 1, 2, 24))
        self.assertIn(failure, result.problems)
        self.assertEqual(result.itinerary.model_dump(), first.model_dump())


if __name__ == "__main__":
    unittest.main()
