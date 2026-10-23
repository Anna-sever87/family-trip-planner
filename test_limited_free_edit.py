"""A schema-compliant offline model can edit only server-authorized free days."""

from contextlib import ExitStack
from copy import deepcopy
import json
import socket
import sqlite3
import unittest
from unittest.mock import patch

import planner
import prompts
import source_planning
from grounded_response import _validate
from regional_skeleton import build_regional_skeleton
from test_regional_generation_repair import _scope
from test_regional_skeleton import fixture, request_for
from yandex_ai import GenerationResult


class LimitedFreeEditTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "getaddrinfo"), (sqlite3, "connect")):
            blocker = patch.object(owner, name, side_effect=AssertionError("external I/O forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)
        self.request = request_for(3)
        self.grounding, self.facts, self.graph = fixture(self.request, 3, 240)
        partial = build_regional_skeleton(
            self.request, self.grounding, self.facts[:2], travel_facts=self.graph, allow_free_days=True,
        )
        full = build_regional_skeleton(self.request, self.grounding, self.facts, travel_facts=self.graph)
        self.source = source_planning.itinerary_from_skeleton(self.request, self.grounding, partial)
        self.full = source_planning.itinerary_from_skeleton(self.request, self.grounding, full)

    def wire(self, day):
        schema = planner.build_itinerary_day_json_schema(self.request, day.day)
        values = day.model_dump()
        result = {key: values[key] for key in schema["properties"]}
        item_keys = schema["properties"]["items"]["items"]["properties"]
        result["items"] = [{key: item[key] for key in item_keys} for item in values["items"]]
        for item in result["items"]:
            item["area"] = "Город А"
            item["description"] = "Плановый пункт дня"
        return result

    def run_edit(self, payload, target=3, *, source=None, enforce_schema=True):
        source = source or self.source
        before = source.model_dump()
        observed = {}

        def completion(messages, **kwargs):
            observed.update(schema=kwargs["json_schema"], messages=messages)
            if enforce_schema:
                _validate(payload, kwargs["json_schema"])
            return GenerationResult(json.dumps(payload, ensure_ascii=False), 1, 1)

        def map_reply(itinerary, *args, **kwargs):
            for day in itinerary.days:
                for item in day.items:
                    if item.kind == "attraction":
                        hint = next(h for h in self.grounding.confirmed_hints if h["place_query"] == item.place_query)
                        item.verified, item.lat, item.lon = True, hint["lat"], hint["lon"]
            return []

        with ExitStack() as stack:
            stack.enter_context(patch.object(planner, "resolve_region", return_value=_scope()))
            stack.enter_context(patch.object(planner, "curated_blocked_place_terms", return_value=[]))
            generate = stack.enter_context(patch.object(planner, "generate_json", side_effect=completion))
            audit = stack.enter_context(patch.object(planner, "_audit_edit_fulfillment", return_value=([], 0, 0)))
            maps = stack.enter_context(patch.object(planner, "_map_verification_problems", side_effect=map_reply))
            stack.enter_context(patch.object(source_planning, "visit_facts_for_grounding", return_value=self.facts))
            stack.enter_context(patch.object(source_planning, "build_travel_graph", return_value=(self.graph, [])))
            stack.enter_context(patch.object(source_planning, "verify_route_edges", return_value=[]))
            stack.enter_context(patch("places.geocode", return_value={"lat": 50, "lon": 100}))
            result = planner.edit_itinerary(self.request, source, target, "Измени только выбранный день", max_attempts=1)
        self.assertEqual(source.model_dump(), before)
        self.assertEqual(generate.call_count, 1)
        return result, observed, maps, audit

    def test_schema_compliant_free_edit_retains_only_meal_rest_and_conditional_depth(self):
        payload = self.wire(self.source.days[2])
        payload["theme"] = "Спокойный свободный день"
        payload["items"][0]["time"] = "12:15"
        payload["items"][1]["time"] = "13:15"
        result, observed, maps, audit = self.run_edit(payload)
        self.assertEqual(result.problems, [])
        self.assertEqual(result.itinerary.days[2].day_mode, "free")
        self.assertEqual(len(result.itinerary.days[2].items), 2)
        self.assertEqual(observed["schema"]["properties"]["items"]["minItems"], 2)
        self.assertNotIn("day_mode", observed["schema"]["properties"])
        self.assertNotIn("program_report", observed["schema"]["properties"])
        self.assertIn("Серверное исключение", observed["messages"][0]["content"])
        self.assertNotIn("Для изменяемого дня ориентир содержательности", observed["messages"][1]["content"])
        self.assertIn("Если добавляешь посещения", observed["messages"][1]["content"])
        self.assertIn("240 минут", observed["messages"][1]["content"])
        maps.assert_called_once()
        audit.assert_called_once()

    def test_ordinary_program_schema_and_prompt_keep_strict_minimum(self):
        payload = self.wire(self.source.days[0])
        payload["theme"] = "Обновлённая программа"
        result, observed, _, _ = self.run_edit(payload, target=1)
        self.assertEqual(result.problems, [])
        self.assertEqual(observed["schema"]["properties"]["items"]["minItems"], 3)
        self.assertNotIn("Серверное исключение", observed["messages"][0]["content"])
        self.assertIn("Для изменяемого дня ориентир содержательности — 240 минут", observed["messages"][1]["content"])

    def test_free_to_program_uses_full_source_checks_and_removes_report(self):
        payload = self.wire(self.full.days[2])
        result, observed, _, audit = self.run_edit(payload)
        self.assertEqual(result.problems, [])
        self.assertEqual(result.itinerary.days[2].day_mode, "program")
        self.assertIsNone(result.itinerary.program_report)
        self.assertTrue(any(tip.startswith("Если в маршруте есть свободные дни,")
                            for tip in result.itinerary.tips))
        self.assertFalse(any(tip.startswith("В свободные дни") for tip in result.itinerary.tips))
        self.assertIn(self.source.tips[0], result.itinerary.tips)
        self.assertEqual(observed["schema"]["properties"]["items"]["minItems"], 2)
        audit.assert_called_once()

    def test_model_fields_cannot_authorize_program_to_free(self):
        payload = self.wire(self.source.days[0])
        payload.update(day_mode="free", program_report=self.source.program_report.model_dump(), theme="Подмена режима")
        payload["items"] = [item for item in payload["items"] if item["kind"] != "attraction"]
        result, observed, maps, audit = self.run_edit(payload, target=1, enforce_schema=False)
        self.assertEqual(result.itinerary.days[0].day_mode, "program")
        self.assertTrue(any("достопримечательности" in issue for issue in result.problems))
        self.assertEqual(observed["schema"]["properties"]["items"]["minItems"], 3)
        maps.assert_not_called()
        audit.assert_not_called()

    def test_source_flag_or_free_label_alone_cannot_relax_the_wire_schema(self):
        for update in ({"source_schedule": False}, {"program_report": None}):
            with self.subTest(update=update):
                source = self.source.model_copy(deep=True, update=update)
                payload = self.wire(source.days[2])
                payload["theme"] = "Изменённый свободный день"
                result, observed, _, _ = self.run_edit(payload, source=source, enforce_schema=False)
                self.assertTrue(result.problems)
                self.assertEqual(observed["schema"]["properties"]["items"]["minItems"], 3)
                self.assertNotIn("Серверное исключение", observed["messages"][0]["content"])

    def test_schema_compliant_free_edit_still_enforces_meal_price(self):
        payload = self.wire(self.source.days[2])
        payload["items"][0]["cost"] = 0
        result, _, maps, audit = self.run_edit(payload)
        self.assertTrue(any("стоимость обеда" in issue for issue in result.problems))
        maps.assert_not_called()
        audit.assert_not_called()

    def test_free_to_program_cannot_lower_fact_duration_to_pass(self):
        payload = self.wire(self.full.days[2])
        visit = next(item for item in payload["items"] if item["kind"] == "attraction")
        visit["duration_min"] = 120
        result, _, _, audit = self.run_edit(payload)
        self.assertTrue(any("длительностью" in issue for issue in result.problems))
        audit.assert_not_called()

    def test_invalid_free_base_returns_rejection_not_report_exception(self):
        payload = self.wire(self.source.days[2])
        payload["base_area"] = "Город Б, Россия"
        result, _, maps, audit = self.run_edit(payload)
        self.assertTrue(any("базу" in issue for issue in result.problems))
        maps.assert_not_called()
        audit.assert_not_called()

    def test_stored_source_description_of_185_chars_survives_wire_without_truncation(self):
        description = "Резерв на обед: " + "я" * (185 - len("Резерв на обед: "))
        self.source.days[2].items[0].description = description
        payload = self.wire(self.source.days[2])
        payload["theme"] = "Уточнённый свободный день"
        payload["items"][0]["description"] = description
        before = deepcopy(payload)
        result, observed, _, audit = self.run_edit(payload)
        self.assertEqual(result.problems, [])
        self.assertEqual(payload, before)
        self.assertEqual(result.itinerary.days[2].items[0].description, description)
        self.assertEqual(len(result.itinerary.days[2].items[0].description), 185)
        self.assertGreaterEqual(observed["schema"]["properties"]["items"]["items"]["properties"]
                                ["description"]["maxLength"], 185)
        for number in (1, 2):
            self.assertEqual(prompts._day_for_edit_audit(result.itinerary, number),
                             prompts._day_for_edit_audit(self.source, number))
        audit.assert_called_once()

    def test_source_description_wire_limit_is_capped_at_1000(self):
        self.source.days[2].items[0].description = "я" * 5000
        payload = self.wire(self.source.days[2])
        payload["theme"] = "Уточнённый свободный день"
        result, observed, _, _ = self.run_edit(payload)
        self.assertEqual(result.problems, [])
        self.assertEqual(observed["schema"]["properties"]["items"]["items"]["properties"]
                         ["description"]["maxLength"], 1000)
        payload["items"][0]["description"] = "я" * 1000
        _validate(payload, observed["schema"])
        payload["items"][0]["description"] += "я"
        with self.assertRaises(ValueError):
            _validate(payload, observed["schema"])

    def test_non_source_description_wire_limit_stays_at_180(self):
        ordinary = self.full.model_copy(deep=True, update={"source_schedule": False})
        ordinary.days[0].items[0].description = "я" * 185
        payload = self.wire(ordinary.days[0])
        payload["theme"] = "Уточнённая программа"
        result, observed, _, _ = self.run_edit(payload, target=1, source=ordinary)
        self.assertEqual(result.problems, [])
        self.assertEqual(observed["schema"]["properties"]["items"]["items"]["properties"]
                         ["description"]["maxLength"], 180)
        self.assertEqual(observed["schema"]["properties"]["items"]["minItems"], 3)
        payload["items"][0]["description"] = "я" * 185
        with self.assertRaises(ValueError):
            _validate(payload, observed["schema"])

    def test_source_description_extension_does_not_change_other_wire_guards(self):
        self.full.days[0].items[0].description = "я" * 185
        payload = self.wire(self.full.days[0])
        payload["theme"] = "Уточнённая программа"
        result, observed, _, _ = self.run_edit(payload, target=1, source=self.full)
        self.assertEqual(result.problems, [])
        baseline = planner.build_itinerary_day_json_schema(self.request, 1)
        actual = deepcopy(observed["schema"])
        actual["properties"]["items"]["items"]["properties"]["description"]["maxLength"] = 180
        self.assertEqual(actual, baseline)


if __name__ == "__main__":
    unittest.main()
