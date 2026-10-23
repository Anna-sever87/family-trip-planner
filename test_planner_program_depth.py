"""Offline integration tests: quality feedback participates in real retry flow."""

from __future__ import annotations

import json
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import planner
import test_regional_generation_repair as regional


class PlannerProgramDepthTests(unittest.TestCase):
    def _context(self, stack, responses, *, travel=None):
        for name, value in (
            ("resolve_region", regional._scope()), ("get_weather", None),
            ("find_city", None), ("cached_place_hints", []),
            ("curated_place_hints", []), ("curated_blocked_place_terms", []),
        ):
            stack.enter_context(patch.object(planner, name, return_value=value))

        def verify(_itinerary, *args, **kwargs):
            kwargs["travel_minutes_by_day"].update(travel or {})
            return []

        stack.enter_context(patch.object(planner, "_map_verification_problems", side_effect=verify))
        return stack.enter_context(patch.object(
            planner, "generate_json", side_effect=[
                SimpleNamespace(text=text, input_tokens=11, output_tokens=13)
                for text in responses
            ],
        ))

    def test_short_day_gets_scoped_quality_feedback_and_then_passes(self):
        source = regional._valid_source()
        regional._attraction(source.days[1]).duration_min = 90
        repaired = regional._valid_source()
        with ExitStack() as stack:
            generate = self._context(stack, [
                regional._model_response_json(source), regional._patch_json(repaired, [2, 3]),
            ])
            result = planner.generate_itinerary(regional._request(), max_attempts=3)
        self.assertEqual((result.problems, result.attempts, generate.call_count), ([], 2, 2))
        data = regional._retry_data(generate.call_args_list[1].args[0])
        self.assertEqual(data["replace_full_days"], [2])
        self.assertEqual(data["transfer_note_only_days"], [3])
        schema = generate.call_args_list[1].kwargs["json_schema"]
        self.assertEqual(set(schema["properties"]), {"day_2", "day_3"})
        self.assertEqual(set(schema["required"]), {"day_2", "day_3"})
        for day_number in (2, 3):
            fields = schema["properties"][f"day_{day_number}"]["properties"]
            self.assertEqual(fields["day"]["enum"], [day_number])
            self.assertEqual(fields["date"]["enum"], [source.days[day_number - 1].date])
        self.assertIn("недостаточно содержательная программа", data["problems_by_day"]["2"][0])
        self.assertEqual(data["rejected_place_queries"], [], "Depth is not a failed map identity")
        self.assertEqual(result.itinerary.days[0], source.days[0])
        self.assertEqual(regional._attraction(source.days[1]).duration_min, 90)

    def test_repeated_short_day_stops_after_three_calls_and_is_not_success(self):
        source = regional._valid_source()
        regional._attraction(source.days[1]).duration_min = 90
        with ExitStack() as stack:
            generate = self._context(stack, [
                regional._model_response_json(source),
                regional._patch_json(source, [2, 3]), regional._patch_json(source, [2, 3]),
            ])
            result = planner.generate_itinerary(regional._request(), max_attempts=3)
        self.assertEqual((generate.call_count, result.attempts), (3, 3))
        self.assertEqual(len(result.problems), 1)
        self.assertIn("День 2: недостаточно содержательная программа", result.problems[0])

    def test_proven_map_travel_contributes_without_inflating_visit(self):
        source = regional._valid_source()
        regional._attraction(source.days[1]).duration_min = 180
        with ExitStack() as stack:
            generate = self._context(stack, [regional._model_response_json(source)], travel={2: 60})
            result = planner.generate_itinerary(regional._request(), max_attempts=3)
        self.assertEqual((generate.call_count, result.problems), (1, []))
        self.assertEqual(regional._attraction(result.itinerary.days[1]).duration_min, 180)

    def test_edit_checks_target_only_and_does_not_rewrite_old_short_days(self):
        current = regional._valid_source()
        for day in current.days:
            regional._attraction(day).duration_min = 90
        changed = regional._replacement(regional._valid_source()).days[1]
        before = current.model_dump()
        with ExitStack() as stack:
            generate = self._context(stack, [changed.model_dump_json()])
            audit = stack.enter_context(patch.object(planner, "_audit_edit_fulfillment", return_value=([], 0, 0)))
            result = planner.edit_itinerary(regional._request(), current, 2, "Замени музей на храм", max_attempts=1)
        self.assertEqual(result.problems, [])
        audit.assert_called_once()
        self.assertIn("240 минут", generate.call_args.args[0][1]["content"])
        self.assertEqual(current.model_dump(), before)
        for index in (0, 2, 3, 4, 5, 6):
            self.assertEqual(result.itinerary.days[index], current.days[index])

    def test_short_edit_rejected_before_semantic_ai_audit(self):
        current = regional._valid_source()
        changed = regional._replacement(current).days[1]
        regional._attraction(changed).duration_min = 90
        with ExitStack() as stack:
            generate = self._context(stack, [changed.model_dump_json()])
            audit = stack.enter_context(patch.object(planner, "_audit_edit_fulfillment"))
            result = planner.edit_itinerary(regional._request(), current, 2, "Замени музей на храм", max_attempts=1)
        self.assertEqual(generate.call_count, 1)
        audit.assert_not_called()
        self.assertTrue(any("недостаточно содержательная программа" in problem for problem in result.problems))

    def test_stale_travel_stats_are_cleared_before_missing_query_return(self):
        source = regional._valid_source()
        regional._attraction(source.days[1]).place_query = ""
        travel = {2: 1000}
        with patch.object(planner, "verify_itinerary") as verify:
            problems = planner._map_verification_problems(source, None, regional._scope(), travel_minutes_by_day=travel)
        verify.assert_not_called()
        self.assertTrue(problems)
        self.assertEqual(travel, {})


if __name__ == "__main__":
    unittest.main()
