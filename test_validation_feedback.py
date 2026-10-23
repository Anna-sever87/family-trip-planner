"""Regression cases from control 3: collect all feedback before spending retries."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import planner
import test_regional_generation_repair as regional
import test_planner_program_depth as depth


class ValidationFeedbackTests(unittest.TestCase):
    def _generate(self, source, responses, map_results):
        with ExitStack() as stack:
            generate = depth.PlannerProgramDepthTests()._context(stack, [
                regional._model_response_json(source), *responses,
            ])
            maps = stack.enter_context(patch.object(planner, "_map_verification_problems", side_effect=map_results))
            result = planner.generate_itinerary(regional._request(), max_attempts=3)
        return result, generate, maps

    def test_empty_query_and_short_other_day_are_reported_together_before_map(self):
        source = regional._valid_source()
        regional._attraction(source.days[0]).duration_min = 90
        regional._attraction(source.days[1]).place_query = ""
        fixed = regional._valid_source()
        result, generate, maps = self._generate(source, [regional._patch_json(fixed, [1, 2, 3])], [[]])
        self.assertEqual((result.problems, generate.call_count, maps.call_count), ([], 2, 1))
        feedback = regional._retry_data(generate.call_args_list[1].args[0])
        self.assertEqual(feedback["replace_full_days"], [1, 2])
        self.assertTrue(any("недостаточно содержательная программа" in text for text in feedback["problems_by_day"]["1"]))
        self.assertTrue(any("Дорога этого дня ещё не подтверждена" in text for text in feedback["problems_by_day"]["1"]))
        self.assertTrue(any("не заполнено точное" in text for text in feedback["problems_by_day"]["2"]))

    def test_map_error_does_not_hide_short_program_in_another_day(self):
        source = regional._valid_source()
        regional._attraction(source.days[0]).duration_min = 90
        fixed = regional._replacement(regional._valid_source())
        result, generate, maps = self._generate(source, [regional._patch_json(fixed, [1, 2, 3])], [[regional._map_failure(source)], []])
        self.assertEqual((result.problems, generate.call_count, maps.call_count), ([], 2, 2))
        feedback = regional._retry_data(generate.call_args_list[1].args[0])
        self.assertEqual(feedback["replace_full_days"], [1, 2])
        self.assertTrue(any("недостаточно содержательная программа" in text for text in feedback["problems_by_day"]["1"]))
        self.assertTrue(any("не найдено на карте" in text for text in feedback["problems_by_day"]["2"]))

    def test_category_and_depth_are_fixed_in_the_same_retry(self):
        source = regional._valid_source()
        item = regional._attraction(source.days[1])
        item.title, item.place_query, item.duration_min = "Природные тропы", "Природные тропы, Листвянка, Россия", 90
        result, generate, maps = self._generate(source, [regional._patch_json(regional._valid_source(), [2, 3])], [[]])
        self.assertEqual((result.problems, generate.call_count, maps.call_count), ([], 2, 1))
        feedback = regional._retry_data(generate.call_args_list[1].args[0])
        self.assertTrue(any("слишком общее название" in text for text in feedback["problems_by_day"]["2"]))
        self.assertTrue(any("недостаточно содержательная программа" in text for text in feedback["problems_by_day"]["2"]))

    def test_empty_queries_are_not_ranked_as_map_checked_over_better_later_attempt(self):
        source = regional._valid_source()
        for day in source.days[1:]:
            regional._attraction(day).place_query = ""
        second = regional._valid_source()
        earlier = regional._attraction(second.days[1])
        later = regional._attraction(second.days[6])
        later.title, later.place_query = earlier.title, earlier.place_query
        result, generate, maps = self._generate(source, [
            regional._patch_json(second, [2, 3, 4, 5, 6, 7]), '{"day_2": {}}',
        ], [])
        self.assertEqual(generate.call_count, 3)
        maps.assert_not_called()
        self.assertEqual(len(result.problems), 1)
        self.assertIn("повторяется место", result.problems[0])
        self.assertTrue(all(regional._attraction(day).place_query for day in result.itinerary.days))


if __name__ == "__main__":
    unittest.main()
