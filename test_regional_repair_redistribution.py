"""Repair may move visits between failed days, never out of untouched days."""

import unittest

import planner
from grounded_schedule import GROUNDED_SYSTEM_PROMPT
from test_regional_generation_repair import (
    _attraction, _patch_json, _retry_data, _valid_source,
)


class RegionalRepairRedistributionTests(unittest.TestCase):
    def _messages(self, source, targets, transfers=()):
        return planner._regional_repair_messages(
            [{"role": "system", "content": "Test rules"},
             {"role": "user", "content": "Test request"}],
            source, list(targets), list(transfers),
            [f"День {day}: нет ни одной достопримечательности" for day in targets],
            {}, {"confirmed": {
                "day": 1, "title": _attraction(source.days[0]).title,
                "place_query": _attraction(source.days[0]).place_query,
            }},
        )

    def test_pool_identity_allows_redistribution_but_day_specific_history_stays_locked(self):
        source = _valid_source()
        before = source.model_dump()
        messages = self._messages(source, range(1, 8))
        data = _retry_data(messages)
        self.assertEqual(data["fixed_place_queries"], [])
        self.assertEqual(data["coverage_requirements"], [
            {"day": day.day, "date": day.date, "minimum_attractions": 1}
            for day in source.days
        ])
        self.assertEqual(len(data["verified_places"]), 1)
        self.assertEqual(data["verified_places"][0]["day"], 1)
        self.assertIn("разрешено перераспределять", messages[0]["content"])
        self.assertIn("КРОМЕ мест из verified_places", messages[0]["content"])
        self.assertIn("нельзя переносить даже между исправляемыми днями", messages[0]["content"])
        self.assertIn("кандидата в общем пуле само по себе не закрепляет", messages[0]["content"])
        self.assertEqual(source.model_dump(), before)

    def test_runtime_still_rejects_moving_a_day_specific_verified_visit(self):
        source = _valid_source()
        moved = _attraction(source.days[0])
        history = {"1:1": {"day": 1, "slot": 1, "title": moved.title, "place_query": moved.place_query}}
        self.assertEqual(planner._enforce_verified_place_locks(source.model_copy(deep=True), history), [])
        changed = source.model_copy(deep=True)
        first, sixth = _attraction(changed.days[0]), _attraction(changed.days[5])
        changed.days[0].items[changed.days[0].items.index(first)] = sixth
        changed.days[5].items[changed.days[5].items.index(sixth)] = first
        self.assertEqual(planner._enforce_verified_place_locks(changed.model_copy(deep=True), {}), [])
        problems = planner._enforce_verified_place_locks(changed, history)
        self.assertEqual(len(problems), 1)
        self.assertIn("День 1", problems[0])
        self.assertIn("Оно найдено в другом дне", problems[0])
        self.assertEqual(_attraction(source.days[0]).place_query, moved.place_query)

    def test_partial_repair_reserves_untouched_and_transfer_only_visits(self):
        source = _valid_source()
        data = _retry_data(self._messages(source, [2, 4], [3, 5]))
        self.assertEqual(data["fixed_place_queries"], [
            _attraction(source.days[number - 1]).place_query
            for number in (1, 3, 5, 6, 7)
        ])
        self.assertEqual([item["day"] for item in data["coverage_requirements"]], [2, 4])
        self.assertEqual(set(data["response_keys"]), {"day_2", "day_3", "day_4", "day_5"})

    def test_merger_can_move_visit_to_empty_failed_day_without_mutating_neighbors(self):
        source = _valid_source()
        moved = _attraction(source.days[5]).model_copy(deep=True)
        source.days[0].items.append(moved)
        source.days[5].items = [item for item in source.days[5].items if item.kind != "attraction"]
        before = source.model_dump()
        repaired = source.model_copy(deep=True)
        repaired.days[0].items = [item for item in repaired.days[0].items
                                 if item.place_query != moved.place_query]
        repaired.days[5].items.insert(0, moved)
        merged = planner._merge_regional_day_repair(source, _patch_json(repaired, [1, 6]), [1, 6], [])
        self.assertEqual(sum(item.kind == "attraction" for item in merged.days[0].items), 1)
        self.assertEqual(_attraction(merged.days[5]).place_query, moved.place_query)
        for number in (2, 3, 4, 5, 7):
            self.assertEqual(merged.days[number - 1].model_dump(), before["days"][number - 1])
        self.assertEqual(source.model_dump(), before)

    def test_initial_prompt_prioritizes_coverage_before_extra_visits(self):
        self.assertIn("для каждой даты из request_rules.dates", GROUNDED_SYSTEM_PROMPT)
        self.assertIn("только после этого добавляй вторые и третьи", GROUNDED_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
