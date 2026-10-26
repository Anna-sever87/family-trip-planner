"""Bounded reserve review with unchanged free-text restrictions, entirely offline."""

from contextlib import ExitStack
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from city_preferences import select_city_preferences
from city_visit_options import visit_option
from test_city_deterministic import candidates, request
from yandex_ai import YandexAIError


def response(ids, status="selected", *, pool=None, excluded_reason="not_standalone_visit"):
    value = {"status": status, "candidate_ids": ids}
    if pool is not None:
        value["candidate_decisions"] = {
            item.candidate_id: "include" if item.candidate_id in ids else excluded_reason for item in pool
        }
    return SimpleNamespace(text=json.dumps(value, ensure_ascii=False))


def family_pool():
    # Eighteen individually bound synthetic observations: nine malls, six parks,
    # one zoo, one broad attraction and one museum. No personal/live fixtures.
    pool = candidates(18)
    kinds = [("tourism", "zoo"), ("leisure", "park"), ("shop", "mall"),
             ("tourism", "attraction"), ("tourism", "museum"), ("leisure", "park"),
             ("shop", "mall"), ("leisure", "park"), ("shop", "mall"),
             ("leisure", "park"), ("shop", "mall"), ("leisure", "park"),
             ("shop", "mall"), ("leisure", "park"), ("shop", "mall"),
             ("shop", "mall"), ("shop", "mall"), ("shop", "mall")]
    for item, (category, kind) in zip(pool, kinds):
        item.record.update(category=category, type=kind)
    value = request(5, end_time="18:00", children_ages=[5, 10],
                    trip_goal="Познакомиться с направлением",
                    wishes="Зоопарк и торговые центры, посильная программа для детей 5 и 10 лет.",
                    interest_categories=["main_sights", "parks", "zoos", "shopping_malls"])
    first = [pool[index].candidate_id for index in (0, 1, 2, 10, 12, 14, 16)]
    return value, pool, first


class CityPreferenceDepthWishesTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.forbidden = [self.stack.enter_context(patch(name, side_effect=AssertionError("External I/O forbidden")))
                          for name in ("httpx.get", "httpx.post", "httpx.Client.send", "sqlite3.connect",
                                       "places.geocode", "places._load_cache", "places._save_cache")]

    def tearDown(self):
        for call in self.forbidden:
            call.assert_not_called()

    def test_family_wishes_receive_one_full_reserve_review_without_automatic_restore(self):
        value, pool, first = family_pool()
        self.assertTrue(all(visit_option(item, value) is not None for item in pool))
        self.assertEqual(sum(visit_option(item, value).duration_min for item in pool), 1605)
        # The model still rejects a broad attraction and one commercial branch.
        reviewed = [item.candidate_id for index, item in enumerate(pool) if index not in (3, 17)]
        before = deepcopy((value, pool))
        completion = Mock(side_effect=[response(first), response(reviewed, pool=pool)])
        result = select_city_preferences(value, pool, memory_context="Справочная память",
                                         completion=completion, allow_depth_repair=True)
        self.assertEqual((result.status, result.candidate_priority), ("success", reviewed))
        self.assertEqual(result.diagnostics["selection_calls"], 2)
        self.assertEqual(result.diagnostics["selection_repair_status"], "improved")
        self.assertFalse(result.diagnostics["selection_depth_shortfall"])
        self.assertEqual((value, pool), before)
        self.assertEqual(completion.call_count, 2)
        initial, review = [json.loads(call.args[0][1]["content"]) for call in completion.call_args_list]
        self.assertEqual(initial["request"], value.model_dump(mode="json"))
        self.assertEqual(review["request"], initial["request"])
        self.assertEqual(review["verified_candidates"], initial["verified_candidates"])
        self.assertEqual(review["prior_memory_data"], initial["prior_memory_data"])
        self.assertEqual(review["depth_review"]["selected_candidate_ids"], first)
        self.assertEqual(review["depth_review"]["selected_planning_minutes"], 660)
        self.assertEqual(review["depth_review"]["required_planning_minutes"], 825)
        self.assertIn("не отменяет и не ослабляет исходные wishes", completion.call_args.args[0][0]["content"])

    def test_strict_malls_only_no_museums_can_keep_short_subset_without_filler(self):
        value, pool, _ = family_pool()
        value.wishes = "Только торговые центры, не включать музеи и зоопарки."
        selected = [pool[index].candidate_id for index in (2, 6, 8)]
        completion = Mock(side_effect=[response(selected),
            response(selected, pool=pool, excluded_reason="current_request_restriction")])
        result = select_city_preferences(value, pool, completion=completion, allow_depth_repair=True)
        self.assertEqual(result.candidate_priority, selected)
        self.assertEqual(result.diagnostics["selection_repair_status"], "unchanged")
        self.assertTrue(result.diagnostics["selection_depth_shortfall"])
        self.assertTrue(all(item.record["category"] == "shop" for item in result.candidates))
        self.assertEqual(completion.call_count, 2)
        for call in completion.call_args_list:
            self.assertEqual(json.loads(call.args[0][1]["content"])["request"]["wishes"], value.wishes)

    def test_failed_refused_or_invalid_review_keeps_initial_subset_and_no_third_call(self):
        value, pool, first = family_pool()
        # Refusal without the required per-ID decisions is incomplete. A valid
        # complete refusal is handled separately in complete-review tests.
        bad = [YandexAIError("Synthetic unavailable provider"), response([], "unsupported_preferences"),
               response([], "insufficient_data"), response(["osm-node-999999999999"]),
               response(first + [first[0]])]
        for reply in bad:
            with self.subTest(reply_type=type(reply).__name__):
                completion = Mock(side_effect=[response(first), reply])
                result = select_city_preferences(value, pool, completion=completion, allow_depth_repair=True)
                self.assertEqual((result.status, result.candidate_priority), ("success", first))
                self.assertEqual(result.diagnostics["selection_repair_status"], "failed")
                self.assertTrue(result.warnings)
                self.assertTrue(result.diagnostics["selection_depth_shortfall"])
                self.assertEqual(completion.call_count, 2)

    def test_review_rechecks_explicit_shopping_ban_instead_of_accepting_more_minutes(self):
        value, pool, _ = family_pool()
        value.wishes = "Без торговых центров. Музеи и парки."
        selected = [pool[1].candidate_id, pool[4].candidate_id]
        completion = Mock(side_effect=[response(selected),
            response([item.candidate_id for item in pool], pool=pool)])
        result = select_city_preferences(value, pool, completion=completion, allow_depth_repair=True)
        self.assertEqual(result.candidate_priority, selected)
        self.assertEqual(result.diagnostics["selection_repair_status"], "failed")
        self.assertFalse(any(item.record["category"] == "shop" for item in result.candidates))
        self.assertEqual(completion.call_count, 2)

    def test_no_remaining_review_budget_or_full_reserve_never_adds_a_call(self):
        value, pool, first = family_pool()
        for identifiers, allowed in ((first, False), ([item.candidate_id for item in pool], True)):
            with self.subTest(allow_review=allowed):
                completion = Mock(side_effect=[response(identifiers)])
                result = select_city_preferences(value, pool, completion=completion, allow_depth_repair=allowed)
                self.assertEqual(result.candidate_priority, identifiers)
                self.assertFalse(result.diagnostics["selection_repair_attempted"])
                completion.assert_called_once()

    def test_review_itself_remains_bounded_even_when_reserve_stays_too_small(self):
        value, pool, first = family_pool()
        reviewed = first + [pool[4].candidate_id]
        completion = Mock(side_effect=[response(first), response(reviewed, pool=pool)])
        result = select_city_preferences(value, pool, completion=completion, allow_depth_repair=True)
        self.assertEqual(result.candidate_priority, reviewed)
        self.assertTrue(result.diagnostics["selection_depth_shortfall"])
        self.assertEqual(result.diagnostics["selection_calls"], 2)
        self.assertEqual(completion.call_count, 2)


if __name__ == "__main__":
    unittest.main()
