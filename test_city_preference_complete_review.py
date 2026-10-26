"""One bounded semantic review must decide every observed ID, not repeat a prefix."""

from contextlib import ExitStack
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from city_deterministic import build_deterministic_city_itinerary
from city_preferences import select_city_preferences
from generation_quality import safe_coverage
from test_city_deterministic import CENTER, candidates, request, visits


def response(value):
    return SimpleNamespace(text=json.dumps(value, ensure_ascii=False))


def selection(ids):
    return {"status": "selected", "candidate_ids": list(ids)}


def reviewed(pool, ids):
    value = selection(ids)
    value["candidate_decisions"] = {
        item.candidate_id: "include" if item.candidate_id in ids else "current_request_restriction"
        for item in pool
    }
    return value


class CompletePreferenceReviewTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.blockers = [self.stack.enter_context(patch(name, side_effect=AssertionError("No external IO")))
                         for name in ("httpx.get", "httpx.post", "places.geocode", "places._load_cache",
                                      "places._save_cache", "sqlite3.connect", "socket.socket.connect")]

    def tearDown(self):
        for value in self.blockers:
            value.assert_not_called()

    def fixture(self, days=14, count=32):
        req = request(days, children_ages=[14], budget=500000, end_time="18:00",
                      interest_categories=["museums"],
                      wishes="Музеи и прогулки утром и после обеда; подростку не нужен дневной сон.")
        return req, candidates(count)

    def test_first20_then_complete32_fills14days_without_new_places_or_longer_visits(self):
        req, pool = self.fixture()
        original = deepcopy(pool)
        first, all_ids = [x.candidate_id for x in pool[:20]], [x.candidate_id for x in pool]
        complete = Mock(side_effect=[response(selection(first)), response(reviewed(pool, all_ids))])
        selected = select_city_preferences(req, pool, completion=complete, allow_depth_repair=True)
        self.assertEqual(selected.problems, [])
        self.assertEqual(selected.candidate_priority, all_ids)
        self.assertEqual(selected.diagnostics["selection_repair_status"], "improved")
        self.assertEqual(safe_coverage(selected.diagnostics)["selection_review"], {
            "reviewed_count": 32, "included_count": 32, "exclusion_counts": {}})
        self.assertEqual(complete.call_count, 2)
        first_schema = complete.call_args_list[0].kwargs["json_schema"]
        review_schema = complete.call_args_list[1].kwargs["json_schema"]
        self.assertNotIn("candidate_decisions", first_schema["properties"])
        self.assertEqual(set(review_schema["properties"]["candidate_decisions"]["required"]), set(all_ids))
        self.assertFalse(review_schema["properties"]["candidate_decisions"]["additionalProperties"])
        context = json.loads(complete.call_args_list[1].args[0][1]["content"])
        self.assertEqual(context["request"], req.model_dump(mode="json"))
        self.assertEqual(len(context["verified_candidates"]), 32)
        built = build_deterministic_city_itinerary(req, selected.candidates, resolved_city=CENTER,
                                                  candidate_priority=selected.candidate_priority)
        self.assertEqual(built.problems, [], built.coverage)
        self.assertEqual(len(built.itinerary.days), 14)
        self.assertTrue(all(day.day_mode == "program" for day in built.itinerary.days))
        used = visits(built.itinerary)
        self.assertEqual(len({x.place_query for x in used}), len(used))
        self.assertTrue(all(x.duration_min == 90 for x in used))
        self.assertEqual(pool, original)

    def test_short_pool_review_also_requires_decision_for_tail_and_preserves_explicit_exclusion(self):
        req, pool = self.fixture(days=5, count=18)
        req.wishes += " Не посещать музей городской истории номер 18."
        first = [x.candidate_id for x in pool[:7]]
        allowed = [x.candidate_id for x in pool[:-1]]
        complete = Mock(side_effect=[response(selection(first)), response(reviewed(pool, allowed))])
        result = select_city_preferences(req, pool, completion=complete, allow_depth_repair=True)
        self.assertEqual(result.candidate_priority, allowed)
        self.assertNotIn(pool[-1], result.candidates)
        self.assertEqual(result.diagnostics["selection_repair_status"], "improved")
        self.assertEqual(complete.call_count, 2)
        self.assertEqual(len(complete.call_args.kwargs["json_schema"]["properties"]["candidate_decisions"]["required"]), 18)

    def test_complete_review_can_confirm_same_smaller_reserve_without_restoring_ids(self):
        req, pool = self.fixture(days=5, count=18)
        req.wishes = "Только выбранные музеи; остальные не посещать."
        first = [x.candidate_id for x in pool[:7]]
        complete = Mock(side_effect=[response(selection(first)), response(reviewed(pool, first))])
        result = select_city_preferences(req, pool, completion=complete, allow_depth_repair=True)
        self.assertEqual(result.candidate_priority, first)
        self.assertEqual(result.diagnostics["selection_repair_status"], "unchanged")
        self.assertEqual(complete.call_count, 2)

    def test_incomplete_inconsistent_or_unknown_decisions_do_not_authorize_tail(self):
        req, pool = self.fixture()
        first, all_ids = [x.candidate_id for x in pool[:20]], [x.candidate_id for x in pool]
        for defect in ("missing_manifest", "missing_last_id", "unknown_id", "unknown_reason", "include_mismatch"):
            with self.subTest(defect=defect):
                value = reviewed(pool, all_ids)
                if defect == "missing_manifest":
                    value.pop("candidate_decisions")
                elif defect == "missing_last_id":
                    value["candidate_decisions"].pop(all_ids[-1])
                elif defect == "unknown_id":
                    value["candidate_decisions"]["osm-node-unknown"] = "include"
                elif defect == "unknown_reason":
                    value["candidate_decisions"][all_ids[-1]] = "low_priority"
                else:
                    value["candidate_decisions"][all_ids[-1]] = "current_request_restriction"
                complete = Mock(side_effect=[response(selection(first)), response(value)])
                result = select_city_preferences(req, pool, completion=complete, allow_depth_repair=True)
                self.assertEqual(result.candidate_priority, first)
                self.assertEqual(result.diagnostics["selection_repair_status"], "failed")
                self.assertTrue(result.warnings)
                self.assertEqual(complete.call_count, 2)

    def test_no_review_permission_keeps_single_completion_and_original_schema(self):
        req, pool = self.fixture()
        first = [x.candidate_id for x in pool[:20]]
        complete = Mock(return_value=response(selection(first)))
        result = select_city_preferences(req, pool, completion=complete, allow_depth_repair=False)
        self.assertEqual(result.candidate_priority, first)
        self.assertFalse(result.diagnostics["selection_repair_attempted"])
        self.assertNotIn("candidate_decisions", complete.call_args.kwargs["json_schema"]["properties"])
        complete.assert_called_once()

    def test_complete_audit_removes_an_initial_id_even_with_fewer_or_equal_minutes(self):
        req, pool = self.fixture()
        req.wishes += " Не посещать музей городской истории номер 1."
        first = [x.candidate_id for x in pool[:20]]
        for final_ids in (first[1:], first[1:] + [pool[20].candidate_id]):
            with self.subTest(final_size=len(final_ids)):
                complete = Mock(side_effect=[response(selection(first)), response(reviewed(pool, final_ids))])
                result = select_city_preferences(req, pool, completion=complete, allow_depth_repair=True)
                self.assertEqual(result.candidate_priority, final_ids)
                self.assertNotIn(first[0], result.candidate_priority)
                self.assertEqual(result.diagnostics["selection_repair_status"], "revised")
                clean = safe_coverage(result.diagnostics)
                self.assertEqual(clean["selection_repair_status"], "revised")
                self.assertEqual(clean["selection_review"]["included_count"], len(final_ids))
                self.assertEqual(clean["selection_review"]["exclusion_counts"],
                                 {"current_request_restriction": 32 - len(final_ids)})
                self.assertEqual(complete.call_count, 2)

    def test_complete_semantic_refusal_never_revives_the_first_selection(self):
        req, pool = self.fixture()
        first = [x.candidate_id for x in pool[:20]]
        for status in ("unsupported_preferences", "insufficient_data"):
            with self.subTest(status=status):
                refusal = reviewed(pool, [])
                refusal["status"] = status
                complete = Mock(side_effect=[response(selection(first)), response(refusal)])
                result = select_city_preferences(req, pool, completion=complete, allow_depth_repair=True)
                self.assertEqual(result.status, status)
                self.assertTrue(result.problems)
                self.assertFalse(result.candidates)
                self.assertEqual(result.diagnostics["selection_repair_status"], "revised")
                self.assertEqual(complete.call_count, 2)

    def test_42_id_review_accepts12000_character_boundary_and_rejects_overflow(self):
        req, pool = self.fixture(count=42)
        first, all_ids = [x.candidate_id for x in pool[:20]], [x.candidate_id for x in pool]
        raw = json.dumps(reviewed(pool, all_ids))
        self.assertLess(len(raw), 12000)
        for length, expected in ((12000, "improved"), (12001, "failed")):
            with self.subTest(length=length):
                complete = Mock(side_effect=[response(selection(first)),
                    SimpleNamespace(text=raw + " " * (length - len(raw)))])
                result = select_city_preferences(req, pool, completion=complete, allow_depth_repair=True)
                self.assertEqual(result.diagnostics["selection_repair_status"], expected)
                self.assertEqual(len(result.candidates), 42 if length == 12000 else 20)
                self.assertEqual(complete.call_count, 2)

    def test_duplicate_decision_keys_fail_closed_and_finite_summary_rejects_arbitrary_text(self):
        req, pool = self.fixture()
        first, all_ids = [x.candidate_id for x in pool[:20]], [x.candidate_id for x in pool]
        raw = json.dumps(reviewed(pool, all_ids))
        field = json.dumps(all_ids[0]) + ': "include"'
        raw = raw.replace(field, field + ", " + field, 1)
        complete = Mock(side_effect=[response(selection(first)), SimpleNamespace(text=raw)])
        result = select_city_preferences(req, pool, completion=complete, allow_depth_repair=True)
        self.assertEqual(result.candidate_priority, first)
        self.assertEqual(result.diagnostics["selection_repair_status"], "failed")
        self.assertNotIn("selection_review", result.diagnostics)
        good = {"reviewed_count": 32, "included_count": 20,
                "exclusion_counts": {"current_request_restriction": 12}}
        for bad in ({**good, "private_provider_text": "secret"},
                    {**good, "reviewed_count": 43}, {**good, "included_count": True},
                    {**good, "exclusion_counts": {"raw-private-reason": 12}},
                    {**good, "exclusion_counts": {"current_request_restriction": 11}}):
            self.assertEqual(safe_coverage({"selection_review": bad}), {})


if __name__ == "__main__":
    unittest.main()
