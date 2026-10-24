"""Quality persistence through the real bot jobs, without starting Telegram."""

from contextlib import ExitStack, nullcontext
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch

import city_context
from generation_quality import KEY
import test_bot_generation_outcome as fixtures


class BotQualityPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("access_guard.operation", side_effect=lambda *a, **k: nullcontext()))
        self.stack.enter_context(patch("access_control.require_access", return_value=None))
        self.harness = fixtures.BotGenerationOutcomeTests(methodName="runTest")
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)
        self.namespace = self.harness.namespace
        self.namespace["validate_city_edit_context"] = city_context.validate_city_edit_context
        self.request, self.route, self.context = fixtures.city_fixture()
        self.harness.request = self.request
        self.harness.result.itinerary = self.route
        self.harness.result.city_map_context = self.context

    def test_create_keeps_safe_partial_report_and_discloses_partial_result(self):
        marker = "synthetic-private-quality-marker"
        self.harness.result.generation_coverage = {
            "quality_status": "partial", "program_days": 1, "free_days": 1,
            "quality_reasons": ["incomplete_calendar"], "private_notes": marker,
            "discovery": {"map_requests": 5, "raw_provider_text": marker}}
        self.harness.run_job()
        saved = self.namespace["db"].save_trip.call_args.args[1][KEY]
        self.assertEqual(saved["quality_status"], "partial")
        self.assertEqual((saved["program_days"], saved["free_days"]), (1, 1))
        self.assertNotIn(marker, json.dumps(saved))
        self.assertEqual(saved["discovery"], {"map_requests": 5})
        self.assertTrue(any("Программа составлена частично" in call.args[1]
                            for call in self.namespace["bot"].send_message.call_args_list))
        self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", True)

    def test_edit_recomputes_partial_report_instead_of_copying_stale_success(self):
        prior = {"quality_status": "complete", "program_days": 7, "free_days": 0,
                 "selected_unique_visits": 999, "quality_reasons": ["budget_constraint"],
                 "discovery": {"map_requests": 5}}
        stored = {**self.request.model_dump(), city_context.CONTEXT_KEY: deepcopy(self.context), KEY: prior}
        before = deepcopy(stored)
        self.namespace["db"].get_trip.return_value = {
            "request": stored, "itinerary": self.route.model_dump(), "revision": 1}
        self.namespace["db"].update_trip_plan.return_value = True
        self.harness.result.itinerary = self.route.model_copy(deep=True)
        self.harness.result.generation_coverage = {}
        self.namespace["edit_itinerary"] = Mock(return_value=self.harness.result)
        fixtures._job(self.namespace, "edit_generation_job")(
            100, 99, 1, "Измени время обеда", "test-claim",
            city_map_context=deepcopy(self.context), expected_revision=1)
        self.namespace["db"].update_trip_plan.assert_called_once()
        saved = self.namespace["db"].update_trip_plan.call_args.args[2][KEY]
        self.assertEqual(saved["quality_status"], "partial")
        self.assertEqual((saved["requested_days"], saved["program_days"], saved["free_days"]), (2, 1, 1))
        self.assertEqual(saved["selected_unique_visits"], 2)
        # The stored program day contains only two morning visits. The edit
        # must disclose both its missing dates and its observable lightness.
        self.assertEqual(saved["quality_reasons"],
                         ["day_program_light", "day_window_unbalanced", "incomplete_calendar"])
        self.assertEqual((saved["desired_program_minutes"], saved["light_program_days"],
                          saved["unbalanced_program_days"]), (300, 1, 1))
        self.assertEqual(saved["discovery"], {"map_requests": 5})
        self.assertEqual(stored, before)
        self.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", True)


if __name__ == "__main__":
    unittest.main()
