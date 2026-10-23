"""No network/AI/DB: bounded metadata and observable legacy calendar."""
from types import SimpleNamespace
import json
import unittest
from generation_quality import memory_note, public_quality, safe_coverage


class GenerationQualityTests(unittest.TestCase):
    def test_allowlist_drops_private_and_malformed_fields(self):
        raw = {"quality_status": "partial", "prompt": "private-secret", "map_requests": True,
               "quality_reasons": ["incomplete_calendar", {"secret": 1}, "private-secret"],
               "discovery": {"confirmed_candidates": 3, "raw_response": "private-secret",
                             "detail_rejection_reasons": {"name_mismatch": 2, "private-secret": 1},
                             "discovery": {"discovery": {"prompt": "secret"}}},
               "interest_coverage": {"zoos": {"status": "not_confirmed", "confirmed_count": 0,
                                               "selected_count": 0, "raw": "secret"},
                                     "museums": {"status": "covered", "confirmed_count": 1, "selected_count": 2}}}
        clean = safe_coverage(raw)
        self.assertNotIn("secret", json.dumps(clean))
        self.assertNotIn("map_requests", clean)
        self.assertNotIn("museums", clean["interest_coverage"])
        self.assertEqual(clean["discovery"]["detail_rejection_reasons"], {"name_mismatch": 2})

    def test_legacy_partial_does_not_invent_completed_audit(self):
        plan = SimpleNamespace(city_schedule=True, days=[SimpleNamespace(day_mode=mode) for mode in ["program"] + ["free"] * 6])
        quality = public_quality(None, plan)
        self.assertEqual(quality["quality_status"], "partial")
        self.assertEqual((quality["program_days"], quality["free_days"]), (1, 6))
        self.assertEqual(quality["interest_texts"], [])

    def test_stale_complete_cannot_override_actual_calendar(self):
        plan = SimpleNamespace(city_schedule=True, days=[SimpleNamespace(day_mode="free")])
        quality = public_quality({"quality_status": "complete", "program_days": 7}, plan)
        self.assertEqual(quality["quality_status"], "insufficient_data")
        self.assertEqual(quality["program_days"], 0)

    def test_category_text_distinguishes_absence_from_missing_confirmation(self):
        plan = SimpleNamespace(city_schedule=True, days=[SimpleNamespace(day_mode="program")])
        quality = public_quality({"quality_status": "partial", "interest_coverage": {
            "zoos": {"status": "not_confirmed", "confirmed_count": 0, "selected_count": 0}}}, plan)
        self.assertIn("не удалось подтвердить", quality["interest_texts"][0])

    def test_legacy_non_city_keeps_existing_program_report(self):
        self.assertEqual(public_quality({}, SimpleNamespace(city_schedule=False, days=[])), {})

    def test_unhashable_values_do_not_break_serialization(self):
        self.assertEqual(safe_coverage({"quality_status": [], "pool_quality": {}}), {})

    def test_selection_mode_is_an_allowlisted_value_not_provider_text(self):
        for mode in ("structured_preferences", "semantic_preferences", "memory_ranking"):
            self.assertEqual(safe_coverage({"selection_mode": mode, "selection_calls": 0}),
                             {"selection_mode": mode, "selection_calls": 0})
        self.assertEqual(safe_coverage({"selection_mode": "private provider response"}), {})
        self.assertEqual(safe_coverage({"selection_mode": {"secret": "value"}}), {})

    def test_memory_note_distinguishes_ranked_from_unavailable(self):
        self.assertIn("используется для приоритетов", memory_note({
            "selection_mode": "memory_ranking", "memory_ranking_status": "applied"}))
        self.assertIn("Не удалось применить", memory_note({
            "selection_mode": "memory_ranking", "memory_ranking_status": "unavailable"}))
        self.assertEqual(memory_note({"selection_mode": "structured_preferences"}), "")
