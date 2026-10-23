"""Offline quality recomputation after an edit; never stale creation claims."""

from copy import deepcopy
import unittest

from city_deterministic import build_deterministic_city_itinerary
from city_planner import evaluate_saved_city_quality
from test_city_deterministic import CENTER, candidates, request


class SavedCityQualityTests(unittest.TestCase):
    def test_filled_calendar_drops_old_partial_reasons_and_recounts_visits(self):
        req, pool = request(2, interest_categories=["museums"]), candidates(4)
        itinerary = build_deterministic_city_itinerary(req, pool, resolved_city=CENTER).itinerary
        prior = {"quality_status": "partial", "quality_reasons": ["incomplete_calendar", "search_limit"],
                 "selected_unique_visits": 99, "states_explored": 20000,
                 "discovery": {"map_requests": 12, "prompt": "synthetic-private-marker"}}
        before = deepcopy((req.model_dump(), pool, itinerary.model_dump(), prior))
        quality = evaluate_saved_city_quality(req, pool, itinerary, prior)
        self.assertEqual(quality["quality_status"], "complete")
        self.assertEqual(quality["quality_reasons"], [])
        self.assertEqual(quality["selected_unique_visits"], 4)
        self.assertEqual(quality["interest_coverage"]["museums"]["selected_count"], 4)
        self.assertNotIn("states_explored", quality)
        self.assertEqual(quality["discovery"], {"map_requests": 12})
        self.assertEqual((req.model_dump(), pool, itinerary.model_dump(), prior), before)

    def test_removed_mall_is_not_still_reported_as_covered_after_edit(self):
        req = request(1, interest_categories=["museums", "shopping_malls"])
        pool = candidates(4)
        pool[-1].record.update(category="shop", type="mall")
        itinerary = build_deterministic_city_itinerary(req, pool[:3], resolved_city=CENTER).itinerary
        prior = {"quality_status": "complete", "interest_coverage": {
            "shopping_malls": {"status": "covered", "confirmed_count": 1, "selected_count": 1}}}
        quality = evaluate_saved_city_quality(req, pool, itinerary, prior)
        self.assertEqual(quality["quality_status"], "partial")
        self.assertEqual(quality["quality_reasons"], ["requested_interest_missing"])
        self.assertEqual(quality["interest_coverage"]["shopping_malls"],
                         {"status": "not_scheduled", "confirmed_count": 1, "selected_count": 0})

    def test_remaining_free_days_have_no_invented_budget_weather_attribution(self):
        req, pool = request(7), candidates(3)
        itinerary = build_deterministic_city_itinerary(req, pool, resolved_city=CENTER).itinerary
        quality = evaluate_saved_city_quality(req, pool, itinerary, {
            "quality_reasons": ["budget_constraint", "indoor_evidence_missing", "search_limit"]})
        self.assertEqual(quality["quality_status"], "partial")
        self.assertEqual(quality["quality_reasons"], ["incomplete_calendar"])
        self.assertEqual((quality["program_days"], quality["free_days"], quality["selected_unique_visits"]), (1, 6, 3))

    def test_unbound_route_cannot_supply_quality_claims(self):
        req, pool = request(1), candidates(2)
        itinerary = build_deterministic_city_itinerary(req, pool, resolved_city=CENTER).itinerary
        with self.assertRaises(ValueError):
            evaluate_saved_city_quality(req, pool[:1], itinerary)
        with self.assertRaises(ValueError):
            evaluate_saved_city_quality(req.model_copy(update={"destination_kind": "region"}), pool, itinerary)


if __name__ == "__main__":
    unittest.main()
