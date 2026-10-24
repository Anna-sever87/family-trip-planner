"""Verified type selects a fixed product estimate, never new visitor facts."""
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
import unittest
from unittest.mock import patch

import city_visit_options as options
import test_city_visit_options as fixtures


class CityTypePlanningEstimateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CityVisitOptionsTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_whole_facility_blocks_are_fixed_across_three_and_fourteen_days_and_ages(self):
        for kind, duration in (("zoo", 120), ("theme_park", 120), ("aquarium", 90)):
            candidate = fixtures.candidate(kind=kind)
            before = deepcopy(candidate)
            for days in (3, 14):
                for children, pace in (([], "low"), ([4], "medium"), ([14], "high")):
                    with self.subTest(kind=kind, days=days, children=children, pace=pace):
                        req = fixtures.request(end_date=(date(2026, 10, 28) + timedelta(days=days - 1)).isoformat(),
                                               children_ages=children, pace=pace)
                        value = options.visit_option(candidate, req)
                        self.assertEqual((value.duration_min, value.basis), (duration, "planning_estimate"))
                        self.assertLessEqual(value.duration_min, options.get_city_planning_cap(candidate))
                        self.assertEqual((value.allowed_dates, value.opening_windows, value.sources), (None, (), ()))
                        self.assertIsNone(value.indoor)
                        self.assertIsNone(value.estimated_cost)
                        self.assertIsNone(value.children_allowed)
                        self.assertIsNone(value.min_child_age)
                        self.assertEqual(set(value.unknown_fields), {"price", "hours", "indoor", "children", "minimum_age"})
                        self.assertIn("не подтверждённая длительность", value.description)
            self.assertEqual(candidate, before)

    def test_existing_positive_specialized_classifier_gets_shorter_block_not_whole_zoo(self):
        tagged = fixtures.candidate("Animal Centre", kind="zoo")
        tagged.record["extratags"] = {"zoo": "petting_zoo"}
        for candidate in (tagged, fixtures.candidate("Пони-ферма", kind="zoo"),
                          fixtures.candidate("Contact Petting Zoo", kind="zoo")):
            value = options.visit_option(candidate, fixtures.request())
            self.assertEqual((value.duration_min, value.basis), (60, "planning_estimate"))
            self.assertIsNone(value.children_allowed)
            self.assertIsNone(value.estimated_cost)

    def test_small_ride_and_unconfirmed_performance_do_not_receive_whole_facility_block(self):
        ride = fixtures.candidate("Test Ride", kind="attraction")
        ride.record["extratags"] = {"attraction": "roller_coaster"}
        for candidate in (ride, fixtures.candidate("Test Theatre", kind="theatre"),
                          fixtures.candidate("Test Cinema", kind="cinema")):
            value = options.visit_option(candidate, fixtures.request())
            self.assertEqual(value.duration_min, 45)
            self.assertIsNone(value.allowed_dates)
            if candidate.record["type"] in {"theatre", "cinema"}:
                self.assertIn("Спектакль или киносеанс в этот блок не включён", value.description)

    def test_longer_blocks_require_exact_category_and_valid_identity(self):
        for kind in ("zoo", "theme_park", "aquarium"):
            candidate = fixtures.candidate(kind=kind)
            candidate.record["category"] = "building"
            self.assertEqual(options.visit_option(candidate, fixtures.request()).duration_min, 45)
            candidate.record["category"] = "tourism"
            candidate.candidate_id = "osm-node-9999"
            self.assertIsNone(options.visit_option(candidate, fixtures.request()))

    def test_published_or_reviewed_conditions_keep_priority_and_limits(self):
        candidate, req = fixtures.candidate(kind="zoo"), fixtures.request(children_ages=[])
        card = self.fixture.card(candidate, indoor=None, minimum_age=0, duration_min=60)
        value = options.visit_option(candidate, req, visit_cards=[card])
        self.assertEqual((value.duration_min, value.basis, value.allowed_dates), (60, "published", ("2026-10-29",)))
        published = replace(options.lookup_city_visit_estimate(fixtures.yersin()),
                            typical_min_minutes=40, typical_max_minutes=50, planning_cap_minutes=50)
        with patch.object(options, "lookup_city_visit_estimate", return_value=published):
            self.assertEqual(options.visit_option(candidate, req).duration_min, 40)
        with patch.object(options, "get_city_planning_cap", return_value=90):
            self.assertIsNone(options.visit_option(candidate, req))  # Do not truncate to fit a cap.

    def test_longer_estimate_still_cannot_fit_short_sourced_window_or_override_age(self):
        candidate = fixtures.candidate(kind="zoo")
        req = fixtures.request(children_ages=[14])
        card = self.fixture.card(candidate, indoor=None, minimum_age=16, duration_min=120)
        self.assertIsNone(options.visit_option(candidate, req, visit_cards=[card]))
        card.update(minimum_age=0, hours=[{"dates": ["2026-10-29"], "start": "10:00", "end": "11:00"}])
        plan = self.fixture.itinerary(candidate, on_date="2026-10-29", time="10:00", duration=120, indoor=False)
        problems = options.check_city_visit_conditions(plan, [candidate], req, visit_cards=[card])
        self.assertTrue(any("целиком не помещается" in problem for problem in problems))

    def test_saved_old_forty_five_minute_zoo_remains_readable_without_rewriting_visit(self):
        import city_context
        import city_deterministic
        import web_service
        from city_visit_estimates import check_city_visit_estimates
        from test_city_deterministic import CENTER, candidates, request

        req, pool = request(1), candidates(4)
        pool[-1].record["type"] = "zoo"
        old_options = [options.visit_option(candidate, req) for candidate in pool]
        old_options[-1] = replace(old_options[-1], duration_min=45,
                                  description="Плановый блок 45 минут — прежняя оценка приложения.")
        built = city_deterministic.build_deterministic_city_itinerary(req, pool,
                    resolved_city=CENTER, options=old_options)
        self.assertFalse(built.problems)
        stored = {"id": 7, "start_date": req.start_date, "end_date": req.end_date,
                  "request": {**req.model_dump(), city_context.CONTEXT_KEY:
                  city_context.build_city_context(req, pool, CENTER)}, "itinerary": built.itinerary.model_dump()}
        before = deepcopy(stored)
        self.assertEqual(web_service._city_schedule_metadata_problems(built.itinerary, req), [])
        self.assertEqual(options.check_city_visit_conditions(built.itinerary, pool, req), [])
        self.assertEqual(check_city_visit_estimates(built.itinerary, pool), [])
        with patch.object(web_service.db, "get_trip", return_value=stored), \
             patch.object(web_service.db, "get_checklist", return_value=[]), \
             patch.object(web_service.db, "get_reminder_statuses", return_value={}):
            detail = web_service.get_trip_detail(1, 7)
        old_visits = [item for day in detail["itinerary"]["days"] for item in day["items"]
                      if item["place_query"] == pool[-1].place_query]
        self.assertEqual([item["duration_min"] for item in old_visits], [45])
        self.assertEqual(stored, before)
        self.assertEqual(options.visit_option(pool[-1], req).duration_min, 120)


if __name__ == "__main__":
    unittest.main()
