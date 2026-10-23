"""No network: city calendar capacity and explicit free-date assembly."""
from copy import deepcopy
from itertools import product
import unittest
from unittest.mock import patch

from city_coverage import (
    build_city_free_day, build_city_program_witness, city_program_day_capacity, select_program_days,
)
from city_quality import city_candidate_may_have_shelter
from city_visit_estimates import get_city_planning_cap
from pricing import minimum_meal_cost
from schemas import TripRequest
import test_city_candidate_planner as fixture
from test_city_candidate_planner import candidate, payload, request
from test_city_visit_estimates import yersin_candidate


class CityCoverageTests(unittest.TestCase):
    def pool(self, kinds):
        result = []
        for index, kind in enumerate(kinds):
            item = candidate()
            item.candidate_id = f"osm-node-{101 + index}"
            item.title = f"Городской объект {index + 1}"
            item.place_query = f"{item.title}, Тестгород, Россия"
            item.record.update(type=kind, osm_id=101 + index, name=item.place_query, lon=83.0 + index * .0001)
            result.append(item)
        return result

    def test_adult_capacity_uses_indivisible_visits_not_sum_of_invented_maxima(self):
        req = request().model_copy(update={"end_date": "2099-11-01"})
        for kinds, expected in (([], 0), (["artwork"], 0), (["artwork"] * 2, 0),
                                (["artwork"] * 3, 1), (["monument"] * 6, 2),
                                (["museum", "museum", "artwork", "artwork"], 2),
                                (["museum", "artwork", "monument", "memorial"], 1),
                                (["museum", "museum"] + ["artwork"] * 5, 3)):
            with self.subTest(kinds=kinds):
                self.assertEqual(city_program_day_capacity(req, self.pool(kinds)), expected)

    def test_default_caps_cannot_stretch_one_museum_or_park_into_an_adult_day(self):
        adult = request().model_copy(update={"end_date": "2099-10-30"})
        for kinds, expected in ((["museum"], 0), (["park"], 0), (["museum", "museum"], 1),
                                (["museum", "park"], 1), (["museum", "place_of_worship"], 1)):
            with self.subTest(kinds=kinds):
                self.assertEqual(city_program_day_capacity(adult, self.pool(kinds)), expected)
        family = adult.model_copy(update={"children_ages": [4, 10]})
        self.assertEqual(city_program_day_capacity(family, self.pool(["museum"])), 1)
        self.assertEqual(city_program_day_capacity(family, self.pool(["park"])), 1)

    def test_reviewed_yersin_cap_cannot_fill_ninety_or_one_eighty_minutes(self):
        adult = request().model_copy(update={"end_date": "2099-10-30"})
        family = adult.model_copy(update={"children_ages": [4, 10]})
        for req in (adult, family):
            with self.subTest(ages=req.children_ages):
                self.assertEqual(city_program_day_capacity(req, [yersin_candidate()]), 0)
                self.assertEqual(select_program_days(req, [yersin_candidate()], [req.start_date]), [])
        # 120 + 45 still cannot cover an adult target of 180.
        self.assertEqual(city_program_day_capacity(adult, [yersin_candidate(), candidate()]), 0)

    def test_two_distinct_reviewed_forty_five_minute_visits_can_cover_family_minimum(self):
        req = request().model_copy(update={"end_date": "2099-10-30", "children_ages": [4, 10]})
        candidates = self.pool(["museum", "museum"])
        # Generic capacity regression: not two copies of the real Yersin POI.
        with patch("city_coverage.get_city_planning_cap", return_value=45):
            self.assertEqual(city_program_day_capacity(req, candidates), 1)
            self.assertEqual(select_program_days(req, candidates, [req.start_date]), [2])
            self.assertEqual(city_program_day_capacity(req.model_copy(update={"children_ages": []}), candidates), 0)
            with patch("planner._max_attractions_for_request", return_value=1):
                self.assertEqual(city_program_day_capacity(req, candidates), 0)

    def test_caps_cannot_be_combined_beyond_per_day_visit_count(self):
        req = request().model_copy(update={"end_date": "2099-10-30"})
        with patch("city_coverage.get_city_planning_cap", return_value=45), \
                patch("planner._max_attractions_for_request", return_value=3):
            self.assertEqual(city_program_day_capacity(req, self.pool(["museum"] * 4)), 0)

    def test_family_capacity_groups_short_stops_under_existing_ninety_minute_target(self):
        req = request().model_copy(update={"end_date": "2099-10-31", "children_ages": [5]})
        for kinds, expected in ((["artwork"], 0), (["artwork", "monument"], 1),
                                (["artwork", "monument", "memorial"], 1),
                                (["artwork"] * 4, 2), (["museum", "artwork"], 1),
                                (["museum", "artwork", "monument"], 2)):
            with self.subTest(kinds=kinds):
                self.assertEqual(city_program_day_capacity(req, self.pool(kinds)), expected)
        self.assertEqual(select_program_days(req, self.pool(["artwork", "monument"])), [1])

    def test_higher_adult_target_visit_cap_and_requested_days_bound_capacity(self):
        req = request().model_copy(update={"end_date": "2099-10-31", "end_time": "23:00"})
        self.assertEqual(city_program_day_capacity(req, self.pool(["artwork"] * 3)), 0)
        self.assertEqual(city_program_day_capacity(req, self.pool(["artwork"] * 4)), 1)
        self.assertEqual(city_program_day_capacity(request(), self.pool(["museum"] * 5)), 1)
        family = req.model_copy(update={"children_ages": [5]})
        with patch("planner._max_attractions_for_request", return_value=1):
            self.assertEqual(city_program_day_capacity(family, self.pool(["artwork"] * 4)), 0)
            self.assertEqual(city_program_day_capacity(family, self.pool(["museum"] + ["artwork"] * 4)), 1)

    def test_calm_short_and_implicit_windows_keep_one_candidate_per_possible_day(self):
        pool = self.pool(["artwork", "monument", "memorial"])
        base = request().model_copy(update={"end_date": "2099-10-30", "children_ages": [5]})
        for req in (base.model_copy(update={"pace": "спокойный"}),
                    base.model_copy(update={"end_time": "16:59"}),
                    TripRequest(**{key: value for key, value in base.model_dump().items()
                                   if key not in {"start_time", "end_time"}})):
            with self.subTest(request=req):
                self.assertEqual(city_program_day_capacity(req, pool), 3)

    def test_capacity_does_not_invent_or_change_candidate_duration_and_weather_bound_remains(self):
        req = request().model_copy(update={"end_date": "2099-10-31"})
        pool = self.pool(["museum", "museum", "artwork", "monument"])
        for item in pool:
            item.record["duration_min"] = 240  # Untrusted metadata does not create extra capacity.
        before = deepcopy(pool)
        self.assertEqual(city_program_day_capacity(req, pool), 2)
        self.assertEqual(select_program_days(req, pool, ["2099-10-28", "2099-10-29", "2099-10-30"]), [1, 4])
        self.assertEqual(pool, before)

    def test_rain_limits_program_dates_without_reusing_museum(self):
        req = request().model_copy(update={"end_date": "2099-10-31"})
        candidates = [candidate() for _ in range(4)]
        for index, item in enumerate(candidates):
            item.candidate_id = f"id-{index}"
            item.record["type"] = "park" if index else "museum"
        self.assertEqual(select_program_days(req, candidates, ["2099-10-30", "2099-10-31"]), [1, 2])
        self.assertEqual(select_program_days(req, candidates, ["2099-10-28", "2099-10-30"]), [2, 4])

    def test_empty_capacity_does_not_authorize_empty_trip(self):
        self.assertEqual(select_program_days(request(), []), [])
        park = candidate()
        park.record["type"] = "park"
        self.assertEqual(select_program_days(request(), [park], ["2099-10-28"]), [])

    def test_open_air_museum_has_dry_capacity_but_cannot_create_a_rainy_program_date(self):
        req = request().model_copy(update={"children_ages": [5], "end_date": "2099-10-29"})
        pool = self.pool(["museum"])
        pool[0].record["name"] = "Folklorik Yörük Parkı Açık Hava Müzesi, Kemer, Türkiye"
        before = deepcopy(pool)
        self.assertEqual(city_program_day_capacity(req, pool), 1)
        self.assertEqual(select_program_days(req, pool, ["2099-10-28", "2099-10-29"]), [])
        self.assertEqual(select_program_days(req, pool, ["2099-10-28"]), [2])
        self.assertEqual(pool, before)

    def test_namedetails_outdoor_museum_does_not_consume_an_indoor_candidate_slot(self):
        req = request().model_copy(update={"children_ages": [5], "end_date": "2099-10-30"})
        pool = self.pool(["museum", "museum", "museum"])
        pool[0].record["namedetails"] = {"official_name:en": "Open-Air Museum"}
        rainy = ["2099-10-28", "2099-10-29", "2099-10-30"]
        self.assertEqual(select_program_days(req, pool, rainy), [1, 2])
        pool[0].record["namedetails"] = {"official_name:en": "Air Museum"}
        self.assertEqual(select_program_days(req, pool, rainy), [1, 2, 3])

    def test_duplicate_candidate_id_and_external_rain_date_rejected(self):
        with self.assertRaises(ValueError):
            select_program_days(request(), [candidate(), candidate()])
        with self.assertRaises(ValueError):
            select_program_days(request(), [candidate()], ["2099-10-29"])

    def test_free_date_preserves_date_budget_and_no_place_claims(self):
        req = request().model_copy(update={"end_date": "2099-10-31", "children_ages": [4, 10]})
        day = build_city_free_day(req, 4)
        self.assertEqual((day.day, day.date, day.day_mode, day.base_area), (4, "2099-10-31", "free", ""))
        self.assertEqual([item.kind for item in day.items], ["meal", "rest"])
        self.assertEqual(sum(item.cost for item in day.items), minimum_meal_cost(req))
        self.assertTrue(all(not item.verified and item.lat is None and not item.place_query for item in day.items))
        self.assertGreaterEqual(day.items[1].duration_min, 60)

    def test_free_day_time_bounds_and_invalid_index(self):
        req = request().model_copy(update={"start_time": "17:00", "end_time": "19:00"})
        self.assertEqual([item.time for item in build_city_free_day(req, 1).items], ["17:00", "18:00"])
        with self.assertRaises(ValueError):
            build_city_free_day(req.model_copy(update={"end_time": "18:59"}), 1)
        for number in (0, 2, True):
            with self.assertRaises(ValueError):
                build_city_free_day(req, number)

    def test_joint_rain_depth_packing_does_not_request_three_impossible_days(self):
        req = request().model_copy(update={"end_date": "2099-10-30", "children_ages": [5]})
        candidates = self.pool(["park", "park", "place_of_worship", "place_of_worship"])
        rainy = ["2099-10-29", "2099-10-30"]
        before = deepcopy((req, candidates))
        self.assertEqual(city_program_day_capacity(req, candidates), 3)  # Still only an upper bound.
        witness = build_city_program_witness(req, candidates, rainy)
        self.assertEqual(witness.program_day_numbers, (1, 2))
        self.assertEqual(select_program_days(req, candidates, rainy), [1, 2])
        self.assertFalse(witness.search_exhausted)
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        flattened = [identity for group in witness.groups_by_day.values() for identity in group]
        self.assertEqual(len(flattened), len(set(flattened)))
        for number, identities in witness.groups_by_day.items():
            group = [by_id[identity] for identity in identities]
            self.assertGreaterEqual(sum(get_city_planning_cap(candidate) for candidate in group), 90)
            self.assertLessEqual(len(group), 3)
            if number != 1:
                self.assertGreaterEqual(sum(city_candidate_may_have_shelter(candidate.record)
                                            for candidate in group) * 2, len(group))
        self.assertEqual((req, candidates), before)  # No duration, indoor or source facts invented.

    def test_groups_use_pairwise_eight_km_limit_not_just_an_anchor_distance(self):
        req = request().model_copy(update={"end_date": "2099-10-30"})
        candidates = self.pool(["artwork"] * 3)
        # A--B and A--C fit, but B--C is too far: a centroid/anchor-only check is insufficient.
        with patch("city_coverage.distance_km", side_effect=[4, 4, 8.000001]):
            self.assertEqual(select_program_days(req, candidates), [])
        with patch("city_coverage.distance_km", side_effect=[4, 4, 8.0]):
            witness = build_city_program_witness(req, candidates)
        self.assertEqual(witness.program_day_numbers, (1,))
        self.assertEqual(len(witness.groups_by_day[1]), 3)

    def test_disjoint_geographic_clusters_get_separate_days(self):
        req = request().model_copy(update={"end_date": "2099-10-30"})
        candidates = self.pool(["museum"] * 4)
        for index, candidate in enumerate(candidates):
            candidate.record["lon"] = 83.0 + (0.25 if index >= 2 else 0) + index * .0001
        witness = build_city_program_witness(req, candidates)
        self.assertEqual(witness.program_day_numbers, (1, 2))
        self.assertEqual({frozenset(group) for group in witness.groups_by_day.values()},
                         {frozenset(candidate.candidate_id for candidate in candidates[:2]),
                          frozenset(candidate.candidate_id for candidate in candidates[2:])})

    def test_rain_packing_uses_existing_count_rule_and_does_not_assert_indoor(self):
        req = request().model_copy(update={"children_ages": [5]})
        candidates = self.pool(["park", "place_of_worship"])
        witness = build_city_program_witness(req, candidates, [req.start_date])
        self.assertEqual(witness.program_day_numbers, (1,))
        self.assertEqual(len(witness.groups_by_day[1]), 2)
        # The group only admits a possible future schedule. It has no indoor flags or durations.
        self.assertTrue(all(isinstance(identity, str) for identity in witness.groups_by_day[1]))
        self.assertTrue(all("indoor" not in candidate.record and "duration_min" not in candidate.record
                            for candidate in candidates))

    def test_search_limit_counts_group_checks_and_returns_only_found_groups(self):
        req = request().model_copy(update={"end_date": "2099-10-30"})
        candidates = self.pool(["museum"] * 2)
        empty = build_city_program_witness(req, candidates, max_states=1)
        self.assertEqual(empty.groups_by_day, {})
        self.assertTrue(empty.search_exhausted)
        found = build_city_program_witness(req, candidates, max_states=4)
        self.assertEqual(found.program_day_numbers, (1,))
        self.assertEqual(set(found.groups_by_day[1]), {candidate.candidate_id for candidate in candidates})
        self.assertTrue(found.search_exhausted)
        self.assertEqual(found.groups_examined, 3)
        self.assertLessEqual(found.states_explored, 4)
        self.assertGreater(found.states_explored, found.groups_examined)

    def test_witness_is_deterministic_and_input_order_independent(self):
        req = request().model_copy(update={"end_date": "2099-10-31", "children_ages": [5]})
        candidates = self.pool(["park", "museum", "place_of_worship", "museum", "park"])
        rainy = ["2099-10-28", "2099-10-30"]
        forward = build_city_program_witness(req, candidates, rainy)
        reverse = build_city_program_witness(req, list(reversed(candidates)), list(reversed(rainy)))
        self.assertEqual(forward, reverse)
        self.assertEqual(forward.program_day_numbers, (1, 2, 3, 4))

    def test_invalid_or_missing_coordinates_fail_closed(self):
        for value in (None, True, float("inf"), float("nan"), 91, 10 ** 1000):
            candidates = self.pool(["museum"] * 2)
            candidates[0].record["lat"] = value
            with self.subTest(latitude=value), self.assertRaises(ValueError):
                build_city_program_witness(request(), candidates)
        candidates = self.pool(["museum"] * 2)
        del candidates[0].record["lon"]
        with self.assertRaises(ValueError):
            build_city_program_witness(request(), candidates)
        for limit in (0, True, 20_001, 1.5):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                build_city_program_witness(request(), self.pool(["museum"]), max_states=limit)
        with self.assertRaises(ValueError):
            build_city_program_witness(request(), self.pool(["museum"] * 19))

    def test_maximum_pool_is_bounded_and_does_not_reuse_ids(self):
        req = request().model_copy(update={"end_date": "2099-11-10"})
        candidates = self.pool(["museum"] * 18)
        witness = build_city_program_witness(req, candidates)
        self.assertLessEqual(witness.states_explored, 20_000)
        self.assertLessEqual(witness.groups_examined, witness.states_explored)
        self.assertEqual(witness.program_day_numbers, tuple(range(1, 10)))
        identities = [identity for group in witness.groups_by_day.values() for identity in group]
        self.assertEqual(len(identities), 18)
        self.assertEqual(len(set(identities)), 18)

    def test_small_joint_pools_match_independent_exhaustive_assignment(self):
        req = request().model_copy(update={"end_date": "2099-10-30", "children_ages": [5]})
        for kinds in (["park", "park", "place_of_worship", "place_of_worship"],
                      ["museum", "park", "artwork", "place_of_worship"],
                      ["museum", "museum", "artwork", "artwork"]):
            candidates = self.pool(kinds)
            for wet_numbers in ((), (2, 3), (1, 2, 3)):
                best = (0, 0)
                for assignment in product(range(4), repeat=len(candidates)):
                    total = dry_count = 0
                    for day in (1, 2, 3):
                        group = [candidate for candidate, number in zip(candidates, assignment) if number == day]
                        if not group:
                            continue
                        if len(group) > 3 or sum(get_city_planning_cap(candidate) for candidate in group) < 90:
                            break
                        if day in wet_numbers and sum(city_candidate_may_have_shelter(candidate.record)
                                                       for candidate in group) * 2 < len(group):
                            break
                        total += 1
                        dry_count += day not in wet_numbers
                    else:
                        best = max(best, (total, dry_count))
                rainy = [f"2099-10-{27 + number}" for number in wet_numbers]
                witness = build_city_program_witness(req, candidates, rainy)
                with self.subTest(kinds=kinds, wet_numbers=wet_numbers):
                    self.assertFalse(witness.search_exhausted)
                    self.assertEqual((len(witness.program_day_numbers),
                                      sum(day not in wet_numbers for day in witness.program_day_numbers)), best)


class CityPartialPlannerTests(unittest.TestCase):
    # Composition, not inheritance: do not rerun the helper's entire suite.
    def run_plan(self, *args, **kwargs):
        return fixture.CityCandidatePlannerTests().run_plan(*args, **kwargs)

    def test_joint_witness_allows_valid_two_plus_one_without_relabelling_failed_days(self):
        req = request().model_copy(update={"end_date": "2099-10-30", "children_ages": [5]})
        candidates = CityCoverageTests().pool(["park", "park", "place_of_worship", "place_of_worship"])
        weather = {"city": "Тестгород", "source": "forecast", "days": [
            {"date": "2099-10-28", "tmin": 10, "tmax": 20, "rain_mm": 0},
            {"date": "2099-10-29", "tmin": 10, "tmax": 20, "rain_mm": 5},
            {"date": "2099-10-30", "tmin": 10, "tmax": 20, "rain_mm": 5}]}
        witness = build_city_program_witness(req, candidates, ["2099-10-29", "2099-10-30"])
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        response = {}
        for number, identities in witness.groups_by_day.items():
            day = deepcopy(payload(second_visit=False)["day_1"])
            day.update(day=number, date=f"2099-10-{27 + number}")
            day["meal"]["cost"] = 1800
            day["attractions"] = []
            for index, identity in enumerate(identities):
                inside = by_id[identity].record["type"] == "place_of_worship"
                day["attractions"].append(dict(
                    candidate_id=identity, time=("09:00", "11:15")[index],
                    duration_min=60 if inside else 90, cost=0, kids_note="", indoor=inside,
                    description="Осмотр внутреннего зала" if inside else "Прогулка по парку",
                ))
            response[f"day_{number}"] = day
        before = deepcopy(response)
        result, calls, _ = self.run_plan([response], trip_request=req, candidates=candidates,
                                        weather=weather, max_attempts=2)
        self.assertEqual(result.problems, [])
        self.assertEqual(calls.call_count, 2)
        self.assertEqual([day.day_mode for day in result.itinerary.days], ["program", "program", "free"])
        self.assertEqual(result.itinerary.program_report.program_days, 2)
        self.assertEqual(sum(item.cost for day in result.itinerary.days for item in day.items), 5400)
        self.assertEqual(response, before)
        schema = calls.call_args_list[1].kwargs["json_schema"]
        self.assertEqual(set(schema["properties"]), {"day_1", "day_2"})

    def test_one_program_plus_two_free_days_pass_real_guards(self):
        req = request().model_copy(update={"end_date": "2099-10-30"})
        response = {"day_1": payload()["day_1"]}
        result, calls, _ = self.run_plan([response], trip_request=req, max_attempts=2)
        self.assertEqual(result.problems, [])
        self.assertEqual(calls.call_count, 2)
        itinerary = result.itinerary
        self.assertTrue(itinerary.city_schedule)
        self.assertFalse(itinerary.source_schedule)
        self.assertEqual([day.day_mode for day in itinerary.days], ["program", "free", "free"])
        report = itinerary.program_report
        self.assertEqual((report.coverage_basis, report.program_days, report.free_days,
                          report.eligible_unique_visits, report.selected_unique_visits), ("map", 1, 2, 2, 2))
        self.assertEqual(sum(item.cost for day in itinerary.days for item in day.items), 6200)

    def test_rainy_first_date_is_free_without_renumbering_day_two(self):
        req = request().model_copy(update={"end_date": "2099-10-29", "children_ages": [5]})
        park = candidate()
        park.record["type"] = "park"
        day = payload(duration=90, indoor=False, second_visit=False)["day_1"]
        day.update(day=2, date="2099-10-29")
        day["meal"]["cost"] = 1800
        weather = {"city": "Тестгород", "source": "forecast", "days": [
            {"date": "2099-10-28", "tmin": 10, "tmax": 20, "rain_mm": 5},
            {"date": "2099-10-29", "tmin": 10, "tmax": 20, "rain_mm": 0}]}
        result, _, _ = self.run_plan([{"day_2": day}], trip_request=req, candidates=[park],
                                     weather=weather, max_attempts=2)
        self.assertEqual(result.problems, [])
        self.assertEqual([(day.day, day.date, day.day_mode) for day in result.itinerary.days],
                         [(1, "2099-10-28", "free"), (2, "2099-10-29", "program")])

    def test_invalid_short_program_is_not_relabelled_free_to_pass(self):
        req = request().model_copy(update={"end_date": "2099-10-29"})
        result, calls, _ = self.run_plan([{"day_1": payload(duration=15)["day_1"]}],
                                        trip_request=req, max_attempts=2)
        self.assertTrue(any("недостаточно содержательная" in value for value in result.problems))
        self.assertEqual(calls.call_count, 2)

    def test_model_cannot_invent_free_dates_or_markers(self):
        req = request().model_copy(update={"end_date": "2099-10-29"})
        for field in ("city_schedule", "source_schedule", "program_report", "day_2"):
            value = {"day_1": deepcopy(payload()["day_1"]), field: True}
            result, _, _ = self.run_plan([value], trip_request=req, max_attempts=2)
            self.assertTrue(result.problems)

    def test_insufficient_meal_budget_stops_before_schedule(self):
        req = request().model_copy(update={"end_date": "2099-10-30", "budget": 4000})
        result, calls, _ = self.run_plan([], trip_request=req, max_attempts=2)
        self.assertEqual(calls.call_count, 1)
        self.assertTrue(any("обеды" in value for value in result.problems))

    def test_two_short_stops_can_fill_one_family_program_day_plus_one_server_free_day(self):
        req = request().model_copy(update={"end_date": "2099-10-29", "children_ages": [5]})
        candidates = CityCoverageTests().pool(["artwork", "monument"])
        day = payload(duration=45, indoor=False, second_visit=False)["day_1"]
        day["meal"]["cost"] = 1800
        second = deepcopy(day["attractions"][0])
        second.update(candidate_id=candidates[1].candidate_id, time="10:00")
        day["attractions"].append(second)
        result, calls, _ = self.run_plan([{"day_1": day}], candidates=candidates,
                                        trip_request=req, max_attempts=2)
        self.assertEqual(result.problems, [])
        self.assertEqual(calls.call_count, 2)
        self.assertEqual([day.day_mode for day in result.itinerary.days], ["program", "free"])
        self.assertEqual(result.itinerary.program_report.selected_unique_visits, 2)
        self.assertEqual(sum(item.cost for day in result.itinerary.days for item in day.items), 5600)
        visits = [item for item in result.itinerary.days[0].items if item.kind == "attraction"]
        self.assertEqual([item.duration_min for item in visits], [45, 45])
        self.assertTrue(all(item.verified and not item.indoor for item in visits))


if __name__ == "__main__":
    unittest.main()
