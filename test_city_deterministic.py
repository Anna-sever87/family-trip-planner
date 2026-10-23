"""Offline deterministic schedules: actual guards, not mocked acceptance."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
import unittest
from unittest.mock import patch

import city_deterministic as engine
from city_candidates import CityCandidate
from city_quality import check_city_quality
from city_visit_estimates import check_city_visit_estimates
from city_visit_options import CityVisitOption
from places import verify_itinerary
from pricing import minimum_meal_cost
from rules import check_program_depth, check_rules, check_verified_place_duplicates
from schemas import TripRequest


CENTER = {"name": "Тестгород", "lat": 55.0, "lon": 83.0, "country_code": "RU"}


def request(days=3, **changes):
    values = dict(destination="Тестгород, Россия", start_date="2099-10-28",
                  end_date=(date(2099, 10, 28) + timedelta(days=days - 1)).isoformat(),
                  adults=2, children_ages=[], budget=60000, currency="RUB",
                  start_time="09:00", end_time="19:00", pace="средний")
    values.update(changes)
    return TripRequest(**values)


def candidates(count=6, *, kind="museum", separation=0.0001):
    values = []
    for index in range(count):
        title = f"Музей городской истории номер {index + 1}"
        query = f"{title}, Тестгород, Россия"
        values.append(CityCandidate(
            candidate_id=f"osm-node-{101 + index}", title=title, place_query=query, area="Тестгород",
            record=dict(lat=55.0, lon=83.0 + separation * index, name=query, provider="osm",
                        osm_type="node", osm_id=101 + index, category="tourism", type=kind,
                        address={"city": "Тестгород", "country": "Россия", "country_code": "ru"})))
    return values


def options(pool, *, duration=90, indoor=None, cost=1000):
    return [CityVisitOption(value.candidate_id, duration, indoor, cost,
                           "Осмотр экспозиции; длительность является плановой оценкой.",
                           "planning_estimate", ("indoor", "hours", "children", "minimum_age")) for value in pool]


def visits(itinerary):
    return [item for day in itinerary.days for item in day.items if item.kind == "attraction"]


def start(item):
    return int(item.time[:2]) * 60 + int(item.time[3:])


class DeterministicCityTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.external = [self.stack.enter_context(patch(name, side_effect=AssertionError("External IO forbidden")))
                         for name in ("httpx.get", "httpx.post", "places.geocode", "places._load_cache",
                                      "places._save_cache", "sqlite3.connect", "planner.generate_json")]

    def tearDown(self):
        for mocked in self.external:
            mocked.assert_not_called()

    def build(self, req=None, pool=None, *, chosen_options=None, rainy=(), **kwargs):
        req = req or request()
        pool = candidates() if pool is None else pool
        return engine.build_deterministic_city_itinerary(
            req, pool, rainy, resolved_city=CENTER, options=chosen_options, **kwargs)

    def assert_valid(self, result, req, pool, rainy=()):
        self.assertEqual((result.status, result.problems), ("success", []), result.coverage)
        itinerary = result.itinerary
        records = {value.place_query: value.record for value in pool}
        self.assertEqual(verify_itinerary(itinerary, verified_records=records), [])
        self.assertEqual(check_rules(itinerary, req, set(rainy), allow_city_limited_program=True), [])
        self.assertEqual(check_program_depth(itinerary, req, allow_city_limited_program=True), [])
        self.assertEqual(check_city_quality(itinerary, req, records, allow_city_limited_program=True), [])
        self.assertEqual(check_city_visit_estimates(itinerary, pool), [])
        self.assertEqual(check_verified_place_duplicates(itinerary, verified_records=records), [])
        self.assertEqual([day.date for day in itinerary.days],
                         [(date.fromisoformat(req.start_date) + timedelta(days=index)).isoformat()
                          for index in range(req.days_count)])
        self.assertEqual(len({item.place_query for item in visits(itinerary)}), len(visits(itinerary)))
        self.assertLessEqual(sum(item.cost for day in itinerary.days for item in day.items), req.program_budget)
        for day in itinerary.days:
            self.assertEqual(sum(item.kind == "meal" for item in day.items), 1)
            self.assertEqual(sum(item.kind == "rest" for item in day.items), 1)
            self.assertEqual(next(item.cost for item in day.items if item.kind == "meal"), minimum_meal_cost(req))
            for item in day.items:
                if item.kind != "attraction":
                    self.assertFalse(item.verified)
                    self.assertEqual((item.place_query, item.provider, item.source_url, item.lat, item.lon),
                                     ("", "", "", None, None))

    def test_full_calendar_without_ai_and_durations_are_fixed_not_caps(self):
        req, pool = request(), candidates()
        result = self.build(req, pool)
        self.assert_valid(result, req, pool)
        self.assertIsNone(result.itinerary.program_report)
        self.assertEqual(result.coverage["program_days"], 3)
        self.assertEqual([item.duration_min for item in visits(result.itinerary)], [90] * 6)
        self.assertTrue(all(not item.indoor for item in visits(result.itinerary)))
        self.assertIn(engine.UNKNOWN_INDOOR_NOTE, result.itinerary.tips)

    def test_partial_calendar_reserves_meals_on_every_free_date(self):
        req, pool = request(), candidates(4)
        result = self.build(req, pool, chosen_options=options(pool))
        self.assert_valid(result, req, pool)
        report = result.itinerary.program_report
        self.assertEqual((report.coverage_basis, report.program_days, report.free_days), ("map", 2, 1))
        self.assertEqual(sum(item.cost for day in result.itinerary.days for item in day.items), 8200)
        free = next(day for day in result.itinerary.days if day.day_mode == "free")
        self.assertEqual((free.base_area, free.transfer_note), ("", ""))
        self.assertEqual([item.kind for item in free.items], ["meal", "rest"])

    def test_unique_visits_are_not_repeated_or_stretched_to_cover_days(self):
        req, pool = request(), candidates(1)
        result = self.build(req, pool)
        self.assertIsNone(result.itinerary)
        self.assertEqual(result.status, "insufficient_data")
        self.assertIn("no_feasible_program_day", result.reasons)

    def test_budget_chooses_fewer_program_days_without_changing_request(self):
        req, pool = request(budget=6200), candidates(4)
        before = req.model_dump()
        result = self.build(req, pool, chosen_options=options(pool))
        self.assert_valid(result, req, pool)
        self.assertEqual((result.coverage["program_days"], result.coverage["free_days"]), (1, 2))
        self.assertEqual(req.model_dump(), before)

    def test_insufficient_meal_budget_returns_no_all_free_plan(self):
        result = self.build(request(budget=4199))
        self.assertEqual((result.status, result.itinerary), ("insufficient_data", None))
        self.assertIn("meal_budget", result.reasons)

    def test_unknown_price_is_disclosed_as_unpriced_not_free_admission(self):
        req, pool = request(1), candidates(2)
        result = self.build(req, pool, chosen_options=options(pool, cost=None))
        self.assert_valid(result, req, pool)
        self.assertEqual(result.coverage["unknown_price_visits"], 2)
        self.assertTrue(all(item.cost == 0 for item in visits(result.itinerary)))
        self.assertTrue(any("неуточнённой ценой" in note and "не означает бесплатный вход" in note
                            for note in result.itinerary.tips))

    def test_rain_cannot_promote_unknown_museum_to_indoor(self):
        req, pool = request(), candidates(6)
        rainy = ["2099-10-28", "2099-10-29", "2099-10-30"]
        result = self.build(req, pool, rainy=rainy)
        self.assertEqual((result.status, result.itinerary), ("insufficient_data", None))

    def test_rain_uses_known_shelter_on_distinct_dates(self):
        req, pool = request(children_ages=[5]), candidates(3)
        chosen = options(pool)
        chosen[1] = replace(chosen[1], indoor=True)
        chosen[2] = replace(chosen[2], indoor=True)
        rainy = ["2099-10-29", "2099-10-30"]
        result = self.build(req, pool, chosen_options=chosen, rainy=rainy)
        self.assert_valid(result, req, pool, rainy)
        self.assertEqual(result.coverage["program_days"], 3)
        self.assertFalse(visits(result.itinerary)[0].indoor)

    def test_geometric_spread_never_becomes_a_travel_claim(self):
        req, pool = request(1), candidates(2, separation=.2)
        result = self.build(req, pool, chosen_options=options(pool))
        self.assertEqual((result.status, result.itinerary), ("insufficient_data", None))

    def test_transfer_reserve_is_at_least_thirty_minutes_for_separate_places(self):
        req, pool = request(1), candidates(2, separation=.02)
        result = self.build(req, pool, chosen_options=options(pool))
        self.assert_valid(result, req, pool)
        first, second = visits(result.itinerary)
        self.assertGreaterEqual(start(second) - start(first) - first.duration_min, 30)
        self.assertIn(engine.TRANSFER_RESERVE_NOTE, result.itinerary.tips)

    def test_colocated_visits_need_no_fabricated_transfer(self):
        req, pool = request(1), candidates(2, separation=0)
        result = self.build(req, pool, chosen_options=options(pool))
        self.assert_valid(result, req, pool)
        first, second = visits(result.itinerary)
        self.assertEqual(start(second), start(first) + first.duration_min)

    def test_source_windows_reorder_visits_and_never_count_pauses_as_transfer(self):
        req, pool = request(1), candidates(2, separation=.02)
        chosen = options(pool)
        def window(left, right):
            return ({"date": req.start_date, "start": left, "end": right, "basis": "source"},)
        chosen[0] = replace(chosen[0], opening_windows=window("14:00", "15:30"))
        chosen[1] = replace(chosen[1], opening_windows=window("09:00", "10:30"))
        result = self.build(req, pool, chosen_options=chosen)
        self.assert_valid(result, req, pool)
        actual = visits(result.itinerary)
        self.assertEqual([item.place_query for item in actual], [pool[1].place_query, pool[0].place_query])
        self.assertEqual([item.time for item in actual], ["09:00", "14:00"])
        first, second = actual
        occupied = sum(max(0, min(start(item) + item.duration_min, start(second))
                           - max(start(item), start(first) + first.duration_min))
                       for item in result.itinerary.days[0].items if item.kind in {"meal", "rest"})
        self.assertGreaterEqual(start(second) - start(first) - first.duration_min - occupied, 30)

    def test_no_time_for_transfer_causes_failure_not_compression(self):
        req, pool = request(1), candidates(2, separation=.02)
        chosen = options(pool)
        chosen[0] = replace(chosen[0], opening_windows=({"date": req.start_date, "start": "09:00", "end": "10:30", "basis": "source"},))
        chosen[1] = replace(chosen[1], opening_windows=({"date": req.start_date, "start": "10:30", "end": "12:00", "basis": "source"},))
        result = self.build(req, pool, chosen_options=chosen)
        self.assertEqual((result.status, result.itinerary), ("insufficient_data", None))

    def test_source_windows_with_non_grid_minutes_do_not_produce_false_infeasibility(self):
        req, pool = request(1, start_time="09:07", end_time="14:07"), candidates(2, separation=0)
        chosen = options(pool)
        chosen[0] = replace(chosen[0], opening_windows=(
            {"date": req.start_date, "start": "09:07", "end": "11:00", "basis": "source"},))
        chosen[1] = replace(chosen[1], opening_windows=(
            {"date": req.start_date, "start": "12:00", "end": "14:07", "basis": "source"},))
        prepared = [engine._checked_option(value, option, [req.start_date]) for value, option in zip(pool, chosen)]
        counter = engine._Budget(20000)
        template = engine._template((0, 1), prepared, (((547, 660),), ((720, 847),)),
                                    547, 847, {(0, 1): 0}, counter, 15000)
        self.assertIsNotNone(template)
        self.assertEqual(template.meal_start, 637)
        self.assertEqual(template.starts, (547, 757))
        self.assertFalse(counter.limited)

    def test_closed_dates_remain_free_and_original_calendar_is_preserved(self):
        req, pool = request(3, children_ages=[5]), candidates(2)
        chosen = options(pool)
        chosen[0] = replace(chosen[0], allowed_dates=("2099-10-29",))
        chosen[1] = replace(chosen[1], allowed_dates=())
        result = self.build(req, pool, chosen_options=chosen)
        self.assert_valid(result, req, pool)
        self.assertEqual([day.day_mode for day in result.itinerary.days], ["free", "program", "free"])
        self.assertEqual(result.coverage["exclusions"][0]["reason"], "no_eligible_dates")

    def test_known_windows_do_not_allow_unlisted_dates(self):
        req, pool = request(3, children_ages=[5]), candidates(1)
        chosen = [replace(options(pool)[0], opening_windows=(
            {"date": "2099-10-30", "start": "10:00", "end": "12:00", "basis": "source"},))]
        result = self.build(req, pool, chosen_options=chosen)
        self.assert_valid(result, req, pool)
        self.assertEqual(result.coverage["program_day_numbers"], [3])

    def test_known_age_and_child_restrictions_are_not_ignored(self):
        req, pool = request(3, children_ages=[5]), candidates(3)
        chosen = options(pool)
        chosen[0] = replace(chosen[0], min_child_age=10)
        chosen[1] = replace(chosen[1], children_allowed=False)
        result = self.build(req, pool, chosen_options=chosen)
        self.assert_valid(result, req, pool)
        self.assertEqual(result.coverage["program_days"], 1)
        self.assertEqual([entry["reason"] for entry in result.coverage["exclusions"]], ["age_restriction"] * 2)

    def test_eighteen_plus_age_rule_excludes_family_but_not_adults(self):
        pool = candidates(2)
        chosen = [replace(value, min_child_age=18) for value in options(pool)]
        result = self.build(request(1, children_ages=[17]), pool, chosen_options=chosen)
        self.assertEqual((result.status, result.itinerary), ("insufficient_data", None))
        adult = request(1)
        self.assert_valid(self.build(adult, pool, chosen_options=chosen), adult, pool)

    def test_curated_block_is_applied_before_group_search(self):
        req, pool = request(1), candidates(4)
        result = self.build(req, pool, chosen_options=options(pool),
                            blocked_places=[{"aliases": [pool[0].title], "reason": "Synthetic closure"}])
        self.assert_valid(result, req, pool)
        self.assertNotIn(pool[0].place_query, [item.place_query for item in visits(result.itinerary)])
        self.assertIn({"candidate_id": pool[0].candidate_id, "reason": "blocked_place"}, result.coverage["exclusions"])

    def test_conflicting_known_indoor_and_outdoor_map_is_excluded(self):
        req, pool = request(1, children_ages=[5]), candidates(2, kind="park")
        chosen = options(pool, indoor=True)
        result = self.build(req, pool, chosen_options=chosen)
        self.assertEqual((result.status, result.itinerary), ("insufficient_data", None))
        self.assertTrue(all(value["reason"] == "conflicting_indoor_evidence" for value in result.coverage["exclusions"]))

    def test_two_short_published_visits_can_make_one_family_day_not_two(self):
        req, pool = request(2, children_ages=[5]), candidates(2)
        chosen = options(pool, duration=45)
        result = self.build(req, pool, chosen_options=chosen)
        self.assert_valid(result, req, pool)
        self.assertEqual(result.coverage["program_days"], 1)
        self.assertEqual([item.duration_min for item in visits(result.itinerary)], [45, 45])

    def test_option_over_cap_or_negative_nan_cost_is_invalid_not_repaired(self):
        pool = candidates(2)
        for change in ({"duration_min": 121}, {"estimated_cost": -1}, {"estimated_cost": float("nan")},
                       {"estimated_cost": True}, {"indoor": "true"}):
            with self.subTest(change=change):
                chosen = options(pool)
                chosen[0] = replace(chosen[0], **change)
                result = self.build(request(1), pool, chosen_options=chosen)
                self.assertEqual((result.status, result.itinerary), ("invalid_request", None))

    def test_bad_request_and_rain_dates_fail_closed(self):
        for req in (request().model_copy(update={"budget": float("nan")}),
                    request().model_copy(update={"adults": True}), request(start_time="25:00"),
                    request(currency="ZZZ"), request(destination_kind="region")):
            self.assertEqual(self.build(req).status, "invalid_request")
        self.assertEqual(self.build(rainy=["2099-11-11"]).status, "invalid_request")
        self.assertEqual(self.build(max_states=True).status, "invalid_request")

    def test_duplicate_or_forged_pool_cannot_supply_map_authority(self):
        pool = candidates(2)
        for bad in ([pool[0], deepcopy(pool[0])], [replace(pool[0], candidate_id="osm-node-999")]):
            self.assertEqual(self.build(pool=bad).status, "invalid_request")

    def test_missing_options_are_excluded_not_replaced_by_generic_visits(self):
        result = self.build(chosen_options=[])
        self.assertEqual((result.status, result.itinerary), ("insufficient_data", None))
        self.assertTrue(all(value["reason"] == "visit_conditions_unavailable" for value in result.coverage["exclusions"]))

    def test_limit_is_reported_and_never_exceeded(self):
        result = self.build(pool=candidates(18), max_states=1)
        self.assertEqual((result.status, result.itinerary), ("search_limit", None))
        self.assertFalse(result.coverage["search_complete"])
        self.assertLessEqual(result.coverage["states_explored"], 1)

    def test_found_partial_under_search_limit_still_passes_all_guards(self):
        req, pool = request(), candidates(18)
        result = self.build(req, pool, chosen_options=options(pool), max_states=90)
        self.assert_valid(result, req, pool)
        self.assertLessEqual(result.coverage["states_explored"], 90)
        self.assertFalse(result.coverage["search_complete"])
        self.assertTrue(any("не доказательство" in value for value in result.itinerary.tips))

    def test_order_independent_and_no_input_mutation(self):
        req, pool = request(), candidates(6)
        chosen = options(pool)
        before = deepcopy((req.model_dump(), pool, chosen))
        first = self.build(req, pool, chosen_options=chosen)
        second = self.build(req, list(reversed(pool)), chosen_options=list(reversed(chosen)))
        self.assert_valid(first, req, pool)
        self.assertEqual(first.itinerary.model_dump(), second.itinerary.model_dump())
        self.assertEqual(first.coverage, second.coverage)
        self.assertEqual((req.model_dump(), pool, chosen), before)

    def test_explicit_candidate_priority_selects_preferred_feasible_ids(self):
        req, pool = request(1), candidates(4)
        priority = [pool[3].candidate_id, pool[2].candidate_id]
        result = self.build(req, pool, candidate_priority=priority)
        self.assert_valid(result, req, pool)
        self.assertEqual(result.coverage["selected_ids_by_day"]["1"][:2], priority)
        self.assertEqual(result.coverage["selected_unique_visits"], 4)

    def test_six_place_mixed_pool_honestly_covers_two_of_five_requested_days(self):
        req = request(5, start_time="10:00", end_time="18:00", budget=70000,
                      travel_cost=20000, lodging_cost=25000)
        pool = candidates(6)
        kinds = ("museum", "museum", "theatre", "place_of_worship", "theatre", "park")
        for candidate, kind in zip(pool, kinds):
            candidate.record["type"] = kind
        result = self.build(req, pool)
        self.assert_valid(result, req, pool)
        self.assertEqual((result.coverage["program_days"], result.coverage["free_days"]), (2, 3))
        self.assertEqual(result.coverage["daily_program_target_minutes"], 180)
        self.assertEqual(result.coverage["meal_reserve_total"], 7000)
        self.assertEqual(result.coverage["selected_unique_visits"], 6)

    def test_rich_eighteen_place_pool_covers_five_long_adult_days(self):
        req, pool = request(5, start_time="08:00", end_time="21:00"), candidates(18)
        result = self.build(req, pool)
        self.assert_valid(result, req, pool)
        self.assertEqual(result.coverage["program_days"], 5)
        self.assertEqual(result.coverage["selected_unique_visits"], 18)
        self.assertLessEqual(result.coverage["states_explored"], engine.MAX_SEARCH_STATES)

    def test_conflicting_early_windows_do_not_starve_sixteen_other_visits(self):
        req, pool = request(5, start_time="08:00", end_time="21:00"), candidates(18)
        chosen = options(pool, cost=0)
        windows = tuple({"date": (date.fromisoformat(req.start_date) + timedelta(days=index)).isoformat(),
                         "start": "09:00", "end": "10:30", "basis": "source"} for index in range(5))
        for index in (0, 1):
            chosen[index] = replace(chosen[index], opening_windows=windows)
        first = self.build(req, pool, chosen_options=chosen)
        reversed_result = self.build(req, list(reversed(pool)), chosen_options=list(reversed(chosen)))
        self.assert_valid(first, req, pool)
        self.assertEqual(first.coverage["program_days"], 5)
        self.assertEqual(first.coverage, reversed_result.coverage)
        self.assertLessEqual(first.coverage["states_explored"], engine.MAX_SEARCH_STATES)

    def test_three_museums_for_seven_days_are_partial_and_third_visit_is_not_discarded(self):
        req = request(7, start_time="11:00", end_time="20:00", budget=100000,
                      travel_cost=10000, lodging_cost=35000)
        pool = candidates(3)
        result = self.build(req, pool)
        self.assert_valid(result, req, pool)
        self.assertEqual(result.coverage["quality_status"], "partial")
        self.assertEqual((result.coverage["program_days"], result.coverage["free_days"]), (1, 6))
        self.assertEqual(result.coverage["selected_unique_visits"], 3)
        self.assertEqual(result.coverage["enrichment_visits"], 1)
        self.assertEqual([item.duration_min for item in visits(result.itinerary)], [90, 90, 90])
        self.assertEqual(result.coverage["quality_reasons"], ["incomplete_calendar", "insufficient_confirmed_places"])
        self.assertNotIn("1 из 7", result.itinerary.summary)
        self.assertEqual(result.itinerary.program_report.selected_unique_visits, 3)

    def test_rich_pool_gives_complete_seven_day_program(self):
        req, pool = request(7, start_time="10:00", end_time="18:00"), candidates(18)
        result = self.build(req, pool)
        self.assert_valid(result, req, pool)
        self.assertEqual(result.coverage["quality_status"], "complete")
        self.assertEqual(result.coverage["quality_reasons"], [])
        self.assertEqual((result.coverage["program_days"], result.coverage["free_days"]), (7, 0))
        self.assertEqual(result.coverage["selected_unique_visits"], 18)

    def test_seventeen_independent_visits_reach_six_day_bound_without_searching_for_impossible_seventh(self):
        req, pool = request(7, start_time="11:00", end_time="20:00"), candidates(17)
        for index, item in enumerate(pool[8:], 1):
            item.title = f"Городской театр номер {index}"
            item.place_query = f"{item.title}, Тестгород, Россия"
            item.record.update(name=item.place_query, category="amenity", type="theatre")
        short_first = [item.candidate_id for item in pool[8:]] + [item.candidate_id for item in pool[:8]]
        for priority in (None, short_first):
            with self.subTest(priority=bool(priority)):
                result = self.build(req, pool, candidate_priority=priority)
                self.assert_valid(result, req, pool)
                self.assertEqual(sum(item.duration_min for item in visits(result.itinerary)), 1125)
                self.assertEqual(result.coverage["daily_program_target_minutes"], 180)
                self.assertEqual((result.coverage["program_days"], result.coverage["free_days"]), (6, 1))
                self.assertEqual(result.coverage["quality_status"], "partial")
                self.assertEqual(result.coverage["selected_unique_visits"], 17)
                self.assertTrue(result.coverage["search_complete"])
                self.assertLess(result.coverage["states_explored"], 20000)
                if priority:
                    self.assertEqual(result.coverage["selected_ids_by_day"]["1"][:4], short_first[:4])

    def test_partial_upper_bound_uses_family_target_and_does_not_override_rain_or_exact_dates(self):
        req, pool = request(7, start_time="11:00", end_time="20:00", children_ages=[5]), candidates(6)
        chosen = options(pool, cost=0)
        dates = [(date.fromisoformat(req.start_date) + timedelta(days=index)).isoformat() for index in range(1, 7)]
        for index, day_date in enumerate(dates):
            chosen[index] = replace(chosen[index], indoor=None if index == 0 else True,
                                    allowed_dates=(day_date,), opening_windows=(
                                        {"date": day_date, "start": "11:00", "end": "12:30", "basis": "source"},))
        result = self.build(req, pool, chosen_options=chosen, rainy=dates[1:])
        self.assert_valid(result, req, pool, dates[1:])
        self.assertEqual(result.coverage["daily_program_target_minutes"], 90)
        self.assertEqual(result.coverage["program_day_numbers"], [2, 3, 4, 5, 6, 7])
        self.assertEqual(result.coverage["quality_status"], "partial")
        self.assertEqual(result.itinerary.days[0].day_mode, "free")
        self.assertFalse(next(item for item in result.itinerary.days[1].items if item.kind == "attraction").indoor)
        self.assertLessEqual(result.coverage["states_explored"], 20000)

    def test_twenty_four_pool_covers_seven_short_and_long_days_with_original_budget(self):
        for start_time, end_time, target in (("10:00", "18:00", 180), ("08:00", "21:00", 240)):
            with self.subTest(target=target):
                req = request(7, start_time=start_time, end_time=end_time)
                pool = candidates(24)
                result = self.build(req, pool)
                self.assert_valid(result, req, pool)
                self.assertEqual(result.coverage["quality_status"], "complete")
                self.assertEqual(result.coverage["program_days"], 7)
                self.assertEqual(result.coverage["daily_program_target_minutes"], target)
                self.assertLessEqual(result.coverage["states_explored"], 20000)
        self.assertEqual(self.build(pool=candidates(25)).status, "invalid_request")

    def test_spanning_tree_bound_avoids_impossible_transfers_without_search(self):
        req, pool = request(1), candidates(4)
        prepared = [engine._checked_option(value, option, [req.start_date])
                    for value, option in zip(pool, options(pool))]
        distances = {(left, right): 1 for left in range(4) for right in range(left + 1, 4)}
        counter = engine._Budget(20000)
        result = engine._template((0, 1, 2, 3), prepared, (((600, 1080),),) * 4,
                                  600, 1080, distances, counter, 15000)
        self.assertIsNone(result)
        self.assertEqual(counter.used, 0)

    def test_legacy_twenty_four_pool_reserves_states_for_calendar(self):
        from city_coverage import build_city_program_witness
        result = build_city_program_witness(request(7, start_time="08:00", end_time="21:00"), candidates(24))
        self.assertEqual(len(result.program_day_numbers), 7)
        self.assertLessEqual(result.states_explored, 20000)

    def test_partial_rain_calendar_reports_missing_indoor_evidence_not_missing_city(self):
        req, pool = request(7, start_time="10:00", end_time="18:00"), candidates(18)
        rainy = [(date.fromisoformat(req.start_date) + timedelta(days=index)).isoformat() for index in range(6)]
        result = self.build(req, pool, rainy=rainy)
        self.assert_valid(result, req, pool, rainy)
        self.assertEqual(result.coverage["quality_status"], "partial")
        self.assertIn("indoor_evidence_missing", result.coverage["quality_reasons"])
        self.assertNotIn("insufficient_confirmed_places", result.coverage["quality_reasons"])
        self.assertEqual(result.coverage["program_days"], 1)

    def test_enrichment_does_not_override_budget_or_source_windows(self):
        req, pool = request(1, start_time="09:00", end_time="19:00", budget=3500), candidates(3)
        chosen = options(pool, cost=1000)
        result = self.build(req, pool, chosen_options=chosen)
        self.assert_valid(result, req, pool)
        self.assertEqual(result.coverage["selected_unique_visits"], 2)
        chosen = options(pool, cost=0)
        chosen[-1] = replace(chosen[-1], opening_windows=(
            {"date": req.start_date, "start": "09:00", "end": "09:30", "basis": "source"},))
        result = self.build(req, pool, chosen_options=chosen)
        self.assert_valid(result, req, pool)
        self.assertEqual(result.coverage["selected_unique_visits"], 2)

    def test_no_program_is_explicitly_insufficient_quality(self):
        result = self.build(pool=candidates(1))
        self.assertIsNone(result.itinerary)
        self.assertEqual(result.coverage["quality_status"], "insufficient_data")
        self.assertIn("no_feasible_program_day", result.coverage["quality_reasons"])

    def test_group_budget_is_bounded_and_reports_unfinished_search(self):
        req, pool = request(1, start_time="09:07", end_time="14:07"), candidates(2, separation=0)
        chosen = options(pool)
        prepared = [engine._checked_option(value, option, [req.start_date]) for value, option in zip(pool, chosen)]
        counter = engine._Budget(engine.MAX_SEARCH_STATES)
        # Two visits cannot both use the same exclusive 90-minute window;
        # this impossibility is rejected before any costly lunch search.
        self.assertIsNone(engine._template((0, 1), prepared, (((547, 637),), ((547, 637),)),
                                           547, 847, {(0, 1): 0}, counter, 15000, attempt_limit=3))
        self.assertEqual(counter.used, 0)
        with patch.object(engine, "_place_visit", return_value=None):
            self.assertIsNone(engine._template((0, 1), prepared, (((547, 847),), ((547, 847),)),
                                               547, 847, {(0, 1): 0}, counter, 15000, attempt_limit=3))
        self.assertEqual((counter.used, counter.group_limits, counter.limited), (3, 1, True))

    @staticmethod
    def shopping_pool():
        pool = candidates(4)
        mall = pool[-1]
        mall.title = "Торговый центр Северный"
        mall.place_query = f"{mall.title}, Тестгород, Россия"
        mall.record.update(name=mall.place_query, category="shop", type="mall")
        return pool

    def test_early_pair_bound_does_not_require_nonadjacent_transfer(self):
        req, pool = request(1), candidates(3)
        chosen = options(pool)
        chosen[2] = replace(chosen[2], duration_min=1)
        prepared = [engine._checked_option(value, option, [req.start_date])
                    for value, option in zip(pool, chosen)]
        counter = engine._Budget(engine.MAX_SEARCH_STATES)
        template = engine._template((0, 1, 2), prepared,
            (((540, 630),), ((631, 721),), ((630, 631),)), 540, 900,
            {(0, 1): .04, (0, 2): .02, (1, 2): .02}, counter, 15000,
            attempt_limit=engine.MAX_GROUP_SEARCH_STATES)
        self.assertIsNotNone(template)
        self.assertEqual(template.order, (0, 2, 1))

    def test_requested_mall_survives_minimal_subset_pruning_and_priority(self):
        req, pool = request(1), self.shopping_pool()
        mall_id = pool[-1].candidate_id
        chosen = options(pool, cost=0)
        chosen[-1] = replace(chosen[-1], duration_min=45)
        result = self.build(req, pool, candidate_priority=[item.candidate_id for item in pool],
                            required_any_of_ids=[mall_id], chosen_options=chosen)
        self.assert_valid(result, req, pool)
        self.assertTrue(result.coverage["requested_category_satisfied"])
        self.assertIn(mall_id, result.coverage["selected_ids_by_day"]["1"])
        self.assertEqual(result.coverage["program_days"], 1)

    def test_requested_mall_cannot_override_budget_or_closed_dates(self):
        req, pool = request(1, budget=3000), self.shopping_pool()
        for changes in ({"estimated_cost": 5000}, {"allowed_dates": ()}):
            chosen = options(pool, cost=0)
            chosen[-1] = replace(chosen[-1], duration_min=45, **changes)
            result = self.build(req, pool, chosen_options=chosen,
                                required_any_of_ids=[pool[-1].candidate_id])
            self.assert_valid(result, req, pool)
            self.assertFalse(result.coverage["requested_category_satisfied"])
            self.assertIn("requested_shopping_not_scheduled", result.reasons)
            self.assertIn("requested_interest_missing", result.coverage["quality_reasons"])
            self.assertFalse(any("торговый центр не вошёл" in tip for tip in result.itinerary.tips))
            self.assertEqual(result.coverage["program_days"], 1)

    def test_requested_categories_survive_minimal_groups_and_child_visit_limit(self):
        req, pool = request(1, children_ages=[2]), candidates(4)
        pool[-1].record.update(category="tourism", type="zoo")
        result = self.build(req, pool, required_category_groups={
            "museums": [item.candidate_id for item in pool[:3]],
            "zoos": [pool[-1].candidate_id]})
        self.assert_valid(result, req, pool)
        identifiers = result.coverage["selected_ids_by_day"]["1"]
        self.assertEqual(len(identifiers), 2)
        self.assertIn(pool[-1].candidate_id, identifiers)
        self.assertTrue(set(identifiers) & {item.candidate_id for item in pool[:3]})
        self.assertLessEqual(result.coverage["states_explored"], 20000)

    def test_twenty_four_mixed_visits_cover_seven_long_days_and_all_requested_categories(self):
        req, pool = request(7, start_time="08:00", end_time="21:00"), candidates(24)
        for index in (18, 19):
            pool[index].record.update(category="tourism", type="zoo")
        for index in (20, 21):
            pool[index].record.update(category="amenity", type="theatre")
        for index in (22, 23):
            pool[index].record.update(category="shop", type="mall")
        groups = {"museums": [item.candidate_id for item in pool[:18]],
                  "zoos": [item.candidate_id for item in pool[18:20]],
                  "theatre": [item.candidate_id for item in pool[20:22]],
                  "shopping_malls": [item.candidate_id for item in pool[22:]]}
        result = self.build(req, pool, required_category_groups=groups,
                            required_any_of_ids=groups["shopping_malls"])
        self.assert_valid(result, req, pool)
        self.assertEqual((result.coverage["quality_status"], result.coverage["program_days"]), ("complete", 7))
        chosen = {identifier for values in result.coverage["selected_ids_by_day"].values() for identifier in values}
        self.assertEqual(len(chosen), 24)
        for identifiers in groups.values():
            self.assertTrue(chosen & set(identifiers))
        # The checked full incumbent reaches the day/category upper bound;
        # do not spend 20k states proving it before retaining spare visits.
        self.assertLess(result.coverage["states_explored"], 1000)

    def test_diverse_incumbent_cannot_override_known_age_dates_or_budget(self):
        req, pool = request(1, children_ages=[8], budget=5000), candidates(4)
        pool[2].record.update(category="tourism", type="zoo")
        pool[3].record.update(category="shop", type="mall")
        groups = {"museums": [item.candidate_id for item in pool[:2]],
                  "zoos": [pool[2].candidate_id], "shopping_malls": [pool[3].candidate_id]}
        for changes in ({"children_allowed": False}, {"allowed_dates": ()}):
            chosen = options(pool, cost=0)
            chosen[2] = replace(chosen[2], duration_min=45, **changes)
            chosen[3] = replace(chosen[3], estimated_cost=10000)
            result = self.build(req, pool, chosen_options=chosen, required_category_groups=groups,
                                required_any_of_ids=groups["shopping_malls"])
            self.assert_valid(result, req, pool)
            selected = set(result.coverage["selected_ids_by_day"]["1"])
            self.assertFalse(selected & {pool[2].candidate_id, pool[3].candidate_id})
            self.assertFalse(result.coverage["requested_category_satisfied"])

    def test_category_groups_cannot_authorize_unknown_or_duplicate_ids(self):
        for groups in ({"zoos": ["invented"]}, {"zoos": ["osm-node-101"] * 2},
                       {"unknown": ["osm-node-101"]}, {"zoos": "osm-node-101"}):
            with self.subTest(groups=groups):
                self.assertEqual(self.build(required_category_groups=groups).status, "invalid_request")

    def test_required_candidate_ids_cannot_invent_or_repeat_a_place(self):
        for identifiers in (["osm-node-99999"], [True], ["osm-node-101"] * 2, "osm-node-101"):
            with self.subTest(identifiers=identifiers):
                self.assertEqual(self.build(required_any_of_ids=identifiers).status, "invalid_request")

    def test_requested_mall_preserves_full_calendar_before_expensive_interest_search(self):
        req, pool = request(5, start_time="10:00", end_time="18:00", budget=17000), candidates(11)
        pool[-1].record.update(category="shop", type="mall")
        chosen = options(pool, cost=1000)
        chosen[-1] = replace(chosen[-1], estimated_cost=8000)
        baseline = self.build(req, pool, chosen_options=chosen)
        result = self.build(req, pool, chosen_options=chosen,
                            required_any_of_ids=[pool[-1].candidate_id])
        self.assert_valid(result, req, pool)
        self.assertEqual(baseline.coverage["program_days"], 5)
        self.assertEqual(result.coverage["program_days"], baseline.coverage["program_days"])
        self.assertFalse(result.coverage["requested_category_satisfied"])
        self.assertLessEqual(result.coverage["states_explored"], engine.MAX_SEARCH_STATES)

    def test_requested_mall_does_not_override_children_or_rain_conditions(self):
        pool = self.shopping_pool()
        for changes in ({"children_allowed": False}, {"min_child_age": 12}):
            req = request(1, children_ages=[8])
            chosen = options(pool, cost=0)
            chosen[-1] = replace(chosen[-1], **changes)
            result = self.build(req, pool, chosen_options=chosen,
                                required_any_of_ids=[pool[-1].candidate_id])
            self.assert_valid(result, req, pool)
            self.assertFalse(result.coverage["requested_category_satisfied"])
            self.assertIn({"candidate_id": pool[-1].candidate_id, "reason": "age_restriction"},
                          result.coverage["exclusions"])
        # Unknown mall shelter is not promoted to indoor evidence. Only the
        # separate museum with an explicitly supplied indoor option shelters
        # this rainy program, and no three-visit group has an indoor majority.
        req = request(1)
        chosen = options(pool, cost=0)
        chosen[0] = replace(chosen[0], indoor=True)
        chosen[-1] = replace(chosen[-1], duration_min=45)
        result = self.build(req, pool, chosen_options=chosen, rainy=[req.start_date],
                            required_any_of_ids=[pool[-1].candidate_id])
        self.assert_valid(result, req, pool, [req.start_date])
        self.assertFalse(result.coverage["requested_category_satisfied"])
        self.assertEqual(result.coverage["program_days"], 1)

    def test_candidate_priority_cannot_authorize_unknown_duplicate_or_unaffordable_ids(self):
        req, pool = request(1, budget=3400), candidates(4)
        for bad in ([pool[0].candidate_id] * 2, ["osm-node-999"], "osm-node-101", [True]):
            self.assertEqual(self.build(req, pool, candidate_priority=bad).status, "invalid_request")
        chosen = options(pool)
        chosen[3] = replace(chosen[3], estimated_cost=3000)
        result = self.build(req, pool, chosen_options=chosen, candidate_priority=[pool[3].candidate_id])
        self.assert_valid(result, req, pool)
        self.assertNotIn(pool[3].candidate_id, result.coverage["selected_ids_by_day"]["1"])

    def test_sources_scope_estimates_and_unknowns_survive_into_disclosure(self):
        req, pool = request(1), candidates(2)
        chosen = options(pool)
        chosen[0] = replace(chosen[0], indoor=True, description="Осмотр только внутренних выставочных залов основной площадки.",
                            sources=({"field": "indoor", "basis": "published", "source_url": "https://example.org/rooms",
                                      "checked_at": "2026-10-22"},), unknown_fields=("price", "children", "minimum_age"))
        result = self.build(req, pool, chosen_options=chosen)
        self.assert_valid(result, req, pool)
        self.assertEqual(visits(result.itinerary)[0].description, chosen[0].description)
        disclosure = " ".join(result.itinerary.tips)
        self.assertIn("https://example.org/rooms", disclosure)
        self.assertIn("2026-10-22", disclosure)
        self.assertIn("плановая оценка, не факт источника", disclosure)
        self.assertNotIn("посещение с детьми", disclosure)

    def test_final_guard_failure_is_not_misreported_as_insufficient_data(self):
        with patch.object(engine, "check_rules", return_value=["Synthetic invariant failure"]):
            result = self.build(request(1), candidates(2))
        self.assertEqual((result.status, result.itinerary), ("validation_error", None))
        self.assertIn("schedule_validation_failed", result.reasons)

    def test_reviewed_nha_sources_have_scope_and_calendar_without_external_lookup(self):
        from test_city_visit_options import FixedDate, ocean, yersin
        req = request(5, destination="Нячанг, Вьетнам", start_date="2026-10-28", end_date="2026-11-01",
                      children_ages=[4, 10], currency="USD", budget=3000)
        pool = [yersin(), ocean()]
        center = {"name": "Нячанг", "lat": 12.25, "lon": 109.19, "country_code": "VN"}
        rainy = ["2026-10-28", "2026-10-29", "2026-10-30", "2026-10-31", "2026-11-01"]
        with patch("city_visit_options.date", FixedDate):
            result = engine.build_deterministic_city_itinerary(req, pool, rainy, resolved_city=center)
            self.assert_valid(result, req, pool, rainy)
        self.assertEqual((result.coverage["program_days"], result.coverage["free_days"]), (1, 4))
        self.assertEqual([item.duration_min for item in visits(result.itinerary)], [90, 30])
        self.assertIn("выставочных залах", visits(result.itinerary)[0].description)
        notes = " ".join(result.itinerary.tips)
        self.assertIn("https://baotanghdh.vn/khu-trung-bay-hdh/", notes)
        self.assertIn("https://baotanghdh.vn/en/contact-us/", notes)
        self.assertIn("06:00–18:00", notes)
        self.assertIn("2026-10-22", notes)
        self.assertIn("90 минут — оценка планирования, не опубликованная длительность", notes)

    def test_reviewed_thirty_minute_yersin_is_never_stretched_into_family_day(self):
        from test_city_visit_options import FixedDate, yersin
        req = request(1, destination="Нячанг, Вьетнам", start_date="2026-10-28", end_date="2026-10-28",
                      children_ages=[5], currency="USD", budget=1000)
        center = {"name": "Нячанг", "lat": 12.25, "lon": 109.19, "country_code": "VN"}
        with patch("city_visit_options.date", FixedDate):
            result = engine.build_deterministic_city_itinerary(req, [yersin()], resolved_city=center)
        self.assertEqual((result.status, result.itinerary), ("insufficient_data", None))


if __name__ == "__main__":
    unittest.main()
