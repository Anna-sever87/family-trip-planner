"""The opt-in city pipeline has one pool, a fixed budget and no map retry."""

from contextlib import ExitStack
from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import city_planner
import planner
from city_candidates import CityCandidate, CityCandidateResult
from schemas import TripRequest
from weather import CityResolutionError
from yandex_ai import YandexAIError


def request():
    return TripRequest(destination="Тестгород, Россия", start_date="2099-10-28", end_date="2099-10-28",
                       adults=2, children_ages=[], budget=30000, currency="RUB",
                       pace="средний", start_time="09:00", end_time="19:00")


def candidate():
    title = "Городской исторический музей"
    return CityCandidate(candidate_id="osm-node-101", title=title,
                         place_query=f"{title}, Тестгород, Россия", area="Тестгород",
                         record=dict(lat=55.0, lon=83.0, name=f"{title}, Тестгород, Россия",
                                     provider="osm", osm_type="node", osm_id=101, type="museum",
                                     address={"city": "Тестгород", "country": "Россия", "country_code": "ru"}))


def default_candidates():
    first, second = candidate(), candidate()
    second.candidate_id = "osm-node-102"
    second.title = "Городской научный музей"
    second.place_query = f"{second.title}, Тестгород, Россия"
    second.record.update(osm_id=102, name=second.place_query, lon=83.0001)
    return [first, second]


def payload(*, duration=90, indoor=True, second_visit=True):
    def pause(kind, time, cost):
        return dict(kind=kind, time=time, title="Обед" if kind == "meal" else "Отдых",
                    area="Тестгород", place_query="", description="Перерыв",
                    duration_min=60, cost=cost, kids_note="", indoor=True)
    visits = [dict(candidate_id="osm-node-101", time="09:00", duration_min=duration, cost=1000,
                   description="Осмотр исторической экспозиции", kids_note="", indoor=indoor)]
    if second_visit:
        visits.append({**visits[0], "candidate_id": "osm-node-102", "time": "10:45",
                       "description": "Осмотр научной экспозиции"})
    return dict(title="Тестовая программа", summary="Синтетическая проверка", tips=["Уточните часы работы"],
                day_1=dict(day=1, date="2099-10-28", theme="История города", base_area="", transfer_note="",
                           rainy_alternative="", attractions=visits,
                           meal=pause("meal", "13:00", 1400), rest=pause("rest", "14:00", 0)))


class CityCandidatePlannerTests(unittest.TestCase):
    def run_plan(self, responses, *, center=None, empty_pool=False, weather=None, max_attempts=3,
                 candidates=None, trip_request=None):
        if center is None:
            center = {"lat": 55.0, "lon": 83.0, "name": "Тестгород", "country_code": "RU"}

        def ground(req, resolved, *, completion, rainy_dates=(), allow_partial_program=False):
            self.assertTrue(allow_partial_program)
            completion([{"role": "user", "content": "Synthetic candidates"}], json_schema={"type": "object"})
            return CityCandidateResult(candidates=[] if empty_pool else
                                       (default_candidates() if candidates is None else candidates), problems=[],
                                       usage={"completion_calls": 1, "input_tokens": 11, "output_tokens": 13}, map_requests=0)

        with ExitStack() as stack:
            stack.enter_context(patch("httpx.get", side_effect=AssertionError("No HTTP")))
            stack.enter_context(patch("places.geocode", side_effect=AssertionError("No map lookup")))
            stack.enter_context(patch.object(city_planner, "find_city", return_value=center))
            weather_mock = stack.enter_context(patch.object(city_planner, "get_weather", return_value=weather))
            stack.enter_context(patch.object(city_planner, "ground_city_candidates", side_effect=ground))
            stack.enter_context(patch.object(planner, "curated_blocked_place_terms", return_value=[]))
            completion = stack.enter_context(patch.object(planner, "generate_json", side_effect=[
                SimpleNamespace(text=json.dumps(value, ensure_ascii=False), input_tokens=11, output_tokens=13)
                if not isinstance(value, Exception) else value for value in [{"titles": []}, *responses]
            ]))
            result = planner.generate_itinerary(trip_request or request(), max_attempts=max_attempts, city_candidates_first=True)
        return result, completion, weather_mock

    def test_two_calls_accept_and_reuse_exact_records_without_network(self):
        result, completion, weather_mock = self.run_plan([payload()])
        self.assertEqual(result.problems, [])
        self.assertEqual((completion.call_count, result.attempts, result.total_tokens), (2, 2, 48))
        self.assertFalse(result.itinerary.source_schedule)
        self.assertTrue(result.itinerary.city_schedule)
        self.assertIsNone(result.itinerary.program_report)
        visit = result.itinerary.days[0].items[0]
        self.assertTrue(visit.verified)
        self.assertEqual((visit.title, visit.area, visit.place_query),
                         (candidate().title, candidate().area, candidate().place_query))
        self.assertIn(city_planner.MAP_ONLY_WARNING, result.itinerary.tips)
        visits = [item for item in result.itinerary.days[0].items if item.kind == "attraction"]
        self.assertEqual([item.duration_min for item in visits], [90, 90])
        self.assertEqual([item.place_query for item in visits], [item.place_query for item in default_candidates()])
        self.assertTrue(all(item.verified for item in visits))
        self.assertEqual(weather_mock.call_args.kwargs["resolved_city"]["name"], "Тестгород")

    def test_same_day_pairs_use_map_geometry_and_do_not_mutate_candidates(self):
        # Synthetic park-shaped coordinates, not newly asserted POI facts.
        first, second = default_candidates()
        first.title = "Парк Ривьера"
        second.title = "Олимпийский парк"
        first.record.update(lat=43.5925, lon=39.7187)
        second.record.update(lat=43.4052, lon=39.9558)
        pool = [first, second]
        before = deepcopy(pool)
        self.assertEqual(city_planner._incompatible_same_day_pairs(pool), [
            {"candidate_ids": [first.candidate_id, second.candidate_id], "distance_km": 28.28},
        ])
        self.assertEqual(pool, before)

    def test_same_day_pair_limit_is_strict_before_distance_rounding(self):
        pool = default_candidates()
        for distance, forbidden in ((7.999999, False), (8.0, False), (8.000001, True)):
            with self.subTest(distance=distance), patch.object(city_planner, "distance_km", return_value=distance):
                pairs = city_planner._incompatible_same_day_pairs(pool)
                self.assertEqual(bool(pairs), forbidden)
        pool[1].record.update(lat=pool[0].record["lat"], lon=pool[0].record["lon"])
        self.assertEqual(city_planner._incompatible_same_day_pairs(pool), [])

    def test_same_day_pairs_are_stable_unique_and_bounded_by_pool_size(self):
        pool = []
        for number in range(18):
            value = candidate()
            value.candidate_id = f"osm-node-{100 + number}"
            pool.append(value)
        with patch.object(city_planner, "distance_km", return_value=9.0) as distance:
            pairs = city_planner._incompatible_same_day_pairs(pool)
        self.assertEqual((len(pairs), distance.call_count), (153, 153))
        self.assertEqual(len({tuple(pair["candidate_ids"]) for pair in pairs}), 153)
        self.assertEqual(pairs[0]["candidate_ids"], ["osm-node-100", "osm-node-101"])
        self.assertEqual(pairs[-1]["candidate_ids"], ["osm-node-116", "osm-node-117"])
        self.assertEqual(city_planner._incompatible_same_day_pairs([]), [])
        self.assertEqual(city_planner._incompatible_same_day_pairs(pool[:1]), [])

    def test_actual_ids_reach_initial_and_repair_prompt_but_final_spread_still_rejects(self):
        pool = default_candidates()
        pool[0].candidate_id = "osm-node-415"
        pool[0].record.update(osm_id=415, lat=43.5925, lon=39.7187)
        pool[1].candidate_id = "osm-way-819"
        pool[1].record.update(osm_type="way", osm_id=819, lat=43.4052, lon=39.9558)
        value = payload()
        for visit, entry in zip(value["day_1"]["attractions"], pool):
            visit["candidate_id"] = entry.candidate_id
        # A valid nearby pair lets the scheduling stage run. The model still
        # selects the forbidden distant pair and must fail the final verifier.
        nearby = deepcopy(pool[0])
        nearby.candidate_id = "osm-node-416"
        nearby.title = "Музей науки и техники"
        nearby.place_query = f"{nearby.title}, Тестгород, Россия"
        nearby.record.update(osm_id=416, name=nearby.place_query, lon=39.719)
        pool.append(nearby)
        before = deepcopy((pool, value))
        center = {"lat": 43.5925, "lon": 39.7187, "name": "Тестгород", "country_code": "RU"}
        result, calls, _ = self.run_plan([value, value], candidates=pool, center=center)
        self.assertEqual(calls.call_count, 3)
        self.assertTrue(any("далеко друг от друга" in problem for problem in result.problems))
        for call in calls.call_args_list[1:]:
            prompt = call.args[0][1]["content"]
            data = prompt.split("CITY_INCOMPATIBLE_SAME_DAY_PAIRS (серверный расчёт):\n", 1)[1].split("\n\n", 1)[0]
            self.assertIn({"candidate_ids": ["osm-node-415", "osm-way-819"], "distance_km": 28.28},
                          json.loads(data))
            self.assertIn("CITY_FEASIBLE_GROUPS", prompt)
            self.assertIn("нельзя включать вместе в один день", prompt)
            self.assertIn("не подтверждение дороги", prompt)
        self.assertEqual((pool, value), before)

    def test_bad_depth_repair_gets_only_one_remaining_call(self):
        result, completion, _ = self.run_plan([payload(duration=60), payload()])
        self.assertEqual(result.problems, [])
        self.assertEqual((completion.call_count, result.attempts, result.total_tokens), (3, 3, 72))
        self.assertIn("недостаточно содержательная", completion.call_args.args[0][-1]["content"])

    def test_planning_cap_runs_after_decode_without_shrinking_the_visit(self):
        for minutes, rejected in ((120, False), (121, True)):
            value = payload()
            value["day_1"]["attractions"][0]["duration_min"] = minutes
            value["day_1"]["attractions"][1]["time"] = "11:15"
            with self.subTest(minutes=minutes):
                result, calls, _ = self.run_plan([value], max_attempts=2)
                self.assertEqual(bool(result.problems), rejected)
                self.assertEqual(calls.call_count, 2)
                first = result.itinerary.days[0].items[0]
                self.assertEqual(first.duration_min, minutes)
                if rejected:
                    self.assertTrue(any("120" in problem and "планирован" in problem for problem in result.problems))

    def two_candidates_and_visits(self, *, offset=0.0001, second_time="10:45"):
        first, second = default_candidates()
        second.record.update(osm_id=102, name=second.place_query, lon=83.0 + offset)
        value = payload(duration=90, second_visit=False)
        next_visit = deepcopy(value["day_1"]["attractions"][0])
        next_visit.update(candidate_id=second.candidate_id, time=second_time)
        value["day_1"]["attractions"].append(next_visit)
        return [first, second], value

    def test_fresh_distinct_ids_nearby_survive_final_duplicate_check(self):
        candidates, value = self.two_candidates_and_visits()
        with patch("places._load_cache", side_effect=AssertionError("No legacy cache resolution")):
            result, completion, _ = self.run_plan([value], candidates=candidates, max_attempts=2)
        self.assertEqual(result.problems, [])
        self.assertEqual(completion.call_count, 2)
        self.assertEqual(sum(item.verified for item in result.itinerary.days[0].items), 2)

    def test_zero_travel_gap_between_separated_visits_fails_real_pipeline(self):
        candidates, value = self.two_candidates_and_visits(offset=0.11, second_time="10:30")
        result, _, _ = self.run_plan([value], candidates=candidates, max_attempts=2)
        self.assertTrue(any("геометрическая" in problem for problem in result.problems))

    def test_family_day_cannot_be_fifteen_minutes_plus_meal_and_rest(self):
        value = payload(duration=15, second_visit=False)
        value["day_1"]["meal"]["cost"] = 1800
        family = request().model_copy(update={"children_ages": [5]})
        result, _, _ = self.run_plan([value], trip_request=family, max_attempts=2)
        self.assertTrue(any("семейная программа" in problem for problem in result.problems))

    def test_artwork_only_capacity_stops_before_requesting_an_unfillable_adult_day(self):
        sculpture = candidate()
        sculpture.title = "Скульптура Вдохновение"
        sculpture.place_query = f"{sculpture.title}, Тестгород, Россия"
        sculpture.record.update(name=sculpture.place_query, type="artwork")
        result, calls, _ = self.run_plan([], candidates=[sculpture], max_attempts=2)
        self.assertEqual(calls.call_count, 1)  # Candidate proposal only; no schedule completion.
        self.assertTrue(result.problems)
        self.assertEqual(result.itinerary.days, [])

    def test_repeated_failure_stops_at_three_calls_without_fallback(self):
        result, completion, _ = self.run_plan([payload(duration=60), payload(duration=60)])
        self.assertEqual(completion.call_count, 3)
        self.assertTrue(any("недостаточно содержательная" in problem for problem in result.problems))

    def test_two_call_budget_cannot_repair(self):
        result, completion, _ = self.run_plan([payload(duration=60)], max_attempts=2)
        self.assertEqual(completion.call_count, 2)
        self.assertTrue(result.problems)

    def test_insufficient_pool_stops_after_candidate_call(self):
        result, completion, _ = self.run_plan([], empty_pool=True)
        self.assertEqual((completion.call_count, result.attempts), (1, 1))
        self.assertIn("Это не означает", result.problems[0])
        self.assertEqual(result.itinerary.days, [])

    def test_unknown_id_and_forged_fields_cannot_bypass_decoder(self):
        for mutate in (lambda p: p["day_1"]["attractions"][0].update(candidate_id="invented"),
                       lambda p: p["day_1"]["attractions"][0].update(verified=True),
                       lambda p: p.update(source_schedule=True)):
            value = payload()
            mutate(value)
            with self.subTest(value=value):
                result, completion, _ = self.run_plan([value], max_attempts=2)
                self.assertTrue(result.problems)
                self.assertEqual(result.itinerary.days, [])
                self.assertEqual(completion.call_count, 2)

    def test_real_rain_not_historical_rain_is_mandatory(self):
        for source, rejected in (("forecast", True), ("fact", True), ("history", False)):
            weather = {"city": "Тестгород", "source": source,
                       "days": [{"date": "2099-10-28", "tmin": 10, "tmax": 20, "rain_mm": 5}]}
            with self.subTest(source=source):
                result, _, _ = self.run_plan([payload(indoor=False)], weather=weather, max_attempts=2)
                self.assertEqual(bool(result.problems), rejected)

    def test_impossible_rain_pool_does_not_spend_schedule_completions(self):
        park = candidate()
        park.title = "Парк Северный"
        park.place_query = f"{park.title}, Тестгород, Россия"
        park.record.update(name=park.place_query, type="park")
        other = deepcopy(park)
        other.candidate_id = "osm-node-102"
        other.title = "Парк Южный"
        other.place_query = f"{other.title}, Тестгород, Россия"
        other.record.update(osm_id=102, name=other.place_query, lon=83.0001)
        weather = {"city": "Тестгород", "source": "forecast",
                   "days": [{"date": "2099-10-28", "tmin": 10, "tmax": 20, "rain_mm": 5}]}
        result, completion, _ = self.run_plan([], candidates=[park, other], weather=weather)
        self.assertEqual((completion.call_count, result.attempts), (1, 1))
        self.assertIn("Расписание не запрашивалось", result.problems[0])
        self.assertEqual(result.itinerary.days, [])

    def test_explicit_outdoor_contradiction_is_rejected(self):
        value = payload()
        value["day_1"]["attractions"][0]["description"] = "Посещение проходит под открытым небом"
        result, _, _ = self.run_plan([value], max_attempts=2)
        self.assertTrue(any("противоречит" in problem for problem in result.problems))

    def test_identity_error_and_missing_center_make_zero_ai_calls(self):
        for outcome in (None, CityResolutionError("Кемер", reason="ambiguous")):
            with self.subTest(outcome=outcome), patch.object(city_planner, "find_city", side_effect=
                outcome if isinstance(outcome, Exception) else None, return_value=None), \
                patch.object(planner, "generate_json") as completion, \
                patch.object(city_planner, "get_weather") as weather, self.assertRaises(CityResolutionError):
                planner.generate_itinerary(request(), city_candidates_first=True)
            completion.assert_not_called()
            weather.assert_not_called()

    def test_provider_failure_does_not_retry_automatically(self):
        result, completion, _ = self.run_plan([YandexAIError("Synthetic provider unavailable")])
        self.assertTrue(result.problems)
        self.assertEqual((completion.call_count, result.attempts, result.total_tokens), (2, 2, 24))

    def test_invalid_budget_and_region_rejected_before_ai(self):
        for budget in (1, 4, True, 2.5):
            with self.subTest(budget=budget), patch.object(planner, "generate_json") as completion, self.assertRaises(ValueError):
                planner.generate_itinerary(request(), max_attempts=budget, city_candidates_first=True)
            completion.assert_not_called()
        region = request().model_copy(update={"destination_kind": "region"})
        with patch.object(planner, "generate_json") as completion, self.assertRaises(ValueError):
            planner.generate_itinerary(region, city_candidates_first=True)
        completion.assert_not_called()


if __name__ == "__main__":
    unittest.main()
