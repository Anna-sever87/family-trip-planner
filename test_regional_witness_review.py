"""Independent witness review: real validators, synthetic map geometry only."""
from contextlib import ExitStack
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

from grounded_schedule import _pool
from grounded_response import build_grounded_response_schema
import places
import planner
import regional_schedule_witness as witness_module
from rules import check_program_depth, check_rules
import test_regional_map_only_schedule as pipeline
from test_regional_schedule_witness import fixture, witness_route


def plain_fixture():
    data = list(fixture())
    data[0] = data[0].model_copy(update={"wishes": "", "trip_goal": ""})
    return tuple(data)


class RegionalWitnessIndependentTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack(); self.addCleanup(self.stack.close)
        for name in ("httpx.get", "httpx.post", "socket.socket.connect", "sqlite3.connect", "places._save_cache"):
            self.stack.enter_context(patch(name, side_effect=AssertionError("No external IO")))

    def full_result(self, data, route):
        with patch.object(pipeline, "fixture", return_value=data):
            return pipeline.RegionalMapOnlyScheduleTests().run_route(route)[0]

    def test_week_reserves_76_minute_geometric_base_change_and_passes_all_validators(self):
        data = plain_fixture(); req, _, grounding, records, _ = data
        base = grounding.base_facts[1]
        delta = .49
        base["lat"] += delta
        records[base["base_query"]]["lat"] += delta
        for hint in grounding.confirmed_hints:
            if hint["base_key"] == base["base_key"]:
                hint["lat"] += delta
                records[hint["place_query"]]["lat"] += delta
        # Deliberately synthetic geometry reproduces the 76-minute lower
        # bound. These are not claimed coordinates or road times in Karelia.
        bases, _ = _pool(grounding)
        lower = math.ceil(places.distance_km(bases[0]["lat"], bases[0]["lon"], bases[1]["lat"], bases[1]["lon"]) + 10)
        self.assertGreaterEqual(lower, 76)
        self.assertLessEqual(lower, 78)
        before = deepcopy(data)
        witness, route = witness_route(deepcopy(data))
        self.assertIsNotNone(witness)
        self.assertTrue(any(day["transfer_reserve_min"] >= lower for day in witness["days"]))
        self.assertEqual(check_program_depth(route, req), [])
        self.assertEqual(check_rules(route, req, []), [])
        result = self.full_result(data, route)
        self.assertEqual(result.problems, [])
        self.assertEqual(len(result.itinerary.days), 7)
        self.assertEqual(len({item.place_query for day in result.itinerary.days for item in day.items
                              if item.kind == "attraction"}), 14)
        self.assertTrue(all(item.duration_min == 90 for day in result.itinerary.days for item in day.items
                            if item.kind == "attraction"))
        self.assertFalse(witness["road_access_verified"])
        self.assertFalse(witness["hours_and_prices_verified"])
        self.assertEqual(data, before)

    def test_lunch_and_rest_cannot_consume_intervisit_travel_reserve(self):
        req, scope, grounding, records, route = plain_fixture()
        req = req.model_copy(update={"end_date": req.start_date, "end_time": "19:00",
                                     "interest_categories": ["museums", "zoos"]})
        grounding.base_facts = grounding.base_facts[:1]
        grounding.confirmed_hints = grounding.confirmed_hints[:2]
        first, second = grounding.confirmed_hints
        first.update(map_poi_type="zoo", lat=51., lon=101.)
        second.update(lat=51.62, lon=101.)
        for hint in grounding.confirmed_hints:
            records[hint["place_query"]].update(lat=hint["lat"], lon=hint["lon"], type=hint["map_poi_type"])
        route.days = route.days[:1]
        data = (req, scope, grounding, records, route)
        witness, route = witness_route(deepcopy(data))
        self.assertIsNotNone(witness)
        result = self.full_result(data, route)
        self.assertEqual(result.problems, [])
        self.assertEqual(sorted(item.duration_min for item in result.itinerary.days[0].items
                                if item.kind == "attraction"), [90, 120])

    def test_custom_exclusions_cannot_be_replaced_with_forced_museum_days(self):
        request, _, grounding, _, _ = plain_fixture()
        for field, text in (("wishes", "Без музеев, только парки"),
                            ("wishes", "Не посещать зоопарки и мемориалы"),
                            ("trip_goal", "Только природные прогулки, никаких музеев"),
                            ("stay_area", "Оставаться в одной базе без смены жилья")):
            with self.subTest(field=field, text=text):
                modified = request.model_copy(update={field: text})
                witness = witness_module.build_witness(modified, *_pool(grounding), 5)
                self.assertIsNotNone(witness)
                self.assertIs(witness["binding"], False)
                schema = build_grounded_response_schema(planner.build_itinerary_json_schema(modified), modified)
                self.assertEqual(witness_module.constrain_witness_schema(schema, witness), schema)
                _, route = witness_route(plain_fixture())
                route.days[0].items[0].time = "08:01"
                before = route.model_dump()
                self.assertEqual(witness_module.check_witness(route, witness), [])
                self.assertEqual(route.model_dump(), before)

    def test_source_catalogue_success_and_failure_both_bypass_map_type_witness(self):
        from test_source_planning import SourcePlanningTests
        probe = SourcePlanningTests()
        probe.setUp(); self.addCleanup(probe.doCleanups)
        with patch.object(witness_module, "build_witness", side_effect=AssertionError("Source path must stay authoritative")) as build, \
                patch.object(planner, "check_witness", side_effect=AssertionError("Source path needs no map-type contract")) as checker:
            probe.test_planner_direct_source_success_or_failure_never_requests_ai_schedule()
        build.assert_not_called(); checker.assert_not_called()

    def test_witness_does_not_make_water_or_budget_violations_valid(self):
        data = plain_fixture(); witness, original = witness_route(deepcopy(data))
        for mode in ("water", "budget"):
            route = original.model_copy(deep=True)
            if mode == "water":
                route.days[0].transfer_note = "Если паром доступен, переправа к базе."
            else:
                for day in route.days:
                    for item in day.items:
                        if item.kind == "attraction": item.cost = 6000
            self.assertEqual(witness_module.check_witness(route, witness), [])
            result = self.full_result(data, route)
            self.assertTrue(result.problems)
            fragment = "водного/ледового" if mode == "water" else "бюджет"
            self.assertTrue(any(fragment in problem.lower() for problem in result.problems), result.problems)

    def test_binding_matches_each_place_to_its_slot_without_rewriting_model_text(self):
        witness, route = witness_route(plain_fixture())
        visits = [item for item in route.days[0].items if item.kind == "attraction"]
        visits[0].time, visits[1].time = visits[1].time, visits[0].time
        visits[0].description = "Осмотр экспозиции в течение 60 минут."
        visits[0].duration_min = 60
        before = route.model_dump()
        problems = witness_module.check_witness(route, witness)
        self.assertGreaterEqual(len(problems), 2)
        self.assertEqual(route.model_dump(), before)


if __name__ == "__main__": unittest.main()
