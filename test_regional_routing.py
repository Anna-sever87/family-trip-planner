"""Offline OSRM graph contract: mocked HTTP and temporary cache databases."""

from copy import deepcopy
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

import httpx
import regional_routing as routing
from regional_skeleton import TravelFact, VisitFact, _travel_problem, _visit_access_valid, _visit_problem
from schemas import TripRequest


POINTS = [(50.0, 100.0), (50.01, 100.0)]


def table(points=POINTS, *, seconds=600, metres=2000):
    n = len(points)
    waypoints = [{"distance": 0, "location": [lon, lat]} for lat, lon in points]
    return {"code": "Ok", "durations": [[0 if i == j else seconds for j in range(n)] for i in range(n)],
            "distances": [[0 if i == j else metres for j in range(n)] for i in range(n)],
            "sources": deepcopy(waypoints), "destinations": deepcopy(waypoints)}


def response(data):
    result = Mock()
    result.content = b"mock table"
    result.raise_for_status.return_value = None
    result.json.return_value = deepcopy(data)
    return result


def route(points=POINTS, *, mode="walking", seconds=600, metres=2000):
    return {"code": "Ok", "waypoints": [{"distance": 0, "location": [lon, lat]} for lat, lon in points],
            "routes": [{"duration": seconds, "distance": metres, "legs": [{"duration": seconds, "distance": metres,
                "steps": [{"mode": mode, "duration": seconds, "distance": metres},
                          {"mode": mode, "duration": 0, "distance": 0}]}]}]}


class Clock:
    def __init__(self):
        self.now = 10000.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class RegionalRoutingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="routing-offline-")
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name) / "routing.db"
        self.clock = Clock()
        for owner, name, options in (
            (socket.socket, "connect", {"side_effect": AssertionError("No network")}),
            (routing.httpx, "get", {"side_effect": AssertionError("HTTP must be injected")}),
            (routing.time, "time", {"side_effect": lambda: self.clock.now}),
            (routing.time, "sleep", {"side_effect": self.clock.sleep}),
        ):
            patcher = patch.object(owner, name, **options)
            patcher.start()
            self.addCleanup(patcher.stop)
        environment = patch.dict(routing.os.environ, {
            "ROUTING_ENABLED": "true", "OSRM_ROUTING_BASE_URL": routing.DEFAULT_ROUTER,
        })
        environment.start()
        self.addCleanup(environment.stop)
        query = "Тестовый музей, Тестовая база, Россия"
        self.grounding = SimpleNamespace(
            base_facts=[{"base_key": "base_1", "lat": POINTS[0][0], "lon": POINTS[0][1]}],
            confirmed_hints=[{"place_query": query, "lat": POINTS[1][0], "lon": POINTS[1][1]}],
        )
        self.visits = [VisitFact("visit_1", query, "Синтетическая программа", 120, "published")]

    def build(self, *data, get=None):
        injected = get or Mock(side_effect=[response(value) for value in (data or (table(), table()))])
        edges, warnings = routing.build_travel_graph(self.grounding, self.visits, cache_file=self.cache, get=injected)
        return edges, warnings, injected

    def cache_rows(self):
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            return connection.execute("SELECT key, checked, payload FROM routing_cache").fetchall()

    def test_directed_matrices_and_walking_preference_use_two_bounded_requests(self):
        foot = table()
        foot["durations"][1][0] = 1200
        edges, warnings, get = self.build(foot, table())
        self.assertEqual(warnings, [])
        self.assertEqual(get.call_count, 2)
        self.assertEqual([(edge.from_key, edge.to_key, edge.minutes) for edge in edges],
                         [("base_1", "visit_1", 17), ("visit_1", "base_1", 29)])
        self.assertTrue(all(edge.mode == "пешком" for edge in edges))
        for index, call in enumerate(get.call_args_list):
            self.assertIn("routed-foot" if index == 0 else "routed-car", call.args[0])
            self.assertEqual(call.kwargs["params"], {"annotations": "duration,distance", "generate_hints": "false"})
            self.assertEqual(call.kwargs["headers"], {"User-Agent": routing.USER_AGENT})
            self.assertEqual(call.kwargs["timeout"], 30.0)
        self.assertGreaterEqual(sum(self.clock.sleeps), 1.0)
        self.assertTrue(all("Предварительная" in edge.note and "без паромов" not in edge.note for edge in edges))

    def test_unreachable_pair_is_absent_and_is_not_mirrored_or_filled(self):
        data = table()
        data["durations"][1][0] = data["distances"][1][0] = None
        edges, _, get = self.build(data, data)
        self.assertEqual([(edge.from_key, edge.to_key) for edge in edges], [("base_1", "visit_1")])
        self.assertEqual(get.call_count, 2)

    def test_no_route_never_creates_straight_line_edges_or_retry(self):
        edges, warnings, get = self.build({"code": "NoTable"}, {"code": "NoRoute"})
        self.assertEqual(edges, [])
        self.assertEqual((len(warnings), get.call_count), (2, 2))
        self.assertEqual(self.cache_rows(), [])

    def test_aerial_fallback_flag_is_rejected_even_with_usable_matrices(self):
        data = table()
        data["fallback_speed_cells"] = [[0, 1]]
        edges, warnings, get = self.build(data, data)
        self.assertEqual((edges, get.call_count, len(warnings)), ([], 2, 2))
        self.assertEqual(self.cache_rows(), [])

    def test_empty_fallback_list_is_valid_but_malformed_flag_is_not(self):
        good = table()
        good["fallback_speed_cells"] = []
        routing._validate_table(good, 2)
        for value in (False, None, "", {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                routing._validate_table({**good, "fallback_speed_cells": value}, 2)

    def test_invalid_matrix_shapes_are_not_cached(self):
        bad_values = [[], {}, {**table(), "durations": []}, {**table(), "sources": []}]
        malformed = table()
        malformed["distances"][1] = [0]
        bad_values.append(malformed)
        for index, data in enumerate(bad_values):
            with self.subTest(index=index):
                edges, warnings, get = self.build(data, data)
                self.assertEqual((edges, get.call_count, len(warnings)), ([], 2, 2))
                self.assertEqual(self.cache_rows(), [])

    def test_invalid_numeric_cells_reject_bool_nonfinite_negative_and_null_mismatch(self):
        for value in (True, -1, "123", float("nan"), float("inf"), 10 ** 400, None):
            data = table()
            data["durations"][0][1] = value
            with self.subTest(value_type=type(value).__name__), self.assertRaises(ValueError):
                routing._validate_table(data, 2)

    def test_reported_snap_distance_over_walking_limit_uses_only_car(self):
        foot = table()
        foot["sources"][0]["distance"] = 101
        foot["destinations"][0]["distance"] = 101
        edges, warnings, _ = self.build(foot, table())
        self.assertEqual(warnings, [])
        self.assertTrue(all(edge.mode == "автомобиль/такси" for edge in edges))

    def test_remote_snapped_location_cannot_claim_distance_zero(self):
        data = table()
        for key in ("sources", "destinations"):
            data[key][0] = {"distance": 0, "location": [110, 60]}
        edges, warnings, _ = self.build(data, data)
        self.assertEqual(edges, [])
        self.assertTrue(warnings)

    def test_waypoint_coordinates_and_reported_distance_must_be_valid(self):
        for wp in (None, {}, {"distance": True, "location": [100, 50]},
                   {"distance": 0, "location": [True, 50]}, {"distance": 0, "location": [100]},
                   {"distance": 251, "location": [100, 50]},
                   {"distance": 0, "location": [100, float("nan")]}):
            with self.subTest(waypoint=wp):
                self.assertFalse(routing._waypoint_matches(wp, POINTS[0], 250))

    def test_walking_length_and_duration_caps_do_not_become_paid_walks(self):
        for foot in (table(metres=5001), table(seconds=5401)):
            with self.subTest(foot=foot["durations"][0][1]):
                self.cache.unlink(missing_ok=True)
                edges, _, _ = self.build(foot, table())
                self.assertTrue(all(edge.mode == "автомобиль/такси" for edge in edges))

    def test_car_unknown_price_uses_explicit_reserve_and_unbuffered_depth_credit(self):
        foot = table()
        foot["durations"][0][1] = foot["distances"][0][1] = None
        foot["durations"][1][0] = foot["distances"][1][0] = None
        edges, _, _ = self.build(foot, table(metres=12000))
        request = TripRequest(destination="Тестовый край", start_date="2099-01-01", end_date="2099-01-01", budget=10000, currency="RUB")
        for edge in edges:
            self.assertIsInstance(edge, TravelFact)
            self.assertEqual((edge.group_cost, edge.planning_reserve, edge.price_source), (None, 600, ""))
            self.assertEqual((edge.minutes, edge.depth_credit_min), (22, 10))
            self.assertIn("не тариф", edge.price_note)
            self.assertIsNone(_travel_problem(edge, request))

    def test_walk_cost_zero_is_explicit_and_travelfact_contract_is_compatible(self):
        edges, _, _ = self.build()
        request = TripRequest(destination="Тестовый край", start_date="2099-01-01", end_date="2099-01-01", budget=10000, currency="RUB")
        for edge in edges:
            self.assertEqual((edge.group_cost, edge.planning_reserve, edge.currency), (0.0, None, "RUB"))
            self.assertEqual(edge.depth_credit_min, 10)
            self.assertLess(edge.depth_credit_min, edge.minutes)
            self.assertIsNone(_travel_problem(edge, request))

    def test_physically_impossible_road_distance_does_not_become_an_edge(self):
        edges, warnings, _ = self.build(table(metres=1), table(metres=1))
        self.assertEqual(edges, [])
        self.assertTrue(warnings)

    def test_invalid_grounded_coordinates_stop_before_cache_or_http(self):
        for value in (True, None, float("nan"), 91, 10 ** 400):
            with self.subTest(value_type=type(value).__name__):
                self.grounding.base_facts[0]["lat"] = value
                edges, warnings, get = self.build()
                self.assertEqual(edges, [])
                self.assertTrue(warnings)
                get.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_ambiguous_missing_and_duplicate_nodes_do_not_send_requests(self):
        original = deepcopy(self.grounding)
        for change in ("duplicate_query", "missing_query", "duplicate_key", "bad_record"):
            with self.subTest(change=change):
                self.grounding = deepcopy(original)
                if change == "duplicate_query":
                    self.grounding.confirmed_hints.append(deepcopy(self.grounding.confirmed_hints[0]))
                elif change == "missing_query":
                    self.grounding.confirmed_hints = []
                elif change == "duplicate_key":
                    self.grounding.base_facts[0]["base_key"] = "visit_1"
                else:
                    self.grounding.confirmed_hints = [None]
                edges, warnings, get = self.build()
                self.assertEqual(edges, [])
                self.assertTrue(warnings)
                get.assert_not_called()

    def test_table_rejects_bad_profile_coordinates_and_point_count_before_io(self):
        for points, profile in ((POINTS, "ferry"), ([], "car"), (POINTS * 11, "foot"),
                                ([(True, 0), (1, 1)], "foot"), (None, "car")):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                routing._table(points, profile, cache_file=self.cache, get=Mock())
        self.assertFalse(self.cache.exists())

    def test_disabled_graph_does_not_send_or_write_anything(self):
        with patch.dict(routing.os.environ, {"ROUTING_ENABLED": "false"}):
            edges, warnings, get = self.build()
        self.assertEqual(edges, [])
        self.assertTrue(warnings)
        get.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_untrusted_origin_is_rejected_without_leaking_credentials(self):
        for origin in ("http://router.invalid", "https://user:SECRET@router.invalid", "https://router.invalid?token=SECRET"):
            with self.subTest(origin=origin), patch.dict(routing.os.environ, {"OSRM_ROUTING_BASE_URL": origin}):
                edges, warnings, get = self.build()
                self.assertEqual(edges, [])
                self.assertNotIn("SECRET", json.dumps(warnings))
                get.assert_not_called()

    def test_cache_reuses_both_profiles_until_ttl_then_refreshes_without_extra_calls(self):
        self.build()
        cached_get = Mock(side_effect=AssertionError("Fresh cache"))
        edges, warnings, _ = self.build(get=cached_get)
        self.assertEqual((len(edges), warnings), (2, []))
        cached_get.assert_not_called()
        self.clock.now += routing.CACHE_TTL + 2
        _, _, refresh = self.build()
        self.assertEqual(refresh.call_count, 2)
        self.assertEqual(len(self.cache_rows()), 2)

    def test_corrupted_or_future_cache_is_not_reused(self):
        self.build()
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            connection.execute("UPDATE routing_cache SET payload='not JSON'")
        _, _, get = self.build()
        self.assertEqual(get.call_count, 2)
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            connection.execute("UPDATE routing_cache SET checked=?", (self.clock.now + 1000,))
        _, _, get = self.build()
        self.assertEqual(get.call_count, 2)

    def test_failed_fast_requests_preserve_shared_clock_and_rate_limit(self):
        request_times = []

        def fail(*args, **kwargs):
            request_times.append(self.clock.now)
            raise httpx.ReadTimeout("SECRET provider detail")

        edges, warnings, get = self.build(get=Mock(side_effect=fail))
        self.assertEqual((edges, get.call_count), ([], 2))
        self.assertGreaterEqual(request_times[1] - request_times[0], 1.0)
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            saved = connection.execute("SELECT requested FROM routing_clock WHERE id=1").fetchone()[0]
        self.assertEqual(saved, request_times[-1])
        self.assertEqual(self.cache_rows(), [])
        self.assertNotIn("SECRET", json.dumps(warnings))

    def test_oversized_and_malformed_json_failures_also_preserve_clock(self):
        huge = response(table())
        huge.content = b"x" * 2_000_001
        invalid = response(table())
        invalid.json.side_effect = ValueError("SECRET invalid payload")
        edges, warnings, get = self.build(get=Mock(side_effect=[huge, invalid]))
        self.assertEqual((edges, get.call_count, len(warnings)), ([], 2, 2))
        self.assertGreaterEqual(sum(self.clock.sleeps), 1.0)
        self.assertEqual(self.cache_rows(), [])
        self.assertNotIn("SECRET", json.dumps(warnings))

    def selected_edge(self, *, car=False, **changes):
        edge = TravelFact("base_1", "visit_1", 22 if car else 17,
                          "автомобиль/такси" if car else "пешком", "routing",
                          group_cost=None if car else 0.0,
                          planning_reserve=500.0 if car else None, depth_credit_min=10)
        return replace(edge, **changes)

    def verify(self, edges, *data, get=None):
        injected = get or Mock(side_effect=[response(value) for value in (data or (route(),))])
        problems = routing.verify_route_edges(edges, self.grounding, self.visits, cache_file=self.cache, get=injected)
        return problems, injected

    def test_selected_walking_and_driving_steps_are_strict_and_deduplicated(self):
        foot, car = self.selected_edge(), self.selected_edge(car=True)
        problems, get = self.verify([foot, foot, car], route(), route(mode="driving"))
        self.assertEqual(problems, [])
        self.assertEqual(get.call_count, 2)
        for index, call in enumerate(get.call_args_list):
            self.assertIn("routed-foot/route" if index == 0 else "routed-car/route", call.args[0])
            self.assertEqual(call.kwargs["params"], {"steps": "true", "overview": "false", "alternatives": "false", "generate_hints": "false"})
            self.assertNotIn("exclude", call.kwargs["params"])
        self.assertEqual((foot.minutes, foot.depth_credit_min, car.planning_reserve), (17, 10, 500))

    def test_directed_reverse_is_checked_separately_and_not_inferred(self):
        edge = self.selected_edge()
        reverse = replace(edge, from_key="visit_1", to_key="base_1")
        problems, get = self.verify([edge, reverse], route(), route(list(reversed(POINTS))))
        self.assertEqual(problems, [])
        self.assertEqual(get.call_count, 2)
        self.assertNotEqual(get.call_args_list[0].args[0], get.call_args_list[1].args[0])

    def test_ferry_unknown_wrong_mode_and_missing_mode_fail_closed(self):
        for mode in ("ferry", "train", "cycling", "driving", "other", "", None):
            with self.subTest(mode=mode):
                self.cache.unlink(missing_ok=True)
                data = route()
                data["routes"][0]["legs"][0]["steps"][-1]["mode"] = mode
                problems, get = self.verify([self.selected_edge()], data)
                self.assertEqual((len(problems), get.call_count), (1, 1))
                self.assertEqual(self.cache_rows(), [])

    def test_empty_missing_or_multiple_routes_legs_steps_are_rejected(self):
        variants = [{"code": "NoRoute"}, {}, {**route(), "routes": []}]
        for key in ("routes", "legs", "steps"):
            data = route()
            parent = data if key == "routes" else data["routes"][0] if key == "legs" else data["routes"][0]["legs"][0]
            parent[key] = []
            variants.append(data)
            if key != "steps":
                data = route()
                parent = data if key == "routes" else data["routes"][0]
                parent[key] *= 2
                variants.append(data)
        for index, data in enumerate(variants):
            with self.subTest(index=index):
                self.cache.unlink(missing_ok=True)
                problems, get = self.verify([self.selected_edge()], data)
                self.assertEqual((len(problems), get.call_count), (1, 1))

    def test_selected_snap_actual_and_reported_distance_cannot_be_faked(self):
        for waypoint in ({"distance": 101, "location": [100, 50]}, {"distance": 0, "location": [110, 60]},
                         {"distance": 0, "location": [True, 50]}):
            with self.subTest(waypoint=waypoint):
                self.cache.unlink(missing_ok=True)
                data = route()
                data["waypoints"][0] = waypoint
                problems, get = self.verify([self.selected_edge()], data)
                self.assertEqual((len(problems), get.call_count), (1, 1))

    def test_route_and_every_step_numeric_data_is_validated(self):
        for value in (True, -1, "600", None, float("inf"), float("nan")):
            with self.subTest(value=value):
                data = route()
                data["routes"][0]["legs"][0]["steps"][0]["duration"] = value
                with self.assertRaises(ValueError):
                    routing._validate_route(data, POINTS, "foot")

    def test_changed_route_duration_rejects_stale_schedule_without_retry(self):
        problems, get = self.verify([self.selected_edge()], route(seconds=17 * 60 + 1))
        self.assertEqual((len(problems), get.call_count), (1, 1))

    def test_shorter_route_rejects_stale_depth_credit_without_mutation(self):
        edge = self.selected_edge()
        problems, get = self.verify([edge], route(seconds=60))
        self.assertEqual((len(problems), get.call_count, edge.depth_credit_min), (1, 1, 10))

    def test_fractional_minutes_never_round_up_into_program_depth_credit(self):
        edges, warnings, _ = self.build(table(seconds=541), table(seconds=541))
        self.assertEqual(warnings, [])
        self.assertEqual([(edge.minutes, edge.depth_credit_min) for edge in edges], [(17, 9), (17, 9)])
        problems, get = self.verify([edges[0]], route(seconds=541))
        self.assertEqual((problems, get.call_count), ([], 1))
        problems, get = self.verify([replace(edges[0], depth_credit_min=10)],
                                   get=Mock(side_effect=AssertionError("Cached route")))
        self.assertEqual(len(problems), 1)
        get.assert_not_called()

    def test_longer_car_route_rejects_underreserved_cost(self):
        problems, get = self.verify([self.selected_edge(car=True)], route(mode="driving", metres=12000))
        self.assertEqual((len(problems), get.call_count), (1, 1))
        good = self.selected_edge(car=True, planning_reserve=600)
        problems, unused = self.verify([good], get=Mock(side_effect=AssertionError("Already cached raw route")))
        self.assertEqual(problems, [])
        unused.assert_not_called()

    def test_totals_cannot_hide_large_step_duration_or_distance(self):
        for field, value in (("duration", 5000), ("distance", 12000)):
            with self.subTest(field=field):
                self.cache.unlink(missing_ok=True)
                data = route(mode="driving")
                data["routes"][0]["legs"][0]["steps"][0][field] = value
                problems, get = self.verify([self.selected_edge(car=True)], data)
                self.assertEqual((len(problems), get.call_count), (1, 1))

    def test_duplicate_pair_validates_each_occurrence_budget_not_only_first(self):
        edge = self.selected_edge()
        problems, get = self.verify([edge, replace(edge, minutes=5, depth_credit_min=5)])
        self.assertEqual((len(problems), get.call_count), (1, 1))

    def test_impossible_distance_and_excessive_walk_never_pass(self):
        for data in (route(metres=1), route(metres=5001), route(seconds=5401)):
            with self.subTest(metres=data["routes"][0]["distance"]):
                self.cache.unlink(missing_ok=True)
                problems, get = self.verify([self.selected_edge(minutes=600)], data)
                self.assertEqual((len(problems), get.call_count), (1, 1))

    def test_invalid_selected_facts_fail_before_http_or_cache(self):
        for change in ({"mode": "паром"}, {"evidence": "source"}, {"minutes": True}, {"from_key": "unknown"},
                       {"depth_credit_min": True}, {"group_cost": True}, {"currency": "USD"}, {"planning_reserve": 0}):
            with self.subTest(change=change):
                problems, get = self.verify([self.selected_edge(**change)])
                self.assertTrue(problems)
                get.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_selected_max32_is_preflighted_before_any_http(self):
        # Seven valid grounded nodes offer 42 directed pairs; 33 must reject
        # before issuing even the first request, not after consuming a budget.
        query = self.visits[0].place_query
        self.visits = [replace(self.visits[0], visit_id=f"v{i}", place_query=f"{query}{i}") for i in range(6)]
        self.grounding.confirmed_hints = [{"place_query": fact.place_query, "lat": 50 + (i + 1) * .001, "lon": 100}
                                         for i, fact in enumerate(self.visits)]
        keys = ["base_1"] + [fact.visit_id for fact in self.visits]
        edges = [self.selected_edge(from_key=start, to_key=end) for start in keys for end in keys if start != end][:33]
        problems, get = self.verify(edges)
        self.assertTrue(problems)
        get.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_selected32_exactly_remains_bounded_without_retries(self):
        query = self.visits[0].place_query
        self.visits = [replace(self.visits[0], visit_id=f"v{i}", place_query=f"{query}{i}") for i in range(6)]
        self.grounding.confirmed_hints = [{"place_query": fact.place_query, "lat": 50 + (i + 1) * .001, "lon": 100}
                                         for i, fact in enumerate(self.visits)]
        keys = ["base_1"] + [fact.visit_id for fact in self.visits]
        edges = [self.selected_edge(from_key=start, to_key=end) for start in keys for end in keys if start != end][:32]
        problems, get = self.verify(edges + edges, get=Mock(side_effect=httpx.ReadTimeout("SECRET")))
        self.assertEqual((len(problems), get.call_count), (32, 32))
        self.assertNotIn("SECRET", json.dumps(problems))

    def test_route_cache_ttl_and_shared_table_clock_are_used(self):
        self.build()
        now = self.clock.now
        problems, get = self.verify([self.selected_edge()])
        self.assertEqual((problems, get.call_count), ([], 1))
        self.assertGreaterEqual(self.clock.now - now, 1)
        problems, get = self.verify([self.selected_edge()], get=Mock(side_effect=AssertionError("Cache only")))
        self.assertEqual(problems, [])
        get.assert_not_called()
        self.clock.now += routing.CACHE_TTL + 1
        problems, get = self.verify([self.selected_edge()])
        self.assertEqual((problems, get.call_count), ([], 1))
        self.assertEqual(len(self.cache_rows()), 3)

    def test_fast_route_failure_preserves_clock_for_next_table_profile(self):
        problems, get = self.verify([self.selected_edge()], get=Mock(side_effect=httpx.ConnectError("SECRET")))
        self.assertEqual((len(problems), get.call_count), (1, 1))
        before = self.clock.now
        self.build()
        self.assertGreaterEqual(self.clock.now - before, 2.0)
        self.assertNotIn("SECRET", json.dumps(problems))

    def test_corrupt_route_cache_modes_must_be_revalidated(self):
        self.verify([self.selected_edge()])
        with closing(sqlite3.connect(self.cache)) as connection, connection:
            connection.execute("UPDATE routing_cache SET payload=?", (json.dumps(route(mode="ferry")),))
        problems, get = self.verify([self.selected_edge()])
        self.assertEqual((problems, get.call_count), ([], 1))

    def test_selected_skeleton_extracts_only_travel_slots_without_mutating(self):
        edge = self.selected_edge()
        slot = SimpleNamespace(kind="travel", duration_min=17, travel_fact=edge)
        skeleton = SimpleNamespace(status="ready", days=[SimpleNamespace(slots=[slot, SimpleNamespace(kind="rest")])])
        injected = Mock(return_value=response(route()))
        problems = routing.verify_selected_routes(skeleton, self.grounding, self.visits, cache_file=self.cache, get=injected)
        self.assertEqual(problems, [])
        self.assertEqual(injected.call_count, 1)
        self.assertIs(slot.travel_fact, edge)

    def test_incomplete_skeleton_never_passes_or_sends_http(self):
        for skeleton in (SimpleNamespace(status="needs_facts", days=[]), SimpleNamespace(status="ready", days=[]),
                         SimpleNamespace(status="ready", days=[SimpleNamespace(slots=[SimpleNamespace(kind="travel", duration_min=1, travel_fact=None)])])):
            injected = Mock()
            self.assertTrue(routing.verify_selected_routes(skeleton, self.grounding, self.visits, cache_file=self.cache, get=injected))
            injected.assert_not_called()

    def loop_visit(self, **changes):
        return replace(self.visits[0], route_kind="loop", access_point=(50.0105, 100.0005),
                       access_point_source="https://park.example/official-loop.pdf", **changes)

    def test_default_point_fields_preserve_legacy_map_coordinates(self):
        visit = self.visits[0]
        self.assertEqual((visit.route_kind, visit.access_point, visit.access_point_source), ("point", None, ""))
        self.assertTrue(_visit_access_valid(visit))
        nodes = routing._routing_nodes(self.grounding, self.visits)
        self.assertEqual([(value["lat"], value["lon"]) for _, value in nodes], POINTS)

    def test_confirmed_loop_routes_to_source_entry_both_ways_without_map_mutation_or_ai(self):
        before = deepcopy(self.grounding)
        visit = self.loop_visit()
        self.visits = [visit]
        points = [POINTS[0], visit.access_point]
        with patch("yandex_ai.generate_json", side_effect=AssertionError("No AI")) as ai:
            edges, warnings, get = self.build(table(points), table(points))
            self.assertEqual(warnings, [])
            self.assertEqual([(edge.from_key, edge.to_key) for edge in edges], [("base_1", "visit_1"), ("visit_1", "base_1")])
            for call in get.call_args_list:
                self.assertIn("100.0000000,50.0000000;100.0005000,50.0105000", call.args[0])
            problems, route_get = self.verify(edges, route(points), route(list(reversed(points))))
            self.assertEqual((problems, route_get.call_count), ([], 2))
            ai.assert_not_called()
        self.assertEqual(self.grounding, before)
        self.assertEqual(visit.place_query, before.confirmed_hints[0]["place_query"])
        self.assertEqual((visit.duration_min, [edge.depth_credit_min for edge in edges]), (120, [10, 10]))

    def test_source_entry_is_not_snapped_or_replaced_by_nearby_summit(self):
        self.visits = [self.loop_visit()]
        nodes = dict(routing._routing_nodes(self.grounding, self.visits))
        self.assertEqual((nodes["visit_1"]["lat"], nodes["visit_1"]["lon"]), self.visits[0].access_point)
        self.assertNotEqual((nodes["visit_1"]["lat"], nodes["visit_1"]["lon"]), POINTS[1])
        # Source access point cannot replace invalid/missing map proof.
        self.grounding.confirmed_hints[0]["lat"] = None
        edges, problems, get = self.build()
        self.assertEqual(edges, [])
        self.assertTrue(problems)
        get.assert_not_called()

    def test_source_loop_coordinate_near_chersky_passport_is_within_strict_cap(self):
        self.grounding.confirmed_hints[0].update(lat=51.877471, lon=104.837395)
        self.visits = [replace(self.visits[0], route_kind="loop", access_point=(51.877837, 104.838069),
                               access_point_source="https://baikal-1.ru/tourism/routes-and-maps/chersky-stone-2/")]
        nodes = dict(routing._routing_nodes(self.grounding, self.visits))
        self.assertEqual((nodes["visit_1"]["lat"], nodes["visit_1"]["lon"]), (51.877837, 104.838069))
        self.assertEqual(self.grounding.confirmed_hints[0]["lat"], 51.877471)

    def test_source_loop_entry_over250m_fails_before_any_request_or_cache(self):
        self.visits = [replace(self.loop_visit(), access_point=(50.014, 100.0))]
        edges, problems, get = self.build()
        self.assertEqual(edges, [])
        self.assertTrue(problems)
        get.assert_not_called()
        problems, get = self.verify([self.selected_edge()])
        self.assertTrue(problems)
        get.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_loop_proximity_boundary_is_closed_at250m(self):
        self.visits = [self.loop_visit()]
        with patch.object(routing, "distance_km", return_value=.25):
            self.assertEqual(len(routing._routing_nodes(self.grounding, self.visits)), 2)
        with patch.object(routing, "distance_km", return_value=.250001), self.assertRaises(ValueError):
            routing._routing_nodes(self.grounding, self.visits)

    def test_reviewed_route_anchor_allows_distant_entry_not_a_shifted_poi(self):
        before = deepcopy(self.grounding)
        visit = replace(self.loop_visit(), access_point=(50.0, 100.001),
                        map_anchor_point=POINTS[1], length_km=3.7)
        self.visits = [visit]
        self.assertTrue(_visit_access_valid(visit))
        nodes = dict(routing._routing_nodes(self.grounding, self.visits))
        self.assertEqual((nodes["visit_1"]["lat"], nodes["visit_1"]["lon"]), visit.access_point)
        self.assertEqual(self.grounding, before)
        # A source anchor is not an independent map result: disagreement still
        # refuses the route before making any HTTP request or writing a cache.
        self.grounding.confirmed_hints[0]["lat"] += .01
        edges, problems, get = self.build()
        self.assertEqual(edges, [])
        self.assertTrue(problems)
        get.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_route_anchor_requires_a_plausible_documented_closed_length(self):
        valid = replace(self.loop_visit(), access_point=(50.0, 100.001),
                        map_anchor_point=POINTS[1], length_km=3.7)
        for change in (dict(length_km=None), dict(length_km=0), dict(length_km=True),
                       dict(length_km=1), dict(length_km=51), dict(length_km=float("inf")),
                       dict(map_anchor_point=[50.01, 100]), dict(map_anchor_point=(91, 100)),
                       dict(map_anchor_point=(True, 100)), dict(map_anchor_point=(float("nan"), 100)),
                       dict(route_kind="point"), dict(route_kind="linear")):
            with self.subTest(change=change):
                self.assertFalse(_visit_access_valid(replace(valid, **change)))

    def test_loop_access_coordinates_and_source_are_strict_no_linear_fallback(self):
        loop = self.loop_visit()
        invalid = [replace(loop, access_point=value) for value in
                   (None, (), [50.0105, 100.0005], (50,), (True, 100), (float("nan"), 100),
                    (50, float("inf")), (91, 100), (50, 181), (10 ** 400, 0))]
        invalid += [replace(loop, access_point_source=value) for value in
                    ("", "http://park.example/loop", "https://name:SECRET@park.example/loop", "https://@park.example/loop",
                     "https://park example/loop", "https://park.example:0/loop", "https://park.example:99999/loop",
                     "file:///loop", "https://[", None)]
        invalid += [replace(loop, route_kind=value) for value in ("linear", "out_and_back", "", None)]
        for visit in invalid:
            with self.subTest(kind=visit.route_kind, source=visit.access_point_source):
                self.assertFalse(_visit_access_valid(visit))
                self.assertEqual(_visit_problem(visit, None), "route_access_point")
                self.visits = [visit]
                edges, warnings, get = self.build()
                self.assertEqual(edges, [])
                self.assertTrue(warnings)
                self.assertNotIn("SECRET", json.dumps(warnings))
                get.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_point_visit_cannot_smuggle_access_coordinates_or_source(self):
        for visit in (replace(self.visits[0], access_point=(50.01, 100.0)),
                      replace(self.visits[0], access_point_source="https://park.example/loop")):
            with self.subTest(visit=visit):
                self.assertFalse(_visit_access_valid(visit))
                self.visits = [visit]
                edges, warnings, get = self.build()
                self.assertEqual(edges, [])
                self.assertTrue(warnings)
                get.assert_not_called()

    def test_final_route_validation_uses_entry_snapping_not_summit_snapping(self):
        self.visits = [replace(self.loop_visit(), access_point=(50.0115, 100.0))]
        # Summit is ~167m from the entry: an OSRM route to the summit must not
        # pass the 100m foot snap check just because the POI identity is valid.
        problems, get = self.verify([self.selected_edge()], route())
        self.assertEqual((len(problems), get.call_count), (1, 1))


if __name__ == "__main__":
    unittest.main()
