"""Offline tests for local fact-based allocation; no maps, AI, or real DB."""

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import date, timedelta
from types import SimpleNamespace
import socket
import unittest
from unittest.mock import patch

from regional_skeleton import (
    SkeletonResult, TravelFact, VisitFact, VisitWindow, build_regional_skeleton,
)
from schemas import TripRequest


def request_for(days=1, **changes):
    data = dict(destination="Тестовый регион, Россия", destination_kind="region",
                start_date="2099-10-28", end_date=(date(2099, 10, 28) + timedelta(days=days - 1)).isoformat(),
                adults=2, children_ages=[], budget=100000, currency="RUB", pace="средний",
                start_time="08:00", end_time="21:00", interest_categories=["museums", "parks"])
    data.update(changes)
    return TripRequest(**data)


def windows_for(request, start="08:00", end="21:00"):
    return tuple(VisitWindow((date.fromisoformat(request.start_date) + timedelta(days=i)).isoformat(), start, end,
                             evidence="estimate", note="Плановое окно; часы требуют уточнения.")
                 for i in range(request.days_count))


def fixture(request, count=2, duration=120):
    grounding = SimpleNamespace(base_facts=[dict(base_key="base_1", base_query="Город А, Россия", lat=50.0, lon=100.0)],
                                confirmed_hints=[])
    facts = []
    for i in range(1, count + 1):
        query = f"Музей {i}, Город А, Россия"
        grounding.confirmed_hints.append(dict(title=f"Музей {i}", place_query=query,
                                             lat=50.0 + i * .01, lon=100.0, base_key="base_1",
                                             allowed_base_keys=["base_1"]))
        facts.append(VisitFact(f"visit_{i}", query, f"Содержание посещения {i}", duration, "published",
                               duration_source="https://example.org/visits", indoor=True,
                               date_windows=windows_for(request), group_cost=200, price_source="https://example.org/tariff",
                               children_allowed=True, interest_categories=("museums",)))
    nodes = ["base_1", *(fact.visit_id for fact in facts)]
    graph = [TravelFact(origin, destination, 10, "walking", "routing", source="fixture-router",
                        group_cost=0, price_source="walking-no-fare")
             for origin in nodes for destination in nodes if origin != destination]
    return grounding, facts, graph


class RegionalSkeletonTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "create_connection"), (socket, "getaddrinfo")):
            blocker = patch.object(owner, name, side_effect=AssertionError("network forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)
        blocker = patch("places.geocode", side_effect=AssertionError("map calls forbidden"))
        blocker.start()
        self.addCleanup(blocker.stop)

    def build(self, request, grounding, facts, graph, **kwargs):
        return build_regional_skeleton(request, grounding, facts, travel_facts=graph, **kwargs)

    def test_two_days_are_complete_unique_budgeted_and_inputs_unchanged(self):
        request = request_for(2)
        grounding, facts, graph = fixture(request, 4)
        before = deepcopy((request.model_dump(), vars(grounding), facts, graph))
        result = self.build(request, grounding, facts, graph)
        self.assertEqual(result.status, "ready", result)
        self.assertEqual([day.day for day in result.days], [1, 2])
        self.assertEqual(len({visit for day in result.days for visit in day.visit_ids}), 4)
        self.assertEqual((result.known_cost_total, result.planning_reserve_total), (800, 2800))
        for day in result.days:
            kinds = [slot.kind for slot in day.slots]
            self.assertEqual((kinds.count("attraction"), kinds.count("meal"), kinds.count("rest")), (2, 1, 1))
            self.assertEqual((day.visit_minutes, day.travel_minutes, day.depth_credit_min), (240, 30, 0))
            for earlier, later in zip(day.slots, day.slots[1:]):
                self.assertLessEqual(earlier.end_time, later.start_time)
            for slot in day.slots:
                self.assertGreaterEqual(slot.start_time, request.start_time)
                self.assertLessEqual(slot.end_time, request.end_time)
                if slot.kind == "meal":
                    self.assertIsNone(slot.group_cost)
                    self.assertEqual(slot.planning_reserve, 1400)
        self.assertEqual(before, (request.model_dump(), vars(grounding), facts, graph))
        self.assertEqual(asdict(result)["status"], "ready")

    def test_missing_or_one_way_graph_never_creates_connections_from_coordinates(self):
        request = request_for()
        grounding, facts, graph = fixture(request, 1, 240)
        for edges in ([], [edge for edge in graph if edge.from_key == "base_1"]):
            with self.subTest(edges=edges):
                result = self.build(request, grounding, facts, edges)
                self.assertEqual(result.status, "needs_facts")
                self.assertEqual(result.days, ())
                self.assertTrue(any(item.startswith("travel:") for item in result.missing_facts))

    def test_base_change_requires_explicit_directed_edge_in_current_day(self):
        request = request_for(2)
        grounding, facts, _ = fixture(request, 2, 240)
        grounding.base_facts.append(dict(base_key="base_2", base_query="Город Б, Россия", lat=50.5, lon=100))
        grounding.confirmed_hints[1].update(base_key="base_2", allowed_base_keys=["base_2"])
        facts = [replace(fact, date_windows=(windows_for(request)[i],)) for i, fact in enumerate(facts)]
        graph = [TravelFact(origin, destination, 20, "car", "routing", source="router",
                            group_cost=0, price_source="prepaid") for i in (1, 2)
                 for origin, destination in ((f"base_{i}", f"visit_{i}"), (f"visit_{i}", f"base_{i}"))]
        refused = self.build(request, grounding, facts, graph)
        self.assertEqual(refused.status, "needs_facts")
        graph.append(TravelFact("base_1", "base_2", 70, "car", "routing", source="router",
                                group_cost=300, price_source="quote", depth_credit_min=60))
        result = self.build(request, grounding, facts, graph)
        self.assertEqual(result.status, "ready", result)
        self.assertEqual([day.base_key for day in result.days], ["base_1", "base_2"])
        second = result.days[1]
        self.assertEqual((second.slots[0].travel_fact.from_key, second.slots[0].travel_fact.to_key), ("base_1", "base_2"))
        self.assertEqual((second.travel_minutes, second.depth_credit_min), (110, 60))

    def test_no_duration_inflation_or_repeated_places_to_fill_a_week(self):
        request = request_for(7)
        grounding, facts, graph = fixture(request, 8, 120)
        result = self.build(request, grounding, facts, graph, max_search_states=2500)
        self.assertEqual(result.status, "insufficient_content")
        self.assertLess(result.search_states, 10)
        self.assertEqual(result.days, ())
        self.assertTrue(all(fact.duration_min == 120 for fact in facts))

    def test_documented_long_visit_not_split_and_estimate_provenance_is_preserved(self):
        request = request_for()
        grounding, facts, graph = fixture(request, 1, 400)
        facts[0] = replace(facts[0], duration_basis="planning_estimate", duration_source="",
                           basis_note="Обоснованный резерв на длинное содержание; не опубликованная длительность.",
                           restrictions=("Требуется предварительная запись",))
        result = self.build(request, grounding, facts, graph)
        self.assertEqual(result.status, "ready", result)
        visits = [slot for slot in result.days[0].slots if slot.kind == "attraction"]
        self.assertEqual(len(visits), 1)
        self.assertEqual(visits[0].duration_min, 400)
        self.assertEqual(visits[0].visit_fact, facts[0])

    def test_unknown_duration_not_calculated_from_length(self):
        request = request_for()
        grounding, facts, graph = fixture(request, 1, 240)
        facts[0] = replace(facts[0], duration_min=None, length_km=12.0)
        result = self.build(request, grounding, facts, graph)
        self.assertEqual(result.status, "needs_facts")
        self.assertIn("visit:visit_1:duration_min", result.missing_facts)

    def test_unknown_price_requires_explicit_reserve_and_affects_budget(self):
        request = request_for(budget=2300)
        grounding, facts, graph = fixture(request, 1, 240)
        facts[0] = replace(facts[0], group_cost=None, price_source="")
        result = self.build(request, grounding, facts, graph)
        self.assertIn("visit:visit_1:price_or_explicit_reserve", result.missing_facts)
        facts[0] = replace(facts[0], planning_reserve=1000, price_note="Резерв; тариф не известен.")
        self.assertNotEqual(self.build(request, grounding, facts, graph).status, "ready")
        result = self.build(request.model_copy(update={"budget": 2500}), grounding, facts, graph)
        self.assertEqual(result.status, "ready", result)
        self.assertEqual((result.known_cost_total, result.planning_reserve_total), (0, 2400))
        attraction = next(slot for slot in result.days[0].slots if slot.kind == "attraction")
        self.assertIsNone(attraction.group_cost)
        self.assertEqual(attraction.planning_reserve, 1000)

    def test_published_free_price_allowed_but_currency_never_converted(self):
        request = request_for()
        grounding, facts, graph = fixture(request, 1, 240)
        facts[0] = replace(facts[0], group_cost=0)
        self.assertEqual(self.build(request, grounding, facts, graph).status, "ready")
        facts[0] = replace(facts[0], currency="EUR")
        self.assertEqual(self.build(request, grounding, facts, graph).status, "needs_facts")

    def test_source_and_routing_credit_can_complete_day_but_estimates_never_can(self):
        request = request_for()
        grounding, facts, graph = fixture(request, 1, 120)
        for evidence in ("source", "routing"):
            edges = [replace(edge, minutes=60, depth_credit_min=60, evidence=evidence) for edge in graph]
            result = self.build(request, grounding, facts, edges)
            self.assertEqual(result.status, "ready", result)
            self.assertEqual(result.days[0].depth_credit_min, 120)
        edges = [replace(edge, minutes=70, depth_credit_min=60, evidence="estimate", note="Оценка времени.") for edge in graph]
        refused = self.build(request, grounding, facts, edges)
        self.assertTrue(any("depth_credit_min" in value for value in refused.missing_facts))
        edges = [replace(edge, depth_credit_min=0) for edge in edges]
        self.assertNotEqual(self.build(request, grounding, facts, edges).status, "ready")

    def test_credits_are_not_moved_between_days_and_travel_windows_are_real_constraints(self):
        request = request_for(2)
        grounding, facts, graph = fixture(request, 2, 120)
        facts = [replace(fact, date_windows=(windows_for(request)[i],)) for i, fact in enumerate(facts)]
        graph = [replace(edge, minutes=120, depth_credit_min=120)
                 if "visit_1" in (edge.from_key, edge.to_key) else edge for edge in graph]
        result = self.build(request, grounding, facts, graph)
        self.assertNotEqual(result.status, "ready")
        self.assertEqual(result.days, ())
        one_day = request_for()
        grounding, facts, graph = fixture(one_day, 1, 240)
        graph[0] = replace(graph[0], date_windows=windows_for(one_day, "20:00", "21:00"))
        self.assertNotEqual(self.build(one_day, grounding, facts, graph).status, "ready")

    def test_schedule_keeps_existing_base_and_does_not_add_credit_only_detours(self):
        request = request_for(2)
        grounding, facts, graph = fixture(request, 2, 240)
        grounding.base_facts.append(dict(base_key="base_2", base_query="Город Б, Россия", lat=50.3, lon=100.0))
        for hint in grounding.confirmed_hints:
            hint["allowed_base_keys"] = ["base_1", "base_2"]
        graph.extend(TravelFact(origin, destination, 180, "car", "routing", source="router",
                                group_cost=1000, price_source="quote", depth_credit_min=180)
                     for origin, destination in (("base_1", "base_2"), ("base_2", "base_1")))
        result = self.build(request, grounding, facts, graph)
        self.assertEqual(result.status, "ready", result)
        self.assertEqual([day.base_key for day in result.days], ["base_1", "base_1"])
        self.assertEqual(sum(day.depth_credit_min for day in result.days), 0)

    def test_credit_must_be_integer_not_boolean_and_not_exceed_actual_time(self):
        request = request_for()
        grounding, facts, graph = fixture(request, 1, 240)
        for credit in (True, 11, -1, 1.5):
            with self.subTest(credit=credit):
                result = self.build(request, grounding, facts, [replace(graph[0], depth_credit_min=credit), graph[1]])
                self.assertNotEqual(result.status, "ready")

    def test_family_eligibility_rest_and_attraction_limit(self):
        request = request_for(children_ages=[2])
        grounding, facts, graph = fixture(request, 3, 30)
        result = self.build(request, grounding, facts, graph)
        self.assertEqual(result.status, "ready", result)
        self.assertLessEqual(len(result.days[0].visit_ids), 2)
        self.assertEqual(next(slot.duration_min for slot in result.days[0].slots if slot.kind == "rest"), 60)
        for change in (dict(children_allowed=False), dict(children_allowed=None), dict(minimum_age=7)):
            blocked = self.build(request, grounding, [replace(fact, **change) for fact in facts], graph)
            unknown = change == dict(children_allowed=None)
            self.assertEqual(blocked.status, "needs_facts" if unknown else "insufficient_content")
            self.assertEqual(bool(blocked.missing_facts), unknown)
            self.assertEqual(bool(blocked.coverage.exclusions), not unknown)
        with self.assertRaises(ValueError):
            self.build(request, grounding, facts, graph, rest_minutes=15)

    def test_interest_and_rain_constraints_are_not_ignored(self):
        request = request_for()
        grounding, facts, graph = fixture(request, 1, 240)
        facts[0] = replace(facts[0], interest_categories=("shopping_malls",))
        rejected = self.build(request, grounding, facts, graph)
        self.assertEqual(rejected.status, "insufficient_content")
        self.assertEqual(rejected.missing_facts, ())
        self.assertEqual(rejected.coverage.exclusions, (("visit_1", "interests_not_matched"),))
        facts[0] = replace(facts[0], interest_categories=("parks",), indoor=False)
        self.assertNotEqual(self.build(request, grounding, facts, graph, rainy_dates=[request.start_date]).status, "ready")
        self.assertEqual(self.build(request, grounding, facts, graph).status, "ready")

    def test_actual_windows_and_source_notes_required(self):
        request = request_for()
        grounding, facts, graph = fixture(request, 1, 240)
        for supplied in ((), (VisitWindow(request.start_date, "08:00", "21:00"),),
                         (VisitWindow(request.start_date, "08:00", "21:00", evidence="source"),),
                         windows_for(request, "09:00", "10:00")):
            with self.subTest(supplied=supplied):
                self.assertNotEqual(self.build(request, grounding, [replace(facts[0], date_windows=supplied)], graph).status, "ready")
        facts[0] = replace(facts[0], date_windows=windows_for(request, "10:00", "14:00"))
        result = self.build(request, grounding, facts, graph)
        visit = next(slot for slot in result.days[0].slots if slot.kind == "attraction")
        self.assertEqual((visit.start_time, visit.end_time), ("10:00", "14:00"))

    def test_scarce_date_visit_is_reserved_in_bounded_search(self):
        request = request_for(2)
        grounding, facts, graph = fixture(request, 2, 240)
        facts[0] = replace(facts[0], date_windows=(windows_for(request)[0],))
        result = self.build(request, grounding, facts, graph)
        self.assertEqual(result.status, "ready", result)
        self.assertEqual([day.visit_ids for day in result.days], [("visit_1",), ("visit_2",)])

    def test_aliases_of_same_map_object_cannot_supply_extra_visits_or_days(self):
        request = request_for()
        for same_point in (True, False):
            grounding, facts, graph = fixture(request, 2, 120)
            if same_point:
                grounding.confirmed_hints[1]["lat"] = grounding.confirmed_hints[0]["lat"] + 0.0001
            else:
                for hint in grounding.confirmed_hints:
                    hint.update(osm_type="way", osm_id=123)
            with self.subTest(same_point=same_point):
                self.assertNotEqual(self.build(request, grounding, facts, graph).status, "ready")
                facts = [replace(fact, duration_min=240) for fact in facts]
                result = self.build(request, grounding, facts, graph)
                self.assertEqual(result.status, "ready", result)
                self.assertEqual(len(result.days[0].visit_ids), 1)

    def test_exact_grounded_query_allowed_base_and_requested_stay_are_enforced(self):
        request = request_for()
        grounding, facts, graph = fixture(request, 1, 240)
        bad = replace(facts[0], place_query="Музей 1, Другое место, Россия")
        self.assertEqual(self.build(request, grounding, [bad], graph).status, "needs_facts")
        self.assertEqual(self.build(request.model_copy(update={"stay_area": "Город"}), grounding, facts, graph).status, "needs_facts")
        self.assertEqual(self.build(request.model_copy(update={"stay_area": "Город А"}), grounding, facts, graph).status, "ready")
        grounding.confirmed_hints[0]["allowed_base_keys"] = ["unknown_base"]
        with self.assertRaises(ValueError):
            self.build(request, grounding, facts, graph)

    def test_no_unsupported_water_transfer_while_permit_conditions_are_preserved(self):
        request = request_for()
        grounding, facts, graph = fixture(request, 1, 240)
        facts[0] = replace(facts[0], restrictions=("Нужны разрешение парка и предварительная запись",))
        self.assertEqual(self.build(request, grounding, facts, graph).status, "ready")
        graph[0] = replace(graph[0], mode="Паром", note="Сезонный паром: расписание и доступность не подтверждены")
        result = self.build(request, grounding, facts, graph)
        self.assertTrue(any("unconfirmed_water_transfer" in value for value in result.missing_facts))

    def test_missing_estimate_note_invalid_numbers_duplicate_ids_and_search_limit(self):
        request = request_for()
        grounding, facts, graph = fixture(request, 1, 240)
        for change in (dict(duration_basis="planning_estimate", basis_note=""), dict(duration_min=True),
                       dict(group_cost=float("nan")), dict(group_cost=10**1000), dict(indoor=None)):
            with self.subTest(change=change):
                self.assertNotEqual(self.build(request, grounding, [replace(facts[0], **change)], graph).status, "ready")
        with self.assertRaises(ValueError):
            self.build(request, grounding, [facts[0], facts[0]], graph)
        with self.assertRaises(ValueError):
            self.build(request, grounding, facts, [*graph, graph[0]])
        result = self.build(request, grounding, facts, graph, max_search_states=1)
        self.assertEqual((result.status, result.days, result.search_states), ("search_limit", (), 1))
        self.assertIsInstance(result, SkeletonResult)

    def test_sparse_week_does_not_search_groups_that_leave_later_days_empty(self):
        request = request_for(7)
        grounding, facts, graph = fixture(request, 7, 120)
        # Generous graph-wide credit makes the cheap global upper bound pass,
        # but every single-visit day is genuinely too short. Two-visit groups
        # cannot help when seven unique places must cover seven different days.
        graph = [replace(edge, minutes=60, depth_credit_min=60)
                 if "base_1" not in (edge.from_key, edge.to_key) else edge for edge in graph]
        before = deepcopy((request.model_dump(), vars(grounding), facts, graph))
        with patch("regional_skeleton._timeline", side_effect=AssertionError("No sufficient day exists")):
            result = self.build(request, grounding, facts, graph)
        self.assertEqual((result.status, result.days), ("insufficient_content", ()))
        self.assertLess(result.search_states, 100)
        self.assertEqual(before, (request.model_dump(), vars(grounding), facts, graph))

    def test_sparse_week_capacity_counts_only_remaining_eligible_identities(self):
        request = request_for(7)
        grounding, facts, graph = fixture(request, 9, 120)
        # These extra cards cannot create extra capacity on any remaining date.
        facts[7] = replace(facts[7], date_windows=(replace(windows_for(request)[0], date="2099-11-20"),))
        facts[8] = replace(facts[8], interest_categories=("shopping_malls",))
        graph = [replace(edge, minutes=60, depth_credit_min=60)
                 if "base_1" not in (edge.from_key, edge.to_key) else edge for edge in graph]
        with patch("regional_skeleton._timeline", side_effect=AssertionError("No sufficient day exists")):
            result = self.build(request, grounding, facts, graph)
        self.assertEqual((result.status, result.days), ("insufficient_content", ()))
        self.assertLess(result.search_states, 100)
        self.assertEqual(result.excluded_visit_ids, ("visit_9",))
        self.assertEqual(result.missing_facts, ())
        self.assertEqual(result.coverage.exclusions, (("visit_9", "interests_not_matched"),))

    def test_future_meal_lower_bound_is_checked_before_timeline(self):
        for budget, visit_cost in ((2799, 0), (3000, 500)):
            request = request_for(2, budget=budget)
            grounding, facts, graph = fixture(request, 2, 240)
            facts = [replace(fact, group_cost=visit_cost) for fact in facts]
            with self.subTest(budget=budget), patch(
                "regional_skeleton._timeline", side_effect=AssertionError("Future meals cannot be funded")
            ):
                result = self.build(request, grounding, facts, graph)
            self.assertEqual((result.status, result.days), ("insufficient_content", ()))
            self.assertLess(result.search_states, 10)

    def test_exact_budget_keeps_same_complete_feasible_plan(self):
        request = request_for(2)
        grounding, facts, graph = fixture(request, 4, 120)
        original = self.build(request, grounding, facts, graph)
        tight = self.build(request.model_copy(update={"budget": 3600}), grounding, facts, graph)
        self.assertEqual(original.status, "ready")
        self.assertEqual(tight.status, "ready")
        self.assertEqual(tight.days, original.days)
        self.assertEqual([day.visit_ids for day in tight.days], [("visit_1", "visit_2"), ("visit_3", "visit_4")])
        self.assertEqual(tight.known_cost_total + tight.planning_reserve_total, 3600)

    def test_failed_state_at_smaller_budget_does_not_prune_a_cheaper_path(self):
        request = request_for(2, budget=4400)
        grounding, facts, graph = fixture(request, 4, 120)
        facts = [replace(fact, date_windows=(windows_for(request)[i // 2],)) for i, fact in enumerate(facts)]
        graph = [replace(edge, group_cost=1000) if (edge.from_key, edge.to_key) == ("visit_1", "visit_2")
                 else edge for edge in graph]
        before = deepcopy((vars(grounding), facts, graph))
        result = self.build(request, grounding, facts, graph)
        self.assertEqual(result.status, "ready", result)
        self.assertEqual([day.visit_ids for day in result.days], [("visit_2", "visit_1"), ("visit_3", "visit_4")])
        self.assertEqual(result.known_cost_total + result.planning_reserve_total, 3600)
        self.assertEqual(before, (vars(grounding), facts, graph))

    def test_available_pool_is_not_a_quota_that_forces_an_unneeded_base_change(self):
        request = request_for(2)
        grounding, facts, graph = fixture(request, 7, 120)
        grounding.base_facts.append(dict(base_key="base_2", base_query="Город Б, Россия", lat=50.5, lon=100))
        for hint in grounding.confirmed_hints[4:]:
            hint.update(base_key="base_2", allowed_base_keys=["base_2"])
        graph.extend(TravelFact(origin, destination, 10, "walking", "routing", source="fixture-router",
                                group_cost=0, price_source="walking-no-fare")
                     for fact in facts[4:] for origin, destination in (("base_2", fact.visit_id), (fact.visit_id, "base_2")))
        graph.extend(TravelFact(origin, destination, 240, "car", "routing", source="fixture-router",
                                planning_reserve=10000, price_note="Explicit synthetic reserve", depth_credit_min=200)
                     for origin, destination in (("base_1", "base_2"), ("base_2", "base_1")))
        result = self.build(request, grounding, facts, graph)
        self.assertEqual(result.status, "ready", result)
        self.assertEqual([day.base_key for day in result.days], ["base_1", "base_1"])
        self.assertEqual([day.visit_ids for day in result.days], [("visit_1", "visit_2"), ("visit_3", "visit_4")])
        self.assertEqual(result.known_cost_total + result.planning_reserve_total, 3600)
        self.assertTrue(all(slot.travel_fact is None or "base_2" not in (slot.travel_fact.from_key, slot.travel_fact.to_key)
                            for day in result.days for slot in day.slots))

    def test_full_day_lunch_must_fit_noon_to_three_and_cannot_split_a_visit(self):
        request = request_for()
        grounding, facts, graph = fixture(request, 1, 300)
        facts = [replace(facts[0], date_windows=windows_for(request, "12:00", "17:00"))]
        result = self.build(request, grounding, facts, graph)
        self.assertEqual((result.status, result.days), ("insufficient_content", ()))
        self.assertEqual(facts[0].duration_min, 300)
        facts = [replace(facts[0], duration_min=240, date_windows=windows_for(request, "14:00", "18:00"))]
        result = self.build(request, grounding, facts, graph)
        self.assertEqual(result.status, "ready", result)
        meal = next(slot for slot in result.days[0].slots if slot.kind == "meal")
        self.assertGreaterEqual(meal.start_time, "12:00")
        self.assertLessEqual(meal.end_time, "15:00")

    def test_late_requested_day_does_not_gain_an_unrequested_noon_constraint(self):
        request = request_for(start_time="16:00", end_time="23:00")
        grounding, facts, graph = fixture(request, 1, 240)
        facts = [replace(facts[0], date_windows=windows_for(request, "16:00", "23:00"))]
        result = self.build(request, grounding, facts, graph)
        self.assertEqual(result.status, "ready", result)
        self.assertTrue(all("16:00" <= slot.start_time < slot.end_time <= "23:00" for slot in result.days[0].slots))

    def test_grounded_provider_identity_deduplicates_aliases_without_becoming_osm(self):
        request = request_for(2)
        grounding, facts, graph = fixture(request, 2, 240)
        for hint in grounding.confirmed_hints:
            hint.update(provider="2gis", provider_id="70000001000000001")
        before = deepcopy(vars(grounding))
        result = self.build(request, grounding, facts, graph)
        self.assertEqual((result.status, result.days), ("insufficient_content", ()))
        self.assertEqual(vars(grounding), before)
        self.assertTrue(all("osm_id" not in hint and "osm_type" not in hint for hint in grounding.confirmed_hints))
        for marker in (True, 1, "", "0", "invalid", "1" * 33):
            for hint in grounding.confirmed_hints:
                hint["provider_id"] = marker
            with self.subTest(marker=marker):
                self.assertEqual(self.build(request, grounding, facts, graph).status, "ready")


if __name__ == "__main__":
    unittest.main()
