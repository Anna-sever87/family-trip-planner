"""Offline adapter/repair regressions; all maps/routing/catalogue are mocked."""

from copy import deepcopy
from contextlib import ExitStack
from dataclasses import replace
import socket
import unittest
from unittest.mock import patch

import source_planning
from regional_skeleton import SkeletonResult, TravelFact, build_regional_skeleton
from schemas import Itinerary, ItineraryDay, ItineraryItem
from test_regional_skeleton import fixture, request_for, windows_for


class SourcePlanningTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "create_connection"), (socket, "getaddrinfo")):
            blocker = patch.object(owner, name, side_effect=AssertionError("network forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)
        for target in ("places.geocode", "regional_routing._table", "regional_routing._route",
                       "regional_routing.sqlite3.connect", "yandex_ai.generate_json"):
            blocker = patch(target, side_effect=AssertionError("real I/O forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)
        for name in ("verify_selected_routes", "verify_route_edges"):
            verifier = patch.object(source_planning, name, return_value=[])
            verifier.start()
            self.addCleanup(verifier.stop)

    def scenario(self, days=1):
        request = request_for(days)
        grounding, facts, graph = fixture(request, 2 * days, 120)
        graph = [replace(edge, group_cost=None, planning_reserve=250,
                         price_note="Резерв 250 на переезд, не тариф.", price_source="") for edge in graph]
        skeleton = build_regional_skeleton(request, grounding, facts, travel_facts=graph)
        self.assertEqual(skeleton.status, "ready", skeleton)
        itinerary = source_planning.itinerary_from_skeleton(request, grounding, skeleton)
        by_query = {hint["place_query"]: hint for hint in grounding.confirmed_hints}
        for day in itinerary.days:
            for item in day.items:
                if item.kind == "attraction":
                    hint = by_query[item.place_query]
                    item.verified, item.lat, item.lon = True, hint["lat"], hint["lon"]
        return request, grounding, facts, graph, skeleton, itinerary

    def validate(self, request, itinerary, facts, graph):
        with patch.object(source_planning, "visit_facts_for_grounding", return_value=facts), \
                patch.object(source_planning, "build_travel_graph", return_value=(graph, [])), \
                patch("places.geocode", return_value={"lat": 50, "lon": 100}):
            return source_planning.validate_source_edit(request, itinerary)

    def test_adapter_preserves_slots_times_and_exact_totals_without_disguised_visits(self):
        request, grounding, facts, graph, skeleton, itinerary = self.scenario(2)
        self.assertEqual(sum(item.cost for day in itinerary.days for item in day.items),
                         skeleton.known_cost_total + skeleton.planning_reserve_total)
        for day, planned in zip(itinerary.days, skeleton.days):
            expected = [slot for slot in planned.slots if slot.kind != "travel"]
            self.assertEqual([(item.kind, item.time, item.duration_min, item.place_query) for item in day.items],
                             [(slot.kind, slot.start_time, slot.duration_min, slot.place_query) for slot in expected])
            self.assertEqual(sum(item.cost for item in day.items), planned.known_cost_total + planned.planning_reserve_total)
            self.assertEqual(sum(item.kind == "meal" for item in day.items), 1)
            self.assertEqual(sum(item.kind == "rest" for item in day.items), 1)
            self.assertEqual(sum(item.kind == "attraction" for item in day.items), len(planned.visit_ids))
            self.assertIn("дорожные расходы/резерв", next(item.description for item in day.items if item.kind == "meal"))
            self.assertTrue(day.transfer_note)

    def test_adapter_keeps_estimate_unknown_price_and_booking_provenance(self):
        request = request_for()
        grounding, facts, graph = fixture(request, 1, 240)
        facts[0] = replace(facts[0], duration_basis="planning_estimate", basis_note="Редакторская оценка, не сеанс.",
                           group_cost=None, planning_reserve=1000, price_source="",
                           price_note="Резерв, не опубликованная цена.", restrictions=("Нужна предварительная запись.",))
        skeleton = build_regional_skeleton(request, grounding, facts, travel_facts=graph)
        itinerary = source_planning.itinerary_from_skeleton(request, grounding, skeleton)
        description = next(item.description for item in itinerary.days[0].items if item.kind == "attraction")
        for text in ("Плановая оценка", "точная цена неизвестна", "не опубликованная цена", "предварительная запись", facts[0].duration_source):
            self.assertIn(text, description)
        self.assertIn("не бронирование", itinerary.summary)

    def test_real_catalog_museum_and_nerpa_adult_text_has_no_child_only_wording(self):
        import visit_facts
        from rules import _CHILD_WORD_RE
        request = request_for(start_date="2026-10-28", end_date="2026-11-03", destination="Байкал, Россия")
        cards = {card["id"]: card for card in visit_facts.load_visit_cards(request.destination)}
        for key in ("baikal_museum", "listvyanka_nerpinarium"):
            card = cards[key]
            from types import SimpleNamespace
            grounding = SimpleNamespace(confirmed_hints=[dict(place_query=card["place_queries"][0])])
            fact = visit_facts.visit_facts_for_grounding(request, grounding)[0]
            description = source_planning._visit_description(fact, request)
            with self.subTest(key=key):
                self.assertIsNone(_CHILD_WORD_RE.search(description))
                self.assertIn(f"{fact.group_cost:g} ₽", description)
                self.assertIn(fact.duration_source, description)
                self.assertIn(fact.price_source, description)

    def test_adapter_rejects_non_ready_or_partial_days(self):
        request, grounding, _, _, skeleton, _ = self.scenario(2)
        for result in (SkeletonResult("needs_facts"), replace(skeleton, days=skeleton.days[:1]),
                       replace(skeleton, status="search_limit")):
            with self.subTest(status=result.status), self.assertRaises(ValueError):
                source_planning.itinerary_from_skeleton(request, grounding, result)

    def test_source_schedule_marker_survives_model_json_roundtrip_without_db(self):
        *_, itinerary = self.scenario()
        self.assertTrue(itinerary.source_schedule)
        restored = Itinerary.model_validate_json(itinerary.model_dump_json())
        self.assertTrue(restored.source_schedule)
        self.assertEqual(restored.model_dump(), itinerary.model_dump())

    def test_unknown_catalogue_retains_legacy_without_routing(self):
        request, grounding, *_ = self.scenario()
        with patch.object(source_planning, "load_visit_cards", return_value=[]), \
                patch.object(source_planning, "build_travel_graph") as routing, \
                patch.object(source_planning, "visit_facts_for_grounding") as facts:
            self.assertIsNone(source_planning.prepare_source_plan(request, grounding))
        routing.assert_not_called()
        facts.assert_not_called()

    def test_known_exact_source_identity_without_usable_facts_stops_not_paid_retry(self):
        request, grounding, facts, *_ = self.scenario()
        cards = [dict(id=fact.visit_id, place_queries=[fact.place_query]) for fact in facts]
        with patch.object(source_planning, "load_visit_cards", return_value=cards), \
                patch.object(source_planning, "visit_facts_for_grounding", return_value=[]), \
                patch.object(source_planning, "build_travel_graph") as routing:
            result = source_planning.prepare_source_plan(request, grounding)
        self.assertIsNotNone(result, "Returning None enables paid scheduling despite known missing/stale source facts")
        self.assertTrue(result.problems)
        self.assertTrue(result.problems[0].startswith("SOURCE_PLAN_UNAVAILABLE: SOURCE_VISIT_DATA_INCOMPLETE requestedDays=1 usableCards=0"))
        self.assertEqual(result.coverage.unique_candidates, 0)
        self.assertEqual(result.itinerary.days, [])
        routing.assert_not_called()

    def test_failed_local_plan_never_publishes_partial_itinerary(self):
        request, grounding, facts, *_ = self.scenario()
        cards = [dict(id=fact.visit_id, place_queries=[fact.place_query]) for fact in facts]
        with patch.object(source_planning, "load_visit_cards", return_value=cards), \
                patch.object(source_planning, "visit_facts_for_grounding", return_value=facts), \
                patch.object(source_planning, "build_travel_graph", return_value=([], ["no connected roads"])):
            result = source_planning.prepare_source_plan(request, grounding)
        self.assertTrue(result.problems)
        self.assertNotEqual(result.status, "ready")
        self.assertEqual((result.itinerary.days, result.depth_by_day), ([], {}))

    def test_ready_local_plan_and_credits_are_passed_through_without_ai(self):
        request, grounding, facts, graph, skeleton, _ = self.scenario()
        cards = [dict(id=fact.visit_id, place_queries=[fact.place_query]) for fact in facts]
        with patch.object(source_planning, "load_visit_cards", return_value=cards), \
                patch.object(source_planning, "visit_facts_for_grounding", return_value=facts), \
                patch.object(source_planning, "build_travel_graph", return_value=(graph, [])):
            result = source_planning.prepare_source_plan(request, grounding)
        self.assertEqual((result.status, result.problems), ("ready", []))
        self.assertEqual(result.depth_by_day, {day.day: day.depth_credit_min for day in skeleton.days})

    def test_selected_route_guard_failure_atomically_refuses_generation_and_edit(self):
        request, grounding, facts, graph, _, itinerary = self.scenario()
        cards = [dict(id=fact.visit_id, place_queries=[fact.place_query]) for fact in facts]
        with patch.object(source_planning, "load_visit_cards", return_value=cards), \
                patch.object(source_planning, "visit_facts_for_grounding", return_value=facts), \
                patch.object(source_planning, "build_travel_graph", return_value=(graph, [])), \
                patch.object(source_planning, "verify_selected_routes", return_value=["Unknown or ferry step"]) as selected:
            result = source_planning.prepare_source_plan(request, grounding)
        selected.assert_called_once()
        self.assertTrue(result.problems)
        self.assertIn("SOURCE_ROUTES_UNCONFIRMED", result.problems[0])
        self.assertEqual((result.itinerary.days, result.depth_by_day), ([], {}))
        before = itinerary.model_dump()
        with patch.object(source_planning, "verify_route_edges", return_value=["Unconfirmed route"]) as edges:
            problems, credits = self.validate(request, itinerary, facts, graph)
        edges.assert_called_once()
        self.assertTrue(problems)
        self.assertEqual(credits, {})
        self.assertEqual(itinerary.model_dump(), before)

    def test_source_skeleton_failure_statuses_keep_distinct_safe_markers(self):
        from generation_feedback import generation_failure_reason
        request, grounding, facts, graph, *_ = self.scenario()
        cards = [dict(id=fact.visit_id, place_queries=[fact.place_query]) for fact in facts]
        cases = [
            ("needs_facts", ("visit:private_id:date_windows",), "SOURCE_VISIT_DATA_INCOMPLETE"),
            ("needs_facts", ("travel:private_start->private_end",), "SOURCE_ROUTES_UNCONFIRMED"),
            ("needs_facts", (), "SOURCE_VISIT_DATA_INCOMPLETE"),
            ("insufficient_content", (), "SOURCE_CONSTRAINTS_INCOMPATIBLE"),
            ("search_limit", ("visit:private_id:date_windows", "travel:private_start->private_end"), "SOURCE_SEARCH_LIMIT"),
            ("future_status", (), ""),
        ]
        for status, missing, marker in cases:
            with self.subTest(status=status, missing=missing), ExitStack() as stack:
                stack.enter_context(patch.object(source_planning, "load_visit_cards", return_value=cards))
                stack.enter_context(patch.object(source_planning, "visit_facts_for_grounding", return_value=facts))
                stack.enter_context(patch.object(source_planning, "build_travel_graph", return_value=(graph, ["place_query SECRET budget warning"])))
                stack.enter_context(patch.object(source_planning, "build_regional_skeleton", return_value=SkeletonResult(status, missing_facts=missing)))
                verify = stack.enter_context(patch.object(source_planning, "verify_selected_routes"))
                result = source_planning.prepare_source_plan(request, grounding)
            self.assertEqual(result.status, status)
            self.assertEqual((result.itinerary.days, result.depth_by_day), ([], {}))
            self.assertTrue(result.problems[0].startswith("SOURCE_PLAN_UNAVAILABLE:"))
            self.assertIn(marker, result.problems[0])
            verify.assert_not_called()
            reason = generation_failure_reason(result.problems)
            self.assertNotIn("SECRET", reason)
            self.assertNotIn("private_", reason)
            self.assertNotIn("меньше дней", reason)
            if not marker:
                self.assertIn("Причину отказа не удалось уточнить", reason)

    def test_short_source_coverage_can_try_free_days_but_never_invents_missing_roads(self):
        request, grounding, facts, *_ = self.scenario(2)
        cards = [dict(id=fact.visit_id, place_queries=[fact.place_query]) for fact in facts]
        with patch.object(source_planning, "load_visit_cards", return_value=cards), \
                patch.object(source_planning, "visit_facts_for_grounding", return_value=facts[:1]), \
                patch.object(source_planning, "build_travel_graph", return_value=([], [])) as routing:
            result = source_planning.prepare_source_plan(request, grounding)
        self.assertTrue(result.problems[0].startswith("SOURCE_PLAN_UNAVAILABLE: SOURCE_VISIT_DATA_INCOMPLETE requestedDays=2 usableCards=1"))
        self.assertEqual(result.coverage.unique_candidates, 1)
        self.assertEqual(result.itinerary.days, [])
        routing.assert_called_once()

    def test_source_failure_coverage_formatter_never_coerces_unbounded_or_bool_values(self):
        from types import SimpleNamespace
        for value in (True, False, -1, 101, 2.0, "7"):
            result = source_planning._source_failure("SOURCE_SEARCH_LIMIT", SimpleNamespace(days_count=value), value, "search_limit")
            self.assertEqual(result.problems[0], "SOURCE_PLAN_UNAVAILABLE: SOURCE_SEARCH_LIMIT")
        result = source_planning._source_failure("SOURCE_SEARCH_LIMIT", SimpleNamespace(days_count=7), 6, "search_limit")
        self.assertEqual(result.problems[0], "SOURCE_PLAN_UNAVAILABLE: SOURCE_SEARCH_LIMIT requestedDays=7 usableCards=6")

    def test_planner_direct_source_success_or_failure_never_requests_ai_schedule(self):
        import planner
        from candidate_grounding import GroundingResult
        import test_regional_generation_repair as regional
        request, grounding, _, _, skeleton, itinerary = self.scenario()
        grounded = GroundingResult(base_facts=grounding.base_facts, confirmed_hints=grounding.confirmed_hints,
                                   completion_calls=1, input_tokens=11, output_tokens=13)
        for success in (True, False):
            plan = source_planning.SourcePlan(itinerary if success else Itinerary(title="Отказ", summary="", days=[]),
                                              [] if success else ["SOURCE_PLAN_UNAVAILABLE: missing facts"],
                                              {1: 0}, "ready" if success else "needs_facts")
            with self.subTest(success=success), ExitStack() as stack:
                for name, value in (("resolve_region", regional._scope()), ("get_weather", None),
                                    ("find_city", None), ("cached_place_hints", []), ("curated_place_hints", []),
                                    ("curated_blocked_place_terms", []), ("ground_region_candidates", grounded),
                                    ("prepare_source_plan", plan)):
                    stack.enter_context(patch.object(planner, name, return_value=value))
                generate = stack.enter_context(patch.object(planner, "generate_json", side_effect=AssertionError("No AI schedule")))
                verify = stack.enter_context(patch.object(planner, "_map_verification_problems", return_value=[]))
                result = planner.generate_itinerary(request, ground_candidates=True, max_attempts=3)
                self.assertEqual((result.attempts, result.prompt_tokens, result.completion_tokens), (1, 11, 13))
                self.assertEqual(bool(result.problems), not success)
                generate.assert_not_called()
                self.assertEqual(verify.call_count, 1 if success else 0)
                if success:
                    self.assertTrue(result.itinerary.source_schedule)
                else:
                    self.assertEqual(result.itinerary.days, [])

    def test_planner_edit_keeps_source_marker_and_runs_source_validation(self):
        import planner
        from test_regional_edit_safety import RegionalEditSafetyTests, _source
        source = _source()
        source.source_schedule = True
        changed = source.days[0].model_copy(deep=True)
        changed.theme = "Обновлённая тема дня"
        with patch.object(planner, "validate_source_edit", return_value=([], {1: 0, 2: 0})) as validate:
            result, _, _ = RegionalEditSafetyTests()._run(source, changed, 1)
        self.assertEqual(result.problems, [])
        self.assertTrue(result.itinerary.source_schedule)
        validate.assert_called_once()

    def test_unchanged_valid_edit_keeps_matching_costs_windows_and_graph(self):
        request, _, facts, graph, _, itinerary = self.scenario()
        problems, credits = self.validate(request, itinerary, facts, graph)
        self.assertEqual((problems, credits), ([], {1: 0}))

    def test_edit_cannot_change_duration_window_or_indoor_fact(self):
        request, _, facts, graph, _, itinerary = self.scenario()
        for field, value in (("duration_min", 121), ("time", "20:30"), ("indoor", False)):
            edited = deepcopy(itinerary)
            item = next(item for item in edited.days[0].items if item.kind == "attraction")
            setattr(item, field, value)
            with self.subTest(field=field):
                problems, credits = self.validate(request, edited, facts, graph)
                self.assertTrue(problems)
                self.assertEqual(credits, {})

    def test_edit_cannot_move_ticket_cost_to_meal_while_day_total_stays_equal(self):
        request, _, facts, graph, _, itinerary = self.scenario()
        attraction = next(item for item in itinerary.days[0].items if item.kind == "attraction")
        meal = next(item for item in itinerary.days[0].items if item.kind == "meal")
        meal.cost += attraction.cost
        attraction.cost = 0
        problems, credits = self.validate(request, itinerary, facts, graph)
        self.assertTrue(problems)
        self.assertEqual(credits, {})

    def test_edit_cannot_erase_source_restrictions_and_replace_with_false_price_claim(self):
        request, _, facts, graph, _, itinerary = self.scenario()
        attraction = next(item for item in itinerary.days[0].items if item.kind == "attraction")
        attraction.description = "Вход бесплатный, можно прийти без записи; время гарантировано."
        problems, _ = self.validate(request, itinerary, facts, graph)
        self.assertTrue(problems or attraction.description != "Вход бесплатный, можно прийти без записи; время гарантировано.")

    def test_successful_price_change_does_not_leave_old_numeric_summary(self):
        request, _, facts, graph, _, itinerary = self.scenario()
        original_summary = itinerary.summary
        attraction = next(item for item in itinerary.days[0].items if item.kind == "attraction")
        facts = [replace(fact, group_cost=400) if fact.place_query == attraction.place_query else fact for fact in facts]
        attraction.cost = 400
        problems, _ = self.validate(request, itinerary, facts, graph)
        self.assertEqual(problems, [])
        self.assertTrue(itinerary.summary != original_summary or "Программа:" not in itinerary.summary,
                        "Source edit must refresh or omit stale numeric source-plan totals")

    def test_edit_cannot_bypass_source_child_eligibility_minimum_age_or_interests(self):
        request, _, facts, graph, _, itinerary = self.scenario()
        request = request.model_copy(update={"children_ages": [6]})
        meal = next(item for item in itinerary.days[0].items if item.kind == "meal")
        meal.cost += 400
        next(item for item in itinerary.days[0].items if item.kind == "rest").duration_min = 60
        from rules import check_rules
        self.assertEqual(check_rules(itinerary, request), [])
        for change in (dict(children_allowed=False), dict(children_allowed=None), dict(minimum_age=18),
                       dict(interest_categories=("shopping_malls",))):
            changed_facts = [replace(facts[0], **change), *facts[1:]]
            with self.subTest(change=change):
                problems, credits = self.validate(request, deepcopy(itinerary), changed_facts, graph)
                self.assertTrue(problems)
                self.assertEqual(credits, {})

    def test_missing_return_edge_or_travel_budget_refuses_all_credits(self):
        request, _, facts, graph, _, itinerary = self.scenario()
        last = [item for item in itinerary.days[0].items if item.kind == "attraction"][-1]
        visit_id = next(fact.visit_id for fact in facts if fact.place_query == last.place_query)
        problems, credits = self.validate(request, itinerary, facts,
                                          [edge for edge in graph if (edge.from_key, edge.to_key) != (visit_id, "base_1")])
        self.assertTrue(problems)
        self.assertEqual(credits, {})
        meal = next(item for item in itinerary.days[0].items if item.kind == "meal")
        meal.cost = 1400
        problems, credits = self.validate(request, itinerary, facts, graph)
        self.assertTrue(problems)
        self.assertEqual(credits, {})

    def test_interbase_transfer_must_not_overlap_meal_or_rest(self):
        request = request_for(2)
        grounding, facts, _ = fixture(request, 2, 240)
        facts = [replace(fact, date_windows=(windows_for(request)[i],)) for i, fact in enumerate(facts)]
        graph = [TravelFact(origin, destination, 10, "car", "routing", source="router",
                            group_cost=0, price_source="prepaid") for i in (1, 2)
                 for origin, destination in ((f"base_{i}", f"visit_{i}"), (f"visit_{i}", f"base_{i}"))]
        graph.append(TravelFact("base_1", "base_2", 70, "car", "routing", source="router",
                                group_cost=0, price_source="prepaid", depth_credit_min=60))
        days = []
        for i, fact in enumerate(facts, 1):
            days.append(ItineraryDay(day=i, date=fact.date_windows[0].date, theme="Визит", base_area=f"Город {i}, Россия", items=[
                ItineraryItem(time="09:30", kind="attraction", title=f"Музей {i}", place_query=fact.place_query,
                              description=source_planning._visit_description(fact), duration_min=240, cost=200,
                              indoor=True, verified=True, lat=50 + i * .01, lon=100),
                ItineraryItem(time="08:00", kind="meal", title="Обед", description="Обед", duration_min=60, cost=1400),
                ItineraryItem(time="14:00", kind="rest", title="Отдых", description="Пауза", duration_min=15, cost=0),
            ]))
        itinerary = Itinerary(title="План", summary="План", days=days, source_schedule=True)
        problems, credits = self.validate(request, itinerary, facts, graph)
        self.assertTrue(problems, "08:00–09:10 transfer overlaps 08:00–09:00 meal")
        self.assertEqual(credits, {})


if __name__ == "__main__":
    unittest.main()
