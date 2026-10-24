"""The new pipeline never pays an AI to repair a server-owned timetable."""

from contextlib import ExitStack
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import city_planner
import city_rollout
import planner
from city_candidates import CityCandidateResult
from schemas import Itinerary
from test_city_candidate_planner import default_candidates, request


class DeterministicIntegrationTests(unittest.TestCase):
    def run_plan(self, *, builder=None, validators=None, trip_request=None, selection_response=None,
                 candidate_result=None, repair_response=None, max_attempts=3, discovery_calls=1,
                 memory_context=""):
        candidates = default_candidates()
        for item in candidates:
            item.record["category"] = "tourism"
        center = {"lat": 55.0, "lon": 83.0, "name": "Тестгород", "country_code": "RU"}

        def ground(req, resolved, *, completion, **kwargs):
            for _ in range(discovery_calls):
                completion([{"role": "user", "content": "Synthetic names only"}])
            return candidate_result if candidate_result is not None else CityCandidateResult(candidates=candidates)

        with ExitStack() as stack:
            stack.enter_context(patch("httpx.get", side_effect=AssertionError("No network")))
            stack.enter_context(patch("places.geocode", side_effect=AssertionError("No map query")))
            stack.enter_context(patch.object(city_planner, "city_deterministic_enabled", return_value=True))
            stack.enter_context(patch.object(city_planner, "find_city", return_value=center))
            stack.enter_context(patch.object(city_planner, "get_weather", return_value=None))
            stack.enter_context(patch.object(city_planner, "ground_city_candidates", side_effect=ground))
            stack.enter_context(patch.object(planner, "_load_blocked_places", return_value=[]))
            if builder is not None:
                stack.enter_context(patch("city_deterministic.build_deterministic_city_itinerary", return_value=builder))
            if validators:
                for name, value in validators.items():
                    stack.enter_context(patch.object(city_planner, name, return_value=value))
            completion = stack.enter_context(patch.object(planner, "generate_json", side_effect=[
                *[SimpleNamespace(text='{"titles":[]}', input_tokens=11, output_tokens=7)
                  for _ in range(discovery_calls)],
                SimpleNamespace(text=selection_response, input_tokens=11, output_tokens=7)
                if selection_response is not None else AssertionError("No second AI call allowed"),
                SimpleNamespace(text=repair_response, input_tokens=11, output_tokens=7)
                if repair_response is not None else AssertionError("No depth review call allowed"),
                AssertionError("No additional AI call allowed"),
            ]))
            result = planner.generate_itinerary(trip_request or request(), max_attempts=max_attempts,
                                                city_candidates_first=True, memory_context=memory_context)
        return result, completion

    def test_schedule_and_budget_are_server_owned_with_only_discovery_completion(self):
        result, completion = self.run_plan()
        self.assertEqual(result.problems, [])
        self.assertEqual(result.generation_status, "success")
        self.assertEqual((completion.call_count, result.attempts, result.total_tokens), (1, 1, 18))
        self.assertTrue(result.itinerary.city_schedule)
        self.assertFalse(result.itinerary.source_schedule)
        self.assertIsNotNone(result.city_map_context)
        visits = [item for day in result.itinerary.days for item in day.items if item.kind == "attraction"]
        self.assertEqual([item.duration_min for item in visits], [90, 90])
        self.assertTrue(all(item.verified for item in visits))
        self.assertTrue(all(not item.indoor for item in visits))

    def test_insufficient_data_never_falls_back_to_model_schedule(self):
        built = SimpleNamespace(itinerary=None, problems=["Недостаточно подтверждённых данных"],
                                status="insufficient_data", coverage={"eligible": 1})
        result, completion = self.run_plan(builder=built)
        self.assertEqual(result.generation_status, "insufficient_data")
        self.assertEqual(result.generation_coverage["eligible"], 1)
        self.assertEqual(result.itinerary.days, [])
        completion.assert_called_once()

    def test_final_boundary_still_rejects_invalid_builder_output(self):
        built = SimpleNamespace(itinerary=Itinerary(title="Invalid test", summary="", days=[], city_schedule=True),
                                problems=[], status="success", coverage={})
        result, completion = self.run_plan(builder=built)
        self.assertTrue(result.problems)
        self.assertEqual(result.generation_status, "validation_error")
        self.assertIsNone(result.city_map_context)
        completion.assert_called_once()

    def test_all_final_validator_findings_reject_without_paid_repair(self):
        names = ("check_rules", "verify_itinerary", "check_city_radius", "check_verified_place_duplicates",
                 "check_program_depth", "check_city_quality", "check_city_visit_estimates",
                 "check_city_visit_conditions")
        valid, _ = self.run_plan()
        self.assertFalse(valid.problems)
        built = SimpleNamespace(itinerary=valid.itinerary, problems=[], status="success", coverage={})
        for name in names:
            with self.subTest(validator=name):
                result, completion = self.run_plan(builder=built, validators={name: ["Synthetic guard finding"]})
                self.assertEqual(result.generation_status, "validation_error")
                self.assertIsNone(result.city_map_context)
                completion.assert_called_once()

    def test_new_switch_is_separate_strict_and_off_without_configuration(self):
        for value, expected in ((None, False), ("", False), ("1", True), ("true", True),
                                ("TRUE", False), ("yes", False), (" true ", False)):
            with self.subTest(value=value), patch.object(city_rollout.os, "getenv", return_value=value):
                self.assertIs(city_rollout.city_deterministic_enabled(), expected)

    def test_semantic_selection_uses_second_call_but_never_model_timetable(self):
        value = request()
        value.interest_categories = ["museums"]
        value.trip_goal = "Подробное знакомство с историей"
        result, completion = self.run_plan(trip_request=value, selection_response=
                                          '{"status":"selected","candidate_ids":["osm-node-102","osm-node-101"]}')
        self.assertEqual(result.problems, [])
        self.assertEqual(result.generation_status, "success")
        self.assertEqual((completion.call_count, result.attempts, result.total_tokens), (2, 2, 36))
        self.assertEqual(set(completion.call_args.kwargs["json_schema"]["properties"]), {"status", "candidate_ids"})
        self.assertEqual(result.generation_coverage["preference_candidates"], 2)

    def depth_fixture(self):
        """Real failure's five selected types; ample independently named reserve."""
        from test_city_deterministic import candidates, request as city_request
        pool = candidates(24)
        for index in range(14, 18):
            pool[index].record.update(category="amenity", type="theatre")
        for index in range(18, 22):
            pool[index].record.update(category="tourism", type="zoo")
        for index in (22, 23):
            pool[index].record.update(category="shop", type="mall")
        value = city_request(7, start_time="11:00", end_time="20:00",
                             trip_goal="Познакомиться с городом",
                             interest_categories=["museums", "theatre", "zoos", "shopping_malls"])
        first = [pool[index].candidate_id for index in (0, 14, 18, 22, 23)]
        # Three low-value alternatives of the represented zoo category remain
        # rejected. Depth review cannot blindly restore the whole map pool.
        curated = [item.candidate_id for index, item in enumerate(pool) if index not in (19, 20, 21)]
        return value, pool, first, curated

    def test_rich_pool_short_selection_gets_one_depth_review_within_three_calls(self):
        value, pool, first, curated = self.depth_fixture()
        value.trip_goal = "Подробное знакомство с историей и культурой"
        result, completion = self.run_plan(trip_request=value,
            memory_context="Предпочитаю разнообразие",
            candidate_result=CityCandidateResult(candidates=pool),
            selection_response=json.dumps({"status": "selected", "candidate_ids": first}),
            repair_response=json.dumps({"status": "selected", "candidate_ids": curated}))
        self.assertFalse(result.problems)
        self.assertEqual((completion.call_count, result.attempts), (3, 3))
        report = result.generation_coverage
        self.assertEqual((report["quality_status"], report["program_days"]), ("partial", 7))
        self.assertIn("day_program_light", report["quality_reasons"])
        self.assertTrue(report["selection_repair_attempted"])
        self.assertEqual(report["selection_repair_status"], "improved")
        self.assertTrue(report["selection_depth_shortfall"])
        self.assertEqual(report["selection_calls"], 2)
        self.assertEqual(report["preference_candidates"], 21)
        used = {item.place_query for day in result.itinerary.days for item in day.items if item.kind == "attraction"}
        self.assertFalse(used & {pool[index].place_query for index in (19, 20, 21)})
        initial = json.loads(completion.call_args_list[1].args[0][1]["content"])
        review = json.loads(completion.call_args_list[2].args[0][1]["content"])
        self.assertEqual(initial["planning_depth"]["required_planning_minutes"], 1890)
        self.assertEqual(review["depth_review"]["selected_planning_minutes"], 90 + 45 + 120 + 90 + 90)
        self.assertEqual(review["depth_review"]["selected_candidate_ids"], first)
        self.assertEqual(set(completion.call_args.kwargs["json_schema"]["properties"]), {"status", "candidate_ids"})

    def test_depth_review_never_exceeds_remaining_total_completion_budget(self):
        value, pool, first, _ = self.depth_fixture()
        value.trip_goal = "Подробное знакомство с историей и культурой"
        for budget, discovery_calls in ((2, 1), (3, 2)):
            with self.subTest(budget=budget):
                result, completion = self.run_plan(trip_request=value, max_attempts=budget,
                    memory_context="Предпочитаю разнообразие",
                    discovery_calls=discovery_calls, candidate_result=CityCandidateResult(candidates=pool),
                    selection_response=json.dumps({"status": "selected", "candidate_ids": first}))
                self.assertFalse(result.problems)
                self.assertEqual((completion.call_count, result.attempts), (budget, budget))
                self.assertEqual(result.generation_coverage["quality_status"], "partial")
                self.assertFalse(result.generation_coverage["selection_repair_attempted"])
                self.assertIn("selection_depth_shortfall", result.generation_coverage["quality_reasons"])
                self.assertNotIn("insufficient_confirmed_places", result.generation_coverage["quality_reasons"])

    def test_failed_or_unchanged_depth_review_keeps_valid_partial_subset_without_fourth_call(self):
        value, pool, first, _ = self.depth_fixture()
        value.trip_goal = "Подробное знакомство с историей и культурой"
        for identifiers, expected in ((["osm-node-invented"], "failed"), (first, "unchanged")):
            with self.subTest(review=expected):
                result, completion = self.run_plan(trip_request=value,
                    memory_context="Предпочитаю разнообразие",
                    candidate_result=CityCandidateResult(candidates=pool),
                    selection_response=json.dumps({"status": "selected", "candidate_ids": first}),
                    repair_response=json.dumps({"status": "selected", "candidate_ids": identifiers}))
                self.assertFalse(result.problems)
                self.assertEqual((completion.call_count, result.attempts), (3, 3))
                self.assertEqual(result.generation_coverage["quality_status"], "partial")
                self.assertEqual(result.generation_coverage["preference_candidates"], 5)
                self.assertEqual(result.generation_coverage["selection_repair_status"], expected)
                self.assertEqual(result.generation_coverage["program_days"], 2)
                self.assertEqual(result.generation_coverage["selected_unique_visits"], 5)
                self.assertNotIn("insufficient_confirmed_places", result.generation_coverage["quality_reasons"])
                if expected == "failed":
                    self.assertIn("selection_repair_failed", result.generation_coverage["quality_reasons"])

    def test_free_text_restriction_does_not_trigger_depth_pressure_or_restore_rejected_places(self):
        value, pool, first, _ = self.depth_fixture()
        value.wishes = "Без пони-ферм и маленьких площадок с животными"
        result, completion = self.run_plan(trip_request=value,
            candidate_result=CityCandidateResult(candidates=pool),
            selection_response=json.dumps({"status": "selected", "candidate_ids": first}))
        self.assertFalse(result.problems)
        self.assertEqual(completion.call_count, 2)
        self.assertFalse(result.generation_coverage["selection_repair_attempted"])
        self.assertEqual(result.generation_coverage["preference_candidates"], 5)

    def test_selection_refusal_does_not_silently_drop_preferences(self):
        value = request()
        value.wishes = "Посещения только с подтверждённой бесплатностью"
        result, completion = self.run_plan(trip_request=value, selection_response=
                                          '{"status":"unsupported_preferences","candidate_ids":[]}')
        self.assertEqual(result.generation_status, "unsupported_preferences")
        self.assertTrue(result.problems)
        self.assertFalse(result.itinerary.days)
        self.assertEqual(completion.call_count, 2)

    def test_proximity_preflight_does_not_pay_for_known_missing_anchor(self):
        value = request()
        value.location_preference, value.stay_area = "nearby", "Северный район"
        result, completion = self.run_plan(trip_request=value)
        self.assertEqual(result.generation_status, "unsupported_preferences")
        self.assertEqual(result.attempts, 0)
        completion.assert_not_called()

    def test_requested_shopping_reaches_final_calendar_not_only_ai_selection(self):
        from test_city_deterministic import DeterministicCityTests
        pool = DeterministicCityTests.shopping_pool()
        value = request()
        value.interest_categories = ["museums", "shopping_malls"]
        result, completion = self.run_plan(trip_request=value,
            candidate_result=CityCandidateResult(candidates=pool), selection_response=json.dumps({
                "status": "selected", "candidate_ids": [item.candidate_id for item in pool]}))
        self.assertEqual(result.problems, [])
        self.assertEqual(result.generation_status, "success")
        self.assertTrue(result.generation_coverage["requested_category_satisfied"])
        self.assertTrue(any(item.place_query == pool[-1].place_query
                            for day in result.itinerary.days for item in day.items))
        self.assertEqual(completion.call_count, 1)
        self.assertEqual(result.generation_coverage["selection_mode"], "structured_preferences")

    def test_explicit_shopping_exclusion_beats_checkbox_in_final_schedule(self):
        from test_city_deterministic import DeterministicCityTests
        pool = DeterministicCityTests.shopping_pool()
        value = request()
        value.interest_categories = ["museums", "shopping_malls"]
        value.wishes = "Без торговых центров"
        result, completion = self.run_plan(trip_request=value,
            candidate_result=CityCandidateResult(candidates=pool), selection_response=json.dumps({
                "status": "selected", "candidate_ids": [item.candidate_id for item in pool[:-1]]}))
        self.assertEqual(result.problems, [])
        self.assertEqual(result.generation_coverage["requested_category_candidates"], 0)
        self.assertFalse(any(item.place_query == pool[-1].place_query
                             for day in result.itinerary.days for item in day.items))
        self.assertEqual(completion.call_count, 2)

    def test_missing_confirmed_mall_is_disclosed_without_rejecting_valid_trip(self):
        value = request()
        value.interest_categories = ["museums", "shopping_malls"]
        result, completion = self.run_plan(trip_request=value, selection_response=json.dumps({
            "status": "selected", "candidate_ids": [item.candidate_id for item in default_candidates()]}))
        self.assertEqual(result.problems, [])
        self.assertEqual(result.generation_status, "success")
        self.assertEqual(result.generation_coverage["quality_status"], "partial")
        self.assertIn("requested_interest_missing", result.generation_coverage["quality_reasons"])
        self.assertEqual(result.generation_coverage["interest_coverage"]["shopping_malls"],
                         {"status": "not_confirmed", "confirmed_count": 0, "selected_count": 0})
        self.assertFalse(any("Не удалось подтвердить подходящий торговый центр" in tip
                             for tip in result.itinerary.tips))
        self.assertEqual(completion.call_count, 1)

    def test_discovery_diagnostics_keep_only_safe_counters_and_static_user_warning(self):
        marker = "synthetic-private-content-not-for-report"
        pool = CityCandidateResult(candidates=default_candidates(), map_requests=12,
            rejected=[{"title": marker, "reason": "name_mismatch", "detail": marker},
                      {"reason": marker}, {"reason": {"text": marker}}],
            warnings=[marker], diagnostics={"attempted_names": 14, "name_limit": 28,
                "usable_limit": 18, "limits_reached": ["name_limit", marker], "wishes": marker})
        result, _ = self.run_plan(candidate_result=pool)
        self.assertFalse(result.problems)
        self.assertEqual(result.generation_coverage["discovery"], {
            "confirmed_candidates": 2, "rejected_count": 3,
            "rejection_reasons": {"name_mismatch": 1, "other": 2}, "warning_count": 1,
            "map_requests": 12, "attempted_names": 14, "name_limit": 28, "usable_limit": 18,
            "limits_reached": ["name_limit"]})
        self.assertIn(city_planner.DISCOVERY_WARNING, result.itinerary.tips)
        self.assertNotIn(marker, json.dumps(result.generation_coverage, ensure_ascii=False))
        self.assertNotIn(marker, result.itinerary.model_dump_json())

    def test_invalid_diagnostic_types_cannot_leak_details_or_invent_counts(self):
        pool = SimpleNamespace(candidates=[], rejected="not a list", warnings={"private": "text"},
                               map_requests=True, diagnostics={"attempted_names": True,
                                   "name_limit": -1, "usable_limit": "secret", "limits_reached": [{}, True]})
        self.assertEqual(city_planner._discovery_diagnostics(pool), {
            "confirmed_candidates": 0, "rejected_count": 0, "rejection_reasons": {},
            "warning_count": 0, "map_requests": 0, "limits_reached": []})

    def test_real_shape_three_museum_pool_has_partial_quality_and_all_three_visits(self):
        from test_city_deterministic import candidates, request as city_request
        pool = candidates(3)
        value = city_request(7, start_time="11:00", end_time="20:00", budget=100000,
                             travel_cost=10000, lodging_cost=35000,
                             interest_categories=["museums", "shopping_malls", "zoos"])
        result, completion = self.run_plan(trip_request=value,
            candidate_result=CityCandidateResult(candidates=pool), selection_response=json.dumps({
                "status": "selected", "candidate_ids": [item.candidate_id for item in pool]}))
        self.assertFalse(result.problems)
        self.assertEqual(result.generation_status, "success")
        self.assertEqual(result.generation_coverage["quality_status"], "partial")
        self.assertEqual(result.generation_coverage["selected_unique_visits"], 3)
        self.assertEqual(result.generation_coverage["program_days"], 1)
        self.assertIn("insufficient_confirmed_places", result.generation_coverage["quality_reasons"])
        self.assertEqual(result.generation_coverage["interest_coverage"]["museums"],
                         {"status": "covered", "confirmed_count": 3, "selected_count": 3})
        for key in ("zoos", "shopping_malls"):
            self.assertEqual(result.generation_coverage["interest_coverage"][key]["status"], "not_confirmed")
        self.assertEqual(completion.call_count, 1)

    def test_selector_omission_of_known_interest_is_visible_as_partial(self):
        from test_city_deterministic import candidates
        pool = candidates(3)
        pool[-1].title = "Театр Северный"
        pool[-1].place_query = "Театр Северный, Тестгород, Россия"
        pool[-1].record.update(name=pool[-1].place_query, category="amenity", type="theatre")
        value = request()
        value.interest_categories = ["museums", "theatre"]
        value.wishes = "Не включать театр"
        result, _ = self.run_plan(trip_request=value,
            candidate_result=CityCandidateResult(candidates=pool), selection_response=json.dumps({
                "status": "selected", "candidate_ids": [item.candidate_id for item in pool[:2]]}))
        self.assertFalse(result.problems)
        self.assertEqual(result.generation_coverage["quality_status"], "partial")
        self.assertEqual(result.generation_coverage["interest_coverage"]["theatre"],
                         {"status": "not_scheduled", "confirmed_count": 1, "selected_count": 0})
        self.assertIn("requested_interest_missing", result.generation_coverage["quality_reasons"])
        self.assertFalse(any("не отражены выбранные интересы" in tip for tip in result.itinerary.tips))

    def test_default_form_preserves_all_checked_categories_without_semantic_selection(self):
        from test_city_deterministic import candidates
        pool = candidates(4)
        for item, title, category, kind in zip(pool[1:],
                ("Зоопарк Северный", "Театр Северный", "Торговый центр Северный"),
                ("tourism", "amenity", "shop"), ("zoo", "theatre", "mall")):
            item.title = title
            item.place_query = f"{title}, Тестгород, Россия"
            item.record.update(name=item.place_query, category=category, type=kind)
        value = request()
        value.trip_goal = "Познакомиться с городом"
        value.interest_categories = ["museums", "zoos", "theatre", "shopping_malls"]
        result, completion = self.run_plan(trip_request=value,
            candidate_result=CityCandidateResult(candidates=pool), selection_response=json.dumps({
                "status": "selected", "candidate_ids": [pool[0].candidate_id]}))
        self.assertFalse(result.problems)
        self.assertEqual(result.generation_coverage["quality_status"], "partial")
        self.assertEqual(result.generation_coverage["quality_reasons"], ["day_window_unbalanced"])
        self.assertEqual([item.duration_min for day in result.itinerary.days for item in day.items
                          if item.place_query == pool[1].place_query], [120])
        self.assertEqual(result.generation_coverage["selected_unique_visits"], 4)
        self.assertTrue(all(item["status"] == "covered"
                            for item in result.generation_coverage["interest_coverage"].values()))
        self.assertEqual(completion.call_count, 1)
        self.assertEqual(result.generation_coverage["selection_calls"], 0)

    def test_final_schedule_preserves_ai_curation_when_zoo_category_is_already_covered(self):
        from test_city_deterministic import candidates
        pool = candidates(6)
        for item, title in zip(pool[2:], ("Городской зоопарк", "Пони-ферма",
                                         "Пони ферма Северная", "Пони-ферма у парка")):
            item.title = title
            item.place_query = f"{title}, Тестгород, Россия"
            item.record.update(name=item.place_query, category="tourism", type="zoo")
        value = request()
        value.wishes = "Без небольших пони-ферм"
        value.trip_goal = "Познакомиться с городом"
        value.interest_categories = ["museums", "zoos"]
        selected = [item.candidate_id for item in pool[:3]]
        result, completion = self.run_plan(trip_request=value,
            memory_context="Предпочитаю большой городской зоопарк, не небольшие фермы",
            candidate_result=CityCandidateResult(candidates=pool), selection_response=json.dumps({
                "status": "selected", "candidate_ids": selected}))
        self.assertFalse(result.problems)
        self.assertEqual(result.generation_coverage["quality_status"], "partial")
        self.assertEqual(result.generation_coverage["quality_reasons"], ["day_window_unbalanced"])
        self.assertEqual([item.duration_min for day in result.itinerary.days for item in day.items
                          if item.place_query == pool[2].place_query], [120])
        self.assertEqual(result.generation_coverage["interest_coverage"]["zoos"],
                         {"status": "covered", "confirmed_count": 4, "selected_count": 1})
        self.assertEqual({item.place_query for day in result.itinerary.days for item in day.items
                          if item.kind == "attraction"}, {item.place_query for item in pool[:3]})
        self.assertEqual(completion.call_count, 2)

    def test_qualitative_interests_are_not_promoted_to_map_facts(self):
        value = request()
        value.interest_categories = ["main_sights", "water_trips"]
        result, _ = self.run_plan(trip_request=value, selection_response=json.dumps({
            "status": "selected", "candidate_ids": [item.candidate_id for item in default_candidates()]}))
        self.assertFalse(result.problems)
        self.assertEqual(result.generation_coverage["quality_status"], "partial")
        self.assertIn("day_program_light", result.generation_coverage["quality_reasons"])
        for item in result.generation_coverage["interest_coverage"].values():
            self.assertEqual(item, {"status": "not_verifiable", "confirmed_count": 0, "selected_count": 0})

    def test_extended_discovery_diagnostics_have_closed_field_and_category_allowlists(self):
        marker = "synthetic-private-marker"
        pool = CityCandidateResult(candidates=default_candidates(), diagnostics={
            "requested_categories": ["zoos", marker], "missing_categories": ["zoos", {}],
            "category_counts": {"museums": 2, "zoos": 0, marker: 3, "theatre": True},
            "detail_rejection_reasons": {"empty_result": 2, marker: 100, "name_mismatch": False},
            "pool_target": 18, "pool_quality": "limited", "map_discovery_status": "unavailable",
            "replenishment_attempted": True, "private_text": marker, "source_alternatives_confirmed": 2})
        value = city_planner._discovery_diagnostics(pool)
        self.assertEqual(value["requested_categories"], ["zoos"])
        self.assertEqual(value["missing_categories"], ["zoos"])
        self.assertEqual(value["category_counts"], {"museums": 2, "zoos": 0})
        self.assertEqual(value["detail_rejection_reasons"], {"empty_result": 2})
        self.assertEqual(value["pool_target"], 18)
        self.assertEqual(value["pool_quality"], "limited")
        self.assertEqual(value["map_discovery_status"], "unavailable")
        self.assertTrue(value["replenishment_attempted"])
        self.assertEqual(value["source_alternatives_confirmed"], 2)
        self.assertNotIn(marker, json.dumps(value))

    def test_structured_rich_reserve_keeps_all_verified_ids_and_needs_no_selector(self):
        value, pool, _, _ = self.depth_fixture()
        # This fixture is a server-curated reserve of independent places, not
        # an invitation to re-add objects rejected by discovery quality gates.
        result, completion = self.run_plan(trip_request=value,
            candidate_result=CityCandidateResult(candidates=pool))
        self.assertFalse(result.problems)
        report = result.generation_coverage
        self.assertEqual((report["selection_mode"], report["selection_calls"]), ("structured_preferences", 0))
        self.assertEqual(report["preference_candidates"], 24)
        self.assertEqual((report["quality_status"], report["program_days"]), ("complete", 7))
        self.assertEqual(report["quality_reasons"], [])
        zoo_queries = {item.place_query for item in pool if item.record["type"] == "zoo"}
        self.assertEqual({item.duration_min for day in result.itinerary.days for item in day.items
                          if item.place_query in zoo_queries}, {120})
        self.assertLessEqual(report["states_explored"], 20000)
        self.assertFalse(report["selection_repair_attempted"])
        completion.assert_called_once()  # Discovery only.

    def test_memory_ranking_cannot_discard_rich_pool_even_when_desired_volume_is_higher(self):
        value, pool, first, _ = self.depth_fixture()
        result, completion = self.run_plan(trip_request=value, memory_context="Справочные предпочтения",
            candidate_result=CityCandidateResult(candidates=pool),
            selection_response=json.dumps({"candidate_ids": first}))
        self.assertFalse(result.problems)
        report = result.generation_coverage
        self.assertEqual((report["selection_mode"], report["memory_ranking_status"]), ("memory_ranking", "applied"))
        self.assertEqual((report["selection_calls"], report["preference_candidates"]), (1, 24))
        self.assertEqual((report["quality_status"], report["program_days"]), ("complete", 7))
        self.assertEqual(report["quality_reasons"], [])
        zoo_queries = {item.place_query for item in pool if item.record["type"] == "zoo"}
        self.assertEqual({item.duration_min for day in result.itinerary.days for item in day.items
                          if item.place_query in zoo_queries}, {120})
        self.assertFalse(report["selection_depth_shortfall"])
        self.assertFalse(report["selection_repair_attempted"])
        self.assertEqual(completion.call_count, 2)  # One discovery + one ranking; no depth review.

    def test_structured_reserve_still_enforces_known_child_and_date_restrictions(self):
        from dataclasses import replace
        import city_visit_options
        from test_city_deterministic import candidates
        pool = candidates(4)
        value = request()
        value.children_ages = [8]
        value.trip_goal = "Познакомиться с городом"
        value.interest_categories = ["museums"]
        actual = city_visit_options.visit_option
        def conditions(item, req, **kwargs):
            option = actual(item, req, **kwargs)
            if item.candidate_id == pool[2].candidate_id:
                return replace(option, children_allowed=False)
            if item.candidate_id == pool[3].candidate_id:
                return replace(option, allowed_dates=())
            return option
        with patch.object(city_visit_options, "visit_option", side_effect=conditions), \
                patch("city_preferences.visit_option", side_effect=conditions):
            result, completion = self.run_plan(trip_request=value,
                candidate_result=CityCandidateResult(candidates=pool))
        self.assertFalse(result.problems)
        used = {item.place_query for day in result.itinerary.days for item in day.items if item.kind == "attraction"}
        self.assertEqual(used, {item.place_query for item in pool[:2]})
        self.assertEqual(result.generation_coverage["preference_candidates"], 4)
        self.assertEqual(result.generation_coverage["eligible_candidates"], 2)
        completion.assert_called_once()


if __name__ == "__main__":
    unittest.main()
