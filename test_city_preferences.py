"""Semantic ID selection preserves inputs; numeric planning stays on the server."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from city_preferences import is_shopping_candidate, select_city_preferences, shopping_preference
from test_city_candidate_planner import default_candidates, request
from yandex_ai import YandexAIError


class CityPreferenceTests(unittest.TestCase):
    def setUp(self):
        self.request = request()
        self.request.wishes = "Не включать музеи, только парки."
        self.candidates = default_candidates()
        self.candidates[1].title = "Городской исторический парк"
        self.candidates[1].place_query = "Городской исторический парк, Тестгород, Россия"
        self.candidates[1].record.update(name=self.candidates[1].place_query, type="park")

    def complete(self, value):
        return Mock(return_value=SimpleNamespace(text=json.dumps(value, ensure_ascii=False)))

    def test_selector_schema_accepts_twenty_four_but_rejects_twenty_five(self):
        from test_city_deterministic import candidates
        self.request.wishes = ""
        self.request.trip_goal = "Подробное знакомство с историей"
        self.request.interest_categories = ["museums"]
        pool = candidates(24)
        identifiers = [item.candidate_id for item in pool]
        completion = self.complete({"status": "selected", "candidate_ids": identifiers})
        result = select_city_preferences(self.request, pool, memory_context="Предпочитаю разнообразие", completion=completion)
        self.assertEqual(result.status, "success")
        self.assertEqual(len(result.candidates), 24)
        self.assertEqual(completion.call_args.kwargs["json_schema"]["properties"]["candidate_ids"]["maxItems"], 24)
        completion.reset_mock()
        result = select_city_preferences(self.request, candidates(25), completion=completion)
        self.assertEqual(result.status, "insufficient_data")
        completion.assert_not_called()

    def test_selected_subset_and_priority_keep_exact_identity_not_model_facts(self):
        before = deepcopy((self.request, self.candidates))
        completion = self.complete({"status": "selected", "candidate_ids": [self.candidates[1].candidate_id]})
        result = select_city_preferences(self.request, self.candidates, completion=completion)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.candidates, [self.candidates[1]])
        self.assertEqual(result.candidate_priority, [self.candidates[1].candidate_id])
        self.assertEqual((self.request, self.candidates), before)
        completion.assert_called_once()
        data = json.loads(completion.call_args.args[0][1]["content"])
        self.assertEqual(data["request"]["wishes"], self.request.wishes)
        self.assertEqual(data["verified_candidates"][1]["map_type"], "park")
        self.assertEqual(set(completion.call_args.kwargs["json_schema"]["properties"]), {"status", "candidate_ids"})

    def test_goal_interests_and_prepared_memory_reach_selection_as_data(self):
        self.request.trip_goal = "Природа и тихие прогулки"
        self.request.interest_categories = ["parks"]
        completion = self.complete({"status": "selected", "candidate_ids": [self.candidates[1].candidate_id]})
        select_city_preferences(self.request, self.candidates, memory_context="Подготовленная память", completion=completion)
        data = json.loads(completion.call_args.args[0][1]["content"])
        self.assertEqual(data["request"], self.request.model_dump(mode="json"))
        self.assertEqual(data["prior_memory_data"], "Подготовленная память")

    def test_no_preferences_need_no_additional_completion(self):
        self.request.wishes = ""
        completion = Mock(side_effect=AssertionError("No extra AI"))
        result = select_city_preferences(self.request, self.candidates, completion=completion)
        self.assertEqual(result.candidates, self.candidates)
        self.assertEqual(result.status, "success")
        completion.assert_not_called()

    def test_structured_form_preserves_full_reserve_without_calling_ai(self):
        from test_city_deterministic import candidates, request as city_request
        pool = candidates(24)
        for goal in ("", "Познакомиться с городом", "  Познакомиться с городом.  "):
            with self.subTest(goal=goal):
                value = city_request(7, trip_goal=goal, interest_categories=["museums"])
                before = deepcopy(pool)
                completion = Mock(side_effect=AssertionError("No semantic AI for a structured form"))
                result = select_city_preferences(value, pool, memory_context=" \n", completion=completion,
                                                 allow_depth_repair=True)
                self.assertEqual(result.candidates, before)
                self.assertEqual(pool, before)
                self.assertEqual(result.diagnostics["selection_calls"], 0)
                self.assertEqual(result.diagnostics["selection_mode"], "structured_preferences")
                self.assertFalse(result.diagnostics["selection_repair_attempted"])
                completion.assert_not_called()

    def test_any_custom_goal_or_wish_requires_semantic_selection_even_with_memory(self):
        from test_city_deterministic import candidates, request as city_request
        pool = candidates(3)
        chosen = [pool[0].candidate_id]
        for goal, wishes, memory in (("Познакомиться с городом и архитектурой", "", ""),
                                    ("Познакомиться с городом", "Без долгих прогулок", ""),
                                    ("Подробное знакомство с историей", "", "Предпочитаю только этот музей")):
            with self.subTest(goal=goal, wishes=wishes, memory=memory):
                value = city_request(1, trip_goal=goal, wishes=wishes, interest_categories=["museums"])
                completion = self.complete({"status": "selected", "candidate_ids": chosen})
                result = select_city_preferences(value, pool, memory_context=memory, completion=completion)
                self.assertEqual(result.candidate_priority, chosen)
                self.assertEqual(result.diagnostics["selection_mode"], "semantic_preferences")
                completion.assert_called_once()

    def test_structured_priority_prefers_requested_types_but_does_not_promote_generic_attraction(self):
        from test_city_deterministic import candidates, request as city_request
        pool = candidates(4)
        pool[0].record.update(type="attraction")
        pool[1].record.update(category="leisure", type="park")
        pool[2].record.update(category="shop", type="mall")
        value = city_request(1, trip_goal="Познакомиться с городом",
                             interest_categories=["main_sights", "shopping_malls", "museums"])
        result = select_city_preferences(value, pool, completion=Mock(side_effect=AssertionError("No AI")))
        self.assertEqual(result.candidate_priority, [pool[i].candidate_id for i in (2, 3, 1, 0)])
        self.assertEqual(len(result.candidates), 4)  # Non-requested park is not a forbidden park.

    def test_structured_form_still_rejects_empty_duplicate_or_oversized_pool(self):
        from test_city_deterministic import candidates, request as city_request
        value = city_request(1, trip_goal="Познакомиться с городом")
        for pool in ([], candidates(1) * 2, candidates(25)):
            with self.subTest(count=len(pool)):
                completion = Mock(side_effect=AssertionError("No AI"))
                result = select_city_preferences(value, pool, completion=completion)
                self.assertEqual(result.status, "insufficient_data")
                completion.assert_not_called()

    def test_unverified_proximity_is_explicit_not_ignored_or_claimed(self):
        self.request.location_preference = "nearby"
        self.request.stay_area = "Северный район"
        completion = Mock(side_effect=AssertionError("No location inference"))
        result = select_city_preferences(self.request, self.candidates, completion=completion)
        self.assertEqual(result.status, "unsupported_preferences")
        self.assertFalse(result.candidates)
        completion.assert_not_called()

    def test_unknown_duplicate_or_changed_ids_and_extra_model_facts_are_rejected(self):
        identifier = self.candidates[0].candidate_id
        bad = [
            {"status": "selected", "candidate_ids": ["osm-node-unknown"]},
            {"status": "selected", "candidate_ids": [identifier, identifier]},
            {"status": "selected", "candidate_ids": []},
            {"status": "insufficient_data", "candidate_ids": [identifier]},
            {"status": "selected", "candidate_ids": [identifier], "indoor": True},
            {"status": ["selected"], "candidate_ids": [identifier]},
            {"status": "selected", "candidate_ids": [True]},
        ]
        for value in bad:
            with self.subTest(value=value):
                completion = self.complete(value)
                result = select_city_preferences(self.request, self.candidates, completion=completion)
                self.assertEqual(result.status, "invalid_ai_response")
                self.assertFalse(result.candidates)
                completion.assert_called_once()

    def test_duplicate_keys_nonfinite_json_and_oversized_response_are_rejected(self):
        for text in ('{"status":"selected","status":"insufficient_data","candidate_ids":[]}',
                     '{"status":"selected","candidate_ids":NaN}', ' ' * 12001, 'not json'):
            with self.subTest(size=len(text)):
                completion = Mock(return_value=SimpleNamespace(text=text))
                result = select_city_preferences(self.request, self.candidates, completion=completion)
                self.assertEqual(result.status, "invalid_ai_response")
                completion.assert_called_once()

    def test_unmet_requirements_do_not_fall_back_to_source_seed_museums(self):
        for status in ("unsupported_preferences", "insufficient_data"):
            with self.subTest(status=status):
                completion = self.complete({"status": status, "candidate_ids": []})
                result = select_city_preferences(self.request, self.candidates, completion=completion)
                self.assertEqual(result.status, status)
                self.assertFalse(result.candidates)
                self.assertTrue(result.problems)

    def test_provider_failure_has_no_hidden_retry_or_preference_bypass(self):
        completion = Mock(side_effect=YandexAIError("Synthetic failure"))
        result = select_city_preferences(self.request, self.candidates, completion=completion)
        self.assertEqual(result.status, "ai_unavailable")
        self.assertFalse(result.candidates)
        completion.assert_called_once()

    def test_failed_depth_review_retains_first_valid_subset_and_reports_failure(self):
        from test_city_deterministic import candidates, request as city_request
        self.request = city_request(7, interest_categories=["museums"], trip_goal="Подробное знакомство с историей")
        pool = candidates(24)
        first = [item.candidate_id for item in pool[:5]]
        completion = Mock(side_effect=[SimpleNamespace(text=json.dumps({"status": "selected", "candidate_ids": first})),
                                       YandexAIError("Synthetic review failure")])
        result = select_city_preferences(self.request, pool, memory_context="Предпочитаю разнообразие",
                                         completion=completion, allow_depth_repair=True)
        self.assertEqual(result.status, "success")
        self.assertFalse(result.problems)
        self.assertEqual(result.candidate_priority, first)
        self.assertEqual(result.diagnostics["selection_calls"], 2)
        self.assertEqual(result.diagnostics["selection_repair_status"], "failed")
        self.assertTrue(result.warnings)
        self.assertEqual(completion.call_count, 2)

    def test_small_pool_does_not_pay_for_impossible_depth_review(self):
        from test_city_deterministic import candidates, request as city_request
        value = city_request(7, interest_categories=["museums"], trip_goal="Подробное знакомство с историей")
        pool = candidates(3)
        completion = self.complete({"status": "selected", "candidate_ids": [pool[0].candidate_id]})
        result = select_city_preferences(value, pool, memory_context="Предпочитаю разнообразие",
                                         completion=completion, allow_depth_repair=True)
        self.assertEqual(result.candidate_priority, [pool[0].candidate_id])
        self.assertFalse(result.diagnostics["selection_repair_attempted"])
        completion.assert_called_once()

    def test_shopping_checkbox_without_wishes_is_an_explicit_request(self):
        self.request.wishes = ""
        self.request.trip_goal = "Познакомиться с городом"
        self.request.interest_categories = ["main_sights", "museums", "theatre", "shopping_malls"]
        self.assertEqual(shopping_preference(self.request), "requested")

    def test_shopping_ru_en_morphology_and_acronyms(self):
        for text in ("Хочу в ТЦ", "Добавьте ТРЦ", "Посетить ТРК", "Посещение торговых центров",
                     "Прогуляться по торговому центру", "Торгово-развлекательные центры",
                     "Интересуют шопинг и музеи", "shopping", "Visit shopping centres",
                     "Shopping centers and museums", "A shopping mall", "Malls please",
                     "Не только ТЦ, но и музеи", "Не против торговых центров", "Not only malls",
                     "Не хочу музеи, хочу ТЦ", "Не музеи, а торговые центры",
                     "Не забудьте ТЦ", "Don't forget malls", "Не нужны музеи, только ТЦ",
                     "Не хочу длинных переездов, интересуют торговые центры",
                     "Торгово‑развлекательные центры"):
            with self.subTest(text=text):
                self.request.wishes = text
                self.assertEqual(shopping_preference(self.request), "requested")

    def test_explicit_shopping_exclusion_wins_over_checkbox(self):
        self.request.interest_categories = ["shopping_malls"]
        for text in ("Не хочу ТЦ", "Без торговых центров", "Исключить торговые центры",
                     "Не включать музеи и ТЦ", "ТЦ не нужны", "Торговые центры мне не интересны",
                     "No malls", "Avoid shopping centres", "Without shopping", "Don't include malls",
                     "Shopping centers are not needed", "Не хочу музеи, ТЦ и рестораны"):
            with self.subTest(text=text):
                self.request.wishes = text
                self.assertEqual(shopping_preference(self.request), "excluded")

    def test_reviewed_postfix_negations_and_other_category_exclusions(self):
        self.request.interest_categories = ["shopping_malls"]
        cases = {
            "ТЦ не надо": "excluded", "Шопинг не планируем": "excluded", "ТЦ — нет": "excluded",
            "Без музеев и с посещением ТЦ": "requested", "Музеи не нужны, ТЦ интересуют": "requested",
            "Без музеев, но ТЦ интересуют": "requested", "Shopping is not planned": "excluded",
            "No museums and with visits to malls": "requested", "Museums not needed, malls interest me": "requested",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.request.wishes = text
                self.assertEqual(shopping_preference(self.request), expected)

    def test_ambiguous_negation_is_left_to_semantic_selector_not_a_category_ban(self):
        self.request.interest_categories = ["shopping_malls"]
        for text in ("Без длинных очередей хочу ТЦ", "Музеи не нужны, ТЦ",
                     "Не знаю, заинтересуют ли торговые центры", "No long queues around malls"):
            with self.subTest(text=text):
                self.request.wishes = text
                self.assertEqual(shopping_preference(self.request), "unspecified")
                completion = self.complete({"status": "selected", "candidate_ids": [self.candidates[0].candidate_id]})
                result = select_city_preferences(self.request, self.candidates, completion=completion)
                self.assertEqual(result.status, "success")
                data = json.loads(completion.call_args.args[0][1]["content"])
                self.assertEqual(data["request"]["wishes"], text)
                self.assertEqual(data["shopping_preference"], "unspecified")

    def test_mall_is_classified_by_provider_type_not_its_title(self):
        item = deepcopy(self.candidates[0])
        item.title = "Mall museum"
        self.assertFalse(is_shopping_candidate(item))
        item.record.update(category="shop", type="mall")
        self.assertTrue(is_shopping_candidate(item))
        item.record["type"] = "shopping_centre"
        self.assertTrue(is_shopping_candidate(item))
        item.record["type"] = "department_store"
        self.assertFalse(is_shopping_candidate(item))
        item.record.update(category="place", type="mall")
        self.assertFalse(is_shopping_candidate(item))

    def test_missing_requested_mall_preserves_other_choices_with_explicit_warning(self):
        self.request.wishes = ""
        self.request.interest_categories = ["shopping_malls", "museums"]
        ids = [item.candidate_id for item in self.candidates]
        completion = self.complete({"status": "selected", "candidate_ids": ids})
        result = select_city_preferences(self.request, self.candidates, completion=completion)
        self.assertEqual(result.status, "success")
        self.assertIn("торговый центр", result.warnings[0])
        self.assertEqual(result.candidates, self.candidates)
        completion.assert_not_called()

    def test_only_malls_requirement_is_not_relaxed_when_malls_missing(self):
        self.request.wishes = "Только торговые центры"
        completion = self.complete({"status": "insufficient_data", "candidate_ids": []})
        result = select_city_preferences(self.request, self.candidates, completion=completion)
        self.assertEqual(result.status, "insufficient_data")
        self.assertFalse(result.candidates)
        self.assertIn("только торговые центры", completion.call_args.args[0][0]["content"])

    def test_unrestricted_mall_is_restored_but_forbidden_mall_is_rejected(self):
        mall = deepcopy(self.candidates[1])
        mall.record.update(category="shop", type="mall")
        self.candidates[1] = mall
        self.request.interest_categories = ["shopping_malls"]
        cases = [("", [self.candidates[0].candidate_id]), ("Без ТЦ", [mall.candidate_id])]
        for wishes, ids in cases:
            with self.subTest(wishes=wishes):
                self.request.wishes = wishes
                completion = self.complete({"status": "selected", "candidate_ids": ids})
                result = select_city_preferences(self.request, self.candidates, completion=completion)
                if wishes:
                    self.assertEqual(result.status, "invalid_ai_response")
                    self.assertFalse(result.candidates)
                else:
                    self.assertEqual(result.status, "success")
                    self.assertIn(mall.candidate_id, result.candidate_priority)
                if wishes:
                    completion.assert_called_once()
                else:
                    completion.assert_not_called()

    def test_default_goal_preserves_checked_categories_without_semantic_completion(self):
        from test_city_deterministic import candidates
        pool = candidates(3)
        pool[1].record.update(category="tourism", type="zoo")
        pool[2].record.update(category="shop", type="mall")
        self.request.wishes = ""
        self.request.trip_goal = "Познакомиться с городом"
        self.request.interest_categories = ["museums", "zoos", "shopping_malls"]
        completion = self.complete({"status": "selected", "candidate_ids": [pool[0].candidate_id]})
        result = select_city_preferences(self.request, pool, completion=completion)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.candidate_priority, [item.candidate_id for item in pool])
        self.assertEqual(result.diagnostics["selection_mode"], "structured_preferences")
        self.assertEqual(result.diagnostics["selection_calls"], 0)
        completion.assert_not_called()

    def test_free_text_constraints_prevent_category_restoration(self):
        from test_city_deterministic import candidates
        pool = candidates(2)
        pool[-1].record.update(category="tourism", type="zoo")
        self.request.interest_categories = ["museums", "zoos"]
        for goal, wishes in (("Познакомиться с городом", "Без зоопарка"),
                             ("Только музеи без животных", "")):
            with self.subTest(goal=goal, wishes=wishes):
                self.request.trip_goal, self.request.wishes = goal, wishes
                result = select_city_preferences(self.request, pool, completion=self.complete({
                    "status": "selected", "candidate_ids": [pool[0].candidate_id]}))
                self.assertEqual(result.candidate_priority, [pool[0].candidate_id])

    def test_existing_zoo_selection_does_not_restore_three_rejected_pony_farms(self):
        from test_city_deterministic import candidates
        pool = candidates(5)
        for index, title in enumerate(("Городской зоопарк", "Пони-ферма", "Пони ферма Северная", "Пони-ферма у парка"), 1):
            pool[index].title = title
            pool[index].record.update(name=title, category="tourism", type="zoo")
        self.request.wishes = "Без пони-ферм"
        self.request.trip_goal = "Познакомиться с городом"
        self.request.interest_categories = ["museums", "zoos"]
        selected = [pool[0].candidate_id, pool[1].candidate_id]
        completion = self.complete({"status": "selected", "candidate_ids": selected})
        result = select_city_preferences(self.request, pool, memory_context="Предпочитаю большой зоопарк",
                                         completion=completion)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.candidate_priority, selected)
        self.assertEqual(result.candidates, pool[:2])
        completion.assert_called_once()

    def test_custom_goal_with_memory_never_restores_a_missing_category(self):
        from dataclasses import replace
        import city_preferences
        from test_city_deterministic import candidates
        pool = candidates(5)
        for item in pool[1:]:
            item.record.update(category="tourism", type="zoo")
        self.request.wishes = ""
        self.request.trip_goal = "Подробное знакомство с историей"
        self.request.children_ages = [8]
        self.request.interest_categories = ["museums", "zoos"]
        actual = city_preferences.visit_option
        def known_conditions(item, request):
            option = actual(item, request)
            if item.candidate_id == pool[1].candidate_id:
                return replace(option, children_allowed=False)
            if item.candidate_id == pool[2].candidate_id:
                return replace(option, allowed_dates=())
            return option
        with patch.object(city_preferences, "visit_option", side_effect=known_conditions):
            result = select_city_preferences(self.request, pool, memory_context="Не интересуют животные", completion=self.complete({
                "status": "selected", "candidate_ids": [pool[0].candidate_id]}))
        self.assertEqual(result.status, "success")
        self.assertEqual(result.candidate_priority, [pool[0].candidate_id])
        self.assertEqual(result.diagnostics["selection_mode"], "semantic_preferences")

    def test_memory_ranks_but_never_truncates_full_structured_reserve(self):
        from test_city_deterministic import candidates, request as city_request
        pool = candidates(24)
        value = city_request(7, trip_goal="Познакомиться с городом", interest_categories=["museums"])
        completion = self.complete({"candidate_ids": [pool[10].candidate_id, pool[3].candidate_id]})
        result = select_city_preferences(value, pool, memory_context="Предпочитаю этот музей",
                                         completion=completion, allow_depth_repair=True)
        self.assertEqual(result.candidate_priority[:2], [pool[10].candidate_id, pool[3].candidate_id])
        self.assertEqual(set(result.candidate_priority), {item.candidate_id for item in pool})
        self.assertEqual(len(result.candidates), 24)
        self.assertEqual(result.diagnostics["selection_mode"], "memory_ranking")
        self.assertEqual(result.diagnostics["memory_ranking_status"], "applied")
        self.assertEqual(result.diagnostics["selection_calls"], 1)
        self.assertFalse(result.diagnostics["selection_repair_attempted"])
        completion.assert_called_once()

    def test_invalid_or_unavailable_memory_ranking_keeps_server_order_with_explicit_status(self):
        from test_city_deterministic import candidates, request as city_request
        pool = candidates(4)
        value = city_request(1, trip_goal="Познакомиться с городом", interest_categories=["museums"])
        bad = [YandexAIError("offline"), {"status": "unsupported_preferences", "candidate_ids": []},
               {"candidate_ids": ["unknown"]}, {"candidate_ids": [pool[0].candidate_id] * 2},
               {"candidate_ids": [], "facts": "forbidden"}, {"candidate_ids": [True]}]
        for answer in bad:
            with self.subTest(answer_type=type(answer).__name__):
                completion = Mock(side_effect=answer) if isinstance(answer, Exception) else self.complete(answer)
                result = select_city_preferences(value, pool, memory_context="Справочная память",
                                                 completion=completion, allow_depth_repair=True)
                self.assertEqual(result.status, "success")
                self.assertEqual(result.candidates, pool)
                self.assertEqual(result.diagnostics["memory_ranking_status"], "unavailable")
                self.assertTrue(result.warnings)
                completion.assert_called_once()

    def test_memory_priority_cannot_promote_broad_attraction_over_meaningful_types(self):
        from test_city_deterministic import candidates, request as city_request
        pool = candidates(3)
        pool[0].record.update(type="attraction")
        value = city_request(1, trip_goal="Познакомиться с городом", interest_categories=["museums", "main_sights"])
        result = select_city_preferences(value, pool, memory_context="Справочная память",
            completion=self.complete({"candidate_ids": [pool[0].candidate_id, pool[2].candidate_id]}))
        self.assertEqual(result.candidate_priority, [pool[i].candidate_id for i in (2, 1, 0)])

    def test_known_ineligible_selected_zoo_does_not_hide_available_zoo_category(self):
        from dataclasses import replace
        import city_preferences
        from test_city_deterministic import candidates
        pool = candidates(4)
        for item in pool[2:]:
            item.record.update(category="tourism", type="zoo")
        self.request.wishes = ""
        self.request.trip_goal = "Познакомиться с городом"
        self.request.children_ages = [8]
        self.request.interest_categories = ["museums", "zoos"]
        actual = city_preferences.visit_option
        for restriction in ({"children_allowed": False}, {"allowed_dates": ()}):
            def known_conditions(item, request):
                option = actual(item, request)
                return replace(option, **restriction) if item.candidate_id == pool[2].candidate_id else option
            with self.subTest(restriction=restriction), \
                    patch.object(city_preferences, "visit_option", side_effect=known_conditions):
                completion = self.complete({"status": "selected", "candidate_ids": [item.candidate_id for item in pool[:3]]})
                result = select_city_preferences(self.request, pool, completion=completion, allow_depth_repair=True)
            self.assertEqual(result.status, "success")
            self.assertIn(pool[3].candidate_id, result.candidate_priority)
            self.assertEqual(result.diagnostics["preference_planning_minutes"], 225)
            self.assertFalse(result.diagnostics["selection_repair_attempted"])
            completion.assert_not_called()

    def test_qualitative_main_sights_does_not_restore_rejected_map_attraction(self):
        from test_city_deterministic import candidates
        pool = candidates(2)
        pool[1].title = "Воздушные шары"
        pool[1].record.update(category="tourism", type="attraction")
        self.request.wishes = ""
        self.request.trip_goal = "Подробное знакомство с музеями"
        self.request.interest_categories = ["museums", "main_sights"]
        result = select_city_preferences(self.request, pool, completion=self.complete({
            "status": "selected", "candidate_ids": [pool[0].candidate_id]}))
        self.assertEqual(result.status, "success")
        self.assertEqual(result.candidate_priority, [pool[0].candidate_id])

    def test_requested_mall_uses_exact_known_id_and_unknown_visit_facts(self):
        self.request.wishes = "Добавить ТЦ"
        self.candidates[1].record.update(category="shop", type="mall")
        ids = [item.candidate_id for item in self.candidates]
        completion = self.complete({"status": "selected", "candidate_ids": ids})
        before = deepcopy(self.candidates)
        result = select_city_preferences(self.request, self.candidates, completion=completion)
        self.assertEqual((result.status, result.candidate_priority), ("success", ids))
        self.assertEqual(self.candidates, before)
        data = json.loads(completion.call_args.args[0][1]["content"])
        self.assertEqual(data["shopping_preference"], "requested")
        self.assertEqual(data["shopping_candidate_ids"], [ids[1]])
        conditions = data["verified_candidates"][1]["visit_conditions"]
        self.assertIsNone(conditions["estimated_cost"])
        self.assertIsNone(conditions["indoor"])

    def test_bot_unsupported_proximity_stops_before_profile_quota_and_thread(self):
        from schemas import TripRequest
        from test_bot_city_edit_gate import function
        namespace = {"TripRequest": TripRequest, "bot": Mock(), "db": Mock(),
                     "_message_user_id": lambda message: 123,
                     "minimum_meal_cost": lambda request: 1,
                     "city_candidates_enabled": lambda: True,
                     "_reserve_generation_job": Mock(), "_claim_route_ai_request": Mock(),
                     "threading": Mock()}
        self.request.location_preference, self.request.stay_area = "nearby", "Северный район"
        message = SimpleNamespace(chat=SimpleNamespace(id=123), from_user=SimpleNamespace(first_name="Test"))
        with patch("city_rollout.city_deterministic_enabled", return_value=True):
            function("start_generation", namespace)(message, self.request.model_dump())
        namespace["bot"].send_message.assert_called_once()
        namespace["db"].save_user.assert_not_called()
        namespace["_claim_route_ai_request"].assert_not_called()
        namespace["_reserve_generation_job"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
