"""Semantic ID selection preserves inputs; numeric planning stays on the server."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from city_preferences import select_city_preferences
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
