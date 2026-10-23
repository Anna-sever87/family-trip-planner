"""Prompt estimates can be corrected; map identity and rain guards cannot."""

from copy import deepcopy
import socket
import sqlite3
import unittest
from unittest.mock import patch

import city_quality
from city_schedule import CITY_CANDIDATE_REPAIR_RULE, CITY_CANDIDATE_SCHEDULE_RULE
from grounded_response import _validate
import test_city_candidate_planner as fixture


def rainy_weather():
    return {"city": "Тестгород", "source": "forecast",
            "days": [{"date": "2099-10-28", "tmin": 10, "tmax": 20, "rain_mm": 5}]}


class CityIndoorPromptTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "getaddrinfo"), (sqlite3, "connect")):
            blocker = patch.object(owner, name, side_effect=AssertionError("External I/O forbidden"))
            mock = blocker.start()
            self.addCleanup(blocker.stop)
            self.addCleanup(mock.assert_not_called)

    def run_plan(self, responses, **kwargs):
        # Compose the existing fixture, never inherit its testcase methods.
        return fixture.CityCandidatePlannerTests().run_plan(responses, weather=rainy_weather(), **kwargs)

    def test_prompt_distinguishes_immutable_identity_from_revisable_unverified_estimates(self):
        rule = CITY_CANDIDATE_SCHEDULE_RULE
        self.assertIn("Неизменяемые данные кандидата", rule)
        self.assertIn("indoor, time, duration_min и cost", rule)
        self.assertIn("их можно исправлять по смыслу программы", rule)
        self.assertIn("indoor=false не означает «карта не доказала помещение»", rule)
        self.assertIn("Одного слова museum или типа карты недостаточно", rule)
        self.assertIn("При недостатке знаний не выдумывай помещение", rule)
        self.assertIn("Не выдавай модельные оценки за проверенные факты", rule)
        self.assertNotIn("Не утверждай неподтверждённые свойства", rule)
        self.assertIn("не общий запрет indoor=true", city_quality.CITY_QUALITY_RULE)
        self.assertIn("Тип museum также не подтверждает indoor", city_quality.CITY_QUALITY_RULE)

    def test_real_pipeline_accepts_model_false_to_true_correction_without_identity_change(self):
        pool = fixture.default_candidates()
        before_pool = deepcopy(pool)
        first = fixture.payload(indoor=False)
        for item in first["day_1"]["attractions"]:
            item["description"] = "Осмотр экспозиции во внутренних залах; доступ и часы уточнить заранее."
        repaired = deepcopy(first)
        for item in repaired["day_1"]["attractions"]:
            item["indoor"] = True
        original = deepcopy((first, repaired))
        result, completion, _ = self.run_plan([first, repaired], candidates=pool)
        self.assertEqual(result.problems, [])
        self.assertEqual((result.attempts, completion.call_count), (3, 3))
        visits = [item for item in result.itinerary.days[0].items if item.kind == "attraction"]
        self.assertTrue(all(item.indoor is True for item in visits))
        self.assertEqual([item.place_query for item in visits], [candidate.place_query for candidate in pool])
        self.assertEqual([(item.lat, item.lon) for item in visits],
                         [(candidate.record["lat"], candidate.record["lon"]) for candidate in pool])
        self.assertEqual([item.title for item in visits], [candidate.title for candidate in pool])
        self.assertEqual(pool, before_pool)
        self.assertEqual((first, repaired), original)
        repair_prompt = completion.call_args_list[-1].args[0][-1]["content"]
        self.assertIn(CITY_CANDIDATE_REPAIR_RULE, repair_prompt)
        self.assertIn("дождливый", repair_prompt)
        self.assertNotIn("Не меняй ID и свойства мест", repair_prompt)
        schema = completion.call_args_list[-1].kwargs["json_schema"]
        indoor_schema = schema["properties"]["day_1"]["properties"]["attractions"]["items"]["properties"]["indoor"]
        self.assertEqual(indoor_schema, {"type": "boolean"})
        _validate(True, indoor_schema)
        _validate(False, indoor_schema)

    def test_repeated_false_plan_is_still_rejected_not_automatically_relabelled(self):
        payload = fixture.payload(indoor=False)
        result, completion, _ = self.run_plan([payload, deepcopy(payload)])
        self.assertEqual((result.attempts, completion.call_count), (3, 3))
        self.assertTrue(any("дождливый" in problem for problem in result.problems))
        self.assertTrue(all(item.indoor is False for item in result.itinerary.days[0].items
                            if item.kind == "attraction"))

    def test_false_indoor_claim_for_explicit_open_air_program_still_fails(self):
        payload = fixture.payload(indoor=True)
        payload["day_1"]["attractions"][0]["description"] = "Посещение музея под открытым небом."
        result, completion, _ = self.run_plan([payload, deepcopy(payload)])
        self.assertEqual((result.attempts, completion.call_count), (3, 3))
        self.assertTrue(any("под открытым небом, но indoor = true" in problem for problem in result.problems))

    def test_open_air_map_identity_still_rejects_claim_of_inner_halls(self):
        pool = fixture.default_candidates()
        pool[0].title = "Учебный музей под открытым небом"
        pool[0].place_query = f"{pool[0].title}, Тестгород, Россия"
        pool[0].record["name"] = pool[0].place_query
        payload = fixture.payload(indoor=True)
        payload["day_1"]["attractions"][0]["description"] = "Осмотр внутренней экспозиции."
        before = deepcopy(pool)
        result, completion, _ = self.run_plan([payload, deepcopy(payload)], candidates=pool)
        self.assertEqual((result.attempts, completion.call_count), (3, 3))
        self.assertTrue(any("не обосновывает indoor=true" in problem for problem in result.problems))
        self.assertEqual(pool, before)


if __name__ == "__main__":
    unittest.main()
