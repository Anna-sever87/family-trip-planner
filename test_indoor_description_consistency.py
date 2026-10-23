"""Only explicit self-contradictions fail; place names are not indoor evidence."""

from contextlib import ExitStack
from copy import deepcopy
import json
import socket
import sqlite3
import unittest
from unittest.mock import patch

from city_response import build_city_response_schema
from grounded_response import _validate, decode_grounded_response
import planner
from rules import _explicit_outdoor_description, check_rules
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest
from yandex_ai import GenerationResult


DESCRIPTION = "Посещение музея под открытым небом Аспендос"


def request_for(**updates):
    values = dict(destination="Тестовый город, Россия", start_date="2099-10-28",
                  end_date="2099-10-28", adults=2, children_ages=[5], budget=30_000,
                  currency="RUB", start_time="09:00", end_time="19:00", pace="средний")
    values.update(updates)
    return TripRequest(**values)


def itinerary_for(description=DESCRIPTION, title="Аспендос"):
    return Itinerary(title="Программа", summary="Проверка согласованности", tips=["Уточните условия посещения"], days=[
        ItineraryDay(day=1, date="2099-10-28", theme="Местная история", items=[
            ItineraryItem(kind="attraction", time="09:00", title=title, area="Центр",
                          place_query=f"{title}, Тестовый город, Россия", description=description,
                          duration_min=90, indoor=True),
            ItineraryItem(kind="meal", time="12:00", title="Обед", area="Центр",
                          description="Обед", duration_min=60, cost=1800),
            ItineraryItem(kind="rest", time="13:00", title="Отдых", area="Центр",
                          description="Спокойный перерыв", duration_min=60),
        ]),
    ])


def model_day(day, schema):
    values = day.model_dump()
    result = {key: values[key] for key in schema["properties"]}
    item_fields = schema["properties"]["items"]["items"]["properties"]
    result["items"] = [{key: item[key] for key in item_fields} for item in values["items"]]
    return result


class IndoorDescriptionConsistencyTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "getaddrinfo"), (sqlite3, "connect")):
            blocker = patch.object(owner, name, side_effect=AssertionError("No external I/O"))
            blocker.start()
            self.addCleanup(blocker.stop)

    def contradictions(self, itinerary, request=None, rainy_dates=None):
        return [problem for problem in check_rules(itinerary, request or request_for(), rainy_dates)
                if "признак помещения противоречит" in problem]

    def test_actual_open_air_museum_description_is_rejected_without_weather(self):
        problems = check_rules(itinerary_for(), request_for())
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("под открытым небом, но indoor = true", problems[0])

    def test_explicit_grammar_normalizes_case_spacing_and_final_punctuation(self):
        for description in (
            DESCRIPTION, "  ПОСЕЩЕНИЕ\u00a0музея под открытым небом\nАСПЕНДОС.  ",
            "Осмотр музея под открытым небом Аспендос!", "Экскурсия по музею под открытым небом Аспендос",
            "Посещение музея под открытым небом", "Посещение проходит под открытым небом.",
            "Осмотр проходит под открытым небом", "Экскурсия проходит под открытым небом",
            "Прогулка проходит под открытым небом", "Прогулка под открытым небом",
            "Экскурсия под открытым небом",
        ):
            with self.subTest(description=description):
                self.assertEqual(len(self.contradictions(itinerary_for(description))), 1)

    def test_museum_template_uses_exact_arbitrary_title_not_a_poi_list(self):
        for title in ("Этнографический комплекс", "Northern Heritage", "Двор старых ремёсел"):
            with self.subTest(title=title):
                self.assertTrue(_explicit_outdoor_description(f"Посещение музея под открытым небом {title}", title))
        self.assertFalse(_explicit_outdoor_description(DESCRIPTION, "Другой объект"))

    def test_negation_condition_quotation_and_alternative_are_not_hard_rejected(self):
        for description in (
            "Посещение проходит не под открытым небом",
            "Не посещаем музей под открытым небом Аспендос",
            "Без прогулки под открытым небом: посещаем зал",
            "Вместо прогулки под открытым небом посещаем выставку",
            "Если сухо, экскурсия проходит под открытым небом",
            "При хорошей погоде прогулка под открытым небом",
            "Возможна прогулка под открытым небом",
            "Прогулка под открытым небом?",
            "«Посещение проходит под открытым небом» — название выставки",
            "Описание альтернативы: прогулка под открытым небом",
        ):
            with self.subTest(description=description):
                self.assertEqual(self.contradictions(itinerary_for(description)), [])

    def test_mixed_and_explicit_covered_part_visits_are_outside_narrow_grammar(self):
        for description in (
            DESCRIPTION + ", но посещаем только крытый павильон",
            DESCRIPTION + "; только крытая экспозиция",
            DESCRIPTION + ". При дожде остаёмся в крытом зале.",
            "Посещение крытой части музея под открытым небом Аспендос",
            "Посещение музея под открытым небом: крытая галерея и открытый двор",
            "Экскурсия проходит под открытым небом и в помещении",
            "Прогулка под открытым небом, затем выставка в здании",
        ):
            with self.subTest(description=description):
                route = itinerary_for(description)
                self.assertEqual(self.contradictions(route), [])
                self.assertFalse(route.days[0].items[0].verified)

    def test_place_title_query_and_category_words_do_not_classify_indoor(self):
        for title in ("Сочинский национальный парк", "Пещера Бельдиби", "Музей под открытым небом",
                      "Open Air Museum", "Галерея северной истории"):
            with self.subTest(title=title):
                self.assertEqual(self.contradictions(itinerary_for("Посещение выбранной экспозиции", title)), [])

    def test_copied_qualifying_title_cannot_turn_mixed_description_into_assertion(self):
        for title in ("Аспендос, но только крытая часть", "Аспендос только крытая экспозиция",
                      "Аспендос (павильон)", "Аспендос внутренний зал", "Аспендос и галерея",
                      "Аспендос при хорошей погоде", "Аспендос без прогулки"):
            with self.subTest(title=title):
                description = f"Посещение музея под открытым небом {title}"
                self.assertEqual(self.contradictions(itinerary_for(description, title)), [])

    def test_only_attraction_with_exact_true_flag_is_checked(self):
        route = itinerary_for("Посещение экспозиции")
        for item in route.days[0].items[1:]:
            item.description = "Посещение проходит под открытым небом"
            item.indoor = True
        self.assertEqual(self.contradictions(route), [])
        for value in (False, 1, "true", None):
            route = itinerary_for()
            route.days[0].items[0].indoor = value
            with self.subTest(value=value):
                self.assertEqual(self.contradictions(route), [])

    def test_rule_does_not_mutate_values_or_replace_existing_rain_check(self):
        route = itinerary_for()
        before = deepcopy(route.model_dump())
        self.assertEqual(len(self.contradictions(route, rainy_dates=["2099-10-28"])), 1)
        self.assertEqual(route.model_dump(), before)
        route.days[0].items[0].indoor = False
        problems = check_rules(route, request_for(), ["2099-10-28"])
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("дождливый", problems[0])
        self.assertEqual(self.contradictions(route), [])

    def test_region_and_source_flags_do_not_grant_false_semantic_evidence(self):
        for kind in ("city", "region"):
            for source in (False, True):
                route = itinerary_for()
                route.source_schedule = source
                with self.subTest(kind=kind, source=source):
                    self.assertEqual(len(self.contradictions(route, request_for(destination_kind=kind))), 1)
                route.days[0].items[0].description = "Посещение крытой экспозиции"
                self.assertEqual(self.contradictions(route, request_for(destination_kind=kind)), [])

    def test_schema_compliant_city_repair_is_still_rejected_by_rules(self):
        request, route = request_for(), itinerary_for()
        canonical = planner.build_itinerary_json_schema(request)
        day = model_day(route.days[0], canonical["properties"]["days"]["items"])
        items = day.pop("items")
        day.update(attractions=[items[0]], meal=items[1], rest=items[2])
        payload = dict(title=route.title, summary=route.summary, tips=route.tips, day_1=day)
        wire = build_city_response_schema(canonical, request)
        decoded = Itinerary.model_validate_json(decode_grounded_response(json.dumps(payload, ensure_ascii=False), wire))
        self.assertEqual(len(self.contradictions(decoded, request)), 1)

    def invoke_planner(self, *, editing):
        request, candidate = request_for(), itinerary_for()
        original = itinerary_for("Посещение крытой экспозиции")
        before = original.model_dump()
        canonical = planner.build_itinerary_json_schema(request)
        day = model_day(candidate.days[0], canonical["properties"]["days"]["items"])
        payload = day if editing else dict(title=candidate.title, summary=candidate.summary,
                                           tips=candidate.tips, days=[day])

        def completion(messages, **kwargs):
            _validate(payload, kwargs["json_schema"])
            return GenerationResult(json.dumps(payload, ensure_ascii=False), 1, 1)

        with ExitStack() as stack:
            for name, value in (("get_weather", None), ("find_city", None), ("cached_place_hints", []),
                                ("curated_place_hints", []), ("_load_blocked_places", [])):
                stack.enter_context(patch.object(planner, name, return_value=value))
            generated = stack.enter_context(patch.object(planner, "generate_json", side_effect=completion))
            maps = stack.enter_context(patch.object(planner, "_map_verification_problems", return_value=[]))
            audit = stack.enter_context(patch.object(planner, "_audit_edit_fulfillment", return_value=([], 0, 0)))
            if editing:
                result = planner.edit_itinerary(request, original, 1, "Измени описание посещения", max_attempts=1)
            else:
                result = planner.generate_itinerary(request, max_attempts=1)
        self.assertEqual(original.model_dump(), before)
        generated.assert_called_once()
        maps.assert_not_called()
        audit.assert_not_called()
        self.assertTrue(any("признак помещения противоречит" in problem for problem in result.problems), result.problems)

    def test_create_path_rejects_contradiction_before_map(self):
        self.invoke_planner(editing=False)

    def test_edit_path_rejects_contradiction_before_map_or_semantic_audit(self):
        self.invoke_planner(editing=True)


if __name__ == "__main__":
    unittest.main()
