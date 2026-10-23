"""No network or storage: public errors must never echo AI repair instructions."""

import unittest

from generation_feedback import generation_failure_message, generation_failure_reason


RAIN_REASON = "Не удалось подобрать подходящие посещения в помещении для дождливых дней."
RAIN_DAY_PROBLEM = (
    "День 3 (2026-10-30) дождливый, но под открытым небом: Музей истории Сочи. "
    "Не менее половины достопримечательностей должны быть в помещении с indoor = true. "
    "Если достопримечательность одна, она обязана быть в помещении."
)
RAIN_POOL_PROBLEM = (
    "Для 3 дождливых дней в проверенном наборе только 1 мест, "
    "не исключённых по уличному/смешанному типу или явному названию карты. "
    "Их крытая программа также не подтверждена. "
    "Расписание не запрашивалось: повторять одно место или выдумывать помещения нельзя. "
    "Ограниченный поиск не означает, что других мест в городе нет."
)


class GenerationFeedbackTests(unittest.TestCase):
    def test_recognized_rain_rule_precedes_generic_attraction_category_without_raw_detail(self):
        for problem in (RAIN_DAY_PROBLEM, RAIN_DAY_PROBLEM.upper(),
                        RAIN_DAY_PROBLEM.replace("Музей истории Сочи", "secret-sentinel https://private.invalid/token")):
            with self.subTest(problem=problem):
                self.assertEqual(generation_failure_reason([problem]), RAIN_REASON)
                message = generation_failure_message([problem])
                self.assertIn(RAIN_REASON, message)
                for forbidden in ("Музей", "secret-sentinel", "private.invalid", "indoor", "подтверждённых картой",
                                  "час", "повтор", "заплат"):
                    self.assertNotIn(forbidden, message)
        self.assertEqual(generation_failure_reason([RAIN_DAY_PROBLEM, "place_query не подтверждён"]), RAIN_REASON)

    def test_recognized_rain_pool_has_same_fixed_reason_without_claiming_city_has_no_places(self):
        for problem in (RAIN_POOL_PROBLEM, RAIN_POOL_PROBLEM.replace(" или явному названию", "")):
            with self.subTest(problem=problem):
                self.assertEqual(generation_failure_reason([problem]), RAIN_REASON)
        self.assertEqual(generation_failure_reason([RAIN_DAY_PROBLEM, RAIN_POOL_PROBLEM]), RAIN_REASON)

    def test_rain_reason_preserves_explicit_source_and_map_outage_priority(self):
        source = "SOURCE_PLAN_UNAVAILABLE: SOURCE_SEARCH_LIMIT requestedDays=7 usableCards=6"
        self.assertEqual(generation_failure_reason([RAIN_DAY_PROBLEM, source]), generation_failure_reason([source]))
        unknown_source = "SOURCE_PLAN_UNAVAILABLE: SOURCE_NEW_CATEGORY"
        self.assertEqual(generation_failure_reason([RAIN_POOL_PROBLEM, unknown_source]),
                         generation_failure_reason([unknown_source]))
        outage = "Сервис карты временно недоступен"
        self.assertEqual(generation_failure_reason([RAIN_DAY_PROBLEM, outage]), generation_failure_reason([outage]))

    def test_indoor_words_weather_mentions_and_malformed_diagnostics_do_not_gain_rain_category(self):
        for problem in (
            "В пожеланиях указан дождь и indoor=true", "Дождливый день: неизвестны часы музея",
            "Не найдено на карте место «Дождливый день»", "indoor должен иметь тип boolean",
            "День 1: у объекта описание под открытым небом, но indoor = true",
            RAIN_DAY_PROBLEM.replace("дождливый, но под открытым небом", "солнечный, но под открытым небом"),
            RAIN_DAY_PROBLEM.replace("indoor = true", "indoor = false"),
            RAIN_DAY_PROBLEM.replace("День 3", "День 99"),
            "Пользователь написал: " + RAIN_DAY_PROBLEM,
            RAIN_DAY_PROBLEM + " Дополнительная неизвестная инструкция.",
            RAIN_POOL_PROBLEM.replace("только 1 мест", "только 4 мест"),
            RAIN_POOL_PROBLEM.replace("Для 3", "Для 99"),
            "Для 3 дождливых дней в проверенном наборе только 1 мест, но они доступны.",
        ):
            with self.subTest(problem=problem):
                self.assertNotEqual(generation_failure_reason([problem]), RAIN_REASON)

    def test_rain_edit_failure_keeps_original_without_model_repair_instructions(self):
        message = generation_failure_message([RAIN_DAY_PROBLEM], editing=True)
        self.assertEqual(message, f"Изменения не сохранены. {RAIN_REASON} Старый маршрут сохранён без изменений.")

    def test_repeated_internal_place_errors_become_one_short_reason(self):
        problems = [
            f"День {day}: place_query пуст. Укажи официальное название. Исправь base_area."
            for day in range(1, 8)
        ]
        before = list(problems)
        message = generation_failure_message(problems)
        self.assertIn("подтверждённых картой", message)
        self.assertIn("Новая поездка не сохранена", message)
        self.assertLess(len(message), 300)
        for internal in ("place_query", "base_area", "Укажи", "Исправь", "День"):
            self.assertNotIn(internal, message)
        self.assertEqual(problems, before)

    def test_explicit_map_outage_is_not_reported_as_missing_places(self):
        reason = generation_failure_reason([
            "День 2: карта временно недоступна, место не проверено (place_query).",
        ])
        self.assertIn("Сервис карты временно недоступен", reason)
        self.assertNotIn("подобрать", reason)

    def test_missing_place_is_not_reported_as_map_outage(self):
        reason = generation_failure_reason(["День 1: место не найдено на карте"])
        self.assertIn("подтверждённых картой", reason)
        self.assertNotIn("недоступен", reason)

    def test_grounding_shortfall_has_no_suggestion_to_change_approved_parameters(self):
        reason = generation_failure_reason([
            "Для поездки на 7 дней удалось подтвердить только 6 конкретных мест. "
            "Расписание не создавалось. Попробуйте более компактную территорию или меньше дней.",
        ])
        self.assertIn("подтверждённых картой", reason)
        self.assertNotIn("меньше дней", reason)

    def test_time_and_budget_errors_have_plain_language_reason(self):
        for problem in (
            "Расходы на программу превышают доступный бюджет программы после дороги и жилья",
            "День 2: не оставлено достаточно времени на переезд",
            "День 1: недостаточно содержательная программа: увеличь duration_min",
        ):
            with self.subTest(problem=problem):
                self.assertEqual(
                    generation_failure_reason([problem]),
                    "Не удалось согласовать программу с доступным временем и бюджетом.",
                )

    def test_unknown_diagnostic_content_is_never_echoed(self):
        message = generation_failure_message(["JSON: secret-sentinel " + "INSTRUCTION " * 1000])
        self.assertNotIn("JSON", message)
        self.assertNotIn("secret-sentinel", message)
        self.assertNotIn("INSTRUCTION", message)
        self.assertLess(len(message), 300)

    def test_edit_explicitly_keeps_original(self):
        message = generation_failure_message(["Повтор места"], editing=True)
        self.assertIn("Изменения не сохранены", message)
        self.assertIn("Старый маршрут сохранён без изменений", message)
        self.assertNotIn("Новая поездка", message)

    def test_empty_unknown_category_is_safe(self):
        self.assertEqual(
            generation_failure_reason([]),
            "Не удалось согласовать маршрут с условиями поездки.",
        )

    def test_explicit_source_categories_override_conflicting_generic_diagnostics(self):
        cases = {
            "SOURCE_VISIT_DATA_INCOMPLETE": "Это не означает, что в регионе нет подходящих мест",
            "SOURCE_SEARCH_LIMIT": "Локальный поиск достиг безопасного предела",
            "SOURCE_CONSTRAINTS_INCOMPATIBLE": "с заданными датами, временем и бюджетом",
            "SOURCE_ROUTES_UNCONFIRMED": "подтвердить необходимые дорожные связи",
        }
        for code, expected in cases.items():
            with self.subTest(code=code):
                message = generation_failure_message([
                    "Карта временно недоступна; place_query SECRET; бюджет; Исправь все дни",
                    f"SOURCE_PLAN_UNAVAILABLE: {code} requestedDays=7 usableCards=6",
                ])
                self.assertIn(expected, message)
                self.assertIn("Дней в запросе — 7; посещений с подходящими данными — 6", message)
                for forbidden in ("SECRET", "place_query", "Исправь", code, "меньше дней", "Повторные запросы", "Сервис карты"):
                    self.assertNotIn(forbidden, message)

    def test_conflicting_source_markers_have_safe_deterministic_priority(self):
        prefix = "SOURCE_PLAN_UNAVAILABLE: "
        markers = ["SOURCE_CONSTRAINTS_INCOMPATIBLE", "SOURCE_VISIT_DATA_INCOMPLETE", "SOURCE_ROUTES_UNCONFIRMED", "SOURCE_SEARCH_LIMIT"]
        reasons = ["с заданными датами", "Это не означает", "дорожные связи", "безопасного предела"]
        for count in range(1, 5):
            diagnostics = [prefix + marker for marker in markers[:count]]
            self.assertIn(reasons[count - 1], generation_failure_reason(diagnostics))
            self.assertEqual(generation_failure_reason(diagnostics), generation_failure_reason(reversed(diagnostics)))

    def test_legacy_unknown_and_marker_suffix_do_not_mislabel_source_failure(self):
        for header in ("SOURCE_PLAN_UNAVAILABLE: missing facts; place_query secret",
                       "SOURCE_PLAN_UNAVAILABLE: SOURCE_NEW_CATEGORY бюджет secret",
                       "SOURCE_PLAN_UNAVAILABLE: SOURCE_SEARCH_LIMIT_extra secret"):
            with self.subTest(header=header):
                reason = generation_failure_reason([header])
                self.assertIn("Причину отказа не удалось уточнить", reason)
                for forbidden in ("secret", "place_query", "меньше дней", "недостаточно", "безопасного предела"):
                    self.assertNotIn(forbidden, reason)

    def test_coverage_requires_single_strict_bounded_integer_values(self):
        prefix = "SOURCE_PLAN_UNAVAILABLE: SOURCE_VISIT_DATA_INCOMPLETE "
        for valid in ("requestedDays=0 usableCards=100", "requestedDays=100 usableCards=0"):
            self.assertIn("Дней в запросе", generation_failure_reason([prefix + valid]))
        for value in ("True", "false", "-1", "101", "7.0", "007", "9" * 1000, "<script>"):
            for key, other in (("requestedDays", "usableCards=6"), ("usableCards", "requestedDays=7")):
                with self.subTest(value=value, key=key):
                    reason = generation_failure_reason([f"{prefix}{key}={value} {other}"])
                    self.assertNotIn("Дней в запросе", reason)
                    self.assertNotIn(value, reason)
        for suffix in ("requestedDays=7", "requestedDays=7 usableCards=6 requestedDays=8",
                       "requestedDays=7 usableCards=6 usableCards=6"):
            self.assertNotIn("Дней в запросе", generation_failure_reason([prefix + suffix]))
        self.assertNotIn("Дней в запросе", generation_failure_reason([
            prefix + "requestedDays=7 usableCards=6", prefix + "requestedDays=7 usableCards=5",
        ]))

    def test_source_edit_refusal_keeps_old_route_without_promising_ai_retry(self):
        message = generation_failure_message(["SOURCE_PLAN_UNAVAILABLE: SOURCE_SEARCH_LIMIT"], editing=True)
        self.assertIn("Старый маршрут сохранён без изменений", message)
        self.assertIn("безопасного предела", message)
        self.assertNotIn("Повторные", message)
        self.assertNotIn("меньше дней", message)


if __name__ == "__main__":
    unittest.main()
