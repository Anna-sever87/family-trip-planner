"""Офлайн-регрессии контракта генерации: без шаблона из трёх пунктов."""

import unittest

from prompts import (
    ADULT_ITINERARY_SYSTEM_PROMPT,
    ITINERARY_SYSTEM_PROMPT,
    build_edit_day_system_prompt,
    build_itinerary_prompt,
    build_itinerary_system_prompt,
)
from schemas import TripRequest


def _request(**updates) -> TripRequest:
    data = {
        "destination": "Байкал, Россия",
        "destination_kind": "region",
        "start_date": "2099-10-28",
        "end_date": "2099-11-03",
        "adults": 2,
        "children_ages": [],
        "budget": 60_000,
        "travel_cost": 10_000,
        "lodging_cost": 15_000,
        "currency": "RUB",
        "pace": "средний",
        "start_time": "08:00",
        "end_time": "21:00",
        "location_preference": "anywhere",
    }
    data.update(updates)
    return TripRequest(**data)


class GenerationStructurePromptTests(unittest.TestCase):
    def test_family_and_adult_have_fields_not_a_fixed_json_example(self):
        for prompt in (ITINERARY_SYSTEM_PROMPT, ADULT_ITINERARY_SYSTEM_PROMPT):
            with self.subTest(family=prompt is ITINERARY_SYSTEM_PROMPT):
                self.assertIn("строго по переданной JSON-схеме", prompt)
                self.assertIn("Поля маршрута:", prompt)
                self.assertIn("Поля дня:", prompt)
                self.assertIn("Поля пункта:", prompt)
                for field in ("base_area", "transfer_note", "items", "place_query", "duration_min", "cost"):
                    self.assertIn(field, prompt)
                self.assertNotIn('"items": [', prompt)
                self.assertNotIn('"kind": "attraction"', prompt)
                self.assertNotIn("JSON-примере", prompt)
                self.assertNotIn("из примера", prompt)

    def test_meal_and_rest_count_does_not_limit_attractions(self):
        for children in ([], [2], [10]):
            prompt = build_itinerary_system_prompt(_request(children_ages=children))
            with self.subTest(children=children):
                self.assertIn("Длина items переменная", prompt)
                self.assertIn("ровно один meal и ровно один rest", prompt)
                self.assertIn("только к обеду и отдыху", prompt)
                self.assertIn("kind = attraction может быть несколько", prompt)
                self.assertIn("Не ограничивай день тремя пунктами", prompt)
                self.assertNotIn("ровно по одному пункту каждого вида", prompt)

    def test_adult_quality_guide_is_conditional_not_a_minimum(self):
        prompt = build_itinerary_system_prompt(_request())
        self.assertIn("Для среднего темпа", prompt)
        self.assertIn("не меньше 8 часов", prompt)
        self.assertIn("ориентир — 2–3 конкретных разных attraction", prompt)
        self.assertIn("не обязательная квота", prompt)
        self.assertNotIn("не менее 2 attraction", prompt)
        self.assertNotIn("минимум 2 attraction", prompt)
        self.assertNotIn("сократи число attraction до одного или двух", prompt)

    def test_single_visit_exceptions_require_real_content_or_transfer(self):
        prompt = build_itinerary_system_prompt(_request())
        self.assertIn("Один attraction допустим", prompt)
        self.assertIn("длительного посещения или природного маршрута", prompt)
        self.assertIn("в день значительного переезда", prompt)
        self.assertIn("поясни содержание", prompt)
        self.assertIn("не растягивай короткую остановку искусственно", prompt)
        self.assertIn("не записывай дорогу в duration_min посещения или в rest", prompt)

    def test_insufficient_known_places_are_disclosed_not_filled_with_fakes(self):
        prompt = build_itinerary_system_prompt(_request())
        self.assertIn("честно объясни ограничение в summary или tips", prompt)
        self.assertIn("не добавляй выдуманные, общие или повторные места", prompt)
        self.assertIn("Не подменяй полноценный день короткой остановкой", prompt)
        self.assertIn("день может закончиться раньше", prompt)

    def test_family_age_rules_are_separate_from_adult_guidance(self):
        family = build_itinerary_system_prompt(_request(children_ages=[2]))
        adult = build_itinerary_system_prompt(_request())
        self.assertIn("до 3 лет: не более 2 достопримечательностей", family)
        self.assertIn("для детей до 6 лет не позже 19:00", family)
        self.assertNotIn("ориентир — 2–3 конкретных разных attraction", family)
        self.assertIn('Во всех пунктах без исключения ставь kids_note = ""', adult)
        self.assertNotIn("подсказку именно для детей", adult)
        self.assertNotIn("до 3 лет: не более 2 достопримечательностей", adult)

    def test_short_calm_request_preserves_actual_settings(self):
        request = _request(pace="спокойный", start_time="12:00", end_time="16:00")
        prompt = build_itinerary_prompt(request)
        self.assertIn("Темп отдыха: спокойный", prompt)
        self.assertIn("Режим каждого дня: с 12:00 до 16:00", prompt)
        self.assertIn("Для среднего темпа", build_itinerary_system_prompt(request))


class RegionalGenerationPromptTests(unittest.TestCase):
    def test_regional_system_plans_consistent_bases_before_visits(self):
        prompt = build_itinerary_system_prompt(_request())
        self.assertIn("До распределения посещений", prompt)
        self.assertIn("последовательность локальных баз на всю поездку", prompt)
        self.assertIn("одну или несколько близких баз", prompt)
        self.assertIn("не чередуй удалённые базы ради новых названий", prompt)
        self.assertIn("Название базы должно быть одинаковым", prompt)
        self.assertIn("Не подменяй выбранное озеро или регион ближайшим крупным городом", prompt)

    def test_transfers_have_endpoints_time_gap_and_seasonal_limits(self):
        prompt = build_itinerary_system_prompt(_request())
        self.assertIn("исходную и следующую базы", prompt)
        self.assertIn("реалистичную ориентировочную длительность и сезонные ограничения", prompt)
        self.assertIn("Согласуй этот переезд со свободным временем расписания", prompt)
        self.assertIn("одного текстового обещания недостаточно", prompt)
        self.assertIn("не выдавай переезд за kind = rest", prompt)
        self.assertIn("не создавай новый kind", prompt)
        self.assertIn("Не обещай водную прогулку или переправу в ноябре", prompt)

    def test_region_defaults_do_not_describe_a_city(self):
        prompt = build_itinerary_prompt(_request())
        self.assertIn("по выбранной территории, компактными группами вокруг баз дней", prompt)
        self.assertIn("познакомиться с выбранной территорией и отдохнуть", prompt)
        self.assertNotIn("по всему городу", prompt)
        self.assertNotIn("познакомиться с городом", prompt)

    def test_city_defaults_remain_unchanged(self):
        request = _request(destination="Казань, Россия", destination_kind="city")
        prompt = build_itinerary_prompt(request)
        self.assertIn("Расположение программы: по всему городу", prompt)
        self.assertIn("Цель поездки: познакомиться с городом и отдохнуть", prompt)
        self.assertNotIn("Дополнительные правила для регионального направления", build_itinerary_system_prompt(request))

    def test_stay_area_preference_and_explicit_goal_are_preserved(self):
        request = _request(location_preference="nearby", stay_area="Листвянка", trip_goal="Неспешные прогулки у воды")
        prompt = build_itinerary_prompt(request)
        self.assertIn("рядом с местом проживания: Листвянка", prompt)
        self.assertIn("Цель поездки: Неспешные прогулки у воды", prompt)
        self.assertNotIn("по выбранной территории, компактными группами", prompt)

    def test_day_edit_keeps_regional_contract_and_single_day_output(self):
        prompt = build_edit_day_system_prompt(_request())
        self.assertIn("последовательность локальных баз на всю поездку", prompt)
        self.assertIn("исходную и следующую базы", prompt)
        self.assertIn("Верни только один полный JSON-объект целевого дня", prompt)
        self.assertNotIn("Поля маршрута:", prompt)


if __name__ == "__main__":
    unittest.main()
