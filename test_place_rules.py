"""Offline regressions for concrete attraction names and map queries."""

from __future__ import annotations

import unittest

from prompts import (
    EDIT_DAY_SYSTEM_RULES,
    ITINERARY_SYSTEM_PROMPT,
    ROUTE_QUALITY_SYSTEM_RULES,
    build_edit_day_system_prompt,
    build_itinerary_prompt,
    build_itinerary_system_prompt,
)
from rules import (
    _normalize_place_name,
    check_rules,
    check_verified_place_duplicates,
)
from schemas import Itinerary, TripRequest


def _request() -> TripRequest:
    return TripRequest(
        destination="Магадан, Россия",
        start_date="2026-12-01",
        end_date="2026-12-01",
        adults=2,
        children_ages=[],
        budget=45_000,
        currency="RUB",
        pace="насыщенный",
        start_time="10:00",
        end_time="19:00",
        trip_goal="Познакомиться с городом",
        interest_categories=["main_sights", "museums", "shopping_malls"],
        location_preference="anywhere",
        stay_area="",
        wishes="",
    )


def _regional_request() -> TripRequest:
    return _request().model_copy(
        update={
            "destination": "Байкал, Россия",
            "destination_kind": "region",
            "start_date": "2026-11-03",
            "end_date": "2026-11-06",
            "trip_goal": "Познакомиться с природой региона",
        }
    )


def _itinerary(
    title: str,
    place_query: str,
    *,
    verified: bool = False,
    official_name: str = "",
) -> Itinerary:
    return Itinerary.model_validate(
        {
            "title": "Знакомство с Магаданом",
            "summary": "Однодневная программа по городу.",
            "days": [
                {
                    "day": 1,
                    "date": "2026-12-01",
                    "theme": "Городские места",
                    "items": [
                        {
                            "time": "10:00",
                            "kind": "attraction",
                            "title": title,
                            "area": "Центр",
                            "place_query": place_query,
                            "description": "Посещение конкретного городского объекта.",
                            "duration_min": 90,
                            "cost": 0,
                            "kids_note": "",
                            "indoor": True,
                            "verified": verified,
                            "lat": 59.56 if verified else None,
                            "lon": 150.80 if verified else None,
                            "official_name": official_name,
                        },
                        {
                            "time": "12:30",
                            "kind": "meal",
                            "title": "Обед",
                            "area": "Центр",
                            "place_query": "",
                            "description": "Обед в городском кафе.",
                            "duration_min": 60,
                            "cost": 1400,
                            "kids_note": "",
                            "indoor": True,
                        },
                        {
                            "time": "13:30",
                            "kind": "rest",
                            "title": "Отдых",
                            "area": "Центр",
                            "place_query": "",
                            "description": "Спокойный перерыв после обеда.",
                            "duration_min": 60,
                            "cost": 0,
                            "kids_note": "",
                            "indoor": True,
                        },
                    ],
                    "rainy_alternative": "",
                }
            ],
            "tips": ["Проверить режим работы перед посещением."],
        }
    )


def _generic_name_problems(itinerary: Itinerary) -> list[str]:
    return [
        problem
        for problem in check_rules(itinerary, _request())
        if "слишком общее название" in problem
    ]


def _two_day_itinerary(
    first_title: str,
    first_query: str,
    second_title: str,
    second_query: str,
) -> Itinerary:
    route = _itinerary(first_title, first_query)
    second_day = route.days[0].model_copy(deep=True)
    second_day.day = 2
    second_day.date = "2026-12-02"
    second_attraction = next(
        item
        for item in second_day.items
        if item.kind == "attraction"
    )
    second_attraction.title = second_title
    second_attraction.place_query = second_query
    route.days.append(second_day)
    return route


class PlaceNameNormalizationTests(unittest.TestCase):
    def test_cyrillic_short_i_survives_diacritic_normalization(self) -> None:
        expected = {
            "Торговый центр": "торговый центр",
            "Исторический центр": "исторический центр",
            "Местный парк": "местный парк",
            "Детский парк": "детский парк",
            "Йошкар-Ола": "йошкар ола",
        }

        for raw, normalized in expected.items():
            with self.subTest(raw=raw):
                self.assertEqual(_normalize_place_name(raw), normalized)

    def test_generic_labels_with_short_i_are_rejected_stably(self) -> None:
        for generic_title in (
            "Торговый центр",
            "ТОРГОВЫЙ-ЦЕНТР",
            "Исторический центр",
            "Местный парк",
            "Детский парк",
        ):
            with self.subTest(generic_title=generic_title):
                route = _itinerary(
                    generic_title,
                    "Торговый дом «Север», Магадан, Россия",
                )
                self.assertTrue(_generic_name_problems(route))

    def test_store_category_is_not_a_concrete_title(self) -> None:
        for generic_title in (
            "Универмаг",
            "Shopping center",
            "Department store",
        ):
            with self.subTest(generic_title=generic_title):
                route = _itinerary(
                    generic_title,
                    "Торговый дом «Север», Магадан, Россия",
                )
                self.assertTrue(_generic_name_problems(route))

    def test_bare_waterfront_category_is_not_a_concrete_place(self) -> None:
        for generic_title in (
            "Набережная",
            "Waterfront",
            "Embankment",
        ):
            with self.subTest(generic_title=generic_title):
                route = _itinerary(
                    generic_title,
                    "Набережная Цесаревича, Владивосток, Россия",
                )
                self.assertTrue(_generic_name_problems(route))

    def test_named_waterfront_is_allowed(self) -> None:
        route = _itinerary(
            "Набережная Цесаревича",
            "Набережная Цесаревича, Владивосток, Россия",
        )

        self.assertEqual(_generic_name_problems(route), [])

    def test_concrete_retailer_name_is_allowed(self) -> None:
        route = _itinerary(
            "Торговый дом «Север»",
            "Торговый дом «Север», Магадан, Россия",
        )

        self.assertEqual(_generic_name_problems(route), [])

    def test_generic_query_is_rejected_even_after_a_map_match(self) -> None:
        route = _itinerary(
            "Торговый дом «Север»",
            "Торговый центр, Владивосток, Россия",
            verified=True,
            official_name=(
                "Торговый дом «Север», Владивосток, Россия"
            ),
        )

        self.assertTrue(_generic_name_problems(route))

    def test_specific_foreign_name_and_address_query_are_not_rejected(self) -> None:
        cases = (
            (
                "Musée d'Orsay",
                "Musée d'Orsay, Paris, France",
            ),
            (
                "東京ステーションギャラリー",
                "〒100-0005 東京都千代田区丸の内1-9-1",
            ),
        )

        for title, query in cases:
            with self.subTest(title=title):
                self.assertEqual(
                    _generic_name_problems(_itinerary(title, query)),
                    [],
                )


class PlaceDuplicateTests(unittest.TestCase):
    def test_same_title_in_different_localities_is_allowed(self) -> None:
        route = _two_day_itinerary(
            "Набережная",
            "Набережная Листвянки, Листвянка, Россия",
            "Набережная",
            "Набережная Байкальска, Байкальск, Россия",
        )
        request = _request().model_copy(
            update={"end_date": "2026-12-02"}
        )

        duplicate_problems = [
            problem
            for problem in check_rules(route, request)
            if "повторяется место" in problem
        ]

        self.assertEqual(duplicate_problems, [])

    def test_same_full_place_query_is_rejected_before_map_check(self) -> None:
        place_query = "Набережная Листвянки, Листвянка, Россия"
        route = _two_day_itinerary(
            "Набережная Листвянки",
            place_query,
            "Прогулочная набережная",
            place_query,
        )
        request = _request().model_copy(
            update={"end_date": "2026-12-02"}
        )

        duplicate_problems = [
            problem
            for problem in check_rules(route, request)
            if "повторяется место" in problem
        ]

        self.assertEqual(len(duplicate_problems), 1)

    def test_same_official_name_far_apart_is_not_a_duplicate(self) -> None:
        route = _two_day_itinerary(
            "Набережная Листвянки",
            "Набережная Листвянки, Листвянка, Россия",
            "Набережная Байкальска",
            "Набережная Байкальска, Байкальск, Россия",
        )
        first = route.days[0].items[0]
        second = route.days[1].items[0]
        first.verified = second.verified = True
        first.official_name = second.official_name = "Набережная"
        first.lat, first.lon = 51.8535, 104.8693
        second.lat, second.lon = 51.5183, 104.1518

        self.assertEqual(check_verified_place_duplicates(route), [])

    def test_different_names_at_same_coordinates_are_a_duplicate(self) -> None:
        route = _two_day_itinerary(
            "Набережная Листвянки",
            "Набережная Листвянки, Листвянка, Россия",
            "Причал Листвянка",
            "Причал Листвянка, Листвянка, Россия",
        )
        first = route.days[0].items[0]
        second = route.days[1].items[0]
        first.verified = second.verified = True
        first.official_name = "Набережная Листвянки"
        second.official_name = "Причал Листвянка"
        first.lat = second.lat = 51.8535
        first.lon = second.lon = 104.8693

        duplicate_problems = check_verified_place_duplicates(route)

        self.assertEqual(len(duplicate_problems), 1)
        self.assertIn("одним и тем же местом", duplicate_problems[0])


class PlacePromptTests(unittest.TestCase):
    def test_long_adult_region_system_has_flexible_program_guidance(self) -> None:
        request = _regional_request().model_copy(
            update={"pace": "средний", "start_time": "08:00", "end_time": "21:00"}
        )
        system_prompt = build_itinerary_system_prompt(request)
        self.assertIn("ориентир — 2–3 конкретных разных attraction", system_prompt)
        self.assertIn("не меньше 8 часов", system_prompt)
        self.assertIn("длительного посещения или природного маршрута", system_prompt)
        self.assertIn("не обязательная квота", system_prompt)
        self.assertIn("не растягивай короткую остановку искусственно", system_prompt)
        self.assertIn("не добавляй выдуманные, общие или повторные места", system_prompt)
        family_prompt = build_itinerary_system_prompt(
            request.model_copy(update={"children_ages": [2]})
        )
        self.assertNotIn("ориентир — 2–3 конкретных разных attraction", family_prompt)
        for prompt in (system_prompt, family_prompt):
            with self.subTest(family=prompt is family_prompt):
                self.assertIn("Длина items переменная", prompt)
                self.assertNotIn('"items": [', prompt)
                self.assertNotIn("ровно по одному пункту каждого вида", prompt)
                self.assertIn("только к обеду и отдыху", prompt)

    def test_prompts_require_specific_consistent_official_names(self) -> None:
        for prompt in (
            ITINERARY_SYSTEM_PROMPT,
            ROUTE_QUALITY_SYSTEM_RULES,
            EDIT_DAY_SYSTEM_RULES,
        ):
            lowered = prompt.casefold()
            with self.subTest(prompt=prompt[:40]):
                self.assertIn("торговый центр", lowered)
                self.assertIn("<точное официальное название>", prompt)
                self.assertIn("title", prompt)
                self.assertIn("place_query", prompt)
                self.assertNotIn(
                    "Центральный универмаг, Магадан, Россия",
                    prompt,
                )

    def test_prompts_forbid_invented_museums_and_department_stores(self) -> None:
        for prompt in (
            ITINERARY_SYSTEM_PROMPT,
            ROUTE_QUALITY_SYSTEM_RULES,
            EDIT_DAY_SYSTEM_RULES,
        ):
            lowered = prompt.casefold()
            with self.subTest(prompt=prompt[:40]):
                self.assertIn("не придумывай", lowered)
                self.assertIn("музе", lowered)
                self.assertIn("универмаг", lowered)


class RegionalDestinationPromptTests(unittest.TestCase):
    def test_missing_destination_kind_keeps_city_prompt(self) -> None:
        request = _request()

        self.assertNotIn(
            "Дополнительные правила для регионального направления",
            build_itinerary_system_prompt(request),
        )
        self.assertNotIn(
            "Тип направления: регион или территория",
            build_itinerary_prompt(request),
        )

    def test_region_prompt_uses_concrete_poi_and_locality(self) -> None:
        request = _regional_request()
        system_prompt = build_itinerary_system_prompt(request)
        user_prompt = build_itinerary_prompt(request)

        self.assertIn(
            "Дополнительные правила для регионального направления",
            system_prompt,
        )
        self.assertIn(
            "<ближайший населённый пункт или конкретная "
            "географическая местность>",
            system_prompt,
        )
        self.assertIn(
            "place_query «Байкал, Россия» не обозначает "
            "конкретный объект",
            system_prompt,
        )
        self.assertIn("локальной базой дня", system_prompt)
        self.assertIn("дневной кластер", system_prompt)
        self.assertIn("base_area", system_prompt)
        self.assertIn("transfer_note", system_prompt)
        self.assertIn("весь регион или все берега Байкала", system_prompt)
        self.assertIn("Тип направления: регион или территория", user_prompt)
        self.assertIn("ближайшим населённым пунктом", user_prompt)
        self.assertIn("Заполни base_area", user_prompt)
        self.assertIn("При смене базы заполни transfer_note", user_prompt)

    def test_region_prompt_models_transfers_without_new_item_kind(self) -> None:
        request = _regional_request()
        system_prompt = build_itinerary_system_prompt(request)

        self.assertIn("не создавай новый kind", system_prompt)
        self.assertIn("не выдавай переезд за kind = rest", system_prompt)
        self.assertIn("промежутками между временем пунктов", system_prompt)
        self.assertIn("границы времени дня", system_prompt)
        self.assertIn("возрастные лимиты", system_prompt)
        self.assertIn("программный бюджет", system_prompt)
        self.assertIn("первого связанного attraction", system_prompt)
        self.assertIn("его description", system_prompt)
        self.assertIn("Не обещай внутренний транспорт бесплатным", system_prompt)

    def test_region_prompt_does_not_promise_november_boat(self) -> None:
        system_prompt = build_itinerary_system_prompt(
            _regional_request()
        )

        self.assertIn("Сезонный транспорт", system_prompt)
        self.assertIn("лодки", system_prompt)
        self.assertIn("в ноябре", system_prompt)
        self.assertIn("надёжно известной доступности", system_prompt)

    def test_region_edit_prompt_inherits_same_constraints(self) -> None:
        edit_prompt = build_edit_day_system_prompt(
            _regional_request()
        )

        self.assertIn(
            "Дополнительные правила для регионального направления",
            edit_prompt,
        )
        self.assertIn("не выдавай переезд за kind = rest", edit_prompt)
        self.assertIn(
            "город или ближайший населённый пункт либо "
            "конкретная местность",
            edit_prompt,
        )
        self.assertIn("base_area, transfer_note", edit_prompt)


if __name__ == "__main__":
    unittest.main()
