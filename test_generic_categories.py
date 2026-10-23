"""Exact category labels must not pass as named attractions."""

import unittest

from rules import check_rules, is_generic_attraction
from schemas import INTEREST_LABELS, Itinerary, ItineraryDay, ItineraryItem, TripRequest


OBSERVED_CATEGORIES = (
    "Природные достопримечательности",
    "Сувенирные лавки",
    "Природные тропы",
    "Музеи и выставки",
)


class GenericCategoryTests(unittest.TestCase):
    def test_observed_placeholders_are_rejected_even_with_empty_query(self):
        for title in OBSERVED_CATEGORIES:
            with self.subTest(title=title):
                self.assertTrue(is_generic_attraction(title, ""))

    def test_all_questionnaire_interest_labels_are_categories(self):
        for category, title in INTEREST_LABELS.items():
            with self.subTest(category=category, title=title):
                self.assertTrue(is_generic_attraction(title, f"{title}, Листвянка, Россия"))

    def test_unambiguous_component_labels_are_also_categories(self):
        for title in (
            "Достопримечательности", "Музеи", "Выставки", "Парки", "Природа",
            "Зоопарки", "Аквариумы", "Торговые центры", "Шопинг",
            "Морские прогулки", "Речные прогулки",
        ):
            with self.subTest(title=title):
                self.assertTrue(is_generic_attraction(title, ""))

    def test_case_whitespace_and_punctuation_do_not_disguise_a_category(self):
        for title in ("  ПРИРОДНЫЕ   ТРОПЫ ", "Сувенирные-лавки", "«Музеи и выставки»", "ПРИРОДНЫЕ ДОСТОПРИМЕЧАТЕЛЬНОСТИ"):
            with self.subTest(title=title):
                self.assertTrue(is_generic_attraction(title, ""))

    def test_specific_title_does_not_hide_a_generic_query_head(self):
        for title in OBSERVED_CATEGORIES:
            with self.subTest(title=title):
                self.assertTrue(is_generic_attraction("Байкальский музей ИНЦ СО РАН", f"{title}, Листвянка, Россия"))

    def test_specific_query_does_not_hide_a_generic_title(self):
        for title in OBSERVED_CATEGORIES:
            with self.subTest(title=title):
                self.assertTrue(is_generic_attraction(title, "Байкальский музей ИНЦ СО РАН, Листвянка, Россия"))

    def test_named_waterfronts_trails_and_museums_are_not_categories(self):
        for title, locality in (
            ("Набережная Цесаревича", "Владивосток"),
            ("Набережная Листвянки", "Листвянка"),
            ("Набережная Байкальска", "Байкальск"),
            ("Большая Байкальская тропа", "Листвянка"),
            ("Байкальский музей ИНЦ СО РАН", "Листвянка"),
            ("Парк Горького", "Москва"),
            ("Московский зоопарк", "Москва"),
            ("Торговый дом «Север»", "Магадан"),
        ):
            with self.subTest(title=title):
                self.assertFalse(is_generic_attraction(title, f"{title}, {locality}, Россия"))

    def test_named_foreign_places_remain_eligible_for_map_verification(self):
        for title, query in (
            ("Musée d'Orsay", "Musée d'Orsay, Paris, France"),
            ("Museum of Natural History", "Museum of Natural History, London, United Kingdom"),
            ("東京ステーションギャラリー", "〒100-0005 東京都千代田区丸の内1-9-1"),
        ):
            with self.subTest(title=title):
                self.assertFalse(is_generic_attraction(title, query))

    def test_real_rule_check_reports_observed_categories_before_map_stage(self):
        request = TripRequest(
            destination="Байкал, Россия", destination_kind="region",
            start_date="2099-10-28", end_date="2099-10-28",
            adults=2, children_ages=[], budget=60_000, currency="RUB",
            start_time="08:00", end_time="21:00",
        )
        for title in OBSERVED_CATEGORIES:
            route = Itinerary(title="Байкал", summary="Программа", days=[
                ItineraryDay(day=1, date="2099-10-28", theme="День у озера", base_area="Листвянка", items=[
                    ItineraryItem(time="09:00", kind="attraction", title=title, place_query="", description="Программа", duration_min=60),
                    ItineraryItem(time="12:00", kind="meal", title="Обед", description="Обед", duration_min=60, cost=2000),
                    ItineraryItem(time="13:00", kind="rest", title="Отдых", description="Пауза", duration_min=60),
                ]),
            ])
            with self.subTest(title=title):
                problems = check_rules(route, request)
                self.assertTrue(any("слишком общее название" in problem and f"«{title}»" in problem for problem in problems), problems)


if __name__ == "__main__":
    unittest.main()
