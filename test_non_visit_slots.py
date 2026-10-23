"""Meal/rest labels cannot disguise themselves as named attractions."""

import unittest

from rules import check_rules, is_generic_attraction
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def _request():
    return TripRequest(
        destination="Сочи, Россия", start_date="2099-10-28", end_date="2099-10-28",
        adults=2, children_ages=[5], budget=30_000, currency="RUB",
        start_time="09:00", end_time="19:00", pace="средний",
    )


def _route(title="Сочинский художественный музей имени Д. Д. Жилинского", query=None):
    return Itinerary(title="Программа", summary="Проверка посещений", days=[
        ItineraryDay(day=1, date="2099-10-28", theme="История города", items=[
            ItineraryItem(
                kind="attraction", time="09:00", title=title,
                place_query=query or f"{title}, Сочи, Россия",
                description="Посещение", duration_min=90, indoor=True,
                kids_note="Пауза по необходимости",
            ),
            ItineraryItem(kind="meal", time="12:00", title="Обед", description="Обед",
                          duration_min=60, cost=1800),
            ItineraryItem(kind="rest", time="13:00", title="Отдых", description="Пауза",
                          duration_min=60),
        ]),
    ])


class NonVisitSlotTests(unittest.TestCase):
    def test_exact_meal_and_rest_titles_are_rejected_regardless_of_query(self):
        for title in ("Обед", "Отдых"):
            for query in ("", "Кафе, Сочи, Россия", "Парк Ривьера, Сочи, Россия",
                          "Сочинский художественный музей имени Д. Д. Жилинского, Сочи, Россия"):
                with self.subTest(title=title, query=query):
                    self.assertTrue(is_generic_attraction(title, query))

    def test_case_whitespace_and_wrapping_punctuation_do_not_hide_a_pause(self):
        for title in ("  ОБЕД  ", "отдых", "«Обед»", "  (Отдых) ", "Отдых!"):
            with self.subTest(title=title):
                self.assertTrue(is_generic_attraction(title, "Парк Ривьера, Сочи, Россия"))

    def test_named_places_containing_pause_words_are_not_rejected(self):
        for title in ("Дом отдыха", "Дом отдыха «Сосны»", "Музей истории отдыха",
                      "Музей «Обед»", "Отдых (картина)", "Обед (скульптура)"):
            with self.subTest(title=title):
                self.assertFalse(is_generic_attraction(title, f"{title}, Сочи, Россия"))

    def test_non_overlapping_fake_visits_fail_rules_even_with_specific_queries(self):
        for title in ("Обед", "Отдых"):
            with self.subTest(title=title):
                problems = check_rules(_route(title, "Парк Ривьера, Сочи, Россия"), _request())
                self.assertTrue(any("слишком общее название" in problem and f"«{title}»" in problem
                                    for problem in problems), problems)
                self.assertFalse(any("пересека" in problem for problem in problems), problems)

    def test_real_visit_with_normal_meal_and_rest_remains_valid(self):
        self.assertEqual(check_rules(_route(), _request()), [])


if __name__ == "__main__":
    unittest.main()
