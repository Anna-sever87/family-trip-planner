"""Regression checks for reserved travel/lodging and the remaining program budget."""

import unittest

from pydantic import ValidationError

import anketa
from formatting import format_itinerary
from prompts import build_itinerary_prompt
from rules import check_rules
from schemas import Itinerary, TripRequest


class TripBudgetTests(unittest.TestCase):
    def request(self, **changes):
        values = dict(destination="Воркута, Россия", start_date="2026-11-03",
                      end_date="2026-11-03", adults=2, budget=45000, currency="RUB",
                      travel_cost=15000, lodging_cost=20000)
        return TripRequest(**{**values, **changes})

    def itinerary(self, cost):
        return Itinerary(title="Программа", summary="Программа дня", days=[
            {"day": 1, "date": "2026-11-03", "theme": "Город", "items": [
                {"time": "12:30", "kind": "meal", "title": "Обед",
                 "description": "Обед на всех", "cost": cost, "indoor": True}
            ]}
        ])

    def test_program_budget_reserves_fixed_totals_once(self):
        req = self.request()
        self.assertEqual(req.fixed_costs, 35000)
        self.assertEqual(req.program_budget, 10000)
        prompt = build_itinerary_prompt(req)
        self.assertIn("Бюджет на программу: 10000 RUB", prompt)
        self.assertIn("Общий бюджет поездки: 45000 RUB", prompt)

    def test_legacy_request_keeps_existing_budget(self):
        req = self.request(travel_cost=0, lodging_cost=0)
        self.assertEqual(req.program_budget, 45000)

    def test_program_cannot_spend_the_reserved_housing_budget(self):
        req = self.request()
        problems = check_rules(self.itinerary(10000.01), req)
        self.assertTrue(any("превышают доступный бюджет" in p for p in problems))
        boundary = check_rules(self.itinerary(10000), req)
        self.assertFalse(any("превышают доступный бюджет" in p for p in boundary))

    def test_fixed_totals_reject_negative_nan_or_infinite(self):
        for field in ("travel_cost", "lodging_cost"):
            for value in (-1, float("nan"), float("inf"), 1_000_000_001):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    self.request(**{field: value})

    def test_telegram_total_budget_rejects_negative_and_overflow(self):
        self.assertEqual(anketa.parse_budget("45 000 руб"), (45000, "RUB"))
        for text in ("-45000 RUB", "9" * 400, "45000 BTC", "200 и 300", "0 RUB"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                anketa.parse_budget(text)

    def test_telegram_currency_does_not_silently_convert(self):
        self.assertEqual(anketa.parse_fixed_cost("15 000,50 руб", "RUB"), 15000.5)
        self.assertEqual(anketa.parse_fixed_cost("-", "RUB"), 0)
        for text in ("-500", "12 EUR", "20 и 30", "nan"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                anketa.parse_fixed_cost(text, "RUB")

    def test_telegram_summary_shows_full_trip_total_and_remainder(self):
        text = format_itinerary(self.itinerary(5600), "RUB", 45000,
                                travel_cost=15000, lodging_cost=20000)
        self.assertIn("Итого запланировано: около 40600 RUB", text)
        self.assertIn("Остаток бюджета: 4400 RUB", text)
        self.assertIn("Жильё за все ночи: 20000 RUB", text)


if __name__ == "__main__":
    unittest.main()
