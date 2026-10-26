"""Synthetic four-day read-model regression, not a replay of private user data.

The validated-route boundary is mocked deliberately: these tests exercise the
web/PDF budget presentation, not map or schedule validation. No stored trip,
provider, paid model, memory service or real database is opened.
"""

from copy import deepcopy
from contextlib import ExitStack
import socket
import sqlite3
import unittest
from unittest.mock import patch

import city_estimates
import generation_quality
import pdf_export
import web_service
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def synthetic_four_day_trip():
    request = TripRequest(
        destination="Synthetic City, Test Country", start_date="2099-06-01", end_date="2099-06-04",
        adults=2, children_ages=[13], currency="USD", budget=20000,
        travel_cost=3000, lodging_cost=5000, start_time="09:00", end_time="20:00",
        interest_categories=["museums", "theatre", "cinema", "shopping_malls"],
    )
    days = []
    for number, count in enumerate((2, 3, 2, 2), start=1):
        visits = [ItineraryItem(
            time=f"{9 + index:02d}:00", kind="attraction", title=f"Synthetic place {number}-{index}",
            description="Synthetic planning fixture; admission price is unknown.", duration_min=45, cost=0.0,
        ) for index in range(count)]
        days.append(ItineraryDay(day=number, date=f"2099-06-0{number}", theme="Synthetic day", items=visits + [
            ItineraryItem(time="13:00", kind="meal", title="Lunch", description="Planning reserve.",
                          duration_min=60, cost=31.0),
            ItineraryItem(time="14:00", kind="rest", title="Rest", description="Break.",
                          duration_min=60, cost=0.0),
        ]))
    itinerary = Itinerary(title="Synthetic itinerary", summary="Synthetic read model.", days=days,
                          tips=[], city_schedule=True, source_schedule=False)
    stored = request.model_dump()
    stored[generation_quality.KEY] = {"quality_status": "complete", "quality_reasons": []}
    return request, itinerary, {"id": 17, "revision": 1, "request": stored}


class FourDayBudgetRegressionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for owner, name in ((socket.socket, "connect"), (socket, "getaddrinfo"), (sqlite3, "connect")):
            blocked = self.stack.enter_context(patch.object(owner, name, side_effect=AssertionError("No external I/O")))
            self.addCleanup(blocked.assert_not_called)
        self.request, self.itinerary, self.trip = synthetic_four_day_trip()
        self.stack.enter_context(patch.object(web_service, "_validated_trip",
                                             return_value=(self.trip, self.request, self.itinerary)))
        self.stack.enter_context(patch.object(web_service.db, "get_checklist", return_value=[]))
        self.stack.enter_context(patch.object(web_service.db, "get_reminder_statuses", return_value=[]))
        self.stack.enter_context(patch.object(web_service.reminders, "describe_plan", return_value=""))

    def detail(self):
        return web_service.get_trip_detail(101, 17)

    def test_all_four_days_are_summed_and_day_progress_is_separate_from_whole_trip(self):
        before = deepcopy((self.request.model_dump(), self.itinerary.model_dump(), self.trip))
        budget = self.detail()["budget"]
        self.assertEqual([day["day_total"] for day in budget["days"]], [31, 31, 31, 31])
        self.assertEqual([day["planned_through_day"] for day in budget["days"]], [8031, 8062, 8093, 8124])
        self.assertEqual([day["remaining_after_day"] for day in budget["days"]], [11969, 11938, 11907, 11876])
        self.assertEqual((budget["program_total"], budget["planned_total"], budget["remaining"]),
                         (124, 8124, 11876))
        pdf = pdf_export._budget_values(self.request.model_dump(), self.itinerary)
        self.assertEqual((pdf["program"], pdf["total"], pdf["remaining"]), (124, 8124, 11876))
        self.assertEqual((self.request.model_dump(), self.itinerary.model_dump(), self.trip), before)

    def test_complete_calendar_does_not_claim_complete_budget_or_free_admission(self):
        detail = self.detail()
        coverage = detail["budget"]["coverage"]
        self.assertEqual(detail["generation_quality"]["quality_status"], "complete")
        self.assertEqual(coverage["status"], "incomplete")
        self.assertEqual(detail["budget"]["unpriced_visit_count"], 9)
        self.assertTrue({"admission", "local_transport", "other_meals"}.issubset(
            {item["code"] for item in coverage["missing"]}))
        self.assertIn("не означает бесплатный вход", detail["itinerary"]["unpriced_visit_note"])
        self.assertEqual(coverage, city_estimates.budget_coverage(self.request, self.itinerary))
        self.assertEqual(coverage, city_estimates.budget_coverage(self.request.model_dump(), self.itinerary))

    def test_an_entered_positive_estimate_changes_only_its_day_without_inventing_other_prices(self):
        self.itinerary.days[1].items[0].cost = 25.0
        before = self.itinerary.model_dump()
        budget = self.detail()["budget"]
        self.assertEqual([day["day_total"] for day in budget["days"]], [31, 56, 31, 31])
        self.assertEqual((budget["program_total"], budget["planned_total"], budget["remaining"]),
                         (149, 8149, 11851))
        self.assertEqual(budget["unpriced_visit_count"], 8)
        self.assertEqual(budget["coverage"]["status"], "incomplete")
        self.assertEqual(self.itinerary.model_dump(), before)

    def test_absent_fixed_costs_are_missing_inputs_not_free_travel_and_lodging(self):
        self.request.travel_cost = self.request.lodging_cost = 0
        budget = self.detail()["budget"]
        codes = {item["code"] for item in budget["coverage"]["missing"]}
        self.assertTrue({"travel", "lodging", "admission", "local_transport", "other_meals"}.issubset(codes))
        self.assertEqual((budget["planned_total"], budget["remaining"]), (124, 19876))

    def test_legacy_zero_admission_is_unknown_but_source_confirmed_zero_is_preserved(self):
        for city_schedule, source_schedule in ((True, True), (False, False), (False, True)):
            with self.subTest(city=city_schedule, source=source_schedule):
                itinerary = self.itinerary.model_copy(deep=True)
                itinerary.city_schedule = city_schedule
                itinerary.source_schedule = source_schedule
                before = itinerary.model_dump()
                coverage = city_estimates.budget_coverage(self.request, itinerary)
                if source_schedule:
                    self.assertEqual(coverage, {"status": "estimate", "missing": [], "notes": []})
                else:
                    self.assertEqual(coverage["status"], "incomplete")
                    self.assertEqual([entry["code"] for entry in coverage["missing"]], ["admission"])
                    self.assertIn("9", coverage["missing"][0]["message"])
                self.assertEqual(itinerary.model_dump(), before)

    def test_malformed_legacy_interests_never_break_pdf_or_expose_raw_values(self):
        for interests in ({"theatre": "private-fixture"}, 7, [{"private-fixture": "cinema"}]):
            with self.subTest(kind=type(interests).__name__):
                saved = self.request.model_dump()
                saved["interest_categories"] = interests
                before = deepcopy((saved, self.itinerary.model_dump()))
                coverage = city_estimates.budget_coverage(saved, self.itinerary)
                self.assertEqual(coverage["status"], "incomplete")
                self.assertFalse(any("спектакль" in note or "Покупки" in note for note in coverage["notes"]))
                self.assertNotIn("private-fixture", str(coverage))
                content = pdf_export.build_trip_pdf({"request": saved, "itinerary": self.itinerary.model_dump()}, [])
                self.assertTrue(content.startswith(b"%PDF"))
                self.assertEqual((saved, self.itinerary.model_dump()), before)

    def test_transport_money_is_counted_per_day_and_zero_is_not_a_paid_reserve(self):
        for day in self.itinerary.days:
            day.items.append(ItineraryItem(time="15:00", kind="transport", title="Synthetic transfer",
                                            description="Planning fixture.", duration_min=15, cost=0.0))

        def transport_message():
            return next((item["message"] for item in self.detail()["budget"]["coverage"]["missing"]
                         if item["code"] == "local_transport"), None)

        self.assertIn("в 4 днях программы", transport_message())
        self.assertEqual(self.detail()["budget"]["program_total"], 124)
        self.itinerary.days[0].items[-1].cost = 5.0
        self.assertIn("в 3 днях программы", transport_message())
        budget = self.detail()["budget"]
        self.assertEqual([day["day_total"] for day in budget["days"]], [36, 31, 31, 31])
        self.assertEqual(budget["program_total"], 129)
        self.assertEqual(budget["planned_total"], 8129)
        for day in self.itinerary.days[1:]:
            day.items[-1].cost = 5.0
        self.assertIsNone(transport_message())
        self.assertEqual(self.detail()["budget"]["program_total"], 144)

    def test_other_meal_titles_are_not_falsely_described_as_lunch_only(self):
        titles = ("Breakfast", "Ужин", "Обед и ужин", "Питание")
        for day, title in zip(self.itinerary.days, titles):
            next(item for item in day.items if item.kind == "meal").title = title
        before = self.itinerary.model_dump()
        coverage = city_estimates.budget_coverage(self.request, self.itinerary)
        self.assertNotIn("other_meals", {item["code"] for item in coverage["missing"]})
        self.assertTrue(any("Питание" in note for note in coverage["notes"]))
        self.assertEqual(self.detail()["budget"]["program_total"], 124)
        self.assertEqual(self.itinerary.model_dump(), before)

    def test_pdf_displays_the_shared_disclosures_without_changing_the_saved_route(self):
        expected = city_estimates.budget_coverage(self.request, self.itinerary)
        before = self.itinerary.model_dump()
        rendered = []
        original = pdf_export._p

        def paragraph(text, *args, **kwargs):
            rendered.append(str(text))
            return original(text, *args, **kwargs)

        with patch.object(pdf_export, "_p", side_effect=paragraph):
            content = pdf_export.build_trip_pdf(
                {"request": self.request.model_dump(), "itinerary": self.itinerary.model_dump()}, [])
        self.assertTrue(content.startswith(b"%PDF"))
        output = "\n".join(rendered)
        for entry in expected["missing"]:
            self.assertIn(entry["message"], output)
        for note in expected["notes"]:
            self.assertIn(note, output)
        self.assertEqual(self.itinerary.model_dump(), before)


class SafeDayQualityRegressionTests(unittest.TestCase):
    def test_four_short_mornings_are_partial_even_when_all_calendar_days_have_visits(self):
        from city_quality import apply_city_day_coverage

        request, itinerary, _ = synthetic_four_day_trip()
        coverage = {"quality_status": "complete", "quality_reasons": [], "program_days": 4, "free_days": 0}
        before = itinerary.model_dump()
        apply_city_day_coverage(request, itinerary, coverage)
        public = generation_quality.public_quality(coverage, itinerary)
        self.assertEqual(public["quality_status"], "partial")
        self.assertEqual((public["program_days"], public["free_days"]), (4, 0))
        self.assertEqual((public["desired_program_minutes"], public["light_program_days"],
                          public["unbalanced_program_days"]), (300, 4, 4))
        self.assertEqual(set(public["quality_reasons"]), {"day_program_light", "day_window_unbalanced"})
        self.assertEqual(itinerary.model_dump(), before)

    def test_depth_and_distribution_diagnostics_use_bounded_counts_and_fixed_reasons(self):
        raw = {"desired_program_minutes": 180, "light_program_days": 2, "unbalanced_program_days": 1,
               "quality_status": "partial", "quality_reasons": ["day_program_light", "day_window_unbalanced"],
               "private_request": "must not be serialized"}
        clean = generation_quality.safe_coverage(raw)
        self.assertEqual({key: clean[key] for key in
                          ("desired_program_minutes", "light_program_days", "unbalanced_program_days")},
                         {"desired_program_minutes": 180, "light_program_days": 2, "unbalanced_program_days": 1})
        self.assertNotIn("private_request", clean)
        _, itinerary, _ = synthetic_four_day_trip()
        public = generation_quality.public_quality(raw, itinerary)
        self.assertEqual(public["quality_status"], "partial")
        self.assertEqual(set(public["reason_texts"]), {
            generation_quality.REASONS["day_program_light"], generation_quality.REASONS["day_window_unbalanced"]})
        for invalid in (-1, True, "2", 1.5, 100001):
            self.assertEqual(generation_quality.safe_coverage({"light_program_days": invalid}), {})


if __name__ == "__main__":
    unittest.main()
