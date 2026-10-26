"""Unknown admission prices stay visible in city and saved regional routes."""

from copy import deepcopy
import io
import socket
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # The guarded full runner supplies the bundled dependency.

import web_service
from city_estimates import (
    UNPRICED_VISIT_LABEL, budget_coverage, city_unpriced_visit_note, is_unpriced_city_visit,
    unpriced_city_visit_count,
)
from formatting import format_itinerary
from pdf_export import _budget_values, build_trip_pdf
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def price_fixture():
    request = TripRequest(destination="Тестгород, Россия", start_date="2099-10-01", end_date="2099-10-01",
                          adults=2, budget=10000, currency="RUB", travel_cost=1000, lodging_cost=2000)
    items = [
        ItineraryItem(time="09:00", title="Музей без тарифа", description="Учебное посещение.", kind="attraction", cost=0, duration_min=60),
        ItineraryItem(time="10:30", title="Театр без тарифа", description="Учебное посещение.", kind="attraction", cost=0, duration_min=60),
        ItineraryItem(time="12:00", title="Оценённое посещение", description="Учебное посещение.", kind="attraction", cost=300, duration_min=60),
        ItineraryItem(time="13:00", title="Обед", description="Учебная пауза.", kind="meal", cost=0, duration_min=60),
        ItineraryItem(time="14:00", title="Отдых", description="Учебная пауза.", kind="rest", cost=0, duration_min=30),
    ]
    itinerary = Itinerary(title="Учебный маршрут", summary="Тест отображения, не рекомендация.",
                          city_schedule=True, days=[ItineraryDay(day=1, date=request.start_date,
                          theme="Учебный день", items=items)], tips=["Уточните условия посещения."])
    return request, itinerary


class CityPriceDisplayTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "getaddrinfo"), (sqlite3, "connect")):
            blocker = patch.object(owner, name, side_effect=AssertionError("External I/O forbidden"))
            mock = blocker.start()
            self.addCleanup(blocker.stop)
            self.addCleanup(mock.assert_not_called)

    def test_unknown_count_is_zero_attractions_only_without_mutation(self):
        _, itinerary = price_fixture()
        before = deepcopy(itinerary.model_dump())
        self.assertEqual([is_unpriced_city_visit(itinerary, item) for item in itinerary.days[0].items],
                         [True, True, False, False, False])
        self.assertEqual(unpriced_city_visit_count(itinerary), 2)
        self.assertIn("Посещений с неуточнённой ценой: 2.", city_unpriced_visit_note(itinerary))
        self.assertIn("не включает неизвестную стоимость входа", city_unpriced_visit_note(itinerary))
        self.assertIn("не означает бесплатный вход", city_unpriced_visit_note(itinerary))
        self.assertEqual(itinerary.model_dump(), before)

    def test_source_and_no_unknown_prices_do_not_gain_the_disclosure(self):
        for flags in ({"city_schedule": False, "source_schedule": True},
                      {"city_schedule": True, "source_schedule": True}):
            with self.subTest(flags=flags):
                _, itinerary = price_fixture()
                itinerary = itinerary.model_copy(update=flags)
                self.assertEqual(unpriced_city_visit_count(itinerary), 0)
                self.assertEqual(city_unpriced_visit_note(itinerary), "")
        _, itinerary = price_fixture()
        for item in itinerary.days[0].items:
            if item.kind == "attraction":
                item.cost = 300
        self.assertEqual(city_unpriced_visit_note(itinerary), "")

    def test_legacy_flags_and_verified_map_coordinates_do_not_confirm_free_admission(self):
        for flags in ({}, {"city_schedule": False}, {"source_schedule": False},
                      {"city_schedule": False, "source_schedule": False},
                      {"source_schedule": "true"}):
            with self.subTest(flags=flags):
                visit = SimpleNamespace(kind="attraction", cost=0, verified=True,
                                        lat=61.78, lon=34.35, provider="osm")
                legacy = SimpleNamespace(days=[SimpleNamespace(items=[visit])], **flags)
                before = deepcopy(vars(legacy))
                self.assertTrue(is_unpriced_city_visit(legacy, visit))
                self.assertEqual(unpriced_city_visit_count(legacy), 1)
                self.assertIn("Посещений с неуточнённой ценой: 1.", city_unpriced_visit_note(legacy))
                self.assertEqual(vars(legacy), before)

    def test_telegram_labels_exactly_unknown_visits_and_preserves_totals(self):
        request, itinerary = price_fixture()
        before = itinerary.model_dump()
        text = format_itinerary(itinerary, "RUB", request.budget, travel_cost=1000, lodging_cost=2000)
        self.assertEqual(text.count(UNPRICED_VISIT_LABEL), 2)
        self.assertIn("Оценённое посещение, ~300 RUB", text)
        for line in text.splitlines():
            if "Обед" in line or "Отдых" in line:
                self.assertNotIn(UNPRICED_VISIT_LABEL, line)
        self.assertIn(city_unpriced_visit_note(itinerary), text)
        self.assertIn("Расходы на программу: около 300 RUB", text)
        self.assertIn("Итого запланировано: около 3300 RUB", text)
        self.assertIn("Остаток бюджета: 6700 RUB", text)
        self.assertEqual(itinerary.model_dump(), before)

    def test_telegram_legacy_zero_visits_are_unknown_and_source_zero_is_unchanged(self):
        for source in (False, True):
            _, itinerary = price_fixture()
            itinerary.city_schedule, itinerary.source_schedule = False, source
            before = deepcopy(itinerary.model_dump())
            text = format_itinerary(itinerary, "RUB", 10000, travel_cost=1000, lodging_cost=2000)
            self.assertEqual(text.count(UNPRICED_VISIT_LABEL), 0 if source else 2)
            self.assertEqual("Посещений с неуточнённой ценой" in text, not source)
            self.assertIn("Итого запланировано: около 3300 RUB", text)
            self.assertEqual(itinerary.model_dump(), before)

    def test_saved_regional_read_adds_admission_disclosure_without_repricing_or_rewriting(self):
        request, itinerary = price_fixture()
        request.destination_kind = "region"
        itinerary.city_schedule = False
        itinerary.days[0].base_area = "Учебная база"
        itinerary.days[0].transfer_note = "Переезд требует отдельной проверки."
        itinerary.days[0].items.append(ItineraryItem(
            kind="transport", time="15:00", title="Переезд", description="Учебный переезд.",
            cost=0, duration_min=60))
        before = deepcopy(itinerary.model_dump())
        trip = {"id": 30, "request": request.model_dump(), "revision": 1}
        with patch.object(web_service, "_validated_trip", return_value=(trip, request, itinerary)), \
                patch.object(web_service.db, "get_checklist", return_value=[]), \
                patch.object(web_service.db, "get_reminder_statuses", return_value=[]), \
                patch.object(web_service.reminders, "describe_plan", return_value=""):
            detail = web_service.get_trip_detail(1, 30)
        self.assertEqual(detail["budget"]["unpriced_visit_count"], 2)
        self.assertEqual(detail["itinerary"]["unpriced_visit_note"], city_unpriced_visit_note(itinerary))
        self.assertIn(city_unpriced_visit_note(itinerary), detail["budget"]["scope_note"])
        coverage = budget_coverage(request, itinerary)
        self.assertEqual([entry["code"] for entry in coverage["missing"]], ["admission"])
        self.assertEqual(coverage["status"], "incomplete")
        self.assertEqual((detail["budget"]["program_total"], detail["budget"]["planned_total"],
                          detail["budget"]["remaining"]), (300, 3300, 6700))
        self.assertEqual([item["cost"] for item in detail["itinerary"]["days"][0]["items"]],
                         [0, 0, 300, 0, 0, 0])
        self.assertEqual(itinerary.model_dump(), before)
        self.assertEqual(trip, {"id": 30, "request": request.model_dump(), "revision": 1})

    def test_web_detail_exposes_count_note_and_unchanged_numeric_budget(self):
        request, itinerary = price_fixture()
        before = itinerary.model_dump()
        trip = {"id": 7, "request": request.model_dump(), "revision": 1}
        with patch.object(web_service, "_validated_trip", return_value=(trip, request, itinerary)), \
                patch.object(web_service.db, "get_checklist", return_value=[]), \
                patch.object(web_service.db, "get_reminder_statuses", return_value=[]), \
                patch.object(web_service.reminders, "describe_plan", return_value=""):
            detail = web_service.get_trip_detail(1, 7)
        self.assertEqual(detail["budget"]["unpriced_visit_count"], 2)
        self.assertEqual(detail["itinerary"]["unpriced_visit_note"], city_unpriced_visit_note(itinerary))
        self.assertIn(city_unpriced_visit_note(itinerary), detail["budget"]["scope_note"])
        self.assertEqual((detail["budget"]["program_total"], detail["budget"]["planned_total"],
                          detail["budget"]["remaining"]), (300, 3300, 6700))
        self.assertEqual([item["cost"] for item in detail["itinerary"]["days"][0]["items"]], [0, 0, 300, 0, 0])
        self.assertEqual(itinerary.model_dump(), before)

    def test_web_scope_note_keeps_source_and_other_cost_warnings_and_discloses_legacy(self):
        request, itinerary = price_fixture()
        request.travel_cost = 0
        note = web_service._budget_scope_note(request, itinerary)
        self.assertIn("Не указаны и не учтены: дорога", note)
        self.assertIn(city_unpriced_visit_note(itinerary), note)
        itinerary.city_schedule = False
        for source in (False, True):
            itinerary.source_schedule = source
            self.assertEqual("Посещений с неуточнённой ценой" in
                             web_service._budget_scope_note(request, itinerary), not source)

    @unittest.skipIf(PdfReader is None, "Run with bundled dependencies for PDF extraction")
    def test_pdf_labels_unknown_admission_and_keeps_original_totals_and_payload(self):
        request, itinerary = price_fixture()
        trip = {"request": request.model_dump(), "itinerary": itinerary.model_dump()}
        before = deepcopy(trip)
        reader = PdfReader(io.BytesIO(build_trip_pdf(trip, [])))
        text = " ".join("\n".join(page.extract_text() for page in reader.pages).split())
        self.assertEqual(text.count(UNPRICED_VISIT_LABEL), 2)
        self.assertIn(city_unpriced_visit_note(itinerary), text)
        for expected in ("300 RUB", "3 300 RUB", "6 700 RUB"):
            self.assertIn(expected, text)
        self.assertEqual(_budget_values(trip["request"], itinerary)["total"], 3300)
        self.assertEqual(trip, before)
        self.assertEqual(len(reader.pages), 2)

    @unittest.skipIf(PdfReader is None, "Run with bundled dependencies for PDF extraction")
    def test_pdf_legacy_zero_visits_are_unknown_and_source_prices_stay_unchanged(self):
        request, itinerary = price_fixture()
        itinerary.city_schedule = False
        for source in (False, True):
            itinerary.source_schedule = source
            trip = {"request": request.model_dump(), "itinerary": itinerary.model_dump()}
            before = deepcopy(trip)
            reader = PdfReader(io.BytesIO(build_trip_pdf(trip, [])))
            text = " ".join("\n".join(page.extract_text() for page in reader.pages).split())
            self.assertEqual(text.count(UNPRICED_VISIT_LABEL), 0 if source else 2)
            self.assertEqual("Посещений с неуточнённой ценой" in text, not source)
            self.assertIn("3 300 RUB", text)
            self.assertEqual(trip, before)


if __name__ == "__main__":
    unittest.main()
