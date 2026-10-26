"""PDF totals and link safety using offline fixtures, without AI or a database."""
import io
import unittest
from copy import deepcopy

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # Available in the bundled artifact runtime; not needed by the app.

from pdf_export import _budget_values, _verified_route_points, build_trip_pdf
from schemas import Itinerary
from formatting import format_itinerary
from program_coverage import FREE_DAY_NOTE, program_report_text


def sample_trip():
    points = [
        ("Музей истории Воркуты", 67.49640, 64.06080),
        ("Центральная площадь", 67.50080, 64.04980),
        ("Памятник первооткрывателям", 67.50470, 64.04380),
    ]
    days = []
    for number in range(1, 5):
        items = []
        for index, (title, lat, lon) in enumerate(points):
            items.append({"time": ["10:00", "16:15", "18:00"][index], "title": title,
                          "kind": "attraction", "description": "Прогулка и знакомство с историей города. "
                          "Осмотр архитектуры и памятных мест, время для фотографий.",
                          "duration_min": 45, "cost": 200 if index == 0 else 0,
                          "verified": True, "lat": lat, "lon": lon})
        items.insert(1, {"time": "13:00", "title": "Обед", "kind": "meal",
                         "description": "Горячий обед для всех участников в кафе поблизости.",
                         "duration_min": 60, "cost": 1400})
        days.append({"day": number, "date": f"2026-11-{number + 2:02d}",
                     "theme": "Городские истории и прогулки", "items": items,
                     "rainy_alternative": "При плохой погоде сократите прогулку и уточните режим работы музея."})
    return {"request": {"destination": "Воркута, Россия", "start_date": "2026-11-03", "end_date": "2026-11-06",
                        "adults": 2, "children_ages": [], "budget": 45000, "currency": "RUB",
                        "travel_cost": 11000, "lodging_cost": 18000,
                        "weather_text": "Указанные цены являются оценкой для демонстрации PDF."},
            "itinerary": {"title": "Четыре дня в Воркуте", "summary": "Демонстрационный маршрут с городской программой до 19:00.",
                          "days": days, "tips": ["Уточните режим работы мест перед поездкой.", "Возьмите тёплую одежду и удобную обувь."]}}


def limited_program_trip():
    """Synthetic display fixture, not a real travel recommendation."""
    trip = sample_trip()
    trip["itinerary"].update(source_schedule=True, program_report={
        "schema_version": 1, "reason": "limited_verified_program", "requested_days": 4,
        "program_days": 2, "free_days": 2, "eligible_unique_visits": 8, "selected_unique_visits": 6,
    })
    trip["itinerary"]["summary"] = "Предварительный план. Сохраняются все четыре даты поездки."
    for day in trip["itinerary"]["days"]:
        day["base_area"] = "Воркута, Россия"
        if day["day"] <= 2:
            day["day_mode"] = "program"
            for item in day["items"]:
                if item["kind"] == "attraction":
                    item["title"] += f" — пример дня {day['day']}"
                    item["lat"] += (day["day"] - 1) * .02
            continue
        day.update(day_mode="free", theme="Свободный день", rainy_alternative="", items=[
            {"time": "13:00", "kind": "meal", "title": "Обед", "description": "Питание учтено.",
             "duration_min": 60, "cost": 1400},
            {"time": "14:00", "kind": "rest", "title": "Отдых", "description": "Без обещанных мест.",
             "duration_min": 30, "cost": 0},
        ])
    return trip


class PdfExportTests(unittest.TestCase):
    @unittest.skipIf(PdfReader is None, "Run with bundled Python dependencies for PDF extraction")
    def test_blank_display_area_does_not_leave_empty_parentheses(self):
        for area in ("", " \t\n ", " 🌳 "):
            with self.subTest(area=area):
                trip = sample_trip()
                trip["itinerary"]["days"] = trip["itinerary"]["days"][:1]
                trip["itinerary"]["days"][0]["items"][1]["area"] = area
                original = deepcopy(trip)
                reader = PdfReader(io.BytesIO(build_trip_pdf(trip, [])))
                text = " ".join("\n".join(page.extract_text() for page in reader.pages).split())
                self.assertIn("Обед", text)
                self.assertNotRegex(text, r"Обед\s*\(\s*\)")
                self.assertEqual(trip, original)

    @unittest.skipIf(PdfReader is None, "Run with bundled Python dependencies for PDF extraction")
    def test_nonblank_display_area_keeps_parentheses_and_escaped_text(self):
        trip = sample_trip()
        trip["itinerary"]["days"] = trip["itinerary"]["days"][:1]
        trip["itinerary"]["days"][0]["items"][1]["area"] = " Центр & набережная "
        original = deepcopy(trip)
        reader = PdfReader(io.BytesIO(build_trip_pdf(trip, [])))
        text = " ".join("\n".join(page.extract_text() for page in reader.pages).split())
        self.assertIn("Обед (Центр & набережная)", text)
        self.assertEqual(trip, original)

    @unittest.skipIf(PdfReader is None, "Run with bundled Python dependencies for PDF extraction")
    def test_free_dates_keep_disclosure_food_costs_and_no_missing_map_warning(self):
        trip = limited_program_trip()
        reader = PdfReader(io.BytesIO(build_trip_pdf(trip, [])))
        text = " ".join("\n".join(page.extract_text() for page in reader.pages).split())
        for expected in ("Проверено подходящих мест: 8", "Включено: 6", "Дней с программой: 2",
                         "Дней свободного режима: 2", "Даты сохранены", "не утверждение, что других мест нет",
                         "Резерв на обед учтён во все дни; личные расходы не запланированы", "6 000 RUB", "35 000 RUB"):
            self.assertIn(expected, text)
        for day in trip["itinerary"]["days"]:
            self.assertIn(f"День {day['day']} ({day['date']})", text)
        for number in (3, 4):
            page_text = next(page.extract_text() for page in reader.pages if f"День {number} (" in page.extract_text())
            self.assertIn("Свободный режим", page_text)
            self.assertIn("точки на карте не требуются", page_text)
            self.assertIn("Обед", page_text)
            self.assertIn("Отдых", page_text)
            self.assertNotIn("нет подтверждённых координат", page_text)
        self.assertEqual(len(reader.pages), 5)

    def test_telegram_free_dates_show_the_same_server_summary_and_unchanged_totals(self):
        trip = limited_program_trip()
        itinerary = Itinerary.model_validate(trip["itinerary"])
        output = format_itinerary(itinerary, "RUB", 45000, children_ages=[], travel_cost=11000, lodging_cost=18000)
        self.assertIn(program_report_text(itinerary), output)
        self.assertIn("Проверено подходящих мест: 8. Включено: 6.", output)
        self.assertEqual(output.count(FREE_DAY_NOTE), 2)
        self.assertIn("Расходы на программу: около 6000 RUB", output)
        self.assertIn("Итого запланировано: около 35000 RUB", output)
        self.assertIn("Остаток бюджета: 10000 RUB", output)
        for day in itinerary.days:
            self.assertIn(f"День {day.day} ({day.date})", output)
        for tip in itinerary.tips:
            self.assertIn(tip, output)

    def test_legacy_telegram_output_does_not_infer_free_days_from_empty_items(self):
        trip = sample_trip()
        trip["itinerary"]["days"][0]["items"] = []
        itinerary = Itinerary.model_validate(trip["itinerary"])
        output = format_itinerary(itinerary, "RUB", 45000)
        self.assertNotIn("Программа и свободные дни", output)
        self.assertNotIn(FREE_DAY_NOTE, output)

    @unittest.skipIf(PdfReader is None, "Run with the bundled Python runtime for PDF extraction checks")
    def test_source_backed_region_bases_and_transfers_survive_pdf_export(self):
        trip = sample_trip()
        trip["request"]["destination_kind"] = "region"
        trip["itinerary"]["source_schedule"] = True
        trip["itinerary"]["days"][0]["base_area"] = "Хужир"
        trip["itinerary"]["days"][0]["transfer_note"] = "Ориентировочный переезд; сезонность и стоимость уточнить перед поездкой."
        reader = PdfReader(io.BytesIO(build_trip_pdf(trip, [])))
        text = "\n".join(page.extract_text() for page in reader.pages)
        self.assertIn("База дня: Хужир", text)
        self.assertIn("Перемещения:", text)
        self.assertIn("сезонность и стоимость уточнить", text)

    def test_all_trip_budget_includes_every_part(self):
        trip = sample_trip()
        costs = _budget_values(trip["request"], Itinerary.model_validate(trip["itinerary"]))
        self.assertEqual(costs["program"], 6400)
        self.assertEqual(costs["program_budget"], 16000)
        self.assertEqual(costs["total"], 35400)
        self.assertEqual(costs["remaining"], 9600)
        trip["request"]["budget"] = 30000
        self.assertEqual(_budget_values(trip["request"], Itinerary.model_validate(trip["itinerary"]))["remaining"], -5400)

    @unittest.skipIf(PdfReader is None, "Run with the bundled Python runtime for PDF extraction checks")
    def test_pdf_totals_and_every_verified_map_link(self):
        trip = sample_trip()
        reader = PdfReader(io.BytesIO(build_trip_pdf(trip, [])))
        text = "\n".join(page.extract_text() for page in reader.pages)
        for expected in ("Общий бюджет поездки: 45 000 RUB", "11 000 RUB", "18 000 RUB", "6 400 RUB",
                         "35 400 RUB", "9 600 RUB", "Схема маршрута дня", "без географической подложки"):
            self.assertIn(expected, text)
        links = [annotation.get_object().get("/A", {}).get("/URI", "")
                 for page in reader.pages for annotation in page.get("/Annots", [])]
        self.assertEqual(len([link for link in links if str(link).startswith("https://www.openstreetmap.org/")]), 12)
        self.assertEqual(len(reader.pages), 5)

    @unittest.skipIf(PdfReader is None, "Run with the bundled Python runtime for PDF extraction checks")
    def test_legacy_missing_costs_and_coords_are_explicit(self):
        trip = sample_trip()
        del trip["request"]["travel_cost"]
        del trip["request"]["lodging_cost"]
        trip["itinerary"]["days"] = trip["itinerary"]["days"][:1]
        for item in trip["itinerary"]["days"][0]["items"]:
            item["verified"] = False
        reader = PdfReader(io.BytesIO(build_trip_pdf(trip, [])))
        text = "\n".join(page.extract_text() for page in reader.pages)
        self.assertIn("Дорога туда и обратно (не указано)", text)
        self.assertIn("Жильё за все ночи (не указано)", text)
        self.assertIn("нет подтверждённых координат", text)
        self.assertNotIn("Открыть на OpenStreetMap", text)

    @unittest.skipIf(PdfReader is None, "Run with the bundled Python runtime for PDF extraction checks")
    def test_partial_and_invalid_coordinates_are_not_mapped(self):
        trip = sample_trip()
        day = trip["itinerary"]["days"][0]
        day["items"][0]["lat"] = float("nan")
        day["items"][2]["lon"] = 181
        parsed = Itinerary.model_validate(trip["itinerary"])
        self.assertEqual(len(_verified_route_points(parsed.days[0])), 1)
        reader = PdfReader(io.BytesIO(build_trip_pdf(trip, [])))
        self.assertIn("Показано 1 из 3 мест", "\n".join(page.extract_text() for page in reader.pages))

    @unittest.skipIf(PdfReader is None, "Run with the bundled Python runtime for PDF extraction checks")
    def test_over_budget_is_presented_as_excess(self):
        trip = sample_trip()
        trip["request"]["budget"] = 30000
        text = "\n".join(page.extract_text() for page in PdfReader(io.BytesIO(build_trip_pdf(trip, []))).pages)
        self.assertIn("Превышение бюджета", text)
        self.assertIn("5 400 RUB", text)


if __name__ == "__main__":
    unittest.main()
