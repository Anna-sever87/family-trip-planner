"""Foreign POI labels are display-only in Telegram, reminders and exported PDF."""

import ast
from copy import deepcopy
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pypdf import PdfReader
from reportlab.pdfbase import pdfmetrics

import pdf_export
from city_candidates import CityCandidate
from city_context import CONTEXT_KEY, build_city_context
from formatting import format_itinerary
from pdf_export import _verified_route_points, build_trip_pdf
from place_labels import build_place_labels, display_place_title
from reminders import build_reminder_text
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def foreign_trip(*, title="合成博物馆", russian="Синтетический музей"):
    """Synthetic Chinese museum, not a travel recommendation or live map result."""
    request = TripRequest(destination="Пекин, Китай", start_date="2099-06-01", end_date="2099-06-01",
                          adults=2, children_ages=[], budget=50000, currency="RUB")
    query = f"{title}, Пекин, Китай"
    details = {"name": title}
    if russian:
        details["name:ru"] = russian
    record = {"name": query, "namedetails": details, "lat": 39.90, "lon": 116.40,
              "category": "tourism", "type": "museum", "osm_type": "node", "osm_id": 987654321,
              "provider": "osm", "address": {"city": "Пекин", "country": "Китай", "country_code": "cn"}}
    candidate = CityCandidate("osm-node-987654321", title, query, "Пекин", record)
    context = build_city_context(request, [candidate],
                                 {"name": "Пекин", "lat": 39.90, "lon": 116.40,
                                  "country_code": "CN", "country": "Китай"})
    item = ItineraryItem(time="10:00", kind="attraction", title=title, place_query=query, area="Пекин",
                         description="Синтетическая проверка отображения.", duration_min=90, cost=0,
                         verified=True, lat=record["lat"], lon=record["lon"], official_name=query, provider="osm")
    meal = ItineraryItem(time="13:00", kind="meal", title="Lunch", description="Перерыв", duration_min=60)
    itinerary = Itinerary(title="Проверка отображения", summary="Только автономный тест.", city_schedule=True,
                          days=[ItineraryDay(day=1, date=request.start_date, theme="Музей", items=[item, meal])])
    return {"id": 9, "request": {**request.model_dump(), CONTEXT_KEY: context}, "itinerary": itinerary.model_dump()}


class PlaceLabelRenderingTests(unittest.TestCase):
    def test_legacy_attribute_itinerary_needs_no_model_dump_or_regional_fields(self):
        item = SimpleNamespace(time="12:00", kind="attraction", title="Legacy Museum", area="", cost=0,
                               description="Старый маршрут.", kids_note="", verified=False)
        day = SimpleNamespace(day=1, date="2099-06-01", theme="Город", items=[item], rainy_alternative="")
        itinerary = SimpleNamespace(title="Город", summary="Старая версия.", days=[day], tips=[])
        before = deepcopy(itinerary)
        for saved_request in (None, {CONTEXT_KEY: {}}):
            with self.subTest(request=saved_request):
                text = format_itinerary(itinerary, "RUB", 10000, saved_request=saved_request)
                self.assertIn("Legacy Museum (название на языке источника)", text)
                self.assertNotIn("База дня:", text)
                self.assertNotIn("Переезд:", text)
                self.assertEqual(itinerary, before)

    def test_shared_label_in_telegram_reminder_and_pdf_without_identity_changes(self):
        trip = foreign_trip()
        before = deepcopy(trip)
        itinerary = Itinerary.model_validate(trip["itinerary"])
        expected = "合成博物馆 (Синтетический музей)"
        labels = build_place_labels(trip)
        self.assertEqual(display_place_title(itinerary.days[0].items[0], labels), expected)
        telegram = format_itinerary(itinerary, "RUB", 50000, saved_request=trip["request"])
        reminder = build_reminder_text("day1", trip, [])
        pdf = PdfReader(io.BytesIO(build_trip_pdf(trip, [])))
        pdf_text = " ".join("\n".join(page.extract_text() for page in pdf.pages).split())
        self.assertIn(expected, telegram)
        self.assertIn(expected, reminder)
        # The route table and its map legend use the same display projection.
        self.assertEqual(pdf_text.count(expected), 2)
        self.assertIn("Lunch", telegram)
        self.assertNotIn("Lunch (", telegram)
        self.assertEqual(trip, before)
        self.assertEqual(itinerary.model_dump(), before["itinerary"])

    def test_verified_map_copy_keeps_original_title_coordinates_and_source(self):
        trip = foreign_trip()
        day = Itinerary.model_validate(trip["itinerary"]).days[0]
        original = day.model_dump()
        old = _verified_route_points(day)
        projected = _verified_route_points(day, build_place_labels(trip))
        self.assertEqual(len(projected), 1)
        self.assertEqual({key: value for key, value in old[0].items() if key != "display_title"},
                         {key: value for key, value in projected[0].items() if key != "display_title"})
        self.assertEqual(projected[0]["title"], "合成博物馆")
        self.assertEqual(projected[0]["display_title"], "合成博物馆 (Синтетический музей)")
        self.assertTrue(projected[0]["source_url"].startswith("https://www.openstreetmap.org/"))
        self.assertEqual(day.model_dump(), original)

    def test_legacy_and_invalid_context_cannot_invent_russian_name_or_category(self):
        for invalid in (None, {}, {"version": 1}):
            with self.subTest(context=invalid):
                trip = foreign_trip()
                if invalid is None:
                    del trip["request"][CONTEXT_KEY]
                else:
                    trip["request"][CONTEXT_KEY] = invalid
                before = deepcopy(trip)
                itinerary = Itinerary.model_validate(trip["itinerary"])
                telegram = format_itinerary(itinerary, "RUB", 50000, saved_request=trip["request"])
                reminder = build_reminder_text("day1", trip, [])
                for text in (telegram, reminder):
                    self.assertIn("合成博物馆 (название на языке источника)", text)
                    self.assertNotIn("Синтетический музей", text)
                self.assertEqual(trip, before)

    def test_type_explanation_is_not_a_claim_of_translation(self):
        trip = foreign_trip(russian="")
        output = format_itinerary(Itinerary.model_validate(trip["itinerary"]), "RUB", 50000,
                                  saved_request=trip["request"])
        self.assertIn("合成博物馆 (музей)", output)
        self.assertNotIn("Синтетический музей", output)

    def test_pdf_escapes_display_text_and_keeps_link_attribution(self):
        trip = foreign_trip(title="Synthetic & Test Museum", russian="Музей теста")
        original = deepcopy(trip)
        pdf = PdfReader(io.BytesIO(build_trip_pdf(trip, [])))
        text = " ".join("\n".join(page.extract_text() for page in pdf.pages).split())
        self.assertEqual(text.count("Synthetic & Test Museum (Музей теста)"), 2)
        links = [annotation.get_object().get("/A", {}).get("/URI", "")
                 for page in pdf.pages for annotation in page.get("/Annots", [])]
        self.assertEqual(sum(str(link).startswith("https://www.openstreetmap.org/") for link in links), 1)
        self.assertEqual(trip, original)

    def test_legacy_reminder_with_partial_itinerary_stays_available(self):
        trip = {"id": 9, "destination": "Пекин", "request": {},
                "itinerary": {"days": [{"items": [{"title": "合成博物馆", "kind": "attraction"},
                                                  {"title": "Обед", "kind": "meal"}]}]}}
        before = deepcopy(trip)
        result = build_reminder_text("day1", trip, [])
        self.assertIn("合成博物馆 (название на языке источника)", result)
        self.assertIn("Обед", result)
        self.assertEqual(trip, before)

    def test_saved_bot_trip_passes_its_server_context_to_real_formatter(self):
        trip = foreign_trip()
        namespace = {"bot": Mock(), "db": Mock(), "_trip_number": Mock(return_value=9),
                     "Itinerary": Itinerary, "format_itinerary": Mock(wraps=format_itinerary), "send_parts": Mock()}
        namespace["db"].get_trip.return_value = trip
        tree = ast.parse(Path(__file__).with_name("bot.py").read_text(encoding="utf-8"))
        node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == "handle_trip")
        node.decorator_list = []
        exec(compile(ast.Module(body=[node], type_ignores=[]), "bot.py", "exec"), namespace)
        namespace["handle_trip"](SimpleNamespace(chat=SimpleNamespace(id=101)))
        self.assertIs(namespace["format_itinerary"].call_args.kwargs["saved_request"], trip["request"])
        self.assertIn("合成博物馆 (Синтетический музей)", namespace["send_parts"].call_args.args[1])

    def test_all_three_bot_paths_pass_saved_context_without_changing_itinerary(self):
        tree = ast.parse(Path(__file__).with_name("bot.py").read_text(encoding="utf-8"))
        for name, request_name in (("handle_trip", "request"), ("generation_job", "request_dict"),
                                   ("edit_generation_job", "request_dict")):
            with self.subTest(handler=name):
                function = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
                calls = [item for item in ast.walk(function) if isinstance(item, ast.Call)
                         and isinstance(item.func, ast.Name) and item.func.id == "format_itinerary"]
                self.assertEqual(len(calls), 1)
                keywords = {kw.arg: kw.value for kw in calls[0].keywords}
                self.assertIsInstance(keywords["saved_request"], ast.Name)
                self.assertEqual(keywords["saved_request"].id, request_name)

    def test_pdf_has_embedded_cjk_glyphs_not_only_a_unicode_text_map(self):
        reader = PdfReader(io.BytesIO(build_trip_pdf(foreign_trip(), [])))
        font = pdfmetrics.getFont(pdf_export._cjk_font)
        self.assertTrue(all(font.face.charToGlyph.get(ord(char)) for char in "合成博物馆"))
        font_name = font.face.name.decode("ascii") if isinstance(font.face.name, bytes) else font.face.name
        descriptors = [value.get_object().get("/FontDescriptor")
                       for page in reader.pages for value in page["/Resources"]["/Font"].get_object().values()]
        cjk = [value.get_object() for value in descriptors if value is not None
               and font_name in str(value.get_object().get("/FontName", ""))]
        self.assertTrue(cjk)
        self.assertTrue(all(value.get("/FontFile2") is not None for value in cjk))

    def test_pdf_font_markup_escapes_source_text_and_never_enters_uri_attributes(self):
        markup = pdf_export._text_markup('合成博物馆 & <script> "test"')
        self.assertIn('<font name="TripCJK">合成博物馆</font>', markup)
        self.assertIn('&amp; &lt;script&gt;', markup)
        self.assertNotIn('<script>', markup)
        self.assertEqual(pdf_export._esc('https://example.invalid/合成博物馆?a=1&b=2'),
                         'https://example.invalid/合成博物馆?a=1&amp;b=2')
        self.assertEqual(pdf_export._clean('\U00020000🙂'), '\U00020000')

    def test_missing_cjk_font_fails_explicitly_instead_of_printing_empty_boxes(self):
        with patch.object(pdf_export, '_cjk_font', None), patch.object(pdf_export, 'CJK_FONT_CANDIDATES', ()):
            self.assertEqual(pdf_export._text_markup('Русское название'), 'Русское название')
            with self.assertRaisesRegex(RuntimeError, 'fonts-wqy-zenhei'):
                pdf_export._text_markup('合成博物馆')


if __name__ == "__main__":
    unittest.main()
