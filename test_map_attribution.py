"""Map-provider provenance survives app persistence, but never becomes AI input."""

from copy import deepcopy
from datetime import date
import io
import json
import socket
import sqlite3
import unittest
from unittest.mock import patch

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # Optional extraction dependency, supplied by the bundled test runtime.

import places
import planner
import prompts
import pdf_export
import verified_map_places as catalog
import web_service
from formatting import format_itinerary
from pdf_export import _verified_route_points, build_trip_pdf
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


class MapAttributionTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "getaddrinfo"),
                            (sqlite3, "connect"), (places.httpx, "get"), (places, "_save_cache")):
            blocker = patch.object(owner, name, side_effect=AssertionError("External I/O forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)
        rows = json.loads(catalog.CATALOG_FILE.read_text(encoding="utf-8"))["places"]
        self.record = catalog._record(rows[0], date(2026, 10, 21))
        loader = patch.object(catalog, "load_verified_map_places", return_value=[deepcopy(self.record)])
        loader.start()
        self.addCleanup(loader.stop)
        self.item = ItineraryItem(time="10:00", title=self.record["namedetails"]["name"],
                                  description="Самостоятельный осмотр.", place_query=self.record["catalog_queries"][0],
                                  verified=True, lat=self.record["lat"], lon=self.record["lon"],
                                  official_name=self.record["name"], provider="2gis", source_url=self.record["source_url"])
        self.itinerary = Itinerary(title="Маршрут", summary="План", days=[
            ItineraryDay(day=1, date="2026-10-28", theme="Музей", items=[self.item])])
        self.request = TripRequest(destination="Байкал, Россия", destination_kind="region", currency="RUB",
                                   start_date="2026-10-28", end_date="2026-10-28", budget=35000)

    def test_app_fields_survive_serialization_and_system_map_verification(self):
        self.item.provider, self.item.source_url = "fake", "javascript:alert(1)"
        with patch.object(places, "geocode", return_value=deepcopy(self.record)):
            self.assertEqual(places.verify_itinerary(self.itinerary), [])
        self.assertEqual((self.item.provider, self.item.source_url), ("2gis", self.record["source_url"]))
        restored = Itinerary.model_validate_json(self.itinerary.model_dump_json())
        self.assertEqual(catalog.item_map_attribution(restored.days[0].items[0]),
                         {"provider": "2gis", "source_url": self.record["source_url"]})

    def test_failed_or_expired_verification_clears_stale_metadata(self):
        for response in (None, {**self.record, "source_url": "https://example.invalid/fake"}):
            with self.subTest(response=response), patch.object(places, "geocode", return_value=response):
                self.item.provider, self.item.source_url, self.item.verified = "2gis", self.record["source_url"], True
                self.assertTrue(places.verify_itinerary(self.itinerary))
                self.assertEqual((self.item.provider, self.item.source_url, self.item.verified), ("", "", False))

    def test_osm_system_record_replaces_old_provider_without_trusting_old_url(self):
        ordinary = {"lat": 51., "lon": 104., "name": "Музей, Город, Россия"}
        with patch.object(places, "geocode", return_value=ordinary):
            self.assertEqual(places.verify_itinerary(self.itinerary), [])
        self.assertEqual(self.item.provider, "osm")
        self.assertTrue(self.item.source_url.startswith("https://www.openstreetmap.org/"))
        self.assertNotIn("2gis", self.item.source_url)

    def test_independent_attribution_requires_exact_current_identity_coordinates_and_url(self):
        expected = {"provider": "", "source_url": ""}
        for change in ({"lat": True}, {"lat": float("nan")}, {"lon": 181}, {"verified": False},
                       {"source_url": "https://2gis.ru/irkutsk/firm/123"}, {"place_query": "Неполное название"},
                       {"lat": self.item.lat + .001}, {"provider": "unknown"}):
            with self.subTest(change=change):
                self.assertEqual(catalog.item_map_attribution({**self.item.model_dump(), **change}), expected)
        with patch.object(catalog, "load_verified_map_places", return_value=[]):
            self.assertEqual(catalog.item_map_attribution(self.item), expected)
        self.assertEqual(catalog.item_map_attribution({**self.item.model_dump(), "provider": ""}), expected)

    def test_model_schema_prompt_and_reset_never_trust_provenance_fields(self):
        schema = planner.build_itinerary_json_schema(self.request)
        props = schema["properties"]["days"]["items"]["properties"]["items"]["items"]["properties"]
        self.assertNotIn("provider", props)
        self.assertNotIn("source_url", props)
        for payload in (prompts._itinerary_for_edit_prompt(self.itinerary)["days"][0],
                        prompts._day_for_edit_audit(self.itinerary, 1),
                        planner._day_content_for_edit_compare(self.itinerary.days[0])):
            self.assertNotIn("provider", payload["items"][0])
            self.assertNotIn("source_url", payload["items"][0])
        planner._clear_map_verification_state(self.itinerary)
        self.assertEqual((self.item.verified, self.item.provider, self.item.source_url), (False, "", ""))

    def test_web_map_and_timeline_only_expose_current_safe_source_without_mutating_saved_data(self):
        before = self.itinerary.model_dump()
        trip = {"id": 9, "request": self.request.model_dump(), "revision": 1}
        for fresh in (True, False):
            with self.subTest(fresh=fresh), patch.object(web_service, "_validated_trip", return_value=(trip, self.request, self.itinerary)), \
                    patch.object(web_service.db, "get_checklist", return_value=[]), \
                    patch.object(web_service.db, "get_reminder_statuses", return_value=[]), \
                    patch.object(web_service.reminders, "describe_plan", return_value=""), \
                    patch.object(catalog, "load_verified_map_places", return_value=[self.record] if fresh else []):
                detail = web_service.get_trip_detail(1, 9)
            self.assertEqual(len(detail["map_points"]), 1 if fresh else 0)
            exposed_items = [detail["itinerary"]["days"][0]["items"][0], *detail["map_points"]]
            for exposed in exposed_items:
                self.assertEqual(exposed["verified"], fresh)
                self.assertEqual(exposed["provider"], "2gis" if fresh else "")
                self.assertEqual(exposed["source_url"], self.record["source_url"] if fresh else "")
            self.assertEqual(self.itinerary.model_dump(), before)

    def test_web_map_excludes_non_attractions_invalid_coordinates_and_untrusted_provenance(self):
        trip = {"id": 9, "request": self.request.model_dump(), "revision": 1}
        changes = (
            {"kind": "meal"}, {"kind": "rest"}, {"verified": False},
            {"lat": float("nan")}, {"lon": float("inf")}, {"lat": 91.0}, {"lon": -181.0},
            {"lat": True}, {"provider": "unknown"},
            {"provider": "2gis", "source_url": "https://2gis.ru/irkutsk/firm/123"},
        )
        for change in changes:
            with self.subTest(change=change):
                invalid = self.item.model_copy(update=change)
                itinerary = self.itinerary.model_copy(deep=True)
                itinerary.days[0].items = [invalid]
                with patch.object(web_service, "_validated_trip", return_value=(trip, self.request, itinerary)), \
                        patch.object(web_service.db, "get_checklist", return_value=[]), \
                        patch.object(web_service.db, "get_reminder_statuses", return_value=[]), \
                        patch.object(web_service.reminders, "describe_plan", return_value=""):
                    detail = web_service.get_trip_detail(1, 9)
                self.assertEqual(detail["map_points"], [])
                self.assertFalse(detail["itinerary"]["days"][0]["items"][0]["verified"])

    def test_web_map_keeps_legacy_osm_coordinate_provenance(self):
        legacy = self.item.model_copy(update={"provider": "", "source_url": ""})
        itinerary = self.itinerary.model_copy(deep=True)
        itinerary.days[0].items = [legacy]
        trip = {"id": 9, "request": self.request.model_dump(), "revision": 1}
        with patch.object(web_service, "_validated_trip", return_value=(trip, self.request, itinerary)), \
                patch.object(web_service.db, "get_checklist", return_value=[]), \
                patch.object(web_service.db, "get_reminder_statuses", return_value=[]), \
                patch.object(web_service.reminders, "describe_plan", return_value=""):
            detail = web_service.get_trip_detail(1, 9)
        self.assertEqual(len(detail["map_points"]), 1)
        self.assertEqual(detail["map_points"][0]["provider"], "osm")
        self.assertTrue(detail["map_points"][0]["source_url"].startswith("https://www.openstreetmap.org/"))

    def _public_map_decisions(self, item, *, fresh=True):
        """Exercise all three presentation paths, including a real PDF build."""
        itinerary = self.itinerary.model_copy(deep=True)
        itinerary.days[0].items = [item]
        trip = {"id": 9, "request": self.request.model_dump(), "revision": 1}
        paragraphs = []
        paragraph_class = pdf_export.Paragraph

        def capture_paragraph(*args, **kwargs):
            paragraph = paragraph_class(*args, **kwargs)
            paragraphs.append(paragraph.getPlainText())
            return paragraph

        with patch.object(web_service, "_validated_trip", return_value=(trip, self.request, itinerary)), \
                patch.object(web_service.db, "get_checklist", return_value=[]), \
                patch.object(web_service.db, "get_reminder_statuses", return_value=[]), \
                patch.object(web_service.reminders, "describe_plan", return_value=""), \
                patch.object(catalog, "load_verified_map_places", return_value=[self.record] if fresh else []), \
                patch.object(pdf_export, "Paragraph", side_effect=capture_paragraph):
            detail = web_service.get_trip_detail(1, 9)
            points = _verified_route_points(itinerary.days[0])
            telegram = format_itinerary(itinerary, "RUB", self.request.budget)
            # A model instance also preserves malformed legacy/model_copy values.
            pdf = build_trip_pdf({**trip, "itinerary": itinerary}, [])
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertEqual(detail["itinerary"]["days"][0]["items"][0]["verified"], bool(detail["map_points"]))
        return {
            "web": bool(detail["map_points"]),
            "pdf_map": bool(points),
            "telegram_without_warning": "⚠️" not in telegram,
            "pdf_without_warning": "место не подтверждено на карте" not in "\n".join(paragraphs),
        }

    def test_expired_independent_source_is_unconfirmed_in_every_public_view(self):
        decisions = self._public_map_decisions(self.item, fresh=False)
        self.assertEqual(decisions, dict.fromkeys(decisions, False))

    def test_incomplete_or_model_provenance_is_unconfirmed_in_every_public_view(self):
        for change in ({"source_url": ""}, {"provider": ""}, {"place_query": ""},
                       {"provider": "unknown"},
                       {"verified": False, "source_url": "https://example.invalid/model-place"},
                       {"provider": "", "source_url": "https://example.invalid/model-place"}):
            with self.subTest(change=change):
                decisions = self._public_map_decisions(self.item.model_copy(update=change))
                self.assertEqual(decisions, dict.fromkeys(decisions, False))

    def test_invalid_saved_coordinates_are_unconfirmed_in_every_public_view(self):
        for change in ({"lat": float("nan")}, {"lon": float("inf")},
                       {"lat": 91.0}, {"lon": -181.0}, {"lat": None}):
            with self.subTest(change=change):
                decisions = self._public_map_decisions(self.item.model_copy(update=change))
                self.assertEqual(decisions, dict.fromkeys(decisions, False))

    def test_current_independent_and_legacy_osm_stay_confirmed_in_every_public_view(self):
        for item in (self.item, self.item.model_copy(update={"provider": "", "source_url": ""})):
            with self.subTest(provider=item.provider):
                decisions = self._public_map_decisions(item)
                self.assertEqual(decisions, dict.fromkeys(decisions, True))

    def test_public_predicate_rejects_malformed_models_and_dicts_without_coercion(self):
        changes = ({"lat": True}, {"lon": "104.0"}, {"lat": []}, {"lat": 10 ** 1000},
                   {"verified": 1}, {"kind": "meal"}, {"kind": "rest"})
        for change in changes:
            with self.subTest(change=change):
                invalid = self.item.model_copy(update=change)
                self.assertFalse(catalog.is_public_map_point(invalid))
                self.assertFalse(catalog.is_public_map_point({**self.item.model_dump(), **change}))
                day = self.itinerary.days[0].model_copy(update={"items": [invalid]})
                self.assertEqual(_verified_route_points(day), [])
        self.assertTrue(catalog.is_public_map_point(self.item))
        self.assertTrue(catalog.is_public_map_point(self.item.model_dump()))

    @unittest.skipIf(PdfReader is None, "Run with bundled Python dependencies for PDF link extraction")
    def test_pdf_organisation_link_uses_2gis_and_expired_record_never_falls_back_to_osm(self):
        trip = {"request": self.request.model_dump(), "itinerary": self.itinerary.model_dump()}
        for fresh in (True, False):
            with self.subTest(fresh=fresh), patch.object(catalog, "load_verified_map_places", return_value=[self.record] if fresh else []):
                points = _verified_route_points(self.itinerary.days[0])
                reader = PdfReader(io.BytesIO(build_trip_pdf(trip, [])))
            links = [annotation.get_object().get("/A", {}).get("/URI", "")
                     for page in reader.pages for annotation in page.get("/Annots", [])]
            self.assertEqual(len(points), 1 if fresh else 0)
            if fresh:
                self.assertEqual(points[0]["source_url"], self.record["source_url"])
            self.assertEqual(self.record["source_url"] in links, fresh)
            self.assertFalse(any(str(link).startswith("https://www.openstreetmap.org/") for link in links))
            text = "\n".join(page.extract_text() for page in reader.pages)
            self.assertIn("Источник координат: 2ГИС" if fresh else "нет подтверждённых координат", text)
            self.assertEqual("место не подтверждено на карте" in text, not fresh)


if __name__ == "__main__":
    unittest.main()
