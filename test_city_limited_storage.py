"""Synthetic city-map plans: isolated SQLite lifecycle, no external services."""

from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import db
import pdf_export
import planner
import web_service
import city_context
from city_candidates import CityCandidate
from formatting import format_itinerary
from pricing import minimum_meal_cost
from program_coverage import FREE_DAY_NOTE, limited_program_problems, program_report_text, rebuild_program_report
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def city_partial_fixture():
    """Storage/display fixture only; these invented POIs are not travel advice."""
    request = TripRequest(
        destination="Тестовый город, Россия", start_date="2026-10-28", end_date="2026-10-31",
        adults=2, children_ages=[5], budget=80000, travel_cost=10000, lodging_cost=15000,
        currency="RUB", start_time="08:00", end_time="21:00", pace="средний",
    )
    days = []
    for index in range(4):
        free = index in (1, 3)
        items = [
            ItineraryItem(time="13:00", kind="meal", title="Обед", description="Резерв на питание, не цена кафе.",
                          duration_min=60, cost=minimum_meal_cost(request)),
            ItineraryItem(time="14:00", kind="rest", title="Отдых", description="Пауза по желанию семьи.",
                          duration_min=15, cost=0),
        ]
        if not free:
            for offset, start in enumerate(("09:00", "15:00")):
                identity = 1000 + index * 2 + offset
                title = f"Учебный музей {identity}"
                items.append(ItineraryItem(
                    time=start, title=title, description="Синтетическое посещение для проверки сохранения.",
                    area="Тестовый город", place_query=f"{title}, Тестовый город, Россия",
                    duration_min=120, cost=500, verified=True, lat=43.6 + index * .001,
                    lon=39.72 + offset * .001, official_name=title, provider="osm",
                    source_url=f"https://www.openstreetmap.org/node/{identity}",
                ))
        days.append(ItineraryDay(
            day=index + 1, date=(date.fromisoformat(request.start_date) + timedelta(days=index)).isoformat(),
            theme="Свободный день" if free else "Учебная программа", day_mode="free" if free else "program",
            items=sorted(items, key=lambda item: item.time),
        ))
    itinerary = Itinerary(title="Синтетическая городская программа", summary="Только проверка жизненного цикла.",
                          days=days, city_schedule=True)
    rebuild_program_report(itinerary, request, eligible_unique_visits=5, allow_city_limited_program=True)
    return request, itinerary


def city_fixture_context(request, itinerary):
    """Synthetic full map observations, not inference from a saved URL."""
    candidates = []
    for day in itinerary.days:
        for item in day.items:
            if item.kind != "attraction":
                continue
            identity = int(item.title.rsplit(" ", 1)[1])
            record = dict(lat=item.lat, lon=item.lon, name=item.official_name, provider="osm",
                          osm_type="node", osm_id=identity, type="museum",
                          address={"city": item.area, "country": "Россия", "country_code": "ru"})
            candidates.append(CityCandidate(f"osm-node-{identity}", item.title, item.place_query, item.area, record))
    title = "Учебная галерея 1999"
    candidates.append(CityCandidate("osm-node-1999", title, f"{title}, Тестовый город, Россия", "Тестовый город",
                                    dict(lat=43.605, lon=39.725, name=title, provider="osm", osm_type="node",
                                         osm_id=1999, type="gallery", address={"city": "Тестовый город",
                                         "country": "Россия", "country_code": "ru"})))
    return city_context.build_city_context(request, candidates,
                                           dict(name="Тестовый город", lat=43.6, lon=39.72, country_code="ru"))


class CityLimitedStorageTests(unittest.TestCase):
    OWNER = 92001
    OTHER = 92002

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="city-limited-storage-")
        self.addCleanup(temporary.cleanup)
        self.database = (Path(temporary.name) / "isolated-planner.db").resolve()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(db, "DB_PATH", str(self.database)))
        real_connect = sqlite3.connect

        def isolated_connect(database, *args, **kwargs):
            if kwargs.get("uri") or Path(database).resolve() != self.database:
                raise AssertionError("SQLite outside disposable test database")
            return real_connect(database, *args, **kwargs)

        self.stack.enter_context(patch.object(sqlite3, "connect", side_effect=isolated_connect))
        for owner, name in ((socket.socket, "connect"), (socket, "create_connection"), (socket, "getaddrinfo")):
            self.stack.enter_context(patch.object(owner, name, side_effect=AssertionError("network forbidden")))
        for target in ("places.geocode", "places._load_cache", "places._save_cache",
                       "yandex_ai.generate_json", "httpx.HTTPTransport.handle_request"):
            self.stack.enter_context(patch(target, side_effect=AssertionError("external I/O forbidden")))
        self.recall = self.stack.enter_context(patch.object(
            web_service.vector_memory, "recall_family_memories", return_value=[]))
        self.remember = self.stack.enter_context(patch.object(web_service, "_save_trip_memory_safely", return_value=0))
        self.stack.enter_context(patch.object(web_service, "_daily_ai_limit", return_value=30))
        self.stack.enter_context(patch.object(web_service, "city_candidates_enabled", return_value=False))
        self.stack.enter_context(patch.object(web_service.reminders, "get_tz", return_value=timezone.utc))
        self.stack.enter_context(patch.object(web_service.reminders, "get_days_before", return_value=3))
        db.init_db()
        db.save_user(self.OWNER, name="Учебная семья", currency="RUB")
        db.save_user(self.OTHER, name="Другой владелец", currency="RUB")
        self.usage_date = datetime.now(timezone.utc).date()
        self.request, self.itinerary = city_partial_fixture()
        self.request = web_service._normalized_trip_request(self.request)
        self.result = planner.PlanResult(itinerary=self.itinerary, weather={}, attempts=1)
        self.result.city_map_context = city_fixture_context(self.request, self.itinerary)
        self.generate = self.stack.enter_context(patch.object(web_service, "generate_itinerary", return_value=self.result))
        self.edit = self.stack.enter_context(patch.object(
            web_service, "edit_itinerary", side_effect=AssertionError("edit must be rejected before AI")))

    def create(self):
        return web_service.create_trip(self.OWNER, self.request)

    def assert_detail(self, detail):
        itinerary = Itinerary.model_validate(detail["itinerary"])
        self.assertTrue(itinerary.city_schedule)
        self.assertFalse(itinerary.source_schedule)
        self.assertEqual(itinerary.program_report.coverage_basis, "map")
        self.assertEqual((itinerary.program_report.program_days, itinerary.program_report.free_days), (2, 2))
        self.assertEqual([day.date for day in itinerary.days], [day.date for day in self.itinerary.days])
        self.assertEqual([day.day_mode for day in itinerary.days], ["program", "free", "program", "free"])
        self.assertEqual(limited_program_problems(itinerary, self.request, allow_city_limited_program=True), [])
        self.assertEqual(detail["itinerary"]["program_report_text"], program_report_text(itinerary))
        self.assertIn("но не условия посещения", detail["itinerary"]["program_report_text"])
        self.assertTrue(detail["edit_available"])
        self.assertEqual(detail["edit_unavailable_reason"], "")
        self.assertEqual(detail["edit_scope_note"], city_context.CITY_EDIT_NOTE)
        self.assertNotIn(city_context.CONTEXT_KEY, detail["request"])
        meals = [item for day in itinerary.days for item in day.items if item.kind == "meal"]
        self.assertEqual(len(meals), 4)
        self.assertEqual(sum(item.cost for item in meals), minimum_meal_cost(self.request) * 4)
        program_total = minimum_meal_cost(self.request) * 4 + 2000
        self.assertEqual(detail["budget"]["program_total"], program_total)
        self.assertEqual(detail["budget"]["planned_total"], program_total + 25000)
        self.assertEqual({point["day"] for point in detail["map_points"]}, {1, 3})
        for day in itinerary.days:
            self.assertEqual(day.base_area, "")
            if day.day_mode == "free":
                self.assertEqual([item.kind for item in day.items], ["meal", "rest"])
                self.assertTrue(all(not item.verified and not item.place_query and item.lat is None for item in day.items))

    def test_create_and_reload_preserve_contract_calendar_budget_without_new_ai(self):
        original = deepcopy(self.itinerary.model_dump())
        created = self.create()
        self.assert_detail(created)
        stored = db.get_trip(created["id"], self.OWNER)
        self.assertEqual(stored["itinerary"], original)
        self.assertNotIn("program_report_text", stored["itinerary"])
        self.generate.side_effect = AssertionError("read must not generate")
        for _ in range(2):
            reloaded = web_service.get_trip_detail(self.OWNER, created["id"])
            self.assert_detail(reloaded)
            self.assertEqual(reloaded["revision"], 1)
        self.assertEqual(db.get_trip(created["id"], self.OWNER), stored)
        self.assertEqual(len(db.list_trips(self.OWNER)), 1)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.usage_date), 1)
        self.generate.assert_called_once()
        self.assertEqual(self.generate.call_args.kwargs, {"memory_context": "", "ground_candidates": False})
        self.edit.assert_not_called()

    def test_bad_map_context_rejected_for_program_and_free_days_before_quota_claim(self):
        created = self.create()
        damaged_request = deepcopy(db.get_trip(created["id"], self.OWNER)["request"])
        damaged_request[city_context.CONTEXT_KEY] = {}
        self.assertTrue(db.update_trip_plan(created["id"], self.OWNER, damaged_request,
                                           self.itinerary.model_dump(), expected_revision=1))
        stored = deepcopy(db.get_trip(created["id"], self.OWNER))
        with patch.object(db, "claim_ai_request", side_effect=AssertionError("must not claim quota")):
            for day in (1, 2):
                with self.subTest(day=day), self.assertRaisesRegex(ValueError, "Не удалось подтвердить.*данные карты"):
                    web_service.edit_trip_day(self.OWNER, created["id"], day, "Измени время обеда")
        self.edit.assert_not_called()
        self.assertEqual(db.get_trip(created["id"], self.OWNER), stored)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.usage_date), 1)

    def test_web_edit_persists_same_private_context_and_only_changed_day(self):
        created = self.create()
        before = deepcopy(db.get_trip(created["id"], self.OWNER))
        edited = self.itinerary.model_copy(deep=True)
        edited.days[1].items[0].time = "13:30"
        edited.days[1].items[1].time = "14:30"
        self.edit.side_effect = None
        self.edit.return_value = planner.PlanResult(itinerary=edited, weather={}, attempts=2,
                                                    city_map_context=self.result.city_map_context)
        result = web_service.edit_trip_day(self.OWNER, created["id"], 2, "Перенеси обед на 13:30")
        self.assertEqual(result["revision"], 2)
        self.assertEqual(self.edit.call_args.kwargs["city_map_context"], self.result.city_map_context)
        self.assertEqual(self.edit.call_args.kwargs["prior_weather"], before["request"].get("weather"))
        stored = db.get_trip(created["id"], self.OWNER)
        self.assertEqual(stored["request"][city_context.CONTEXT_KEY], before["request"][city_context.CONTEXT_KEY])
        for number in (0, 2, 3):
            self.assertEqual(stored["itinerary"]["days"][number], before["itinerary"]["days"][number])
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.usage_date), 2)
        self.assertNotIn(city_context.CONTEXT_KEY, result["request"])

    def test_new_city_missing_context_cannot_be_saved(self):
        self.result.city_map_context = None
        with self.assertRaises(web_service.TripCreateValidationError):
            self.create()
        self.assertEqual(db.list_trips(self.OWNER), [])

    def test_planner_rejects_city_edit_before_any_external_or_source_validation(self):
        before = self.itinerary.model_dump()
        with ExitStack() as stack:
            for name in ("_load_blocked_places", "get_weather", "find_city", "generate_json",
                         "_map_verification_problems", "validate_source_edit", "_audit_edit_fulfillment"):
                stack.enter_context(patch.object(planner, name, side_effect=AssertionError(f"unexpected {name}")))
            with self.assertRaisesRegex(ValueError, "Редактирование.*пока недоступно"):
                planner.edit_itinerary(self.request, self.itinerary, 2, "Перенеси обед")
        self.assertEqual(self.itinerary.model_dump(), before)

    def test_foreign_owner_cannot_read_or_edit_or_spend_quota(self):
        created = self.create()
        with patch.object(db, "claim_ai_request", side_effect=AssertionError("must not claim quota")):
            with self.assertRaises(web_service.TripNotFoundError):
                web_service.get_trip_detail(self.OTHER, created["id"])
            with self.assertRaises(web_service.TripNotFoundError):
                web_service.edit_trip_day(self.OTHER, created["id"], 1, "Измени обед")
        self.assertEqual(db.get_ai_request_usage(self.OTHER, self.usage_date), 0)
        self.edit.assert_not_called()

    def test_invalid_city_metadata_cannot_be_saved_as_success(self):
        self.itinerary.source_schedule = True
        with self.assertRaises(web_service.TripCreateValidationError):
            self.create()
        self.assertEqual(db.list_trips(self.OWNER), [])
        with db.connect() as connection:
            statuses = [row[0] for row in connection.execute("SELECT status FROM ai_requests")]
        self.assertEqual(statuses, ["failed"])
        self.remember.assert_not_called()

    def test_loaded_invalid_map_metadata_is_rejected_without_mutation(self):
        raw = self.itinerary.model_dump()
        raw["city_schedule"] = False
        trip_id = db.save_trip(self.OWNER, self.request.model_dump(), raw)
        before = deepcopy(db.get_trip(trip_id, self.OWNER))
        with self.assertRaises(web_service.TripDataError):
            web_service.get_trip_detail(self.OWNER, trip_id)
        with self.assertRaises(web_service.TripDataError):
            web_service.get_trip_pdf(self.OWNER, trip_id)
        self.assertEqual(db.get_trip(trip_id, self.OWNER), before)
        self.generate.assert_not_called()

    def test_api_telegram_pdf_share_honest_map_disclosure_and_free_day_note(self):
        created = self.create()
        disclosure = created["itinerary"]["program_report_text"]
        text = format_itinerary(self.itinerary, "RUB", self.request.budget)
        self.assertIn(disclosure, text)
        self.assertEqual(text.count(FREE_DAY_NOTE), 2)
        with patch.object(pdf_export, "_p", wraps=pdf_export._p) as paragraph:
            document = web_service.get_trip_pdf(self.OWNER, created["id"])
        self.assertTrue(document.startswith(b"%PDF-"))
        rendered_text = [call.args[0] for call in paragraph.call_args_list if call.args]
        self.assertIn(disclosure, rendered_text)
        self.assertEqual(rendered_text.count(FREE_DAY_NOTE), 2)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.usage_date), 1)

    def test_ordinary_trip_remains_editable_and_unchanged_after_city_plan_creation(self):
        ordinary = self.itinerary.model_copy(deep=True)
        ordinary.city_schedule = False
        ordinary.program_report = None
        for day in ordinary.days:
            day.day_mode = "program"
        legacy = ordinary.model_dump()
        legacy.pop("city_schedule")
        old_id = db.save_trip(self.OWNER, self.request.model_dump(), legacy)
        before = deepcopy(db.get_trip(old_id, self.OWNER))
        self.create()
        detail = web_service.get_trip_detail(self.OWNER, old_id)
        self.assertTrue(detail["edit_available"])
        self.assertEqual(detail["edit_unavailable_reason"], "")
        self.assertEqual(detail["itinerary"]["program_report_text"], "")
        self.assertEqual(db.get_trip(old_id, self.OWNER), before)


if __name__ == "__main__":
    unittest.main()
