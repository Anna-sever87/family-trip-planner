"""Real temporary SQLite create/read/edit lifecycle; no real AI, maps or memory."""

from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import db
from planner import PlanResult
from pricing import minimum_meal_cost
from program_coverage import limited_program_problems, program_report_text
from regional_skeleton import build_regional_skeleton
from schemas import Itinerary, ItineraryDay, ItineraryItem
from source_planning import itinerary_from_skeleton
from test_regional_skeleton import fixture, request_for, windows_for
import web_service


class LimitedProgramStorageTests(unittest.TestCase):
    OWNER = 91001
    OTHER = 91002

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="limited-program-storage-")
        self.addCleanup(temporary.cleanup)
        self.database = (Path(temporary.name) / "isolated-planner.db").resolve()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(db, "DB_PATH", str(self.database)))
        original_connect = sqlite3.connect
        self.connections = []

        def temporary_database_only(database, *args, **kwargs):
            # All real DB and quota operations must target this one disposable
            # file, even if an unexpected code path bypasses db.DB_PATH.
            if kwargs.get("uri") or Path(database).resolve() != self.database:
                raise AssertionError("SQLite access outside the isolated test database")
            self.connections.append(str(database))
            return original_connect(database, *args, **kwargs)

        self.stack.enter_context(patch.object(sqlite3, "connect", side_effect=temporary_database_only))
        for owner, name in ((socket.socket, "connect"), (socket, "create_connection"),
                            (socket, "getaddrinfo")):
            self.stack.enter_context(patch.object(owner, name, side_effect=AssertionError("network forbidden")))
        for target in ("places.geocode", "places._load_cache", "places._save_cache",
                       "yandex_ai.generate_json", "httpx.HTTPTransport.handle_request"):
            self.stack.enter_context(patch(target, side_effect=AssertionError("external I/O forbidden")))
        self.recall = self.stack.enter_context(patch.object(
            web_service.vector_memory, "recall_family_memories", return_value=[]))
        self.remember = self.stack.enter_context(patch.object(
            web_service, "_save_trip_memory_safely", return_value=0))
        self.stack.enter_context(patch.object(web_service, "_daily_ai_limit", return_value=5))
        self.stack.enter_context(patch.object(web_service.reminders, "get_tz", return_value=timezone.utc))
        self.stack.enter_context(patch.object(web_service.reminders, "get_days_before", return_value=3))
        db.init_db()
        db.save_user(self.OWNER, name="Тестовая семья", currency="RUB")
        db.save_user(self.OTHER, name="Другая тестовая семья", currency="RUB")
        self.usage_date = datetime.now(timezone.utc).date()
        self.request = request_for(7, budget=60000, travel_cost=10000, lodging_cost=15000)
        grounding, facts, graph = fixture(self.request, 4, 120)
        self.skeleton = build_regional_skeleton(self.request, grounding, facts, travel_facts=graph,
                                                allow_free_days=True)
        self.assertEqual(self.skeleton.status, "limited_program")
        self.assertTrue(self.skeleton.search_complete)
        self.itinerary = itinerary_from_skeleton(self.request, grounding, self.skeleton)
        self.assertEqual(limited_program_problems(
            self.itinerary, self.request, allow_limited_program=True), [])
        self.result = PlanResult(itinerary=self.itinerary, weather={}, attempts=1)
        self.generate = self.stack.enter_context(patch.object(
            web_service, "generate_itinerary", return_value=self.result))
        self.edit = self.stack.enter_context(patch.object(
            web_service, "edit_itinerary", side_effect=AssertionError("unexpected edit generation")))

    def create(self):
        return web_service.create_trip(self.OWNER, self.request)

    def assert_limited_detail(self, detail):
        public = detail["itinerary"]
        restored = Itinerary.model_validate({key: value for key, value in public.items()
                                            if key != "program_report_text"})
        report = restored.program_report
        self.assertTrue(restored.source_schedule)
        self.assertEqual((report.requested_days, report.program_days, report.free_days), (7, 2, 5))
        self.assertEqual((report.eligible_unique_visits, report.selected_unique_visits), (4, 4))
        self.assertEqual([day.date for day in restored.days], [w.date for w in windows_for(self.request)])
        self.assertEqual([day.day_mode for day in restored.days], ["program"] * 2 + ["free"] * 5)
        self.assertEqual(public["program_report_text"], program_report_text(restored))
        self.assertTrue(public["program_report_text"])
        self.assertNotIn("требуют перепроверки", public["program_report_text"])
        self.assertEqual(limited_program_problems(restored, self.request, allow_limited_program=True), [])
        meals = [item for day in restored.days for item in day.items if item.kind == "meal"]
        self.assertEqual(len(meals), 7)
        self.assertEqual(sum(item.cost for item in meals), 7 * minimum_meal_cost(self.request))
        for day in restored.days[2:]:
            self.assertEqual([item.kind for item in day.items], ["meal", "rest"])
            self.assertEqual(day.transfer_note, "")
            self.assertEqual(day.base_area, restored.days[1].base_area)
            self.assertTrue(all(not item.verified and item.lat is None and item.lon is None
                                and not item.place_query for item in day.items))
        self.assertEqual(detail["budget"]["program_total"], 10600)
        self.assertEqual(detail["budget"]["fixed_costs"], 25000)
        self.assertEqual(detail["budget"]["planned_total"], 35600)
        self.assertEqual(detail["budget"]["remaining"], 24400)
        self.assertFalse(detail["budget"]["over_budget"])
        self.assertEqual([row["day_total"] for row in detail["budget"]["days"][2:]], [1400] * 5)
        return restored

    def test_create_persists_complete_free_calendar_and_api_disclosure(self):
        before = deepcopy(self.itinerary.model_dump())
        detail = self.create()
        self.assert_limited_detail(detail)
        persisted = db.get_trip(detail["id"], self.OWNER)
        self.assertEqual(persisted["itinerary"], before)
        self.assertNotIn("program_report_text", persisted["itinerary"])
        self.assertEqual(self.itinerary.model_dump(), before)
        self.assertEqual(detail["revision"], 1)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.usage_date), 1)
        with db.connect() as connection:
            rows = connection.execute("SELECT kind, source, status FROM ai_requests").fetchall()
        self.assertEqual([tuple(row) for row in rows], [("plan", "web", "succeeded")])
        self.generate.assert_called_once()
        self.assertTrue(self.generate.call_args.kwargs["ground_candidates"])
        self.recall.assert_called_once()
        self.remember.assert_called_once()
        self.assertTrue(self.connections)
        self.assertEqual(set(self.connections), {str(self.database)})

    def test_fresh_sqlite_reload_does_not_regenerate_consume_quota_or_duplicate_trip(self):
        created = self.create()
        trip_id = created["id"]
        original_summary = created["itinerary"]["summary"]
        # Prove the following reads come from serialized SQLite, not the fake
        # generator's retained object. Every db operation opens a new connection.
        self.itinerary.summary = "Изменено только в памяти генератора, не в БД"
        self.generate.side_effect = AssertionError("reload must not invoke AI")
        db.init_db()  # Reopen the existing temporary DB and apply normal migrations.
        for _ in range(3):
            reloaded = web_service.get_trip_detail(self.OWNER, trip_id)
            self.assert_limited_detail(reloaded)
            self.assertEqual(reloaded["itinerary"]["summary"], original_summary)
            self.assertEqual(reloaded["revision"], 1)
            self.assertNotIn("generation", reloaded)
        with db.connect() as connection:
            raw = connection.execute("SELECT itinerary_json FROM trips WHERE id=?", (trip_id,)).fetchone()[0]
            trip_count = connection.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
            request_count = connection.execute("SELECT COUNT(*) FROM ai_requests").fetchone()[0]
        restored = Itinerary.model_validate_json(raw)
        self.assertEqual(restored.program_report.program_days, 2)
        self.assertEqual(restored.program_report.free_days, 5)
        self.assertEqual(json.loads(raw)["source_schedule"], True)
        self.assertEqual((trip_count, request_count), (1, 1))
        self.assertEqual(len(web_service.list_user_trips(self.OWNER)), 1)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.usage_date), 1)
        self.generate.assert_called_once()
        self.edit.assert_not_called()
        self.recall.assert_called_once()
        self.remember.assert_called_once()

    def test_existing_ordinary_legacy_trip_is_unchanged(self):
        ordinary_request = request_for(1, destination_kind="city", destination="Тестовый город, Россия")
        ordinary = Itinerary(title="Обычная старая поездка", summary="Сохранённый обычный маршрут", days=[
            ItineraryDay(day=1, date=ordinary_request.start_date, theme="Знакомство с городом", items=[
                ItineraryItem(time="10:00", title="Старый музей", description="Посещение", duration_min=60, cost=100)
            ])
        ]).model_dump()
        ordinary.pop("program_report")
        ordinary.pop("source_schedule")
        ordinary["days"][0].pop("day_mode")
        ordinary_id = db.save_trip(self.OWNER, ordinary_request.model_dump(), ordinary)
        before = deepcopy(db.get_trip(ordinary_id, self.OWNER))
        limited = self.create()
        self.assertNotEqual(ordinary_id, limited["id"])
        detail = web_service.get_trip_detail(self.OWNER, ordinary_id)
        self.assertEqual(detail["itinerary"]["title"], ordinary["title"])
        self.assertIsNone(detail["itinerary"]["program_report"])
        self.assertEqual(detail["itinerary"]["program_report_text"], "")
        self.assertFalse(detail["itinerary"]["source_schedule"])
        self.assertEqual(detail["itinerary"]["days"][0]["day_mode"], "program")
        self.assertEqual(db.get_trip(ordinary_id, self.OWNER), before)
        self.assertEqual(len(web_service.list_user_trips(self.OWNER)), 2)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.usage_date), 1)

    def test_other_owner_cannot_read_or_edit_limited_trip_or_consume_its_quota(self):
        detail = self.create()
        trip_id = detail["id"]
        before = deepcopy(db.get_trip(trip_id, self.OWNER))
        self.assertIsNone(db.get_trip(trip_id, self.OTHER))
        self.assertEqual(web_service.list_user_trips(self.OTHER), [])
        with self.assertRaises(web_service.TripNotFoundError):
            web_service.get_trip_detail(self.OTHER, trip_id)
        with self.assertRaises(web_service.TripNotFoundError):
            web_service.edit_trip_day(self.OTHER, trip_id, 1, "Уточни описание дня")
        self.edit.assert_not_called()
        self.assertEqual(db.get_trip(trip_id, self.OWNER), before)
        self.assertEqual(db.get_ai_request_usage(self.OTHER, self.usage_date), 0)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.usage_date), 1)

    def test_trusted_edit_result_preserves_free_metadata_in_new_revision(self):
        created = self.create()
        trip_id = created["id"]
        before = deepcopy(db.get_trip(trip_id, self.OWNER)["itinerary"])

        def edited_result(request, current, target_day, instruction):
            self.assertTrue(current.source_schedule)
            self.assertEqual(current.program_report.free_days, 5)
            self.assertEqual(target_day, 1)
            changed = current.model_copy(deep=True)
            changed.days[0].theme = "Уточнённое описание первого программного дня"
            self.assertEqual(limited_program_problems(changed, request, allow_limited_program=True), [])
            return PlanResult(itinerary=changed, weather={}, attempts=1)

        self.edit.side_effect = edited_result
        updated = web_service.edit_trip_day(self.OWNER, trip_id, 1, "Уточни описание первого дня")
        self.assert_limited_detail(updated)
        self.assertEqual(updated["revision"], 2)
        self.assertEqual(updated["edit"]["day"], 1)
        after = db.get_trip(trip_id, self.OWNER)["itinerary"]
        self.assertEqual(after["program_report"], before["program_report"])
        self.assertEqual(after["days"][1:], before["days"][1:])
        self.assertNotEqual(after["days"][0]["theme"], before["days"][0]["theme"])
        reloaded = web_service.get_trip_detail(self.OWNER, trip_id)
        self.assert_limited_detail(reloaded)
        self.assertEqual(reloaded["revision"], 2)
        self.assertEqual(len(web_service.list_user_trips(self.OWNER)), 1)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.usage_date), 2)
        with db.connect() as connection:
            statuses = connection.execute("SELECT status FROM ai_requests ORDER BY rowid").fetchall()
        self.assertEqual([row[0] for row in statuses], ["succeeded", "succeeded"])
        self.generate.assert_called_once()
        self.edit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
