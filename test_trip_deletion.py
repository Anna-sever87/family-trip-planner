"""Single-trip deletion uses disposable SQLite/files and no external services."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

with patch("dotenv.load_dotenv", return_value=False):
    import access_control
    import db
    import generation_jobs
    import trip_covers
    import trip_deletion as deletion
    import web_app
    import web_auth
    import web_service
    import yandex_art
from fastapi.testclient import TestClient
import httpx
from schemas import TripRequest


class TripDeletionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="trip-deletion-")))
        self.stack.enter_context(patch.object(db, "DB_PATH", str(self.directory / "synthetic.db")))
        self.output = self.directory / "output"
        self.covers = self.output / "covers"
        self.covers.mkdir(parents=True)
        self.stack.enter_context(patch.object(deletion, "OUTPUT_DIR", self.output))
        self.stack.enter_context(patch.object(trip_covers.manager, "cache_dir", self.covers))
        self.stack.enter_context(patch.object(trip_covers.manager, "_cache_override", self.covers))
        self.stack.enter_context(patch.dict("os.environ", {
            "ACCESS_MODE": "open", "ACCESS_OWNER_IDS": "9001", "WEB_COOKIE_SECURE": "0",
            "WEB_BASE_URL": "http://testserver", "WEB_SESSION_SECRET": "synthetic-trip-delete-secret-1234567890",
        }))
        db.init_db()
        trip_covers.init_schema()
        for owner in (1, 2, 9001):
            db.save_user(owner, "Synthetic", 2, [], "RUB")
        self.account = db.get_account_key(1)
        self.request = TripRequest(destination="Тест, Россия", start_date="2099-06-01", end_date="2099-06-03",
                                   budget=50000, currency="RUB")
        self.trip = self.save(1)
        self.second = self.save(1)
        self.other = self.save(2)
        self.day = datetime.now(timezone.utc).date().isoformat()
        self.network = self.stack.enter_context(patch.object(httpx.HTTPTransport, "handle_request", side_effect=AssertionError("No network")))
        self.stack.enter_context(patch.object(httpx.AsyncHTTPTransport, "handle_async_request", side_effect=AssertionError("No async network")))
        self.image_provider = self.stack.enter_context(patch.object(yandex_art, "submit", side_effect=AssertionError("No images")))
        self.memory = self.stack.enter_context(patch.object(web_service.vector_memory, "recall_family_memories", side_effect=AssertionError("No memory lookup")))
        self.generate = self.stack.enter_context(patch.object(web_service, "generate_itinerary", side_effect=AssertionError("No route generation")))

    def save(self, owner):
        trip = db.save_trip(owner, self.request.model_dump(), {"title": "Synthetic", "days": []})
        db.save_checklist(trip, [("Synthetic", "Keep correctly scoped")])
        return trip

    def rows(self, table):
        with db.connect() as conn:
            return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]

    def client(self, owner=1):
        client = TestClient(web_app.app)
        self.addCleanup(client.close)
        if owner is not None:
            token, csrf, _ = web_auth.create_session(owner)
            client.cookies.set(web_auth.SESSION_COOKIE, token)
            client.cookies.set(web_auth.CSRF_COOKIE, csrf)
            client.headers["X-CSRF-Token"] = csrf
        return client

    def delete(self, trip=None, account=None):
        return web_service.delete_trip(1, self.trip if trip is None else trip,
                                       expected_account_key=self.account if account is None else account)

    def cover(self, trip=None, *, status="ready", name=None, override=False):
        trip = self.trip if trip is None else trip
        owner = db.get_trip(trip)["tg_id"]
        account = db.get_account_key(owner)
        name = name or uuid4().hex + ".png"
        if "/" not in name and "\\" not in name:
            (self.covers / name).write_bytes(b"synthetic image fixture")
        with db.connect() as conn:
            if override:
                conn.execute("INSERT INTO trip_cover_overrides "
                    "(trip_id,account_key,file_name,mime_type,source_receipt,prompt_version,image_sha256,created_at) "
                    "VALUES (?,?,?,'image/png','synthetic-receipt','test','synthetic-sha',?)",
                    (trip, account, name, db._now()))
            else:
                conn.execute("INSERT INTO trip_covers(trip_id,account_key,cache_key,status,file_name,mime_type,created_at) "
                             "VALUES (?,?,'synthetic-cache',?,?,'image/png',?)", (trip, account, status, name, db._now()))
        return self.covers / name

    def assert_no_paid_work(self):
        self.network.assert_not_called()
        self.image_provider.assert_not_called()
        self.memory.assert_not_called()
        self.generate.assert_not_called()

    def test_delete_only_own_trip_files_checklist_and_reminders_preserves_memory_profile_and_ledgers(self):
        original = self.cover()
        reviewed = self.cover(override=True)
        other_image = self.cover(self.other)
        pdf = self.output / f"trip_{self.trip}.pdf"
        pdf.write_bytes(b"synthetic pdf")
        other_pdf = self.output / f"trip_{self.other}.pdf"
        other_pdf.write_bytes(b"other pdf")
        shared = self.output / "geocode_cache.json"
        shared.write_text("shared map cache", encoding="utf-8")
        db.save_memory_item(1, "synthetic-family-memory", "custom", "Keep family memory", source_trip_id=self.trip)
        claim = db.claim_ai_request(1, self.day, "edit", "test", trip_id=self.trip)
        db.finish_ai_request(claim.claim_id, succeeded=True)
        with db.connect() as conn:
            conn.execute("INSERT INTO cover_attempts VALUES ('synthetic-attempt',?)", (self.day,))
            conn.execute("INSERT INTO reminder_deliveries(trip_id,kind,status,attempts,claimed_at) VALUES (?,'eve','sent',1,?)",
                         (self.trip, db._now()))
        users = self.rows("users")
        quota = {table: self.rows(table) for table in ("cover_attempts", "access_route_attempts", "ai_quota_retention")}
        result = self.delete()
        self.assertEqual(result, {"deleted": True, "trip_id": self.trip, "files_deleted": 3, "cleanup_pending": False,
                                 "message": "Поездка удалена. Общая память семьи и использованные попытки сохранены."})
        self.assertIsNone(db.get_trip(self.trip))
        self.assertIsNotNone(db.get_trip(self.second))
        self.assertIsNotNone(db.get_trip(self.other))
        self.assertEqual(self.rows("users"), users)
        self.assertEqual(db.get_checklist(self.trip), [])
        self.assertEqual(self.rows("reminder_deliveries"), [])
        self.assertEqual(len(self.rows("memory_items")), 1)
        self.assertIsNone(self.rows("memory_items")[0]["source_trip_id"])
        self.assertEqual(self.rows("ai_requests")[0]["status"], "succeeded")
        self.assertIsNone(self.rows("ai_requests")[0]["trip_id"])
        self.assertEqual(len(self.rows("ai_requests")), 1)
        self.assertEqual({table: self.rows(table) for table in quota}, quota)
        self.assertTrue(all(not p.exists() for p in (original, reviewed, pdf)))
        self.assertTrue(all(p.exists() for p in (other_image, other_pdf, shared)))
        self.assertEqual(self.rows("trip_file_cleanup"), [])
        self.assert_no_paid_work()

    def test_unknown_other_owner_and_stale_account_are_indistinguishable_without_changes(self):
        before = self.rows("trips")
        for trip, account in ((self.other, self.account), (999999, self.account), (self.trip, "old-account"),
                              (self.trip, ""), (True, self.account), (-1, self.account)):
            with self.subTest(trip=trip, account=account), self.assertRaises(web_service.TripNotFoundError):
                self.delete(trip, account)
        self.assertEqual(self.rows("trips"), before)
        self.assert_no_paid_work()

    def test_running_create_edit_cover_and_reminder_reject_before_any_deletion(self):
        for kind in ("create", "edit", "queued_cover", "running_cover", "reminder", "job", "account"):
            with self.subTest(kind=kind):
                with db.connect() as conn:
                    if kind in {"create", "edit"}:
                        conn.execute("INSERT INTO ai_requests(claim_id,tg_id,usage_date,kind,source,trip_id,status,claimed_at) "
                                     "VALUES ('busy',1,?,'plan','test',?,'running',?)",
                                     (self.day, self.trip if kind == "edit" else None, db._now()))
                    elif kind.endswith("cover"):
                        conn.execute("INSERT INTO trip_covers(trip_id,account_key,cache_key,status,created_at) VALUES (?,?,'busy',?,?)",
                                     (self.trip, self.account, kind.split("_")[0], db._now()))
                    elif kind == "reminder":
                        conn.execute("INSERT INTO reminder_deliveries(trip_id,kind,status,attempts,claimed_at) VALUES (?,'eve','sending',1,?)",
                                     (self.trip, db._now()))
                    elif kind == "job":
                        conn.execute("INSERT INTO generation_jobs(job_id,tg_id,account_key,idempotency_key,request_digest,status,created_at) "
                                     "VALUES ('busy',1,?,'busy','digest','queued',?)", (self.account, db._now()))
                    else:
                        conn.execute("UPDATE users SET deletion_token='busy' WHERE tg_id=1")
                before = self.rows("trips")
                with self.assertRaises(deletion.TripDeleteBusyError):
                    self.delete()
                self.assertEqual(self.rows("trips"), before)
                self.assertEqual(self.rows("trip_file_cleanup"), [])
                with db.connect() as conn:
                    conn.execute("DELETE FROM ai_requests WHERE claim_id='busy'")
                    conn.execute("DELETE FROM trip_covers")
                    conn.execute("DELETE FROM reminder_deliveries")
                    conn.execute("DELETE FROM generation_jobs WHERE job_id='busy'")
                    conn.execute("UPDATE users SET deletion_token='' WHERE tg_id=1")

    def test_deleted_generation_receipt_never_restarts_or_claims_success(self):
        key, job_id = str(uuid4()), str(uuid4())
        digest = hashlib.sha256(json.dumps(self.request.model_dump(mode="json"), ensure_ascii=False,
                                          sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        with db.connect() as conn:
            conn.execute("INSERT INTO generation_jobs(job_id,tg_id,account_key,idempotency_key,request_digest,status,trip_id,created_at) "
                         "VALUES (?,1,?,?,?,'succeeded',?,?)", (job_id, self.account, key, digest, self.trip, db._now()))
        self.delete()
        status = generation_jobs.get_job(1, self.account, job_id=job_id)
        self.assertEqual((status["status"], status["trip_id"]), ("failed", None))
        self.assertIn("удалена", status["message"])
        executor = Mock()
        manager = generation_jobs.GenerationJobs(executor_factory=lambda **_: executor)
        manager.start()
        self.addCleanup(manager.close)
        self.assertEqual(manager.submit(1, self.account, key, self.request), status)
        executor.submit.assert_not_called()
        self.assert_no_paid_work()

    def test_missing_trip_cannot_be_recreated_by_edit_claim_or_stale_reminder(self):
        self.delete()
        with self.assertRaises(ValueError):
            db.claim_ai_request(1, self.day, "edit", "test", trip_id=self.trip)
        self.assertFalse(db.claim_reminder(self.trip, "eve"))
        self.assertFalse(db.update_trip_plan(self.trip, 1, self.request.model_dump(), {"title": "late", "days": []}))
        self.assertEqual(self.rows("ai_requests"), [])
        self.assertEqual(self.rows("reminder_deliveries"), [])

    def test_sql_write_barrier_prevents_an_edit_claim_racing_deletion(self):
        reached, release = threading.Event(), threading.Event()
        original = deletion._files_for_trip
        def held(*args):
            reached.set()
            self.assertTrue(release.wait(5))
            return original(*args)
        with patch.object(deletion, "_files_for_trip", side_effect=held), ThreadPoolExecutor(max_workers=2) as pool:
            deleting = pool.submit(self.delete)
            self.assertTrue(reached.wait(5))
            claiming = pool.submit(db.claim_ai_request, 1, self.day, "edit", "test", trip_id=self.trip)
            release.set()
            self.assertTrue(deleting.result(timeout=5)["deleted"])
            with self.assertRaises(ValueError):
                claiming.result(timeout=5)
        self.assertEqual(self.rows("ai_requests"), [])

    def test_shared_cover_is_preserved_for_another_trip(self):
        shared = self.cover()
        self.cover(self.other, name=shared.name)
        self.delete()
        self.assertTrue(shared.exists())
        self.assertEqual(self.rows("trip_file_cleanup"), [])
        self.assertEqual(self.rows("trip_covers")[0]["trip_id"], self.other)

    def test_unsafe_cover_name_or_symlink_fails_before_route_deletion(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO trip_covers(trip_id,account_key,cache_key,status,file_name,created_at) "
                         "VALUES (?,?,'unsafe','ready','../outside.png',?)", (self.trip, self.account, db._now()))
        with self.assertRaises(deletion.TripDeleteStorageError):
            self.delete()
        self.assertIsNotNone(db.get_trip(self.trip))
        self.assertEqual(self.rows("trip_file_cleanup"), [])
        with db.connect() as conn:
            conn.execute("DELETE FROM trip_covers WHERE trip_id=?", (self.trip,))
        pdf = self.output / f"trip_{self.trip}.pdf"
        with patch.object(Path, "is_symlink", autospec=True, side_effect=lambda value: value.name == pdf.name):
            with self.assertRaises(deletion.TripDeleteStorageError):
                self.delete()
        self.assertIsNotNone(db.get_trip(self.trip))

    def test_sql_failure_rolls_back_before_files_are_deleted(self):
        image = self.cover()
        with db.connect() as conn:
            conn.execute("CREATE TRIGGER synthetic_delete_failure BEFORE DELETE ON trips "
                         "BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.delete()
        self.assertIsNotNone(db.get_trip(self.trip))
        self.assertTrue(image.exists())
        self.assertEqual(self.rows("trip_file_cleanup"), [])

    def test_disk_failure_returns_honest_success_and_owner_bound_retry_cleans_exact_file(self):
        image = self.cover()
        with patch.object(Path, "unlink", side_effect=OSError("synthetic disk problem")):
            result = self.delete()
        self.assertTrue(result["deleted"])
        self.assertTrue(result["cleanup_pending"])
        self.assertIsNone(db.get_trip(self.trip))
        self.assertTrue(image.exists())
        with self.assertRaises(deletion.TripDeleteNotFoundError):
            deletion.delete_owned_trip(2, db.get_account_key(2), self.trip)
        with self.assertRaises(web_service.TripNotFoundError):
            self.delete(account="recreated-account")
        result = self.delete()
        self.assertFalse(result["cleanup_pending"])
        self.assertEqual(result["files_deleted"], 1)
        self.assertFalse(image.exists())
        with self.assertRaises(web_service.TripNotFoundError):
            self.delete()

    def test_pending_cleanup_survives_init_and_never_follows_changed_root_or_live_reference(self):
        image = self.cover()
        with patch.object(Path, "unlink", side_effect=OSError("synthetic")):
            self.delete()
        db.init_db()
        other_root = self.directory / "other-cache"
        other_root.mkdir()
        elsewhere = other_root / image.name
        elsewhere.write_bytes(b"unrelated")
        with patch.object(trip_covers.manager, "_cache_override", other_root):
            self.assertTrue(deletion.cleanup_pending_files()[1])
        self.assertTrue(elsewhere.exists())
        self.assertTrue(image.exists())
        self.cover(self.other, name=image.name)
        removed, pending = deletion.cleanup_pending_files()
        self.assertEqual((removed, pending), (0, False))
        self.assertTrue(image.exists())

    def test_inactive_cover_manager_uses_configured_cache_without_starting_workers(self):
        original, reviewed = self.cover(), self.cover(override=True)
        inactive = trip_covers.TripCovers()
        with patch.object(trip_covers, "manager", inactive), \
                patch.dict("os.environ", {"IMAGE_CACHE_DIR": str(self.covers)}), \
                patch.object(inactive, "start", side_effect=AssertionError("No workers")) as start:
            result = self.delete()
        self.assertEqual(result["files_deleted"], 2)
        self.assertFalse(result["cleanup_pending"])
        self.assertFalse(original.exists())
        self.assertFalse(reviewed.exists())
        self.assertIsNone(inactive.executor)
        start.assert_not_called()

    def test_http_requires_session_csrf_and_current_owner(self):
        anonymous = self.client(None)
        self.assertEqual(anonymous.delete(f"/api/trips/{self.trip}").status_code, 401)
        own = self.client()
        csrf = own.headers.pop("X-CSRF-Token")
        self.assertEqual(own.delete(f"/api/trips/{self.trip}").status_code, 403)
        own.headers["X-CSRF-Token"] = csrf
        self.assertEqual(own.delete(f"/api/trips/{self.other}").status_code, 404)
        response = own.delete(f"/api/trips/{self.trip}")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["deleted"])
        self.assertEqual(own.delete(f"/api/trips/{self.trip}").status_code, 404)
        self.assert_no_paid_work()

    def test_http_busy_is_409_and_bad_file_preflight_is_503_without_deletion(self):
        client = self.client()
        claim = db.claim_ai_request(1, self.day, "plan", "test")
        self.assertEqual(client.delete(f"/api/trips/{self.trip}").status_code, 409)
        db.finish_ai_request(claim.claim_id, succeeded=False)
        with patch.object(deletion, "_files_for_trip", side_effect=deletion.TripDeleteStorageError("unsafe")):
            response = client.delete(f"/api/trips/{self.trip}")
        self.assertEqual(response.status_code, 503)
        self.assertIn("сохранена", response.json()["detail"])
        self.assertIsNotNone(db.get_trip(self.trip))

    def test_exhausted_invitation_allows_delete_without_refund_but_revoked_access_stays_forbidden(self):
        with patch.dict("os.environ", {"ACCESS_MODE": "invite"}):
            invite = access_control.create_invite(9001)
            access_control.redeem_invite(1, invite["token"])
            for _ in range(5):
                claim = db.claim_ai_request(1, self.day, "edit", "test", trip_id=self.trip)
                db.finish_ai_request(claim.claim_id, succeeded=True)
            self.assertEqual(access_control.status(1)["route_remaining"], 0)
            ledgers = self.rows("access_route_attempts")
            client = self.client()
            self.assertEqual(client.delete(f"/api/trips/{self.trip}").status_code, 200)
            self.assertEqual(self.rows("access_route_attempts"), ledgers)
            self.assertEqual(access_control.status(1)["route_remaining"], 0)
            access_control.revoke_invite(9001, invite["invite_id"])
            self.assertEqual(client.delete(f"/api/trips/{self.second}").status_code, 403)
            self.assertIsNotNone(db.get_trip(self.second))
        self.assert_no_paid_work()


if __name__ == "__main__":
    unittest.main()
