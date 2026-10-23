"""Проверки удаления аккаунта используют только временную БД и мок Pinecone."""

from __future__ import annotations

import os
import importlib
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import db
import privacy
import web_auth


class UserDataDeletionTests(unittest.TestCase):
    OWNER = 1001
    OTHER = 1002

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.original_db = db.DB_PATH
        db.DB_PATH = str(self.directory / "test.db")
        self.output = self.directory / "output"
        self.output.mkdir()
        self.output_patch = patch.object(privacy, "OUTPUT_DIR", self.output)
        self.output_patch.start()
        self.env_patch = patch.dict(os.environ, {"WEB_SESSION_SECRET": "test-secret-for-account-deletion-only-123456"})
        self.env_patch.start()
        self.usage_day = datetime.now(privacy.reminders.get_tz()).date().isoformat()
        db.init_db()
        self.owner_trip = self._populate(self.OWNER)
        self.other_trip = self._populate(self.OTHER)
        self.owner_pdf = self.output / f"trip_{self.owner_trip}.pdf"
        self.other_pdf = self.output / f"trip_{self.other_trip}.pdf"
        self.owner_pdf.write_bytes(b"owner-pdf")
        self.other_pdf.write_bytes(b"other-pdf")
        self.cache = self.output / "geocode_cache.json"
        self.cache.write_text("global-cache", encoding="utf-8")
        self.session, _, _ = web_auth.create_session(self.OWNER)
        self.login = web_auth.issue_login_token(self.OWNER)

    def tearDown(self):
        self.env_patch.stop()
        self.output_patch.stop()
        db.DB_PATH = self.original_db
        self.temporary.cleanup()

    def _populate(self, owner):
        db.save_user(owner, "Test")
        trip = db.save_trip(owner, {
            "destination": "Тест", "start_date": "2026-11-03", "end_date": "2026-11-03",
        }, {"title": "Тест", "days": []})
        db.save_checklist(trip, [("Одежда", "Куртка")])
        db.save_memory_item(owner, f"family:{owner}", "custom", "Тестовая память", source_trip_id=trip)
        claim = db.claim_ai_request(owner, self.usage_day, "plan", "test", trip_id=trip)
        db.finish_ai_request(claim.claim_id, succeeded=True)
        with db.connect() as conn:
            conn.execute(
                "INSERT INTO reminder_deliveries (trip_id,kind,status,claimed_at) VALUES (?, 'eve', 'sent', ?)",
                (trip, datetime.now(timezone.utc).isoformat()),
            )
        return trip

    def _delete(self):
        return privacy.delete_all_user_data(self.OWNER, actor_tg_id=self.OWNER, confirm=True)

    def _assert_owner_preserved(self):
        self.assertIsNotNone(db.get_user(self.OWNER))
        self.assertIsNotNone(db.get_trip(self.owner_trip, self.OWNER))
        self.assertEqual(db.get_user(self.OWNER)["deletion_token"], "")
        self.assertEqual(len(db.get_checklist(self.owner_trip)), 1)
        self.assertEqual(len(db.list_memory_items(self.OWNER)), 1)

    def test_complete_deletion_keeps_other_family_and_global_files(self):
        old_key = db.get_account_key(self.OWNER)
        with patch.object(privacy.vector_memory, "clear_family_memories", return_value=True) as remote:
            result = self._delete()
            remote.assert_called_once_with(self.OWNER, self.OWNER, confirm=True)
            self.assertTrue(result.deleted)
            self.assertEqual(result.trips_deleted, 1)
            self.assertEqual(result.pdfs_deleted, 1)
            self.assertFalse(self._delete().deleted)
            self.assertEqual(remote.call_count, 2)
        self.assertIsNone(db.get_user(self.OWNER))
        self.assertFalse(self.owner_pdf.exists())
        self.assertEqual(self.other_pdf.read_bytes(), b"other-pdf")
        self.assertEqual(self.cache.read_text(encoding="utf-8"), "global-cache")
        self.assertIsNotNone(db.get_trip(self.other_trip, self.OTHER))
        self.assertEqual(len(db.list_memory_items(self.OTHER)), 1)
        with db.connect() as conn:
            for table in ("ai_requests", "memory_items", "web_login_tokens"):
                self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE tg_id = ?", (self.OWNER,)).fetchone()[0], 0)
            for table in ("checklist_items", "reminder_deliveries"):
                self.assertEqual(conn.execute(f"SELECT COUNT(*) FROM {table} WHERE trip_id = ?", (self.owner_trip,)).fetchone()[0], 0)
            self.assertIsNone(conn.execute("PRAGMA foreign_key_check").fetchone())
        self.assertIsNone(web_auth.verify_session(self.session))
        db.save_user(self.OWNER, "Recreated")
        self.assertNotEqual(db.get_account_key(self.OWNER), old_key)
        self.assertIsNone(web_auth.verify_session(self.session))
        self.assertIsNone(web_auth.consume_login_token(self.login))
        new_session, _, _ = web_auth.create_session(self.OWNER)
        self.assertIsNotNone(web_auth.verify_session(new_session))

    def test_confirmation_and_owner_are_required_before_any_remote_action(self):
        with patch.object(privacy.vector_memory, "clear_family_memories") as remote:
            with self.assertRaises(ValueError):
                privacy.delete_all_user_data(self.OWNER, actor_tg_id=self.OWNER)
            with self.assertRaises(PermissionError):
                privacy.delete_all_user_data(self.OWNER, actor_tg_id=self.OTHER, confirm=True)
            with self.assertRaises(PermissionError):
                privacy.delete_all_user_data(-1001, actor_tg_id=-1001, confirm=True)
            remote.assert_not_called()
        self._assert_owner_preserved()

    def test_remote_failure_keeps_local_data_and_allows_retry(self):
        with patch.object(privacy.vector_memory, "clear_family_memories", return_value=False):
            with self.assertRaises(privacy.AccountDeletionError):
                self._delete()
        self._assert_owner_preserved()
        self.assertEqual(self.owner_pdf.read_bytes(), b"owner-pdf")
        self.assertIsNotNone(web_auth.verify_session(self.session))
        with patch.object(privacy.vector_memory, "clear_family_memories", return_value=True):
            self.assertTrue(self._delete().deleted)

    def test_active_route_blocks_deletion_without_clearing_memory(self):
        active = db.claim_ai_request(self.OWNER, self.usage_day, "edit", "test", trip_id=self.owner_trip)
        with patch.object(privacy.vector_memory, "clear_family_memories") as remote:
            with self.assertRaises(privacy.AccountDeletionBusyError):
                self._delete()
            remote.assert_not_called()
        self._assert_owner_preserved()
        db.finish_ai_request(active.claim_id, succeeded=False)

    def test_lease_blocks_owner_writes_but_does_not_lock_other_users(self):
        def clear_namespace(*args, **kwargs):
            self.assertIsNone(web_auth.verify_session(self.session))
            claim = db.claim_ai_request(self.OWNER, self.usage_day, "plan", "test")
            self.assertEqual(claim.status, "busy")
            with self.assertRaises(RuntimeError):
                db.save_user(self.OWNER, "Blocked")
            with self.assertRaises(RuntimeError):
                db.save_trip(self.OWNER, {}, {})
            with self.assertRaises(RuntimeError):
                db.save_memory_item(self.OWNER, "new", "custom", "Blocked")
            with self.assertRaises(RuntimeError):
                web_auth.issue_login_token(self.OWNER)
            db.save_user(self.OTHER, "Other family writes without waiting")
            self.assertEqual(db.get_user(self.OTHER)["name"], "Other family writes without waiting")
            return True
        with patch.object(privacy.vector_memory, "clear_family_memories", side_effect=clear_namespace):
            self.assertTrue(self._delete().deleted)

    def test_local_transaction_failure_rolls_back_and_releases_lease(self):
        with db.connect() as conn:
            conn.execute(
                "CREATE TRIGGER fail_test_deletion BEFORE DELETE ON users "
                f"WHEN OLD.tg_id = {self.OWNER} BEGIN SELECT RAISE(ABORT, 'test failure'); END"
            )
        with patch.object(privacy.vector_memory, "clear_family_memories", return_value=True):
            with self.assertRaises(privacy.AccountDeletionError):
                self._delete()
        self._assert_owner_preserved()
        with db.connect() as conn:
            conn.execute("DROP TRIGGER fail_test_deletion")
        with patch.object(privacy.vector_memory, "clear_family_memories", return_value=True):
            self.assertTrue(self._delete().deleted)

    def test_pdf_preflight_failure_does_not_clear_memory(self):
        self.owner_pdf.unlink()
        self.owner_pdf.mkdir()
        with patch.object(privacy.vector_memory, "clear_family_memories") as remote:
            with self.assertRaises(privacy.AccountDeletionError):
                self._delete()
            remote.assert_not_called()
        self._assert_owner_preserved()

    def test_abandoned_lease_can_be_retried_and_identity_migration_is_stable(self):
        original_key = db.get_account_key(self.OWNER)
        db.init_db()
        self.assertEqual(db.get_account_key(self.OWNER), original_key)
        with db.connect() as conn:
            conn.execute(
                "UPDATE users SET deletion_token = 'abandoned', deletion_started_at = ? WHERE tg_id = ?",
                ((datetime.now(timezone.utc) - timedelta(minutes=11)).isoformat(), self.OWNER),
            )
        with patch.object(privacy.vector_memory, "clear_family_memories", return_value=True):
            self.assertTrue(self._delete().deleted)

    def test_legacy_accounts_migrate_with_a_stable_nonempty_identity(self):
        old_path = db.DB_PATH
        db.DB_PATH = str(self.directory / "legacy.db")
        try:
            with closing(sqlite3.connect(db.DB_PATH)) as conn, conn:
                conn.execute(
                    "CREATE TABLE users (tg_id INTEGER PRIMARY KEY, name TEXT, adults INTEGER, "
                    "children_ages TEXT, currency TEXT, created_at TEXT)"
                )
                conn.execute("INSERT INTO users VALUES (1, 'Legacy', 2, '[]', 'EUR', '2026-01-01')")
            db.init_db()
            identity = db.get_account_key(1)
            self.assertEqual(len(identity), 32)
            db.init_db()
            self.assertEqual(db.get_account_key(1), identity)
            self.assertEqual(db.get_user(1)["name"], "Legacy")
        finally:
            db.DB_PATH = old_path

    def test_authorized_but_delayed_route_cannot_recreate_a_deleted_account(self):
        identity = db.get_account_key(self.OWNER)
        with patch.object(privacy.vector_memory, "clear_family_memories", return_value=True):
            self.assertTrue(self._delete().deleted)
        claim = db.claim_ai_request(self.OWNER, self.usage_day, "plan", "web", expected_account_key=identity)
        self.assertEqual(claim.status, "busy")
        self.assertIsNone(db.get_user(self.OWNER))
        db.save_user(self.OWNER, "Recreated")
        claim = db.claim_ai_request(self.OWNER, self.usage_day, "plan", "web", expected_account_key=identity)
        self.assertEqual(claim.status, "busy")
        current_claim = db.claim_ai_request(
            self.OWNER, self.usage_day, "plan", "web", expected_account_key=db.get_account_key(self.OWNER)
        )
        self.assertEqual(current_claim.status, "ok")

    def test_legacy_edit_cannot_introduce_or_remove_fixed_costs(self):
        trip = db.get_trip(self.owner_trip, self.OWNER)
        request = dict(trip["request"])
        for cost in ("travel_cost", "lodging_cost"):
            with self.subTest(cost=cost):
                with self.assertRaises(ValueError):
                    db.update_trip_plan(self.owner_trip, self.OWNER, {**request, cost: 500}, trip["itinerary"])
        self.assertTrue(db.update_trip_plan(
            self.owner_trip, self.OWNER, {**request, "travel_cost": 0, "lodging_cost": 0}, trip["itinerary"]
        ))
        # Реально сохранённая ненулевая стоимость также не исчезает при
        # передаче старой анкеты в редактор.
        with db.connect() as conn:
            conn.execute(
                "UPDATE trips SET request_json = json_set(request_json, '$.travel_cost', 500) WHERE id = ?",
                (self.owner_trip,),
            )
        with self.assertRaises(ValueError):
            db.update_trip_plan(self.owner_trip, self.OWNER, request, trip["itinerary"])

    def test_deletion_and_recreation_do_not_reset_daily_paid_quota(self):
        for attempt in range(2):
            claim = db.claim_ai_request(self.OWNER, self.usage_day, "plan", "test")
            self.assertEqual(claim.status, "ok")
            db.finish_ai_request(claim.claim_id, succeeded=attempt % 2 == 0)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.usage_day), 3)
        with patch.object(privacy.vector_memory, "clear_family_memories", return_value=True):
            self._delete()
            self._delete()
        db.save_user(self.OWNER, "New profile")
        claim = db.claim_ai_request(self.OWNER, self.usage_day, "plan", "test")
        self.assertEqual(claim.status, "limit")
        self.assertEqual(claim.used, 3)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.usage_day), 3)
        with db.connect() as conn:
            retention = dict(conn.execute("SELECT * FROM ai_quota_retention").fetchone())
            self.assertEqual(set(retention), {"key_hash", "usage_date", "used"})
            self.assertEqual(len(retention["key_hash"]), 64)
            self.assertNotEqual(retention["key_hash"], str(self.OWNER))
            self.assertEqual(retention["used"], 3)
        tomorrow = (datetime.fromisoformat(self.usage_day) + timedelta(days=1)).date().isoformat()
        claim = db.claim_ai_request(self.OWNER, tomorrow, "plan", "test")
        self.assertEqual(claim.status, "ok")
        self.assertEqual(claim.used, 1)
        with db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM ai_quota_retention").fetchone()[0], 0)

    def test_repeated_deletion_retains_only_new_usage_once(self):
        with patch.object(privacy.vector_memory, "clear_family_memories", return_value=True):
            self._delete()
            db.save_user(self.OWNER, "New profile")
            claim = db.claim_ai_request(self.OWNER, self.usage_day, "plan", "test")
            self.assertEqual(claim.used, 2)
            db.finish_ai_request(claim.claim_id, succeeded=False)
            self._delete()
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.usage_day), 2)

    def test_absent_local_profile_still_clears_possible_orphan_remote_memory(self):
        missing = 5001
        with patch.object(privacy.vector_memory, "clear_family_memories", return_value=False) as remote:
            with self.assertRaises(privacy.AccountDeletionError):
                privacy.delete_all_user_data(missing, actor_tg_id=missing, confirm=True)
            remote.assert_called_once_with(missing, missing, confirm=True)
        self.assertIsNone(db.get_user(missing))
        with patch.object(privacy.vector_memory, "clear_family_memories", return_value=True) as remote:
            result = privacy.delete_all_user_data(missing, actor_tg_id=missing, confirm=True)
            remote.assert_called_once_with(missing, missing, confirm=True)
            self.assertFalse(result.deleted)
        self.assertIsNone(db.get_user(missing))

    def test_bot_confirmation_ownership_and_job_locks(self):
        # Импорт Telegram-клиента заменён локальным FakeBot; polling и сеть
        # запрещены этим fake-клиентом, db.DB_PATH остаётся временным.
        import dotenv
        import telebot
        from try_bot_memory import FakeBot, FakeThread
        fake_bot = FakeBot()
        existing_bot = sys.modules.pop("bot", None)
        try:
            with patch.object(dotenv, "load_dotenv", return_value=False), patch.object(
                telebot, "TeleBot", return_value=fake_bot
            ), patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "123456:LOCAL_TEST_ONLY"}):
                module = importlib.import_module("bot")
            message = SimpleNamespace(
                chat=SimpleNamespace(id=self.OWNER, type="private"),
                from_user=SimpleNamespace(id=self.OWNER), message_id=42, text="/delete_data",
            )
            callback = SimpleNamespace(id="local-callback", message=message,
                from_user=SimpleNamespace(id=self.OWNER), data="account_delete:confirm")
            with patch.object(module.privacy, "delete_all_user_data", return_value=privacy.AccountDeletionResult(True, 1, 1)) as service:
                module.handle_delete_data(message)
                service.assert_not_called()
                callback.from_user = SimpleNamespace(id=self.OTHER)
                module.handle_account_delete(callback)
                service.assert_not_called()
                callback.from_user = SimpleNamespace(id=self.OWNER)
                FakeThread.created.clear()
                with patch.object(module.threading, "Thread", FakeThread):
                    module.handle_account_delete(callback)
                self.assertEqual(len(FakeThread.created), 1)
                self.assertIn(self.OWNER, module.generating)
                self.assertIn(self.OWNER, module.memory_jobs)
                job = FakeThread.created[0]
                job.target(*job.args, **job.kwargs)
                service.assert_called_once_with(self.OWNER, actor_tg_id=self.OWNER, confirm=True)
                self.assertNotIn(self.OWNER, module.generating)
                self.assertNotIn(self.OWNER, module.memory_jobs)
            module.generating.add(self.OWNER)
            module.memory_jobs.add(self.OWNER)
            with patch.object(module.privacy, "delete_all_user_data", side_effect=privacy.AccountDeletionError("test")), patch.object(module.logger, "exception"):
                module._delete_account_job(self.OWNER, self.OWNER)
            self.assertNotIn(self.OWNER, module.generating)
            self.assertNotIn(self.OWNER, module.memory_jobs)
        finally:
            sys.modules.pop("bot", None)
            if existing_bot is not None:
                sys.modules["bot"] = existing_bot


if __name__ == "__main__":
    unittest.main()
