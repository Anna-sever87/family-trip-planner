"""Reminder claims must precede owner-bound reads of the current itinerary.

Compile the actual dispatcher without importing the bot's .env/startup. All
storage is disposable; Telegram, weather and scheduling are local test doubles.
"""

import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket
import tempfile
from typing import Optional
import unittest
from unittest.mock import Mock, patch

import db


def dispatcher(namespace):
    source = Path(__file__).with_name("bot.py").read_text(encoding="utf-8")
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "dispatch_reminders")
    namespace.update(datetime=datetime, timedelta=timedelta, Optional=Optional)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "bot.py", "exec"), namespace)
    return namespace["dispatch_reminders"]


class BotReminderSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="reminder-snapshot-")
        self.addCleanup(temporary.cleanup)
        self.addCleanup(patch.stopall)
        patch.object(db, "DB_PATH", str(Path(temporary.name) / "test.db")).start()
        patch.object(socket.socket, "connect", side_effect=AssertionError("No network")).start()
        db.init_db()
        db.save_user(101, "Synthetic owner")
        self.trip_id = db.save_trip(101, {"destination": "Synthetic", "start_date": "2099-06-01",
                                        "end_date": "2099-06-02"},
                                    {"title": "Before swap", "days": []})
        self.snapshot = db.get_trip(self.trip_id, 101)
        self.now = datetime(2099, 6, 1, 8, tzinfo=timezone.utc)
        self.reminders = Mock()
        self.reminders.get_tz.return_value = timezone.utc
        self.reminders.normalize_datetime.side_effect = lambda value, _tz: value
        self.reminders.get_days_before.return_value = 3
        self.reminders.due_reminders.return_value = [(deepcopy(self.snapshot), "day1")]
        self.reminders.build_reminder_text.side_effect = lambda _kind, trip, _checklist: trip["itinerary"]["title"]
        self.bot = Mock()
        self.refresh = Mock(side_effect=lambda trip: (trip, True))
        self.dispatch = dispatcher({"db": db, "reminders": self.reminders, "bot": self.bot,
                                    "logger": Mock(), "_trip_with_fresh_eve_weather": self.refresh})
        self.ai = patch.object(db, "claim_ai_request", side_effect=AssertionError("No quota claim")).start()

    def status(self):
        with db.connect() as conn:
            row = conn.execute("SELECT status,attempts FROM reminder_deliveries WHERE trip_id=?", (self.trip_id,)).fetchone()
            return tuple(row) if row else None

    def replace_plan(self):
        # Reproduce another transaction committing after the due-list read but
        # before the reminder claim. No actual user row is involved.
        with db.connect() as conn:
            conn.execute("UPDATE trips SET itinerary_json=?,revision=revision+1 WHERE id=? AND tg_id=?",
                         (json.dumps({"title": "After swap", "days": []}), self.trip_id, 101))

    def test_committed_swap_before_claim_uses_new_plan_and_preserves_sent_ledger(self):
        claim = db.claim_reminder
        get_trip = db.get_trip

        def claim_after_swap(*args, **kwargs):
            self.replace_plan()
            return claim(*args, **kwargs)

        def read_after_claim(trip_id, owner):
            self.assertEqual(self.status(), ("sending", 1))
            return get_trip(trip_id, owner)

        with patch.object(db, "claim_reminder", side_effect=claim_after_swap), \
                patch.object(db, "get_trip", side_effect=read_after_claim) as current:
            self.dispatch(self.now)
        current.assert_called_once_with(self.trip_id, 101)
        self.bot.send_message.assert_called_once_with(101, "After swap")
        self.assertEqual(self.status(), ("sent", 1))
        self.assertEqual(self.snapshot["itinerary"]["title"], "Before swap")
        self.dispatch(self.now)
        self.bot.send_message.assert_called_once()
        self.assertEqual(self.status(), ("sent", 1))
        self.ai.assert_not_called()

    def test_eve_weather_refresh_receives_current_owner_bound_snapshot(self):
        self.reminders.due_reminders.return_value = [(self.snapshot, "eve")]
        self.replace_plan()
        self.dispatch(self.now)
        self.assertEqual(self.refresh.call_args.args[0]["itinerary"]["title"], "After swap")
        self.bot.send_message.assert_called_once_with(101, "After swap")

    def test_claimed_lease_blocks_real_swap_until_current_message_is_sent(self):
        from test_day_swap import fixture
        import web_service

        data = fixture(2)
        self.trip_id = db.save_trip(101, data["request"], data["itinerary"])
        self.snapshot = db.get_trip(self.trip_id, 101)
        self.reminders.due_reminders.return_value = [(self.snapshot, "day1")]
        account = db.get_account_key(101)
        get_trip = db.get_trip

        def read_with_concurrent_swap(trip_id, owner):
            self.assertEqual(self.status(), ("sending", 1))
            with self.assertRaises(web_service.EditBusyError):
                web_service.swap_trip_days(101, self.trip_id, 1, 2, 1, expected_account_key=account)
            return get_trip(trip_id, owner)

        with patch.dict("os.environ", {"ACCESS_MODE": "open"}), \
                patch.object(db, "get_trip", side_effect=read_with_concurrent_swap):
            self.dispatch(self.now)
        self.bot.send_message.assert_called_once_with(101, data["itinerary"]["title"])
        self.assertEqual(self.status(), ("sent", 1))
        self.assertEqual(db.get_trip(self.trip_id, 101)["revision"], 1)

    def test_failed_claim_never_reads_or_sends_stale_snapshot(self):
        with patch.object(db, "claim_reminder", return_value=False), patch.object(db, "get_trip") as current:
            self.dispatch(self.now)
        current.assert_not_called()
        self.bot.send_message.assert_not_called()
        self.assertIsNone(self.status())

    def test_missing_current_owner_trip_fails_closed_without_stale_fallback(self):
        with patch.object(db, "get_trip", return_value=None) as current:
            self.dispatch(self.now)
        current.assert_called_once_with(self.trip_id, 101)
        self.bot.send_message.assert_not_called()
        self.reminders.build_reminder_text.assert_not_called()
        self.refresh.assert_not_called()
        self.assertEqual(self.status(), ("failed", 1))

    def test_current_read_error_does_not_send_old_plan_or_reset_attempts(self):
        with patch.object(db, "get_trip", side_effect=RuntimeError("Synthetic unavailable storage")):
            self.dispatch(self.now)
        self.bot.send_message.assert_not_called()
        self.reminders.build_reminder_text.assert_not_called()
        self.assertEqual(self.status(), ("failed", 1))

    def test_owner_mismatch_never_reads_another_owners_plan(self):
        stale = deepcopy(self.snapshot)
        stale["tg_id"] = 202
        self.reminders.due_reminders.return_value = [(stale, "day1")]
        with patch.object(db, "get_trip", wraps=db.get_trip) as current:
            self.dispatch(self.now)
        current.assert_called_once_with(self.trip_id, 202)
        self.bot.send_message.assert_not_called()
        self.assertEqual(self.status(), ("failed", 1))


if __name__ == "__main__":
    unittest.main()
