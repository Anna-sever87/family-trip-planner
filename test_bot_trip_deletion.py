"""Actual Telegram deletion handler; synthetic messages and disposable storage."""

import ast
from pathlib import Path
import socket
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import Mock, patch

import access_control
import db
import trip_deletion
import web_service


def handler(namespace):
    tree = ast.parse(Path(__file__).with_name("bot.py").read_text(encoding="utf-8"))
    names = {"_private_actor", "_trip_number", "handle_delete"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in functions:
        node.decorator_list = []
    namespace["Optional"] = Optional
    exec(compile(ast.Module(body=functions, type_ignores=[]), "bot.py", "exec"), namespace)
    return namespace["handle_delete"]


def message(text="/delete 7", *, actor=101, chat=101, kind="private"):
    return SimpleNamespace(text=text, from_user=SimpleNamespace(id=actor),
                           chat=SimpleNamespace(id=chat, type=kind))


class BotTripDeletionTests(unittest.TestCase):
    def setUp(self):
        self.service = ModuleType("web_service")
        self.service.TripNotFoundError = web_service.TripNotFoundError
        self.service.delete_trip = Mock(return_value={
            "deleted": True, "trip_id": 7, "files_deleted": 2, "cleanup_pending": False,
            "message": "Поездка удалена. Общая память семьи и использованные попытки сохранены.",
        })
        self.namespace = {"bot": Mock(), "db": Mock(), "logger": Mock()}
        self.namespace["db"].get_account_key.return_value = "synthetic-account-lifetime"
        self.handle = handler(self.namespace)
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"web_service": self.service}).start()
        patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network")).start()
        self.access = patch.object(access_control, "require_access", side_effect=AssertionError("No invite gate")).start()

    def text(self):
        return "\n".join(call.args[1] for call in self.namespace["bot"].send_message.call_args_list)

    def test_shared_service_receives_actor_trip_and_account_lifetime_without_quota(self):
        self.handle(message())
        self.service.delete_trip.assert_called_once_with(101, 7, expected_account_key="synthetic-account-lifetime")
        self.namespace["db"].get_account_key.assert_called_once_with(101)
        self.namespace["db"].delete_trip.assert_not_called()
        self.namespace["db"].claim_ai_request.assert_not_called()
        self.access.assert_not_called()
        self.assertIn("память семьи", self.text())
        self.assertIn("попытки сохранены", self.text())

    def test_direct_call_cannot_delete_from_group_or_mismatched_actor(self):
        for candidate in (message(kind="group"), message(actor=202),
                          SimpleNamespace(chat=SimpleNamespace(id=101, type="private"), text="/delete 7")):
            self.handle(candidate)
        self.service.delete_trip.assert_not_called()
        self.namespace["db"].get_account_key.assert_not_called()
        self.assertIn("личном чате", self.text())

    def test_invalid_numbers_do_not_touch_storage(self):
        for value in ("/delete", "/delete 0", "/delete -1", "/delete nope", "/delete ²", "/delete " + "9" * 5000):
            self.handle(message(value))
        self.service.delete_trip.assert_not_called()
        self.namespace["db"].get_account_key.assert_not_called()
        self.assertIn("например: /delete 3", self.text())

    def test_missing_or_deleting_account_does_not_start_deletion(self):
        self.namespace["db"].get_account_key.return_value = None
        self.handle(message())
        self.service.delete_trip.assert_not_called()
        self.assertIn("не запущено", self.text())

    def test_busy_is_honest_and_does_not_claim_success_or_echo_exception(self):
        self.service.delete_trip.side_effect = trip_deletion.TripDeleteBusyError("private-details")
        self.handle(message())
        self.assertIn("Дождитесь", self.text())
        self.assertIn("/delete 7", self.text())
        self.assertIn("Поездка сохранена", self.text())
        self.assertNotIn("Поездка удалена", self.text())
        self.assertNotIn("private-details", self.text())

    def test_not_owned_missing_and_stale_account_share_not_found_response(self):
        self.service.delete_trip.side_effect = web_service.TripNotFoundError("private-owner")
        self.handle(message())
        self.assertIn("Такой поездки у вас нет", self.text())
        self.assertNotIn("private-owner", self.text())

    def test_precommit_storage_failure_reports_preserved_trip(self):
        self.service.delete_trip.side_effect = trip_deletion.TripDeleteStorageError("private-path")
        self.handle(message())
        self.assertIn("Поездка сохранена", self.text())
        self.assertNotIn("private-path", self.text())

    def test_postcommit_cleanup_pending_reports_deleted_trip_and_safe_retry(self):
        self.service.delete_trip.return_value.update(
            cleanup_pending=True,
            message="Поездка удалена. Очистка некоторых файлов пока не подтверждена.",
        )
        self.handle(message())
        self.assertIn("Поездка удалена", self.text())
        self.assertIn("файлов пока не подтверждена", self.text())
        self.assertIn("/delete 7", self.text())
        self.assertNotIn("Поездка сохранена", self.text())

    def test_unknown_failure_does_not_claim_preserved_or_deleted_and_hides_details(self):
        self.service.delete_trip.side_effect = RuntimeError("private-path-secret")
        self.handle(message())
        self.assertIn("Не удалось подтвердить результат", self.text())
        self.assertIn("/trips", self.text())
        self.assertNotIn("Поездка сохранена", self.text())
        self.assertNotIn("Поездка удалена", self.text())
        self.assertNotIn("private-path-secret", self.text())
        self.namespace["logger"].warning.assert_called_once_with("trip_delete_result_unconfirmed")


class BotTripDeletionStorageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="bot-trip-deletion-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.output = self.directory / "output"
        self.output.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(db, "DB_PATH", str(self.directory / "test.db")).start()
        patch.object(trip_deletion, "OUTPUT_DIR", self.output).start()
        patch.object(socket.socket, "connect", side_effect=AssertionError("Unexpected network")).start()
        db.init_db()
        db.save_user(101, "Synthetic owner")
        db.save_user(202, "Synthetic other")
        self.trip = db.save_trip(101, {"destination": "Synthetic", "start_date": "2099-06-01", "end_date": "2099-06-01"},
                                 {"title": "Synthetic", "days": []})
        db.save_memory_item(101, "synthetic-memory", "custom", "Keep this preference", source_trip_id=self.trip)
        db.save_checklist(self.trip, [("Synthetic", "Keep until deleted")])
        self.account_key = db.get_account_key(101)
        self.namespace = {"db": db, "bot": Mock(), "logger": Mock()}
        self.handle = handler(self.namespace)
        self.paid = patch.object(web_service, "generate_itinerary", side_effect=AssertionError("No model call")).start()
        self.memory = patch.object(web_service.vector_memory, "recall_family_memories", side_effect=AssertionError("No memory recall")).start()
        self.access = patch.object(access_control, "require_access", side_effect=AssertionError("No invite gate")).start()

    def tearDown(self):
        self.paid.assert_not_called()
        self.memory.assert_not_called()
        self.access.assert_not_called()

    def text(self):
        return "\n".join(call.args[1] for call in self.namespace["bot"].send_message.call_args_list)

    def test_real_common_service_deletes_own_trip_and_pdf_preserves_memory(self):
        pdf = self.output / f"trip_{self.trip}.pdf"
        pdf.write_bytes(b"synthetic-pdf")
        with patch.object(db, "claim_ai_request", side_effect=AssertionError("No quota charge")):
            self.handle(message(f"/delete {self.trip}"))
        self.assertIsNone(db.get_trip(self.trip, 101))
        self.assertFalse(pdf.exists())
        self.assertEqual(len(db.list_memory_items(101)), 1)
        self.assertEqual(db.get_account_key(101), self.account_key)
        self.assertIn("Поездка удалена", self.text())

    def test_real_common_service_refuses_other_owner(self):
        self.handle(message(f"/delete {self.trip}", actor=202, chat=202))
        self.assertIsNotNone(db.get_trip(self.trip, 101))
        self.assertIn("Такой поездки у вас нет", self.text())

    def test_real_common_service_refuses_inflight_reminder(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO reminder_deliveries (trip_id,kind,status,claimed_at) VALUES (?, 'eve', 'sending', ?)",
                         (self.trip, "2099-06-01T00:00:00+00:00"))
        self.handle(message(f"/delete {self.trip}"))
        self.assertIsNotNone(db.get_trip(self.trip, 101))
        self.assertEqual(len(db.get_checklist(self.trip)), 1)
        self.assertIn("Дождитесь", self.text())


if __name__ == "__main__":
    unittest.main()
