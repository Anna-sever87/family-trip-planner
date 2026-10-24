"""Telegram invite UX, actual function bodies with no .env, DB or network access."""
import ast
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import Mock, patch


SOURCE = Path(__file__).with_name("bot.py").read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


class Denied(Exception):
    pass


class Quota(Denied):
    pass


class Keyboard:
    def __init__(self, **_kwargs):
        self.buttons = []

    def add(self, *buttons):
        self.buttons.extend(buttons)


def message(user=1, text="/invite", *, chat_type="private", chat_id=None):
    return SimpleNamespace(text=text, from_user=SimpleNamespace(id=user, first_name="Tester"),
                           chat=SimpleNamespace(id=user if chat_id is None else chat_id, type=chat_type))


def callback(user=1, data="invite_revoke:" + "a" * 32, **kwargs):
    return SimpleNamespace(id="callback", data=data, from_user=SimpleNamespace(id=user),
                           message=message(user, **kwargs))


class BotAccessTests(unittest.TestCase):
    def setUp(self):
        self.access = ModuleType("access_control")
        self.access.AccessDeniedError = Denied
        self.access.AccessQuotaError = Quota
        self.access.enabled = Mock(return_value=True)
        self.access.is_owner = Mock(side_effect=lambda user: user == 1)
        self.access.require_access = Mock(return_value={"allowed": True, "grant_id": "grant-A"})
        self.access.reserve_aux = Mock()
        self.access.create_invite = Mock(return_value={"invite_id": "a" * 32, "token": "s" * 43,
            "expires_at": "2099-11-01T00:00:00+00:00", "max_users": 1, "route_limit": 5})
        self.access.redeem_invite = Mock()
        self.access.status = Mock(return_value={"allowed": True, "expires_at": "2099-11-01T00:00:00+00:00", "route_remaining": 5, "route_limit": 5})
        self.access.list_invites = Mock(return_value=[{"invite_id": "a" * 32, "expires_at": "2099-11-01T00:00:00+00:00",
            "max_users": 3, "route_limit": 3, "joined_users": 2, "revoked_at": None}])
        self.access.revoke_invite = Mock(return_value=True)
        self.guard = ModuleType("access_guard")
        self.guard.snapshot = Mock(return_value="grant-A")

        @contextmanager
        def operation(user, expected_grant=None):
            grant = self.access.require_access(user)
            if expected_grant is not None and expected_grant != grant.get("grant_id"):
                raise Denied("Доступ изменился; старая операция не запускается.")
            yield
        self.guard.operation = operation
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"access_control": self.access, "access_guard": self.guard}).start()
        self.ns = {"access_control": self.access, "re": re, "datetime": datetime, "timezone": timezone,
            "Optional": Optional, "bot": Mock(), "states": {}, "threading": Mock(), "logger": Mock(),
            "types": SimpleNamespace(ReplyKeyboardRemove=lambda: "remove", InlineKeyboardMarkup=Keyboard,
                InlineKeyboardButton=lambda text, **kwargs: SimpleNamespace(text=text, **kwargs),
                BotCommand=lambda command, description: SimpleNamespace(command=command, description=description)),
            "db": Mock(), "vector_memory": Mock(), "TripRequest": Mock(), "Itinerary": Mock(),
            "privacy": SimpleNamespace(delete_all_user_data=Mock(), AccountDeletionBusyError=type("AccountDeletionBusyError", (Exception,), {})),
            "CityResolutionError": type("CityResolutionError", (Exception,), {}),
            "_finish_memory_job": Mock(), "_finish_generation_job": Mock(), "_finish_ai_claim": Mock(),
            "_reserve_memory_job": Mock(return_value=True), "_reserve_generation_job": Mock(return_value=True),
            "_is_generation_active": Mock(return_value=False), "used_today": Mock(return_value=3), "DAILY_LIMIT": 3, "ask": Mock(),
            "_claim_route_ai_request": Mock(return_value=SimpleNamespace(status="ok", claim_id="claim")),
            "minimum_meal_cost": Mock(return_value=0), "city_candidates_enabled": Mock(return_value=False),
            "resolve_city_for_planning": Mock(return_value=None),
            "_clean_memory_text": lambda text, _limit: text, "MAX_MANUAL_MEMORY_CHARS": 600, "MAX_FEEDBACK_CHARS": 600}
        self.ns["bot"].get_me.return_value = SimpleNamespace(username="SyntheticTravelBot")
        names = {"_private_actor", "_message_access_blocked", "handle_access_denied", "_callback_access_blocked",
            "handle_callback_access_denied", "_invite_expiry", "_owner_invite_chat", "handle_invite", "handle_invites",
            "_send_owner_invitation", "_handle_invite_command", "handle_invite_one", "_owner_invite_callback", "handle_invite_create",
            "handle_revoke", "handle_invite_revoke", "handle_start", "setup_commands", "handle_delete_data", "_delete_account_job", "_message_user_id", "handle_remember", "handle_feedback",
            "_save_custom_memory_job", "_save_feedback_memory_job", "generation_job", "edit_generation_job", "start_generation", "handle_edit", "handle_plan"}
        body = []
        for item in TREE.body:
            if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                    {"ACCESS_SAFE_COMMANDS", "ACCESS_DENIED_TEXT", "HELP_TEXT"} for t in item.targets):
                body.append(item)
            if isinstance(item, ast.FunctionDef) and item.name in names:
                item = ast.parse(ast.get_source_segment(SOURCE, item)).body[0]
                item.decorator_list = []
                body.append(item)
        exec(compile(ast.Module(body=body, type_ignores=[]), "bot.py", "exec"), self.ns)

    def text(self):
        return "\n".join(str(call.args[1]) for call in self.ns["bot"].send_message.call_args_list)

    def test_gate_is_registered_before_commands_questionnaire_and_callbacks(self):
        positions = {item.name: index for index, item in enumerate(TREE.body) if isinstance(item, ast.FunctionDef)}
        for name in ("handle_plan", "handle_web", "handle_answer", "handle_edit", "handle_remember", "handle_feedback"):
            self.assertLess(positions["handle_access_denied"], positions[name])
        for name in ("handle_check", "handle_memory_clear", "handle_account_delete", "handle_invite_revoke", "handle_invite_create"):
            self.assertLess(positions["handle_callback_access_denied"], positions[name])

    def test_denied_or_failed_read_blocks_commands_and_questionnaire_but_preserves_privacy(self):
        for failure in (Denied("No access"), RuntimeError("unavailable")):
            self.access.require_access.side_effect = failure
            for text in ("/plan", "/web", "/edit 1 1 text", "/remember text", "/feedback 1 text", "Сочи, Россия", "/trips"):
                self.assertTrue(self.ns["_message_access_blocked"](message(2, text)))
            for text in ("/start invite_secret", "/help", "/cancel", "/delete_data", "/forget_memory", "/memory", "/delete 1"):
                self.assertFalse(self.ns["_message_access_blocked"](message(2, text)))
        self.assertTrue(self.ns["_message_access_blocked"](message(2, "/help", chat_type="group")))
        self.assertTrue(self.ns["_message_access_blocked"](message(2, "/help", chat_id=3)))
        self.ns["states"][2] = {"step": "budget"}
        self.ns["handle_access_denied"](message(2))
        self.assertNotIn(2, self.ns["states"])

    def test_callbacks_fail_closed_except_owner_bound_privacy_actions(self):
        self.access.require_access.side_effect = Denied("No access")
        for data in ("check:1", "invite_revoke:" + "a" * 32):
            self.assertTrue(self.ns["_callback_access_blocked"](callback(2, data)))
        for data in ("memory_clear:confirm", "account_delete:confirm"):
            self.assertFalse(self.ns["_callback_access_blocked"](callback(2, data)))
            self.assertTrue(self.ns["_callback_access_blocked"](callback(2, data, chat_id=3)))

    def test_owner_invite_link_is_bounded_explicit_private_and_not_logged(self):
        self.ns["handle_invite"](message())
        self.access.create_invite.assert_called_once_with(1, days=7, max_users=1, route_limit=5)
        text = self.text()
        self.assertIn("https://t.me/SyntheticTravelBot?start=invite_" + "s" * 43, text)
        self.assertIn("за весь срок", text)
        self.assertIn("не ежедневно", text)
        self.assertLessEqual(len("invite_" + "s" * 43), 64)
        self.assertEqual(self.ns["logger"].mock_calls, [])

    def test_new_invitations_have_no_owner_count_cap_and_days_are_validated(self):
        for _ in range(4):
            self.ns["handle_invite"](message(text="/invite 30"))
        self.assertEqual(self.access.create_invite.call_count, 4)
        for text in ("/invite 0", "/invite 31", "/invite -1", "/invite 7 3", "/invite nope", "/invite ²", "/invite " + "9" * 5000):
            self.ns["handle_invite"](message(text=text))
        self.assertEqual(self.access.create_invite.call_count, 4)

    def test_nonowner_and_group_cannot_manage_invites_even_direct_handler_calls(self):
        for handler in ("handle_invite", "handle_invite_one", "handle_invites", "handle_revoke"):
            self.ns[handler](message(2))
            self.ns[handler](message(1, chat_type="group"))
        self.ns["handle_invite_revoke"](callback(2))
        self.ns["handle_invite_revoke"](callback(1, chat_type="group"))
        self.access.create_invite.assert_not_called()
        self.access.list_invites.assert_not_called()
        self.access.revoke_invite.assert_not_called()

    def test_redeeming_start_link_uses_actor_and_explains_total_not_daily_allowance(self):
        self.ns["handle_start"](message(2, "/start invite_" + "s" * 43))
        self.access.redeem_invite.assert_called_once_with(2, "s" * 43)
        self.assertIn("Осталось попыток создания и изменения маршрутов: 5", self.text())
        self.assertIn("не ежедневно", self.text())
        self.assertNotIn("s" * 43, self.text())

    def test_owner_read_error_blocks_admin_callback_but_keeps_help_available(self):
        self.access.is_owner.side_effect = RuntimeError("permission store unavailable")
        self.ns["handle_invite_revoke"](callback())
        self.ns["handle_start"](message(2, "/help"))
        self.access.revoke_invite.assert_not_called()
        self.assertIn(self.ns["HELP_TEXT"], self.text())
        self.assertNotIn("Для владельца", self.text())

    def test_invalid_or_used_invite_never_logs_or_echoes_token(self):
        self.access.redeem_invite.side_effect = ValueError("secret-token-unsafe")
        for token in ("secret-token-unsafe", "x" * 58, "a b"):
            self.ns["handle_start"](message(2, "/start invite_" + token))
        self.assertIn("недействительно", self.text())
        self.assertNotIn("secret-token-unsafe", self.text())
        self.assertEqual(self.ns["logger"].mock_calls, [])

    def test_invites_list_and_revoke_callback_never_expose_secret_token(self):
        self.ns["handle_invites"](message(text="/invites"))
        self.assertIn("участников 2/3", self.text())
        self.assertIn("a" * 32, self.text())
        keyboard = self.ns["bot"].send_message.call_args.kwargs["reply_markup"]
        self.assertEqual(keyboard.buttons[0].callback_data, "invite_revoke:" + "a" * 32)
        self.assertNotIn("s" * 43, self.text())
        self.ns["handle_invite_revoke"](callback())
        self.access.revoke_invite.assert_called_once_with(1, "a" * 32)
        self.assertIn("Уже отправленный", self.text())

    def test_full_id_command_can_revoke_older_invite_outside_last_twenty(self):
        self.ns["handle_revoke"](message(text="/revoke " + "b" * 32))
        self.access.revoke_invite.assert_called_once_with(1, "b" * 32)

    def test_direct_paid_handlers_deny_before_any_thread_or_data_access(self):
        self.access.require_access.side_effect = Denied("Доступ закрыт")
        for handler, text in (("handle_remember", "/remember note"), ("handle_feedback", "/feedback 1 note"),
                              ("handle_edit", "/edit 1 1 note")):
            self.ns[handler](message(2, text))
        self.ns["start_generation"](message(2), {})
        self.ns["threading"].Thread.assert_not_called()
        self.ns["db"].get_trip.assert_not_called()
        self.ns["_claim_route_ai_request"].assert_not_called()

    def test_route_workers_recheck_revoked_grant_before_providers_and_release_locks(self):
        self.access.require_access.side_effect = Denied("Доступ закрыт")
        self.ns["generation_job"](2, 2, Mock(), "claim", access_grant="grant-A")
        self.ns["edit_generation_job"](2, 1, 1, "edit", "claim", access_grant="grant-A")
        self.ns["vector_memory"].recall_family_memories.assert_not_called()
        self.ns["db"].get_trip.assert_not_called()
        self.assertEqual(self.ns["_finish_generation_job"].call_count, 2)
        self.assertEqual(self.ns["_finish_ai_claim"].call_count, 2)
        self.assertEqual(self.ns["logger"].mock_calls, [])

    def test_reinvitation_does_not_reanimate_queued_old_grant(self):
        self.access.require_access.return_value = {"allowed": True, "grant_id": "grant-B"}
        self.ns["generation_job"](2, 2, Mock(), "claim", access_grant="grant-A")
        self.ns["vector_memory"].recall_family_memories.assert_not_called()
        self.assertIn("старая операция", self.text())

    def test_manual_memory_quota_is_reserved_before_embedding_and_denial_releases_lock(self):
        self.access.reserve_aux.side_effect = Quota("Лимит памяти исчерпан")
        self.ns["_save_custom_memory_job"](2, 2, "note", access_grant="grant-A")
        self.ns["_save_feedback_memory_job"](2, 2, 1, "City", "note", access_grant="grant-A")
        self.assertEqual(self.access.reserve_aux.call_args_list[0].kwargs, {"kind": "memory"})
        self.assertEqual(self.access.reserve_aux.call_count, 2)
        self.ns["vector_memory"].save_custom_memory.assert_not_called()
        self.ns["vector_memory"].save_feedback_memory.assert_not_called()
        self.assertEqual(self.ns["_finish_memory_job"].call_count, 2)

    def test_manual_memory_enqueue_captures_grant_without_consuming_aux_quota(self):
        self.ns["handle_remember"](message(2, "/remember Important preference"))
        kwargs = self.ns["threading"].Thread.call_args.kwargs
        self.assertEqual(kwargs["kwargs"], {"access_grant": "grant-A"})
        self.access.reserve_aux.assert_not_called()

    def test_generation_enqueue_captures_grant_and_atomic_quota_denial_releases_slot(self):
        req = SimpleNamespace(program_budget=10000, days_count=1, destination_kind="region", adults=2,
                              children_ages=[], currency="RUB", pace="средний", wishes="", destination="City")
        self.ns["TripRequest"].return_value = req
        self.ns["start_generation"](message(2), {})
        self.assertEqual(self.ns["threading"].Thread.call_args.kwargs["kwargs"], {"access_grant": "grant-A"})
        self.ns["threading"].reset_mock()
        self.ns["_claim_route_ai_request"].side_effect = Quota("Три попытки уже использованы")
        self.ns["start_generation"](message(2), {})
        self.ns["threading"].Thread.assert_not_called()
        self.ns["_finish_generation_job"].assert_called_with(2)
        self.assertIn("Три попытки", self.text())

    def test_individual_command_defaults_to_one_account_five_lifetime_attempts(self):
        self.ns["handle_invite_one"](message(text="/invite_one"))
        self.access.create_invite.assert_called_once_with(1, days=7, max_users=1, route_limit=5)
        self.assertIn("Ссылка для 1 аккаунта", self.text())
        self.assertIn("за весь срок", self.text())
        self.assertNotIn("Ссылка для 3 пользователей", self.text())
        self.assertEqual(self.ns["logger"].mock_calls, [])

    def test_reinvitation_between_snapshot_and_claim_never_spends_new_grant_or_starts_worker(self):
        request = SimpleNamespace(program_budget=10000, days_count=1, destination_kind="region", adults=2,
                                  children_ages=[], currency="RUB", pace="средний", wishes="", destination="City")
        self.ns["TripRequest"].return_value = request
        self.ns["db"].get_trip.return_value = {"itinerary": {}}
        self.ns["Itinerary"].model_validate.return_value = SimpleNamespace(days=[1], city_schedule=False)
        self.ns["_clean_edit_request"] = lambda text: text

        def change_grant_before_claim(_chat_id):
            self.access.require_access.return_value = {"allowed": True, "grant_id": "grant-B"}
            return True

        self.ns["_reserve_generation_job"].side_effect = change_grant_before_claim
        for handler in ("start_generation", "handle_edit"):
            with self.subTest(handler=handler):
                self.access.require_access.return_value = {"allowed": True, "grant_id": "grant-A"}
                self.guard.snapshot.return_value = "grant-A"
                if handler == "start_generation":
                    self.ns[handler](message(2), {})
                else:
                    self.ns[handler](message(2, "/edit 1 1 Изменить музей"))
                self.ns["_claim_route_ai_request"].assert_not_called()
                self.ns["threading"].Thread.assert_not_called()
                self.ns["_finish_generation_job"].assert_called_with(2)
        self.assertIn("старая операция", self.text())

    def test_deletion_confirmation_and_result_disclose_preserved_access_ledgers(self):
        self.ns["handle_delete_data"](message(2, "/delete_data"))
        confirmation = self.text()
        self.assertIn("хешированные идентификаторы", confirmation)
        self.assertIn("о выдаче доступа и расходе попыток", confirmation)
        self.assertIn("псевдонимизированы, но не анонимны", confirmation)
        self.assertNotIn("До следующего дня", confirmation)
        self.ns["bot"].reset_mock()
        self.ns["_delete_account_job"](2, 2)
        self.ns["privacy"].delete_all_user_data.assert_called_once_with(2, actor_tg_id=2, confirm=True)
        self.assertIn("сохранены хешированные идентификаторы", self.text())
        self.assertIn("о выдаче доступа и расходе попыток", self.text())
        self.assertNotIn("следующего дня", self.text())

    def test_individual_command_validates_days_and_never_accepts_a_supplied_capacity(self):
        self.ns["handle_invite_one"](message(text="/invite_one 30"))
        self.access.create_invite.assert_called_once_with(1, days=30, max_users=1, route_limit=5)
        for text in ("/invite_one 0", "/invite_one 31", "/invite_one ²", "/invite_one 7 9"):
            self.ns["handle_invite_one"](message(text=text))
        self.assertEqual(self.access.create_invite.call_count, 1)

    def test_list_offers_only_personal_creation_even_when_no_invites_exist(self):
        self.access.list_invites.return_value = []
        self.ns["handle_invites"](message(text="/invites"))
        keyboard = self.ns["bot"].send_message.call_args.kwargs["reply_markup"]
        self.assertEqual([(item.text, item.callback_data) for item in keyboard.buttons], [
            ("Пригласить одного", "invite_create:one")])
        self.assertIn("7 дней", self.text())
        self.assertIn("5 попыток на весь срок", self.text())

    def test_creation_callbacks_allow_only_personal_invites_and_reject_old_group_button(self):
        self.ns["handle_invite_create"](callback(1, "invite_create:one"))
        self.access.create_invite.assert_called_once_with(1, days=7, max_users=1, route_limit=5)
        for data in ("invite_create:three", "invite_create:4", "invite_create:999", "invite_create:one:30", "invite_create:2", "invite_create:"):
            self.ns["handle_invite_create"](callback(1, data))
        self.assertEqual(self.access.create_invite.call_count, 1)

    def test_creation_callback_rejects_guests_groups_impersonation_and_permission_errors(self):
        for call in (callback(2, "invite_create:one"), callback(1, "invite_create:one", chat_type="group"),
                     callback(1, "invite_create:three", chat_id=2)):
            self.ns["handle_invite_create"](call)
        self.access.is_owner.side_effect = RuntimeError("permission unavailable")
        self.ns["handle_invite_create"](callback(1, "invite_create:one"))
        self.access.create_invite.assert_not_called()
        self.ns["bot"].get_me.assert_not_called()

    def test_shared_invite_delivery_fails_closed_for_invalid_actor_or_capacity(self):
        for owner, chat, capacity in ((2, 2, 1), (1, 2, 1), (1, 1, 2), (1, 1, 3), (1, 1, 4), (1, 1, True), (1, 1, "3")):
            self.ns["_send_owner_invitation"](owner, chat, days=7, max_users=capacity)
        self.access.create_invite.assert_not_called()

    def test_owner_help_and_simple_commands_offer_personal_invites(self):
        self.ns["handle_start"](message(text="/help"))
        self.assertIn("/invite_one [дни]", self.text())
        self.assertIn("/invite [дни]", self.text())
        self.assertIn("5 попыток", self.text())
        self.assertNotIn("ссылка на 3", self.text())
        self.ns["setup_commands"]()
        commands = {command.command for command in self.ns["bot"].set_my_commands.call_args.args[0]}
        self.assertTrue({"invite_one", "invite", "invites"}.issubset(commands))

    def test_guest_can_start_questionnaire_after_three_daily_operations(self):
        self.access.require_access.return_value = {"owner": False, "grant_id": "grant-A", "route_remaining": 2}
        self.ns["handle_plan"](message(2, "/plan"))
        self.ns["ask"].assert_called_once_with(2, "destination_kind")
        self.ns["used_today"].assert_not_called()

    def test_exhausted_guest_cannot_start_but_owner_retains_daily_protection(self):
        for grant in ({"owner": False, "grant_id": "grant-A", "route_remaining": 0},
                      {"owner": True, "grant_id": "owner", "route_remaining": None}):
            with self.subTest(grant=grant):
                self.access.require_access.return_value = grant
                self.ns["handle_plan"](message(2, "/plan"))
                self.ns["ask"].assert_not_called()


if __name__ == "__main__":
    unittest.main()
