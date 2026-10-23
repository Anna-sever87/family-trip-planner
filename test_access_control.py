"""Invitation security tests: disposable SQLite only, no network/providers."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import access_control as access
import db


class AccessControlTests(unittest.TestCase):
    OWNER = 9001
    GUEST = 1001

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="offline-access-")))
        self.stack.enter_context(patch.object(db, "DB_PATH", str(self.directory / "synthetic.db")))
        self.stack.enter_context(patch.dict(os.environ, {
            "ACCESS_MODE": "invite", "ACCESS_OWNER_IDS": str(self.OWNER),
            "ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": "12",
        }))
        self.now = datetime(2099, 10, 22, 10, 30, tzinfo=timezone.utc)
        db.init_db()

    def invite(self, **kwargs):
        # Preserve explicit legacy-policy coverage; current defaults are tested separately.
        kwargs.setdefault("max_users", 3)
        kwargs.setdefault("route_limit", 3)
        return access.create_invite(self.OWNER, now=kwargs.pop("now", self.now), **kwargs)

    def join(self, user=None, invite=None, now=None):
        return access.redeem_invite(user or self.GUEST, (invite or self.invite())["token"], now=now or self.now)

    def claim(self, user=None, now=None, **kwargs):
        when = now or self.now
        return db.claim_ai_request(user or self.GUEST, when.date(), "plan", "test", now=when, **kwargs)

    def spend(self, user=None, now=None, success=False, **kwargs):
        claim = self.claim(user, now, **kwargs)
        self.assertEqual(claim.status, "ok")
        db.finish_ai_request(claim.claim_id, succeeded=success, now=now or self.now)
        return claim

    def rows(self, table):
        with db.connect() as conn:
            return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]

    def test_absent_mode_preserves_open_behavior_without_database_reads(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(db, "connect", side_effect=AssertionError("Unexpected DB")):
            self.assertFalse(access.enabled())
            self.assertEqual(access.require_access(self.GUEST)["grant_id"], "open")
            self.assertTrue(access.status(self.GUEST)["allowed"])
            access.reserve_aux(self.GUEST, "cover")

    def test_invalid_modes_fail_closed_including_owner(self):
        for mode in ("", "invte", "disabled", "0"):
            with self.subTest(mode=mode), patch.dict(os.environ, {"ACCESS_MODE": mode}):
                self.assertTrue(access.enabled())
                self.assertFalse(access.status(self.OWNER)["allowed"])
                with self.assertRaises(access.AccessDeniedError):
                    access.require_access(self.OWNER)
                with self.assertRaises(access.AccessDeniedError):
                    access.reserve_aux(self.OWNER, "cover")

    def test_invalid_or_empty_owner_configuration_closes_existing_guest_access(self):
        self.join()
        for owners in ("", "9001,wrong", "-1", "0", "true", str(2**63)):
            with self.subTest(owners=owners), patch.dict(os.environ, {"ACCESS_OWNER_IDS": owners}):
                self.assertFalse(access.status(self.GUEST, self.now)["allowed"])
                self.assertFalse(access.status(self.OWNER, self.now)["allowed"])
                with self.assertRaises(access.AccessDeniedError):
                    access.require_access(self.GUEST, self.now)

    def test_owner_ids_and_owner_status_are_safe_and_do_not_read_database(self):
        with patch.dict(os.environ, {"ACCESS_OWNER_IDS": "9001, 9002;9003\n9004"}):
            for owner in range(9001, 9005):
                self.assertTrue(access.is_owner(owner))
            with patch.object(db, "connect", side_effect=AssertionError("Unexpected DB")):
                self.assertEqual(access.require_access(self.OWNER), {
                    "owner": True, "allowed": True, "expires_at": None,
                    "route_remaining": None, "route_limit": None, "grant_id": "owner",
                })

    def test_invalid_user_ids_are_denied(self):
        for user in (True, 0, -1, "1001", 2**63):
            with self.subTest(user=user):
                self.assertFalse(access.is_owner(user))
                with self.assertRaises(access.AccessDeniedError):
                    access.require_access(user)

    def test_invite_token_has_entropy_is_only_hashed_and_not_listed(self):
        invitation = self.invite()
        self.assertRegex(invitation["token"], r"^[A-Za-z0-9_-]{43}$")
        row = self.rows("access_invites")[0]
        self.assertEqual(len(row["token_hash"]), 64)
        self.assertNotIn(invitation["token"], json.dumps(row))
        self.assertNotIn(invitation["token"].encode(), Path(db.DB_PATH).read_bytes())
        listing = access.list_invites(self.OWNER)
        self.assertEqual(listing[0]["joined_users"], 0)
        self.assertNotIn("token", listing[0])
        self.assertNotIn("token_hash", listing[0])

    def test_invite_expiry_is_fixed_at_creation_and_boundary_is_exclusive(self):
        invitation = self.invite(days=7)
        late = self.now + timedelta(days=6, hours=23)
        result = self.join(invite=invitation, now=late)
        expiry = self.now + timedelta(days=7)
        self.assertEqual(datetime.fromisoformat(result["expires_at"]), expiry)
        self.assertTrue(access.status(self.GUEST, expiry - timedelta(microseconds=1))["allowed"])
        self.assertFalse(access.status(self.GUEST, expiry)["allowed"])
        with self.assertRaises(access.AccessDeniedError):
            access.redeem_invite(1002, invitation["token"], now=expiry)

    def test_parameter_ranges_reject_booleans_and_out_of_bounds(self):
        for field, value in (("days", 0), ("days", 31), ("days", True), ("days", 1.5),
                             ("max_users", 4), ("max_users", 0), ("route_limit", 6), ("route_limit", False)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.invite(**{field: value})
        self.assertEqual(self.rows("access_invites"), [])

    def test_owner_required_for_create_list_and_revoke(self):
        for action in (lambda: access.create_invite(self.GUEST),
                       lambda: access.list_invites(self.GUEST),
                       lambda: access.revoke_invite(self.GUEST, uuid4().hex)):
            with self.assertRaises(access.AccessDeniedError):
                action()

    def test_schema_migration_is_additive_idempotent_and_preserves_key(self):
        self.join()
        before = {table: self.rows(table) for table in ("access_settings", "access_invites", "access_grants")}
        db.save_user(self.GUEST, "Synthetic")
        account = db.get_account_key(self.GUEST)
        db.init_db()
        access.init_schema()
        self.assertEqual(db.get_account_key(self.GUEST), account)
        self.assertEqual(before, {table: self.rows(table) for table in before})

    def test_redeem_same_user_is_idempotent_and_new_link_does_not_reset(self):
        invitation = self.invite()
        initial = self.join(invite=invitation)
        self.spend()
        newer = self.invite(now=self.now + timedelta(days=1))
        repeated = self.join(invite=invitation, now=self.now + timedelta(days=1))
        other_link = self.join(invite=newer, now=self.now + timedelta(days=1))
        self.assertEqual(repeated, other_link)
        self.assertEqual(initial["grant_id"], repeated["grant_id"])
        self.assertEqual(initial["expires_at"], repeated["expires_at"])
        self.assertEqual(repeated["route_remaining"], 2)
        self.assertEqual(len(self.rows("access_grants")), 1)
        self.assertEqual(len(self.rows("access_current_grants")), 1)

    def test_capacity_is_three_accounts_including_repeated_redeems(self):
        invitation = self.invite()
        for user in (1001, 1002, 1003):
            self.join(user, invitation)
        self.join(1001, invitation)
        with self.assertRaises(access.AccessDeniedError):
            self.join(1004, invitation)
        self.assertEqual(access.list_invites(self.OWNER)[0]["joined_users"], 3)

    def test_concurrent_redeems_cannot_exceed_capacity(self):
        invitation = self.invite()
        def redeem(user):
            try:
                self.join(user, invitation)
                return True
            except access.AccessDeniedError:
                return False
        with ThreadPoolExecutor(max_workers=8) as workers:
            results = list(workers.map(redeem, range(1001, 1021)))
        self.assertEqual(sum(results), 3)
        self.assertEqual(len(self.rows("access_grants")), 3)

    def test_concurrent_same_user_join_consumes_only_one_seat(self):
        invitation = self.invite()
        with ThreadPoolExecutor(max_workers=8) as workers:
            results = list(workers.map(lambda _: self.join(invite=invitation), range(16)))
        self.assertEqual(len({row["grant_id"] for row in results}), 1)
        self.assertEqual(len(self.rows("access_grants")), 1)

    def test_owner_redeem_does_not_consume_guest_seat(self):
        invitation = self.invite(max_users=1)
        self.assertEqual(self.join(self.OWNER, invitation)["grant_id"], "owner")
        self.join(invite=invitation)
        self.assertEqual(len(self.rows("access_grants")), 1)

    def test_revoke_immediate_idempotent_and_new_explicit_invite_can_replace_grant(self):
        invitation = self.invite()
        old = self.join(invite=invitation)
        self.spend()
        self.assertTrue(access.revoke_invite(self.OWNER, invitation["invite_id"], now=self.now))
        self.assertTrue(access.revoke_invite(self.OWNER, invitation["invite_id"], now=self.now + timedelta(days=1)))
        self.assertFalse(access.status(self.GUEST, self.now)["allowed"])
        with self.assertRaises(access.AccessDeniedError):
            self.claim()
        with self.assertRaises(access.AccessDeniedError):
            self.join(invite=invitation)
        new = self.join(invite=self.invite())
        self.assertNotEqual(old["grant_id"], new["grant_id"])
        self.assertEqual(new["route_remaining"], 3)
        self.assertEqual(len(self.rows("access_grants")), 2)

    def test_expired_grant_new_explicit_invite_has_new_identity(self):
        previous = self.join(invite=self.invite(days=1))
        later = self.now + timedelta(days=2)
        self.assertFalse(access.status(self.GUEST, later)["allowed"])
        new = self.join(invite=self.invite(now=later), now=later)
        self.assertNotEqual(previous["grant_id"], new["grant_id"])

    def test_bad_tokens_and_unknown_revoke_do_not_change_data_or_echo_token(self):
        for token in ("", "secret-invalid", "x" * 500, "' OR 1=1 --", None, "x" * 43):
            with self.subTest(token=token), self.assertRaises(access.AccessDeniedError) as caught:
                access.redeem_invite(self.GUEST, token, now=self.now)
            if token:
                self.assertNotIn(token, str(caught.exception))
        self.assertFalse(access.revoke_invite(self.OWNER, "invalid"))
        self.assertFalse(access.revoke_invite(self.OWNER, uuid4().hex))
        self.assertEqual(self.rows("access_grants"), [])

    def test_route_quota_is_lifetime_across_days_and_failed_attempts_count(self):
        self.join()
        for offset in range(3):
            self.spend(now=self.now + timedelta(days=offset), success=offset == 1)
        self.assertEqual(access.status(self.GUEST, self.now)["route_remaining"], 0)
        with self.assertRaises(access.AccessQuotaError):
            self.claim(now=self.now + timedelta(days=3))
        self.assertEqual(len(self.rows("access_route_attempts")), 3)

    def test_route_limit_can_be_lowered_per_invite(self):
        self.join(invite=self.invite(route_limit=1))
        self.spend()
        with self.assertRaises(access.AccessQuotaError):
            self.claim()
        self.assertEqual(access.status(self.GUEST, self.now)["route_remaining"], 0)

    def test_profile_deletion_recreation_and_same_token_do_not_reset_grant(self):
        invitation = self.invite()
        original = self.join(invite=invitation)
        for _ in range(3):
            self.spend()
        with db.connect() as conn:
            conn.execute("DELETE FROM users WHERE tg_id = ?", (self.GUEST,))
        db.save_user(self.GUEST, "Recreated")
        repeated = self.join(invite=invitation)
        self.assertEqual(original["grant_id"], repeated["grant_id"])
        self.assertEqual(repeated["route_remaining"], 0)
        with self.assertRaises(access.AccessQuotaError):
            self.claim()
        self.assertEqual(len(self.rows("access_route_attempts")), 3)

    def test_guest_telegram_id_is_not_stored_in_access_ledger(self):
        self.join()
        self.spend()
        for table in ("access_grants", "access_current_grants", "access_route_attempts"):
            for row in self.rows(table):
                self.assertNotIn("tg_id", row)
                self.assertEqual(len(row["subject_hash"]), 64)

    def test_busy_does_not_reserve_invitation_quota_and_owner_keeps_daily_limit(self):
        self.join()
        first = self.claim(daily_limit=1)
        self.assertEqual(self.claim(daily_limit=1).status, "busy")
        self.assertEqual(len(self.rows("access_route_attempts")), 1)
        db.finish_ai_request(first.claim_id, succeeded=False, now=self.now)
        self.spend(self.OWNER, daily_limit=1)
        self.assertEqual(self.claim(self.OWNER, daily_limit=1).status, "limit")
        self.assertEqual(access.status(self.GUEST, self.now)["route_remaining"], 2)

    def test_denied_claim_does_not_create_user_or_paid_attempt(self):
        with self.assertRaises(access.AccessDeniedError):
            self.claim()
        self.assertIsNone(db.get_user(self.GUEST))
        self.assertEqual(self.rows("access_route_attempts"), [])
        self.assertEqual(self.rows("ai_requests"), [])

    def test_insert_failure_rolls_back_both_quota_reservation_and_new_profile(self):
        self.join()
        with db.connect() as conn:
            conn.execute("CREATE TRIGGER deny_test_insert BEFORE INSERT ON ai_requests "
                         "BEGIN SELECT RAISE(ABORT, 'synthetic insertion failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.claim()
        self.assertEqual(access.status(self.GUEST, self.now)["route_remaining"], 3)
        self.assertEqual(self.rows("access_route_attempts"), [])
        self.assertIsNone(db.get_user(self.GUEST))

    def test_global_route_cap_defaults_to_twelve_including_owner_and_failed_attempts(self):
        with patch.dict(os.environ, {"ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": "12"}):
            del os.environ["ACCESS_GLOBAL_ROUTE_DAILY_LIMIT"]
            for _ in range(12):
                self.spend(self.OWNER, daily_limit=50)
            with self.assertRaises(access.AccessQuotaError):
                self.claim(self.OWNER, daily_limit=50)
            self.assertEqual(len(self.rows("access_route_attempts")), 12)
            self.spend(self.OWNER, now=self.now + timedelta(days=1), daily_limit=50)

    def test_global_cap_rejection_does_not_consume_guest_quota(self):
        self.join()
        with patch.dict(os.environ, {"ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": "1"}):
            self.spend(self.OWNER)
            with self.assertRaises(access.AccessQuotaError):
                self.claim()
        self.assertEqual(access.status(self.GUEST, self.now)["route_remaining"], 3)
        self.assertIsNone(db.get_user(self.GUEST))

    def test_invalid_global_cap_fails_closed_before_paid_claim(self):
        for value in ("", "x", "0", "-1", "1001"):
            with self.subTest(value=value), patch.dict(os.environ, {"ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": value}):
                with self.assertRaises(access.AccessQuotaError):
                    self.claim(self.OWNER)
        self.assertEqual(self.rows("access_route_attempts"), [])

    def test_global_budget_day_is_utc_not_the_supplied_usage_date(self):
        local = self.now.replace(hour=0, minute=30, tzinfo=timezone(timedelta(hours=7)))
        self.spend(self.OWNER, now=local)
        self.assertEqual(self.rows("access_route_attempts")[0]["usage_day"], "2099-10-21")

    def test_direct_route_reservation_requires_transaction_and_replay_is_idempotent(self):
        self.join()
        with db.connect() as conn:
            with self.assertRaises(RuntimeError):
                access.reserve_route(conn, self.GUEST, "synthetic", self.now)
            conn.execute("BEGIN IMMEDIATE")
            access.reserve_route(conn, self.GUEST, "synthetic", self.now)
            access.reserve_route(conn, self.GUEST, "synthetic", self.now)
        self.assertEqual(access.status(self.GUEST, self.now)["route_remaining"], 2)
        self.join(1002)
        with db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            with self.assertRaises(access.AccessDeniedError):
                access.reserve_route(conn, 1002, "synthetic", self.now)

    def test_concurrent_route_reservations_never_exceed_grant_limit(self):
        self.join()
        def reserve(_):
            try:
                with db.connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    access.reserve_route(conn, self.GUEST, uuid4().hex, self.now)
                return True
            except access.AccessQuotaError:
                return False
        with ThreadPoolExecutor(max_workers=8) as workers:
            self.assertEqual(sum(workers.map(reserve, range(20))), 3)
        self.assertEqual(len(self.rows("access_route_attempts")), 3)

    def test_auxiliary_limits_are_independent_and_three_for_entire_grant(self):
        self.join()
        for kind in ("cover", "memory"):
            for offset in range(3):
                access.reserve_aux(self.GUEST, kind, self.now + timedelta(days=offset))
            with self.assertRaises(access.AccessQuotaError):
                access.reserve_aux(self.GUEST, kind, self.now)
        self.assertEqual(access.status(self.GUEST, self.now)["route_remaining"], 3)
        self.assertEqual(self.rows("access_route_attempts"), [])

    def test_owner_aux_has_no_invitation_quota_but_unknown_kind_is_rejected(self):
        for _ in range(5):
            access.reserve_aux(self.OWNER, "cover", self.now)
            access.reserve_aux(self.OWNER, "memory", self.now)
        with self.assertRaises(ValueError):
            access.reserve_aux(self.OWNER, "other", self.now)
        self.assertEqual(self.rows("access_grants"), [])

    def test_aux_requires_live_grant_and_revocation_takes_effect(self):
        with self.assertRaises(access.AccessDeniedError):
            access.reserve_aux(self.GUEST, "cover", self.now)
        invitation = self.invite()
        self.join(invite=invitation)
        access.revoke_invite(self.OWNER, invitation["invite_id"], now=self.now)
        with self.assertRaises(access.AccessDeniedError):
            access.reserve_aux(self.GUEST, "memory", self.now)

    def test_concurrent_aux_reservations_never_exceed_three(self):
        self.join()
        def reserve(_):
            try:
                access.reserve_aux(self.GUEST, "cover", self.now)
                return True
            except access.AccessQuotaError:
                return False
        with ThreadPoolExecutor(max_workers=8) as workers:
            self.assertEqual(sum(workers.map(reserve, range(20))), 3)
        self.assertEqual(self.rows("access_grants")[0]["cover_used"], 3)

    def test_read_status_and_require_access_never_consume_budgets(self):
        self.join()
        before = self.rows("access_grants")
        for _ in range(5):
            access.status(self.GUEST, self.now)
            access.require_access(self.GUEST, self.now)
            access.list_invites(self.OWNER)
        self.assertEqual(self.rows("access_grants"), before)
        self.assertEqual(self.rows("access_route_attempts"), [])

    def test_open_mode_route_claims_do_not_spend_invitation_or_global_quota(self):
        with patch.dict(os.environ, {"ACCESS_MODE": "open", "ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": "0"}):
            for _ in range(4):
                self.spend(daily_limit=50)
        self.assertEqual(self.rows("access_route_attempts"), [])

    def test_default_invitation_is_one_person_with_five_lifetime_attempts(self):
        invitation = access.create_invite(self.OWNER, now=self.now)
        self.assertEqual((invitation["max_users"], invitation["route_limit"]), (1, 5))
        grant = self.join(invite=invitation)
        self.assertEqual((grant["route_limit"], grant["route_remaining"]), (5, 5))
        with self.assertRaises(access.AccessDeniedError):
            self.join(1002, invitation)

    def test_five_attempts_can_be_used_today_across_channels_then_sixth_is_denied(self):
        invitation = access.create_invite(self.OWNER, now=self.now)
        self.join(invite=invitation)
        for index in range(5):
            claim = db.claim_ai_request(self.GUEST, self.now.date(),
                                       "plan" if index % 2 == 0 else "edit_day",
                                       "web" if index % 2 == 0 else "telegram",
                                       daily_limit=3, now=self.now)
            self.assertEqual(claim.status, "ok")
            db.finish_ai_request(claim.claim_id, succeeded=index % 2 == 0, now=self.now)
        with self.assertRaises(access.AccessQuotaError):
            self.claim()
        with self.assertRaises(access.AccessQuotaError):
            self.claim(now=self.now + timedelta(days=1))
        self.assertEqual(len(self.rows("ai_requests")), 5)
        self.assertEqual(self.join(invite=invitation)["route_remaining"], 0)

    def test_new_owner_link_after_exhaustion_grants_five_more_without_daily_obstruction(self):
        invitation = access.create_invite(self.OWNER, now=self.now)
        original = self.join(invite=invitation)
        for _ in range(5):
            self.spend()
        self.assertEqual(self.join(invite=invitation)["route_remaining"], 0)
        fresh = access.create_invite(self.OWNER, now=self.now)
        renewed = self.join(invite=fresh)
        self.assertNotEqual(renewed["grant_id"], original["grant_id"])
        self.assertEqual(renewed["route_remaining"], 5)
        for _ in range(5):
            self.spend()
        with self.assertRaises(access.AccessQuotaError):
            self.claim()
        self.assertEqual([row["route_used"] for row in self.rows("access_grants")], [5, 5])

    def test_legacy_invite_schema_migrates_with_all_ledgers_and_existing_quota_unchanged(self):
        invitation = self.invite(max_users=1)
        self.join(invite=invitation)
        self.spend()
        access.reserve_aux(self.GUEST, "memory", self.now)
        tables = ("access_settings", "access_invites", "access_grants", "access_current_grants", "access_route_attempts")
        before = {table: self.rows(table) for table in tables}
        with db.connect() as conn:
            conn.execute("PRAGMA foreign_keys = OFF")
            conn.execute("BEGIN IMMEDIATE")
            schema = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'access_invites'").fetchone()[0]
            conn.execute(schema.replace('"access_invites"', '"legacy_invites"', 1)
                         .replace('CREATE TABLE access_invites', 'CREATE TABLE legacy_invites', 1)
                         .replace('route_limit BETWEEN 1 AND 5', 'route_limit BETWEEN 1 AND 3'))
            conn.execute("INSERT INTO legacy_invites SELECT * FROM access_invites")
            conn.execute("DROP TABLE access_invites")
            conn.execute("ALTER TABLE legacy_invites RENAME TO access_invites")
        original_create = access._create_invites_table
        def fail_copy(conn, name):
            original_create(conn, name)
            if name == "access_invites_route5":
                conn.execute("CREATE TRIGGER migration_failure BEFORE INSERT ON access_invites_route5 "
                             "BEGIN SELECT RAISE(ABORT, 'synthetic migration failure'); END")
        with patch.object(access, "_create_invites_table", side_effect=fail_copy):
            with self.assertRaises(sqlite3.IntegrityError):
                access.init_schema()
        self.assertEqual(before, {table: self.rows(table) for table in tables})
        access.init_schema()
        access.init_schema()
        self.assertEqual(before, {table: self.rows(table) for table in tables})
        self.assertEqual(access.status(self.GUEST, self.now)["route_remaining"], 2)
        self.assertEqual(access.create_invite(self.OWNER, now=self.now)["route_limit"], 5)
        with db.connect() as conn:
            self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
            self.assertEqual(conn.execute("PRAGMA foreign_key_check").fetchall(), [])


if __name__ == "__main__":
    unittest.main()
