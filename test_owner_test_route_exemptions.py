"""Dated owner test exclusions preserve personal/guest quotas and all ledgers.

Only disposable SQLite, synthetic identities and stubbed external I/O are used.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import socket
import sqlite3
import tempfile
import threading
import unittest
from unittest.mock import patch
from uuid import uuid4

with patch("dotenv.load_dotenv", return_value=False):
    import access_control as access
    import db
    import privacy


class OwnerTestRouteExemptionTests(unittest.TestCase):
    OWNER, OTHER, GUEST = 9001, 9002, 1001

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="offline-owner-tests-")))
        self.stack.enter_context(patch.object(db, "DB_PATH", str(self.directory / "synthetic.db")))
        self.stack.enter_context(patch.dict(os.environ, {
            "ACCESS_MODE": "invite", "ACCESS_OWNER_IDS": str(self.OWNER),
            "ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": "10", "OWNER_TEST_ROUTE_EXEMPTIONS": "",
        }))
        self.stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("No network")))
        self.stack.enter_context(patch.object(socket, "getaddrinfo", side_effect=AssertionError("No DNS")))
        self.now = datetime(2099, 10, 22, 10, 30, tzinfo=timezone.utc)
        self.day = self.now.date().isoformat()
        db.init_db()

    def configure(self, identifiers, day=None):
        os.environ["OWNER_TEST_ROUTE_EXEMPTIONS"] = (day or self.day) + ":" + ",".join(identifiers)

    def claim(self, user=None, *, now=None, usage_day=None):
        now = now or self.now
        return db.claim_ai_request(user or self.OWNER, usage_day or now.date(), "plan", "synthetic-test",
                                   now=now, daily_limit=10)

    def spend(self, user=None, *, succeeded=False, now=None, usage_day=None):
        claim = self.claim(user, now=now, usage_day=usage_day)
        self.assertEqual(claim.status, "ok")
        db.finish_ai_request(claim.claim_id, succeeded=succeeded, now=now or self.now)
        return claim.claim_id

    def rows(self, table):
        with db.connect() as conn:
            return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]

    def invite_guest(self, limit=5):
        invite = access.create_invite(self.OWNER, route_limit=limit, now=self.now)
        access.redeem_invite(self.GUEST, invite['token'], now=self.now)

    def test_parser_is_bounded_exact_date_and_all_or_nothing(self):
        identifier = "a" * 32
        self.configure([identifier.upper()])
        self.assertEqual(access.owner_test_claim_ids(self.day), (identifier,))
        self.assertEqual(access.owner_test_claim_ids("2099-10-23"), ())
        bad_values = ["", self.day + ":", self.day + ":" + identifier + ",",
                      self.day + ":" + identifier + "," + identifier.upper(),
                      self.day + ":" + identifier + ",not-a-claim",
                      self.day + ":" + identifier + " ", "20991022:" + identifier,
                      "2099-02-30:" + identifier, self.day + ":" + "я" * 32,
                      self.day + ":" + "0" * 10241,
                      self.day + ":" + ",".join(f"{i:032x}" for i in range(101))]
        for value in bad_values:
            with self.subTest(value_length=len(value)), patch.dict(os.environ, {"OWNER_TEST_ROUTE_EXEMPTIONS": value}):
                self.assertEqual(access.owner_test_claim_ids(self.day), ())
        self.configure([f"{i:032x}" for i in range(100)])
        self.assertEqual(len(access.owner_test_claim_ids(self.day)), 100)
        for invalid_day in (None, True, "20991022", "2099-2-2", "2099-02-30"):
            self.assertEqual(access.owner_test_claim_ids(invalid_day), ())

    def test_three_authorized_tests_leave_ten_personal_attempts_without_deleting_history(self):
        tests = [self.spend(succeeded=i == 0) for i in range(3)]
        before = self.rows("ai_requests")
        self.configure(tests)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.day), 0)
        self.assertEqual(self.rows("ai_requests"), before)
        for _ in range(10):
            self.spend(succeeded=False)
        rejected = self.claim()
        self.assertEqual((rejected.status, rejected.used, rejected.limit), ("limit", 10, 10))
        self.assertEqual(len(self.rows("ai_requests")), 13)
        self.assertEqual(len(self.rows("access_route_attempts")), 13)
        self.assertEqual({r['claim_id'] for r in self.rows("ai_requests") if r['claim_id'] in tests}, set(tests))

    def test_guests_unknown_claims_and_nonowner_rows_are_never_exempt(self):
        self.invite_guest()
        guest = self.spend(self.GUEST)
        owner = self.spend()
        self.configure([guest, owner, uuid4().hex])
        self.assertEqual(db.get_ai_request_usage(self.GUEST, self.day), 1)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.day), 0)
        self.assertEqual(access.status(self.GUEST, self.now)['route_remaining'], 4)
        with patch.dict(os.environ, {"ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": "2"}):
            self.spend()
            with self.assertRaises(access.AccessQuotaError):
                self.claim()
        self.assertEqual(access.status(self.GUEST, self.now)['route_remaining'], 4)

    def test_guest_grant_rows_stay_in_global_count_even_after_account_becomes_owner(self):
        self.invite_guest()
        guest = self.spend(self.GUEST)
        self.configure([guest])
        with patch.dict(os.environ, {"ACCESS_OWNER_IDS": f"{self.OWNER},{self.GUEST}",
                                    "ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": "1"}):
            with self.assertRaises(access.AccessQuotaError):
                self.claim()

    def test_unknown_or_other_owner_claim_id_cannot_reduce_personal_count(self):
        with patch.dict(os.environ, {"ACCESS_OWNER_IDS": f"{self.OWNER},{self.OTHER}"}):
            self.spend()
            other = self.spend(self.OTHER)
            self.configure([other, uuid4().hex])
            self.assertEqual(db.get_ai_request_usage(self.OWNER, self.day), 1)
            self.assertEqual(db.get_ai_request_usage(self.OTHER, self.day), 0)

    def test_changed_owner_configuration_does_not_exempt_former_owner_rows(self):
        test = self.spend()
        self.configure([test])
        with patch.dict(os.environ, {"ACCESS_OWNER_IDS": str(self.OTHER),
                                    "ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": "1"}):
            self.assertEqual(db.get_ai_request_usage(self.OWNER, self.day), 1)
            with self.assertRaises(access.AccessQuotaError):
                self.claim(self.OTHER)

    def test_insert_failure_with_exemptions_rolls_back_both_reservations(self):
        test = self.spend()
        self.configure([test])
        before_ai, before_access = self.rows("ai_requests"), self.rows("access_route_attempts")
        with db.connect() as conn:
            conn.execute("CREATE TRIGGER deny_synthetic_claim BEFORE INSERT ON ai_requests "
                         "BEGIN SELECT RAISE(ABORT, 'synthetic insertion failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.claim()
        self.assertEqual(self.rows("ai_requests"), before_ai)
        self.assertEqual(self.rows("access_route_attempts"), before_access)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.day), 0)

    def test_wrong_date_does_not_refund_old_or_new_personal_or_global_usage(self):
        test = self.spend()
        tomorrow = self.now + timedelta(days=1)
        self.configure([test], tomorrow.date().isoformat())
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.day), 1)
        with patch.dict(os.environ, {"ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": "1"}):
            with self.assertRaises(access.AccessQuotaError):
                self.claim()
            self.spend(now=tomorrow)
        self.configure([test])
        self.assertEqual(db.get_ai_request_usage(self.OWNER, tomorrow.date()), 1)

    def test_personal_date_and_utc_access_day_remain_independent(self):
        local_day = (self.now + timedelta(days=1)).date().isoformat()
        test = self.spend(usage_day=local_day)
        self.configure([test], self.day)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, local_day), 1)
        # UTC access ledger may exempt the same test, without altering the
        # separate requested local-date personal ledger.
        with patch.dict(os.environ, {"ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": "1"}):
            self.spend(usage_day=local_day)
        self.assertEqual(db.get_ai_request_usage(self.OWNER, local_day), 2)

    def test_failed_expired_and_running_nonexempt_attempts_keep_existing_rules(self):
        failed = self.spend()
        expired = self.claim().claim_id
        later = self.now + timedelta(minutes=31)
        running = self.claim(now=later)
        self.assertEqual(running.status, "ok")
        self.configure([failed])
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.day), 2)
        self.assertEqual(self.claim(now=later).status, "busy")
        self.configure([failed, expired, running.claim_id])
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.day), 0)
        self.assertEqual(self.claim(now=later).status, "busy")

    def test_deletion_retains_adjusted_usage_once_and_preserves_global_ledger(self):
        tests = [self.spend() for _ in range(2)]
        self.spend()
        self.spend()
        self.configure(tests)
        original_access = self.rows("access_route_attempts")
        for expected in (2, 3):
            with patch.object(privacy, "OUTPUT_DIR", self.directory / "empty-output"), \
                 patch.object(privacy.vector_memory, "clear_family_memories", return_value=True), \
                 patch.object(privacy, "datetime", wraps=datetime) as clock, \
                 patch.object(privacy.reminders, "get_tz", return_value=timezone.utc):
                clock.now.return_value = self.now
                result = privacy.delete_all_user_data(self.OWNER, actor_tg_id=self.OWNER, confirm=True)
            self.assertTrue(result.deleted)
            self.assertEqual(db.get_ai_request_usage(self.OWNER, self.day), expected)
            self.assertEqual(self.rows("ai_requests"), [])
            self.assertEqual(self.rows("ai_quota_retention")[0]['used'], expected)
            db.save_user(self.OWNER, "Synthetic recreated")
            self.spend()
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.day), 4)
        current = {row['claim_id']: row for row in self.rows("access_route_attempts")}
        for row in original_access:
            self.assertEqual(current[row['claim_id']], row)
        with patch.dict(os.environ, {"ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": "4"}):
            with self.assertRaises(access.AccessQuotaError):
                self.claim()

    def test_concurrent_personal_requests_cannot_exceed_ten(self):
        tests = [self.spend() for _ in range(2)]
        self.configure(tests)
        for _ in range(9):
            self.spend()
        barrier = threading.Barrier(8)
        def attempt(_):
            barrier.wait()
            return self.claim().status
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(attempt, range(8)))
        self.assertEqual(results.count("ok"), 1)
        self.assertTrue(set(results) <= {"ok", "busy", "limit"})
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.day), 10)
        self.assertEqual(len(self.rows("ai_requests")), 12)
        self.assertEqual(len(self.rows("access_route_attempts")), 12)

    def test_global_limit_remains_atomic_across_owners_with_exemptions(self):
        test = self.spend()
        self.configure([test])
        barrier = threading.Barrier(2)
        def attempt(owner):
            barrier.wait()
            try:
                return self.claim(owner).status
            except access.AccessQuotaError:
                return "global_limit"
        with patch.dict(os.environ, {"ACCESS_OWNER_IDS": f"{self.OWNER},{self.OTHER}",
                                    "ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": "1"}):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(attempt, [self.OWNER, self.OTHER]))
        self.assertEqual(sorted(results), ["global_limit", "ok"])
        self.assertEqual(len(self.rows("access_route_attempts")), 2)

    def test_invalid_configuration_never_reduces_any_usage(self):
        test = self.spend()
        self.configure([test, test])
        self.assertEqual(db.get_ai_request_usage(self.OWNER, self.day), 1)
        with patch.dict(os.environ, {"ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": "1"}):
            with self.assertRaises(access.AccessQuotaError):
                self.claim()
        self.configure([test])
        with patch.dict(os.environ, {"ACCESS_OWNER_IDS": ""}):
            self.assertEqual(db.get_ai_request_usage(self.OWNER, self.day), 1)


if __name__ == "__main__":
    unittest.main()
