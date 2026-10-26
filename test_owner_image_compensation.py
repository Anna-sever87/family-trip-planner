"""Dated owner-only relief; disposable SQLite and mocked image generation."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import os
import socket
import unittest
from unittest.mock import patch

import test_trip_covers as fixtures

covers = fixtures.covers
db = fixtures.db


class OwnerImageCompensationTests(unittest.TestCase):
    DAY = "2099-10-22"

    def setUp(self):
        self.fixture = fixtures.TripCoversTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.stack.enter_context(patch.dict(os.environ, {
            "ACCESS_MODE": "open", "ACCESS_OWNER_IDS": "1",
            "OWNER_IMAGE_DAILY_COMPENSATION": self.DAY + ":9",
        }))
        self.fixture.stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("No network")))
        self.fixture.stack.enter_context(patch.object(socket, "getaddrinfo", side_effect=AssertionError("No DNS")))
        clock = self.fixture.stack.enter_context(patch.object(covers, "datetime", wraps=datetime))
        clock.now.return_value = datetime(2099, 10, 22, 12, tzinfo=timezone.utc)

    def seed_attempts(self, count, day=None):
        with db.connect() as conn:
            conn.executemany("INSERT INTO cover_attempts(attempt_id,utc_day) VALUES (?,?)",
                             [(f"{number:032x}", day or self.DAY) for number in range(count)])

    def test_ten_raw_attempts_leave_nine_owner_slots_and_preserve_history(self):
        self.seed_attempts(10)
        original = self.fixture.rows("cover_attempts")
        for _ in range(9):
            self.assertEqual(self.fixture.submit(self.fixture.save_trip())["status"], "queued")
            self.fixture.manager.executor.run_next()
        with self.assertRaises(covers.CoverLimitError):
            self.fixture.submit(self.fixture.save_trip())
        after = self.fixture.rows("cover_attempts")
        self.assertEqual(len(after), 19)
        self.assertTrue(all(row in after for row in original))
        self.assertEqual(self.fixture.submission.call_count, 9)

    def test_guest_does_not_receive_owner_compensation(self):
        self.seed_attempts(10)
        trip = self.fixture.save_trip(owner=2)
        with self.assertRaises(covers.CoverLimitError):
            self.fixture.manager.submit(2, db.get_account_key(2), trip)
        self.assertEqual(len(self.fixture.rows("cover_attempts")), 10)
        self.fixture.submission.assert_not_called()

    def test_compensation_expires_without_configuration_change(self):
        self.assertEqual(covers.daily_limit_for(1, self.DAY), 19)
        self.assertEqual(covers.daily_limit_for(1, "2099-10-23"), 10)
        self.assertEqual(covers.daily_limit_for(1, "2099-10-21"), 10)

    def test_parser_rejects_malformed_noncanonical_or_excessive_values(self):
        values = ["", self.DAY, self.DAY + ":-1", self.DAY + ":+9", self.DAY + ":09",
                  self.DAY + ":9 ", " " + self.DAY + ":9", self.DAY + ":9:9",
                  self.DAY + ":9\n", self.DAY + ":９", self.DAY + ":41",
                  self.DAY + ":999999999999", "20991022:9", "2099-02-30:9"]
        for value in values:
            with self.subTest(value=value), patch.dict(os.environ, {"OWNER_IMAGE_DAILY_COMPENSATION": value}):
                day = "2099-02-30" if value.startswith("2099-02-30") else self.DAY
                self.assertEqual(covers.daily_limit_for(1, day), 10)
        for invalid_day in (None, 20991022, "20991022"):
            self.assertEqual(covers.daily_limit_for(1, invalid_day), 10)

    def test_base_invalid_never_reopened_and_total_bound_is_fifty(self):
        for base in ("0", "-1", "abc", "1000"):
            with self.subTest(base=base), patch.dict(os.environ, {"IMAGE_GENERATION_DAILY_LIMIT": base}):
                self.assertEqual(covers.daily_limit_for(1, self.DAY), 0)
        with patch.dict(os.environ, {"OWNER_IMAGE_DAILY_COMPENSATION": self.DAY + ":40"}):
            self.assertEqual(covers.daily_limit_for(1, self.DAY), 50)
            with patch.dict(os.environ, {"IMAGE_GENERATION_DAILY_LIMIT": "11"}):
                self.assertEqual(covers.daily_limit_for(1, self.DAY), 11)

    def test_unknown_or_invalid_owner_identity_gets_no_bonus(self):
        for owner in (2, 0, -1, True, "1", None):
            with self.subTest(owner=owner):
                self.assertEqual(covers.daily_limit_for(owner, self.DAY), 10)
        with patch.dict(os.environ, {"ACCESS_OWNER_IDS": "invalid"}):
            self.assertEqual(covers.daily_limit_for(1, self.DAY), 10)

    def test_cached_reads_and_submit_at_exhausted_cap_do_not_spend(self):
        self.fixture.complete()
        self.seed_attempts(18)
        before = self.fixture.rows("cover_attempts")
        self.assertEqual(len(before), 19)
        for _ in range(3):
            self.assertEqual(self.fixture.status()["status"], "ready")
            self.assertEqual(self.fixture.submit()["status"], "ready")
        self.assertEqual(self.fixture.rows("cover_attempts"), before)
        self.fixture.submission.assert_called_once()

    def test_concurrent_independent_managers_reserve_only_final_slot(self):
        self.seed_attempts(18)
        second = covers.TripCovers(executor_factory=fixtures.DeferredExecutor,
                                   cache_dir=self.fixture.directory / "second-covers")
        second.start()
        self.addCleanup(second.close)
        trips = [self.fixture.save_trip(), self.fixture.save_trip()]

        def submit(item):
            manager, trip = item
            try:
                return manager.submit(1, self.fixture.account, trip)["status"]
            except covers.CoverLimitError:
                return "limit"

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(submit, zip((self.fixture.manager, second), trips)))
        self.assertCountEqual(results, ["queued", "limit"])
        self.assertEqual(len(self.fixture.rows("cover_attempts")), 19)
        self.assertEqual(len(self.fixture.rows("trip_covers")), 1)
        self.fixture.submission.assert_not_called()


if __name__ == "__main__":
    unittest.main()
