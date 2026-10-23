"""Disposable SQLite/files and mocked YandexART; never paid requests."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image
with patch("dotenv.load_dotenv", return_value=False):
    import db
    import trip_covers as covers
    import web_app
    import web_auth
    import yandex_art
from fastapi.testclient import TestClient


class DeferredExecutor:
    def __init__(self, **kwargs):
        self.calls = []
        self.options = kwargs

    def submit(self, function, *args):
        self.calls.append((function, args))

    def run_next(self):
        function, args = self.calls.pop(0)
        function(*args)

    def shutdown(self, **kwargs):
        self.calls.clear()


class TripCoversTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="offline-trip-covers-"))
        self.directory = Path(directory)
        self.stack.enter_context(patch.object(db, "DB_PATH", str(self.directory / "synthetic.db")))
        self.stack.enter_context(patch.dict("os.environ", {
            "IMAGE_GENERATION_ENABLED": "1", "IMAGE_GENERATION_DAILY_LIMIT": "10",
            "YANDEX_ART_API_KEY": "synthetic-art-key", "YANDEX_FOLDER_ID": "synthetic-folder",
            "YANDEX_ART_MODEL_URI": "", "WEB_SESSION_SECRET": "synthetic-offline-cover-session-secret-123456789",
        }))
        db.init_db()
        for owner in (1, 2):
            db.save_user(owner, "Synthetic", 2, [], "RUB")
        self.account = db.get_account_key(1)
        self.trip = self.save_trip()
        self.manager = covers.TripCovers(executor_factory=DeferredExecutor,
                                        cache_dir=self.directory / "covers")
        self.manager.start()
        self.addCleanup(self.manager.close)
        payload = BytesIO()
        Image.new("RGB", (16, 9), "blue").save(payload, format="JPEG")
        self.image = yandex_art.validate_image(payload.getvalue())
        self.submission = self.stack.enter_context(patch.object(yandex_art, "submit", return_value=
            yandex_art.ArtOperation("synthetic-operation", True, self.image)))
        self.poll = self.stack.enter_context(patch.object(yandex_art, "get_operation", return_value=
            yandex_art.ArtOperation("synthetic-operation", True, self.image)))

    def save_trip(self, owner=1, destination="Москва, Россия"):
        return db.save_trip(owner, {"destination": destination, "start_date": "2099-10-20",
                                   "end_date": "2099-10-23", "wishes": "private-family-information"},
                            {"title": "Synthetic", "days": []})

    def submit(self, trip=None):
        return self.manager.submit(1, self.account, trip or self.trip)

    def status(self, trip=None):
        return self.manager.get_status(1, self.account, trip or self.trip)

    def rows(self, table):
        with db.connect() as conn:
            return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]

    def complete(self):
        self.submit()
        self.manager.executor.run_next()
        return self.status()

    def reviewed_image(self, color="orange"):
        content = BytesIO()
        Image.new("RGB", (16, 9), color).save(content, format="JPEG")
        return yandex_art.validate_image(content.getvalue())

    def install(self, image=None, receipt="local-reviewed-receipt", version="destination-cover-v5"):
        return self.manager.install_reviewed_image(1, self.account, self.trip,
                                                  image or self.reviewed_image(), receipt, version)

    def test_schema_is_additive_and_idempotent(self):
        original = self.rows("trips")
        covers.init_schema()
        covers.init_schema()
        self.assertEqual(self.rows("trips"), original)
        self.assertEqual(self.rows("trip_covers"), [])

    def test_existing_cover_schema_migrates_visibility_without_data_loss(self):
        self.complete()
        original = self.rows("trip_covers")[0]
        without_hidden = {key: value for key, value in original.items() if key != "hidden"}
        before_image = self.manager.get_image(1, self.account, self.trip).data
        attempts = self.rows("cover_attempts")
        with db.connect() as conn:
            schema = conn.execute("SELECT sql FROM sqlite_master WHERE name='trip_covers'").fetchone()[0]
            schema = schema.replace("hidden INTEGER NOT NULL DEFAULT 0 CHECK(hidden IN (0,1)),", "")
            conn.execute("DROP TABLE trip_covers")
            conn.execute(schema)
            columns = ",".join(without_hidden)
            placeholders = ",".join("?" for _ in without_hidden)
            conn.execute(f"INSERT INTO trip_covers({columns}) VALUES({placeholders})", tuple(without_hidden.values()))
            self.assertNotIn("hidden", {row["name"] for row in conn.execute("PRAGMA table_info(trip_covers)")})
        covers.init_schema()
        covers.init_schema()
        self.assertEqual(self.rows("trip_covers"), [original])
        self.assertEqual(self.manager.get_image(1, self.account, self.trip).data, before_image)
        self.assertEqual(self.rows("cover_attempts"), attempts)

    def test_get_is_readonly_and_does_not_submit_or_reserve(self):
        before = Path(db.DB_PATH).read_bytes()
        for _ in range(3):
            self.assertEqual(self.status()["status"], "empty")
        self.assertEqual(Path(db.DB_PATH).read_bytes(), before)
        self.submission.assert_not_called()
        self.assertEqual(self.rows("cover_attempts"), [])

    def test_disabled_default_and_missing_credentials_do_not_reserve(self):
        for values in ({"IMAGE_GENERATION_ENABLED": "0"}, {"YANDEX_ART_API_KEY": "", "YANDEX_API_KEY": ""},
                       {"YANDEX_FOLDER_ID": ""}):
            with self.subTest(values=values), patch.dict("os.environ", values):
                self.assertEqual(self.submit()["status"], "disabled")
        self.submission.assert_not_called()
        self.assertEqual(self.rows("cover_attempts"), [])

    def test_repeated_and_concurrent_submit_is_one_receipt_one_attempt(self):
        with ThreadPoolExecutor(max_workers=6) as callers:
            results = list(callers.map(lambda _: self.submit(), range(12)))
        self.assertTrue(all(result["status"] == "queued" for result in results))
        self.assertEqual(len(self.manager.executor.calls), 1)
        self.assertEqual(len(self.rows("cover_attempts")), 1)
        self.manager.executor.run_next()
        self.assertEqual(self.submit()["status"], "ready")
        self.submission.assert_called_once()

    def test_cap_is_global_and_capacity_is_two_with_one_worker(self):
        self.submit()
        second = self.save_trip(2)
        self.manager.submit(2, db.get_account_key(2), second)
        third = self.save_trip()
        with self.assertRaises(covers.CoverBusyError):
            self.submit(third)
        self.assertEqual(len(self.rows("cover_attempts")), 2)
        self.assertEqual(self.manager.executor.options["max_workers"], 1)

    def test_daily_limit_includes_failed_attempts_and_never_retries(self):
        with patch.dict("os.environ", {"IMAGE_GENERATION_DAILY_LIMIT": "1"}):
            self.submission.side_effect = yandex_art.YandexArtError("forbidden")
            self.submit()
            self.manager.executor.run_next()
            self.assertEqual(self.submit()["status"], "failed")
            with self.assertRaises(covers.CoverLimitError):
                self.submit(self.save_trip())
        self.submission.assert_called_once()
        self.assertEqual(len(self.rows("cover_attempts")), 1)

    def test_invalid_daily_limit_fails_closed(self):
        for limit in ("0", "-1", "1000", "abc"):
            with self.subTest(limit=limit), patch.dict("os.environ", {"IMAGE_GENERATION_DAILY_LIMIT": limit}):
                with self.assertRaises(covers.CoverLimitError):
                    self.submit()
        self.assertEqual(self.rows("cover_attempts"), [])

    def test_owner_isolation_and_old_account_key(self):
        for owner, account in ((2, db.get_account_key(2)), (1, "stale-account-key")):
            for method in (self.manager.submit, self.manager.get_status, self.manager.get_image):
                with self.subTest(owner=owner, method=method.__name__), self.assertRaises(covers.CoverNotFoundError):
                    method(owner, account, self.trip)
        self.assertEqual(self.rows("cover_attempts"), [])

    def test_deletion_lease_before_worker_prevents_paid_submit(self):
        self.submit()
        with db.connect() as conn:
            conn.execute("UPDATE users SET deletion_token='synthetic-lease' WHERE tg_id=1")
        self.manager.executor.run_next()
        self.submission.assert_not_called()
        self.assertEqual(self.rows("trip_covers")[0]["status"], "interrupted")

    def test_destination_country_prompt_excludes_private_trip_fields(self):
        self.complete()
        prompt = self.submission.call_args.args[0]
        self.assertIn("Москва", prompt.user_prompt)
        self.assertIn("Россия", prompt.user_prompt)
        self.assertNotIn("private-family-information", repr(prompt))
        metadata = json.dumps(self.rows("trip_covers"), ensure_ascii=False)
        self.assertNotIn("Москва", metadata)
        self.assertNotIn("private", metadata)
        self.assertNotIn("synthetic-art-key", metadata)

    def test_unknown_country_rejected_before_reservation(self):
        with self.assertRaises(covers.CoverDestinationError):
            self.submit(self.save_trip(destination="Неизвестное направление"))
        self.assertEqual(self.rows("cover_attempts"), [])
        self.submission.assert_not_called()

    def test_local_receipt_is_persisted_before_disk_save_without_poll(self):
        original = self.manager._save_image
        def save(*args):
            self.assertEqual(self.rows("trip_covers")[0]["operation_id"], "synthetic-operation")
            return original(*args)
        with patch.object(self.manager, "_save_image", side_effect=save):
            self.assertEqual(self.complete()["status"], "ready")
        self.submission.assert_called_once()
        self.poll.assert_not_called()

    def test_paid_timeout_is_interrupted_and_replays_never_resubmit(self):
        self.submission.side_effect = yandex_art.YandexArtError("network", submission_uncertain=True)
        self.complete()
        self.assertEqual(self.submit()["status"], "interrupted")
        self.submission.assert_called_once()
        self.poll.assert_not_called()

    def test_legacy_pending_result_is_interrupted_without_poll_or_retry(self):
        self.submission.return_value = yandex_art.ArtOperation("synthetic-operation", False)
        self.complete()
        self.assertEqual(self.submit()["status"], "interrupted")
        self.assertEqual(self.rows("trip_covers")[0]["operation_id"], "synthetic-operation")
        self.submission.assert_called_once()
        self.poll.assert_not_called()

    def test_image_missing_from_complete_response_is_interrupted(self):
        self.submission.return_value = yandex_art.ArtOperation("synthetic-operation", True)
        self.complete()
        self.assertEqual(self.status()["status"], "interrupted")
        self.poll.assert_not_called()
        self.submission.assert_called_once()

    def test_cache_path_resolves_runtime_directory_not_source_directory(self):
        manager = covers.TripCovers(executor_factory=DeferredExecutor)
        self.addCleanup(manager.close)
        selected = self.directory / "persistent-image-cache"
        with patch.dict("os.environ", {"IMAGE_CACHE_DIR": str(selected)}):
            manager.start()
        self.assertEqual(manager.cache_dir, selected.resolve())

    def test_default_cache_path_is_relative_to_runtime_directory(self):
        manager = covers.TripCovers(executor_factory=DeferredExecutor)
        self.addCleanup(manager.close)
        with patch.dict("os.environ", {"IMAGE_CACHE_DIR": ""}):
            manager.start()
        self.assertEqual(manager.cache_dir, Path("output/covers").resolve())

    def test_ready_image_is_valid_and_private_path_never_in_status(self):
        result = self.complete()
        self.assertEqual(result["image_url"], f"/api/trips/{self.trip}/cover/image")
        self.assertNotIn("operation", json.dumps(result))
        self.assertNotIn(str(self.directory), json.dumps(result))
        retrieved = self.manager.get_image(1, self.account, self.trip)
        self.assertEqual(retrieved.mime_type, "image/jpeg")
        self.assertEqual(retrieved.data, self.image.data)
        self.assertEqual(list(self.manager.cache_dir.glob("*.tmp")), [])

    def test_corrupt_provider_image_is_not_saved(self):
        invalid = yandex_art.GeneratedImage(b"not-an-image", "image/jpeg", 16, 9)
        self.submission.return_value = yandex_art.ArtOperation("synthetic-operation", True, invalid)
        self.assertEqual(self.complete()["status"], "failed")
        self.assertFalse(self.manager.cache_dir.exists())

    def test_missing_ready_image_does_not_trigger_regeneration(self):
        self.complete()
        (self.manager.cache_dir / self.rows("trip_covers")[0]["file_name"]).unlink()
        self.assertEqual(self.submit()["status"], "failed")
        with self.assertRaises(covers.CoverNotFoundError):
            self.manager.get_image(1, self.account, self.trip)
        self.submission.assert_called_once()

    def test_invalid_stored_path_cannot_serve_arbitrary_file(self):
        self.complete()
        with db.connect() as conn:
            conn.execute("UPDATE trip_covers SET file_name='../synthetic.db'")
        with self.assertRaises(covers.CoverNotFoundError):
            self.manager.get_image(1, self.account, self.trip)
        self.assertEqual(self.status()["status"], "failed")

    def test_restart_marks_pending_interrupted_without_resubmission(self):
        self.submit()
        self.manager.close()
        self.manager.start()
        self.assertEqual(self.submit()["status"], "interrupted")
        self.submission.assert_not_called()

    def test_shutdown_does_not_start_queued_paid_work(self):
        self.submit()
        self.manager.closing = True
        self.manager.executor.run_next()
        self.manager.closing = False
        self.assertEqual(self.status()["status"], "interrupted")
        self.submission.assert_not_called()

    def test_ready_survives_restart_without_resubmission(self):
        self.complete()
        self.manager.close()
        self.manager.start()
        self.assertEqual(self.submit()["status"], "ready")
        self.submission.assert_called_once()

    def test_hide_and_restore_only_change_visibility_not_paid_receipt_or_file(self):
        self.complete()
        original = self.rows("trip_covers")[0]
        attempts = self.rows("cover_attempts")
        trip_before = self.rows("trips")
        file_name = original["file_name"]
        before_image = (self.manager.cache_dir / file_name).read_bytes()
        result = self.manager.set_hidden(1, self.account, self.trip, True)
        self.assertEqual(result["status"], "ready")
        self.assertIs(result["hidden"], True)
        self.assertIsNone(result["image_url"])
        self.assertEqual(self.rows("trip_covers"), [{**original, "hidden": 1}])
        self.assertEqual(self.status(), result)
        self.assertEqual(self.submit(), result)
        # A hidden cover can still be read by its owner, never by other accounts.
        self.assertEqual(self.manager.get_image(1, self.account, self.trip).data, before_image)
        self.assertEqual(self.manager.set_hidden(1, self.account, self.trip, True), result)
        restored = self.manager.set_hidden(1, self.account, self.trip, False)
        self.assertIs(restored["hidden"], False)
        self.assertEqual(restored["image_url"], f"/api/trips/{self.trip}/cover/image")
        self.assertEqual(self.rows("trip_covers"), [original])
        self.assertEqual((self.manager.cache_dir / file_name).read_bytes(), before_image)
        self.assertEqual(self.rows("cover_attempts"), attempts)
        self.assertEqual(self.rows("trips"), trip_before)
        self.submission.assert_called_once()

    def test_hidden_preference_survives_restart(self):
        self.complete()
        self.manager.set_hidden(1, self.account, self.trip, True)
        self.manager.close()
        self.manager.start()
        self.assertIs(self.status()["hidden"], True)
        self.assertEqual(self.status()["status"], "ready")
        self.assertIsNone(self.status()["image_url"])
        self.assertEqual(self.submit(), self.status())
        self.submission.assert_called_once()

    def test_visibility_change_requires_ready_cover_and_real_boolean(self):
        with self.assertRaises(covers.CoverStateError):
            self.manager.set_hidden(1, self.account, self.trip, True)
        self.submit()
        with self.assertRaises(covers.CoverStateError):
            self.manager.set_hidden(1, self.account, self.trip, True)
        for value in ("true", 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.manager.set_hidden(1, self.account, self.trip, value)
        self.assertIs(self.status()["hidden"], False)
        self.submission.assert_not_called()

    def test_visibility_change_checks_owner_and_deletion_lease(self):
        self.complete()
        for owner, account in ((2, db.get_account_key(2)), (1, "stale-account-key")):
            with self.subTest(owner=owner), self.assertRaises(covers.CoverNotFoundError):
                self.manager.set_hidden(owner, account, self.trip, True)
        with db.connect() as conn:
            conn.execute("UPDATE users SET deletion_token='synthetic-lease' WHERE tg_id=1")
        with self.assertRaises(covers.CoverNotFoundError):
            self.manager.set_hidden(1, self.account, self.trip, True)
        self.assertEqual(self.rows("trip_covers")[0]["hidden"], 0)

    def test_trip_deletion_cascades_metadata_but_not_anonymous_budget(self):
        self.complete()
        with db.connect() as conn:
            conn.execute("DELETE FROM trips WHERE id=?", (self.trip,))
        self.assertEqual(self.rows("trip_covers"), [])
        self.assertEqual(len(self.rows("cover_attempts")), 1)
        with self.assertRaises(covers.CoverNotFoundError):
            self.manager.get_image(1, self.account, self.trip)

    def test_account_change_after_provider_completion_cannot_write_image(self):
        def provider(prompt):
            with db.connect() as conn:
                conn.execute("UPDATE users SET deletion_token='synthetic-lease' WHERE tg_id=1")
            return yandex_art.ArtOperation("synthetic-operation", True, self.image)
        self.submission.side_effect = provider
        self.submit()
        self.manager.executor.run_next()
        self.assertFalse(self.manager.cache_dir.exists())
        self.assertEqual(self.rows("trip_covers")[0]["status"], "interrupted")

    def test_dispatch_failure_has_receipt_and_does_not_retry(self):
        with patch.object(self.manager.executor, "submit", side_effect=RuntimeError("private")):
            self.assertEqual(self.submit()["status"], "interrupted")
        self.assertEqual(self.submit()["status"], "interrupted")
        self.submission.assert_not_called()

    def test_http_auth_csrf_owner_isolation_and_private_image_headers(self):
        with patch.object(covers, "manager", self.manager), TestClient(web_app.app) as client:
            url = f"/api/trips/{self.trip}/cover"
            self.assertEqual(client.get(url).status_code, 401)
            self.assertEqual(client.post(url).status_code, 401)
            session, csrf, _ = web_auth.create_session(1)
            client.cookies.set(web_auth.SESSION_COOKIE, session)
            client.cookies.set(web_auth.CSRF_COOKIE, csrf)
            self.assertEqual(client.get(url).json()["status"], "empty")
            self.assertEqual(client.post(url).status_code, 403)
            self.submission.assert_not_called()
            headers = {"X-CSRF-Token": csrf}
            self.assertEqual(client.post(url, headers=headers).status_code, 202)
            self.manager.executor.run_next()
            self.assertEqual(client.post(url, headers=headers).json()["status"], "ready")
            result = client.get(url + "/image")
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.headers["content-type"], "image/jpeg")
            self.assertIn("private", result.headers["cache-control"])
            self.assertIn("no-store", result.headers["cache-control"])
            self.assertEqual(result.headers["x-content-type-options"], "nosniff")
            session2, csrf2, _ = web_auth.create_session(2)
            client.cookies.set(web_auth.SESSION_COOKIE, session2)
            client.cookies.set(web_auth.CSRF_COOKIE, csrf2)
            self.assertEqual(client.get(url).status_code, 404)
            self.assertEqual(client.get(url + "/image").status_code, 404)
            self.assertEqual(client.post(url, headers={"X-CSRF-Token": csrf2}).status_code, 404)
        self.submission.assert_called_once()

    def test_http_visibility_patch_auth_csrf_strict_boolean_and_restoration(self):
        with patch.object(covers, "manager", self.manager), TestClient(web_app.app) as client:
            url = f"/api/trips/{self.trip}/cover"
            self.assertEqual(client.patch(url, json={"hidden": True}).status_code, 401)
            session, csrf, _ = web_auth.create_session(1)
            client.cookies.set(web_auth.SESSION_COOKIE, session)
            client.cookies.set(web_auth.CSRF_COOKIE, csrf)
            headers = {"X-CSRF-Token": csrf}
            self.assertEqual(client.patch(url, json={"hidden": True}).status_code, 403)
            self.assertEqual(client.patch(url, json={"hidden": True}, headers=headers).status_code, 409)
            self.complete()
            for payload in ({"hidden": "true"}, {"hidden": 1}, {"hidden": None}, {}, {"hidden": True, "retry": True}):
                with self.subTest(payload=payload):
                    self.assertEqual(client.patch(url, json=payload, headers=headers).status_code, 422)
            hidden = client.patch(url, json={"hidden": True}, headers=headers)
            self.assertEqual(hidden.status_code, 200)
            self.assertIs(hidden.json()["hidden"], True)
            self.assertEqual(hidden.json()["status"], "ready")
            self.assertIsNone(hidden.json()["image_url"])
            self.assertEqual(client.get(url).json(), hidden.json())
            self.assertEqual(client.get(url + "/image").status_code, 200)
            session2, csrf2, _ = web_auth.create_session(2)
            client.cookies.set(web_auth.SESSION_COOKIE, session2)
            client.cookies.set(web_auth.CSRF_COOKIE, csrf2)
            self.assertEqual(client.patch(url, json={"hidden": False}, headers={"X-CSRF-Token": csrf2}).status_code, 404)
            self.assertEqual(client.get(url + "/image").status_code, 404)
            client.cookies.set(web_auth.SESSION_COOKIE, session)
            client.cookies.set(web_auth.CSRF_COOKIE, csrf)
            restored = client.patch(url, json={"hidden": False}, headers=headers)
            self.assertEqual(restored.status_code, 200)
            self.assertIs(restored.json()["hidden"], False)
            self.assertEqual(restored.json()["image_url"], url + "/image")
            self.assertIn("no-store", restored.headers["cache-control"])
        self.submission.assert_called_once()
        self.assertEqual(len(self.rows("cover_attempts")), 1)

    def test_reviewed_override_preserves_original_receipt_files_and_quota(self):
        self.complete()
        self.manager.set_hidden(1, self.account, self.trip, True)
        original = self.rows("trip_covers")[0]
        attempts = self.rows("cover_attempts")
        trips = self.rows("trips")
        image = self.reviewed_image()
        result = self.install(image)
        self.assertEqual(result["status"], "ready")
        self.assertIs(result["hidden"], False)
        self.assertEqual(self.rows("trip_covers"), [{**original, "hidden": 0}])
        override = self.rows("trip_cover_overrides")[0]
        self.assertEqual(override["source_receipt"], "local-reviewed-receipt")
        self.assertEqual(override["prompt_version"], "destination-cover-v5")
        self.assertNotEqual(override["file_name"], original["file_name"])
        self.assertEqual((self.manager.cache_dir / original["file_name"]).read_bytes(), self.image.data)
        self.assertEqual((self.manager.cache_dir / override["file_name"]).read_bytes(), image.data)
        self.assertEqual(self.manager.get_image(1, self.account, self.trip).data, image.data)
        self.assertEqual(self.rows("cover_attempts"), attempts)
        self.assertEqual(self.rows("trips"), trips)
        self.assertEqual(list(self.manager.cache_dir.glob("*.tmp")), [])
        self.submission.assert_called_once()
        self.poll.assert_not_called()

    def test_reviewed_override_install_without_worker_or_generation_enabled(self):
        self.complete()
        local = covers.TripCovers(cache_dir=self.manager.cache_dir)
        self.assertIsNone(local.executor)
        image = self.reviewed_image()
        with patch.dict("os.environ", {"IMAGE_GENERATION_ENABLED": "0"}), \
                patch.object(yandex_art, "submit", side_effect=AssertionError("No paid call allowed")):
            result = local.install_reviewed_image(1, self.account, self.trip, image,
                                                  "local-reviewed-receipt", "destination-cover-v5")
        self.assertEqual(result["status"], "ready")
        self.assertEqual(local.get_image(1, self.account, self.trip).data, image.data)
        self.assertIsNone(local.executor)
        self.assertEqual(len(self.rows("cover_attempts")), 1)

    def test_reviewed_override_replay_is_noop_and_preserves_later_hidden_choice(self):
        self.complete()
        self.install()
        self.manager.set_hidden(1, self.account, self.trip, True)
        rows = self.rows("trip_cover_overrides")
        original = self.rows("trip_covers")
        files = sorted(path.name for path in self.manager.cache_dir.iterdir())
        result = self.install()
        self.assertIs(result["hidden"], True)
        self.assertEqual(self.rows("trip_cover_overrides"), rows)
        self.assertEqual(self.rows("trip_covers"), original)
        self.assertEqual(sorted(path.name for path in self.manager.cache_dir.iterdir()), files)
        self.submission.assert_called_once()

    def test_reviewed_override_rejects_different_receipt_version_or_bytes(self):
        self.complete()
        self.install()
        rows = self.rows("trip_cover_overrides")
        files = sorted(path.name for path in self.manager.cache_dir.iterdir())
        for kwargs in ({"receipt": "local-other-receipt"}, {"version": "destination-cover-v6"},
                       {"image": self.reviewed_image("red")}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(covers.CoverStateError):
                self.install(**kwargs)
        self.assertEqual(self.rows("trip_cover_overrides"), rows)
        self.assertEqual(sorted(path.name for path in self.manager.cache_dir.iterdir()), files)
        self.submission.assert_called_once()

    def test_reviewed_override_owner_account_and_deletion_lease_checks(self):
        self.complete()
        for owner, account in ((2, db.get_account_key(2)), (1, "stale-account-key")):
            with self.subTest(owner=owner), self.assertRaises(covers.CoverNotFoundError):
                self.manager.install_reviewed_image(owner, account, self.trip, self.reviewed_image(),
                                                    "local-reviewed", "destination-cover-v5")
        with db.connect() as conn:
            conn.execute("UPDATE users SET deletion_token='synthetic-lease' WHERE tg_id=1")
        with self.assertRaises(covers.CoverNotFoundError):
            self.install()
        self.assertEqual(self.rows("trip_cover_overrides"), [])
        self.assertEqual(len(list(self.manager.cache_dir.iterdir())), 1)

    def test_reviewed_override_requires_ready_original(self):
        with self.assertRaises(covers.CoverStateError):
            self.install()
        self.submit()
        with self.assertRaises(covers.CoverStateError):
            self.install()
        self.assertEqual(self.rows("trip_cover_overrides"), [])
        self.assertFalse(self.manager.cache_dir.exists())

    def test_reviewed_override_validates_image_and_bounded_receipt_metadata(self):
        self.complete()
        invalid = yandex_art.GeneratedImage(b"not-an-image", "image/jpeg", 16, 9)
        with self.assertRaises(yandex_art.YandexArtError):
            self.install(invalid)
        for kwargs in ({"receipt": "../secret"}, {"receipt": "x" * 129}, {"version": "secret\nvalue"}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                self.install(**kwargs)
        self.assertEqual(self.rows("trip_cover_overrides"), [])
        self.assertEqual(len(list(self.manager.cache_dir.iterdir())), 1)

    def test_reviewed_override_survives_restart_and_hiding_does_not_regenerate(self):
        self.complete()
        image = self.reviewed_image()
        self.install(image)
        self.manager.close()
        self.manager.start()
        self.assertEqual(self.manager.get_image(1, self.account, self.trip).data, image.data)
        self.assertIsNone(self.manager.set_hidden(1, self.account, self.trip, True)["image_url"])
        restored = self.manager.set_hidden(1, self.account, self.trip, False)
        self.assertEqual(restored["image_url"], f"/api/trips/{self.trip}/cover/image")
        self.assertEqual(self.submit(), restored)
        self.submission.assert_called_once()

    def test_reviewed_override_corruption_or_missing_file_never_resurfaces_rejected_original(self):
        self.complete()
        self.install()
        original = self.rows("trip_covers")[0]
        override = self.rows("trip_cover_overrides")[0]
        path = self.manager.cache_dir / override["file_name"]
        for content in (b"corrupt", self.reviewed_image("pink").data):
            path.write_bytes(content)
            with self.assertRaises(covers.CoverNotFoundError):
                self.manager.get_image(1, self.account, self.trip)
            self.assertEqual(self.status()["status"], "failed")
            self.assertIsNone(self.status()["image_url"])
        path.unlink()
        with self.assertRaises(covers.CoverNotFoundError):
            self.manager.get_image(1, self.account, self.trip)
        self.assertEqual(self.status()["status"], "failed")
        self.assertIsNone(self.status()["image_url"])
        self.assertEqual(self.rows("trip_covers"), [original])
        self.assertEqual((self.manager.cache_dir / original["file_name"]).read_bytes(), self.image.data)
        self.assertEqual(self.rows("trip_cover_overrides"), [override])
        self.submission.assert_called_once()

    def test_reviewed_override_can_serve_when_original_file_is_missing(self):
        self.complete()
        self.install()
        original = self.rows("trip_covers")[0]
        (self.manager.cache_dir / original["file_name"]).unlink()
        self.assertEqual(self.status()["status"], "ready")
        self.assertEqual(self.manager.get_image(1, self.account, self.trip).data, self.reviewed_image().data)

    def test_status_does_not_offer_corrupt_override_when_original_is_missing(self):
        self.complete()
        self.install()
        original = self.rows("trip_covers")[0]
        override = self.rows("trip_cover_overrides")[0]
        (self.manager.cache_dir / original["file_name"]).unlink()
        (self.manager.cache_dir / override["file_name"]).write_bytes(b"corrupt")
        self.assertEqual(self.status()["status"], "failed")
        with self.assertRaises(covers.CoverNotFoundError):
            self.manager.get_image(1, self.account, self.trip)
        self.submission.assert_called_once()

    def test_reviewed_override_migration_preserves_existing_receipt(self):
        self.complete()
        original = self.rows("trip_covers")
        with db.connect() as conn:
            conn.execute("DROP TABLE trip_cover_overrides")
        covers.init_schema()
        covers.init_schema()
        self.assertEqual(self.rows("trip_covers"), original)
        self.assertEqual(self.rows("trip_cover_overrides"), [])

    def test_reviewed_override_insert_failure_leaves_original_and_no_orphan_file(self):
        self.complete()
        original = self.rows("trip_covers")
        files = sorted(path.name for path in self.manager.cache_dir.iterdir())
        with db.connect() as conn:
            conn.execute("CREATE TRIGGER reject_override BEFORE INSERT ON trip_cover_overrides "
                         "BEGIN SELECT RAISE(ABORT, 'Synthetic failure'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.install()
        self.assertEqual(self.rows("trip_covers"), original)
        self.assertEqual(self.rows("trip_cover_overrides"), [])
        self.assertEqual(sorted(path.name for path in self.manager.cache_dir.iterdir()), files)

    def test_reviewed_override_uuid_collision_cannot_delete_original_file(self):
        self.complete()
        original = self.rows("trip_covers")[0]
        existing_uuid = original["file_name"].split(".")[0]
        with patch.object(covers, "uuid4", side_effect=[SimpleNamespace(hex=existing_uuid), SimpleNamespace(hex="f" * 32)]):
            with self.assertRaises(FileExistsError):
                self.install()
        self.assertEqual((self.manager.cache_dir / original["file_name"]).read_bytes(), self.image.data)
        self.assertEqual(self.rows("trip_cover_overrides"), [])
        self.assertEqual(len(list(self.manager.cache_dir.iterdir())), 1)

    def test_reviewed_override_concurrent_replay_saves_exactly_one_new_file(self):
        self.complete()
        image = self.reviewed_image()
        with ThreadPoolExecutor(max_workers=4) as callers:
            results = list(callers.map(lambda _: self.install(image), range(4)))
        self.assertTrue(all(result["status"] == "ready" for result in results))
        self.assertEqual(len(self.rows("trip_cover_overrides")), 1)
        self.assertEqual(len(list(self.manager.cache_dir.iterdir())), 2)
        self.assertEqual(len(self.rows("cover_attempts")), 1)
        self.submission.assert_called_once()


if __name__ == "__main__":
    unittest.main()
