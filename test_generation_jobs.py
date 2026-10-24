"""Durable receipts with disposable SQLite and controlled workers; no AI/HTTP."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import ast
from datetime import datetime, timedelta
import inspect
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch
from uuid import uuid4

with patch("dotenv.load_dotenv", return_value=False):
    import db
    import generation_jobs as jobs
    import web_app
    import web_auth
    import web_service
from fastapi.testclient import TestClient
from schemas import Itinerary, ItineraryDay, TripRequest
from weather import CityResolutionError

REAL_CREATE_TRIP = web_service.create_trip
REAL_CLAIM = db.claim_ai_request


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
        while self.calls:
            self.run_next()


class GenerationJobsTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="generation-jobs-"))
        self.stack.enter_context(patch.object(db, "DB_PATH", str(Path(directory) / "synthetic.db")))
        self.stack.enter_context(patch.dict("os.environ", {"WEB_SESSION_SECRET": "synthetic-secret-for-offline-generation-tests-12345"}))
        db.init_db()
        for owner in (1, 2, 3):
            db.save_user(owner, "Synthetic", 2, [], "RUB")
        self.account = db.get_account_key(1)
        self.key = str(uuid4())
        self.request = TripRequest(destination="Синтетический город, Россия", start_date="2099-06-01",
                                   end_date="2099-06-01", budget=20000, currency="RUB",
                                   trip_goal="Познакомиться с городом", interest_categories=["main_sights"],
                                   wishes="private-secret-like-questionnaire")
        self.itinerary = Itinerary(title="Synthetic", summary="", days=[
            ItineraryDay(day=1, date=self.request.start_date, theme="Synthetic", items=[])
        ])
        self.worker = self.stack.enter_context(patch.object(web_service, "create_trip", side_effect=AssertionError("Unexpected generation")))
        self.claim = self.stack.enter_context(patch.object(db, "claim_ai_request", side_effect=AssertionError("Unexpected quota claim")))
        self.manager = jobs.GenerationJobs(executor_factory=DeferredExecutor)
        self.manager.start()
        self.addCleanup(self.manager.close)

    def submit(self, owner=1, key=None, request=None):
        return self.manager.submit(owner, db.get_account_key(owner), key or self.key, request or self.request)

    def save(self, owner, request, *, expected_account_key, generation_job_id):
        return db.save_trip_bundle(owner, request.model_dump(), self.itinerary.model_dump(), [("General", "Synthetic")],
                                   generation_job_id=generation_job_id, expected_account_key=expected_account_key)

    def rows(self, table):
        with db.connect() as connection:
            return [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1")]

    def finish_saved(self):
        self.worker.side_effect = self.save
        status = self.submit()
        self.manager.executor.run_next()
        return jobs.get_job(1, self.account, job_id=status["job_id"])

    @staticmethod
    def bot_quota_namespace():
        """Execute only the quota default/helpers, never bot startup or Telegram."""
        source = ast.parse(Path(__file__).with_name("bot.py").read_text(encoding="utf-8"))
        selected = [node for node in source.body if (
            isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "DAILY_LIMIT"
                                                 for target in node.targets)
        ) or (isinstance(node, ast.FunctionDef) and node.name in {"_claim_route_ai_request", "_ai_claim_error_text"})]
        namespace = {"os": os, "db": db, "datetime": datetime, "Optional": Optional,
                     "reminders": web_service.reminders}
        exec(compile(ast.Module(body=selected, type_ignores=[]), "<offline-bot-quota>", "exec"), namespace)
        return namespace

    def test_route_defaults_are_three_for_bot_web_and_database(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.bot_quota_namespace()["DAILY_LIMIT"], 3)
            self.assertEqual(web_service._daily_ai_limit(), 3)
        self.assertEqual(inspect.signature(REAL_CLAIM).parameters["daily_limit"].default, 3)

    def test_three_route_operations_are_shared_across_channels_and_reset_by_day_not_user(self):
        self.claim.side_effect = REAL_CLAIM
        with patch.dict(os.environ, {}, clear=True):
            namespace = self.bot_quota_namespace()
            today = datetime.now(web_service.reminders.get_tz()).date()
            for index in range(3):
                claim = (namespace["_claim_route_ai_request"](1, "plan") if index != 1 else
                         REAL_CLAIM(1, today, "edit", "web", daily_limit=web_service._daily_ai_limit()))
                self.assertEqual(claim.status, "ok")
                self.assertEqual(claim.limit, 3)
                self.assertEqual(claim.used, index + 1)
                # An unsuccessful paid attempt still occupies its slot.
                self.assertTrue(db.finish_ai_request(claim.claim_id, succeeded=index != 1))
            limited_bot = namespace["_claim_route_ai_request"](1, "plan")
            limited_web = REAL_CLAIM(1, today, "plan", "web", daily_limit=web_service._daily_ai_limit())
            self.assertEqual((limited_bot.status, limited_web.status), ("limit", "limit"))
            self.assertIn("лимит 3", namespace["_ai_claim_error_text"](limited_bot))
            self.assertEqual(db.get_ai_request_usage(1, today), 3)
            other = REAL_CLAIM(2, today, "plan", "web")
            self.assertEqual((other.status, other.used), ("ok", 1))
            self.assertTrue(db.finish_ai_request(other.claim_id, succeeded=True))
            tomorrow = REAL_CLAIM(1, today + timedelta(days=1), "plan", "web")
            self.assertEqual((tomorrow.status, tomorrow.used), ("ok", 1))
            self.assertTrue(db.finish_ai_request(tomorrow.claim_id, succeeded=True))
        self.worker.assert_not_called()

    def test_lowering_route_limit_preserves_existing_usage_and_blocks_next_attempt(self):
        day = "2099-06-01"
        for _ in range(4):
            claim = REAL_CLAIM(1, day, "plan", "web", daily_limit=50)
            self.assertEqual(claim.status, "ok")
            self.assertTrue(db.finish_ai_request(claim.claim_id, succeeded=True))
        before = self.rows("ai_requests")
        limited = REAL_CLAIM(1, day, "plan", "telegram", daily_limit=3)
        self.assertEqual((limited.status, limited.used, limited.limit), ("limit", 4, 3))
        self.assertEqual(self.rows("ai_requests"), before)

    def test_saved_route_reads_and_pdf_do_not_consume_ai_quota(self):
        trip_id = db.save_trip(1, self.request.model_dump(), self.itinerary.model_dump())
        before = self.rows("ai_requests")
        self.assertEqual(web_service.list_user_trips(1)[0]["id"], trip_id)
        self.assertEqual(web_service.get_trip_detail(1, trip_id)["id"], trip_id)
        with patch.object(web_service, "build_trip_pdf", return_value=b"%PDF-synthetic"):
            self.assertEqual(web_service.get_trip_pdf(1, trip_id), b"%PDF-synthetic")
        self.assertEqual(self.rows("ai_requests"), before)
        self.claim.assert_not_called()
        self.worker.assert_not_called()

    def test_additive_migration_is_idempotent_and_preserves_existing_rows(self):
        trip_id = db.save_trip(1, self.request.model_dump(), self.itinerary.model_dump())
        with db.connect() as connection:
            connection.execute("DROP TABLE generation_jobs")
        before = self.rows("trips")
        db.init_db()
        db.init_db()
        self.assertEqual(self.rows("trips"), before)
        self.assertEqual(before[0]["id"], trip_id)
        self.assertEqual(self.rows("generation_jobs"), [])

    def test_destination_options_migration_preserves_legacy_receipt_and_trip(self):
        status = self.finish_saved()
        before_trips = self.rows("trips")
        before_checklist = self.rows("checklist_items")
        with db.connect() as connection:
            # Recreate the immediately previous schema, not an empty database.
            connection.execute("ALTER TABLE generation_jobs DROP COLUMN destination_options_json")
        legacy = self.rows("generation_jobs")
        db.init_db()
        db.init_db()
        after = self.rows("generation_jobs")
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0].pop("destination_options_json"), "[]")
        self.assertEqual(after, legacy)
        self.assertEqual(self.rows("trips"), before_trips)
        self.assertEqual(self.rows("checklist_items"), before_checklist)
        self.assertEqual(jobs.get_job(1, self.account, job_id=status["job_id"]), status)

    def test_concurrent_same_key_claims_exactly_once_without_quota_before_worker(self):
        with ThreadPoolExecutor(max_workers=8) as callers:
            results = list(callers.map(lambda _: self.submit(), range(12)))
        self.assertEqual(len({result["job_id"] for result in results}), 1)
        self.assertEqual(len(self.manager.executor.calls), 1)
        self.assertEqual(len(self.rows("generation_jobs")), 1)
        self.worker.assert_not_called()
        self.claim.assert_not_called()

    def test_changed_payload_same_key_conflicts(self):
        self.submit()
        with self.assertRaises(jobs.JobConflictError):
            self.submit(request=self.request.model_copy(update={"budget": 21000}))
        self.assertEqual(len(self.manager.executor.calls), 1)

    def test_capacity_and_one_active_per_account_reject_before_receipt_and_quota(self):
        self.submit()
        with self.assertRaises(jobs.JobBusyError):
            self.submit(key=str(uuid4()))
        self.submit(owner=2)
        with self.assertRaises(jobs.JobBusyError):
            self.submit(owner=3)
        self.assertEqual(len(self.rows("generation_jobs")), 2)
        self.assertEqual(len(self.manager.executor.calls), 2)
        self.claim.assert_not_called()
        self.worker.assert_not_called()

    def test_request_and_private_fields_never_persist_in_receipt_or_status(self):
        status = self.submit()
        self.assertEqual(set(status), {"job_id", "status", "trip_id", "message"})
        raw = json.dumps(self.rows("generation_jobs"), ensure_ascii=False)
        self.assertNotIn(self.request.wishes, raw)
        self.assertNotIn(self.request.destination, raw)
        self.assertEqual(set(self.rows("generation_jobs")[0]), {"job_id", "tg_id", "account_key", "idempotency_key",
                         "request_digest", "status", "trip_id", "failure_code", "destination_options_json",
                         "created_at", "finished_at"})

    def test_success_trip_and_receipt_commit_together_and_replay_never_calls_worker(self):
        status = self.finish_saved()
        self.assertEqual(status["status"], "succeeded")
        self.assertEqual(status["trip_id"], self.rows("trips")[0]["id"])
        self.assertEqual(len(self.rows("checklist_items")), 1)
        self.assertEqual(self.submit(), status)
        self.worker.assert_called_once()
        with self.assertRaises(web_service.TripNotFoundError):
            web_service.get_trip_detail(2, status["trip_id"])
        self.assertEqual(web_service.get_trip_detail(1, status["trip_id"])["id"], status["trip_id"])

    def test_partial_saved_result_is_not_announced_as_complete_and_replay_agrees(self):
        def save_partial(owner, request, *, expected_account_key, generation_job_id):
            envelope = request.model_dump()
            envelope["generation_coverage"] = {"quality_status": "partial", "raw": "private-secret"}
            return db.save_trip_bundle(owner, envelope, self.itinerary.model_dump(), [("General", "Synthetic")],
                                      generation_job_id=generation_job_id, expected_account_key=expected_account_key)
        self.worker.side_effect = save_partial
        receipt = self.submit()
        self.manager.executor.run_next()
        status = jobs.get_job(1, self.account, job_id=receipt["job_id"])
        self.assertEqual(status["status"], "succeeded")
        self.assertIn("частично", status["message"])
        self.assertNotIn("private-secret", json.dumps(status))
        self.assertEqual(self.submit(), status)

    def test_lost_response_after_commit_keeps_success_not_exception_id(self):
        def saved_then_error(*args, **kwargs):
            self.save(*args, **kwargs)
            raise web_service.TripSavedResponseError(999999, "secret-model-output")
        self.worker.side_effect = saved_then_error
        self.submit()
        self.manager.executor.run_next()
        status = jobs.get_job(1, self.account, key=self.key)
        self.assertEqual(status["status"], "succeeded")
        self.assertEqual(status["trip_id"], self.rows("trips")[0]["id"])
        self.assertNotIn("secret", json.dumps(status))

    def test_returned_success_without_atomic_receipt_is_not_accepted(self):
        self.worker.side_effect = None
        self.worker.return_value = {"id": 999}
        self.submit()
        self.manager.executor.run_next()
        self.assertEqual(jobs.get_job(1, self.account, key=self.key)["status"], "failed")

    def test_failure_messages_are_allowlisted_including_insufficient_data(self):
        for code in ("insufficient_data", "search_limit", "invalid_request", "validation_error", "unsupported_preferences",
                     "city_ambiguous", "city_not_found", "city_unavailable", "city_invalid", "private-secret"):
            with self.subTest(code=code):
                key = str(uuid4())
                self.worker.side_effect = web_service.TripCreateValidationError("secret-raw", code=code,
                                                                            diagnostic_problems=["secret-prompt"])
                self.submit(key=key)
                self.manager.executor.run_next()
                status = jobs.get_job(1, self.account, key=key)
                self.assertEqual(status["status"], "failed")
                self.assertEqual(status["message"], jobs.MESSAGES[code if code in jobs.MESSAGES else "validation"])
                self.assertNotIn("secret", json.dumps(status))

    def test_actual_city_preflight_failure_persists_options_without_quota_or_memory(self):
        options = ("Москва, Москва, Россия", "Москва, Тверская область, Россия")
        self.worker.side_effect = REAL_CREATE_TRIP
        with patch.object(web_service, "resolve_city_for_planning", side_effect=CityResolutionError(
                "private-destination", reason="ambiguous", suggestions=options)) as resolve, \
                patch.object(web_service.vector_memory, "recall_family_memories",
                             side_effect=AssertionError("Memory must follow city preflight")) as recall, \
                patch.object(web_service, "generate_itinerary",
                             side_effect=AssertionError("Generation must follow city preflight")) as model:
            queued = self.submit()
            resolve.assert_not_called()  # No external lookup in queue transaction.
            self.manager.executor.run_next()
            status = jobs.get_job(1, self.account, job_id=queued["job_id"])
            self.assertEqual(status["status"], "failed")
            self.assertEqual(status["clarification"], {"kind": "city", "options": list(options), "attempt_used": False})
            self.assertEqual(status["message"], jobs.MESSAGES["city_ambiguous"])
            self.assertEqual(self.submit(), status)
            self.assertEqual(self.manager.executor.calls, [])
            resolve.assert_called_once_with(self.request.destination)
            recall.assert_not_called()
            model.assert_not_called()
        self.claim.assert_not_called()
        self.worker.assert_called_once()
        self.assertEqual(self.rows("ai_requests"), [])
        self.assertEqual(self.rows("access_route_attempts"), [])
        self.assertEqual(self.rows("trips"), [])
        row = self.rows("generation_jobs")[0]
        self.assertEqual(json.loads(row["destination_options_json"]), list(options))
        self.assertNotIn("private-destination", json.dumps(row))

    def test_clarification_is_sanitized_owner_bound_and_survives_restart_without_rerun(self):
        options = ["Москва, Москва, Россия", "Москва, Тверская область, Россия"]
        supplied = [options[0], "<script>, Province, Country", "City, Province\nX, Country",
                    "https://secret.test, Province, Country", {"secret": "private"}, "A" * 101,
                    options[1], options[0]]
        self.worker.side_effect = web_service.TripCreateValidationError(
            "private-raw-error", code="city_ambiguous", destination_options=supplied)
        receipt = self.submit()
        self.manager.executor.run_next()
        status = jobs.get_job(1, self.account, job_id=receipt["job_id"])
        self.assertEqual(status["clarification"]["options"], options)
        self.assertEqual(json.loads(self.rows("generation_jobs")[0]["destination_options_json"]), options)
        for identifier in ({"job_id": receipt["job_id"]}, {"key": self.key}):
            for owner, account in ((2, db.get_account_key(2)), (1, db.get_account_key(2))):
                with self.subTest(owner=owner, identifier=identifier), self.assertRaises(jobs.JobNotFoundError):
                    jobs.get_job(owner, account, **identifier)
        replacement = jobs.GenerationJobs(executor_factory=DeferredExecutor)
        replacement.start()
        self.addCleanup(replacement.close)
        self.assertEqual(replacement.submit(1, self.account, self.key, self.request), status)
        self.assertEqual(replacement.executor.calls, [])
        self.assertNotIn("private", json.dumps(self.rows("generation_jobs"), ensure_ascii=False))
        self.worker.assert_called_once()
        self.claim.assert_not_called()

    def test_actual_rich_city_preflight_survives_worker_persistence_and_restart_without_paid_work(self):
        options = ("Пекин, Пекин, Китай", "Пекин, Шаньси, Китай")
        details = [
            {"label": options[0], "feature_label": "Столица страны", "admin2": "", "population": 20_000_000},
            {"label": options[1], "feature_label": "Населённый пункт", "admin2": "Юньчэн", "population": 2_000},
        ]  # Synthetic provider facts, not a current population claim.
        self.worker.side_effect = REAL_CREATE_TRIP
        with patch.object(web_service, "resolve_city_for_planning", side_effect=CityResolutionError(
                "private-destination", reason="ambiguous", suggestions=options, suggestion_details=details)) as resolve, \
                patch.object(web_service.vector_memory, "recall_family_memories",
                             side_effect=AssertionError("No paid embedding before city clarification")) as recall, \
                patch.object(web_service, "generate_itinerary",
                             side_effect=AssertionError("No model before city clarification")) as model:
            receipt = self.submit()
            resolve.assert_not_called()
            self.manager.executor.run_next()
            status = jobs.get_job(1, self.account, job_id=receipt["job_id"])
            self.assertEqual(status["clarification"], {
                "kind": "city", "options": list(options), "details": details, "attempt_used": False})
            self.assertEqual(status["status"], "failed")
            row = self.rows("generation_jobs")[0]
            self.assertEqual(json.loads(row["destination_options_json"]), {
                "version": 1, "options": list(options), "details": details})
            self.assertNotIn("private", json.dumps(row))
            replacement = jobs.GenerationJobs(executor_factory=DeferredExecutor)
            replacement.start()
            self.addCleanup(replacement.close)
            self.assertEqual(replacement.submit(1, self.account, self.key, self.request), status)
            self.assertEqual(replacement.executor.calls, [])
            self.assertEqual(jobs.get_job(1, self.account, key=self.key), status)
            for owner, account in ((2, db.get_account_key(2)), (1, db.get_account_key(2))):
                with self.subTest(owner=owner), self.assertRaises(jobs.JobNotFoundError):
                    jobs.get_job(owner, account, job_id=receipt["job_id"])
            resolve.assert_called_once_with(self.request.destination)
            recall.assert_not_called()
            model.assert_not_called()
        self.worker.assert_called_once()
        self.claim.assert_not_called()
        self.assertEqual(self.rows("ai_requests"), [])
        self.assertEqual(self.rows("access_route_attempts"), [])
        self.assertEqual(self.rows("trips"), [])

    def test_rich_city_details_cross_actual_legacy_http_and_receipt_get_without_resubmission(self):
        options = ["City, Province, Country"]
        details = [{"label": options[0], "feature_label": "Столица страны", "admin2": "Area", "population": 1234}]
        self.worker.side_effect = web_service.TripCreateValidationError(
            "private-raw-error", code="city_ambiguous", destination_options=options, destination_details=details)
        receipt = self.submit()
        self.manager.executor.run_next()
        client = TestClient(web_app.app)  # No lifespan: only our disposable DB/worker is used.
        self.addCleanup(client.close)
        session, csrf, _ = web_auth.create_session(1)
        client.cookies.set(web_auth.SESSION_COOKIE, session)
        client.cookies.set(web_auth.CSRF_COOKIE, csrf)
        expected = {"kind": "city", "options": options, "details": details, "attempt_used": False}
        for path in (f"/api/generation-jobs/{receipt['job_id']}", f"/api/generation-jobs?key={self.key}"):
            response = client.get(path)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["clarification"], expected)
            self.assertNotIn("private", response.text)
        self.worker.assert_called_once()
        self.assertEqual(self.manager.executor.calls, [])
        response = client.post("/api/trips", json=self.request.model_dump(), headers={"X-CSRF-Token": csrf})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(response.json()["detail"]["clarification"], expected)
        self.assertNotIn("private", response.text)
        self.claim.assert_not_called()
        self.assertEqual(self.rows("ai_requests"), [])

    def test_clarification_versioned_object_is_revalidated_and_legacy_array_remains_readable(self):
        options = ["City, Province, Country"]
        details = [{"label": options[0], "feature_label": "Столица страны", "admin2": "Area", "population": 1234}]
        self.worker.side_effect = web_service.TripCreateValidationError("ignored", code="city_ambiguous")
        receipt = self.submit()
        self.manager.executor.run_next()

        def read_saved(saved):
            with db.connect() as connection:
                connection.execute("UPDATE generation_jobs SET destination_options_json=? WHERE job_id=?",
                                   (json.dumps(saved, ensure_ascii=False), receipt["job_id"]))
            return jobs.get_job(1, self.account, job_id=receipt["job_id"])["clarification"]

        self.assertEqual(read_saved(options), {"kind": "city", "options": options, "attempt_used": False})
        for saved in (
                {"version": True, "options": options, "details": details},
                {"version": "1", "options": options, "details": details},
                {"version": 2, "options": options, "details": details},
                {"version": 1, "options": options},
                {"version": 1, "options": options, "details": details, "secret": "private"},
                {"version": 1, "options": {"private": options}, "details": details},
                {"version": 1, "options": [], "details": details}):
            with self.subTest(saved=saved):
                self.assertEqual(read_saved(saved), {"kind": "city", "options": [], "attempt_used": False})
        for invalid in (None, {}, "private", [{"label": "Wrong, Province, Country", "population": 100}],
                        [{"label": options[0], "feature_label": "<script>private</script>",
                          "admin2": "private\u202eregion", "population": True}]):
            with self.subTest(invalid=invalid):
                self.assertEqual(read_saved({"version": 1, "options": options, "details": invalid}),
                                 {"kind": "city", "options": options, "attempt_used": False})
        raw = {"version": 1, "options": options, "details": [
            {**details[0], "latitude": 1, "private": "secret", "provider_response": {"raw": "private"}}]}
        self.assertEqual(read_saved(raw), {"kind": "city", "options": options, "details": details, "attempt_used": False})
        self.worker.assert_called_once()
        self.claim.assert_not_called()

    def test_wrong_failure_code_or_empty_details_never_persists_or_exposes_rich_clarification(self):
        options = ["City, Province, Country"]
        details = [{"label": options[0], "feature_label": "Столица страны", "admin2": "Area", "population": 1234}]
        for code in ("city_not_found", "city_unavailable", "validation_error", "private"):
            with self.subTest(code=code):
                key = str(uuid4())
                self.worker.side_effect = web_service.TripCreateValidationError(
                    "ignored", code=code, destination_options=options, destination_details=details)
                receipt = self.submit(key=key)
                self.manager.executor.run_next()
                row = next(row for row in self.rows("generation_jobs") if row["job_id"] == receipt["job_id"])
                self.assertEqual(json.loads(row["destination_options_json"]), [])
                with db.connect() as connection:
                    connection.execute("UPDATE generation_jobs SET destination_options_json=? WHERE job_id=?",
                                       (json.dumps({"version": 1, "options": options, "details": details}), receipt["job_id"]))
                exposed = jobs.get_job(1, self.account, job_id=receipt["job_id"])
                if code in web_service.CITY_PREFLIGHT_CODES:
                    self.assertEqual(exposed["clarification"], {"kind": "city", "options": [], "attempt_used": False})
                else:
                    self.assertNotIn("clarification", exposed)
        self.worker.side_effect = web_service.TripCreateValidationError(
            "ignored", code="city_ambiguous", destination_options=options, destination_details=[])
        receipt = self.submit(key=str(uuid4()))
        self.manager.executor.run_next()
        row = next(row for row in self.rows("generation_jobs") if row["job_id"] == receipt["job_id"])
        self.assertEqual(json.loads(row["destination_options_json"]), options)
        self.assertNotIn("details", jobs.get_job(1, self.account, job_id=receipt["job_id"])["clarification"])
        self.claim.assert_not_called()

    def test_clarification_revalidates_persisted_options_and_never_applies_to_other_failures(self):
        self.worker.side_effect = web_service.TripCreateValidationError("ignored", code="city_ambiguous")
        receipt = self.submit()
        self.manager.executor.run_next()
        for raw in ("not-json", '{"raw":"secret"}', "[" + " " * 4096 + "]",
                    '["<b>unsafe</b>, Province, Country", 1, null]'):
            with self.subTest(raw=raw[:30]), db.connect() as connection:
                connection.execute("UPDATE generation_jobs SET destination_options_json=? WHERE job_id=?",
                                   (raw, receipt["job_id"]))
            self.assertEqual(jobs.get_job(1, self.account, job_id=receipt["job_id"])["clarification"]["options"], [])
        with db.connect() as connection:
            connection.execute("UPDATE generation_jobs SET failure_code='validation', destination_options_json=? WHERE job_id=?",
                               (json.dumps(["Москва, Москва, Россия"]), receipt["job_id"]))
        self.assertNotIn("clarification", jobs.get_job(1, self.account, job_id=receipt["job_id"]))

    def test_clarification_keeps_at_most_five_unique_plain_text_options(self):
        options = [f"City {index}, Province, Country" for index in range(8)]
        self.worker.side_effect = web_service.TripCreateValidationError("ignored", code="city_ambiguous",
                                                                      destination_options=options)
        self.submit()
        self.manager.executor.run_next()
        status = jobs.get_job(1, self.account, key=self.key)
        self.assertEqual(status["clarification"]["options"], options[:5])
        self.assertEqual(json.loads(self.rows("generation_jobs")[0]["destination_options_json"]), options[:5])

    def test_generic_error_contains_no_exception_text(self):
        self.worker.side_effect = RuntimeError("SECRET_token=fixture personal-model-text")
        self.submit()
        self.manager.executor.run_next()
        status = jobs.get_job(1, self.account, key=self.key)
        self.assertEqual(status["message"], jobs.MESSAGES["internal"])
        self.assertNotIn("SECRET", str(self.rows("generation_jobs")))

    def test_startup_interrupts_old_pending_without_reexecuting_it(self):
        status = self.submit()
        # The old deferred process disappeared; startup must not resume it.
        self.manager.executor.calls.clear()
        replacement = jobs.GenerationJobs(executor_factory=DeferredExecutor)
        replacement.start()
        self.addCleanup(replacement.close)
        recovered = jobs.get_job(1, self.account, job_id=status["job_id"])
        self.assertEqual(recovered["status"], "interrupted")
        self.assertEqual(replacement.submit(1, self.account, self.key, self.request), recovered)
        self.assertEqual(replacement.executor.calls, [])
        self.worker.assert_not_called()
        self.claim.assert_not_called()

    def test_startup_keeps_saved_receipt_recoverable(self):
        status = self.finish_saved()
        replacement = jobs.GenerationJobs(executor_factory=DeferredExecutor)
        replacement.start()
        self.addCleanup(replacement.close)
        self.assertEqual(jobs.get_job(1, self.account, key=self.key), status)
        self.assertEqual(replacement.executor.calls, [])

    def test_deleted_trip_does_not_regenerate(self):
        status = self.finish_saved()
        db.delete_trip(status["trip_id"], 1)
        recovered = self.submit()
        self.assertEqual(recovered["status"], "succeeded")
        self.assertIsNone(recovered["trip_id"])
        self.assertEqual(recovered["message"], jobs.MESSAGES["deleted"])
        self.worker.assert_called_once()

    def test_owner_isolation_and_recreated_account_namespace(self):
        status = self.submit()
        for identifier in ({"job_id": status["job_id"]}, {"key": self.key}):
            with self.assertRaises(jobs.JobNotFoundError):
                jobs.get_job(2, db.get_account_key(2), **identifier)
        token = jobs.owner_token(self.account)
        with db.connect() as connection:
            connection.execute("DELETE FROM users WHERE tg_id=1")
        db.save_user(1, "Recreated", 2, [], "RUB")
        self.assertNotEqual(jobs.owner_token(db.get_account_key(1)), token)
        self.manager.executor.run_next()
        self.worker.assert_not_called()
        self.assertEqual(self.rows("generation_jobs"), [])
        with self.assertRaises(jobs.JobNotFoundError):
            jobs.get_job(1, db.get_account_key(1), job_id=status["job_id"])

    def test_save_rejects_wrong_account_or_not_running_before_any_write(self):
        status = self.submit()
        for account in (self.account, db.get_account_key(2)):
            with self.assertRaises(ValueError):
                self.save(1, self.request, expected_account_key=account, generation_job_id=status["job_id"])
        self.assertEqual(self.rows("trips"), [])
        self.assertEqual(self.rows("checklist_items"), [])

    def test_aborted_account_deletion_does_not_leave_a_queued_receipt_without_worker(self):
        status = self.submit()
        with db.connect() as connection:
            connection.execute("UPDATE users SET deletion_token='synthetic-deletion-lease' WHERE tg_id=1")
        self.manager.executor.run_next()
        self.worker.assert_not_called()
        self.claim.assert_not_called()
        with self.assertRaises(jobs.JobNotFoundError):
            jobs.get_job(1, self.account, job_id=status["job_id"])
        # External deletion failed: privacy releases the lease, keeping data.
        with db.connect() as connection:
            connection.execute("UPDATE users SET deletion_token='' WHERE tg_id=1")
        recovered = jobs.get_job(1, self.account, key=self.key)
        self.assertEqual(recovered["status"], "interrupted")
        self.assertEqual(recovered["message"], jobs.MESSAGES["account_change"])
        self.assertEqual(self.submit(), recovered)
        self.assertEqual(self.manager.executor.calls, [])
        fresh = self.submit(key=str(uuid4()))
        self.assertEqual(fresh["status"], "queued")
        self.assertNotEqual(fresh["job_id"], recovered["job_id"])
        self.assertEqual(len(self.manager.executor.calls), 1)
        self.assertEqual(self.rows("trips"), [])
        self.worker.assert_not_called()
        self.claim.assert_not_called()

    def test_checklist_failure_rolls_back_trip_and_keeps_no_success_receipt(self):
        self.worker.side_effect = self.save
        self.submit()
        with db.connect() as connection:
            connection.execute("CREATE TRIGGER abort_checklist BEFORE INSERT ON checklist_items BEGIN SELECT RAISE(ABORT, 'synthetic'); END")
        self.manager.executor.run_next()
        self.assertEqual(self.rows("trips"), [])
        self.assertEqual(self.rows("checklist_items"), [])
        self.assertEqual(jobs.get_job(1, self.account, key=self.key)["status"], "failed")

    def test_commit_failure_releases_capacity_and_rolls_back_receipt(self):
        connect = db.connect
        @contextmanager
        def fail_commit():
            with connect() as connection:
                yield connection
                raise sqlite3.OperationalError("synthetic commit failure")
        with patch.object(db, "connect", fail_commit), self.assertRaises(sqlite3.OperationalError):
            self.submit()
        self.assertEqual(self.rows("generation_jobs"), [])
        self.submit()
        self.submit(owner=2)
        self.assertEqual(len(self.manager.executor.calls), 2)

    def test_outer_persistence_failure_marks_failed_when_db_recovers(self):
        status = self.submit()
        with patch.object(jobs, "_owner", side_effect=sqlite3.OperationalError("private-secret")):
            with self.assertLogs("web.jobs", level="ERROR") as output:
                self.manager.executor.run_next()
        self.assertNotIn("secret", str(output.output))
        self.assertEqual(jobs.get_job(1, self.account, job_id=status["job_id"])["status"], "failed")
        self.worker.assert_not_called()

    def test_executor_submission_failure_keeps_interrupted_receipt_and_releases_slot(self):
        with patch.object(self.manager.executor, "submit", side_effect=RuntimeError("secret")):
            status = self.submit()
        self.assertEqual(status["status"], "interrupted")
        self.submit(owner=2)
        self.submit(owner=3)
        self.assertEqual(len(self.manager.executor.calls), 2)

    def test_shutdown_waits_for_workers_and_nested_lifespan_does_not_interrupt(self):
        self.worker.side_effect = self.save
        status = self.submit()
        self.manager.start()  # Nested local TestClient, not a second web process.
        self.manager.close()
        self.assertIsNotNone(self.manager.executor)
        self.assertEqual(jobs.get_job(1, self.account, key=self.key)["status"], "queued")
        self.manager.close()
        self.assertEqual(jobs.get_job(1, self.account, key=self.key)["status"], "succeeded")
        self.manager.start()
        self.assertEqual(self.manager.executor.calls, [])
        self.assertEqual(self.submit()["job_id"], status["job_id"])

    def test_invalid_key_or_dates_leave_no_receipt(self):
        for key in ("", "not-a-uuid", "A" * 36):
            with self.assertRaises(ValueError):
                self.submit(key=key or "invalid")
        with self.assertRaises(ValueError):
            self.submit(request=self.request.model_copy(update={"start_date": "2000-01-01", "end_date": "2000-01-01"}))
        self.assertEqual(self.rows("generation_jobs"), [])
        self.claim.assert_not_called()

    def test_deterministic_city_nearby_preflight_rejects_before_job_or_quota(self):
        nearby = self.request.model_copy(update={"location_preference": "nearby", "stay_area": "Synthetic area"})
        with patch.object(web_service, "city_candidates_enabled", return_value=True), \
                patch.object(web_service, "city_deterministic_enabled", return_value=True):
            with self.assertRaises(web_service.TripCreateValidationError) as raised:
                self.submit(request=nearby)
            self.assertEqual(raised.exception.code, "unsupported_preferences")
            self.assertIn("нет проверенной точки размещения", str(raised.exception))
            with self.assertRaises(web_service.TripCreateValidationError) as raised:
                REAL_CREATE_TRIP(1, nearby, expected_account_key=self.account)
            self.assertEqual(raised.exception.code, "unsupported_preferences")
            self.claim.assert_not_called()
            self.worker.assert_not_called()
            self.assertEqual(self.rows("generation_jobs"), [])
            self.assertEqual(self.rows("trips"), [])
            self.assertEqual(self.manager.executor.calls, [])

    def test_nearby_preflight_does_not_change_region_or_either_legacy_flag(self):
        for candidate_mode, deterministic_mode, kind in ((False, False, "city"), (False, True, "city"),
                                                          (True, False, "city"), (True, True, "region")):
            with self.subTest(candidate_mode=candidate_mode, deterministic_mode=deterministic_mode, kind=kind), \
                    patch.object(web_service, "city_candidates_enabled", return_value=candidate_mode), \
                    patch.object(web_service, "city_deterministic_enabled", return_value=deterministic_mode):
                nearby = self.request.model_copy(update={"location_preference": "nearby", "stay_area": "Synthetic area",
                                                        "destination_kind": kind})
                normalized = web_service._normalized_trip_request(nearby)
                self.assertEqual(normalized.location_preference, "nearby")
                self.assertEqual(normalized.stay_area, "Synthetic area")
                self.assertEqual(normalized.trip_goal, nearby.trip_goal)

    def test_nearby_http_preflight_has_safe_actionable_message_without_acceptance(self):
        with patch.object(jobs, "manager", self.manager), \
                patch.object(web_service, "city_candidates_enabled", return_value=True), \
                patch.object(web_service, "city_deterministic_enabled", return_value=True), \
                TestClient(web_app.app) as client:
            session, csrf, _ = web_auth.create_session(1)
            client.cookies.set(web_auth.SESSION_COOKIE, session)
            client.cookies.set(web_auth.CSRF_COOKIE, csrf)
            body = self.request.model_dump()
            body.update(location_preference="nearby", stay_area="private-specific-address")
            response = client.post("/api/generation-jobs", json=body,
                                   headers={"X-CSRF-Token": csrf, "Idempotency-Key": self.key})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["detail"], jobs.MESSAGES["unsupported_preferences"])
            self.assertNotIn("private-specific-address", response.text)
            self.assertEqual(self.rows("generation_jobs"), [])
            self.claim.assert_not_called()
            self.worker.assert_not_called()

    def test_http_auth_csrf_owner_isolation_and_legacy_post_compatibility(self):
        with patch.object(jobs, "manager", self.manager), TestClient(web_app.app) as client:
            body = self.request.model_dump()
            self.assertEqual(client.post("/api/generation-jobs", json=body).status_code, 401)
            session, csrf, _ = web_auth.create_session(1)
            client.cookies.set(web_auth.SESSION_COOKIE, session)
            client.cookies.set(web_auth.CSRF_COOKIE, csrf)
            self.assertEqual(client.post("/api/generation-jobs", json=body).status_code, 403)
            headers = {"X-CSRF-Token": csrf, "Idempotency-Key": self.key}
            response = client.post("/api/generation-jobs", json=body, headers=headers)
            self.assertEqual(response.status_code, 202, response.text)
            status = response.json()
            self.assertEqual(client.post("/api/generation-jobs", json=body, headers=headers).json(), status)
            self.assertEqual(client.get(f"/api/generation-jobs/{status['job_id']}").json(), status)
            self.assertEqual(client.get("/api/generation-jobs", params={"key": self.key}).json(), status)
            profile = client.get("/api/me").json()
            self.assertEqual(profile["generation_jobs"], {"enabled": True, "owner": jobs.owner_token(self.account)})
            self.assertNotIn(self.account, json.dumps(profile))
            self.assertEqual(len(self.manager.executor.calls), 1)
            session2, _, _ = web_auth.create_session(2)
            client.cookies.set(web_auth.SESSION_COOKIE, session2)
            self.assertEqual(client.get(f"/api/generation-jobs/{status['job_id']}").status_code, 404)
            self.assertEqual(client.get("/api/generation-jobs", params={"key": self.key}).status_code, 404)
            client.cookies.set(web_auth.SESSION_COOKIE, session)
            self.worker.side_effect = None
            self.worker.return_value = {"id": 123}
            legacy = client.post("/api/trips", json=body, headers={"X-CSRF-Token": csrf})
            self.assertEqual(legacy.status_code, 201)
            self.assertNotIn("generation_job_id", self.worker.call_args.kwargs)

    def test_actual_service_claim_save_and_recover_with_only_model_and_memory_mocked(self):
        self.worker.side_effect = REAL_CREATE_TRIP
        self.claim.side_effect = REAL_CLAIM
        result = SimpleNamespace(itinerary=self.itinerary, problems=[], weather=[], weather_text="",
                                 attempts=1, prompt_tokens=10, completion_tokens=20, total_tokens=30)
        with patch.object(web_service, "generate_itinerary", return_value=result) as model, \
                patch.object(web_service, "resolve_city_for_planning", return_value=object()) as resolve, \
                patch.object(web_service.vector_memory, "recall_family_memories", return_value=[]), \
                patch.object(web_service, "_save_trip_memory_safely", return_value=0), \
                patch.object(web_service, "_daily_ai_limit", return_value=5):
            self.submit()
            self.manager.executor.run_next()
            status = jobs.get_job(1, self.account, key=self.key)
            self.assertEqual(status["status"], "succeeded")
            self.assertEqual(self.rows("ai_requests")[0]["status"], "succeeded")
            self.assertEqual(len(self.rows("trips")), 1)
            self.assertGreater(len(self.rows("checklist_items")), 0)
            self.assertEqual(self.submit(), status)
            model.assert_called_once()
            self.claim.assert_called_once()
            resolve.assert_called_once_with(self.request.destination)
            self.assertNotIn("tokens", json.dumps(status))

    def test_actual_thread_executor_has_two_inflight_slots_and_no_third_queued_task(self):
        started = threading.Event()
        finish = threading.Event()
        lock = threading.Lock()
        active = []
        def controlled(*args, **kwargs):
            with lock:
                active.append(args[0])
                if len(active) == 2:
                    started.set()
            if not finish.wait(5):
                raise AssertionError("Synthetic worker timed out")
            return self.save(*args, **kwargs)
        self.worker.side_effect = controlled
        real = jobs.GenerationJobs()
        real.start()
        try:
            real.submit(1, self.account, self.key, self.request)
            real.submit(2, db.get_account_key(2), str(uuid4()), self.request)
            self.assertTrue(started.wait(5))
            with self.assertRaises(jobs.JobBusyError):
                real.submit(3, db.get_account_key(3), str(uuid4()), self.request)
            self.assertEqual(len(self.rows("generation_jobs")), 2)
            self.assertEqual(sorted(active), [1, 2])
        finally:
            finish.set()
            real.close()
        self.assertEqual(len(self.rows("trips")), 2)
        self.assertTrue(all(row["status"] == "succeeded" for row in self.rows("generation_jobs")))


if __name__ == "__main__":
    unittest.main()
