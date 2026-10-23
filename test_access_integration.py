"""Access integration uses disposable data, deferred workers and mocked HTTP."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import hmac
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from uuid import uuid4

with patch("dotenv.load_dotenv", return_value=False):
    import access_control as access
    import access_guard
    import db
    import generation_jobs
    import trip_covers
    import web_app
    import web_auth
    import web_service
    import yandex_ai
    import yandex_art
from fastapi.testclient import TestClient
import httpx
from schemas import TripRequest


class DeferredExecutor:
    def __init__(self, **kwargs):
        self.calls = []

    def submit(self, function, *args):
        self.calls.append((function, args))

    def run_next(self):
        function, args = self.calls.pop(0)
        function(*args)

    def shutdown(self, **kwargs):
        self.calls.clear()


class AccessIntegrationTests(unittest.TestCase):
    OWNER = 9001
    GUEST = 1001

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="offline-access-integration-")))
        self.stack.enter_context(patch.object(db, "DB_PATH", str(self.directory / "synthetic.db")))
        self.stack.enter_context(patch.dict("os.environ", {
            "ACCESS_MODE": "invite", "ACCESS_OWNER_IDS": str(self.OWNER),
            "ACCESS_GLOBAL_ROUTE_DAILY_LIMIT": "12",
            "WEB_SESSION_SECRET": "synthetic-offline-access-integration-secret-123456789",
            "WEB_COOKIE_SECURE": "0", "WEB_BASE_URL": "http://testserver",
            "IMAGE_GENERATION_ENABLED": "1", "IMAGE_GENERATION_DAILY_LIMIT": "10",
            "YANDEX_API_KEY": "synthetic-key", "YANDEX_ART_API_KEY": "synthetic-image-key",
            "YANDEX_FOLDER_ID": "synthetic-folder", "YANDEX_ART_MODEL_URI": "",
        }))
        db.init_db()
        trip_covers.init_schema()
        for user in (self.GUEST, self.OWNER):
            db.save_user(user, "Synthetic", 2, [], "RUB")
        self.invitation = access.create_invite(self.OWNER)
        self.grant = access.redeem_invite(self.GUEST, self.invitation["token"])
        self.account = db.get_account_key(self.GUEST)
        self.request = TripRequest(destination="Москва, Россия", start_date="2099-10-22", end_date="2099-10-22",
                                   budget=20000, currency="RUB", trip_goal="Познакомиться с городом",
                                   interest_categories=["main_sights"])
        self.trip = db.save_trip(self.GUEST, self.request.model_dump(), {"title": "Synthetic", "days": []})
        # Importing these modules must never turn a failing test into a live call.
        self.http = self.stack.enter_context(patch.object(httpx, "post", side_effect=AssertionError("Unexpected HTTP")))
        self.images = self.stack.enter_context(patch.object(yandex_art, "submit", side_effect=AssertionError("Unexpected image generation")))

    def rows(self, table):
        with db.connect() as conn:
            return [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]

    def revoke(self):
        access.revoke_invite(self.OWNER, self.invitation["invite_id"])

    def test_web_status_returns_five_lifetime_attempts_without_spending_any(self):
        client = self.client(self.GUEST)
        for _ in range(3):
            response = client.get("/api/me")
            self.assertEqual(response.status_code, 200)
            state = response.json()["access"]
            self.assertEqual((state["route_limit"], state["route_remaining"]), (5, 5))
        self.assert_no_paid_work()

    def replace_grant(self):
        self.revoke()
        self.invitation = access.create_invite(self.OWNER)
        return access.redeem_invite(self.GUEST, self.invitation["token"])

    def client(self, user=None, session=None):
        client = TestClient(web_app.app)
        self.addCleanup(client.close)
        if user is not None or session is not None:
            token, csrf, _ = session or web_auth.create_session(user)
            client.cookies.set(web_auth.SESSION_COOKIE, token)
            client.cookies.set(web_auth.CSRF_COOKIE, csrf)
            client.headers["X-CSRF-Token"] = csrf
        return client

    def api_operations(self):
        return (
            ("post", "/api/trips", self.request.model_dump(mode="json")),
            ("post", "/api/generation-jobs", self.request.model_dump(mode="json")),
            ("post", f"/api/trips/{self.trip}/days/1/edit", {"instruction": "Начать прогулку позже"}),
            ("post", f"/api/trips/{self.trip}/cover", None),
            ("get", f"/api/trips/{self.trip}/cover/image", None),
        )

    def assert_no_paid_work(self):
        self.http.assert_not_called()
        self.images.assert_not_called()
        self.assertEqual(self.rows("ai_requests"), [])
        self.assertEqual(self.rows("access_route_attempts"), [])

    def assert_slots_released(self, manager, count=2):
        acquired = []
        try:
            for _ in range(count):
                self.assertTrue(manager.slots.acquire(blocking=False))
                acquired.append(True)
            self.assertFalse(manager.slots.acquire(blocking=False))
        finally:
            for _ in acquired:
                manager.slots.release()

    def test_revoked_login_token_cannot_be_consumed(self):
        token = web_auth.issue_login_token(self.GUEST)
        self.revoke()
        with self.assertRaises(access.AccessDeniedError):
            web_auth.consume_login_token(token)
        self.assertIsNone(web_auth.consume_login_token(token))
        self.assert_no_paid_work()

    def test_old_login_token_does_not_revive_after_new_invitation(self):
        token = web_auth.issue_login_token(self.GUEST)
        new = self.replace_grant()
        self.assertNotEqual(self.grant["grant_id"], new["grant_id"])
        with self.assertRaises(access.AccessDeniedError):
            web_auth.consume_login_token(token)
        current = web_auth.issue_login_token(self.GUEST)
        self.assertEqual(web_auth.consume_login_token(current), self.GUEST)
        self.assertIsNone(web_auth.consume_login_token(current))

    def test_consume_login_grant_returns_validated_snapshot_and_consumes_once(self):
        token = web_auth.issue_login_token(self.GUEST)
        self.assertEqual(web_auth.consume_login_grant(token), (self.GUEST, self.grant["grant_id"]))
        self.assertIsNone(web_auth.consume_login_grant(token))
        self.assertIsNone(web_auth.consume_login_grant("invalid"))

    def test_session_creation_rejects_changed_expected_grant(self):
        self.replace_grant()
        with self.assertRaises(access.AccessDeniedError):
            web_auth.create_session(self.GUEST, expected_grant=self.grant["grant_id"])

    def test_api_login_does_not_switch_grant_between_consumption_and_session_creation(self):
        token = web_auth.issue_login_token(self.GUEST)
        original = web_auth.consume_login_grant
        def consume_then_replace(value):
            result = original(value)
            self.replace_grant()
            return result
        client = self.client()
        with patch.object(web_app, "consume_login_grant", side_effect=consume_then_replace):
            response = client.post("/api/login", json={"token": token})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["code"], "access_denied")
        self.assertFalse(response.headers.get_list("set-cookie"))
        self.assertNotIn(web_auth.SESSION_COOKIE, client.cookies)
        self.assertNotIn(web_auth.CSRF_COOKIE, client.cookies)
        self.assertEqual(client.post("/api/login", json={"token": token}).status_code, 401)
        self.assert_no_paid_work()

    def test_api_login_creates_session_bound_to_validated_grant(self):
        client = self.client()
        token = web_auth.issue_login_token(self.GUEST)
        response = client.post("/api/login", json={"token": token})
        self.assertEqual(response.status_code, 200)
        session = web_auth.verify_session(client.cookies[web_auth.SESSION_COOKIE])
        self.assertEqual(session.grant_id, self.grant["grant_id"])
        self.assertEqual(client.get("/api/me").status_code, 200)

    def test_api_login_preserves_owner_old_open_token_in_invite_mode(self):
        with patch.dict("os.environ", {"ACCESS_MODE": "open"}):
            token = web_auth.issue_login_token(self.OWNER)
        client = self.client()
        response = client.post("/api/login", json={"token": token})
        self.assertEqual(response.status_code, 200)
        session = web_auth.verify_session(client.cookies[web_auth.SESSION_COOKIE])
        self.assertEqual(session.grant_id, "owner")
        self.assertEqual(client.get("/api/me").status_code, 200)

    def test_old_session_denied_after_revoke_and_stays_denied_after_reissue(self):
        client = self.client(self.GUEST)
        self.assertEqual(client.get("/api/me").status_code, 200)
        self.revoke()
        self.assertEqual(client.get("/api/me").status_code, 403)
        self.invitation = access.create_invite(self.OWNER)
        new = access.redeem_invite(self.GUEST, self.invitation["token"])
        self.assertEqual(client.get("/api/me").status_code, 403)
        current = self.client(self.GUEST).get("/api/me")
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["access"]["grant_id"], new["grant_id"])

    def test_owner_keeps_existing_open_session_and_open_login_token(self):
        with patch.dict("os.environ", {"ACCESS_MODE": "open"}):
            session = web_auth.create_session(self.OWNER)
            login_token = web_auth.issue_login_token(self.OWNER)
        client = self.client(session=session)
        self.assertEqual(client.get("/api/me").status_code, 200)
        self.assertEqual(web_auth.consume_login_token(login_token), self.OWNER)

    def test_owner_keeps_legacy_signed_session_without_grant_field(self):
        session, csrf, expires = web_auth.create_session(self.OWNER)
        payload = json.loads(web_auth._urlsafe_decode(session.split(".", 1)[0]))
        payload.pop("grant")
        encoded = web_auth._urlsafe_encode(json.dumps(payload, separators=(",", ":")).encode())
        signature = web_auth._urlsafe_encode(hmac.new(web_auth._secret(), encoded.encode(), hashlib.sha256).digest())
        legacy = (encoded + "." + signature, csrf, expires)
        self.assertEqual(self.client(session=legacy).get("/api/me").status_code, 200)

    def test_guest_open_session_does_not_gain_new_invitation_access(self):
        with patch.dict("os.environ", {"ACCESS_MODE": "open"}):
            session = web_auth.create_session(self.GUEST)
            login_token = web_auth.issue_login_token(self.GUEST)
        self.assertEqual(self.client(session=session).get("/api/me").status_code, 403)
        with self.assertRaises(access.AccessDeniedError):
            web_auth.consume_login_token(login_token)

    def test_unauthenticated_creation_edit_and_images_reject_without_paid_work(self):
        client = self.client()
        client.headers["Idempotency-Key"] = str(uuid4())
        with patch.object(web_service, "create_trip") as create, patch.object(web_service, "edit_trip_day") as edit:
            for method, url, payload in self.api_operations():
                with self.subTest(url=url):
                    response = client.request(method, url, **({"json": payload} if payload is not None else {}))
                    self.assertEqual(response.status_code, 401)
            create.assert_not_called()
            edit.assert_not_called()
        self.assertEqual(self.rows("cover_attempts"), [])
        self.assertEqual(self.rows("generation_jobs"), [])
        self.assert_no_paid_work()

    def test_revoked_authenticated_api_creation_edit_and_images_reject_without_quota(self):
        client = self.client(self.GUEST)
        client.headers["Idempotency-Key"] = str(uuid4())
        self.revoke()
        with patch.object(web_service, "create_trip") as create, patch.object(web_service, "edit_trip_day") as edit:
            for method, url, payload in self.api_operations():
                with self.subTest(url=url):
                    response = client.request(method, url, **({"json": payload} if payload is not None else {}))
                    self.assertEqual(response.status_code, 403)
                    self.assertEqual(response.json()["code"], "access_denied")
            create.assert_not_called()
            edit.assert_not_called()
        self.assertEqual(self.rows("cover_attempts"), [])
        self.assertEqual(self.rows("generation_jobs"), [])
        self.assert_no_paid_work()

    def test_logout_still_clears_cookies_after_access_revocation(self):
        client = self.client(self.GUEST)
        self.revoke()
        response = client.post("/api/logout")
        self.assertEqual(response.status_code, 200)
        cleared = response.headers.get_list("set-cookie")
        for cookie in (web_auth.SESSION_COOKIE, web_auth.CSRF_COOKIE):
            self.assertTrue(any(cookie in value and "Max-Age=0" in value for value in cleared))
        self.assert_no_paid_work()

    def test_logout_without_access_still_requires_csrf(self):
        client = self.client(self.GUEST)
        self.revoke()
        del client.headers["X-CSRF-Token"]
        response = client.post("/api/logout")
        self.assertEqual(response.status_code, 403)
        self.assertFalse(response.headers.get_list("set-cookie"))

    def test_generation_queued_then_revoked_does_not_start_provider_and_releases_slots(self):
        manager = generation_jobs.GenerationJobs(executor_factory=DeferredExecutor)
        manager.start()
        self.addCleanup(manager.close)
        accepted = manager.submit(self.GUEST, self.account, str(uuid4()), self.request)
        self.assertEqual(accepted["status"], "queued")
        self.revoke()
        with patch.object(web_service, "create_trip") as create:
            manager.executor.run_next()
            create.assert_not_called()
        result = generation_jobs.get_job(self.GUEST, self.account, job_id=accepted["job_id"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.rows("generation_jobs")[0]["failure_code"], "access")
        self.assert_slots_released(manager)
        self.assert_no_paid_work()

    def test_old_queued_generation_cannot_resume_under_new_grant(self):
        manager = generation_jobs.GenerationJobs(executor_factory=DeferredExecutor)
        manager.start()
        self.addCleanup(manager.close)
        manager.submit(self.GUEST, self.account, str(uuid4()), self.request)
        self.replace_grant()
        with patch.object(web_service, "create_trip") as create:
            manager.executor.run_next()
            create.assert_not_called()
        self.assertEqual(self.rows("generation_jobs")[0]["failure_code"], "access")
        self.assert_slots_released(manager)
        self.assert_no_paid_work()

    def cover_manager(self):
        manager = trip_covers.TripCovers(executor_factory=DeferredExecutor, cache_dir=self.directory / "covers")
        manager.start()
        self.addCleanup(manager.close)
        return manager

    def test_cover_queued_then_revoked_does_not_call_provider_and_releases_slots(self):
        manager = self.cover_manager()
        self.assertEqual(manager.submit(self.GUEST, self.account, self.trip)["status"], "queued")
        self.revoke()
        manager.executor.run_next()
        self.assertEqual(manager.get_status(self.GUEST, self.account, self.trip)["status"], "interrupted")
        self.assertEqual(self.rows("trip_covers")[0]["failure_code"], "access")
        self.assertEqual(self.rows("access_grants")[0]["cover_used"], 0)
        # Existing global cover policy reserves at acceptance, even if cancelled.
        self.assertEqual(len(self.rows("cover_attempts")), 1)
        self.assert_slots_released(manager)
        self.assert_no_paid_work()

    def test_old_queued_cover_cannot_spend_new_grant(self):
        manager = self.cover_manager()
        manager.submit(self.GUEST, self.account, self.trip)
        self.replace_grant()
        manager.executor.run_next()
        self.assertEqual(self.rows("trip_covers")[0]["failure_code"], "access")
        self.assertEqual(sum(row["cover_used"] for row in self.rows("access_grants")), 0)
        self.assert_slots_released(manager)
        self.assert_no_paid_work()

    def test_scope_revocation_blocks_next_yandex_http_request(self):
        self.http.side_effect = None
        self.http.return_value = httpx.Response(200, json={"result": {}})
        with access_guard.operation(self.GUEST):
            yandex_ai._send_request({}, {})
            self.revoke()
            with self.assertRaises(access.AccessDeniedError):
                yandex_ai._send_request({}, {})
        self.assertEqual(self.http.call_count, 1)

    def test_scope_reissue_does_not_authorize_next_yandex_http(self):
        with access_guard.operation(self.GUEST):
            self.replace_grant()
            with self.assertRaises(access.AccessDeniedError):
                yandex_ai._send_request({}, {})
        self.http.assert_not_called()

    def test_scopes_are_thread_isolated_and_clean_up_after_error(self):
        barrier = threading.Barrier(2)
        revoked = threading.Event()
        def guest():
            with access_guard.operation(self.GUEST):
                barrier.wait(timeout=5)
                self.assertTrue(revoked.wait(timeout=5))
                with self.assertRaises(access.AccessDeniedError):
                    access_guard.check()
            access_guard.check()
            return "guest-denied"
        def owner():
            with access_guard.operation(self.OWNER):
                barrier.wait(timeout=5)
                self.revoke()
                revoked.set()
                access_guard.check()
            return "owner-allowed"
        with ThreadPoolExecutor(max_workers=2) as workers:
            guest_result = workers.submit(guest)
            owner_result = workers.submit(owner)
            self.assertEqual(guest_result.result(timeout=10), "guest-denied")
            self.assertEqual(owner_result.result(timeout=10), "owner-allowed")
        access_guard.check()

    def test_nested_scope_cannot_switch_user(self):
        with access_guard.operation(self.GUEST):
            with self.assertRaises(access.AccessDeniedError):
                with access_guard.operation(self.OWNER):
                    self.fail("Switched scope user")
            access_guard.check()

    def test_nested_scope_preserves_same_user_same_grant(self):
        with access_guard.operation(self.GUEST):
            with access_guard.operation(self.GUEST, expected_grant=self.grant["grant_id"]):
                access_guard.check()
            access_guard.check()

    def test_nested_scope_cannot_explicitly_switch_to_new_grant(self):
        with access_guard.operation(self.GUEST):
            new = self.replace_grant()
            with self.assertRaises(access.AccessDeniedError):
                with access_guard.operation(self.GUEST, expected_grant=new["grant_id"]):
                    self.fail("Switched inherited grant")

    def test_reservation_under_old_scope_cannot_spend_new_route_grant(self):
        with access_guard.operation(self.GUEST):
            self.replace_grant()
            with self.assertRaises(access.AccessDeniedError):
                db.claim_ai_request(self.GUEST, db._utc_datetime().date(), "plan", "test")
        self.assertEqual(sum(row["route_used"] for row in self.rows("access_grants")), 0)
        self.assert_no_paid_work()

    def test_reservation_under_old_scope_cannot_spend_new_auxiliary_grant(self):
        with access_guard.operation(self.GUEST):
            self.replace_grant()
            for kind in ("cover", "memory"):
                with self.subTest(kind=kind), self.assertRaises(access.AccessDeniedError):
                    access.reserve_aux(self.GUEST, kind)
        for row in self.rows("access_grants"):
            self.assertEqual(row["cover_used"] + row["memory_used"], 0)
        self.assert_no_paid_work()

    def test_worker_access_denial_returns_safe_403_in_direct_creation_and_edit(self):
        client = self.client(self.GUEST)
        with patch.object(web_service, "create_trip", side_effect=access.AccessDeniedError()), \
             patch.object(web_service, "edit_trip_day", side_effect=access.AccessDeniedError()):
            for method, url, payload in (self.api_operations()[0], self.api_operations()[2]):
                with self.subTest(url=url):
                    response = client.request(method, url, json=payload)
                    self.assertEqual(response.status_code, 403)
                    self.assertEqual(response.json()["code"], "access_denied")
        self.assert_no_paid_work()

    def replace_during_auth(self):
        original = web_auth._require_session_access
        def authenticate_then_replace(session):
            captured = original(session)
            self.replace_grant()
            return captured
        return patch.object(web_auth, "_require_session_access", side_effect=authenticate_then_replace)

    def test_direct_creation_and_edit_preserve_grant_captured_by_http_auth(self):
        reached = []
        @access_guard.guard_user_operation
        def guarded_worker(tg_id, *args, **kwargs):
            reached.append(tg_id)
            return {"ok": True}
        for method, url, payload in (self.api_operations()[0], self.api_operations()[2]):
            client = self.client(self.GUEST)
            with self.subTest(url=url), self.replace_during_auth(), \
                 patch.object(web_service, "create_trip", guarded_worker), \
                 patch.object(web_service, "edit_trip_day", guarded_worker):
                response = client.request(method, url, json=payload)
                self.assertEqual(response.status_code, 403)
        self.assertEqual(reached, [])
        self.assert_no_paid_work()

    def test_generation_enqueue_preserves_grant_captured_by_http_auth(self):
        manager = generation_jobs.GenerationJobs(executor_factory=DeferredExecutor)
        manager.start()
        self.addCleanup(manager.close)
        client = self.client(self.GUEST)
        client.headers["Idempotency-Key"] = str(uuid4())
        with self.replace_during_auth(), patch.object(generation_jobs, "manager", manager):
            response = client.post("/api/generation-jobs", json=self.request.model_dump(mode="json"))
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.rows("generation_jobs"), [])
        self.assertEqual(manager.executor.calls, [])
        self.assert_slots_released(manager)
        self.assert_no_paid_work()

    def test_cover_enqueue_preserves_grant_captured_by_http_auth(self):
        manager = self.cover_manager()
        client = self.client(self.GUEST)
        with self.replace_during_auth(), patch.object(trip_covers, "manager", manager):
            response = client.post(f"/api/trips/{self.trip}/cover")
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.rows("trip_covers"), [])
        self.assertEqual(self.rows("cover_attempts"), [])
        self.assertEqual(manager.executor.calls, [])
        self.assert_slots_released(manager)
        self.assert_no_paid_work()

    def test_reservations_cannot_switch_user_inside_existing_scope(self):
        with access_guard.operation(self.GUEST):
            with self.assertRaises(access.AccessDeniedError):
                db.claim_ai_request(self.OWNER, db._utc_datetime().date(), "plan", "test")
            for kind in ("cover", "memory"):
                with self.subTest(kind=kind), self.assertRaises(access.AccessDeniedError):
                    access.reserve_aux(self.OWNER, kind)
        self.assert_no_paid_work()


if __name__ == "__main__":
    unittest.main()
