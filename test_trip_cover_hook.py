"""The saved-route cover hook is optional; all fixtures and providers are offline."""

from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

with patch("dotenv.load_dotenv", return_value=False):
    import db
    import trip_covers
    import web_service
from schemas import Itinerary, ItineraryDay, TripRequest


class TripCoverHookTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="offline-cover-hook-"))
        self.stack.enter_context(patch.object(db, "DB_PATH", str(Path(directory) / "synthetic.db")))
        db.init_db()
        db.save_user(1, "Synthetic", 2, [], "RUB")
        self.account = db.get_account_key(1)
        self.request = TripRequest(destination="Москва, Россия", start_date="2099-06-01",
                                   end_date="2099-06-01", budget=20000, currency="RUB",
                                   trip_goal="Познакомиться с городом", interest_categories=["main_sights"])
        itinerary = Itinerary(title="Synthetic", summary="", days=[
            ItineraryDay(day=1, date=self.request.start_date, theme="Synthetic", items=[])
        ])
        result = SimpleNamespace(itinerary=itinerary, problems=[], weather=[], weather_text="",
                                 attempts=1, prompt_tokens=10, completion_tokens=10, total_tokens=20)
        self.generate = self.stack.enter_context(patch.object(web_service, "generate_itinerary", return_value=result))
        self.stack.enter_context(patch.object(web_service, "resolve_city_for_planning", return_value=object()))
        self.stack.enter_context(patch.object(web_service.vector_memory, "recall_family_memories", return_value=[]))
        self.stack.enter_context(patch.object(web_service, "_save_trip_memory_safely", return_value=0))
        self.stack.enter_context(patch.object(web_service, "_daily_ai_limit", return_value=20))
        self.detail = self.stack.enter_context(patch.object(web_service, "get_trip_detail", return_value={"id": 1}))
        self.enabled = self.stack.enter_context(patch.object(trip_covers, "enabled", return_value=True))
        self.manager = SimpleNamespace(executor=object(), closing=False, submit=Mock())
        self.stack.enter_context(patch.object(trip_covers, "manager", self.manager))
        self.paid = self.stack.enter_context(patch.object(trip_covers.yandex_art, "submit",
            side_effect=AssertionError("Live image calls forbidden in offline hook tests")))

    def create(self):
        return web_service.create_trip(1, self.request, expected_account_key=self.account)

    def test_enqueues_exactly_once_after_route_commit_with_expected_owner(self):
        def accepted(owner, account_key, trip_id):
            self.assertEqual(owner, 1)
            self.assertEqual(account_key, self.account)
            self.assertIsNotNone(db.get_trip(trip_id, owner))
            return {"status": "queued"}
        self.manager.submit.side_effect = accepted
        response = self.create()
        self.assertEqual(response["id"], 1)
        self.manager.submit.assert_called_once_with(1, self.account, 1)
        self.paid.assert_not_called()

    def test_disabled_integration_never_submits(self):
        self.enabled.return_value = False
        self.assertEqual(self.create()["id"], 1)
        self.manager.submit.assert_not_called()

    def test_unstarted_manager_does_not_initialize_worker(self):
        self.manager.executor = None
        self.assertEqual(self.create()["id"], 1)
        self.manager.submit.assert_not_called()

    def test_closing_manager_does_not_enqueue(self):
        self.manager.closing = True
        self.assertEqual(self.create()["id"], 1)
        self.manager.submit.assert_not_called()

    def test_busy_cover_does_not_fail_saved_route_or_route_quota(self):
        self.manager.submit.side_effect = trip_covers.CoverBusyError("private-provider-details")
        with self.assertLogs("web.service", level="WARNING") as capture:
            self.assertEqual(self.create()["id"], 1)
        self.assertNotIn("private", " ".join(capture.output))
        self.assertIsNotNone(db.get_trip(1, 1))
        with db.connect() as conn:
            self.assertEqual(conn.execute("SELECT status FROM ai_requests").fetchone()[0], "succeeded")

    def test_internal_cover_failure_does_not_fail_saved_route(self):
        self.manager.submit.side_effect = RuntimeError("private-secret-credential")
        with self.assertLogs("web.service", level="WARNING") as capture:
            response = self.create()
        self.assertEqual(response["id"], 1)
        self.assertNotIn("credential", " ".join(capture.output))
        self.assertIsNotNone(db.get_trip(1, 1))

    def test_rejected_route_does_not_enqueue_cover(self):
        self.generate.return_value.problems = ["Synthetic validation error"]
        with self.assertRaises(web_service.TripCreateValidationError):
            self.create()
        self.manager.submit.assert_not_called()
        self.assertEqual(db.list_trips(1), [])

    def test_failed_persistence_does_not_enqueue(self):
        with patch.object(db, "save_trip_bundle", side_effect=RuntimeError("Synthetic failed commit")):
            with self.assertRaises(RuntimeError):
                self.create()
        self.manager.submit.assert_not_called()

    def test_committed_route_still_enqueues_if_detail_render_fails(self):
        self.detail.side_effect = RuntimeError("Synthetic display failure")
        with self.assertRaises(web_service.TripSavedResponseError):
            self.create()
        self.manager.submit.assert_called_once_with(1, self.account, 1)
        self.assertIsNotNone(db.get_trip(1, 1))

    def test_cover_configuration_failure_is_best_effort(self):
        self.enabled.side_effect = RuntimeError("private-environment")
        with self.assertLogs("web.service", level="WARNING"):
            self.assertEqual(self.create()["id"], 1)
        self.manager.submit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
