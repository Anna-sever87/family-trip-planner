"""Telegram city preflight: actual handler/resolver, synthetic data, no providers.

The Moscow fixture preserves the two observed GeoNames identities and regions.
Coordinates below are test coordinates, not a new production city catalogue.
"""

from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import socket
import sqlite3
import sys
import unittest
from unittest.mock import Mock, patch

import weather
from schemas import TripRequest
from test_bot_city_edit_gate import function
import test_bot_generation_outcome as bot_fixtures


def moscow_results():
    return [
        {"id": 524901, "name": "Москва", "country": "Россия", "country_code": "RU",
         "admin1": "Москва", "feature_code": "PPLC", "latitude": 55.75, "longitude": 37.62},
        {"id": 524900, "name": "Москва", "country": "Россия", "country_code": "RU",
         "admin1": "Тверская область", "feature_code": "PPL", "latitude": 56.90, "longitude": 32.20},
    ]


class BotCityResolutionTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.access = ModuleType("access_control")
        self.access.AccessDeniedError = type("AccessDeniedError", (Exception,), {})
        self.access.AccessQuotaError = type("AccessQuotaError", (Exception,), {})
        self.access.require_access = Mock(side_effect=lambda *_: self.events.append("access"))
        self.access.enabled = Mock(return_value=True)
        self.guard = ModuleType("access_guard")
        self.guard.snapshot = Mock(return_value="synthetic-grant")

        @contextmanager
        def operation(*_args, **_kwargs):
            self.access.require_access(123)
            yield
        self.guard.operation = operation
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {"access_control": self.access, "access_guard": self.guard}).start()
        for owner, name in ((weather.httpx, "get"), (socket.socket, "connect"),
                            (socket, "getaddrinfo"), (sqlite3, "connect")):
            patch.object(owner, name, side_effect=AssertionError(f"Unexpected {name}")).start()

        def resolve(destination):
            self.events.append("resolve")
            return weather.resolve_city_for_planning(destination)

        self.namespace = {
            "TripRequest": TripRequest, "CityResolutionError": weather.CityResolutionError,
            "resolve_city_for_planning": Mock(side_effect=resolve),
            "bot": Mock(), "db": Mock(), "threading": Mock(), "types": Mock(),
            "_message_user_id": lambda _message: 123,
            "minimum_meal_cost": lambda _request: 100,
            "city_candidates_enabled": Mock(return_value=False),
            "_reserve_generation_job": Mock(return_value=True),
            "_claim_route_ai_request": Mock(side_effect=lambda *_: (
                self.events.append("claim") or SimpleNamespace(status="ok", claim_id="synthetic-claim"))),
            "_finish_generation_job": Mock(), "_finish_ai_claim": Mock(), "generation_job": Mock(),
            "vector_memory": Mock(),
        }
        self.start = function("start_generation", self.namespace)
        self.message = SimpleNamespace(chat=SimpleNamespace(id=123, type="private"),
                                       from_user=SimpleNamespace(id=123, first_name="Synthetic"))
        self.data = dict(destination="Москва, Россия", destination_kind="city",
                         start_date="2099-06-01", end_date="2099-06-03", budget=60000, currency="RUB")

    def assert_prepaid_work_not_started(self):
        for name in ("_reserve_generation_job", "_claim_route_ai_request", "generation_job"):
            self.namespace[name].assert_not_called()
        self.namespace["db"].save_user.assert_not_called()
        self.namespace["threading"].Thread.assert_not_called()
        self.namespace["vector_memory"].recall_family_memories.assert_not_called()

    def test_moscow_homonyms_show_choices_without_quota_memory_or_automatic_restart(self):
        with patch.object(weather, "_get_json", return_value={"results": moscow_results()}) as provider:
            self.start(self.message, self.data)
        provider.assert_called_once()
        self.namespace["resolve_city_for_planning"].assert_called_once_with("Москва, Россия")
        self.assert_prepaid_work_not_started()
        text = self.namespace["bot"].send_message.call_args.args[1]
        self.assertIn("Москва, Москва, Россия", text)
        self.assertIn("Москва, Тверская область, Россия", text)
        self.assertIn("Попытка создания поездки не списана", text)
        self.assertIn("/plan", text)
        self.assertIn("автоматически не запускается", text)
        self.assertEqual(self.events, ["access", "resolve"])

    def test_missing_city_and_provider_outage_do_not_spend_or_start_work(self):
        for failure in (None, weather.httpx.ConnectError("private provider detail")):
            with self.subTest(failure=type(failure).__name__):
                self.namespace["bot"].reset_mock()
                with patch.object(weather, "_get_json", return_value={"results": []}, side_effect=failure):
                    self.start(self.message, self.data)
                self.assert_prepaid_work_not_started()
                text = self.namespace["bot"].send_message.call_args.args[1]
                self.assertIn("не списана", text)
                self.assertNotIn("private provider detail", text)

    def test_access_denied_is_checked_before_even_free_city_lookup(self):
        self.access.require_access.side_effect = self.access.AccessDeniedError("Доступ закрыт")
        self.start(self.message, self.data)
        self.namespace["resolve_city_for_planning"].assert_not_called()
        self.assert_prepaid_work_not_started()

    def test_explicit_moscow_is_resolved_once_before_claim_and_carried_to_worker(self):
        self.data["destination"] = "Москва, Москва, Россия"
        with patch.object(weather, "_get_json", return_value={"results": moscow_results()}) as provider:
            self.start(self.message, self.data)
        provider.assert_called_once()
        self.namespace["_claim_route_ai_request"].assert_called_once_with(123, "plan")
        self.assertLess(self.events.index("resolve"), self.events.index("claim"))
        thread = self.namespace["threading"].Thread.call_args.kwargs
        self.assertEqual(thread["kwargs"]["access_grant"], "synthetic-grant")
        resolved = thread["kwargs"]["resolved_city"]
        self.assertIsNotNone(resolved)
        self.namespace["threading"].Thread.return_value.start.assert_called_once()

        # Exercise the actual worker with isolated provider/storage mocks too.
        fixture = bot_fixtures.BotGenerationOutcomeTests()
        fixture.setUp()
        fixture.request = thread["args"][2]
        fixture.result.problems = ["Synthetic stop after planner invocation"]
        bot_fixtures._job(fixture.namespace)(*thread["args"], **thread["kwargs"])
        fixture.namespace["generate_itinerary"].assert_called_once()
        self.assertIs(fixture.namespace["generate_itinerary"].call_args.kwargs["resolved_city"], resolved)
        self.namespace["resolve_city_for_planning"].assert_called_once()
        fixture.namespace["db"].save_trip.assert_not_called()

    def test_regional_trip_preserves_existing_path_without_city_resolution(self):
        self.data.update(destination="Байкал, Россия", destination_kind="region")
        self.start(self.message, self.data)
        self.namespace["resolve_city_for_planning"].assert_not_called()
        self.namespace["_claim_route_ai_request"].assert_called_once_with(123, "plan")
        thread = self.namespace["threading"].Thread.call_args.kwargs
        self.assertEqual(thread["kwargs"], {"access_grant": "synthetic-grant"})
        self.namespace["threading"].Thread.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
