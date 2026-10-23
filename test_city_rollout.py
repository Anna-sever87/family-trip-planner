"""Off-by-default routing/disclosure with actual handlers and mocked services."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import city_rollout
import web_app
import web_service
import test_bot_generation_outcome as bot_fixture
import test_web_service_outcomes as web_fixture
from test_bot_city_edit_gate import function
from schemas import TripRequest


class CityRolloutTests(unittest.TestCase):
    def test_switch_accepts_only_literal_true_or_one_and_defaults_off(self):
        for value, expected in (("true", True), ("1", True), ("0", False), ("false", False),
                                ("TRUE", False), ("True", False), (" true ", False),
                                ("yes", False), ("on", False), ("", False), (None, False)):
            with self.subTest(value=value), patch.object(city_rollout.os, "getenv", return_value=value) as get:
                self.assertIs(city_rollout.city_candidates_enabled(), expected)
                get.assert_called_once_with("CITY_CANDIDATES_FIRST", "0")

    def test_me_capabilities_expose_only_boolean_and_public_disclosure(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch.object(web_app.db, "get_user", return_value={"name": "Test"}), \
                    patch.object(web_service, "city_candidates_enabled", return_value=enabled):
                result = web_app.me(123)
                self.assertEqual(result["capabilities"], {
                    "city_candidates_first": enabled,
                    "city_mvp_disclosure": city_rollout.CITY_MVP_DISCLOSURE if enabled else "",
                    "cover_images": True,
                })
                self.assertEqual(result["name"], "Test")
                self.assertEqual(result["generation_jobs"], {"enabled": False, "owner": ""})

    def test_web_new_city_opt_in_and_region_or_off_keep_previous_kwargs(self):
        for enabled in (False, True):
            for kind in ("city", "region"):
                fixture = web_fixture.WebServiceOutcomeTests()
                fixture.setUp()
                try:
                    fixture.request.destination_kind = kind
                    with self.subTest(enabled=enabled, kind=kind), \
                            patch.object(web_service, "city_candidates_enabled", return_value=enabled):
                        web_service.create_trip(123, fixture.request)
                        kwargs = fixture.generate.call_args.kwargs
                        self.assertEqual(kwargs["ground_candidates"], kind == "region")
                        self.assertEqual(kwargs.get("city_candidates_first", False), enabled and kind == "city")
                        if not (enabled and kind == "city"):
                            self.assertNotIn("city_candidates_first", kwargs)
                        fixture.generate.assert_called_once()
                        fixture.save.assert_called_once()
                finally:
                    fixture.doCleanups()

    def test_bot_new_city_opt_in_and_region_or_off_keep_previous_kwargs(self):
        for enabled in (False, True):
            for kind in ("city", "region"):
                with self.subTest(enabled=enabled, kind=kind):
                    fixture = bot_fixture.BotGenerationOutcomeTests()
                    fixture.setUp()
                    fixture.request.destination_kind = kind
                    fixture.namespace["city_candidates_enabled"].return_value = enabled
                    fixture.run_job()
                    kwargs = fixture.namespace["generate_itinerary"].call_args.kwargs
                    self.assertEqual(kwargs["ground_candidates"], kind == "region")
                    self.assertEqual(kwargs.get("city_candidates_first", False), enabled and kind == "city")
                    if not (enabled and kind == "city"):
                        self.assertNotIn("city_candidates_first", kwargs)

    def test_bot_shows_disclosure_before_final_wishes_only_for_enabled_city(self):
        for enabled, kind, step in ((True, "city", "wishes"), (False, "city", "wishes"),
                                    (True, "region", "wishes"), (True, "city", "dates")):
            with self.subTest(enabled=enabled, kind=kind, step=step):
                namespace = {"bot": Mock(), "states": {100: {"data": {"destination_kind": kind}}},
                             "QUESTIONS": {"wishes": "Пожелания?", "dates": "Даты?"},
                             "CITY_MVP_DISCLOSURE": city_rollout.CITY_MVP_DISCLOSURE,
                             "city_candidates_enabled": Mock(return_value=enabled)}
                function("ask", namespace)(100, step)
                text = namespace["bot"].send_message.call_args.args[1]
                shown = enabled and kind == "city" and step == "wishes"
                self.assertEqual(city_rollout.CITY_MVP_DISCLOSURE in text, shown)
                self.assertEqual("/cancel" in text, shown)

    def test_bot_start_disclosure_precedes_quota_and_thread_without_extra_confirmation(self):
        for enabled, kind in ((True, "city"), (False, "city"), (True, "region")):
            with self.subTest(enabled=enabled, kind=kind):
                events = []
                messenger = Mock()
                messenger.send_message.side_effect = lambda chat, text, **kw: events.append(("message", text))
                namespace = {"TripRequest": TripRequest, "bot": messenger, "db": Mock(), "types": Mock(),
                             "threading": Mock(), "_message_user_id": lambda message: 100,
                             "minimum_meal_cost": lambda request: 100,
                             "CITY_MVP_DISCLOSURE": city_rollout.CITY_MVP_DISCLOSURE,
                             "city_candidates_enabled": Mock(return_value=enabled),
                             "_reserve_generation_job": Mock(return_value=True),
                             "_claim_route_ai_request": Mock(side_effect=lambda *args: (
                                 events.append(("claim", None)) or SimpleNamespace(status="ok", claim_id="test"))),
                             "generation_job": Mock()}
                message = SimpleNamespace(chat=SimpleNamespace(id=100), from_user=SimpleNamespace(first_name="Test"))
                data = dict(destination="Тестовый город, Россия", destination_kind=kind,
                            start_date="2099-06-01", end_date="2099-06-02", budget=50000)
                function("start_generation", namespace)(message, data)
                texts = [text for event, text in events if event == "message"]
                self.assertEqual(city_rollout.CITY_MVP_DISCLOSURE in texts, enabled and kind == "city")
                if enabled and kind == "city":
                    self.assertLess(events.index(("message", city_rollout.CITY_MVP_DISCLOSURE)),
                                    events.index(("claim", None)))
                namespace["_claim_route_ai_request"].assert_called_once()
                namespace["threading"].Thread.return_value.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
