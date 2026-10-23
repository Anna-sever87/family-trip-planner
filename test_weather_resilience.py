"""Bounded weather transport retries and explicit unknown-state presentation."""
import socket
import sqlite3
import unittest
from unittest.mock import patch

import httpx
import weather


class WeatherResilienceTests(unittest.TestCase):
    def setUp(self):
        self.patches = []
        for owner, name in ((socket.socket, "connect"), (socket, "getaddrinfo"), (sqlite3, "connect")):
            self.mock(owner, name, side_effect=AssertionError("No external I/O"))
        self.get = self.mock(httpx, "get")
        self.sleep = self.mock(weather.time, "sleep")

    def mock(self, owner, name, **kwargs):
        handle = patch.object(owner, name, **kwargs)
        self.addCleanup(handle.stop)
        return handle.start()

    def response(self, status=200, *, headers=None, content=None):
        return httpx.Response(status, request=httpx.Request("GET", weather.FORECAST_URL),
                              headers=headers, json={"ok": True} if content is None else content)

    def test_timeout_and_transient_status_retry_once_then_succeed(self):
        for first in (httpx.ConnectTimeout("offline"), httpx.ConnectError("offline"),
                      self.response(429), self.response(503), self.response(504)):
            with self.subTest(first=first):
                self.get.reset_mock(); self.sleep.reset_mock()
                self.get.side_effect = [first, self.response()]
                self.assertEqual(weather._get_json(weather.FORECAST_URL, name="city"), {"ok": True})
                self.assertEqual(self.get.call_count, 2)
                self.assertEqual(self.get.call_args_list[0], self.get.call_args_list[1])
                self.sleep.assert_called_once_with(weather.RETRY_DELAY_SECONDS)

    def test_exhaustion_never_uses_third_request(self):
        self.get.side_effect = httpx.ReadTimeout("offline")
        with self.assertRaises(httpx.ReadTimeout):
            weather._get_json(weather.FORECAST_URL)
        self.assertEqual(self.get.call_count, 2)

    def test_nontransient_http_errors_are_not_retried(self):
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.get.reset_mock(); self.sleep.reset_mock()
                self.get.return_value = self.response(status)
                with self.assertRaises(httpx.HTTPStatusError):
                    weather._get_json(weather.FORECAST_URL)
                self.get.assert_called_once(); self.sleep.assert_not_called()

    def test_bad_json_is_not_a_network_retry(self):
        self.get.return_value = httpx.Response(200, request=httpx.Request("GET", weather.FORECAST_URL), content=b"not-json")
        with self.assertRaises(ValueError):
            weather._get_json(weather.FORECAST_URL)
        self.get.assert_called_once(); self.sleep.assert_not_called()

    def test_short_retry_after_is_respected_and_long_or_ambiguous_is_not_ignored(self):
        self.get.side_effect = [self.response(429, headers={"Retry-After": "1.5"}), self.response()]
        weather._get_json(weather.FORECAST_URL)
        self.sleep.assert_called_once_with(1.5)
        for value in ("60", "-1", "NaN", "Thu, 22 Oct 2026 10:00:00 GMT", "wrong"):
            with self.subTest(value=value):
                self.get.reset_mock(); self.sleep.reset_mock()
                self.get.side_effect = None
                self.get.return_value = self.response(429, headers={"Retry-After": value})
                with self.assertRaises(httpx.HTTPStatusError):
                    weather._get_json(weather.FORECAST_URL)
                self.get.assert_called_once(); self.sleep.assert_not_called()

    def test_unknown_weather_is_explicit_not_zero_rain_or_historical_forecast(self):
        for value in (None, {}, {"days": []}):
            self.assertEqual(weather.weather_summary(value), weather.UNKNOWN_WEATHER_NOTE)
        self.assertIn("не означает отсутствие дождя", weather.UNKNOWN_WEATHER_NOTE)
        known = {"city": "Test", "source": "history", "days": [{"date": "2027-10-28", "tmin": 1, "tmax": 2, "rain_mm": 0}]}
        self.assertIn("прошлого года", weather.weather_summary(known))
        self.assertNotIn("Погода неизвестна", weather.weather_summary(known))


if __name__ == "__main__":
    unittest.main()
