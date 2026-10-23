"""Weather dates/values are provider evidence, not positional placeholders."""

from copy import deepcopy
from datetime import date
import socket
import sqlite3
import unittest
from unittest.mock import patch

import weather


class FixedDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 10, 21)


CITY = {"lat": 43.6, "lon": 39.7, "name": "Тестовый город"}


def response(times, *, rain=None):
    size = len(times)
    return {"daily": {"time": times, "temperature_2m_min": list(range(10, 10 + size)),
                      "temperature_2m_max": list(range(20, 20 + size)),
                      "precipitation_sum": list(range(size)) if rain is None else rain}}


class WeatherResponseIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.patch(weather, "date", new=FixedDate)
        self.find = self.patch(weather, "find_city", return_value=deepcopy(CITY))
        self.get = self.patch(weather, "_get_json")
        for owner, name in ((weather.httpx, "get"), (socket.socket, "connect"),
                            (socket, "getaddrinfo"), (sqlite3, "connect")):
            self.patch(owner, name, side_effect=AssertionError("No external I/O"))

    def patch(self, owner, name, **kwargs):
        handle = patch.object(owner, name, **kwargs)
        self.addCleanup(handle.stop)
        return handle.start()

    def fetch(self, data, start="2026-10-28", end="2026-10-29", **kwargs):
        self.get.return_value = data
        return weather.get_weather("Тестовый город, Россия", start, end, **kwargs)

    def test_valid_forecast_uses_exact_provider_calendar_and_preserves_real_zero(self):
        result = self.fetch(response(["2026-10-28", "2026-10-29"], rain=[0, 2.5]))
        self.assertEqual(result["source"], "forecast")
        self.assertEqual([day["date"] for day in result["days"]], ["2026-10-28", "2026-10-29"])
        self.assertEqual([day["rain_mm"] for day in result["days"]], [0, 2.5])
        self.get.assert_called_once()
        self.assertEqual(self.get.call_args.args, (weather.FORECAST_URL,))

    def test_shifted_reversed_duplicate_missing_and_extra_provider_dates_are_rejected(self):
        for times in (["2026-10-29", "2026-10-30"], ["2026-10-29", "2026-10-28"],
                      ["2026-10-28", "2026-10-28"], ["2026-10-28"], [],
                      ["2026-10-28", "2026-10-29", "2026-10-30"], ["20261028", "20261029"],
                      ["2026-10-28T00:00", "2026-10-29T00:00"], [None, "2026-10-29"]):
            with self.subTest(times=times):
                self.assertIsNone(self.fetch(response(times)))

    def test_null_or_missing_rain_is_not_converted_to_dry_weather(self):
        for rain in ([None, 0], [0, None]):
            with self.subTest(rain=rain):
                self.assertIsNone(self.fetch(response(["2026-10-28", "2026-10-29"], rain=rain)))
        value = response(["2026-10-28", "2026-10-29"])
        del value["daily"]["precipitation_sum"]
        self.assertIsNone(self.fetch(value))

    def test_numeric_fields_reject_bool_strings_null_nonfinite_and_rain_below_zero(self):
        for field in ("temperature_2m_min", "temperature_2m_max", "precipitation_sum"):
            for invalid in (True, False, "0", None, float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field, invalid=invalid):
                    value = response(["2026-10-28", "2026-10-29"])
                    value["daily"][field][0] = invalid
                    self.assertIsNone(self.fetch(value))
        self.assertIsNone(self.fetch(response(["2026-10-28", "2026-10-29"], rain=[-0.1, 0])))

    def test_inverted_temperature_range_rejected_but_freezing_temperatures_valid(self):
        value = response(["2026-10-28", "2026-10-29"])
        value["daily"]["temperature_2m_min"][0] = 30
        self.assertIsNone(self.fetch(value))
        value["daily"]["temperature_2m_min"] = [-20, -10]
        value["daily"]["temperature_2m_max"] = [-10, -10]
        self.assertIsNotNone(self.fetch(value))

    def test_all_arrays_must_be_lists_with_identical_expected_lengths(self):
        for field in ("time", "temperature_2m_min", "temperature_2m_max", "precipitation_sum"):
            for replacement in (None, "wrong", {}, [], [0], [0, 1, 2], (0, 1)):
                with self.subTest(field=field, replacement=replacement):
                    value = response(["2026-10-28", "2026-10-29"])
                    value["daily"][field] = replacement
                    self.assertIsNone(self.fetch(value))

    def test_malformed_response_containers_fail_soft_without_retry(self):
        for value in (None, [], "wrong", {}, {"daily": None}, {"daily": []}, {"daily": "wrong"}):
            with self.subTest(value=value):
                before = self.get.call_count
                self.assertIsNone(self.fetch(value))
                self.assertEqual(self.get.call_count - before, 1)

    def test_reversed_request_range_fails_before_any_request(self):
        self.assertIsNone(self.fetch({}, "2026-10-29", "2026-10-28"))
        self.get.assert_not_called()
        self.find.assert_not_called()

    def test_fact_calendar_including_leap_day_is_not_relabelled(self):
        times = ["2024-02-28", "2024-02-29", "2024-03-01"]
        result = self.fetch(response(times), times[0], times[-1])
        self.assertEqual(result["source"], "fact")
        self.assertEqual([item["date"] for item in result["days"]], times)
        self.assertEqual(self.get.call_args.args, (weather.ARCHIVE_URL,))

    def test_forecast_spanning_past_and_today_requires_only_requested_forecast_dates(self):
        result = self.fetch(response(["2026-10-21", "2026-10-22"]), "2026-10-20", "2026-10-22")
        self.assertEqual([item["date"] for item in result["days"]], ["2026-10-21", "2026-10-22"])
        self.assertEqual(self.get.call_args.kwargs["start_date"], "2026-10-21")
        self.assertIsNone(self.fetch(response(["2026-10-20", "2026-10-21"]), "2026-10-20", "2026-10-22"))

    def test_history_maps_matching_month_day_not_leap_year_array_index(self):
        data = response(["2028-02-28", "2028-02-29", "2028-03-01"], rain=[1, 22, 3])
        result = self.fetch(data, "2029-02-28", "2029-03-01")
        self.assertEqual(result["source"], "history")
        self.assertEqual([(item["date"], item["rain_mm"]) for item in result["days"]],
                         [("2029-02-28", 1), ("2029-03-01", 3)])

    def test_history_leap_target_day_uses_existing_feb28_analogue_without_shifting_march(self):
        result = self.fetch(response(["2027-02-28", "2027-03-01"], rain=[2, 9]), "2028-02-28", "2028-03-01")
        self.assertEqual([(item["date"], item["rain_mm"]) for item in result["days"]],
                         [("2028-02-28", 2), ("2028-02-29", 2), ("2028-03-01", 9)])
        self.assertEqual(result["source"], "history")

    def test_history_wrong_year_is_rejected(self):
        self.assertIsNone(self.fetch(response(["2029-02-28", "2029-03-01"]), "2029-02-28", "2029-03-01"))

    def test_resolved_city_still_avoids_new_geocode_and_input_data_is_unchanged(self):
        data = response(["2026-10-28", "2026-10-29"])
        before = deepcopy(data)
        self.assertIsNotNone(self.fetch(data, resolved_city=CITY))
        self.find.assert_not_called()
        self.get.assert_called_once()
        self.assertEqual(data, before)

    def test_city_resolution_error_stays_typed_and_is_not_swallowed(self):
        error = weather.CityResolutionError("Неоднозначный город", reason="ambiguous")
        self.find.side_effect = error
        with self.assertRaises(weather.CityResolutionError) as caught:
            self.fetch(response(["2026-10-28", "2026-10-29"]))
        self.assertIs(caught.exception, error)
        self.get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
