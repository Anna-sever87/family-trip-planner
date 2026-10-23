"""Candidate-first weather reuses one server-owned destination resolution."""
import unittest
from unittest.mock import patch

import weather


class WeatherResolvedCityTests(unittest.TestCase):
    def test_resolved_center_does_not_search_a_second_city(self):
        center = {"lat": 12.2, "lon": 109.1, "name": "Нячанг"}
        daily = {"time": ["2098-10-28"], "temperature_2m_min": [20],
                 "temperature_2m_max": [29], "precipitation_sum": [4]}
        with patch.object(weather, "find_city", side_effect=AssertionError("No repeated resolution")), \
             patch.object(weather, "_get_json", return_value={"daily": daily}) as get:
            result = weather.get_weather("Нячанг, Вьетнам", "2099-10-28", "2099-10-28", resolved_city=center)
        self.assertEqual((result["city"], result["source"]), ("Нячанг", "history"))
        self.assertEqual((get.call_args.kwargs["latitude"], get.call_args.kwargs["longitude"]), (12.2, 109.1))
        self.assertEqual(get.call_count, 1)

    def test_invalid_resolved_coordinates_fail_before_forecast(self):
        for bad in (True, float("nan"), float("inf"), 91, "55"):
            with self.subTest(bad=bad), patch.object(weather, "_get_json") as get, self.assertRaises(weather.CityResolutionError):
                weather.get_weather("Тест", "2099-10-28", "2099-10-28", resolved_city={"lat": bad, "lon": 0, "name": "Тест"})
            get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
