"""Free server-owned geographic preflight; all provider responses are mocked."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
import unittest
from unittest.mock import Mock, patch

import city_planner
import planner
import weather
from schemas import TripRequest
from test_city_resolution import city


def moscow_rows():
    # Provider identities/admin labels from the reported ambiguity. No lookup.
    return [
        {**city("Москва", lat=55.75204, lon=37.61781, country="Россия", code="RU",
                admin="Москва", identity=524901), "feature_code": "PPLC"},
        city("Москва", lat=56.91775, lon=32.16579, country="Россия", code="RU",
             admin="Тверская Область", identity=524900),
    ]


class CityPlanningPreflightTests(unittest.TestCase):
    def setUp(self):
        blocked = patch.object(weather.httpx, "get", side_effect=AssertionError("No live HTTP"))
        blocked.start()
        self.addCleanup(blocked.stop)

    def resolve(self, destination="Москва, Москва, Россия", rows=None):
        with patch.object(weather, "_get_json", return_value={"results": rows or moscow_rows()}):
            return weather.resolve_city_for_planning(destination)

    def test_moscow_ambiguity_offers_both_verified_qualifiers_not_population_choice(self):
        for rows in (moscow_rows(), list(reversed(moscow_rows()))):
            rows[0]["population"] = 100_000_000
            with self.subTest(first=rows[0]["id"]), \
                    patch.object(weather, "_get_json", return_value={"results": rows}), \
                    self.assertRaises(weather.CityResolutionError) as caught:
                weather.resolve_city_for_planning("Москва, Россия")
            self.assertEqual(caught.exception.reason, "ambiguous")
            self.assertEqual(set(caught.exception.suggestions), {
                "Москва, Москва, Россия", "Москва, Тверская Область, Россия"})
            self.assertIn("Уточните город", str(caught.exception))

    def test_explicit_qualifier_reuses_correct_city(self):
        for destination, identity in (("Москва, Москва, Россия", 524901),
                                      ("Москва, Тверская Область, Россия", 524900)):
            resolved = self.resolve(destination)
            self.assertEqual(weather.validated_planning_city(resolved, destination)["geoname_id"], identity)

    def test_same_admin_homonyms_are_not_offered_as_resolved_choices(self):
        rows = moscow_rows()
        rows[1]["admin1"] = "Москва"
        with patch.object(weather, "_get_json", return_value={"results": rows}), \
                self.assertRaises(weather.CityResolutionError) as caught:
            weather.resolve_city_for_planning("Москва, Россия")
        self.assertEqual(caught.exception.suggestions, ())

    def test_singapore_and_vatican_need_no_invented_admin(self):
        for name, country, code, lat, lon in (("Singapore", "Singapore", "SG", 1.29, 103.85),
                                              ("Vatican City", "Vatican City", "VA", 41.90, 12.45)):
            with self.subTest(code=code):
                destination = f"{name}, {country}"
                resolved = self.resolve(destination, [city(name, lat=lat, lon=lon, country=country,
                                                           code=code, admin="", identity=1)])
                result = weather.validated_planning_city(resolved, destination)
                self.assertEqual((result["country_code"], result["admin1"]), (code, ""))

    def test_not_found_network_and_invalid_provider_json_have_distinct_reasons(self):
        cases = (({"results": []}, None, "not_found"),
                 (None, weather.httpx.ConnectError("private provider detail"), "unavailable"),
                 (None, ValueError("private malformed response"), "invalid_response"),
                 ([], None, "invalid_response"))
        for value, error, reason in cases:
            with self.subTest(reason=reason), \
                    patch.object(weather, "_get_json", return_value=value, side_effect=error), \
                    self.assertRaises(weather.CityResolutionError) as caught:
                weather.resolve_city_for_planning("Москва, Россия")
            self.assertEqual(caught.exception.reason, reason)
            self.assertNotIn("private", str(caught.exception))
            self.assertEqual(caught.exception.suggestions, ())

    def test_country_cutoff_and_conflicting_identity_stay_fail_closed(self):
        rows = moscow_rows()
        scenarios = [
            ([{**rows[0], "country_code": "US"}], "Москва, Москва, Россия", "context_mismatch"),
            ([rows[0], {**rows[0], "longitude": 38.0}], "Москва, Москва, Россия", "ambiguous"),
            ([rows[0]] + [{**rows[0], "name": f"Other{index}"} for index in range(9)],
             "Москва, Москва, Россия", "search_limit"),
        ]
        for response, destination, reason in scenarios:
            with self.subTest(reason=reason), patch.object(weather, "_get_json", return_value={"results": response}), \
                    self.assertRaises(weather.CityResolutionError) as caught:
                weather.resolve_city_for_planning(destination)
            self.assertEqual(caught.exception.reason, reason)

    def test_bound_context_is_immutable_and_returned_dict_is_detached(self):
        resolved = self.resolve()
        with self.assertRaises(FrozenInstanceError):
            resolved._destination = "Elsewhere"
        first = weather.validated_planning_city(resolved, "Москва, Москва, Россия")
        first["lat"] = 0
        self.assertEqual(weather.validated_planning_city(resolved, "Москва, Москва, Россия")["lat"], 55.75204)

    def test_dict_forgery_destination_replacement_and_record_tamper_are_rejected(self):
        resolved = self.resolve()
        modified = replace(resolved, _record=tuple((key, 0 if key == "lat" else value)
                                                  for key, value in resolved._record))
        for context, destination in (
                (weather.validated_planning_city(resolved, "Москва, Москва, Россия"), "Москва, Москва, Россия"),
                (resolved, "Москва, Тверская Область, Россия"),
                (replace(resolved, _destination="Elsewhere"), "Elsewhere"),
                (modified, "Москва, Москва, Россия")):
            with self.subTest(destination=destination), self.assertRaises(weather.CityResolutionError) as caught:
                weather.validated_planning_city(context, destination)
            self.assertEqual(caught.exception.reason, "invalid_context")

    def test_suggestions_are_plain_bounded_canonical_text(self):
        good = "Москва, Москва, Россия"
        bad = ["<b>Москва</b>, Москва, Россия", "Москва, Москва, Россия\nprivate", "Москва, Россия",
               "Москва,,Россия", "Москва, район, область, Россия", "M" * 101 + ", X, Y", None, {}]
        self.assertEqual(weather.safe_city_suggestions(bad + [good]), (good,))
        self.assertEqual(len(weather.safe_city_suggestions([f"City{i}, Province, Country" for i in range(10)])), 5)
        self.assertEqual(weather.safe_city_suggestions([good, good]), (good,))
        self.assertEqual(weather.safe_city_suggestions(good), ())

    def test_provider_markup_and_control_chars_never_reach_suggestions(self):
        rows = moscow_rows()
        rows[0]["admin1"] = "<script>private</script>"
        rows[1]["admin1"] = "private\nregion"
        with patch.object(weather, "_get_json", return_value={"results": rows}), \
                self.assertRaises(weather.CityResolutionError) as caught:
            weather.resolve_city_for_planning("Москва, Россия")
        self.assertEqual(caught.exception.suggestions, ())
        self.assertNotIn("private", str(caught.exception))

    def test_city_pipeline_uses_one_preflight_for_weather_and_candidate_center(self):
        class ReachedGrounding(Exception):
            pass

        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        request = TripRequest(destination="Москва, Москва, Россия", start_date=tomorrow,
                              end_date=tomorrow, budget=50000)
        daily = {"time": [tomorrow], "temperature_2m_min": [8],
                 "temperature_2m_max": [12], "precipitation_sum": [0]}
        with patch.object(weather, "_get_json", side_effect=[{"results": moscow_rows()}, {"daily": daily}]) as get:
            resolved = weather.resolve_city_for_planning(request.destination)
            with patch.object(city_planner, "find_city", side_effect=AssertionError("No second geocoder")), \
                    patch.object(planner, "find_city", side_effect=AssertionError("No legacy geocoder")), \
                    patch.object(city_planner, "ground_city_candidates", side_effect=ReachedGrounding) as ground, \
                    patch.object(planner, "generate_json") as completion, \
                    self.assertRaises(ReachedGrounding):
                planner.generate_itinerary(request, city_candidates_first=True, resolved_city=resolved)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args_list[0].args[0], weather.GEOCODING_URL)
        self.assertEqual(get.call_args_list[1].args[0], weather.FORECAST_URL)
        self.assertEqual(ground.call_args.args[1]["geoname_id"], 524901)
        completion.assert_not_called()

    def test_legacy_city_branch_reuses_preflight_for_weather_and_center(self):
        class ReachedPrompt(Exception):
            pass

        request = TripRequest(destination="Москва, Москва, Россия", start_date="2099-11-02",
                              end_date="2099-11-02", budget=50000)
        resolved = self.resolve(request.destination)
        with patch.object(planner, "get_weather", return_value=None) as forecast, \
                patch.object(planner, "find_city", side_effect=AssertionError("No second geocoder")), \
                patch.object(planner, "cached_place_hints", return_value=[]) as hints, \
                patch.object(planner, "curated_place_hints", return_value=[]), \
                patch.object(planner, "_load_blocked_places", return_value=[]), \
                patch.object(planner, "build_itinerary_prompt", side_effect=ReachedPrompt), \
                self.assertRaises(ReachedPrompt):
            planner.generate_itinerary(request, resolved_city=resolved)
        self.assertEqual(forecast.call_args.kwargs["resolved_city"]["geoname_id"], 524901)
        self.assertEqual(hints.call_args.args[1], (55.75204, 37.61781))

    def test_planner_rejects_wrong_destination_or_region_before_any_dependency(self):
        resolved = self.resolve()
        for kind, destination in (("city", "Москва, Тверская Область, Россия"), ("region", "Москва, Москва, Россия")):
            request = TripRequest(destination=destination, destination_kind=kind, start_date="2099-11-02",
                                  end_date="2099-11-02", budget=50000)
            with self.subTest(kind=kind), patch.object(planner, "get_weather") as forecast, \
                    patch.object(planner, "generate_json") as completion, \
                    self.assertRaises(weather.CityResolutionError):
                planner.generate_itinerary(request, resolved_city=resolved)
            forecast.assert_not_called()
            completion.assert_not_called()


if __name__ == "__main__":
    unittest.main()
