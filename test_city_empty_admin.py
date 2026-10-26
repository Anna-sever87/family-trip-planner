"""Empty admin lookup recovery offers choices, never silently relocates a city.

Provider spellings reflect a reviewed free response; these tests mock every
lookup. Coordinates/IDs are fixture data, not a runtime city catalogue.
"""
from copy import deepcopy
import socket
import sqlite3
import unittest
from unittest.mock import patch

import weather


def beijing_rows():
    return [
        {"id": 1816670, "name": "Пекин", "country": "Китай", "country_code": "CN",
         "admin1": "Beijing Shi", "admin2": "Beijing", "feature_code": "PPLC",
         "population": 18960744, "latitude": 39.9075, "longitude": 116.39723},
        {"id": 1816671, "name": "Пекин", "country": "Китай", "country_code": "CN",
         "admin1": "Шаньси", "admin2": "Yuncheng Shi", "feature_code": "PPL",
         "population": None, "latitude": 35.20917, "longitude": 110.73278},
    ]


class EmptyAdminClarificationTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((weather.httpx, "get"), (socket.socket, "connect"),
                            (socket, "getaddrinfo"), (sqlite3, "connect")):
            blocked = patch.object(owner, name, side_effect=AssertionError("No network or database"))
            blocked.start()
            self.addCleanup(blocked.stop)

    def rejected(self, responses, reason="ambiguous", destination="Пекин, Пекин, Китай"):
        with patch.object(weather, "_get_json", side_effect=responses) as get, \
                self.assertRaises(weather.CityResolutionError) as caught:
            weather.resolve_city_for_planning(destination)
        self.assertEqual(caught.exception.reason, reason)
        return caught.exception, get

    def test_observed_provider_admin_is_offered_with_capital_fact_not_guessed(self):
        rows = beijing_rows()
        before = deepcopy(rows)
        error, get = self.rejected([{}, {"results": rows}])
        self.assertEqual(get.call_count, 2)
        self.assertEqual([call.kwargs["name"] for call in get.call_args_list], ["Пекин, Пекин", "Пекин"])
        # The reviewed China alias now uses the provider's dedicated ISO
        # country filter on both queries; the explicit admin is not dropped
        # from the first lookup, and alternatives still require user choice.
        self.assertTrue(all(call.kwargs["countryCode"] == "CN" for call in get.call_args_list))
        self.assertTrue(all(call.kwargs["count"] == weather.MAX_CITY_RESULTS for call in get.call_args_list))
        self.assertEqual(error.suggestions, ("Пекин, Beijing Shi, Китай", "Пекин, Шаньси, Китай"))
        self.assertEqual(error.suggestion_details[0]["feature_label"], "Столица страны")
        self.assertEqual(error.suggestion_details[0]["admin2"], "Beijing")
        self.assertNotIn("Пекин, Пекин, Китай", error.suggestions)
        self.assertEqual(rows, before)

    def test_even_single_alternative_requires_an_explicit_new_choice(self):
        error, get = self.rejected([{"results": []}, {"results": beijing_rows()[:1]}])
        self.assertEqual(error.suggestions, ("Пекин, Beijing Shi, Китай",))
        self.assertEqual(get.call_count, 2)

    def test_explicit_canonical_choice_resolves_in_one_strict_lookup(self):
        destination = "Пекин, Beijing Shi, Китай"
        with patch.object(weather, "_get_json", return_value={"results": beijing_rows()[:1]}) as get:
            context = weather.resolve_city_for_planning(destination)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(get.call_args.kwargs["name"], "Пекин, Beijing Shi")
        self.assertEqual(weather.validated_planning_city(context, destination)["geoname_id"], 1816670)
        with self.assertRaises(weather.CityResolutionError):
            weather.validated_planning_city(context, "Пекин, Пекин, Китай")

    def test_existing_nonempty_context_mismatch_never_discards_admin(self):
        error, get = self.rejected([{"results": beijing_rows()}], "context_mismatch")
        self.assertEqual(get.call_count, 1)
        self.assertEqual(error.suggestions, ())

    def test_wrong_country_wrong_name_type_and_coordinates_are_not_suggestions(self):
        mutations = ({"country": "Россия", "country_code": "RU"}, {"name": "Пекинский район"},
                     {"feature_code": "ADM1"}, {"latitude": True}, {"longitude": float("nan")})
        for change in mutations:
            with self.subTest(change=change):
                error, get = self.rejected([{}, {"results": [{**beijing_rows()[0], **change}]}], "not_found")
                self.assertEqual(error.suggestions, ())
                self.assertEqual(get.call_count, 2)

    def test_known_country_iso_filter_is_retained_for_clarification(self):
        rows = [{**beijing_rows()[0], "name": "Город", "country": "Россия", "country_code": "RU",
                 "admin1": "Точный регион"}]
        error, get = self.rejected([{}, {"results": rows}], destination="Город, Другой регион, Россия")
        self.assertEqual(error.suggestions, ("Город, Точный регион, Россия",))
        self.assertEqual(get.call_args_list[1].kwargs, {"name": "Город", "count": weather.MAX_CITY_RESULTS,
                                                       "language": "ru", "countryCode": "RU"})

    def test_same_admin_homonyms_and_conflicting_coordinates_are_not_selectable(self):
        rows = [beijing_rows()[0], {**beijing_rows()[0], "longitude": 116.5}]
        error, get = self.rejected([{}, {"results": rows}])
        self.assertEqual(error.suggestions, ())
        self.assertEqual(error.suggestion_details, ())
        self.assertEqual(get.call_count, 2)

    def test_truncated_alternative_search_is_not_treated_as_unique(self):
        rows = [beijing_rows()[0]] + [{**beijing_rows()[0], "name": "Other"}
                                    for _ in range(weather.MAX_CITY_RESULTS - 1)]
        error, get = self.rejected([{}, {"results": rows}], "search_limit")
        self.assertEqual(error.suggestions, ())
        self.assertEqual(get.call_count, 2)

    def test_lookup_failure_and_malformed_fallback_do_not_authorize_more_search(self):
        for response, reason in (([], "invalid_response"), ({"results": "wrong"}, "invalid_response"),
                                 (weather.httpx.ConnectError("private"), "unavailable")):
            with self.subTest(reason=reason):
                error, get = self.rejected([{}, response], reason)
                self.assertEqual(error.suggestions, ())
                self.assertEqual(get.call_count, 2)
                self.assertNotIn("private", str(error))

    def test_two_empty_lookups_stop_without_loop_or_relaxed_match(self):
        error, get = self.rejected([{}, {}], "not_found")
        self.assertEqual(get.call_count, 2)
        self.assertEqual(error.suggestions, ())

    def test_unqualified_or_country_only_request_does_not_add_lookup(self):
        for destination in ("Пекин", "Пекин, Китай"):
            with self.subTest(destination=destination):
                _, get = self.rejected([{}], "not_found", destination)
                self.assertEqual(get.call_count, 1)

    def test_total_http_budget_is_at_most_four_sends_with_existing_retries(self):
        def response(code, payload):
            return weather.httpx.Response(code, json=payload, request=weather.httpx.Request("GET", weather.GEOCODING_URL))

        responses = [response(503, {}), response(200, {}), response(503, {}),
                     response(200, {"results": beijing_rows()})]
        with patch.object(weather.httpx, "get", side_effect=responses) as get, \
                patch.object(weather.time, "sleep") as sleep, \
                self.assertRaises(weather.CityResolutionError) as caught:
            weather.resolve_city_for_planning("Пекин, Пекин, Китай")
        self.assertEqual(caught.exception.reason, "ambiguous")
        self.assertEqual(get.call_count, 4)
        self.assertEqual(sleep.call_count, 2)

    def test_first_query_service_failure_does_not_trigger_fallback(self):
        with patch.object(weather.httpx, "get", side_effect=weather.httpx.ConnectError("private")) as get, \
                patch.object(weather.time, "sleep"), self.assertRaises(weather.CityResolutionError) as caught:
            weather.resolve_city_for_planning("Пекин, Пекин, Китай")
        self.assertEqual(caught.exception.reason, "unavailable")
        self.assertEqual(get.call_count, weather.MAX_HTTP_ATTEMPTS)
        self.assertEqual(caught.exception.suggestions, ())


if __name__ == "__main__":
    unittest.main()
