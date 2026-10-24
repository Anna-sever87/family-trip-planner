"""Provider-only city clarification facts, with synthetic offline responses.

Numbers, administrative labels and coordinates in these fixtures are test data,
not an embedded city catalogue or a statement of current population.
"""

from copy import deepcopy
import socket
import sqlite3
import unittest
from unittest.mock import patch

import weather
from test_city_resolution import city


def beijing_rows():
    return [
        {**city("Пекин", country="Китай", code="CN", admin="Пекин", identity=101,
                lat=39.9, lon=116.4, population=20_000_000), "feature_code": "PPLC"},
        {**city("Пекин", country="Китай", code="CN", admin="Шаньси", identity=102,
                lat=35.1, lon=111.0, population=2_000), "admin2": "Юньчэн"},
    ]


class CitySuggestionDetailsTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((weather.httpx, "get"), (socket.socket, "connect"),
                            (socket, "getaddrinfo"), (sqlite3, "connect")):
            boundary = patch.object(owner, name, side_effect=AssertionError(f"{name} forbidden"))
            boundary.start()
            self.addCleanup(boundary.stop)

    def ambiguous(self, rows=None):
        with patch.object(weather, "_get_json", return_value={"results": rows or beijing_rows()}) as get, \
                self.assertRaises(weather.CityResolutionError) as caught:
            weather.resolve_city_for_planning("Пекин, Китай")
        self.assertEqual(get.call_count, 1)
        self.assertEqual(caught.exception.reason, "ambiguous")
        return caught.exception

    def test_capital_and_other_region_are_described_without_changing_selection_values(self):
        error = self.ambiguous()
        self.assertEqual(error.suggestions, ("Пекин, Пекин, Китай", "Пекин, Шаньси, Китай"))
        self.assertEqual(error.suggestion_details, (
            {"label": "Пекин, Пекин, Китай", "feature_label": "Столица страны",
             "admin2": "", "population": 20_000_000},
            {"label": "Пекин, Шаньси, Китай", "feature_label": "Населённый пункт",
             "admin2": "Юньчэн", "population": 2_000},
        ))
        self.assertIn("Пекин, Пекин, Китай\n  Столица страны", str(error))
        self.assertIn("Административная единица: Юньчэн", str(error))
        self.assertIn("Население по справочнику: 20 000 000", str(error))

    def test_ranking_population_and_capital_status_never_select_a_city(self):
        for rows in (beijing_rows(), list(reversed(beijing_rows()))):
            rows[0]["population"] = 100_000_000
            with self.subTest(first=rows[0]["id"]):
                error = self.ambiguous(rows)
                self.assertEqual(error.suggestions[0], f"Пекин, {rows[0]['admin1']}, Китай")
                self.assertEqual(len(error.suggestions), 2)
                for label, identity in (("Пекин, Пекин, Китай", 101), ("Пекин, Шаньси, Китай", 102)):
                    with patch.object(weather, "_get_json", return_value={"results": rows}):
                        resolved = weather.resolve_city_for_planning(label)
                    self.assertEqual(weather.validated_planning_city(resolved, label)["geoname_id"], identity)

    def test_same_admin_homonyms_are_not_made_selectable_by_extra_details(self):
        rows = beijing_rows()
        rows[1]["admin1"] = rows[0]["admin1"]
        error = self.ambiguous(rows)
        self.assertEqual(error.suggestions, ())
        self.assertEqual(error.suggestion_details, ())
        self.assertNotIn("20 000 000", str(error))

    def test_missing_optional_facts_are_not_invented(self):
        rows = beijing_rows()
        for row in rows:
            for key in ("feature_code", "admin2", "population"):
                row.pop(key, None)
        error = self.ambiguous(rows)
        self.assertEqual(len(error.suggestions), 2)
        self.assertEqual(error.suggestion_details, ())
        self.assertNotIn("Столица", str(error))
        self.assertNotIn("Население", str(error))

    def test_geo_names_types_have_fixed_meaning_not_city_name_inference(self):
        mappings = {
            "PPLC": "Столица страны", "PPLA": "Административный центр региона",
            "PPLA2": "Административный центр (уровень 2)",
            "PPLA3": "Административный центр (уровень 3)",
            "PPLA4": "Административный центр (уровень 4)",
            "PPLA5": "Административный центр (уровень 5)",
            "PPLX": "Часть населённого пункта", "PPLCH": "Историческая столица",
            "PPLCD": "Административный центр зависимой территории",
        }
        for feature, expected in mappings.items():
            with self.subTest(feature=feature):
                rows = beijing_rows()
                rows[1]["feature_code"] = feature
                self.assertEqual(self.ambiguous(rows).suggestion_details[1]["feature_label"], expected)

    def test_unknown_or_empty_non_string_feature_never_becomes_a_display_claim(self):
        for feature in ("PPL_UNKNOWN", "PPL<script>", "", None, [], {}):
            with self.subTest(feature=feature):
                rows = beijing_rows()
                rows[0]["feature_code"] = feature
                detail = self.ambiguous(rows).suggestion_details[0]
                self.assertEqual(detail["feature_label"], "")
                self.assertEqual(detail["population"], 20_000_000)

    def test_population_requires_a_bounded_positive_integer_not_bool_or_numeric_string(self):
        for value in (True, False, 0, -1, 100_000_001, 2.5, "2000", float("nan"), None, {}, []):
            with self.subTest(population=value):
                rows = beijing_rows()
                rows[0]["population"] = value
                self.assertIsNone(self.ambiguous(rows).suggestion_details[0]["population"])

    def test_admin2_markup_controls_bidi_and_oversize_are_dropped(self):
        for value in ("<img src=x>", "private\nregion", "private\u202eregion", "X" * 101,
                      "https://private.invalid", {}, ["private"]):
            with self.subTest(admin2=value):
                rows = beijing_rows()
                rows[1]["admin2"] = value
                error = self.ambiguous(rows)
                self.assertEqual(error.suggestion_details[1]["admin2"], "")
                self.assertNotIn("private", str(error))

    def test_non_ambiguous_error_never_discloses_attached_suggestion_metadata(self):
        error = self.ambiguous()
        for reason in ("not_found", "invalid_context", "unavailable", "invalid_response", "search_limit"):
            with self.subTest(reason=reason):
                other = weather.CityResolutionError("Пекин, Китай", reason=reason,
                    suggestions=error.suggestions, suggestion_details=error.suggestion_details)
                self.assertEqual(other.suggestions, ())
                self.assertEqual(other.suggestion_details, ())

    def test_safe_helper_binds_exact_labels_ignores_unknown_keys_and_detaches_data(self):
        options = ("City, Region, Country",)
        details = [{"label": options[0], "feature_label": "Столица страны", "admin2": "Area",
                    "population": 1234, "latitude": 55.0, "geoname_id": "private", "secret": "private"},
                   {"label": " City, Region, Country", "feature_label": "Столица страны"},
                   {"label": "Other, Region, Country", "feature_label": "Столица страны"}]
        result = weather.safe_city_suggestion_details(details, options)
        self.assertEqual(result, ({"label": options[0], "feature_label": "Столица страны",
                                   "admin2": "Area", "population": 1234},))
        details[0]["admin2"] = "Changed"
        self.assertEqual(result[0]["admin2"], "Area")

    def test_safe_helper_sanitizes_even_persisted_or_forged_display_values(self):
        options = ("City, Region, Country",)
        result = weather.safe_city_suggestion_details([
            {"label": options[0], "feature_label": "Huge capital <b>", "admin2": "Area", "population": True}], options)
        self.assertEqual(result, ({"label": options[0], "feature_label": "", "admin2": "Area", "population": None},))
        for value in (None, {}, "private", [None, "private", {}, []]):
            self.assertEqual(weather.safe_city_suggestion_details(value, options), ())
        self.assertEqual(weather.safe_city_suggestion_details(result, ()), ())

    def test_safe_helper_preserves_option_order_caps_size_and_rejects_conflicting_duplicates(self):
        options = tuple(f"City{i}, Region, Country" for i in range(10))
        details = [{"label": label, "feature_label": "Столица страны", "population": 1} for label in options]
        result = weather.safe_city_suggestion_details(list(reversed(details)), options)
        self.assertEqual([item["label"] for item in result], list(options[:5]))
        duplicate = deepcopy(details[0])
        self.assertEqual(len(weather.safe_city_suggestion_details([details[0], duplicate], options)), 1)
        duplicate["population"] = 2
        self.assertEqual(weather.safe_city_suggestion_details([details[0], duplicate], options), ())

    def test_wrong_country_candidates_are_not_exposed_even_with_capital_feature(self):
        rows = beijing_rows()
        rows[0]["country_code"] = "US"
        rows[0]["country"] = "США"
        with patch.object(weather, "_get_json", return_value={"results": rows}):
            resolved = weather.resolve_city_for_planning("Пекин, Китай")
        self.assertEqual(weather.validated_planning_city(resolved, "Пекин, Китай")["geoname_id"], 102)


if __name__ == "__main__":
    unittest.main()
