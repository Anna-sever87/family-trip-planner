"""Pure candidate-ID schedule contracts; model text cannot replace identity."""

from copy import deepcopy
import json
import socket
import sqlite3
import unittest
from unittest.mock import patch

from city_candidates import CityCandidate as Candidate
from city_response import build_city_response_schema
from city_schedule import CITY_CANDIDATE_SCHEDULE_RULE, build_city_schedule_schema, decode_city_schedule
import planner
from rules import check_rules
from schemas import Itinerary
from test_city_response import city_request, city_wire


def candidates_for(count=3):
    return [Candidate(f"c{number}", f"Музей ремёсел {number}",
                      f"Музей ремёсел {number}, Тестовый город, Россия", "Центр",
                      {"lat": 50 + number / 100, "lon": 100, "verified": True,
                       "source_schedule": True, "indoor": True})
            for number in range(1, count + 1)]


def candidate_wire(request):
    payload = city_wire(request)
    for number in range(1, request.days_count + 1):
        for attraction in payload[f"day_{number}"]["attractions"]:
            for field in ("title", "place_query", "area", "kind"):
                attraction.pop(field)
            attraction["candidate_id"] = f"c{number}"
    return payload


class CityScheduleTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "getaddrinfo"), (sqlite3, "connect")):
            blocker = patch.object(owner, name, side_effect=AssertionError("No external I/O"))
            blocker.start()
            self.addCleanup(blocker.stop)
        self.request = city_request()
        self.canonical = planner.build_itinerary_json_schema(self.request)
        self.candidates = candidates_for()
        self.schema = build_city_schedule_schema(self.canonical, self.request, self.candidates)
        self.payload = candidate_wire(self.request)

    def decode(self, payload=None, *, schema=None, candidates=None, canonical=None):
        return json.loads(decode_city_schedule(
            json.dumps(self.payload if payload is None else payload, ensure_ascii=False),
            self.schema if schema is None else schema,
            self.candidates if candidates is None else candidates,
            canonical_schema=self.canonical if canonical is None else canonical))

    def test_schema_has_only_ids_and_model_estimates_in_attractions_without_mutating_inputs(self):
        before = deepcopy((self.canonical, self.candidates))
        schema = build_city_schedule_schema(self.canonical, self.request, self.candidates)
        self.assertEqual((self.canonical, self.candidates), before)
        for number in (1, 2):
            properties = schema["properties"][f"day_{number}"]["properties"]
            attraction = properties["attractions"]["items"]
            self.assertEqual(set(attraction["properties"]),
                             {"candidate_id", "time", "duration_min", "cost", "description", "kids_note", "indoor"})
            self.assertEqual(attraction["properties"]["candidate_id"]["enum"], ["c1", "c2", "c3"])
            self.assertEqual(set(attraction["required"]), set(attraction["properties"]))
            self.assertFalse(attraction["additionalProperties"])
            for key in ("base_area", "transfer_note", "rainy_alternative"):
                self.assertEqual(properties[key]["const"], "")
            self.assertEqual(properties["day"]["const"], number)
            self.assertEqual(properties["date"]["const"], f"2099-06-0{number}")

    def test_existing_city_limits_and_non_identity_fields_are_preserved(self):
        original = build_city_response_schema(self.canonical, self.request)
        for key in ("day_1", "day_2"):
            before = original["properties"][key]["properties"]
            after = self.schema["properties"][key]["properties"]
            for field in ("meal", "rest", "day", "date", "theme"):
                self.assertEqual(after[field], before[field])
            self.assertEqual((after["attractions"]["minItems"], after["attractions"]["maxItems"]), (1, 3))
            for field in ("time", "duration_min", "cost", "description", "kids_note", "indoor"):
                self.assertEqual(after["attractions"]["items"]["properties"][field], before["attractions"]["items"]["properties"][field])

    def test_decode_restores_exact_identity_and_preserves_estimates_without_verified_fields(self):
        self.payload["day_1"]["attractions"][0].update(time="16:00", cost=123.5, indoor=False)
        before = deepcopy((self.payload, self.schema, self.canonical, self.candidates))
        result = self.decode()
        first = result["days"][0]["items"][-1]
        self.assertEqual(first, {**{key: value for key, value in self.payload["day_1"]["attractions"][0].items() if key != "candidate_id"},
                                 "title": self.candidates[0].title, "place_query": self.candidates[0].place_query,
                                 "area": self.candidates[0].area, "kind": "attraction"})
        itinerary = Itinerary.model_validate(result)
        self.assertFalse(itinerary.source_schedule)
        self.assertIsNone(itinerary.program_report)
        self.assertTrue(all(not item.verified and item.lat is None and item.lon is None
                            for day in itinerary.days for item in day.items))
        self.assertFalse(first["indoor"])
        self.assertEqual((self.payload, self.schema, self.canonical, self.candidates), before)

    def test_unknown_and_globally_repeated_ids_are_rejected(self):
        for identifier in ("unknown", "", True, 1, "c1"):
            payload = deepcopy(self.payload)
            payload["day_2"]["attractions"][0]["candidate_id"] = identifier
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                self.decode(payload)
        payload = deepcopy(self.payload)
        payload["day_1"]["attractions"] *= 2
        with self.assertRaises(ValueError):
            self.decode(payload)

    def test_duplicate_missing_empty_or_invalid_candidate_pool_fails_closed(self):
        for candidates in ([], self.candidates[:1], self.candidates + [deepcopy(self.candidates[0])]):
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                build_city_schedule_schema(self.canonical, self.request, candidates)
        for field, value in (("candidate_id", ""), ("candidate_id", 1), ("title", " "),
                             ("place_query", ""), ("area", ""), ("record", None)):
            candidates = deepcopy(self.candidates)
            setattr(candidates[0], field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_city_schedule_schema(self.canonical, self.request, candidates)
            with self.subTest(decoder_field=field), self.assertRaises(ValueError):
                self.decode(candidates=candidates)
        with self.assertRaises(ValueError):
            self.decode(candidates=self.candidates[:2])

    def test_candidate_fields_must_fit_current_canonical_bounds_without_truncation(self):
        for field, length in (("title", 81), ("place_query", 141), ("area", 61)):
            candidates = deepcopy(self.candidates)
            setattr(candidates[0], field, "Я" * length)
            before = deepcopy(candidates)
            with self.subTest(field=field), self.assertRaises(ValueError):
                build_city_schedule_schema(self.canonical, self.request, candidates)
            with self.subTest(decoder_field=field), self.assertRaises(ValueError):
                self.decode(candidates=candidates)
            self.assertEqual(candidates, before)
        canonical = deepcopy(self.canonical)
        canonical["properties"]["days"]["items"]["properties"]["items"]["items"]["properties"]["title"]["maxLength"] = 3
        with self.assertRaises(ValueError):
            build_city_schedule_schema(canonical, self.request, self.candidates)

    def test_canonical_validation_is_required_and_checks_restored_payload(self):
        canonical = deepcopy(self.canonical)
        canonical["properties"]["days"]["items"]["properties"]["items"]["items"]["properties"]["description"]["maxLength"] = 2
        with self.assertRaises(ValueError):
            self.decode(canonical=canonical)
        with self.assertRaises(TypeError):
            decode_city_schedule(json.dumps(self.payload), self.schema, self.candidates)

    def test_plain_json_schema_roundtrip_needs_no_hidden_metadata(self):
        schema = json.loads(json.dumps(self.schema, ensure_ascii=False))
        canonical = json.loads(json.dumps(self.canonical, ensure_ascii=False))
        self.assertEqual(self.decode(schema=schema, canonical=canonical), self.decode())

    def test_model_cannot_inject_identity_kind_trust_or_candidate_ids_into_meal_rest(self):
        for field, value in (("title", "Обед"), ("place_query", "Кафе"), ("area", "Чужой город"),
                             ("kind", "meal"), ("verified", True), ("lat", 50),
                             ("source_schedule", True), ("provider", "osm"), ("source_url", "https://example.org")):
            payload = deepcopy(self.payload)
            payload["day_1"]["attractions"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.decode(payload)
        for field in ("meal", "rest"):
            payload = deepcopy(self.payload)
            payload["day_1"][field]["candidate_id"] = "c1"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.decode(payload)
        payload = deepcopy(self.payload)
        payload["day_1"]["attractions"] = [deepcopy(payload["day_1"]["meal"])]
        with self.assertRaises(ValueError):
            self.decode(payload)

    def test_exact_days_empty_city_fields_and_separate_slots_remain_required(self):
        for field, value in (("day", 2), ("date", "2099-06-02"), ("base_area", "Город"),
                             ("transfer_note", "Переезд"), ("rainy_alternative", "Другой музей"),
                             ("attractions", []), ("rest", [])):
            payload = deepcopy(self.payload)
            payload["day_1"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.decode(payload)
        payload = deepcopy(self.payload)
        payload.pop("day_2")
        with self.assertRaises(ValueError):
            self.decode(payload)

    def test_meal_minimum_rest_minimum_and_visit_maximum_remain_enforced(self):
        for field, value in (("meal", 1799), ("rest", 59)):
            payload = deepcopy(self.payload)
            payload["day_1"][field]["cost" if field == "meal" else "duration_min"] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.decode(payload)
        payload = deepcopy(self.payload)
        payload["day_1"]["attractions"] *= 4
        with self.assertRaises(ValueError):
            self.decode(payload)

    def test_duplicate_json_keys_nonfinite_numbers_and_coercions_are_rejected(self):
        raw = json.dumps(self.payload, ensure_ascii=False)
        duplicate = raw.replace('"candidate_id": "c1"', '"candidate_id": "c1", "candidate_id": "c2"', 1)
        self.assertNotEqual(raw, duplicate)
        for broken in (duplicate, raw[:-1] + ', "day_1": {}}', "{}", "null", "{"):
            with self.subTest(raw=broken), self.assertRaises(ValueError):
                decode_city_schedule(broken, self.schema, self.candidates, canonical_schema=self.canonical)
        for field, value in (("cost", float("nan")), ("cost", float("inf")), ("cost", "1"),
                             ("cost", True), ("duration_min", 60.0), ("indoor", 1)):
            payload = deepcopy(self.payload)
            payload["day_1"]["attractions"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.decode(payload)

    def test_indoor_is_not_promoted_to_evidence_and_downstream_rain_rule_still_applies(self):
        route = Itinerary.model_validate(self.decode())
        problems = check_rules(route, self.request, [self.request.start_date])
        self.assertTrue(any("дождливый" in problem for problem in problems), problems)
        self.assertFalse(route.days[0].items[0].indoor)
        self.assertFalse(route.days[0].items[0].verified)

    def test_partial_day_keys_remain_numeric_and_only_requested_days_are_returned(self):
        canonical = {"type": "object", "additionalProperties": False,
                     "properties": {"day_2": deepcopy(self.canonical["properties"]["days"]["items"])},
                     "required": ["day_2"]}
        schema = build_city_schedule_schema(canonical, self.request, self.candidates)
        result = self.decode({"day_2": self.payload["day_2"]}, schema=schema, canonical=canonical)
        self.assertEqual(list(result), ["day_2"])
        self.assertEqual(result["day_2"]["items"][0]["title"], self.candidates[1].title)

    def test_format_rule_states_identity_only_and_keeps_indoor_unverified(self):
        for text in ("candidate_id", "один раз за всю поездку", "не проверенным фактом",
                     "карта подтверждает только идентичность", "rainy_alternative", "бюджета", "содержательности"):
            self.assertIn(text, CITY_CANDIDATE_SCHEDULE_RULE)


class SelectedCityProgramTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "getaddrinfo"), (sqlite3, "connect")):
            handle = patch.object(owner, name, side_effect=AssertionError("No external I/O"))
            handle.start()
            self.addCleanup(handle.stop)
        self.request = city_request(start_date="2099-10-28", end_date="2099-11-01")
        self.canonical = planner.build_itinerary_json_schema(self.request)
        self.candidates = candidates_for(5)
        self.selected = [1, 2, 4]
        self.schema = build_city_schedule_schema(self.canonical, self.request, self.candidates,
                                                program_day_numbers=self.selected)
        full = candidate_wire(self.request)
        self.payload = {f"day_{number}": full[f"day_{number}"] for number in self.selected}

    def decode(self, payload=None, *, schema=None, canonical=None, selection=None):
        return json.loads(decode_city_schedule(
            json.dumps(self.payload if payload is None else payload, ensure_ascii=False),
            self.schema if schema is None else schema, self.candidates,
            canonical_schema=self.canonical if canonical is None else canonical,
            program_day_numbers=self.selected if selection is None else selection))

    def test_selected_days_keep_original_numbers_dates_and_have_no_model_free_metadata(self):
        self.assertEqual(list(self.schema["properties"]), ["day_1", "day_2", "day_4"])
        self.assertEqual(self.schema["required"], ["day_1", "day_2", "day_4"])
        result = self.decode()
        self.assertEqual(list(result), ["day_1", "day_2", "day_4"])
        for number, expected_date in ((1, "2099-10-28"), (2, "2099-10-29"), (4, "2099-10-31")):
            fields = self.schema["properties"][f"day_{number}"]["properties"]
            self.assertEqual(fields["day"]["const"], number)
            self.assertEqual(fields["date"]["const"], expected_date)
            self.assertEqual(result[f"day_{number}"]["day"], number)
            self.assertEqual(result[f"day_{number}"]["date"], expected_date)
            for key in ("day_mode", "program_report", "source_schedule"):
                self.assertNotIn(key, fields)
                self.assertNotIn(key, result[f"day_{number}"])
        self.assertNotIn("day_3", result)
        self.assertNotIn("day_5", result)

    def test_sorting_selection_is_numeric_and_preserves_inputs(self):
        selected = [4, 1, 2]
        before = deepcopy((self.canonical, self.candidates, selected))
        schema = build_city_schedule_schema(self.canonical, self.request, self.candidates,
                                            program_day_numbers=selected)
        self.assertEqual(schema, self.schema)
        self.assertEqual(self.decode(schema=schema, selection=selected), self.decode())
        self.assertEqual((self.canonical, self.candidates, selected), before)

    def test_none_keeps_previous_full_contract_and_explicit_all_is_keyed(self):
        full_schema = build_city_schedule_schema(self.canonical, self.request, self.candidates)
        explicit_none = build_city_schedule_schema(self.canonical, self.request, self.candidates,
                                                   program_day_numbers=None)
        self.assertEqual(full_schema, explicit_none)
        payload = candidate_wire(self.request)
        decoded = json.loads(decode_city_schedule(json.dumps(payload), full_schema, self.candidates,
                                                 canonical_schema=self.canonical))
        self.assertEqual(set(decoded), {"title", "summary", "tips", "days"})
        selected = [1, 2, 3, 4, 5]
        schema = build_city_schedule_schema(self.canonical, self.request, self.candidates,
                                            program_day_numbers=selected)
        only_days = {key: value for key, value in payload.items() if key.startswith("day_")}
        self.assertEqual(list(self.decode(only_days, schema=schema, selection=selected)),
                         [f"day_{number}" for number in selected])

    def test_invalid_empty_duplicate_noninteger_and_outside_selection_fail_closed(self):
        for selected in ([], [1, 1], [True], [1.0], ["1"], [0], [-1], [6], (1, 2), {1, 2}, "1,2"):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                build_city_schedule_schema(self.canonical, self.request, self.candidates,
                                            program_day_numbers=selected)
            with self.subTest(decoder_selection=selected), self.assertRaises(ValueError):
                self.decode(selection=selected)

    def test_selection_must_match_wire_and_cannot_add_drop_or_renumber_days(self):
        for selected in ([1, 2], [1, 2, 3], [2, 4]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                self.decode(selection=selected)
        with self.assertRaises(ValueError):
            decode_city_schedule(json.dumps(self.payload), self.schema, self.candidates,
                                 canonical_schema=self.canonical)  # Selection cannot be inferred from model/schema.
        for mutation in ("drop", "add", "renumber", "date"):
            payload = deepcopy(self.payload)
            if mutation == "drop": payload.pop("day_4")
            elif mutation == "add": payload["day_3"] = deepcopy(payload["day_4"])
            elif mutation == "renumber": payload["day_4"]["day"] = 3
            else: payload["day_4"]["date"] = "2099-10-30"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.decode(payload)

    def test_root_text_free_days_and_server_flags_are_not_accepted(self):
        for key, value in (("title", "Подмена"), ("summary", "Свободные дни выбраны моделью"),
                           ("tips", []), ("source_schedule", True), ("program_report", {}),
                           ("day_3", {"day_mode": "free"})):
            payload = deepcopy(self.payload)
            payload[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.decode(payload)
        for key, value in (("day_mode", "free"), ("source_schedule", True), ("program_report", {})):
            payload = deepcopy(self.payload)
            payload["day_4"][key] = value
            with self.subTest(day_key=key), self.assertRaises(ValueError):
                self.decode(payload)

    def test_candidate_count_applies_to_selected_days_without_reusing_ids(self):
        pool = [self.candidates[index] for index in (0, 1, 3)]
        schema = build_city_schedule_schema(self.canonical, self.request, pool,
                                            program_day_numbers=self.selected)
        result = json.loads(decode_city_schedule(json.dumps(self.payload), schema, pool,
                                                canonical_schema=self.canonical, program_day_numbers=self.selected))
        self.assertEqual(len(result), 3)
        with self.assertRaises(ValueError):
            build_city_schedule_schema(self.canonical, self.request, pool)  # Full five-day route still fails.
        with self.assertRaises(ValueError):
            build_city_schedule_schema(self.canonical, self.request, pool[:2], program_day_numbers=self.selected)
        for identifier in ("c1", "unknown"):
            payload = deepcopy(self.payload)
            payload["day_4"]["attractions"][0]["candidate_id"] = identifier
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                self.decode(payload)

    def test_selected_response_retains_meal_rest_cost_and_attraction_bounds(self):
        for field, key, value in (("meal", "cost", 1799), ("rest", "duration_min", 59),
                                  ("meal", "candidate_id", "c3")):
            payload = deepcopy(self.payload)
            payload["day_4"][field][key] = value
            with self.subTest(field=field, key=key), self.assertRaises(ValueError):
                self.decode(payload)
        payload = deepcopy(self.payload)
        payload["day_4"]["attractions"] *= 4
        with self.assertRaises(ValueError):
            self.decode(payload)

    def test_canonical_validation_rejects_weakened_wire_calendar_or_length(self):
        schema = deepcopy(self.schema)
        schema["properties"]["day_4"]["properties"]["date"] = {"type": "string"}
        payload = deepcopy(self.payload)
        payload["day_4"]["date"] = "2099-10-30"
        with self.assertRaises(ValueError):
            self.decode(payload, schema=schema)
        canonical = deepcopy(self.canonical)
        canonical["properties"]["days"]["items"]["properties"]["items"]["items"]["properties"]["description"]["maxLength"] = 2
        with self.assertRaises(ValueError):
            self.decode(canonical=canonical)
        wrong_count = deepcopy(self.canonical)
        wrong_count["properties"]["days"].update(minItems=4, maxItems=4)
        with self.assertRaises(ValueError):
            build_city_schedule_schema(wrong_count, self.request, self.candidates, program_day_numbers=[1, 2])

    def test_full_calendar_constraints_are_not_silently_discarded(self):
        original = self.canonical["properties"]["days"]["items"]["properties"]["date"]["enum"]
        for values in (original[:-1], [*original[:-1], original[0]],
                       [*original[:-1], "2099-11-03"], [*original[:-1], "not-a-date"]):
            canonical = deepcopy(self.canonical)
            canonical["properties"]["days"]["items"]["properties"]["date"]["enum"] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                build_city_schedule_schema(canonical, self.request, self.candidates,
                                            program_day_numbers=self.selected)
        canonical = deepcopy(self.canonical)
        canonical["const"] = {}
        with self.assertRaises(ValueError):
            build_city_schedule_schema(canonical, self.request, self.candidates, program_day_numbers=self.selected)
        for field, value in (("const", []), ("enum", [[]])):
            canonical = deepcopy(self.canonical)
            canonical["properties"]["days"][field] = value
            with self.subTest(array_constraint=field), self.assertRaises(ValueError):
                build_city_schedule_schema(canonical, self.request, self.candidates,
                                            program_day_numbers=self.selected)

    def test_existing_keyed_canonical_schema_with_pinned_dates_is_supported(self):
        canonical = {"type": "object", "additionalProperties": False,
                     "properties": {f"day_{number}": planner.build_itinerary_day_json_schema(self.request, number)
                                    for number in self.selected}, "required": [f"day_{number}" for number in self.selected]}
        schema = build_city_schedule_schema(canonical, self.request, self.candidates,
                                            program_day_numbers=self.selected)
        self.assertEqual(self.decode(schema=schema, canonical=canonical), self.decode())

    def test_partial_json_duplicate_keys_and_nonfinite_values_still_fail_closed(self):
        raw = json.dumps(self.payload)
        duplicate = raw.replace('"candidate_id": "c1"', '"candidate_id": "c1", "candidate_id": "c2"', 1)
        for invalid in (duplicate, raw[:-1] + ', "day_4": {}}', "null", "{}"):
            with self.subTest(raw=invalid), self.assertRaises(ValueError):
                decode_city_schedule(invalid, self.schema, self.candidates, canonical_schema=self.canonical,
                                     program_day_numbers=self.selected)
        payload = deepcopy(self.payload)
        payload["day_4"]["attractions"][0]["cost"] = float("nan")
        with self.assertRaises(ValueError):
            self.decode(payload)


if __name__ == "__main__":
    unittest.main()
