"""Pure strict schedule-wire regressions; no AI, map or storage access."""

from copy import deepcopy
import json
import socket
import unittest
from unittest.mock import patch

from grounded_response import build_grounded_response_schema, decode_grounded_response
import planner
from pricing import minimum_meal_cost
from schemas import TripRequest


BASE = "Долинный, Россия"
QUERY = "Музей ремёсел, Долинный, Россия"


def request_for(days=2, **updates):
    args = dict(destination="Тестовая территория, Россия", destination_kind="region",
                start_date="2099-06-01", end_date=f"2099-06-{days:02d}",
                adults=2, children_ages=[], budget=100000, currency="RUB")
    args.update(updates)
    return TripRequest(**args)


def canonical_for(request, targets=None):
    schema = (planner.build_itinerary_json_schema(request) if targets is None
              else planner._regional_day_repair_schema(request, targets, []))
    days = ([schema["properties"]["days"]["items"]] if targets is None
            else list(schema["properties"].values()))
    for day in days:
        day["properties"]["base_area"]["enum"] = [BASE]
        item = day["properties"]["items"]["items"]["properties"]
        item["place_query"]["enum"] = ["", QUERY]
        item["area"]["enum"] = ["Долинный"]
    return schema


def item_for(kind, *, time=None, cost=0):
    return dict(time=time or {"attraction": "10:00", "meal": "13:00", "rest": "15:00"}[kind],
                kind=kind, title={"attraction": "Музей ремёсел", "meal": "Обед", "rest": "Отдых"}[kind],
                area="Долинный", place_query=QUERY if kind == "attraction" else "",
                description="Описание без добавления фактов", duration_min=60, cost=cost,
                kids_note="", indoor=True)


def wire_for(request, targets=None):
    value = {"title": "Маршрут", "summary": "Описание", "tips": ["Учитывайте ограничения"]} if targets is None else {}
    for day in range(1, request.days_count + 1) if targets is None else targets:
        value[f"day_{day}"] = dict(
            day=day, date=f"2099-06-{day:02d}", theme="Местная история", base_area=BASE,
            transfer_note="", rainy_alternative="", attractions=[item_for("attraction")],
            meal=item_for("meal", cost=minimum_meal_cost(request)), rest=item_for("rest"),
        )
    return value


class GroundedResponseTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(socket.socket, "connect", side_effect=AssertionError("No network"))
        network.start()
        self.addCleanup(network.stop)
        self.request = request_for()
        self.canonical = canonical_for(self.request)
        self.schema = build_grounded_response_schema(self.canonical, self.request)
        self.wire = wire_for(self.request)

    def decode(self, value, schema=None):
        return json.loads(decode_grounded_response(json.dumps(value, ensure_ascii=False), schema or self.schema))

    def test_full_schema_has_exact_required_days_separate_items_and_no_input_mutation(self):
        before = deepcopy(self.canonical)
        schema = build_grounded_response_schema(self.canonical, self.request)
        self.assertEqual(self.canonical, before)
        self.assertEqual(set(schema["properties"]), {"title", "summary", "tips", "day_1", "day_2"})
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertFalse(schema["additionalProperties"])
        for number in (1, 2):
            day = schema["properties"][f"day_{number}"]
            props = day["properties"]
            self.assertNotIn("items", props)
            self.assertEqual(set(day["required"]), set(props))
            self.assertEqual(props["day"]["enum"], [number])
            self.assertEqual(props["date"]["enum"], [f"2099-06-0{number}"])
            self.assertEqual(props["attractions"]["minItems"], 1)
            self.assertEqual(props["attractions"]["maxItems"], before["properties"]["days"]["items"]["properties"]["items"]["maxItems"] - 2)
            self.assertEqual(props["attractions"]["items"]["properties"]["place_query"]["enum"], [QUERY])
            for key, item in (("attraction", props["attractions"]["items"]), ("meal", props["meal"]), ("rest", props["rest"])):
                self.assertEqual(item["properties"]["kind"]["const"], key)
            for key in ("meal", "rest"):
                self.assertEqual(props[key]["properties"]["place_query"]["const"], "")
            self.assertEqual(props["meal"]["properties"]["cost"]["minimum"], 1400)
            self.assertEqual(props["rest"]["properties"]["title"]["const"], "Отдых")
        schema["properties"]["day_1"]["properties"]["theme"]["maxLength"] = 1
        self.assertEqual(self.canonical, before)

    def test_full_decode_preserves_every_value_and_sorts_items_without_filling(self):
        self.wire["day_1"]["attractions"][0]["time"] = "16:00"
        self.wire["day_1"]["attractions"][0]["description"] = "Музей ремёсел"
        before_wire, before_schema = deepcopy(self.wire), deepcopy(self.schema)
        result = self.decode(self.wire)
        self.assertEqual(set(result), {"title", "summary", "tips", "days"})
        self.assertEqual([day["day"] for day in result["days"]], [1, 2])
        self.assertEqual(result["days"][0]["items"], [self.wire["day_1"]["meal"], self.wire["day_1"]["rest"], self.wire["day_1"]["attractions"][0]])
        self.assertEqual(result["days"][0]["items"][-1]["cost"], 0)
        for key in ("title", "summary", "tips"):
            self.assertEqual(result[key], self.wire[key])
        self.assertEqual((self.wire, self.schema), (before_wire, before_schema))

    def test_full_days_sort_numerically_not_lexicographically(self):
        request = request_for(10)
        schema = build_grounded_response_schema(canonical_for(request), request)
        payload = dict(reversed(list(wire_for(request).items())))
        self.assertEqual([day["day"] for day in self.decode(payload, schema)["days"]], list(range(1, 11)))

    def test_repair_two_three_seven_retains_only_selected_keys(self):
        request = request_for(7)
        canonical = canonical_for(request, [7, 2, 3])
        schema = build_grounded_response_schema(canonical, request)
        result = self.decode(wire_for(request, [7, 2, 3]), schema)
        self.assertEqual(list(result), ["day_2", "day_3", "day_7"])
        self.assertEqual(set(result), set(canonical["properties"]))
        for key, day in result.items():
            self.assertEqual(len(day["items"]), 3)
            self.assertNotIn("attractions", day)

    def test_repair_two_three_seven_rejects_missing_extra_keys_and_inner_identity_changes(self):
        request = request_for(7)
        schema = build_grounded_response_schema(canonical_for(request, [2, 3, 7]), request)
        original = wire_for(request, [2, 3, 7])
        missing = deepcopy(original)
        missing.pop("day_3")
        extra = {**deepcopy(original), "day_4": deepcopy(original["day_3"])}
        day_mismatch = deepcopy(original)
        day_mismatch["day_7"]["day"] = 3
        date_mismatch = deepcopy(original)
        date_mismatch["day_7"]["date"] = "2099-06-03"
        for broken in (missing, extra, day_mismatch, date_mismatch):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                self.decode(broken, schema)
        raw = json.dumps(original)
        duplicate = raw[:-1] + ', "day_3": ' + json.dumps(original["day_3"]) + '}'
        with self.assertRaises(ValueError):
            decode_grounded_response(duplicate, schema)

    def test_empty_area_roundtrip_when_actual_locality_exceeds_area_length_limit(self):
        from candidate_grounding import GroundingResult
        from grounded_schedule import constrain_grounded_schema

        locality = "Длинное название местности " * 3
        locality = locality.strip()
        query = f"Музей ремёсел, {locality}, Россия"
        self.assertGreater(len(locality), 60)
        grounding = GroundingResult(
            base_facts=[{"base_key": "base_1", "base_query": BASE, "lat": 52, "lon": 105}],
            confirmed_hints=[{"base_key": "base_1", "title": "Музей ремёсел", "place_query": query,
                              "locality": locality, "lat": 52.01, "lon": 105.01}],
        )
        canonical = constrain_grounded_schema(planner.build_itinerary_json_schema(self.request), grounding)
        schema = build_grounded_response_schema(canonical, self.request)
        payload = deepcopy(self.wire)
        for key in ("day_1", "day_2"):
            day = payload[key]
            for item in [*day["attractions"], day["meal"], day["rest"]]:
                item["area"] = ""
            day["attractions"][0]["place_query"] = query
        decoded = self.decode(payload, schema)
        self.assertEqual(schema["properties"]["day_1"]["properties"]["attractions"]["items"]["properties"]["area"]["enum"], [""])
        self.assertTrue(all(item["area"] == "" for day in decoded["days"] for item in day["items"]))

    def test_meal_floor_uses_request_currency_and_whole_group(self):
        for currency, expected in (("RUB", 1800), ("EUR", 31), ("USD", 31)):
            request = request_for(currency=currency, children_ages=[5])
            schema = build_grounded_response_schema(canonical_for(request), request)
            self.assertEqual(schema["properties"]["day_1"]["properties"]["meal"]["properties"]["cost"]["minimum"], expected)
            self.assertEqual(self.decode(wire_for(request), schema)["days"][0]["items"][1]["cost"], expected)

    def test_full_and_nested_extra_or_missing_fields_fail(self):
        variants = []
        for key in ("title", "day_1"):
            value = deepcopy(self.wire)
            value.pop(key)
            variants.append(value)
        for key in ("days", "day_3", "unknown"):
            variants.append({**deepcopy(self.wire), key: []})
        for key in ("attractions", "meal", "rest", "theme"):
            value = deepcopy(self.wire)
            value["day_1"].pop(key)
            variants.append(value)
        for key in ("verified", "lat", "official_name"):
            value = deepcopy(self.wire)
            value["day_1"]["attractions"][0][key] = True
            variants.append(value)
        for value in variants:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    self.decode(value)

    def test_legacy_full_and_keyed_items_formats_are_never_accepted(self):
        canonical_payload = self.decode(self.wire)
        with self.assertRaises(ValueError):
            self.decode(canonical_payload)
        request = request_for(7)
        schema = build_grounded_response_schema(canonical_for(request, [2]), request)
        legacy = {"day_2": canonical_payload["days"][1]}
        with self.assertRaises(ValueError):
            self.decode(legacy, schema)

    def test_duplicate_json_keys_rejected_at_root_day_and_item(self):
        raw = json.dumps(self.wire, ensure_ascii=False)
        for broken in (
            raw.replace('"title": "Маршрут"', '"title": "Маршрут", "title": "Другой"', 1),
            raw.replace('"day": 1', '"day": 1, "day": 1', 1),
            raw.replace('"cost": 1400.0', '"cost": 1400.0, "cost": 1400.0', 1),
        ):
            self.assertNotEqual(broken, raw)
            with self.assertRaises(ValueError):
                decode_grounded_response(broken, self.schema)

    def test_wrong_day_date_and_numeric_coercions_fail(self):
        for field, value in (("day", True), ("day", "1"), ("day", 1.0), ("day", 2), ("date", "2099-06-02")):
            broken = deepcopy(self.wire)
            broken["day_1"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.decode(broken)
        for field, value in (("cost", True), ("cost", "1400"), ("duration_min", True), ("duration_min", 60.0), ("indoor", 1)):
            broken = deepcopy(self.wire)
            broken["day_1"]["meal"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.decode(broken)

    def test_nonfinite_numbers_and_malformed_json_fail(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            broken = deepcopy(self.wire)
            broken["day_1"]["meal"]["cost"] = value
            with self.assertRaises(ValueError):
                self.decode(broken)
        for raw in ("{", "null", "[]", "{}", "```json\n{}\n```", json.dumps(self.wire).replace("1400.0", "1e999", 1)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                decode_grounded_response(raw, self.schema)

    def test_attraction_count_and_kinds_are_strict(self):
        for count in (0, self.schema["properties"]["day_1"]["properties"]["attractions"]["maxItems"] + 1):
            broken = deepcopy(self.wire)
            broken["day_1"]["attractions"] = [item_for("attraction") for _ in range(count)]
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.decode(broken)
        for key in ("meal", "rest", "attractions"):
            broken = deepcopy(self.wire)
            item = broken["day_1"][key] if key != "attractions" else broken["day_1"][key][0]
            item["kind"] = "rest" if key != "rest" else "meal"
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.decode(broken)

    def test_empty_or_unapproved_poi_query_and_nonempty_meal_rest_queries_fail(self):
        for key, query in (("attractions", ""), ("attractions", "Другой музей, Долинный, Россия"), ("meal", QUERY), ("rest", QUERY)):
            broken = deepcopy(self.wire)
            item = broken["day_1"][key] if key != "attractions" else broken["day_1"][key][0]
            item["place_query"] = query
            with self.subTest(key=key, query=query), self.assertRaises(ValueError):
                self.decode(broken)

    def test_meal_floor_zero_and_boundary_are_checked_without_fixing(self):
        for cost in (0, 1399.99, -1):
            broken = deepcopy(self.wire)
            broken["day_1"]["meal"]["cost"] = cost
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                self.decode(broken)
            self.assertEqual(broken["day_1"]["meal"]["cost"], cost)
        self.assertEqual(self.decode(self.wire)["days"][0]["items"][1]["cost"], 1400)
        request = request_for(children_ages=[5, 10])
        schema = build_grounded_response_schema(canonical_for(request), request)
        self.assertEqual(schema["properties"]["day_1"]["properties"]["meal"]["properties"]["cost"]["minimum"], 2200)
        with self.assertRaises(ValueError):
            self.decode(self.wire, schema)

    def test_all_original_bounds_enums_constants_and_patterns_remain_checked(self):
        for field, value in (("description", ""), ("description", "Я" * 181), ("duration_min", 14),
                             ("duration_min", 241), ("area", "Другой район"), ("kids_note", "Для детей"),
                             ("time", "9:00"), ("cost", 100001)):
            broken = deepcopy(self.wire)
            broken["day_1"]["attractions"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.decode(broken)
        broken = deepcopy(self.wire)
        broken["day_1"]["rest"]["title"] = "Второй обед"
        with self.assertRaises(ValueError):
            self.decode(broken)
        canonical = deepcopy(self.canonical)
        canonical["properties"]["days"]["items"]["properties"]["items"]["items"]["properties"]["time"]["pattern"] = r"^\d{2}:\d{2}$"
        schema = build_grounded_response_schema(canonical, self.request)
        broken["day_1"] = deepcopy(self.wire["day_1"])
        broken["day_1"]["meal"]["time"] = "ab:cd"
        with self.assertRaises(ValueError):
            self.decode(broken, schema)

    def test_unrepresentable_schema_and_unknown_constraints_fail_closed(self):
        mutations = [deepcopy(self.canonical) for _ in range(5)]
        mutations[0]["properties"]["days"]["items"]["properties"]["items"]["maxItems"] = 2
        mutations[1]["properties"]["days"]["items"]["properties"]["items"]["items"]["properties"]["cost"]["maximum"] = 1000
        mutations[2]["properties"]["days"]["items"]["properties"]["items"]["items"]["properties"]["place_query"]["enum"] = [""]
        mutations[3]["oneOf"] = []
        mutations[4]["properties"]["days"]["maxItems"] = 7
        for schema in mutations:
            with self.subTest(schema=schema), self.assertRaises(ValueError):
                build_grounded_response_schema(schema, self.request)
        with self.assertRaises(ValueError):
            build_grounded_response_schema({"type": "object", "properties": {}, "required": [], "additionalProperties": False}, self.request)
        with self.assertRaises(ValueError):
            decode_grounded_response(json.dumps(self.wire), self.canonical)


if __name__ == "__main__":
    unittest.main()
