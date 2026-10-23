"""A model may change one city day, never its record authority or other days."""

from contextlib import ExitStack
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from city_candidates import CityCandidate
from city_coverage import build_city_free_day
from city_edit import (
    build_city_edit_schema, city_edit_prompt_data, decode_city_edit,
    prepare_city_edit, validate_city_edit,
)
from city_estimates import label_city_estimates
from places import verify_itinerary
from planner import build_itinerary_day_json_schema
from program_coverage import rebuild_program_report
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def fixture(*, children=(), pool_size=6):
    request = TripRequest(destination="Тестгород, Россия", start_date="2099-10-28", end_date="2099-10-30",
                          adults=2, children_ages=list(children), budget=60000, currency="RUB",
                          pace="средний", start_time="09:00", end_time="19:00")
    center = dict(lat=55.0, lon=83.0, name="Тестгород", country_code="RU")
    titles = ("Городской исторический музей", "Городской научный музей", "Музей морской истории",
              "Музей народных ремёсел", "Музей древней культуры", "Музей военной истории")
    candidates = []
    for index, title in enumerate(titles[:pool_size]):
        query = f"{title}, Тестгород, Россия"
        candidates.append(CityCandidate(
            candidate_id=f"osm-node-{101 + index}", title=title, place_query=query, area="Тестгород",
            record=dict(lat=55.0, lon=83.0 + index * .0001, name=query, provider="osm",
                        osm_type="node", osm_id=101 + index, type="museum",
                        address={"city": "Тестгород", "country": "Россия", "country_code": "ru"}),
        ))
    days = []
    for number in (1, 2):
        visits = [ItineraryItem(time=time, kind="attraction", title=value.title, area=value.area,
                               place_query=value.place_query, description="Осмотр постоянной экспозиции",
                               duration_min=90, cost=1000, indoor=True)
                  for value, time in zip(candidates[(number - 1) * 2:number * 2], ("09:00", "10:45"))]
        pauses = build_city_free_day(request, number).items
        for pause in pauses:
            pause.area = "Тестгород"
        days.append(ItineraryDay(day=number, date=f"2099-10-{27 + number}", theme="История города",
                                 items=visits + pauses))
    days.append(build_city_free_day(request, 3))
    itinerary = Itinerary(title="Синтетическая поездка", summary="Проверка редактирования",
                          days=days, tips=["Уточните условия посещения"], city_schedule=True)
    label_city_estimates(itinerary, request)
    rebuild_program_report(itinerary, request, eligible_unique_visits=len(candidates),
                           allow_city_limited_program=True)
    assert verify_itinerary(itinerary, verified_records={value.place_query: value.record for value in candidates}) == []
    return request, itinerary, candidates, center


def context_for(values, target=1, *, rainy_dates=()):
    request, itinerary, candidates, center = values
    return prepare_city_edit(request, itinerary, target, candidates=candidates, resolved_city=center,
                             canonical_day_schema=build_itinerary_day_json_schema(request, target),
                             rainy_dates=rainy_dates)


def wire_for(context):
    day = deepcopy(city_edit_prompt_data(context)["before_day"])
    items = day.pop("items")
    keys = {"candidate_id", "time", "duration_min", "cost", "description", "kids_note", "indoor"}
    day["attractions"] = [{key: value for key, value in item.items() if key in keys}
                          for item in items if item["kind"] == "attraction"]
    day["meal"] = next(item for item in items if item["kind"] == "meal")
    day["rest"] = next(item for item in items if item["kind"] == "rest")
    return {f"day_{context.target_day}": day}


def decode(value, context):
    return decode_city_edit(json.dumps(value, ensure_ascii=False), context)


def new_visit(identifier, time="09:00"):
    return dict(candidate_id=identifier, time=time, duration_min=90, cost=1000,
                description="Осмотр постоянной экспозиции", kids_note="", indoor=True)


class CityEditTests(unittest.TestCase):
    def setUp(self):
        # Even accidental fallback must fail immediately, not talk to services.
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.blockers = []
        for name in ("httpx.get", "httpx.post", "places.geocode", "planner.generate_json",
                     "places._load_cache", "places._save_cache", "sqlite3.connect", "socket.socket.connect"):
            self.blockers.append(self.stack.enter_context(
                patch(name, side_effect=AssertionError(f"Forbidden I/O: {name}"))))

    def tearDown(self):
        for blocker in self.blockers:
            blocker.assert_not_called()

    def test_prepare_excludes_other_days_and_owns_defensive_snapshots(self):
        values = fixture()
        context = context_for(values)
        prompt = city_edit_prompt_data(context)
        self.assertEqual([item["candidate_id"] for item in prompt["available_candidates"]],
                         ["osm-node-101", "osm-node-102", "osm-node-105", "osm-node-106"])
        self.assertEqual(prompt["unavailable_candidate_ids"], ["osm-node-103", "osm-node-104"])
        self.assertNotIn("record", prompt["available_candidates"][0])
        self.assertNotIn("address", json.dumps(prompt))
        values[1].days[0].theme = "Changed outside"
        values[2][0].record["lat"] = 0
        values[3]["lat"] = 0
        self.assertEqual(context.before_day.theme, "История города")
        self.assertEqual(context.records[context.candidates[0].place_query]["lat"], 55)
        self.assertEqual(context.resolved_city["lat"], 55)

    def test_valid_program_edit_preserves_every_other_day_and_root_exactly(self):
        values = fixture()
        values[1].days[1].items[-2].description = "Сохранённое пользовательское пояснение обеда"
        original = values[1].model_dump()
        context = context_for(values)
        wire = wire_for(context)
        wire["day_1"]["attractions"][0]["time"] = "09:15"
        result = decode(wire, context)
        self.assertFalse(result.days[0].items[0].verified)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        self.assertTrue(result.days[0].items[0].verified)
        after = result.model_dump()
        self.assertEqual(after["days"][1:], original["days"][1:])
        self.assertEqual({key: value for key, value in after.items() if key != "days"},
                         {key: value for key, value in original.items() if key != "days"})
        self.assertEqual(values[1].model_dump(), original)

    def test_unused_candidate_can_replace_visit_without_relabeling_other_days(self):
        context = context_for(fixture())
        wire = wire_for(context)
        wire["day_1"]["attractions"][0]["candidate_id"] = "osm-node-105"
        result = decode(wire, context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        self.assertEqual(result.days[0].items[0].title, "Музей древней культуры")
        self.assertEqual(result.program_report.selected_unique_visits, 4)

    def test_existing_free_day_can_stay_free_with_only_time_change(self):
        context = context_for(fixture(), 3)
        wire = wire_for(context)
        wire["day_3"]["meal"]["time"] = "13:30"
        wire["day_3"]["rest"]["time"] = "14:30"
        result = decode(wire, context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        day = result.days[2]
        self.assertEqual((day.day_mode, day.base_area), ("free", ""))
        self.assertEqual([item.time for item in day.items], ["13:30", "14:30"])
        self.assertTrue(all(not item.verified and item.lat is None and item.lon is None
                            and not item.place_query and not item.provider and not item.source_url
                            for item in day.items))
        self.assertEqual(result.program_report, context.original.program_report)

    def test_free_day_with_no_unused_candidates_still_supports_pause_edit(self):
        context = context_for(fixture(pool_size=4), 3)
        rule = build_city_edit_schema(context)["properties"]["day_3"]["properties"]["attractions"]
        self.assertEqual((rule["minItems"], rule["maxItems"]), (0, 0))
        wire = wire_for(context)
        wire["day_3"]["rest"]["time"] = "14:15"
        result = decode(wire, context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        wire["day_3"]["attractions"] = [new_visit("osm-node-101")]
        with self.assertRaises(ValueError):
            decode(wire, context)

    def test_free_to_full_program_removes_report_but_does_not_rewrite_other_dates(self):
        context = context_for(fixture(), 3)
        wire = wire_for(context)
        wire["day_3"]["theme"] = "Дополнительные музеи"
        wire["day_3"]["attractions"] = [new_visit("osm-node-105"), new_visit("osm-node-106", "10:45")]
        result = decode(wire, context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        self.assertEqual(result.days[2].day_mode, "program")
        self.assertIsNone(result.program_report)
        self.assertEqual(result.days[:2], context.original.days[:2])

    def test_program_day_cannot_become_free_or_accept_server_metadata(self):
        context = context_for(fixture())
        for field, value in (("attractions", []), ("day_mode", "free"), ("program_report", {}),
                             ("city_schedule", True), ("source_schedule", True)):
            with self.subTest(field=field):
                wire = wire_for(context)
                wire["day_1"][field] = value
                with self.assertRaises(ValueError):
                    decode(wire, context)

    def test_number_date_root_and_other_day_scope_are_strict_wire_boundaries(self):
        context = context_for(fixture())
        for field, value in (("day", 2), ("date", "2099-10-29"), ("base_area", "Отель"),
                             ("transfer_note", "Переезд"), ("rainy_alternative", "Другой музей")):
            with self.subTest(field=field):
                wire = wire_for(context)
                wire["day_1"][field] = value
                with self.assertRaises(ValueError):
                    decode(wire, context)
        for field, value in (("title", "Новый заголовок"), ("day_2", {})):
            wire = wire_for(context)
            wire[field] = value
            with self.assertRaises(ValueError):
                decode(wire, context)

    def test_model_cannot_supply_identity_map_fields_or_reuse_occupied_ids(self):
        context = context_for(fixture())
        for field, value in (("candidate_id", "osm-node-103"), ("candidate_id", "unknown"),
                             ("title", "Придуманный музей"), ("place_query", "Другое место"),
                             ("verified", True), ("lat", 55), ("provider", "osm")):
            with self.subTest(field=field, value=value):
                wire = wire_for(context)
                wire["day_1"]["attractions"][0][field] = value
                with self.assertRaises(ValueError):
                    decode(wire, context)
        wire = wire_for(context)
        wire["day_1"]["attractions"][1]["candidate_id"] = "osm-node-101"
        with self.assertRaises(ValueError):
            decode(wire, context)

    def test_no_change_is_not_mislabeled_as_success(self):
        context = context_for(fixture())
        result = decode(wire_for(context), context)
        problems = validate_city_edit(result, context, blocked_places=[])
        self.assertTrue(any("остался без изменений" in item for item in problems))

    def test_direct_candidate_cannot_modify_other_days_root_or_report(self):
        context = context_for(fixture())
        wire = wire_for(context)
        wire["day_1"]["attractions"][0]["time"] = "09:15"
        for mutation in (lambda result: setattr(result.days[1], "theme", "Other"),
                         lambda result: setattr(result, "summary", "Other"),
                         lambda result: setattr(result.program_report, "eligible_unique_visits", 99),
                         lambda result: setattr(result, "source_schedule", 0),
                         lambda result: setattr(result.days[0].items[0], "place_query", "Unknown")):
            result = decode(wire, context)
            mutation(result)
            before = result.model_dump()
            self.assertTrue(validate_city_edit(result, context, blocked_places=[]))
            self.assertEqual(result.model_dump(), before)

    def test_bad_cost_types_nonfinite_and_negative_values_are_not_repaired(self):
        context = context_for(fixture())
        for cost in (-1, float("nan"), float("inf"), "1000", True):
            with self.subTest(cost=cost):
                wire = wire_for(context)
                wire["day_1"]["attractions"][0]["cost"] = cost
                with self.assertRaises(ValueError):
                    decode(wire, context)
                result = context.original.model_copy(deep=True)
                result.days[0].items[0].cost = cost
                self.assertTrue(validate_city_edit(result, context, blocked_places=[]))

    def test_depth_budget_overlap_and_cap_checks_still_run(self):
        context = context_for(fixture())
        def shallow(day):
            for item in day["attractions"]:
                item["duration_min"] = 30
        def expensive(day):
            day["attractions"][0]["cost"] = 59000
        def overlap(day):
            day["attractions"][1]["time"] = "09:30"
        def stretched(day):
            day["attractions"][0]["duration_min"] = 121
            day["attractions"][1]["time"] = "11:15"
        for change, expected in ((shallow, "содержатель"), (expensive, "бюджет"),
                                 (overlap, "пересека"), (stretched, "планирования")):
            with self.subTest(change=change.__name__):
                wire = wire_for(context)
                change(wire["day_1"])
                result = decode(wire, context)
                problems = validate_city_edit(result, context, blocked_places=[])
                self.assertTrue(any(expected in problem.lower() for problem in problems), problems)
                self.assertFalse(result.days[0].items[0].verified)

    def test_rain_and_blocked_catalog_cover_all_days_not_only_target(self):
        values = fixture()
        values[1].days[1].items[0].indoor = False
        values[1].days[1].items[1].indoor = False
        context = context_for(values, rainy_dates=["2099-10-29"])
        wire = wire_for(context)
        wire["day_1"]["attractions"][0]["time"] = "09:15"
        result = decode(wire, context)
        self.assertTrue(any("дожд" in problem.lower()
                            for problem in validate_city_edit(result, context, blocked_places=[])))
        context = context_for(fixture())
        result = decode(wire, context)
        blocked = [{"aliases": [context.original.days[1].items[0].title], "reason": "Синтетический запрет"}]
        self.assertTrue(validate_city_edit(result, context, blocked_places=blocked))

    def test_distant_pair_is_prompted_and_still_rejected_by_final_map_guard(self):
        values = fixture()
        values[2][4].record["lat"] = 55.1
        context = context_for(values)
        pairs = city_edit_prompt_data(context)["incompatible_same_day_pairs"]
        self.assertTrue(any(set(pair["candidate_ids"]) == {"osm-node-101", "osm-node-105"} for pair in pairs))
        wire = wire_for(context)
        wire["day_1"]["attractions"][1]["candidate_id"] = "osm-node-105"
        result = decode(wire, context)
        self.assertTrue(any("далеко друг от друга" in problem
                            for problem in validate_city_edit(result, context, blocked_places=[])))

    def test_prepare_rejects_stale_mismatched_forged_or_outside_records(self):
        mutations = (
            lambda values: values[2][0].record["address"].update(country_code="vn"),
            lambda values: values[2][0].record.update(lat=float("nan")),
            lambda values: values[2][0].record.update(lat=56),
            lambda values: values[2][0].record.update(osm_id=999),
            lambda values: values[2][0].record.update(osm_type="invented"),
            lambda values: values[2][0].record.update(provider="model"),
            lambda values: values[2][0].record.update(type="locality"),
            lambda values: setattr(values[1].days[0].items[0], "lat", 55.01),
            lambda values: setattr(values[1].days[0].items[0], "verified", False),
            lambda values: setattr(values[2][0], "area", "Другой город"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                values = fixture()
                mutation(values)
                with self.assertRaises(ValueError):
                    context_for(values)

    def test_invalid_context_calendar_and_authorization_fail_before_model(self):
        for mutation in (lambda values: setattr(values[1], "city_schedule", False),
                         lambda values: setattr(values[1], "source_schedule", True),
                         lambda values: setattr(values[1].days[1], "date", "2099-10-30"),
                         lambda values: setattr(values[0], "destination_kind", "region"),
                         lambda values: values[3].pop("country_code")):
            with self.subTest(mutation=mutation):
                values = fixture()
                mutation(values)
                with self.assertRaises(ValueError):
                    context_for(values)
        for target in (0, 4, True, 1.0):
            with self.subTest(target=target), self.assertRaises((ValueError, TypeError)):
                context_for(fixture(), target)
        for dates in (["2099-10-31"], ["2099-10-28", "2099-10-28"], "2099-10-28"):
            with self.subTest(dates=dates), self.assertRaises(ValueError):
                context_for(fixture(), rainy_dates=dates)

    def test_family_labels_fit_edit_contract_and_other_days_keep_exact_labels(self):
        context = context_for(fixture(children=[5]))
        wire = wire_for(context)
        wire["day_1"]["attractions"][0]["time"] = "09:15"
        result = decode(wire, context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        self.assertEqual(result.days[1:], context.original.days[1:])
        self.assertEqual(result.tips, context.original.tips)

    def test_family_free_to_program_accepts_server_label_without_truncation(self):
        context = context_for(fixture(children=[5]), 3)
        wire = wire_for(context)
        wire["day_3"]["attractions"] = [new_visit("osm-node-105")]
        result = decode(wire, context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        self.assertGreater(len(result.days[2].items[0].kids_note), 100)
        self.assertEqual(result.days[:2], context.original.days[:2])

    def test_direct_objects_cannot_coerce_types_or_smuggle_pause_places(self):
        context = context_for(fixture())
        wire = wire_for(context)
        wire["day_1"]["attractions"][0]["time"] = "09:15"
        for field, value in (("duration_min", "90"), ("duration_min", True),
                             ("indoor", "true"), ("kind", "unknown")):
            with self.subTest(field=field, value=value):
                result = decode(wire, context)
                setattr(result.days[0].items[0], field, value)
                self.assertTrue(validate_city_edit(result, context, blocked_places=[]))
        result = decode(wire, context)
        result.days[0].items[-2].place_query = "Кафе без подтверждения"
        self.assertTrue(validate_city_edit(result, context, blocked_places=[]))
        result = decode(wire, context)
        result.days[0].transfer_note = "Поездка в другой город"
        self.assertTrue(validate_city_edit(result, context, blocked_places=[]))

    def test_no_current_independent_record_cannot_become_a_two_gis_candidate(self):
        values = fixture()
        item = values[2][4]
        item.candidate_id = "2gis-123"
        item.record.update(provider="2gis", provider_id="123", source_url="https://2gis.ru/test/firm/123",
                           checked_at="2000-01-01")
        with patch("verified_map_places.load_verified_map_places", return_value=[]):
            with self.assertRaises(ValueError):
                context_for(values)

    def test_last_free_date_retires_only_exact_legacy_server_root_wording(self):
        values = fixture()
        values[1].title = f"{values[0].destination}: поездка со свободными днями"
        values[1].summary = ("Предварительная программа по местам, подтверждённым на карте. "
                             "На часть дат предусмотрен свободный режим. Условия посещений требуют уточнения.")
        context = context_for(values, 3)
        wire = wire_for(context)
        wire["day_3"]["attractions"] = [new_visit("osm-node-105"), new_visit("osm-node-106", "10:45")]
        result = decode(wire, context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        self.assertNotIn("свободн", result.title + result.summary)
        self.assertIn("предварительная", result.title)
        self.assertIn("Условия посещений требуют уточнения", result.summary)
        self.assertEqual([day.model_dump() for day in result.days[:2]],
                         [day.model_dump() for day in context.original.days[:2]])
        self.assertEqual(result.tips, context.original.tips)
        for field in ("title", "summary"):
            altered = result.model_copy(deep=True)
            setattr(altered, field, "Произвольное новое описание модели")
            self.assertTrue(validate_city_edit(altered, context, blocked_places=[]))

    def test_arbitrary_root_wording_is_not_rewritten_when_last_free_date_fills(self):
        values = fixture()
        values[1].title = "Наша поездка с запасом свободного времени"
        values[1].summary = "Собственное пояснение пользователя, не canonical server text."
        context = context_for(values, 3)
        wire = wire_for(context)
        wire["day_3"]["attractions"] = [new_visit("osm-node-105"), new_visit("osm-node-106", "10:45")]
        result = decode(wire, context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        self.assertEqual((result.title, result.summary), (values[1].title, values[1].summary))

    def test_blocked_check_runs_before_and_after_exact_map_reverification(self):
        context = context_for(fixture())
        wire = wire_for(context)
        wire["day_1"]["attractions"][0]["time"] = "09:15"
        result = decode(wire, context)
        seen = []
        def inspect(itinerary, blocked):
            seen.append(itinerary.days[0].items[0].official_name)
            return []
        with patch("planner._blocked_place_problems", side_effect=inspect):
            self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        self.assertEqual(seen, ["", context.candidates[0].place_query])


if __name__ == "__main__":
    unittest.main()
