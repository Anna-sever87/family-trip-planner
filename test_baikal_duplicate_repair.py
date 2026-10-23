"""Offline regression tests for a duplicated Baikal place query."""

from __future__ import annotations

import json
import unittest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import destination_scope
import planner
from destination_scope import RegionScope
from rules import check_rules
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


DUPLICATE_QUERY = (
    "Набережная Листвянки, Листвянка, Россия"
)
GENERIC_DUPLICATE_QUERY = (
    "Набережная, Листвянка, Россия"
)
REPLACEMENT_HINT = {
    "title": "Байкальская астрофизическая обсерватория",
    "place_query": (
        "Байкальская астрофизическая обсерватория, "
        "Листвянка, Россия"
    ),
    "official_name": (
        "Байкальская астрофизическая обсерватория"
    ),
    "lat": 51.879,
    "lon": 104.902,
}

TARGETED_REPLACEMENTS = {
    1: {
        "title": "Храм Николая Чудотворца",
        "place_query": (
            "Храм Николая Чудотворца, Листвянка, Россия"
        ),
        "description": "Осмотр исторического храма.",
        "area": "посёлок Листвянка",
        "cost": 0,
        "kids_note": "",
        "indoor": True,
    },
    7: {
        "title": REPLACEMENT_HINT["title"],
        "place_query": REPLACEMENT_HINT["place_query"],
        "description": "Осмотр научного объекта снаружи.",
        "area": "посёлок Листвянка",
        "cost": 0,
        "kids_note": "",
        "indoor": False,
    },
}

BAIKAL_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[
        [103.20, 51.20],
        [110.10, 51.20],
        [110.10, 55.90],
        [103.20, 55.90],
        [103.20, 51.20],
    ]],
}

UNIQUE_PLACES = {
    2: (
        "Байкальский музей ИНЦ СО РАН",
        "Байкальский музей ИНЦ СО РАН, Листвянка, Россия",
    ),
    3: (
        "Шаман-камень",
        "Шаман-камень, Листвянка, Россия",
    ),
    4: (
        "Камень Черского",
        "Камень Черского, Листвянка, Россия",
    ),
    5: (
        "Нерпинарий в Листвянке",
        "Нерпинарий в Листвянке, Листвянка, Россия",
    ),
    6: (
        "Ретро-парк Осиповых",
        "Ретро-парк Осиповых, Листвянка, Россия",
    ),
}


def _request() -> TripRequest:
    return TripRequest(
        destination="Байкал, Россия",
        destination_kind="region",
        start_date="2026-10-19",
        end_date="2026-10-25",
        adults=2,
        children_ages=[],
        budget=60_000,
        travel_cost=10_000,
        lodging_cost=15_000,
        currency="RUB",
        pace="средний",
        start_time="08:00",
        end_time="21:00",
        trip_goal="Познакомиться с территорией",
        interest_categories=["main_sights", "parks"],
        location_preference="anywhere",
        stay_area="",
        wishes="",
    )


def _item(
    *,
    time: str,
    kind: str,
    title: str,
    description: str,
    duration_min: int,
    cost: float,
    place_query: str = "",
    indoor: bool = False,
) -> ItineraryItem:
    return ItineraryItem(
        time=time,
        kind=kind,
        title=title,
        area="",
        place_query=place_query,
        description=description,
        duration_min=duration_min,
        cost=cost,
        kids_note="",
        indoor=indoor,
    )


def _day(day_number: int) -> ItineraryDay:
    if day_number in (1, 7):
        title = "Набережная Листвянки"
        place_query = DUPLICATE_QUERY
    else:
        title, place_query = UNIQUE_PLACES[day_number]

    day_date = date(2026, 10, 19) + timedelta(
        days=day_number - 1
    )
    return ItineraryDay(
        day=day_number,
        date=day_date.isoformat(),
        theme=f"Знакомство с Байкалом — день {day_number}",
        base_area="Листвянка",
        transfer_note="",
        items=[
            # Synthetic long programme for repair tests, not a recommended
            # real-world visit duration for the named POI.
            _item(
                time="09:00",
                kind="attraction",
                title=title,
                place_query=place_query,
                description="Посещение конкретного места.",
                duration_min=240,
                cost=0,
            ),
            _item(
                time="13:00",
                kind="meal",
                title="Обед",
                description="Обед в кафе.",
                duration_min=60,
                cost=1_500,
                indoor=True,
            ),
            _item(
                time="15:00",
                kind="rest",
                title="Отдых",
                description="Спокойная пауза.",
                duration_min=60,
                cost=0,
                indoor=True,
            ),
        ],
        rainy_alternative="",
    )


def _itinerary() -> Itinerary:
    return Itinerary(
        title="Семь дней на Байкале",
        summary="Маршрут с одной локальной базой.",
        days=[_day(day_number) for day_number in range(1, 8)],
        tips=["Проверить расписание перед поездкой."],
    )


def _generic_duplicate_itinerary() -> Itinerary:
    itinerary = _itinerary()

    for day_number in (1, 7):
        attraction = _attraction(
            itinerary.days[day_number - 1]
        )
        attraction.title = "Набережная"
        attraction.place_query = GENERIC_DUPLICATE_QUERY

    return itinerary


def _attraction(day_plan: ItineraryDay) -> ItineraryItem:
    return next(
        item
        for item in day_plan.items
        if item.kind == "attraction"
    )


def _duplicate_and_unrelated_generic_itinerary() -> Itinerary:
    itinerary = _itinerary()
    generic = _attraction(itinerary.days[2])
    generic.title = "Музей"
    generic.place_query = "Музей, Листвянка, Россия"
    return itinerary


def _candidate_with_replacements(
    source: Itinerary,
    replacements: dict[int, dict[str, object]] | None = None,
) -> Itinerary:
    candidate = source.model_copy(deep=True)

    for day_number, replacement in (
        replacements or TARGETED_REPLACEMENTS
    ).items():
        item = _attraction(
            candidate.days[day_number - 1]
        )

        for field_name, value in replacement.items():
            setattr(item, field_name, value)

    return candidate


def _model_response_json(itinerary: Itinerary) -> str:
    payload = itinerary.model_dump(mode="json")

    for day_plan in payload["days"]:
        for item in day_plan["items"]:
            item.pop("verified", None)
            item.pop("lat", None)
            item.pop("lon", None)
            item.pop("official_name", None)

    return json.dumps(payload, ensure_ascii=False)


def _scope() -> RegionScope:
    return RegionScope(
        name="озеро Байкал",
        country="Россия",
        bounds=(50.50, 56.50, 102.00, 112.00),
        geometry=BAIKAL_GEOMETRY,
    )


def _stable_day_dump(day_plan: ItineraryDay) -> bytes:
    payload = json.dumps(
        day_plan.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return payload.encode("utf-8")


def _fake_geocode_records() -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {
        "Листвянка, Россия": {
            "lat": 51.856,
            "lon": 104.862,
            "name": "Листвянка, Иркутская область, Россия",
            "type": "village",
            "category": "place",
            "address": {"country_code": "ru"},
        },
        DUPLICATE_QUERY: {
            "lat": 51.860,
            "lon": 104.870,
            "name": "Набережная Листвянки",
        },
        REPLACEMENT_HINT["place_query"]: {
            "lat": REPLACEMENT_HINT["lat"],
            "lon": REPLACEMENT_HINT["lon"],
            "name": REPLACEMENT_HINT["official_name"],
        },
    }

    for offset, (_, query) in enumerate(
        UNIQUE_PLACES.values(),
        start=1,
    ):
        records[query] = {
            "lat": 51.860 + offset * 0.004,
            "lon": 104.870 + offset * 0.004,
            "name": query.split(",", 1)[0],
        }

    return records


class BaikalDuplicateRepairTests(unittest.TestCase):
    def test_verified_hint_repairs_only_conflicting_day_and_revalidates(
        self,
    ) -> None:
        request = _request()
        itinerary = _itinerary()
        unchanged_day_dumps = {
            day_plan.day: _stable_day_dump(day_plan)
            for day_plan in itinerary.days[:6]
        }
        day_seven_non_attractions = [
            item.model_dump(mode="json")
            for item in itinerary.days[6].items
            if item.kind != "attraction"
        ]

        initial_problems = check_rules(itinerary, request)
        self.assertEqual(
            [
                problem
                for problem in initial_problems
                if "повторяется место" in problem
            ],
            [
                "Дни 1 и 7: повторяется место "
                "«Набережная Листвянки» "
                "(ранее «Набережная Листвянки»)"
            ],
        )
        self.assertEqual(
            [
                problem
                for problem in initial_problems
                if "повторяется место" not in problem
            ],
            [],
            initial_problems,
        )

        replacement_requirements: dict[
            tuple[int, str],
            dict[str, object],
        ] = {}
        repaired_count = planner._prune_safe_exact_place_duplicates(
            itinerary,
            {},
            map_hints=[REPLACEMENT_HINT],
            replacement_requirements=replacement_requirements,
            replacement_allowed_days={7},
        )

        self.assertEqual(repaired_count, 1)
        for day_plan in itinerary.days[:6]:
            self.assertEqual(
                _stable_day_dump(day_plan),
                unchanged_day_dumps[day_plan.day],
            )

        day_seven = itinerary.days[6]
        attractions = [
            item
            for item in day_seven.items
            if item.kind == "attraction"
        ]
        self.assertEqual(len(attractions), 1)
        self.assertEqual(
            attractions[0].place_query,
            REPLACEMENT_HINT["place_query"],
        )
        self.assertEqual(
            [
                item.model_dump(mode="json")
                for item in day_seven.items
                if item.kind != "attraction"
            ],
            day_seven_non_attractions,
        )
        self.assertEqual(check_rules(itinerary, request), [])

        self.assertTrue(replacement_requirements)
        self.assertTrue(
            planner._replacement_map_verification_problems(
                itinerary,
                replacement_requirements,
            )
        )

        records = _fake_geocode_records()

        def fake_geocode(query: str, *_, **__):
            return records.get(query)

        with patch.object(
            destination_scope.places,
            "geocode",
            side_effect=fake_geocode,
        ):
            region_problems = planner._map_verification_problems(
                itinerary,
                None,
                _scope(),
                start_time=request.start_time,
                end_time=request.end_time,
            )

        self.assertEqual(region_problems, [])
        self.assertEqual(check_rules(itinerary, request), [])
        self.assertEqual(
            planner._replacement_map_verification_problems(
                itinerary,
                replacement_requirements,
            ),
            [],
        )

    def test_only_attraction_is_not_removed_without_verified_replacement(
        self,
    ) -> None:
        request = _request()
        itinerary = _itinerary()
        before = itinerary.model_dump(mode="json")

        repaired_count = planner._prune_safe_exact_place_duplicates(
            itinerary,
            {},
            map_hints=[],
            replacement_allowed_days={7},
        )

        self.assertEqual(repaired_count, 0)
        self.assertEqual(itinerary.model_dump(mode="json"), before)
        self.assertEqual(
            sum(
                item.kind == "attraction"
                for item in itinerary.days[6].items
            ),
            1,
        )
        self.assertTrue(
            any(
                "Дни 1 и 7: повторяется место" in problem
                for problem in check_rules(itinerary, request)
            )
        )


class TargetedDuplicateRepairTests(unittest.TestCase):
    def test_duplicate_and_unrelated_generic_day_are_selected_together(self) -> None:
        source = _duplicate_and_unrelated_generic_itinerary()
        problems = check_rules(source, _request())
        self.assertEqual(len(problems), 2, problems)
        slots = planner._sole_exact_duplicate_repair_slots(source, problems)
        self.assertEqual([(slot.day, slot.item_index) for slot in slots], [(3, 0), (7, 0)])

    def test_two_concrete_duplicate_groups_keep_each_first_occurrence(self) -> None:
        source = _itinerary()
        source.days[5].items[0] = _attraction(source.days[1]).model_copy(deep=True)
        problems = check_rules(source, _request())
        self.assertEqual(len(problems), 2, problems)
        self.assertTrue(all("повторяется место" in problem for problem in problems))
        slots = planner._sole_exact_duplicate_repair_slots(source, problems)
        self.assertEqual([slot.day for slot in slots], [6, 7])

    def test_generic_first_occurrence_does_not_discard_first_concrete_one(self) -> None:
        source = _itinerary()
        source.days[5].items[0] = _attraction(source.days[6]).model_copy(deep=True)
        _attraction(source.days[0]).title = "Набережная"
        problems = check_rules(source, _request())
        slots = planner._sole_exact_duplicate_repair_slots(source, problems)
        self.assertEqual([slot.day for slot in slots], [1, 7])

    def test_unrelated_budget_or_time_failure_disables_scoped_repair(self) -> None:
        for failure in ("budget", "time"):
            with self.subTest(failure=failure):
                source = _duplicate_and_unrelated_generic_itinerary()
                if failure == "budget":
                    source.days[1].items[1].cost = _request().program_budget + 1
                else:
                    _attraction(source.days[1]).time = "23:00"
                problems = check_rules(source, _request())
                self.assertTrue(any(
                    "повторяется место" not in problem
                    and "слишком общее название" not in problem
                    for problem in problems
                ), problems)
                self.assertEqual(planner._sole_exact_duplicate_repair_slots(source, problems), [])

    def test_any_conflicting_day_with_multiple_attractions_disables_all_slots(self) -> None:
        for day_number in (3, 7):
            with self.subTest(day=day_number):
                source = _duplicate_and_unrelated_generic_itinerary()
                extra = _attraction(source.days[1]).model_copy(deep=True)
                extra.title = REPLACEMENT_HINT["title"]
                extra.place_query = REPLACEMENT_HINT["place_query"]
                extra.time = "16:15"
                extra.duration_min = 45
                source.days[day_number - 1].items.append(extra)
                problems = check_rules(source, _request())
                self.assertEqual(len(problems), 2, problems)
                self.assertEqual(planner._sole_exact_duplicate_repair_slots(source, problems), [])

    def test_multiple_conflicts_get_early_scoped_retry_with_three_call_budget(self) -> None:
        request = _request()
        first = _duplicate_and_unrelated_generic_itinerary()
        planner.sanitize_rest_items(first)
        replacements = {
            3: {
                "title": UNIQUE_PLACES[3][0],
                "place_query": UNIQUE_PLACES[3][1],
                "description": "Осмотр природного объекта с берега.",
            },
            7: TARGETED_REPLACEMENTS[7],
        }
        candidate = _candidate_with_replacements(first, replacements)
        candidate.title = "Нежелательная смена названия"
        candidate.summary = "Нежелательная смена описания"
        _attraction(candidate.days[1]).place_query = "Чужое место, Иркутск, Россия"
        for day_number in (3, 7):
            day = candidate.days[day_number - 1]
            day.base_area = "Иркутск"
            day.transfer_note = "Нежелательный переезд"
            _attraction(day).time = "20:30"
            _attraction(day).duration_min = 15
            day.items[1].cost = 99_999

        for succeeds in (True, False):
            with self.subTest(succeeds=succeeds):
                responses = [first, candidate] if succeeds else [first, first, first]
                generations = [SimpleNamespace(
                    text=_model_response_json(route), input_tokens=11, output_tokens=13,
                ) for route in responses]
                with (
                    patch.object(planner, "resolve_region", return_value=_scope()),
                    patch.object(planner, "get_weather") as weather,
                    patch.object(planner, "find_city") as find_city,
                    patch.object(planner, "cached_place_hints", return_value=[]),
                    patch.object(planner, "curated_place_hints", return_value=[]),
                    patch.object(planner, "curated_blocked_place_terms", return_value=[]),
                    patch.object(planner, "generate_json", side_effect=generations) as generate,
                    patch.object(planner, "_map_verification_problems", return_value=[]) as map_check,
                ):
                    result = planner.generate_itinerary(request, max_attempts=3)

                expected_calls = 2 if succeeds else 3
                self.assertEqual(generate.call_count, expected_calls)
                self.assertEqual(result.attempts, expected_calls)
                self.assertEqual(result.total_tokens, expected_calls * 24)
                weather.assert_not_called()
                find_city.assert_not_called()
                self.assertIn("только дней: 3, 7", generate.call_args_list[1].args[0][-1]["content"])
                if not succeeds:
                    self.assertTrue(result.problems)
                    map_check.assert_not_called()
                    continue

                self.assertEqual(result.problems, [])
                map_check.assert_called_once()
                self.assertEqual(result.itinerary.title, first.title)
                self.assertEqual(result.itinerary.summary, first.summary)
                for day_number in (1, 2, 4, 5, 6):
                    self.assertEqual(
                        result.itinerary.days[day_number - 1].model_dump(),
                        first.days[day_number - 1].model_dump(),
                    )
                for day_number in (3, 7):
                    original = first.days[day_number - 1]
                    actual = result.itinerary.days[day_number - 1]
                    self.assertEqual(actual.base_area, original.base_area)
                    self.assertEqual(actual.transfer_note, original.transfer_note)
                    self.assertEqual(_attraction(actual).place_query, replacements[day_number]["place_query"])
                    self.assertEqual(_attraction(actual).time, _attraction(original).time)
                    self.assertEqual(_attraction(actual).duration_min, _attraction(original).duration_min)
                    self.assertEqual(
                        [item.model_dump() for item in actual.items if item.kind != "attraction"],
                        [item.model_dump() for item in original.items if item.kind != "attraction"],
                    )

    def test_long_adult_medium_day_gets_completeness_guidance(self) -> None:
        request = _request()
        constraints = planner._generation_constraints(
            request,
            planner._max_attractions_for_request(request),
        )

        self.assertIn(
            "ориентир — 2–3 конкретных attraction",
            constraints,
        )
        self.assertIn(
            "Не добавляй выдуманные, общие или повторные места",
            constraints,
        )

    def test_invalid_json_during_scoped_retry_preserves_scope_for_third_call(self) -> None:
        first = _generic_duplicate_itinerary()
        planner.sanitize_rest_items(first)
        repaired = _candidate_with_replacements(first)
        repaired.summary = "Нежелательное изменение всей поездки"
        _attraction(repaired.days[1]).place_query = "Чужое место, Иркутск, Россия"
        responses = [
            _model_response_json(first),
            '{"days":',
            _model_response_json(repaired),
        ]
        generations = [SimpleNamespace(
            text=text, input_tokens=11, output_tokens=13,
        ) for text in responses]
        with (
            patch.object(planner, "resolve_region", return_value=_scope()),
            patch.object(planner, "get_weather") as weather,
            patch.object(planner, "find_city") as find_city,
            patch.object(planner, "cached_place_hints", return_value=[]),
            patch.object(planner, "curated_place_hints", return_value=[]),
            patch.object(planner, "curated_blocked_place_terms", return_value=[]),
            patch.object(planner, "generate_json", side_effect=generations) as generate,
            patch.object(planner, "_map_verification_problems", return_value=[]) as map_check,
            self.assertLogs("planner", level="WARNING"),
        ):
            result = planner.generate_itinerary(_request(), max_attempts=3)

        self.assertEqual(generate.call_count, 3)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(result.total_tokens, 72)
        self.assertEqual(result.problems, [])
        map_check.assert_called_once()
        weather.assert_not_called()
        find_city.assert_not_called()
        third_messages = generate.call_args_list[2].args[0]
        self.assertTrue(any("только дней: 1, 7" in message["content"] for message in third_messages))
        self.assertEqual(result.itinerary.summary, first.summary)
        for day_index in range(1, 6):
            self.assertEqual(result.itinerary.days[day_index].model_dump(), first.days[day_index].model_dump())
        for day_number in (1, 7):
            self.assertEqual(
                _attraction(result.itinerary.days[day_number - 1]).place_query,
                TARGETED_REPLACEMENTS[day_number]["place_query"],
            )

    def test_ai_failure_during_scoped_retry_keeps_failure_and_stops_calls(self) -> None:
        first = _generic_duplicate_itinerary()
        planner.sanitize_rest_items(first)
        generations = [
            SimpleNamespace(text=_model_response_json(first), input_tokens=11, output_tokens=13),
            planner.YandexAIError("offline timeout"),
        ]
        with (
            patch.object(planner, "resolve_region", return_value=_scope()),
            patch.object(planner, "get_weather") as weather,
            patch.object(planner, "find_city") as find_city,
            patch.object(planner, "cached_place_hints", return_value=[]),
            patch.object(planner, "curated_place_hints", return_value=[]),
            patch.object(planner, "curated_blocked_place_terms", return_value=[]),
            patch.object(planner, "generate_json", side_effect=generations) as generate,
            patch.object(planner, "_map_verification_problems", return_value=[]) as map_check,
            self.assertLogs("planner", level="WARNING"),
        ):
            result = planner.generate_itinerary(_request(), max_attempts=3)

        self.assertEqual(generate.call_count, 2)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.total_tokens, 24)
        self.assertTrue(any("повторяется место" in problem for problem in result.problems))
        self.assertTrue(any("слишком общее название" in problem for problem in result.problems))
        self.assertEqual(result.itinerary.model_dump(), first.model_dump())
        self.assertTrue(all(not item.verified for day in result.itinerary.days for item in day.items))
        map_check.assert_not_called()
        weather.assert_not_called()
        find_city.assert_not_called()

    def test_both_generic_duplicate_days_are_selected(self) -> None:
        source = _generic_duplicate_itinerary()
        problems = check_rules(source, _request())

        self.assertEqual(
            sum(
                "слишком общее название" in problem
                for problem in problems
            ),
            2,
            problems,
        )
        self.assertEqual(
            sum(
                "повторяется место" in problem
                for problem in problems
            ),
            1,
            problems,
        )

        slots = planner._sole_exact_duplicate_repair_slots(
            source,
            problems,
        )

        self.assertEqual(
            [
                (slot.day, slot.item_index)
                for slot in slots
            ],
            [(1, 0), (7, 0)],
        )
        self.assertEqual(
            {slot.old_query_key for slot in slots},
            {
                planner._lossless_place_identity(
                    GENERIC_DUPLICATE_QUERY
                )
            },
        )

    def test_atomic_merge_uses_only_new_poi_fields(self) -> None:
        source = _generic_duplicate_itinerary()
        source_dump = source.model_dump(mode="json")
        slots = planner._sole_exact_duplicate_repair_slots(
            source,
            check_rules(source, _request()),
        )
        candidate = _candidate_with_replacements(source)

        candidate.title = "Переписанное название поездки"
        candidate.summary = "Переписанное описание поездки."
        candidate.tips = ["Полностью другой совет."]
        candidate.days[1].theme = "Чужое изменение второго дня"
        _attraction(candidate.days[1]).title = (
            "Чужое место второго дня"
        )
        _attraction(candidate.days[1]).place_query = (
            "Чужое место второго дня, Иркутск, Россия"
        )

        for day_number in (1, 7):
            day_plan = candidate.days[day_number - 1]
            day_plan.base_area = "Чужая база"
            day_plan.transfer_note = "Чужой переезд"
            replacement = _attraction(day_plan)
            replacement.time = "20:30"
            replacement.duration_min = 15

            meal = next(
                item
                for item in day_plan.items
                if item.kind == "meal"
            )
            meal.title = "Другой обед"
            meal.cost = 99_999

            rest = next(
                item
                for item in day_plan.items
                if item.kind == "rest"
            )
            rest.title = "Другая пауза"
            rest.duration_min = 5

        merged, accepted = (
            planner._merge_targeted_duplicate_repair(
                source,
                candidate,
                slots,
            )
        )

        self.assertTrue(accepted)
        self.assertEqual(merged.title, source.title)
        self.assertEqual(merged.summary, source.summary)
        self.assertEqual(merged.tips, source.tips)

        for day_index in range(1, 6):
            self.assertEqual(
                merged.days[day_index].model_dump(mode="json"),
                source_dump["days"][day_index],
            )

        for day_number in (1, 7):
            source_day = source.days[day_number - 1]
            merged_day = merged.days[day_number - 1]
            merged_attraction = _attraction(merged_day)
            source_attraction = _attraction(source_day)
            expected = TARGETED_REPLACEMENTS[day_number]

            self.assertEqual(
                merged_day.base_area,
                source_day.base_area,
            )
            self.assertEqual(
                merged_day.transfer_note,
                source_day.transfer_note,
            )
            self.assertEqual(
                merged_attraction.time,
                source_attraction.time,
            )
            self.assertEqual(
                merged_attraction.duration_min,
                source_attraction.duration_min,
            )

            for field_name, expected_value in expected.items():
                self.assertEqual(
                    getattr(merged_attraction, field_name),
                    expected_value,
                )

            self.assertFalse(merged_attraction.verified)
            self.assertIsNone(merged_attraction.lat)
            self.assertIsNone(merged_attraction.lon)
            self.assertEqual(merged_attraction.official_name, "")
            self.assertEqual(
                [
                    item.model_dump(mode="json")
                    for item in merged_day.items
                    if item.kind != "attraction"
                ],
                [
                    item.model_dump(mode="json")
                    for item in source_day.items
                    if item.kind != "attraction"
                ],
            )

        self.assertEqual(check_rules(merged, _request()), [])

    def test_partial_generic_and_old_query_repairs_are_atomic_rejections(
        self,
    ) -> None:
        source = _generic_duplicate_itinerary()
        source_dump = source.model_dump(mode="json")
        slots = planner._sole_exact_duplicate_repair_slots(
            source,
            check_rules(source, _request()),
        )

        partial = _candidate_with_replacements(source)
        partial.days[6].items = [
            item
            for item in partial.days[6].items
            if item.kind != "attraction"
        ]

        generic = _candidate_with_replacements(source)
        generic_item = _attraction(generic.days[6])
        generic_item.title = "Набережная"
        generic_item.place_query = (
            "Набережная Байкальска, Байкальск, Россия"
        )

        old_query = _candidate_with_replacements(source)
        old_item = _attraction(old_query.days[6])
        old_item.title = "Конкретно названная прогулочная зона"
        old_item.place_query = GENERIC_DUPLICATE_QUERY

        for label, candidate in (
            ("partial", partial),
            ("generic", generic),
            ("old_query", old_query),
        ):
            with self.subTest(label=label):
                merged, accepted = (
                    planner._merge_targeted_duplicate_repair(
                        source,
                        candidate,
                        slots,
                    )
                )

                self.assertFalse(accepted)
                self.assertEqual(
                    merged.model_dump(mode="json"),
                    source_dump,
                )

    def test_specific_duplicate_targets_only_later_occurrence(self) -> None:
        source = _itinerary()
        problems = check_rules(source, _request())
        self.assertEqual(
            sum(
                "повторяется место" in problem
                for problem in problems
            ),
            1,
            problems,
        )
        self.assertFalse(
            any(
                "слишком общее название" in problem
                for problem in problems
            ),
            problems,
        )

        slots = planner._sole_exact_duplicate_repair_slots(
            source,
            problems,
        )
        self.assertEqual(
            [slot.day for slot in slots],
            [7],
        )

        candidate = _candidate_with_replacements(source)
        merged, accepted = (
            planner._merge_targeted_duplicate_repair(
                source,
                candidate,
                slots,
            )
        )

        self.assertTrue(accepted)
        self.assertEqual(
            merged.days[0].model_dump(mode="json"),
            source.days[0].model_dump(mode="json"),
        )
        self.assertEqual(
            _attraction(merged.days[6]).place_query,
            TARGETED_REPLACEMENTS[7]["place_query"],
        )
        self.assertEqual(check_rules(merged, _request()), [])

    def test_different_full_queries_do_not_create_repair_slots(self) -> None:
        source = _itinerary()
        last_attraction = _attraction(source.days[6])
        last_attraction.title = "Набережная Байкальска"
        last_attraction.place_query = (
            "Набережная Байкальска, Байкальск, Россия"
        )
        problems = check_rules(source, _request())

        self.assertFalse(
            any(
                "повторяется место" in problem
                for problem in problems
            ),
            problems,
        )
        self.assertEqual(
            planner._sole_exact_duplicate_repair_slots(
                source,
                problems,
            ),
            [],
        )

    def test_generate_itinerary_uses_one_scoped_retry_then_one_map_check(
        self,
    ) -> None:
        request = _request()
        first = _generic_duplicate_itinerary()
        repaired_candidate = _candidate_with_replacements(first)
        generations = [
            SimpleNamespace(
                text=_model_response_json(first),
                input_tokens=11,
                output_tokens=13,
            ),
            SimpleNamespace(
                text=_model_response_json(repaired_candidate),
                input_tokens=17,
                output_tokens=19,
            ),
        ]

        with (
            patch.object(
                planner,
                "resolve_region",
                return_value=_scope(),
            ) as resolve_region,
            patch.object(planner, "get_weather") as get_weather,
            patch.object(planner, "find_city") as find_city,
            patch.object(
                planner,
                "cached_place_hints",
                return_value=[],
            ),
            patch.object(
                planner,
                "curated_place_hints",
                return_value=[],
            ),
            patch.object(
                planner,
                "curated_blocked_place_terms",
                return_value=[],
            ),
            patch.object(
                planner,
                "generate_json",
                side_effect=generations,
            ) as generate_json,
            patch.object(
                planner,
                "_map_verification_problems",
                return_value=[],
            ) as map_check,
        ):
            result = planner.generate_itinerary(
                request,
                max_attempts=2,
            )

        resolve_region.assert_called_once_with(
            request.destination
        )
        get_weather.assert_not_called()
        find_city.assert_not_called()
        self.assertEqual(generate_json.call_count, 2)
        map_check.assert_called_once()
        self.assertEqual(result.problems, [])
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.prompt_tokens, 28)
        self.assertEqual(result.completion_tokens, 32)
        self.assertEqual(result.total_tokens, 60)
        self.assertEqual(
            _attraction(
                result.itinerary.days[0]
            ).place_query,
            TARGETED_REPLACEMENTS[1]["place_query"],
        )
        self.assertEqual(
            _attraction(
                result.itinerary.days[6]
            ).place_query,
            TARGETED_REPLACEMENTS[7]["place_query"],
        )

        second_messages = (
            generate_json.call_args_list[1].args[0]
        )
        self.assertIn(
            "только дней: 1, 7",
            second_messages[-1]["content"],
        )
        self.assertIn(
            "Верни полный маршрут",
            second_messages[-1]["content"],
        )


if __name__ == "__main__":
    unittest.main()
