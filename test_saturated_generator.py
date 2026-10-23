"""Focused regression tests for saturated-route evening enrichment."""

from __future__ import annotations

import json
import unittest
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import planner
from rules import check_verified_place_duplicates
from schemas import Itinerary, TripRequest


def _request() -> TripRequest:
    return TripRequest(
        destination="Воркута, Россия",
        start_date="2026-11-03",
        end_date="2026-11-06",
        adults=2,
        children_ages=[],
        budget=45_000,
        currency="RUB",
        pace="насыщенный",
        start_time="10:00",
        end_time="19:00",
        trip_goal="Насыщенная культурная программа",
        interest_categories=["main_sights", "museums"],
        location_preference="nearby",
        stay_area="центр Воркуты",
        wishes="Продолжать программу до вечера",
    )


def _attraction(day_number: int, position: int) -> dict:
    times = ("10:00", "15:30", "18:00")
    durations = (60, 45, 45)
    title = f"Уникальное место {day_number}-{position}"
    return {
        "time": times[position - 1],
        "kind": "attraction",
        "title": title,
        "area": "Центр",
        "place_query": f"{title}, Воркута, Россия",
        "description": f"Посещение места «{title}».",
        "duration_min": durations[position - 1],
        "cost": 0,
        "kids_note": "",
        "indoor": position % 2 == 0,
    }


def _day(day_number: int, *, enriched: bool) -> dict:
    trip_date = date(2026, 11, 3) + timedelta(days=day_number - 1)
    attractions = [
        _attraction(day_number, 1),
        _attraction(day_number, 2),
    ]
    if enriched:
        attractions.append(_attraction(day_number, 3))

    return {
        "day": day_number,
        "date": trip_date.isoformat(),
        "theme": f"Уникальная тема дня {day_number}",
        "items": [
            attractions[0],
            {
                "time": "12:30",
                "kind": "meal",
                "title": "Обед",
                "area": "Центр",
                "place_query": "",
                "description": "Обед в городском кафе.",
                "duration_min": 60,
                "cost": 1400,
                "kids_note": "",
                "indoor": True,
            },
            {
                "time": "13:30",
                "kind": "rest",
                "title": "Отдых",
                "area": "Центр",
                "place_query": "",
                "description": "Спокойный перерыв после обеда.",
                "duration_min": 60,
                "cost": 0,
                "kids_note": "",
                "indoor": True,
            },
            *attractions[1:],
        ],
        "rainy_alternative": f"Крытая площадка {day_number}",
    }


def _itinerary(*, enriched: bool) -> Itinerary:
    return Itinerary.model_validate(
        {
            "title": "Воркута: четыре насыщенных дня",
            "summary": "Культурная программа с вечерними объектами.",
            "days": [
                _day(day_number, enriched=enriched)
                for day_number in range(1, 5)
            ],
            "tips": ["Проверить режим работы перед выходом."],
        }
    )


def _model_payload(itinerary: Itinerary) -> dict:
    payload = itinerary.model_dump()
    for day_plan in payload["days"]:
        for item in day_plan["items"]:
            item.pop("verified", None)
            item.pop("lat", None)
            item.pop("lon", None)
            item.pop("official_name", None)
    return payload


def _city_retry_payload(itinerary: Itinerary) -> dict:
    """Explicit retry fixture format, independent of the supplied response schema."""
    payload = {
        "title": itinerary.title,
        "summary": itinerary.summary,
        "tips": list(itinerary.tips),
    }
    item_fields = (
        "time", "kind", "title", "area", "place_query", "description",
        "duration_min", "cost", "kids_note", "indoor",
    )
    for day_plan in itinerary.days:
        day = {
            field: getattr(day_plan, field)
            for field in (
                "day", "date", "theme", "base_area", "transfer_note",
                "rainy_alternative",
            )
        }
        items = [
            {field: getattr(item, field) for field in item_fields}
            for item in day_plan.items
        ]
        meals = [item for item in items if item["kind"] == "meal"]
        rests = [item for item in items if item["kind"] == "rest"]
        assert len(meals) == len(rests) == 1
        day["attractions"] = [item for item in items if item["kind"] == "attraction"]
        day["meal"], day["rest"] = meals[0], rests[0]
        payload[f"day_{day_plan.day}"] = day
    return payload


def _generation(itinerary: Itinerary) -> SimpleNamespace:
    return SimpleNamespace(
        text=json.dumps(_model_payload(itinerary), ensure_ascii=False),
        input_tokens=10,
        output_tokens=20,
    )


def _city_generation(itinerary: Itinerary) -> SimpleNamespace:
    return SimpleNamespace(
        text=json.dumps(_city_retry_payload(itinerary), ensure_ascii=False),
        input_tokens=10,
        output_tokens=20,
    )


def _verify_all_places(itinerary: Itinerary) -> list[str]:
    for day_plan in itinerary.days:
        attraction_number = 0
        for item in day_plan.items:
            if item.kind != "attraction":
                continue
            attraction_number += 1
            item.verified = True
            item.lat = 67.49 + day_plan.day / 100 + attraction_number / 1000
            item.lon = 64.04 + day_plan.day / 100 + attraction_number / 1000
            item.official_name = item.title
    return []


def _all_attractions(itinerary: Itinerary):
    return (
        item
        for day_plan in itinerary.days
        for item in day_plan.items
        if item.kind == "attraction"
    )


def _trusted_evening_hints() -> list[dict]:
    return [
        {
            "title": f"Проверенное вечернее место {day_number}",
            "place_query": (
                f"Проверенное вечернее место {day_number}, "
                "Воркута, Россия"
            ),
            "official_name": f"Проверенное вечернее место {day_number}",
            "lat": 67.49 + day_number / 100 + 0.004,
            "lon": 64.04 + day_number / 100 + 0.004,
            "description": "Безопасный внешний осмотр.",
            "indoor": False,
            "access_mode": "public_exterior",
            "trusted_late": True,
            "catalog_entry": True,
            "verified_at": "2026-10-12",
            "source": "https://example.test/place",
        }
        for day_number in range(1, 5)
    ]


class SaturatedGeneratorRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = _request()
        self.sparse = _itinerary(enriched=False)
        self.enriched = _itinerary(enriched=True)

    def _common_patches(self, generations, verifier):
        return (
            patch.object(planner, "get_weather", return_value=None),
            patch.object(planner, "weather_summary", return_value=""),
            patch.object(planner, "find_city", return_value=None),
            patch.object(planner, "cached_place_hints", return_value=[]),
            patch.object(planner, "generate_json", side_effect=generations),
            patch.object(planner, "verify_itinerary", side_effect=verifier),
        )

    def test_enrichment_adds_unique_verified_late_block_in_two_ai_calls(self) -> None:
        generation_mock = None
        patches = self._common_patches(
            [_generation(self.sparse), _city_generation(self.enriched)],
            _verify_all_places,
        )
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4] as generation_mock,
            patches[5],
        ):
            result = planner.generate_itinerary(
                self.request,
                max_attempts=5,
            )

        self.assertEqual(result.problems, [])
        self.assertEqual(result.attempts, 2)
        self.assertEqual(generation_mock.call_count, 2)
        self.assertEqual(
            [
                sum(item.kind == "attraction" for item in day_plan.items)
                for day_plan in result.itinerary.days
            ],
            [3, 3, 3, 3],
        )

        late_finish_threshold = 17 * 60 + 30
        for day_plan in result.itinerary.days:
            late_attractions = []
            for item in day_plan.items:
                if item.kind != "attraction":
                    continue
                hours, minutes = (int(part) for part in item.time.split(":"))
                if hours * 60 + minutes + item.duration_min >= late_finish_threshold:
                    late_attractions.append(item)
            self.assertTrue(late_attractions, f"day {day_plan.day} has no late block")

        attractions = list(_all_attractions(result.itinerary))
        self.assertTrue(
            all(
                item.verified
                and item.lat is not None
                and item.lon is not None
                and item.official_name
                for item in attractions
            )
        )
        self.assertEqual(check_verified_place_duplicates(result.itinerary), [])
        self.assertEqual(
            len({item.place_query.casefold() for item in attractions}),
            len(attractions),
        )

    def test_invalid_enrichment_returns_verified_fallback_without_third_call(self) -> None:
        verification_call = 0

        def reject_one_new_place(itinerary: Itinerary) -> list[str]:
            nonlocal verification_call
            verification_call += 1
            _verify_all_places(itinerary)
            if verification_call == 1:
                return []

            failed = next(
                item
                for item in itinerary.days[0].items
                if item.kind == "attraction" and item.time == "18:00"
            )
            failed.verified = False
            failed.lat = None
            failed.lon = None
            failed.official_name = ""
            return [
                "День 1: место "
                f"«{failed.title}» не найдено на карте ({failed.place_query})"
            ]

        generation_mock = None
        patches = self._common_patches(
            [_generation(self.sparse), _city_generation(self.enriched)],
            reject_one_new_place,
        )
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4] as generation_mock,
            patches[5],
        ):
            result = planner.generate_itinerary(
                self.request,
                max_attempts=5,
            )

        self.assertEqual(result.problems, [])
        self.assertEqual(result.attempts, 2)
        self.assertEqual(generation_mock.call_count, 2)
        self.assertEqual(verification_call, 2)
        self.assertEqual(
            [
                sum(item.kind == "attraction" for item in day_plan.items)
                for day_plan in result.itinerary.days
            ],
            [2, 2, 2, 2],
        )

        fallback_attractions = list(_all_attractions(result.itinerary))
        self.assertTrue(
            all(
                item.verified
                and item.lat is not None
                and item.lon is not None
                and item.official_name
                for item in fallback_attractions
            )
        )
        self.assertEqual(check_verified_place_duplicates(result.itinerary), [])
        self.assertEqual(
            [item.place_query for item in fallback_attractions],
            [item.place_query for item in _all_attractions(self.sparse)],
        )

    def test_duplicate_late_places_are_rejected_and_do_not_leak_into_fallback(self) -> None:
        verification_call = 0

        def map_late_places_to_one_object(itinerary: Itinerary) -> list[str]:
            nonlocal verification_call
            verification_call += 1
            _verify_all_places(itinerary)
            if verification_call == 1:
                return []

            for day_plan in itinerary.days:
                late_item = next(
                    item
                    for item in day_plan.items
                    if item.kind == "attraction" and item.time == "18:00"
                )
                late_item.lat = 67.5000
                late_item.lon = 64.0500
                late_item.official_name = "Один и тот же вечерний объект"
            return []

        generation_mock = None
        patches = self._common_patches(
            [_generation(self.sparse), _city_generation(self.enriched)],
            map_late_places_to_one_object,
        )
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4] as generation_mock,
            patches[5],
        ):
            result = planner.generate_itinerary(
                self.request,
                max_attempts=5,
            )

        self.assertEqual(result.problems, [])
        self.assertEqual(result.attempts, 2)
        self.assertEqual(generation_mock.call_count, 2)
        self.assertEqual(verification_call, 2)
        self.assertEqual(
            [
                sum(item.kind == "attraction" for item in day_plan.items)
                for day_plan in result.itinerary.days
            ],
            [2, 2, 2, 2],
        )
        self.assertEqual(check_verified_place_duplicates(result.itinerary), [])
        self.assertFalse(
            any(
                item.official_name == "Один и тот же вечерний объект"
                for item in _all_attractions(result.itinerary)
            )
        )

    def _assert_ai_failure_completed_from_catalog(
        self,
        interest_categories: list[str],
    ) -> None:
        broad_request = self.request.model_copy(
            update={
                "trip_goal": "Познакомиться с городом",
                "interest_categories": interest_categories,
                "location_preference": "anywhere",
                "stay_area": "",
                "wishes": "",
            }
        )
        trusted_hints = _trusted_evening_hints()

        with (
            patch.object(planner, "get_weather", return_value=None),
            patch.object(planner, "weather_summary", return_value=""),
            patch.object(
                planner,
                "find_city",
                return_value={"lat": 67.51, "lon": 64.06},
            ),
            patch.object(planner, "cached_place_hints", return_value=[]),
            patch.object(
                planner,
                "curated_place_hints",
                return_value=trusted_hints,
            ),
            patch.object(
                planner,
                "generate_json",
                side_effect=[
                    _generation(self.sparse),
                    planner.YandexAIError("temporary failure"),
                ],
            ) as generation_mock,
            patch.object(
                planner,
                "verify_itinerary",
                side_effect=_verify_all_places,
            ),
        ):
            result = planner.generate_itinerary(
                broad_request,
                max_attempts=5,
            )

        self.assertEqual(result.problems, [])
        self.assertEqual(result.attempts, 2)
        self.assertEqual(generation_mock.call_count, 2)
        self.assertEqual(
            [
                sum(item.kind == "attraction" for item in day_plan.items)
                for day_plan in result.itinerary.days
            ],
            [3, 3, 3, 3],
        )
        self.assertEqual(
            planner._saturated_days_needing_enrichment(
                result.itinerary,
                broad_request,
            ),
            [],
        )
        self.assertEqual(
            check_verified_place_duplicates(result.itinerary),
            [],
        )
        self.assertTrue(
            all(
                any(
                    item.kind == "attraction"
                    and item.time >= "17:30"
                    for item in day_plan.items
                )
                for day_plan in result.itinerary.days
            )
        )

    def test_ai_failure_is_completed_from_trusted_local_catalog(self) -> None:
        self._assert_ai_failure_completed_from_catalog(
            list(planner.INTEREST_CATEGORIES)
        )

    def test_default_main_sights_is_completed_after_ai_failure(self) -> None:
        self._assert_ai_failure_completed_from_catalog(["main_sights"])

    def test_museum_only_request_does_not_get_general_city_fallback(self) -> None:
        museum_request = self.request.model_copy(
            update={
                "trip_goal": "Познакомиться с городом",
                "interest_categories": ["museums"],
                "location_preference": "anywhere",
                "stay_area": "",
                "wishes": "",
            }
        )
        sparse = self.sparse.model_copy(deep=True)
        _verify_all_places(sparse)
        original = sparse.model_dump()

        result = planner._deterministic_saturated_enrichment(
            sparse,
            museum_request,
            [],
            (67.51, 64.06),
            {},
            _trusted_evening_hints(),
        )

        self.assertIsNone(result)
        self.assertEqual(sparse.model_dump(), original)

    def test_blocked_place_retries_before_map_verification(self) -> None:
        blocked = self.sparse.model_copy(deep=True)
        blocked_item = next(_all_attractions(blocked))
        blocked_item.title = "Кинотеатр Каскад"
        blocked_item.place_query = (
            "Кинотеатр Каскад, Воркута, Россия"
        )
        blocked_places = [
            {
                "title": "Кинотеатр «Каскад»",
                "aliases": ["Каскад", "Кинотеатр Каскад"],
                "place_queries": [blocked_item.place_query],
                "reason": "Временно закрыто.",
            }
        ]

        with (
            patch.object(planner, "get_weather", return_value=None),
            patch.object(planner, "weather_summary", return_value=""),
            patch.object(planner, "find_city", return_value=None),
            patch.object(planner, "cached_place_hints", return_value=[]),
            patch.object(planner, "curated_place_hints", return_value=[]),
            patch.object(
                planner,
                "curated_blocked_place_terms",
                return_value=blocked_places,
            ),
            patch.object(
                planner,
                "generate_json",
                side_effect=[
                    _generation(blocked),
                    _city_generation(self.sparse),
                ],
            ) as generation_mock,
            patch.object(
                planner,
                "verify_itinerary",
                side_effect=_verify_all_places,
            ) as verifier_mock,
        ):
            result = planner.generate_itinerary(
                self.request,
                max_attempts=2,
            )

        self.assertEqual(result.problems, [])
        self.assertEqual(generation_mock.call_count, 2)
        self.assertEqual(verifier_mock.call_count, 1)
        self.assertFalse(
            any(
                "каскад" in item.title.casefold()
                for item in _all_attractions(result.itinerary)
            )
        )
        self.assertTrue(
            planner._blocked_place_matches(
                "Культурный центр",
                "Центральная площадь, 1, Воркута, Россия",
                "Каскад, 1, Центральная площадь, Воркута",
                blocked_places[0],
            )
        )
        self.assertFalse(
            planner._blocked_place_matches(
                "Каскадный водопад",
                "Каскадный водопад, Россия",
                "Каскадный водопад",
                blocked_places[0],
            )
        )
        self.assertTrue(
            planner._blocked_place_matches(
                "Воркутинский краеведческий музей",
                "Воркутинский краеведческий музей, Воркута, Россия",
                (
                    "Воркутинский краеведческий музей, 38, улица "
                    "Ленина, Воркута, муниципальный округ Воркута"
                ),
                {
                    "aliases": [],
                    "place_queries": [],
                    "official_names": [
                        "Воркутинский краеведческий музей, 38, "
                        "улица Ленина, Воркута"
                    ],
                },
            )
        )

    def test_curated_hint_freshness_boundaries(self) -> None:
        request = self.request.model_copy(
            update={"start_date": "2026-11-03"}
        )

        def hint(title: str, verified_at: str) -> dict:
            return {
                "title": title,
                "place_query": f"{title}, Воркута, Россия",
                "official_name": title,
                "lat": 67.50,
                "lon": 64.05,
                "description": "Внешний осмотр.",
                "indoor": False,
                "access_mode": "public_exterior",
                "trusted_late": True,
                "catalog_entry": True,
                "verified_at": verified_at,
                "source": "https://example.test/place",
            }

        accepted_date = (
            date.fromisoformat(request.start_date)
            - timedelta(days=planner.CURATED_HINT_MAX_AGE_DAYS)
        ).isoformat()
        expired_date = (
            date.fromisoformat(request.start_date)
            - timedelta(days=planner.CURATED_HINT_MAX_AGE_DAYS + 1)
        ).isoformat()
        future_date = (
            date.fromisoformat(request.start_date)
            + timedelta(days=1)
        ).isoformat()
        pool = planner._trusted_saturated_hint_pool(
            request,
            [
                hint("Граница свежести", accepted_date),
                hint("Просрочено", expired_date),
                hint("Из будущего", future_date),
                hint("Некорректная дата", "не-дата"),
            ],
            {},
        )

        self.assertEqual(
            [item["title"] for item in pool],
            ["Граница свежести"],
        )

    def test_rainy_deterministic_enrichment_requires_indoor_catalog(self) -> None:
        broad_request = self.request.model_copy(
            update={
                "trip_goal": "Познакомиться с городом",
                "interest_categories": list(planner.INTEREST_CATEGORIES),
                "location_preference": "anywhere",
                "stay_area": "",
                "wishes": "",
            }
        )
        sparse = self.sparse.model_copy(deep=True)
        _verify_all_places(sparse)
        rainy_dates = [day_plan.date for day_plan in sparse.days]

        def catalog_hints(*, indoor: bool, access_mode: str) -> list[dict]:
            return [
                {
                    "title": f"Резерв {day_number}",
                    "place_query": f"Резерв {day_number}, Воркута, Россия",
                    "official_name": f"Резерв {day_number}",
                    "lat": 67.49 + day_number / 100 + 0.004,
                    "lon": 64.04 + day_number / 100 + 0.004,
                    "description": "Проверенный резерв.",
                    "indoor": indoor,
                    "access_mode": access_mode,
                    "trusted_late": True,
                    "catalog_entry": True,
                    "verified_at": "2026-10-12",
                    "source": "https://example.test/place",
                }
                for day_number in range(1, 5)
            ]

        outdoor_result = planner._deterministic_saturated_enrichment(
            sparse,
            broad_request,
            rainy_dates,
            None,
            {},
            catalog_hints(
                indoor=False,
                access_mode="public_exterior",
            ),
        )
        self.assertIsNone(outdoor_result)
        self.assertEqual(
            [
                sum(item.kind == "attraction" for item in day.items)
                for day in sparse.days
            ],
            [2, 2, 2, 2],
        )

        indoor_result = planner._deterministic_saturated_enrichment(
            sparse,
            broad_request,
            rainy_dates,
            None,
            {},
            catalog_hints(
                indoor=True,
                access_mode="public_24h",
            ),
        )
        self.assertIsNotNone(indoor_result)
        assert indoor_result is not None
        self.assertEqual(
            [
                sum(item.kind == "attraction" for item in day.items)
                for day in indoor_result.days
            ],
            [3, 3, 3, 3],
        )

    def test_final_map_repair_does_not_use_curated_hours_entry(self) -> None:
        itinerary = self.sparse.model_copy(deep=True)
        _verify_all_places(itinerary)
        target_day = itinerary.days[0]
        first = next(
            item
            for item in target_day.items
            if item.kind == "attraction"
        )
        target_day.items = [
            item
            for item in target_day.items
            if item.kind != "attraction" or item is first
        ]
        first.verified = False
        first.lat = None
        first.lon = None
        first.official_name = ""
        problem = (
            f"День 1: место «{first.title}» не найдено на карте "
            f"({first.place_query})"
        )
        curated_hours_entry = {
            "title": "Музей по расписанию",
            "place_query": "Музей по расписанию, Воркута, Россия",
            "official_name": "Музей по расписанию",
            "lat": 67.50,
            "lon": 64.05,
            "catalog_entry": True,
            "trusted_late": False,
            "access_mode": "opening_hours_required",
        }

        repaired = planner._repair_final_map_misses(
            itinerary,
            [problem],
            self.request,
            [],
            None,
            {},
            {},
            [curated_hours_entry],
        )

        self.assertIsNone(repaired)

    def test_first_pass_late_place_requires_textual_catalog_identity(self) -> None:
        broad_request = self.request.model_copy(
            update={
                "trip_goal": "Познакомиться с городом",
                "interest_categories": list(planner.INTEREST_CATEGORIES),
                "location_preference": "anywhere",
                "stay_area": "",
                "wishes": "",
            }
        )
        itinerary = self.sparse.model_copy(deep=True)
        _verify_all_places(itinerary)
        late_item = next(
            item
            for item in itinerary.days[0].items
            if item.kind == "attraction" and item.time == "15:30"
        )
        late_item.time = "18:00"
        trusted_hint = {
            "title": "Проверенная площадь",
            "place_query": "Проверенная площадь, Воркута, Россия",
            "official_name": "Проверенная площадь",
            "lat": float(late_item.lat),
            "lon": float(late_item.lon),
            "description": "Внешний осмотр.",
            "indoor": False,
            "access_mode": "public_exterior",
            "trusted_late": True,
            "catalog_entry": True,
            "verified_at": "2026-10-12",
            "source": "https://example.test/place",
        }

        spoofed_problems = planner._catalog_late_access_problems(
            itinerary,
            broad_request,
            [],
            [trusted_hint],
            {},
        )
        self.assertTrue(spoofed_problems)

        late_item.title = trusted_hint["title"]
        late_item.place_query = trusted_hint["place_query"]
        safe_problems = planner._catalog_late_access_problems(
            itinerary,
            broad_request,
            [],
            [trusted_hint],
            {},
        )
        self.assertEqual(safe_problems, [])

        late_item.lat = float(late_item.lat) + 0.02
        wrong_coordinates = planner._catalog_late_access_problems(
            itinerary,
            broad_request,
            [],
            [trusted_hint],
            {},
        )
        self.assertTrue(wrong_coordinates)

    def test_soft_enrichment_preserves_day_dates_and_order(self) -> None:
        candidate = self.enriched.model_copy(deep=True)
        _verify_all_places(candidate)
        candidate.days[0].date = "2099-01-01"

        problems = planner._soft_enrichment_problems(
            candidate,
            self.sparse,
            [1, 2, 3, 4],
            self.request,
        )

        self.assertTrue(
            any(
                "порядок или даты" in problem
                for problem in problems
            )
        )


if __name__ == "__main__":
    unittest.main()
