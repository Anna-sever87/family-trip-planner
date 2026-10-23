"""Автономные регрессии региональной географии и planner-flow.

Все внешние границы подменены: тест не обращается к карте, погоде, AI,
кэшу или базе данных.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import destination_scope
import planner
from destination_scope import (
    DestinationScopeError,
    RegionScope,
    build_region_context,
    check_region_itinerary,
    resolve_region,
)
from rules import check_city_radius
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


BAIKAL_BOUNDS = (50.50, 56.50, 102.00, 112.00)
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


def _official_baikal_record(**overrides):
    record = {
        "lat": 53.50,
        "lon": 108.00,
        "name": "озеро Байкал, Иркутская область, Россия",
        "boundingbox": [str(value) for value in BAIKAL_BOUNDS],
        "category": "natural",
        "type": "lake",
        "addresstype": "lake",
        "geojson": BAIKAL_GEOMETRY,
        "address": {"country": "Россия", "country_code": "ru"},
    }
    record.update(overrides)
    return record


def _scope() -> RegionScope:
    return RegionScope(
        name="озеро Байкал",
        country="Россия",
        bounds=BAIKAL_BOUNDS,
        geometry=BAIKAL_GEOMETRY,
    )


def _request(*, destination_kind="region", days=2) -> TripRequest:
    return TripRequest(
        destination=(
            "озеро Байкал, Россия"
            if destination_kind == "region"
            else "Иркутск, Россия"
        ),
        destination_kind=destination_kind,
        start_date="2099-06-01",
        end_date="2099-06-02" if days == 2 else "2099-06-01",
        adults=2,
        children_ages=[],
        budget=20_000,
        currency="RUB",
        pace="средний",
        start_time="11:00",
        end_time="18:00",
        trip_goal="Увидеть природу и конкретные места",
        interest_categories=["parks", "museums"],
    )


def _item(
    title: str,
    time: str,
    *,
    lat: float | None = None,
    lon: float | None = None,
    duration: int = 60,
    kind: str = "attraction",
    place_query: str | None = None,
) -> ItineraryItem:
    return ItineraryItem(
        time=time,
        kind=kind,
        title=title,
        area="",
        place_query=(
            place_query
            if place_query is not None
            else f"{title}, Байкал, Россия"
        ),
        description="Тестовый пункт.",
        duration_min=duration,
        cost=0,
        kids_note="",
        indoor=kind != "attraction",
        verified=kind == "attraction" and lat is not None and lon is not None,
        lat=lat,
        lon=lon,
        official_name=title if kind == "attraction" else "",
    )


def _scope_itinerary(*days: ItineraryDay) -> Itinerary:
    return Itinerary(
        title="Региональный маршрут",
        summary="Проверка географии.",
        days=list(days),
        tips=["Проверяйте переезды."],
    )


def _generated_day(day: int, date: str, base: str, *, transfer_note="") -> dict:
    return {
        "day": day,
        "date": date,
        "theme": f"День в районе {base}",
        "base_area": base,
        "transfer_note": transfer_note,
        "items": [
            {
                "time": "11:00",
                "kind": "attraction",
                "title": f"Музей {base}",
                "area": base,
                "place_query": f"Музей {base}, {base}, Россия",
                "description": "Конкретное место.",
                "duration_min": 60,
                "cost": 500,
                "kids_note": "",
                "indoor": True,
            },
            {
                "time": "13:00",
                "kind": "meal",
                "title": "Обед",
                "area": base,
                "place_query": "",
                "description": "Обед.",
                "duration_min": 60,
                "cost": 1000,
                "kids_note": "",
                "indoor": True,
            },
            {
                "time": "15:00",
                "kind": "rest",
                "title": "Отдых",
                "area": base,
                "place_query": "",
                "description": "Спокойный отдых.",
                "duration_min": 60,
                "cost": 0,
                "kids_note": "",
                "indoor": True,
            },
        ],
        "rainy_alternative": "",
    }


def _generated_itinerary_payload() -> dict:
    return {
        "title": "Две базы Байкала",
        "summary": "Конкретные места в двух частях территории.",
        "days": [
            _generated_day(1, "2099-06-01", "Листвянка"),
            _generated_day(
                2,
                "2099-06-02",
                "Хужир",
                transfer_note=(
                    "Переезд между базами ориентировочно 5 часов; "
                    "расписание и сезонность проверить отдельно."
                ),
            ),
        ],
        "tips": ["Проверить транспорт между базами."],
    }


def _base_geocoder(query: str, *_, **__):
    coordinates = {
        "Листвянка, Россия": (51.85, 104.86),
        "Слюдянка, Россия": (51.66, 103.70),
        "Хужир, Россия": (53.19, 107.34),
        "База А, Россия": (52.00, 105.00),
        "База Б, Россия": (52.20, 105.60),
    }
    point = coordinates.get(query)
    if point is None:
        return None
    return {"lat": point[0], "lon": point[1], "name": query,
            "type": "village", "category": "place", "address": {"country_code": "ru"}}


class RegionPromptContextTests(unittest.TestCase):
    def test_context_uses_resolved_coordinates_and_shared_limits_without_geocoding(self):
        geometry = {
            "type": "Polygon",
            "coordinates": [[[20, 10], [24, 10], [24, 12], [20, 12], [20, 10]]],
        }
        scope = RegionScope("Тестовая территория", "Страна", (10, 12, 20, 24), geometry)
        with patch.object(destination_scope.places, "geocode", side_effect=AssertionError("network")):
            context = build_region_context(scope)
        self.assertEqual(context["canonical_name"], scope.name)
        self.assertEqual(context["country"], scope.country)
        self.assertEqual(context["bounding_box"], {"south": 10, "north": 12, "west": 20, "east": 24})
        self.assertEqual(context["limits_km"], {
            "outside_territory_margin": destination_scope.REGION_MARGIN_KM,
            "attraction_from_daily_base": destination_scope.REGION_BASE_RADIUS_KM,
            "attractions_within_day": destination_scope.REGION_DAY_SPREAD_KM,
        })
        preview = context["boundary_preview"]
        self.assertFalse(preview["sampled"])
        self.assertFalse(preview["is_membership_test"])
        self.assertEqual(preview["rings"][0]["vertices_lon_lat"], geometry["coordinates"][0])
        self.assertNotIn("Байкал", json.dumps(context, ensure_ascii=False))
        context["boundary_preview"]["rings"][0]["vertices_lon_lat"][0][0] = 999
        self.assertEqual(geometry["coordinates"][0][0], [20, 10], "context must not mutate validation geometry")

    def test_large_boundary_preview_is_bounded_and_contains_only_real_source_vertices(self):
        ring = [[20 + index / 10000, 10 + (index % 2) / 10000] for index in range(2000)]
        ring.append(list(ring[0]))
        geometry = {"type": "Polygon", "coordinates": [ring]}
        scope = RegionScope("Большая территория", "Страна", (10, 12, 20, 24), geometry)
        context = build_region_context(scope)
        preview = context["boundary_preview"]
        self.assertTrue(preview["sampled"])
        selected = preview["rings"][0]["vertices_lon_lat"]
        self.assertLessEqual(len(selected), 96)
        self.assertTrue(all(point in ring for point in selected))
        self.assertEqual(selected[0], ring[0])
        self.assertEqual(selected[-1], ring[-1])
        self.assertEqual(preview["rings"][0]["source_vertex_count"], 2001)
        self.assertLess(len(json.dumps(context, ensure_ascii=False)), 10_000)

    def test_holes_and_omitted_islands_are_explicit_not_a_new_allowing_polygon(self):
        outer = [[20, 10], [24, 10], [24, 12], [20, 12], [20, 10]]
        hole = [[21, 10.5], [22, 10.5], [22, 11], [21, 11], [21, 10.5]]
        polygons = [[outer, hole]] + [[outer] for _ in range(15)]
        geometry = {"type": "MultiPolygon", "coordinates": polygons}
        scope = RegionScope("Островная территория", "Страна", (10, 12, 20, 24), geometry)
        context = build_region_context(scope)
        preview = context["boundary_preview"]
        self.assertEqual(preview["source_ring_count"], 17)
        self.assertEqual(preview["omitted_ring_count"], 5)
        self.assertTrue(preview["sampled"])
        self.assertEqual(preview["rings"][1]["role"], "hole")
        self.assertFalse(preview["is_membership_test"])
        self.assertLessEqual(sum(len(ring["vertices_lon_lat"]) for ring in preview["rings"]), 96)
        rules = " ".join(context["planning_rules"])
        self.assertIn("не подтверждает", rules)
        self.assertIn("следующему дню", rules)
        self.assertIn("не содержит списка подтверждённых баз", rules)
        self.assertNotIn("geometry", context)

    def test_legacy_scope_does_not_invent_a_boundary(self):
        context = build_region_context(RegionScope("Территория", "Страна", (10, 12, 20, 24)))
        preview = context["boundary_preview"]
        self.assertIsNone(preview["source_geometry_type"])
        self.assertEqual(preview["rings"], [])
        self.assertFalse(preview["is_membership_test"])


class InterbaseTransferTests(unittest.TestCase):
    def itinerary(self, first_time="10:00", *, meal=False, same_base=False, transfer_note="Ориентир переезда; проверить дорогу."):
        base = "База А" if same_base else "База Б"
        point = (52.0, 105.0) if same_base else (52.2, 105.65)
        items = [_item("Второе место", first_time, lat=point[0], lon=point[1], duration=30)]
        if meal:
            items.insert(0, _item("Обед", "09:00", kind="meal", duration=30, place_query=""))
        return _scope_itinerary(
            ItineraryDay(day=1, date="2099-06-01", theme="Первая база", base_area="База А",
                         items=[_item("Первое место", "09:00", lat=52.0, lon=105.0)]),
            ItineraryDay(day=2, date="2099-06-02", theme="Следующая база", base_area=base,
                         transfer_note=transfer_note, items=items),
        )

    def check(self, itinerary):
        with patch.object(destination_scope.places, "geocode", side_effect=_base_geocoder):
            return check_region_itinerary(itinerary, _scope(), start_time="09:00", end_time="20:00")

    def test_interbase_and_base_to_first_travel_require_combined_free_time(self):
        problems = self.check(self.itinerary())
        self.assertTrue(any("между базами" in problem for problem in problems), problems)
        self.assertFalse(any("transfer_note" in problem for problem in problems), problems)

    def test_meal_is_not_transfer_time_and_sufficient_free_gap_passes(self):
        for kind in ("meal", "rest"):
            with self.subTest(kind=kind):
                insufficient = self.itinerary("10:30", meal=True)
                insufficient.days[1].items[0].kind = kind
                problems = self.check(insufficient)
                self.assertTrue(any("между базами" in problem for problem in problems), problems)
                sufficient = self.itinerary("10:45", meal=True)
                sufficient.days[1].items[0].kind = kind
                self.assertEqual(self.check(sufficient), [])

    def test_transfer_note_is_still_required_even_with_enough_free_time(self):
        problems = self.check(self.itinerary("10:45", transfer_note=""))
        self.assertTrue(any("transfer_note" in problem for problem in problems), problems)
        self.assertFalse(any("между базами" in problem for problem in problems), problems)

    def test_same_base_does_not_add_interbase_time(self):
        self.assertEqual(self.check(self.itinerary("09:00", same_base=True, transfer_note="")), [])

    def test_collector_sums_confirmed_interbase_outbound_between_and_return_once(self):
        itinerary = self.itinerary("11:00")
        itinerary.days[1].items.append(_item("Третье место", "13:00", lat=52.2, lon=105.75, duration=30))
        minutes = {99: 9999}
        with patch.object(destination_scope.places, "geocode", side_effect=_base_geocoder) as geocode:
            problems = check_region_itinerary(itinerary, _scope(), start_time="09:00", end_time="20:00",
                                               travel_minutes_by_day=minutes)
        self.assertEqual(problems, [])
        self.assertEqual(geocode.call_count, 2, "collection must reuse existing base lookups")
        expected = sum(destination_scope.math.ceil(destination_scope.places.distance_km(*a, *b) + 10)
                       for a, b in (
                           ((52.0, 105.0), (52.2, 105.6)),
                           ((52.2, 105.6), (52.2, 105.65)),
                           ((52.2, 105.65), (52.2, 105.75)),
                           ((52.2, 105.75), (52.2, 105.6)),
                       ))
        self.assertEqual(minutes, {1: 0, 2: expected})

    def test_collector_excludes_day_with_insufficient_gap_or_missing_transfer_note(self):
        for itinerary in (self.itinerary("10:00"), self.itinerary("11:00", transfer_note="")):
            with self.subTest(itinerary=itinerary):
                minutes = {2: 9999}
                with patch.object(destination_scope.places, "geocode", side_effect=_base_geocoder):
                    problems = check_region_itinerary(itinerary, _scope(), start_time="09:00",
                                                       travel_minutes_by_day=minutes)
                self.assertTrue(problems)
                self.assertNotIn(2, minutes)
                self.assertEqual(minutes[1], 0)

    def test_collector_does_not_treat_transfer_text_or_unverified_poi_as_travel_evidence(self):
        itinerary = self.itinerary("11:00", transfer_note="Дорога 10 часов; текст не является подтверждением.")
        itinerary.days[1].items[0].verified = False
        minutes = {}
        with patch.object(destination_scope.places, "geocode", side_effect=_base_geocoder):
            check_region_itinerary(itinerary, _scope(), start_time="09:00", travel_minutes_by_day=minutes)
        self.assertEqual(minutes, {1: 0})

    def test_collector_excludes_a_day_outside_region_even_when_its_times_fit(self):
        itinerary = self.itinerary("11:00")
        itinerary.days[1].items[0].lat = 20
        itinerary.days[1].items[0].lon = 20
        minutes = {}
        with patch.object(destination_scope.places, "geocode", side_effect=_base_geocoder):
            problems = check_region_itinerary(itinerary, _scope(), start_time="09:00",
                                               travel_minutes_by_day=minutes)
        self.assertTrue(any("за пределами" in problem for problem in problems))
        self.assertNotIn(2, minutes)

    def test_collector_does_not_credit_transfer_from_outside_previous_base(self):
        itinerary = self.itinerary("14:00")

        def base_geocoder(query):
            if query == "База А, Россия":
                return {"lat": 50.7, "lon": 105.0, "name": query,
                        "type": "village", "category": "place", "address": {"country_code": "ru"}}
            return _base_geocoder(query)

        minutes = {}
        with patch.object(destination_scope.places, "geocode", side_effect=base_geocoder):
            problems = check_region_itinerary(itinerary, _scope(), start_time="09:00",
                                               travel_minutes_by_day=minutes)
        self.assertTrue(any("база «База А» находится за пределами" in problem for problem in problems))
        self.assertFalse(any(problem.startswith("День 2:") for problem in problems), problems)
        self.assertEqual(minutes, {}, "invalid previous base cannot prove the next day's transfer")

    def test_collector_keeps_confirmed_previous_base_despite_unrelated_day_error(self):
        itinerary = self.itinerary("11:00")
        itinerary.days[0].items[0].lat = 20
        itinerary.days[0].items[0].lon = 20
        minutes = {}
        with patch.object(destination_scope.places, "geocode", side_effect=_base_geocoder):
            problems = check_region_itinerary(itinerary, _scope(), start_time="09:00",
                                               travel_minutes_by_day=minutes)
        self.assertTrue(any(problem.startswith("День 1:") for problem in problems), problems)
        self.assertNotIn(1, minutes)
        self.assertGreater(minutes[2], 0, "a bad POI does not invalidate the confirmed base itself")


class DestinationScopeTests(unittest.TestCase):
    def test_resolve_official_lake_polygon_and_legacy_constructor(self):
        with patch.object(
            destination_scope.places,
            "geocode",
            return_value=_official_baikal_record(),
        ) as geocode:
            scope = resolve_region("озеро Байкал, Россия")

        geocode.assert_called_once_with(
            "озеро Байкал, Россия",
            include_geometry=True,
        )
        self.assertEqual(scope.bounds, BAIKAL_BOUNDS)
        self.assertEqual(scope.geometry, BAIKAL_GEOMETRY)
        self.assertEqual(scope.country, "Россия")

        legacy = RegionScope("Байкал", "Россия", BAIKAL_BOUNDS)
        self.assertIsNone(legacy.geometry)

    def test_homonymous_settlement_and_museum_are_not_regions(self):
        invalid_records = [
            _official_baikal_record(
                type="village",
                addresstype="village",
                category="place",
            ),
            _official_baikal_record(
                type="museum",
                addresstype="museum",
                category="tourism",
            ),
        ]
        for record in invalid_records:
            with self.subTest(record_type=record["type"]), patch.object(
                destination_scope.places,
                "geocode",
                return_value=record,
            ):
                with self.assertRaises(DestinationScopeError):
                    resolve_region("Байкал, Россия")

    def test_large_administrative_region_is_allowed(self):
        record = _official_baikal_record(
            name="Иркутская область, Россия",
            type="administrative",
            addresstype="state",
            category="boundary",
        )
        with patch.object(
            destination_scope.places,
            "geocode",
            return_value=record,
        ):
            scope = resolve_region("Иркутская область, Россия")
        self.assertEqual(scope.name, "Иркутская область, Россия")
        self.assertEqual(scope.geometry, BAIKAL_GEOMETRY)

    def test_missing_bounds_or_polygon_is_rejected(self):
        invalid_records = [
            _official_baikal_record(boundingbox=[]),
            _official_baikal_record(geojson=None),
        ]
        for record in invalid_records:
            with self.subTest(record=record), patch.object(
                destination_scope.places,
                "geocode",
                return_value=record,
            ):
                with self.assertRaises(DestinationScopeError):
                    resolve_region("озеро Байкал, Россия")

    def test_geometry_rejects_string_and_boolean_coordinates(self):
        invalid_values = ["103.20", True]
        for invalid_lon in invalid_values:
            geometry = {
                "type": "Polygon",
                "coordinates": [[
                    [invalid_lon, 51.20],
                    [110.10, 51.20],
                    [110.10, 55.90],
                    [103.20, 55.90],
                    [invalid_lon, 51.20],
                ]],
            }
            with self.subTest(invalid_lon=invalid_lon), patch.object(
                destination_scope.places,
                "geocode",
                return_value=_official_baikal_record(geojson=geometry),
            ):
                with self.assertRaises(DestinationScopeError):
                    resolve_region("озеро Байкал, Россия")

    def test_missing_bounds_fails_before_weather_city_or_ai(self):
        request = _request(days=1)
        with (
            patch.object(
                destination_scope.places,
                "geocode",
                return_value=_official_baikal_record(boundingbox=[]),
            ),
            patch.object(planner, "get_weather") as weather,
            patch.object(planner, "find_city") as find_city,
            patch.object(planner, "generate_json") as generate_json,
        ):
            with self.assertRaises(DestinationScopeError):
                planner.generate_itinerary(request, max_attempts=1)

        weather.assert_not_called()
        find_city.assert_not_called()
        generate_json.assert_not_called()

    def test_outlier_from_another_region_is_rejected(self):
        itinerary = _scope_itinerary(
            ItineraryDay(
                day=1,
                date="2099-06-01",
                theme="Неверная география",
                base_area="Листвянка",
                transfer_note="",
                items=[
                    _item(
                        "Музей Владивостока",
                        "12:00",
                        lat=43.12,
                        lon=131.89,
                        place_query="Музей Владивостока, Владивосток, Россия",
                    )
                ],
            )
        )
        with patch.object(
            destination_scope.places,
            "geocode",
            side_effect=_base_geocoder,
        ):
            problems = check_region_itinerary(
                itinerary,
                _scope(),
                start_time="11:00",
                end_time="18:00",
            )
        self.assertTrue(
            any("предел" in problem and "Владивостока" in problem for problem in problems),
            problems,
        )

    def test_polygon_rejects_point_that_is_only_inside_broad_bbox(self):
        itinerary = _scope_itinerary(
            ItineraryDay(
                day=1,
                date="2099-06-01",
                theme="Только внутри bbox",
                base_area="Листвянка",
                transfer_note="",
                items=[
                    _item(
                        "Точка вне полигона",
                        "12:00",
                        lat=50.70,
                        lon=102.30,
                    )
                ],
            )
        )
        south, north, west, east = BAIKAL_BOUNDS
        self.assertTrue(south < 50.70 < north and west < 102.30 < east)
        with patch.object(
            destination_scope.places,
            "geocode",
            side_effect=_base_geocoder,
        ):
            problems = check_region_itinerary(itinerary, _scope())
        self.assertTrue(
            any("Точка вне полигона" in problem and "предел" in problem for problem in problems),
            problems,
        )

    def test_region_does_not_apply_city_center_radius(self):
        khuzhir = _item(
            "Мыс Бурхан",
            "12:00",
            lat=53.20,
            lon=107.34,
            place_query="Мыс Бурхан, Хужир, Россия",
        )
        itinerary = _scope_itinerary(
            ItineraryDay(
                day=1,
                date="2099-06-01",
                theme="Ольхон",
                base_area="Хужир",
                transfer_note="",
                items=[khuzhir],
            )
        )
        irkutsk_center = (52.286, 104.280)
        self.assertTrue(check_city_radius(itinerary, irkutsk_center))

        with (
            patch.object(planner, "verify_itinerary", return_value=[]) as verify,
            patch.object(
                destination_scope.places,
                "geocode",
                side_effect=_base_geocoder,
            ),
        ):
            problems = planner._map_verification_problems(
                itinerary,
                irkutsk_center,
                _scope(),
            )
        self.assertEqual(problems, [])
        self.assertEqual(destination_scope.REGION_DAY_SPREAD_KM, 120.0)
        verify.assert_called_once_with(itinerary, max_day_spread_km=120.0)

    def test_missing_base_and_missing_transfer_are_reported(self):
        missing_base = _scope_itinerary(
            ItineraryDay(
                day=1,
                date="2099-06-01",
                theme="Без базы",
                base_area="",
                transfer_note="",
                items=[],
            )
        )
        with patch.object(
            destination_scope.places,
            "geocode",
            side_effect=_base_geocoder,
        ):
            base_problems = check_region_itinerary(missing_base, _scope())
        self.assertTrue(any("base_area" in problem for problem in base_problems))

        changed_base = _scope_itinerary(
            ItineraryDay(
                day=1,
                date="2099-06-01",
                theme="Юг",
                base_area="Листвянка",
                transfer_note="",
                items=[],
            ),
            ItineraryDay(
                day=2,
                date="2099-06-02",
                theme="Другой берег",
                base_area="Хужир",
                transfer_note="",
                items=[],
            ),
        )
        with patch.object(
            destination_scope.places,
            "geocode",
            side_effect=_base_geocoder,
        ):
            transfer_problems = check_region_itinerary(changed_base, _scope())
        self.assertTrue(
            any("transfer_note" in problem for problem in transfer_problems),
            transfer_problems,
        )

    def test_complete_base_query_does_not_duplicate_country(self):
        itinerary = _scope_itinerary(
            ItineraryDay(
                day=1,
                date="2099-06-01",
                theme="Полное имя базы",
                base_area="Иркутск, Россия",
                transfer_note="",
                items=[],
            )
        )

        def geocode(query: str, *_, **__):
            self.assertEqual(query, "Иркутск, Россия")
            return {"lat": 52.286, "lon": 104.280, "name": query,
                    "type": "city", "category": "place", "address": {"country_code": "ru"}}

        with patch.object(
            destination_scope.places,
            "geocode",
            side_effect=geocode,
        ) as mocked:
            problems = check_region_itinerary(itinerary, _scope())

        mocked.assert_called_once_with("Иркутск, Россия")
        self.assertFalse(
            any("не удалось подтвердить" in problem for problem in problems),
            problems,
        )

    def test_short_gap_between_attractions_is_rejected(self):
        itinerary = _scope_itinerary(
            ItineraryDay(
                day=1,
                date="2099-06-01",
                theme="Недостаточный переезд",
                base_area="База А",
                transfer_note="",
                items=[
                    _item("Точка А", "11:00", lat=52.00, lon=105.00),
                    _item("Точка Б", "12:20", lat=52.00, lon=105.60),
                ],
            )
        )
        with patch.object(
            destination_scope.places,
            "geocode",
            side_effect=_base_geocoder,
        ):
            problems = check_region_itinerary(
                itinerary,
                _scope(),
                start_time="11:00",
                end_time="18:00",
            )
        self.assertTrue(
            any("между «Точка А» и «Точка Б»" in problem for problem in problems),
            problems,
        )

    def test_meal_and_rest_do_not_count_as_transfer_time(self):
        itinerary = _scope_itinerary(
            ItineraryDay(
                day=1,
                date="2099-06-01",
                theme="Занятый промежуток",
                base_area="База А",
                transfer_note="",
                items=[
                    _item("Точка А", "11:00", lat=52.00, lon=105.00),
                    _item(
                        "Обед",
                        "12:00",
                        duration=30,
                        kind="meal",
                        place_query="",
                    ),
                    _item(
                        "Отдых",
                        "12:30",
                        duration=30,
                        kind="rest",
                        place_query="",
                    ),
                    _item("Точка Б", "13:10", lat=52.00, lon=105.60),
                ],
            )
        )
        with patch.object(
            destination_scope.places,
            "geocode",
            side_effect=_base_geocoder,
        ):
            problems = check_region_itinerary(
                itinerary,
                _scope(),
                start_time="11:00",
                end_time="18:00",
            )
        self.assertTrue(
            any("свободно 10 мин" in problem for problem in problems),
            problems,
        )

    def test_base_to_first_and_last_to_base_need_time_inside_window(self):
        itinerary = _scope_itinerary(
            ItineraryDay(
                day=1,
                date="2099-06-01",
                theme="Нет времени на дорогу от базы",
                base_area="База А",
                transfer_note="",
                items=[
                    _item(
                        "Удалённая точка",
                        "11:15",
                        lat=52.00,
                        lon=105.60,
                        duration=390,
                    )
                ],
            )
        )
        with patch.object(
            destination_scope.places,
            "geocode",
            side_effect=_base_geocoder,
        ):
            problems = check_region_itinerary(
                itinerary,
                _scope(),
                start_time="11:00",
                end_time="18:00",
            )
        base_travel_problems = [
            problem for problem in problems
            if "баз" in problem.casefold() and "достаточно времени" in problem
        ]
        self.assertGreaterEqual(len(base_travel_problems), 2, problems)

    def test_legacy_request_remains_city_and_city_schema_allows_empty_base(self):
        payload = _request(destination_kind="city", days=1).model_dump()
        payload.pop("destination_kind")
        request = TripRequest.model_validate(payload)
        schema = planner.build_itinerary_json_schema(request)
        day_schema = schema["properties"]["days"]["items"]
        self.assertEqual(request.destination_kind, "city")
        self.assertEqual(day_schema["properties"]["base_area"]["minLength"], 0)

    def test_soft_enrichment_cannot_change_base_or_transfer_on_any_day(self):
        request = _request(days=2)
        fallback = Itinerary.model_validate(_generated_itinerary_payload())
        candidate = fallback.model_copy(deep=True)
        candidate.days[0].base_area = "Другая база"
        candidate.days[1].transfer_note = "Другие условия переезда"

        problems = planner._soft_enrichment_problems(
            candidate,
            fallback,
            [1],
            request,
            trusted_late_hints=[],
        )

        self.assertTrue(
            any("День 1" in problem and "базу дня" in problem for problem in problems),
            problems,
        )
        self.assertTrue(
            any("День 2" in problem and "условия переезда" in problem for problem in problems),
            problems,
        )

    def test_unavailable_poi_blocks_route_even_when_base_is_confirmed(self):
        attraction = _item(
            "Недоступная точка",
            "12:00",
            lat=51.90,
            lon=104.90,
            place_query="Недоступная точка, Листвянка, Россия",
        )
        itinerary = _scope_itinerary(
            ItineraryDay(
                day=1,
                date="2099-06-01",
                theme="Сбой карты POI",
                base_area="Листвянка",
                transfer_note="",
                items=[attraction],
            )
        )
        queries = []

        def geocode(query: str, *_, **__):
            queries.append(query)
            if query == "Листвянка, Россия":
                return {
                    "lat": 51.85,
                    "lon": 104.86,
                    "name": "Листвянка, Иркутская область, Россия",
                    "type": "village", "category": "place", "address": {"country_code": "ru"},
                }
            raise destination_scope.places.GeocodeUnavailable(query)

        with patch.object(
            destination_scope.places,
            "geocode",
            side_effect=geocode,
        ), self.assertLogs("places", level="WARNING"):
            problems = planner._map_verification_problems(
                itinerary,
                None,
                _scope(),
                start_time="11:00",
                end_time="18:00",
            )

        self.assertIn("Недоступная точка, Листвянка, Россия", queries)
        self.assertIn("Листвянка, Россия", queries)
        self.assertTrue(
            any("карта временно недоступна" in problem for problem in problems),
            problems,
        )
        self.assertFalse(attraction.verified)
        self.assertIsNone(attraction.lat)
        self.assertIsNone(attraction.lon)

    def test_generate_region_skips_weather_and_city_and_uses_regional_schema(self):
        request = _request(days=2)
        payload = _generated_itinerary_payload()
        schemas = []

        def fake_generate_json(_messages, **kwargs):
            schemas.append(kwargs["json_schema"])
            return SimpleNamespace(
                text=json.dumps(payload, ensure_ascii=False),
                input_tokens=11,
                output_tokens=22,
            )

        scope = _scope()
        with (
            patch.object(planner, "resolve_region", return_value=scope) as resolve,
            patch.object(planner, "get_weather") as weather,
            patch.object(planner, "find_city") as find_city,
            patch.object(planner, "cached_place_hints", return_value=[]),
            patch.object(planner, "curated_place_hints", return_value=[]),
            patch.object(planner, "curated_blocked_place_terms", return_value=[]),
            patch.object(planner, "generate_json", side_effect=fake_generate_json),
            patch.object(planner, "check_rules", return_value=[]),
            patch.object(
                planner,
                "_map_verification_problems",
                return_value=[],
            ) as map_check,
        ):
            result = planner.generate_itinerary(request, max_attempts=1)

        resolve.assert_called_once_with("озеро Байкал, Россия")
        weather.assert_not_called()
        find_city.assert_not_called()
        self.assertEqual(result.problems, [])
        self.assertEqual(result.attempts, 1)
        self.assertIsNone(result.weather)
        self.assertIn("баз", result.weather_text.casefold())
        self.assertEqual(len(schemas), 1)
        day_schema = schemas[0]["properties"]["days"]["items"]
        self.assertIn("base_area", day_schema["required"])
        self.assertIn("transfer_note", day_schema["required"])
        self.assertEqual(day_schema["properties"]["base_area"]["minLength"], 1)
        checked_args = map_check.call_args.args
        self.assertIsNone(checked_args[1])
        self.assertIs(checked_args[2], scope)
        self.assertEqual(
            map_check.call_args.kwargs,
            {"start_time": "11:00", "end_time": "18:00", "travel_minutes_by_day": {}},
        )
        self.assertEqual(
            result.itinerary.days[0].items[0].place_query,
            "Музей Листвянка, Листвянка, Россия",
        )
        self.assertEqual(result.itinerary.days[1].base_area, "Хужир")

    def test_edit_region_skips_weather_and_city_and_keeps_generated_base(self):
        request = _request(days=2)
        current = Itinerary.model_validate(_generated_itinerary_payload())
        changed_day = _generated_day(
            2,
            "2099-06-02",
            "Слюдянка",
            transfer_note=(
                "Переезд из Листвянки ориентировочно 2 часа; "
                "расписание проверить отдельно."
            ),
        )
        changed_day["theme"] = "Новый день в Слюдянке"
        schemas = []

        def fake_generate_json(_messages, **kwargs):
            schemas.append(kwargs["json_schema"])
            return SimpleNamespace(
                text=json.dumps(changed_day, ensure_ascii=False),
                input_tokens=7,
                output_tokens=9,
            )

        scope = _scope()
        with (
            patch.object(planner, "resolve_region", return_value=scope) as resolve,
            patch.object(planner, "get_weather") as weather,
            patch.object(planner, "find_city") as find_city,
            patch.object(planner, "generate_json", side_effect=fake_generate_json),
            patch.object(planner, "check_rules", return_value=[]),
            patch.object(
                planner,
                "_map_verification_problems",
                return_value=[],
            ) as map_check,
            patch.object(
                planner,
                "_audit_edit_fulfillment",
                return_value=([], 0, 0),
            ),
        ):
            result = planner.edit_itinerary(
                request,
                current,
                2,
                "Сделай второй день в Слюдянке",
                max_attempts=1,
            )

        resolve.assert_called_once_with("озеро Байкал, Россия")
        weather.assert_not_called()
        find_city.assert_not_called()
        self.assertEqual(result.problems, [])
        self.assertEqual(result.attempts, 1)
        self.assertEqual(len(schemas), 1)
        self.assertIn("base_area", schemas[0]["required"])
        self.assertIn("transfer_note", schemas[0]["required"])
        checked_args = map_check.call_args.args
        self.assertIsNone(checked_args[1])
        self.assertIs(checked_args[2], scope)
        self.assertEqual(
            map_check.call_args.kwargs,
            {"start_time": "11:00", "end_time": "18:00", "travel_minutes_by_day": {}},
        )
        edited = result.itinerary.days[1]
        self.assertEqual(edited.base_area, "Слюдянка")
        self.assertEqual(
            edited.items[0].place_query,
            "Музей Слюдянка, Слюдянка, Россия",
        )
        self.assertEqual(result.itinerary.days[0], current.days[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
