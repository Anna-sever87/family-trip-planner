"""Offline opt-in quality bounds, not factual indoor/duration verification."""

from copy import deepcopy
import socket
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import city_quality
import places
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def request(**changes):
    values = dict(destination="Тестовый город, Россия", start_date="2099-06-01", end_date="2099-06-01",
                  adults=2, children_ages=[], budget=60000, currency="RUB", pace="средний",
                  start_time="09:00", end_time="19:00")
    values.update(changes)
    return TripRequest(**values)


def visit(number=1, **changes):
    values = dict(time="09:00", kind="attraction", title=f"Объект {number}",
                  place_query=f"Объект {number}, Тестовый город, Россия", area="Тестовый город",
                  description="Оценка посещения", duration_min=60, cost=0, indoor=False,
                  verified=True, lat=52.0, lon=105.0, provider="osm")
    values.update(changes)
    return ItineraryItem(**values)


def pause(kind, time, duration):
    return ItineraryItem(time=time, kind=kind, title=kind, description="", duration_min=duration)


def route(items):
    return Itinerary(title="Городская программа", summary="", days=[
        ItineraryDay(day=1, date="2099-06-01", theme="", items=items)])


def records_for(items, map_type="museum"):
    return {item.place_query: {"type": map_type, "provider": item.provider,
                               "lat": item.lat, "lon": item.lon, "source_url": item.source_url}
            for item in items if item.kind == "attraction"}


class CityQualityTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((places.httpx, "get"), (places, "geocode"), (places, "_load_cache"),
                            (places, "_save_cache"), (socket.socket, "connect"),
                            (socket, "getaddrinfo"), (sqlite3, "connect")):
            handle = patch.object(owner, name, side_effect=AssertionError("No I/O in city quality"))
            handle.start()
            self.addCleanup(handle.stop)

    def check(self, items, *, trip=None, records=None, map_type="museum"):
        return city_quality.check_city_quality(route(items), trip or request(),
                                              records_for(items, map_type) if records is None else records)

    def test_explicit_outdoor_or_mixed_types_cannot_justify_indoor_true(self):
        types = ("park", "garden", "beach", "viewpoint", "nature_reserve", "waterfall", "peak", "lake",
                 "artwork", "monument", "memorial", "archaeological_site")
        for map_type in types:
            item = visit(indoor=True)
            with self.subTest(map_type=map_type):
                errors = self.check([item], map_type=map_type)
                self.assertEqual(len(errors), 2 if map_type == "peak" else 1)
                self.assertTrue(any("не обосновывает indoor=true" in error for error in errors))
                self.assertTrue(item.indoor)  # No automatic relabelling to pass rain.
                item.indoor = False
                self.assertEqual(len(self.check([item], map_type=map_type)), 1 if map_type == "peak" else 0)

    def test_replayed_peak_is_not_access_evidence_even_with_plausible_model_metadata(self):
        for minutes, cost, indoor in ((180, 0, False), (30, 200, False), (60, 100, True)):
            with self.subTest(minutes=minutes, cost=cost, indoor=indoor):
                item = visit(title="Tahtalı Dağı", duration_min=minutes, cost=cost, indoor=indoor,
                             description="Подъёмник и безопасный доступ включены")
                records = records_for([item], "peak")
                records[item.place_query].update(access="yes", cableway=True, cost=cost)
                before = deepcopy((item.model_dump(), records))
                errors = self.check([item], records=records)
                self.assertTrue(any("координата вершины" in error for error in errors))
                self.assertTrue(any("не подтверждает доступное посещение" in error for error in errors))
                self.assertEqual((item.model_dump(), records), before)
        self.assertIn("не VisitFacts", city_quality.CITY_QUALITY_RULE)

    def test_peak_guard_is_bound_to_record_and_does_not_change_region_or_named_museum(self):
        item = visit(title="Mountain Museum", duration_min=180)
        self.assertEqual(self.check([item], map_type="museum"), [])
        records = records_for([item], "peak")
        self.assertEqual(self.check([item], trip=request(destination_kind="region"), records=records), [])
        records[item.place_query]["lon"] += 1
        self.assertEqual(self.check([item], records=records), [])  # Map validation rejects wrong binding separately.

    def test_museum_pass_is_not_indoor_verification_or_title_classifier(self):
        item = visit(title="Парк скульптур при музее", indoor=True, duration_min=180)
        before = item.model_dump()
        self.assertEqual(self.check([item]), [])
        self.assertEqual(item.model_dump(), before)
        self.assertIn("Тип museum также не подтверждает indoor", city_quality.CITY_QUALITY_RULE)

    def test_observed_open_air_museum_cannot_bypass_type_guard_and_is_not_mutated(self):
        item = visit(indoor=True, duration_min=120)
        records = records_for([item])
        records[item.place_query].update(
            name="Folklorik Yörük Parkı Açık Hava Müzesi, Merkez, Kemer, Türkiye",
            namedetails={"name": "Folklorik Yörük Parkı Açık Hava Müzesi"})
        before = deepcopy((item.model_dump(), records))
        errors = self.check([item], records=records)
        self.assertEqual(len(errors), 1)
        self.assertIn("не обосновывает indoor=true", errors[0])
        self.assertEqual((item.model_dump(), records), before)
        self.assertEqual(self.check([item], records=records, trip=request(destination_kind="region")), [])
        item.indoor = False
        self.assertEqual(self.check([item], records=records), [])

    def test_open_air_name_filter_handles_unicode_case_and_exact_adjacent_words(self):
        for name in ("Folklorik Yörük Parkı Açık Hava Müzesi", "AÇIK HAVA MÜZESİ", "AÇİK HAVA MÜZESİ",
                     "Open-Air Museum", "OPEN AIR MUSEUM", "Open‑Air Museum", "Ｏｐｅｎ Ａｉｒ Museum",
                     "Музей под открытым небом", "МУЗЕЙ ПОД ОТКРЫТЫМ НЕБОМ", "Bảo tàng ngoài trời"):
            with self.subTest(name=name):
                self.assertFalse(city_quality.city_candidate_may_have_shelter({"type": "museum", "name": name}))
        for name in ("Air Museum", "Air Force Museum", "Open History Museum", "Open Airless Museum",
                     "City Museum, Open Air Street, City", "Музей авиации", "Museum of Modern Art"):
            with self.subTest(name=name):
                self.assertTrue(city_quality.city_candidate_may_have_shelter({"type": "museum", "name": name}))
        self.assertFalse(city_quality.city_candidate_may_have_shelter(None))

    def test_open_air_filter_uses_provider_namedetails_not_address_description_or_model_claims(self):
        item = visit(title="Open-Air Museum", description="Под открытым небом", indoor=True)
        records = records_for([item])
        record = records[item.place_query]
        record.update(name="City Museum, City", address={"road": "Open Air Street"},
                      description="open air", namedetails={"description:en": "Open Air"})
        self.assertEqual(self.check([item], records=records), [])
        for key, value in (("name", "Open-Air Museum"), ("name:tr", "Açık Hava Müzesi"),
                           ("official_name:ru", "Музей под открытым небом")):
            with self.subTest(key=key):
                record["namedetails"] = {key: value}
                self.assertEqual(len(self.check([item], records=records)), 1)
        record["lat"] += 1
        self.assertEqual(self.check([item], records=records), [])  # Wrong record is rejected by the map check.

    def test_weather_pool_uses_same_open_air_exclusion_without_claiming_other_museums_are_indoor(self):
        record = {"type": "museum", "name": "Folklorik Yörük Parkı Açık Hava Müzesi"}
        candidates = [SimpleNamespace(record=record)]
        before = deepcopy(record)
        problems = city_quality.city_pool_weather_problems(candidates, ["2099-06-01"])
        self.assertEqual(len(problems), 1)
        self.assertIn("только 0 мест", problems[0])
        self.assertIn("не подтверждена", problems[0])
        self.assertEqual(record, before)
        record["name"] = "Air Museum"
        self.assertEqual(city_quality.city_pool_weather_problems(candidates, ["2099-06-01"]), [])
        self.assertEqual(city_quality.city_pool_weather_problems(candidates, []), [])

    def test_uses_record_type_not_model_title_description_or_extra_metadata(self):
        item = visit(title="Крытый музей", description="Совершенно крытая экспозиция", indoor=True)
        records = records_for([item], "park")
        records[item.place_query].update(indoor=True, duration_min=180)
        self.assertEqual(len(self.check([item], records=records)), 1)
        records[item.place_query]["type"] = "museum"
        records[item.place_query]["category"] = "park"
        self.assertEqual(self.check([item], records=records), [])

    def test_stop_limit_exact_boundary_and_reproduced_inflated_artwork(self):
        for map_type in ("artwork", "monument", "memorial"):
            for minutes, count in ((59, 0), (60, 0), (61, 1), (180, 1)):
                with self.subTest(map_type=map_type, minutes=minutes):
                    errors = self.check([visit(duration_min=minutes)], map_type=map_type)
                    self.assertEqual(len(errors), count)
                    if errors:
                        self.assertIn("ограничение продукта", errors[0])
        errors = self.check([visit(indoor=True, duration_min=180)], map_type="artwork")
        self.assertEqual(len(errors), 2)

    def test_family_minimum_youngest_age_boundaries(self):
        for ages, minimum in (([5], 90), ([10, 4], 90), ([6], 90), ([7], 120), ([12, 10], 120)):
            for minutes, count in ((15, 1), (minimum - 1, 1), (minimum, 0), (minimum + 1, 0)):
                with self.subTest(ages=ages, minutes=minutes):
                    errors = self.check([visit(duration_min=minutes)], trip=request(children_ages=ages))
                    self.assertEqual(len(errors), count)
                    if errors:
                        self.assertIn(f"не менее {minimum} мин.", errors[0])

    def test_family_minimum_counts_visits_only_not_meals_rest_or_travel(self):
        items = [visit(duration_min=15), pause("meal", "12:00", 120), pause("rest", "14:00", 120)]
        itinerary = route(items)
        itinerary.days[0].transfer_note = "Дорога 180 минут"
        errors = city_quality.check_city_quality(itinerary, request(children_ages=[5]), records_for(items))
        self.assertEqual(len(errors), 1)
        self.assertIn("15 мин. посещений", errors[0])

    def test_family_minimum_sum_sensible_museum_90_minutes_passes(self):
        items = [visit(duration_min=45, indoor=True), visit(2, time="10:00", duration_min=45, indoor=True),
                 pause("meal", "13:00", 60), pause("rest", "15:00", 60)]
        self.assertEqual(self.check(items, trip=request(children_ages=[5])), [])

    def test_short_calm_implicit_and_adult_windows_do_not_gain_family_requirement(self):
        item = visit(duration_min=15)
        for changes in ({"children_ages": []}, {"children_ages": [5], "pace": "спокойный"},
                        {"children_ages": [5], "pace": "насыщенный"},
                        {"children_ages": [5], "end_time": "16:59"}):
            with self.subTest(changes=changes):
                self.assertEqual(self.check([item], trip=request(**changes)), [])
        self.assertEqual(len(self.check([item], trip=request(children_ages=[5], end_time="17:00"))), 1)
        values = request(children_ages=[5]).model_dump()
        for removed in (("start_time",), ("end_time",), ("start_time", "end_time")):
            implicit = {key: value for key, value in values.items() if key not in removed}
            self.assertEqual(self.check([item], trip=TripRequest(**implicit)), [])

    def test_reproduced_seven_km_without_transfer_gap_is_rejected(self):
        items = [visit(duration_min=90), visit(2, time="10:30", duration_min=90, lat=52.063)]
        errors = self.check(items)
        self.assertEqual(len(errors), 1)
        self.assertIn("свободно 0 мин.", errors[0])
        self.assertIn("нужно хотя бы 8 мин.", errors[0])
        self.assertIn("не пеший маршрут", errors[0])

    def test_distance_30m_exemption_and_ceiling_thresholds(self):
        for km, gap, count in ((0, 0, 0), (.03, 0, 0), (.030001, 0, 1), (.030001, 1, 0),
                               (1, 1, 0), (1.000001, 1, 1), (1.000001, 2, 0), (7, 7, 0), (7.1, 7, 1)):
            with self.subTest(km=km, gap=gap), patch.object(city_quality, "distance_km", return_value=km):
                items = [visit(), visit(2, time=f"10:{gap:02d}")]
                self.assertEqual(len(self.check(items)), count)

    def test_free_time_before_and_after_pause_can_be_added_not_counting_pause_as_travel(self):
        items = [visit(), pause("meal", "10:03", 15), visit(2, time="10:25", lat=52.05)]
        with patch.object(city_quality, "distance_km", return_value=10):
            self.assertEqual(self.check(items), [])  # 3 + 7 free minutes.
            items[-1].time = "10:24"
            errors = self.check(items)
            self.assertEqual(len(errors), 1)
            self.assertIn("свободно 9 мин.", errors[0])

    def test_occupied_pause_union_is_clipped_and_not_double_counted(self):
        items = [visit(), pause("meal", "10:02", 6), pause("rest", "10:05", 7),
                 visit(2, time="10:15", lat=52.05)]
        with patch.object(city_quality, "distance_km", return_value=5):
            self.assertEqual(self.check(items), [])  # Union 10:02–10:12, five free minutes.
        with patch.object(city_quality, "distance_km", return_value=6):
            self.assertIn("свободно 5 мин.", self.check(items)[0])
        self.assertEqual(city_quality._busy_minutes([pause("rest", "09:50", 20),
                                                    pause("meal", "10:12", 20)], 600, 615), 13)

    def test_travel_uses_time_order_and_only_consecutive_known_visits(self):
        items = [visit(3, time="11:20", lat=52.02), visit(1), visit(2, time="10:10", lat=52.01)]
        with patch.object(city_quality, "distance_km", return_value=10) as distance:
            self.assertEqual(self.check(items), [])
            self.assertEqual(distance.call_count, 2)
        items[2].verified = False
        with patch.object(city_quality, "distance_km", return_value=10) as distance:
            self.assertEqual(self.check(items), [])
            self.assertEqual(distance.call_count, 1)

    def test_travel_does_not_invent_hotel_coordinates_or_cross_day_legs(self):
        itinerary = route([visit()])
        itinerary.days[0].base_area = "Отель за городом"
        itinerary.days.append(ItineraryDay(day=2, date="2099-06-02", theme="", items=[visit(2, lat=60)]))
        with patch.object(city_quality, "distance_km", side_effect=AssertionError("No inferred legs")):
            self.assertEqual(city_quality.check_city_quality(itinerary, request(), {}), [])

    def test_record_binding_and_unverified_items_do_not_supply_type_evidence(self):
        item = visit(indoor=True, duration_min=180)
        records = records_for([item], "artwork")
        for mutation in ({"lat": 53.0}, {"lat": True}, {"provider": "2gis"}, {"type": None}):
            changed = deepcopy(records)
            changed[item.place_query].update(mutation)
            with self.subTest(mutation=mutation):
                self.assertEqual(self.check([item], records=changed), [])
        self.assertEqual(self.check([item], records={item.place_query.lower(): records[item.place_query]}), [])
        item.verified = False
        self.assertEqual(self.check([item], records=records), [])  # Mandatory map check rejects it separately.

    def test_malformed_interval_does_not_wrap_or_create_travel_credit(self):
        item = visit(time="23:30", duration_min=90)
        self.assertIsNone(city_quality._interval(item))
        for changes in ({"duration_min": True}, {"duration_min": -1}, {"duration_min": 10**100},
                        {"time": "25:00"}, {"duration_min": float("nan")}):
            changed = visit().model_copy(update=changes)
            with self.subTest(changes=changes):
                self.assertIsNone(city_quality._interval(changed))

    def test_pure_function_and_region_no_op_preserve_all_existing_data(self):
        items = [visit(indoor=True, duration_min=180), visit(2, time="12:00", lat=52.063)]
        itinerary, records = route(items), records_for(items, "artwork")
        before = deepcopy((itinerary.model_dump(), records))
        self.assertTrue(city_quality.check_city_quality(itinerary, request(), records))
        self.assertEqual((itinerary.model_dump(), records), before)
        self.assertEqual(city_quality.check_city_quality(itinerary, request(destination_kind="region"), records), [])


if __name__ == "__main__":
    unittest.main()
