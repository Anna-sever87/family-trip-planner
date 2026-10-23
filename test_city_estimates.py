"""Neutral wording cannot launder identity, schedule or adult-only errors."""

from copy import deepcopy
import socket
import sqlite3
import unittest
from unittest.mock import patch

import httpx

from city_estimates import (
    CHILD_CONDITIONS_WARNING, CHILD_MEAL_NOTE, CHILD_REST_NOTE, CHILD_VISIT_NOTE,
    MEAL_ESTIMATE_DESCRIPTION, REST_ESTIMATE_DESCRIPTION, label_city_estimates,
)
from rules import _adult_wording_problems
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def request(*, children=True):
    return TripRequest(destination="Тестгород, Россия", start_date="2099-10-01", end_date="2099-10-01",
                       adults=2, children_ages=[5] if children else [], budget=30000, currency="RUB")


def route():
    return Itinerary(title="Городская поездка", summary="Программа", days=[
        ItineraryDay(day=1, date="2099-10-01", theme="Прогулка", items=[
            ItineraryItem(kind="attraction", time="09:00", title="Городской исторический музей",
                place_query="Городской исторический музей, Тестгород, Россия", area="Тестгород",
                description="Историческая экспозиция", duration_min=120, cost=1000,
                kids_note="Есть интерактивные элементы", indoor=True,
                verified=True, lat=55.0, lon=83.0, provider="osm", official_name="Имя карты",
                source_url="https://www.openstreetmap.org/?mlat=55&mlon=83"),
            ItineraryItem(kind="meal", time="13:00", title="Обед", duration_min=60, cost=1800,
                description="Кафе с детским меню", kids_note="В кафе есть детское меню", indoor=True),
            ItineraryItem(kind="rest", time="14:00", title="Отдых", duration_min=60, cost=0,
                description="Тихая оборудованная комната", kids_note="Есть детская игровая зона", indoor=True),
        ]),
    ])


class CityEstimatesTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((httpx, "get"), (httpx, "post"), (socket.socket, "connect"),
                            (socket, "getaddrinfo"), (sqlite3, "connect")):
            handle = patch.object(owner, name, side_effect=AssertionError("No I/O in city estimate labels"))
            mock = handle.start()
            self.addCleanup(handle.stop)
            self.addCleanup(mock.assert_not_called)

    def test_observed_child_facility_claims_become_neutral_recommendations(self):
        itinerary = route()
        label_city_estimates(itinerary, request())
        visit, meal, rest = itinerary.days[0].items
        self.assertEqual(visit.kids_note, CHILD_VISIT_NOTE)
        self.assertEqual((meal.description, meal.kids_note, meal.indoor),
                         (MEAL_ESTIMATE_DESCRIPTION, CHILD_MEAL_NOTE, False))
        self.assertEqual((rest.description, rest.kids_note, rest.indoor),
                         (REST_ESTIMATE_DESCRIPTION, CHILD_REST_NOTE, False))
        self.assertIn(CHILD_CONDITIONS_WARNING, itinerary.tips)
        self.assertTrue(all(len(value) <= 180 for value in (
            MEAL_ESTIMATE_DESCRIPTION, REST_ESTIMATE_DESCRIPTION, CHILD_VISIT_NOTE, CHILD_MEAL_NOTE, CHILD_REST_NOTE)))

    def test_every_identity_time_price_duration_and_attraction_description_is_unchanged(self):
        itinerary = route()
        before = deepcopy(itinerary.model_dump())
        label_city_estimates(itinerary, request())
        after = itinerary.model_dump()
        for index, (old, new) in enumerate(zip(before["days"][0]["items"], after["days"][0]["items"])):
            expected = deepcopy(old)
            expected["kids_note"] = (CHILD_VISIT_NOTE, CHILD_MEAL_NOTE, CHILD_REST_NOTE)[index]
            if index:
                expected["description"] = (MEAL_ESTIMATE_DESCRIPTION, REST_ESTIMATE_DESCRIPTION)[index - 1]
                expected["indoor"] = False
            self.assertEqual(new, expected)
        self.assertEqual(itinerary.days[0].base_area, before["days"][0]["base_area"])
        self.assertFalse(itinerary.city_schedule)  # This helper is not authorization.
        self.assertIsNone(itinerary.program_report)

    def test_adult_nonempty_kids_notes_remain_invalid_for_every_item_kind(self):
        itinerary = route()
        original = [item.kids_note for item in itinerary.days[0].items]
        label_city_estimates(itinerary, request(children=False))
        self.assertEqual([item.kids_note for item in itinerary.days[0].items], original)
        self.assertTrue(_adult_wording_problems(itinerary))
        self.assertNotIn(CHILD_CONDITIONS_WARNING, itinerary.tips)

    def test_adult_empty_notes_stay_empty_and_no_child_warning_is_added(self):
        itinerary = route()
        for item in itinerary.days[0].items:
            item.kids_note = ""
        label_city_estimates(itinerary, request(children=False))
        self.assertTrue(all(item.kids_note == "" for item in itinerary.days[0].items))
        self.assertEqual(_adult_wording_problems(itinerary), [])
        self.assertEqual(itinerary.tips, [])

    def test_free_pauses_keep_no_map_location_or_invented_base(self):
        from city_coverage import build_city_free_day
        req = request()
        itinerary = Itinerary(title="Свободный день", summary="", days=[build_city_free_day(req, 1)])
        before = deepcopy(itinerary.model_dump())
        label_city_estimates(itinerary, req)
        day = itinerary.days[0]
        self.assertEqual(day.day_mode, "free")
        self.assertEqual((day.base_area, day.transfer_note, day.rainy_alternative), ("", "", ""))
        for index, item in enumerate(day.items):
            self.assertEqual((item.verified, item.lat, item.lon, item.place_query, item.provider, item.source_url),
                             (False, None, None, "", "", ""))
            for field in ("time", "cost", "duration_min", "title", "kind"):
                self.assertEqual(getattr(item, field), before["days"][0]["items"][index][field])

    def test_invalid_kind_and_pause_identity_are_not_cleaned_away(self):
        itinerary = route()
        meal = itinerary.days[0].items[1]
        meal.title, meal.place_query, meal.provider = "Выдуманное кафе", "Кафе, Тестгород, Россия", "model"
        unknown = meal.model_copy(update={"kind": "transfer", "description": "Недопустимый пункт"})
        itinerary.days[0].items.append(unknown)
        before = unknown.model_dump()
        label_city_estimates(itinerary, request())
        self.assertEqual((meal.title, meal.place_query, meal.provider), ("Выдуманное кафе", "Кафе, Тестгород, Россия", "model"))
        self.assertEqual(unknown.model_dump(), before)

    def test_attraction_claims_and_indoor_contradictions_are_not_fact_checked_or_repaired(self):
        itinerary = route()
        item = itinerary.days[0].items[0]
        item.description = "Крытая экспозиция под открытым небом с детской площадкой"
        original = item.model_dump()
        label_city_estimates(itinerary, request())
        self.assertEqual(item.description, original["description"])
        self.assertEqual(item.indoor, original["indoor"])

    def test_repeated_call_is_idempotent(self):
        itinerary = route()
        label_city_estimates(itinerary, request())
        before = itinerary.model_dump()
        label_city_estimates(itinerary, request())
        self.assertEqual(itinerary.model_dump(), before)
        self.assertEqual(itinerary.tips.count(CHILD_CONDITIONS_WARNING), 1)

    def test_source_and_region_inputs_fail_before_mutation(self):
        for source in (True, False):
            itinerary, req = route(), request()
            if source:
                itinerary.source_schedule = True
            else:
                req.destination_kind = "region"
            before = itinerary.model_dump()
            with self.subTest(source=source), self.assertRaises(ValueError):
                label_city_estimates(itinerary, req)
            self.assertEqual(itinerary.model_dump(), before)


if __name__ == "__main__":
    unittest.main()
