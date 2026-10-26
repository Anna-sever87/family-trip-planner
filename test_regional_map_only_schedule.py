"""End-to-end legacy regional schedule guards over a synthetic verified pool.

These are not real visitor facts. Only external AI/map/catalogue boundaries
are stubbed; decoding, pool binding, map verification, region, budget, duplicate,
calendar and depth validators remain real. No road route is claimed: this
legacy branch uses geometric lower bounds, not the source-backed road router.
"""

from contextlib import ExitStack
from copy import deepcopy
from datetime import date, timedelta
import json
import unittest
from unittest.mock import patch

from candidate_grounding import GroundingResult
from destination_scope import RegionScope
import places
import planner
from pricing import minimum_meal_cost
import source_planning
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest
from test_grounded_planner import _wire_response_json
from yandex_ai import GenerationResult


def fixture():
    request = TripRequest(destination="Тестовый край, Россия", destination_kind="region",
                          start_date="2099-06-01", end_date="2099-06-07", adults=2,
                          budget=140000, travel_cost=25000, lodging_cost=45000, currency="RUB",
                          pace="средний", start_time="09:00", end_time="18:00",
                          interest_categories=["museums"], wishes="Наземные переезды между базами")
    scope = RegionScope("Тестовый край, Россия", "Россия", (50,53,100,103),
                        {"type":"Polygon", "coordinates":[[[100,50],[103,50],[103,53],[100,53],[100,50]]]},
                        area_type="administrative")
    grounding = GroundingResult(completion_calls=1, input_tokens=10, output_tokens=20)
    records, days = {}, []
    for index, name in enumerate(("Город А", "Город Б")):
        query, key = f"{name}, Россия", f"base_{index+1}"
        row = dict(name=query, namedetails={"name":name}, lat=51.+.1*index, lon=101.+.2*index,
                   category="place", type="town", osm_type="node", osm_id=101+index,
                   address={"town":name, "state":"Тестовый край", "country":"Россия", "country_code":"ru"})
        records[query] = row
        grounding.base_facts.append(dict(base_key=key, base_query=query, lat=row["lat"], lon=row["lon"],
                                        official_name=query, map_base_type="town", map_address=deepcopy(row["address"])))
    for day in range(1,8):
        base = grounding.base_facts[day >= 4]
        locality = base["base_query"].split(",")[0]
        items = []
        for offset, time in enumerate(("10:00", "14:00")):
            number = 2*(day-1)+offset+1
            title = f"Музей региональной истории номер {number}"
            query = f"{title}, {locality}, Россия"
            row = dict(name=query, namedetails={"name":title}, lat=base["lat"]+.001*number,
                       lon=base["lon"]+.001*number, category="tourism", type="museum",
                       osm_type="node", osm_id=201+number, address=deepcopy(base["map_address"]))
            records[query] = row
            grounding.confirmed_hints.append(dict(title=title, place_query=query, locality=locality,
                base_key=base["base_key"], allowed_base_keys=[base["base_key"]], lat=row["lat"], lon=row["lon"],
                official_name=query, map_address=deepcopy(row["address"]),
                osm_type="node", osm_id=row["osm_id"], evidence="map_identity_and_coordinates_only"))
            items.append(ItineraryItem(time=time, title=title, area=locality, place_query=query,
                description="Осмотр экспозиции. 90 минут и 500 рублей — плановые оценки; часы и тариф уточнить.",
                duration_min=90, cost=500, indoor=False))
        items.extend([
            ItineraryItem(time="12:00", kind="meal", title="Обед", description="Резерв на обед для группы.",
                          duration_min=60, cost=minimum_meal_cost(request), indoor=False),
            ItineraryItem(time="13:00", kind="rest", title="Отдых", description="Короткая пауза по желанию.",
                          duration_min=20, cost=0, indoor=False),
        ])
        days.append(ItineraryDay(day=day, date=(date(2099,6,1)+timedelta(days=day-1)).isoformat(),
            theme="Региональные музеи", base_area=base["base_query"], items=sorted(items,key=lambda item:item.time),
            transfer_note=("Город А → Город Б: ориентировочно час наземного переезда утром. "
                           "Фактическую дорогу и условия проживания уточнить." if day == 4 else "")))
    route = Itinerary(title="Семь дней в тестовом регионе", summary="План с двумя базами, не бронирование. "
                       "Длительности, цены и переезды оценочные; часы и дорожная доступность не подтверждены.",
                       days=days, tips=["Перепроверить графики и реальную дорогу перед поездкой."])
    return request, scope, grounding, records, route


class RegionalMapOnlyScheduleTests(unittest.TestCase):
    def run_route(self, route, *, grounding_override=None, records_override=None, expected_format_failure=False):
        request, scope, grounding, records, _ = fixture()
        grounding = deepcopy(grounding if grounding_override is None else grounding_override)
        records = deepcopy(records if records_override is None else records_override)
        with ExitStack() as stack:
            for path in ("httpx.get", "httpx.post", "sqlite3.connect", "socket.socket.connect",
                         "places._save_cache", "regional_routing.build_travel_graph"):
                stack.enter_context(patch(path, side_effect=AssertionError("Unexpected external operation")))
            road = stack.enter_context(patch.object(source_planning,"build_travel_graph",
                                                    side_effect=AssertionError("No VisitFacts means no road graph")))
            cards = stack.enter_context(patch.object(source_planning,"load_visit_cards",return_value=[]))
            prepare = stack.enter_context(patch.object(planner,"prepare_source_plan",wraps=source_planning.prepare_source_plan))
            stack.enter_context(patch.object(planner,"resolve_region",return_value=scope))
            stack.enter_context(patch.object(planner,"ground_region_candidates",return_value=grounding))
            for name in ("cached_place_hints", "curated_place_hints", "_load_blocked_places"):
                stack.enter_context(patch.object(planner,name,return_value=[]))
            stack.enter_context(patch.object(places,"_cache",{key.lower():value for key,value in records.items()}))
            def geocode(query, **kwargs):
                row = records.get(query)
                if row is not None and not places._place_matches_query(query,row):
                    raise places.GeocodeMismatch("Synthetic provider context does not match")
                return deepcopy(row)
            geo = stack.enter_context(patch.object(places,"geocode",side_effect=geocode))
            completion = stack.enter_context(patch.object(planner,"generate_json",
                return_value=GenerationResult(_wire_response_json(route), 30,40)))
            if expected_format_failure:
                with self.assertRaisesRegex(RuntimeError, "Не удалось получить маршрут в нужном формате"):
                    planner.generate_itinerary(request, max_attempts=2, ground_candidates=True)
                result = None
            else:
                result = planner.generate_itinerary(request, max_attempts=2, ground_candidates=True)
            road.assert_not_called()
            return result, completion.call_count, geo.call_count, cards.call_count, prepare.call_count

    def test_week_with_two_bases_and_fourteen_unique_visits_passes_real_validators(self):
        request, _, grounding, _, route = fixture()
        before = deepcopy((grounding, route))
        result, completions, map_calls, cards, prepares = self.run_route(route)
        self.assertEqual(result.problems, [])
        self.assertEqual((completions,result.attempts,cards,prepares), (1,2,1,1))
        self.assertGreaterEqual(map_calls,21)
        self.assertEqual(len(result.itinerary.days),7)
        visits = [item for day in result.itinerary.days for item in day.items if item.kind == "attraction"]
        self.assertEqual(len({item.place_query for item in visits}),14)
        self.assertTrue(all(item.verified and item.provider == "osm" for item in visits))
        self.assertTrue(all(item.duration_min == 90 for item in visits))
        self.assertFalse(result.itinerary.source_schedule)
        self.assertFalse(result.itinerary.city_schedule)
        self.assertIsNone(result.itinerary.program_report)
        self.assertIn("не подтверждены",result.itinerary.summary)
        self.assertEqual((grounding,route),before)

    def test_thin_program_is_not_filled_by_fake_time_or_authorized_free_days(self):
        _, _, _, _, route = fixture()
        for day in route.days:
            for item in day.items:
                if item.kind == "attraction": item.duration_min=30
        result, completions, _, _, _ = self.run_route(route)
        self.assertTrue(any("недостаточно содержательная программа" in value for value in result.problems))
        self.assertEqual(completions,1)
        self.assertTrue(all(day.day_mode == "program" for day in result.itinerary.days))

    def test_missing_transfer_note_blocks_base_change(self):
        _, _, _, _, route = fixture()
        route.days[3].transfer_note=""
        result, _, _, _, _ = self.run_route(route)
        self.assertTrue(any("смене базы" in value and "переезд" in value for value in result.problems))

    def test_conditional_water_crossing_is_not_accepted_as_ground_transport(self):
        _, _, _, _, route = fixture()
        route.days[3].transfer_note="Если паром доступен, переправа из Города А в Город Б."
        result, _, _, _, _ = self.run_route(route)
        self.assertTrue(any("водного/ледового" in value for value in result.problems))

    def test_map_outside_region_cannot_pass_the_rest_of_the_pipeline(self):
        _, _, _, records, route = fixture()
        records[route.days[0].items[0].place_query]["lat"] = 60.
        result, _, _, _, _ = self.run_route(route,records_override=records)
        self.assertTrue(any("пределами выбранной территории" in value for value in result.problems))

    def test_schedule_cannot_invent_a_visit_not_in_confirmed_pool(self):
        _, _, _, _, route = fixture()
        first = route.days[0].items[0]
        first.title="Несуществующее для тестового пула место"
        first.place_query="Несуществующее для тестового пула место, Город А, Россия"
        result, completions, map_calls, _, _ = self.run_route(route,expected_format_failure=True)
        self.assertIsNone(result)
        self.assertEqual((completions,map_calls),(1,0))


if __name__ == "__main__":
    unittest.main()
