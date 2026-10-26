"""A feasible map-only witness never invents visitor facts or relaxes guards."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from grounded_schedule import _pool, build_grounded_messages, constrain_grounded_schema
from grounded_response import build_grounded_response_schema, decode_grounded_response
import planner
from regional_schedule_witness import build_witness, check_witness, constrain_witness_schema
from rules import check_program_depth
from schemas import ItineraryItem
import test_regional_map_only_schedule as pipeline
from test_grounded_planner import _wire_response_json


def fixture():
    request, scope, grounding, records, route = pipeline.fixture()
    request=request.model_copy(update={'wishes':'', 'trip_goal':''})
    for row in grounding.confirmed_hints:
        row.update(map_category='tourism', map_poi_type='museum')
    return request, scope, grounding, records, route


def witness_route(data):
    request, scope, grounding, records, route = data
    bases, candidates = _pool(grounding)
    witness = build_witness(request, bases, candidates, 5)
    if witness is None:
        return None, route
    originals = {item.place_query: item for day in route.days for item in day.items if item.kind=='attraction'}
    for day, schedule in zip(route.days, witness['days']):
        rest = [item.model_copy(deep=True) for item in day.items if item.kind!='attraction']
        day.base_area = schedule['base_area']
        day.items = [originals[slot['place_query']].model_copy(deep=True) for slot in schedule['visits']] + rest
        slots={slot['place_query']:slot for slot in schedule['visits']}
        for item in day.items:
            if item.kind=='attraction':
                item.time,item.duration_min=slots[item.place_query]['time'],slots[item.place_query]['duration_min']
            else:
                item.time,item.duration_min=schedule[item.kind+'_time'],schedule[item.kind+'_duration_min']
        day.items.sort(key=lambda item:item.time)
        if day.day > 1 and route.days[day.day-2].base_area != day.base_area:
            day.transfer_note = (f'{route.days[day.day-2].base_area} → {day.base_area}. '
                                 f'Плановый запас {schedule["transfer_reserve_min"]} минут; фактическую дорогу уточнить.')
        else:
            day.transfer_note = ''
    return witness, route


class RegionalScheduleWitnessTests(unittest.TestCase):
    def test_seven_days_exact_fixed_blocks_and_real_full_pipeline(self):
        data=fixture(); before=deepcopy(data)
        witness, route=witness_route(deepcopy(data))
        self.assertIsNotNone(witness)
        self.assertTrue(all(day['visit_minutes']==180 for day in witness['days']))
        self.assertFalse(witness['road_access_verified'])
        self.assertFalse(witness['hours_and_prices_verified'])
        self.assertEqual(check_program_depth(route,data[0]),[])
        self.assertEqual(data,before)
        with patch.object(pipeline,'fixture',return_value=data):
            result,*_=pipeline.RegionalMapOnlyScheduleTests().run_route(route)
        self.assertEqual(result.problems,[])
        self.assertFalse(result.itinerary.source_schedule)
        self.assertEqual(len({item.place_query for day in result.itinerary.days for item in day.items if item.kind=='attraction'}),14)

    def test_two_short_visits_cannot_prove_three_hour_program(self):
        request,_,grounding,_,_=fixture()
        for row in grounding.confirmed_hints:row['map_poi_type']='attraction'
        self.assertIsNone(build_witness(request,*_pool(grounding),5))

    def test_long_window_uses_actual_240_target_without_stretching_blocks(self):
        request,_,grounding,_,_=fixture()
        request=request.model_copy(update={'start_time':'08:00','end_time':'21:00'})
        self.assertIsNone(build_witness(request,*_pool(grounding),5))

    def test_missing_observed_type_does_not_infer_duration_from_name(self):
        request,_,grounding,_,_=fixture()
        for row in grounding.confirmed_hints:row.pop('map_poi_type')
        self.assertIsNone(build_witness(request,*_pool(grounding),5))

    def test_incompatible_base_pools_do_not_mix_to_fill_days(self):
        request,_,grounding,_,_=fixture()
        request=request.model_copy(update={'end_date':'2099-06-01'})
        grounding.confirmed_hints=grounding.confirmed_hints[:1]+grounding.confirmed_hints[-1:]
        # Total 180, but one 90-minute visit at each incompatible base.
        self.assertIsNone(build_witness(request,*_pool(grounding),5))

    def test_complete_set_binding_never_changes_incompatible_output(self):
        witness,route=witness_route(fixture())
        for kind in ('missing','duplicate','foreign','wrong_base'):
            candidate=route.model_copy(deep=True)
            visits=[item for item in candidate.days[0].items if item.kind=='attraction']
            if kind=='missing':candidate.days[0].items.remove(visits[0])
            elif kind=='duplicate':visits[1].place_query=visits[0].place_query
            elif kind=='foreign':visits[0].place_query='Другой музей, Город А, Россия'
            else:candidate.days[0].base_area='Другая база, Россия'
            before=candidate.model_dump()
            self.assertTrue(check_witness(candidate,witness))
            self.assertEqual(candidate.model_dump(),before)

    def test_contract_check_never_changes_cost_prose_or_water_assertion(self):
        witness,route=witness_route(fixture())
        first=route.days[0].items[0]
        first.cost=777
        first.description='Неизвестные часы и цена; требуется уточнение.'
        route.days[0].transfer_note='Если паром доступен, переправа к базе.'
        before=route.model_dump()
        self.assertEqual(check_witness(route,witness),[])
        self.assertEqual(route.model_dump(),before)

    def test_short_model_duration_and_prose_are_rejected_without_rewriting(self):
        witness,route=witness_route(fixture())
        first=next(item for item in route.days[0].items if item.kind=='attraction')
        first.duration_min=60
        first.description='Осмотр экспозиции в течение 60 минут.'
        before=route.model_dump()
        self.assertTrue(check_witness(route,witness))
        self.assertEqual(route.model_dump(),before)

    def test_thin_live_failure_shape_is_rejected_by_wire_before_map(self):
        data=fixture();request,_,grounding,_,_=data
        witness,route=witness_route(data)
        schema=constrain_witness_schema(build_grounded_response_schema(planner.build_itinerary_json_schema(request),request),witness)
        for day in route.days:
            first=next(item for item in day.items if item.kind=='attraction')
            day.items=[item for item in day.items if item.kind!='attraction']+[first]
        with self.assertRaisesRegex(ValueError,'attractions.*границы'):
            decode_grounded_response(_wire_response_json(route),schema)

    def test_schema_does_not_mutate_input_and_binds_each_day_set(self):
        data=fixture();request,_,grounding,_,_=data
        witness,route=witness_route(data)
        schema=build_grounded_response_schema(constrain_grounded_schema(planner.build_itinerary_json_schema(request),grounding),request)
        before=deepcopy(schema)
        narrowed=constrain_witness_schema(schema,witness)
        self.assertEqual(schema,before)
        decode_grounded_response(_wire_response_json(route),narrowed)
        for day in witness['days']:
            props=narrowed['properties'][f'day_{day["day"]}']['properties']
            self.assertEqual(props['base_area']['const'],day['base_area'])
            self.assertEqual(props['attractions']['minItems'],len(day['visits']))

    def test_prompt_delivers_same_witness_not_a_claim_of_source_schedule(self):
        request,scope,grounding,_,_=fixture()
        payload=json.loads(build_grounded_messages(request,scope,grounding,5)[1]['content'])
        self.assertEqual(len(payload['schedule_witness']['days']),7)
        self.assertEqual(payload['schedule_witness']['basis'],'planning_estimates_only')
        self.assertTrue(all('duration_min' not in candidate for candidate in payload['candidates']))

    def test_two_base_example_requires_real_local_places_not_remote_overnight_labels(self):
        request,_,grounding,_,_=fixture()
        for hint in grounding.confirmed_hints:
            hint['allowed_base_keys']=['base_1','base_2']
            hint['lat']=51.01;hint['lon']=101.01
        self.assertIsNone(build_witness(request,*_pool(grounding),5,minimum_bases=2))
        witness=build_witness(request,*_pool(grounding),5)
        self.assertIsNotNone(witness)
        self.assertEqual({day['base_area'] for day in witness['days']},{grounding.base_facts[0]['base_query']})

    def test_exhausted_cpu_budget_returns_no_witness(self):
        request,_,grounding,_,_=fixture()
        with patch('regional_schedule_witness.MAX_OPTION_CHECKS',0):
            self.assertIsNone(build_witness(request,*_pool(grounding),5))

    def test_unsupported_water_interest_is_not_replaced_with_museums(self):
        request,_,grounding,_,_=fixture()
        request=request.model_copy(update={'interest_categories':['water_trips']})
        self.assertIsNone(build_witness(request,*_pool(grounding),5))


if __name__=='__main__':unittest.main()
