"""Regional reserve is source-bound, finite and entirely offline in tests."""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import Mock, patch

import httpx
import candidate_grounding as grounding
import city_candidates
import places
import regional_discovery as discovery
from destination_scope import RegionScope
from schemas import TripRequest
from yandex_ai import GenerationResult


def scope():
    return RegionScope('Тестовый край, Россия', 'Россия', (50, 53, 100, 103),
                       {'type':'Polygon','coordinates':[[[100,50],[103,50],[103,53],[100,53],[100,50]]]},
                       area_type='administrative')


def record(title='Большой город', *, kind='city', identity=100, county='Первый район', lon=101.):
    return {'name':f'{title}, Тестовый край, Россия','lat':51.,'lon':lon,'type':kind,
            'category':'place' if kind in {'city','town','village'} else 'tourism','addresstype':kind,
            'osm_type':'node','osm_id':identity,'namedetails':{'name':title},
            'address':{'city':'Большой город','county':county,'state':'Тестовый край',
                       'country':'Россия','country_code':'ru'}}


def raw(row):
    return {**deepcopy(row), 'display_name':row['name']}


def base(row=None, key='base_1'):
    row = row or record()
    return {'base_key':key, 'base_query':row['name'], 'lat':row['lat'],'lon':row['lon'],
            'official_name':row['name'], 'map_base_type':row['type'], 'map_address':row['address']}


def response(payload):
    return Mock(content=json.dumps(payload).encode(), json=Mock(return_value=payload), raise_for_status=Mock())


def counter(cap=42, reserved=0):
    return grounding.GroundingResult(http_attempt_limit=cap, reserved_http_attempts_upper_bound=reserved, query_limit=21)


def req():
    return TripRequest(destination='Тестовый край, Россия',destination_kind='region',
                       start_date='2099-06-01',end_date='2099-06-07',budget=140000,
                       interest_categories=['museums','parks','main_sights'])


class RegionalDiscoveryTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((places,'_save_cache'), (grounding,'generate_json')):
            patcher=patch.object(owner,name,side_effect=AssertionError('No writes or paid calls'))
            patcher.start(); self.addCleanup(patcher.stop)
        clock=patch.object(places,'_wait_for_turn',side_effect=lambda:nullcontext())
        clock.start(); self.addCleanup(clock.stop)
        network=patch.object(httpx,'get',side_effect=AssertionError('Unexpected network'))
        self.network=network.start();self.addCleanup(network.stop)
        cache=patch.object(places,'_load_cache',return_value={})
        cache.start();self.addCleanup(cache.stop)

    def town(self, row=None, *, payload=None):
        row=record() if row is None else row
        payload=payload if payload is not None else {'elements':[{'type':'node','id':100,'lat':51.,'lon':101.,
                    'tags':{'name':'Большой город','place':'city','population':'90000'}}]}
        self.network.side_effect=[response(payload), response([raw(row)])]
        result=counter()
        found=discovery.discover_town(scope(),result)
        return found,result

    def test_verified_town_uses_two_charged_calls_exact_identity_and_no_write(self):
        found,result=self.town()
        self.assertEqual(found['base_query'],'Большой город, Россия')
        self.assertEqual((result.regional_discovery_calls,result.reserved_http_attempts_upper_bound),(2,2))
        calls=self.network.call_args_list
        self.assertIn('out body center 40',calls[0].kwargs['params']['data'])
        self.assertEqual(calls[1].kwargs['params']['osm_ids'],'N100')

    def test_exact_cached_region_relation_admin_centre_uses_two_calls(self):
        area=scope()
        cached={**record('Тестовый край',kind='administrative',identity=500),
                'name':area.name,'osm_type':'relation','category':'boundary','geojson':area.geometry}
        relation={'elements':[{'type':'relation','id':500,'tags':{'boundary':'administrative'},
                   'members':[{'type':'node','ref':100,'role':'admin_centre'}]}]}
        self.network.side_effect=[response(relation),response([raw(record())])]
        with patch.object(places,'_load_cache',return_value={'region':cached}):
            result=counter(); town=discovery.discover_town(area,result)
        self.assertEqual(town['osm_id'],100)
        self.assertIn('/relation/500.json',self.network.call_args_list[0].args[0])
        self.assertEqual(result.regional_discovery_calls,2)

    def test_malformed_cache_types_do_not_crash_discovery(self):
        for cache in ([],None,{'bad':{'geojson':scope().geometry,'name':scope().name,'type':[]}}):
            self.network.side_effect=[response({'elements':[]})]
            with patch.object(places,'_load_cache',return_value=cache):
                self.assertIsNone(discovery.discover_town(scope(),counter()))

    def test_town_rejects_identity_country_admin_polygon_type_and_title_changes(self):
        mutations=[lambda r:r.update(osm_id=101),lambda r:r['address'].update(country_code='fi'),
                   lambda r:r['address'].update(state='Другой край'),lambda r:r.update(lat=60.),
                   lambda r:r.update(type='island',addresstype='island'),
                   lambda r:r.update(name='Чужой город, Россия',namedetails={'name':'Чужой город'})]
        for mutate in mutations:
            row=record();mutate(row)
            with self.subTest(row=row):
                self.assertIsNone(self.town(row)[0])

    def test_town_incomplete_or_oversized_response_is_not_silent_success(self):
        for payload in ({'remark':'timeout','elements':[]},{'elements':[{}]*41},{'elements':None},[]):
            self.network.side_effect=[response(payload)]
            self.assertIsNone(discovery.discover_town(scope(),counter()))

    def test_town_call_ceiling_and_nonadmin_region_have_no_network(self):
        self.assertIsNone(discovery.discover_town(scope(),counter(0)))
        self.assertIsNone(discovery.discover_town(replace(scope(),area_type='lake'),counter()))
        self.network.assert_not_called()

    def test_town_network_error_is_one_attempt_and_reserved(self):
        self.network.side_effect=httpx.ConnectError('not logged')
        result=counter()
        self.assertIsNone(discovery.discover_town(scope(),result))
        self.assertEqual(result.regional_discovery_calls,1)
        self.assertEqual(self.network.call_count,1)

    def test_category_records_use_actual_names_identity_and_three_calls_at_most(self):
        row=record('Музей культуры',kind='museum',identity=200)
        self.network.side_effect=[response([raw(row)]),response([]),response([])]
        result=counter()
        rows=discovery.category_records(req(),scope(),base(),result)
        self.assertEqual(len(rows),1)
        self.assertEqual(places.map_record_identity(rows[0]),('osm','node',200))
        self.assertEqual(rows[0]['place_query'],'Музей культуры, Большой город, Россия')
        self.assertEqual((result.regional_discovery_calls,self.network.call_count),(3,3))
        self.assertEqual(self.network.call_args_list[0].kwargs['params']['q'],'[museum]')

    def test_category_foreign_outside_and_missing_identity_are_rejected(self):
        mutations=[lambda r:r.update(osm_id=None),lambda r:r['address'].update(country_code='fi'),
                   lambda r:r['address'].update(state='Чужой край'),lambda r:r.update(lat=55.),
                   lambda r:r.update(namedetails={})]
        for mutate in mutations:
            row=record('Музей культуры',kind='museum',identity=200);mutate(row)
            self.network.side_effect=[response([raw(row)]),response([]),response([])]
            self.assertEqual(discovery.category_records(req(),scope(),base(),counter()),[])

    def test_shared_limit_keeps_categories_balanced_instead_of_starving_museums(self):
        request=req().model_copy(update={'interest_categories':['main_sights','parks','museums']})
        batches=[]
        for category_index,kind in enumerate(('attraction','park','museum')):
            batch=[]
            for index in range(8):
                row=record(f'Объект {category_index} номер {index}',kind=kind,identity=400+category_index*20+index)
                if kind=='park':row['category']='leisure'
                batch.append(raw(row))
            batches.append(response(batch))
        self.network.side_effect=batches
        result=counter()
        rows=discovery.category_records(request,scope(),base(),result)
        selected=rows[:18]
        self.assertEqual([row['type'] for row in selected],['attraction','park','museum']*6)
        self.assertEqual((result.regional_discovery_calls,self.network.call_count),(3,3))

    def test_round_robin_balances_usable_categories_after_duplicates_and_adult_filter(self):
        request=req().model_copy(update={'interest_categories':['main_sights','parks','museums'], 'children_ages':[]})
        groups=[]
        for category_index,kind in enumerate(('attraction','park','museum')):
            group=[]
            for index in range(8):
                row=record(f'Объект {category_index} номер {index}',kind=kind,identity=500+category_index*20+index)
                if kind=='park':row['category']='leisure'
                group.append(row)
            if kind=='museum':
                duplicate=deepcopy(group[0]);duplicate['osm_id']=999
                group[:0]=[duplicate,record('Детский музей',kind='museum',identity=998)]
            groups.append(response([raw(row) for row in group]))
        self.network.side_effect=groups
        result=counter()
        selected=discovery.category_records(request,scope(),base(),result)[:18]
        self.assertEqual([row['type'] for row in selected],['attraction','park','museum']*6)
        self.assertEqual(len({row['place_query'] for row in selected}),18)
        self.assertEqual((result.regional_discovery_calls,self.network.call_count),(3,3))

    def warm_short_pool(self):
        request=req().model_copy(update={'interest_categories':['main_sights','parks','museums'], 'start_time':'09:00','end_time':'18:00'})
        result=counter();result.base_facts=[base()]
        for index in range(18):
            row=record(f'Памятный объект номер {index}',kind='attraction',identity=600+index)
            canonical=city_candidates._canonical_record(row['name'].split(',')[0],row,{'lat':51.,'lon':101.,'country_code':'ru'})
            fact,reason=grounding._confirmed_fact(canonical['place_query'].split(',')[0],canonical,{'base_1':base()},scope(),result)
            self.assertIsNotNone(fact,reason);result.confirmed_hints.append(fact)
        return request,result

    def test_warm_full_short_pool_is_rebalanced_within_same_http_and_candidate_caps(self):
        request,result=self.warm_short_pool()
        first=result.confirmed_hints[0]['place_query']
        self.assertTrue(grounding._regional_needs_reserve(request,result,14))
        groups=[]
        for category_index,kind in enumerate(('attraction','park','museum')):
            rows=[]
            for index in range(8):
                row=record(f'Новое место {category_index} номер {index}',kind=kind,identity=1000+category_index*20+index)
                if kind=='park':row['category']='leisure'
                rows.append(raw(row))
            groups.append(response(rows))
        self.network.side_effect=groups
        with patch.object(places,'_remember_nearby_alias'):
            grounding._regional_map_reserve(request,scope(),{'base_1':base()},result,18)
        self.assertEqual(len(result.confirmed_hints),18)
        self.assertEqual({kind:sum(row['map_poi_type']==kind for row in result.confirmed_hints)
                          for kind in ('attraction','park','museum')},
                         {'attraction':6,'park':6,'museum':6})
        self.assertIn(first,[row['place_query'] for row in result.confirmed_hints])
        self.assertFalse(grounding._regional_needs_reserve(request,result,14))
        self.assertEqual((result.regional_discovery_calls,self.network.call_count),(3,3))
        self.assertLessEqual(result.reserved_http_attempts_upper_bound,42)

    def test_rebalance_preserves_grounded_model_and_literal_user_choices(self):
        request,result=self.warm_short_pool()
        result.confirmed_hints[-1]['original_proposal_query']=result.confirmed_hints[-1]['place_query']
        request=request.model_copy(update={'wishes':f"Хочу посетить {result.confirmed_hints[-2]['title']}"})
        selected=grounding._balanced_regional_hints(request,{'base_1':base()},result.confirmed_hints,3)
        self.assertEqual({row['place_query'] for row in selected},
                         {result.confirmed_hints[index]['place_query'] for index in (-1,-2,0)})

    def test_exhausted_warm_reserve_does_not_spend_or_fabricate_new_data(self):
        request,result=self.warm_short_pool();result.reserved_http_attempts_upper_bound=42
        before=deepcopy(result.confirmed_hints)
        grounding._regional_map_reserve(request,scope(),{'base_1':base()},result,18)
        self.assertEqual(result.confirmed_hints,before)
        self.assertEqual(result.regional_discovery_calls,0)
        self.network.assert_not_called()

    def test_unknown_cached_types_are_not_inferred_from_titles(self):
        request,result=self.warm_short_pool()
        for row in result.confirmed_hints:
            row['title']='Музей '+row['title'];row.pop('map_poi_type')
        before=deepcopy(result.confirmed_hints)
        self.assertTrue(grounding._regional_needs_reserve(request,result,14))
        self.assertEqual(result.confirmed_hints,before)

    def test_unsupported_witness_profiles_keep_existing_count_based_reserve_rule(self):
        request,result=self.warm_short_pool()
        for changes in ({'children_ages':[8]}, {'interest_categories':['water_trips']}, {'pace':'спокойный'}):
            self.assertFalse(grounding._regional_needs_reserve(request.model_copy(update=changes),result,14))
        result.base_facts*=4
        self.assertFalse(grounding._regional_needs_reserve(request,result,14))

    def test_cached_alias_preserves_successful_proposal_by_provider_identity(self):
        _,result=self.warm_short_pool()
        original=result.confirmed_hints[-1]
        alias={**original,'place_query':'Alternate museum label, Большой город, Россия'}
        grounding._protect_grounded_proposal(result,alias,alias['place_query'],'Большой город')
        self.assertEqual(original['original_proposal_query'],alias['place_query'])
        other={**result.confirmed_hints[0],'osm_id':99999,'place_query':'Независимое место, Большой город, Россия'}
        grounding._protect_grounded_proposal(result,other,other['place_query'],'Большой город')
        self.assertNotIn('original_proposal_query',result.confirmed_hints[0])

    def test_map_reserve_respects_remaining_shared_http_budget(self):
        self.network.side_effect=[response([])]
        result=counter(42,41)
        self.assertEqual(discovery.category_records(req(),scope(),base(),result),[])
        self.assertEqual((result.regional_discovery_calls,result.reserved_http_attempts_upper_bound),(1,42))
        self.assertEqual(self.network.call_count,1)

    def test_total_reserve_calls_never_exceed_eight(self):
        self.network.side_effect=lambda *a,**k:response([])
        result=counter()
        for _ in range(4): discovery.category_records(req(),scope(),base(),result)
        self.assertEqual((result.regional_discovery_calls,self.network.call_count),(8,8))

    def test_county_crossing_requires_same_verified_admin_region_and_full_polygon(self):
        row=record('Музей культуры',kind='museum',county='Второй район',lon=101.1)
        row['address']['city']='Малый город'
        self.assertFalse(grounding._administratively_compatible(base()['map_address'],row['address']))
        self.assertTrue(grounding._regional_base_compatible(base(),row,scope()))
        for area in (None,replace(scope(),area_type='lake'),replace(scope(),area_type='island')):
            self.assertFalse(grounding._regional_base_compatible(base(),row,area))
        for mutate in (lambda r:r['address'].update(country_code='fi'),lambda r:r['address'].update(country='Finland'),
                       lambda r:r['address'].update(state='Другой край'),
                       lambda r:r.update(lon=103.),lambda r:r.update(lon=102.5)):
            altered=deepcopy(row);mutate(altered)
            self.assertFalse(grounding._regional_base_compatible(base(),altered,scope()))

    def test_map_only_supplement_is_accepted_only_via_original_poi_gates(self):
        row=record('Музей культуры',kind='museum',identity=200)
        canonical=city_candidates._canonical_record('Музей культуры',row,{'lat':51.,'lon':101.,'country_code':'ru'})
        result=counter(); result.base_facts=[base()]
        with patch.object(discovery,'category_records',return_value=[canonical,canonical]), \
                patch.object(places,'_remember_nearby_alias') as remember:
            grounding._regional_map_reserve(req(),scope(),{'base_1':base()},result,18)
        self.assertEqual(len(result.confirmed_hints),1)
        self.assertEqual(result.confirmed_hints[0]['evidence'],'map_identity_and_coordinates_only')
        self.assertNotIn('duration_min',result.confirmed_hints[0])
        self.assertNotIn('opening_hours',result.confirmed_hints[0])
        remember.assert_called_once()

    def test_same_canonical_query_cannot_duplicate_visit_with_another_osm_id(self):
        row=record('Музей культуры',kind='museum',identity=200)
        canonical=city_candidates._canonical_record('Музей культуры',row,{'lat':51.,'lon':101.,'country_code':'ru'})
        duplicate={**canonical,'osm_id':201,'lon':101.001}
        result=counter();result.base_facts=[base()]
        with patch.object(discovery,'category_records',return_value=[canonical,duplicate]), \
                patch.object(places,'_remember_nearby_alias'):
            grounding._regional_map_reserve(req(),scope(),{'base_1':base()},result,18)
        self.assertEqual(len(result.confirmed_hints),1)

    def test_sparse_village_pool_can_add_verified_town_without_rebinding_old_hint(self):
        village=base(record('Малое село',kind='village'))
        result=counter();result.base_facts=[village]
        prior={'title':'Старинная церковь','base_key':'base_1','allowed_base_keys':['base_1']}
        result.confirmed_hints=[deepcopy(prior)]
        town={**record(),'base_query':'Большой город, Россия'}
        with patch.object(discovery,'discover_town',return_value=town),patch.object(discovery,'category_records',return_value=[]):
            grounding._regional_map_reserve(req(),scope(),{'base_1':village},result,18)
        self.assertEqual(len(result.base_facts),2)
        self.assertEqual(result.confirmed_hints,[prior])
        self.assertEqual(result.base_facts[-1]['base_key'],'base_2')


if __name__=='__main__':unittest.main()
