"""Explicit regional scope gets bounded discovery and honest coverage metadata."""
from contextlib import nullcontext
from copy import deepcopy
import unittest
from unittest.mock import patch

import candidate_grounding as cg
import city_candidates
import places
import regional_discovery as discovery
from grounded_schedule import _pool
from regional_schedule_witness import requested_minimum_bases, regional_coverage, build_witness
import test_regional_discovery as fx
import test_regional_map_only_schedule as route_fx


def request():
    return fx.req().model_copy(update={'wishes':'Наземные переезды между несколькими базами внутри региона; без водных и ледовых маршрутов.',
                                      'pace':'средний','start_time':'09:00','end_time':'18:00'})


class RegionalMultibaseTests(unittest.TestCase):
    def setUp(self):
        for path in ('httpx.get','httpx.post','socket.socket.connect','places._save_cache','candidate_grounding.generate_json'):
            guard=patch(path,side_effect=AssertionError('No external operations'))
            guard.start();self.addCleanup(guard.stop)
        clock=patch.object(places,'_wait_for_turn',side_effect=lambda:nullcontext())
        clock.start();self.addCleanup(clock.stop)

    def test_explicit_requirement_uses_current_request_and_respects_negation_or_permission_only(self):
        positive={'Несколько баз внутри региона':2,'Нужны две разные базы':2,
                  'Хочу три базы':3,'Путешествие со сменой жилья':2,
                  'Наземные переезды между несколькими базами без водных маршрутов':2}
        negative=('Без смены жилья','Одна база, выезды в соседние города','Не хочу несколько баз',
                  'Несколько баз не нужны','Возможна смена жилья','Смена жилья при необходимости',
                  'Можно несколько баз','Допускаю смену жилья','Много городов и достопримечательностей')
        for text,expected in [*positive.items(),*((text,1) for text in negative)]:
            with self.subTest(text=text):
                self.assertEqual(requested_minimum_bases(request().model_copy(update={'wishes':text})),expected)
        self.assertEqual(requested_minimum_bases(request().model_copy(update={'destination_kind':'city'})),1)

    def test_saved_single_base_is_partial_not_claimed_as_full_regional_trip(self):
        req,_,grounding,_,route=route_fx.fixture()
        req=req.model_copy(update={'wishes':request().wishes})
        full=regional_coverage(req,route,grounding)
        self.assertEqual((full['selected_bases'],full['quality_status']),(2,'complete'))
        for day in route.days:day.base_area=grounding.base_facts[0]['base_query']
        before=route.model_dump()
        partial=regional_coverage(req,route,grounding)
        self.assertEqual(partial['quality_status'],'partial')
        self.assertEqual(partial['quality_reasons'],['regional_scope_narrowed'])
        self.assertEqual(partial['requested_bases'],2)
        self.assertEqual(route.model_dump(),before)
        observed=regional_coverage(req,route)
        self.assertEqual(observed['quality_status'],'partial')
        self.assertEqual(observed['selected_bases'],1)
        self.assertNotIn('confirmed_bases',observed)

    def test_full_city_cache_still_requests_multibase_reserve(self):
        req,_,grounding,_,_=route_fx.fixture()
        grounding.base_facts=grounding.base_facts[:1]
        for hint in grounding.confirmed_hints:
            hint.update(base_key='base_1',allowed_base_keys=['base_1'],map_poi_type='museum')
        req=req.model_copy(update={'wishes':request().wishes})
        self.assertTrue(cg._regional_needs_reserve(req,grounding,14))

    def test_secondary_cache_base_requires_all_geography_and_source_identity_gates(self):
        existing=fx.base();valid=fx.record('Малый посёлок',kind='village',identity=201,lon=101.5)
        mutations=(lambda r:r.update(osm_id=None),lambda r:r['address'].update(country_code='fi'),
                   lambda r:r['address'].update(country='Finland'),lambda r:r['address'].update(state='Иной край'),
                   lambda r:r.update(lat=60),lambda r:r.update(lon=103.0),lambda r:r.update(type='island'))
        for mutate in mutations:
            row=deepcopy(valid);mutate(row)
            with self.subTest(row=row),patch.object(places,'_load_cache',return_value={'candidate':row}),\
                 patch.object(discovery,'discover_town',return_value=None) as town:
                self.assertIsNone(discovery.discover_secondary_base(fx.scope(),fx.counter(),(existing,)))
                town.assert_called_once()
        with patch.object(places,'_load_cache',return_value={'existing':fx.record(),'new':valid}),\
             patch.object(discovery,'discover_town',side_effect=AssertionError('Cached source sufficient')):
            result=fx.counter();found=discovery.discover_secondary_base(fx.scope(),result,(existing,))
            self.assertEqual(found['osm_id'],201)
            self.assertEqual(result.regional_discovery_calls,0)

    def test_distinct_neighbouring_town_is_not_rejected_by_arbitrary_three_km_radius(self):
        existing=fx.base();other=fx.record('Соседний город',identity=202,lon=101.02)
        self.assertLess(places.distance_km(existing['lat'],existing['lon'],other['lat'],other['lon']),3)
        with patch.object(places,'_load_cache',return_value={'new':other}):
            self.assertEqual(discovery.discover_secondary_base(fx.scope(),fx.counter(),(existing,))['osm_id'],202)

    def test_local_cache_poi_is_not_hidden_by_city_only_global_name_hints(self):
        row=fx.record('Церковь местного поселения',kind='place_of_worship',identity=900)
        row['category']='amenity'
        with patch.object(places,'_load_cache',return_value={'local':row}),\
             patch.object(city_candidates,'_category_search',return_value=[]):
            found=discovery.category_records(request(),fx.scope(),fx.base(),fx.counter())
        self.assertEqual(len(found),1)
        self.assertEqual(places.map_record_identity(found[0]),('osm','node',900))
        bad=deepcopy(row);bad['type']=[]
        with patch.object(places,'_load_cache',return_value={'malformed':bad}),\
             patch.object(city_candidates,'_category_search',return_value=[]):
            self.assertEqual(discovery.category_records(request(),fx.scope(),fx.base(),fx.counter()),[])

    def test_network_search_skips_existing_largest_city_before_exact_lookup(self):
        existing=fx.base();other=fx.record('Малый город',kind='town',identity=202,lon=101.5)
        payload={'elements':[{'type':'node','id':row['osm_id'],'lat':row['lat'],'lon':row['lon'],
                             'tags':{'place':row['type'],'name':row['namedetails']['name'],'population':population}}
                             for row,population in ((fx.record(),'900000'),(other,'30000'))]}
        with patch.object(places,'_load_cache',return_value={}),\
             patch('httpx.get',side_effect=[fx.response(payload),fx.response([fx.raw(other)])]) as get:
            result=fx.counter();found=discovery.discover_town(fx.scope(),result,near_bases=(existing,),exclude_bases=(existing,))
        self.assertEqual(found['osm_id'],202)
        self.assertEqual(get.call_args_list[1].kwargs['params']['osm_ids'],'N202')
        self.assertEqual(result.regional_discovery_calls,2)

    def test_warm_full_pool_adds_second_base_and_local_program_within_same_http_ceiling(self):
        req=request();scope=fx.scope();first=fx.base();bases={'base_1':first}
        second=fx.record('Малый город',kind='town',identity=202,lon=101.5)
        second['address']['city']='Малый город'
        result=fx.counter();result.base_facts=[first]
        for index in range(18):
            row=fx.record(f'Музей истории номер {index}',kind='museum',identity=500+index,lon=101.001+index*.001)
            row=city_candidates._canonical_record(row['namedetails']['name'],row,{'lat':51.,'lon':101.,'country_code':'ru'})
            fact,reason=cg._confirmed_fact(row['place_query'].split(',')[0],row,bases,scope,result,preferred_base_key='base_1')
            self.assertIsNotNone(fact,reason);result.confirmed_hints.append(fact)
        seen=[]
        def categories(category,country,center,scratch):
            seen.append((category,center['lon']))
            kind={'museums':'museum','parks':'park','main_sights':'attraction'}[category]
            rows=[]
            for index in range(3):
                row=fx.record(f'Местный {kind} номер {index}',kind=kind,identity=700+len(seen)*10+index,lon=center['lon']+.001*index)
                row['address']['city']='Малый город' if center['lon']>101.2 else 'Большой город'
                if kind=='park':row['category']='leisure'
                rows.append({'title':row['namedetails']['name'],'record':row})
            return rows
        with patch.object(places,'_load_cache',return_value={'town':second}),\
             patch.object(places,'_remember_nearby_alias'),patch.object(city_candidates,'_category_search',side_effect=categories):
            cg._regional_map_reserve(req,scope,bases,result,18)
        self.assertEqual(len(result.base_facts),2)
        self.assertLessEqual(len(result.confirmed_hints),18)
        self.assertLessEqual(result.regional_discovery_calls,8)
        self.assertLessEqual(result.reserved_http_attempts_upper_bound,42)
        self.assertEqual(seen[0][1],101.5) # The new base gets its own search first.
        witness=build_witness(req,*_pool(result),5,minimum_bases=2)
        self.assertIsNotNone(witness)
        self.assertEqual(len({day['base_area'] for day in witness['days']}),2)
        self.assertTrue(all(day['visit_minutes']>=180 for day in witness['days']))

    def test_new_base_pairs_are_refreshed_only_from_valid_saved_provider_evidence(self):
        for valid in (True,False):
            with self.subTest(valid=valid):
                req=request();scope=fx.scope();first=fx.base();bases={'base_1':first}
                second=fx.record('Малый город',kind='town',identity=202,lon=101.5)
                second['address']['city']='Малый город'
                row=fx.record('Городской музей истории',kind='museum',identity=501,lon=101.01)
                row=city_candidates._canonical_record('Городской музей истории',row,{'lat':51.,'lon':101.,'country_code':'ru'})
                result=fx.counter();result.base_facts=[first]
                fact,reason=cg._confirmed_fact('Городской музей истории',row,bases,scope,result,preferred_base_key='base_1')
                self.assertIsNotNone(fact,reason)
                if not valid:fact['map_address']['country_code']='fi'
                result.confirmed_hints=[fact]
                with patch.object(places,'_load_cache',return_value={'town':second}),\
                     patch.object(places,'_remember_nearby_alias'),patch.object(city_candidates,'_category_search',return_value=[]):
                    cg._regional_map_reserve(req,scope,bases,result,18)
                self.assertEqual('base_2' in fact['allowed_base_keys'],valid)

    def test_three_requested_bases_share_original_eight_call_ceiling(self):
        req=request().model_copy(update={'wishes':'Хочу три базы'})
        result=fx.counter();bases={}
        for index in range(3):
            row=fx.record(f'Город номер {index}',kind='town',identity=100+index,lon=101.+index*.2)
            fact=fx.base(row,key=f'base_{index+1}');bases[fact['base_key']]=fact
        result.base_facts=list(bases.values());seen=[]
        def empty(category,country,center,scratch):seen.append(center['lon']);return []
        with patch.object(places,'_load_cache',return_value={}),patch.object(city_candidates,'_category_search',side_effect=empty):
            cg._regional_map_reserve(req,fx.scope(),bases,result,18)
        self.assertEqual(set(seen),{101.,101.2,101.4})
        self.assertEqual((len(seen),result.regional_discovery_calls),(8,8))


if __name__=='__main__':unittest.main()
