"""Bound regional admin spelling, with real provider labels and synthetic maps.

The names/types/state reflect the reviewed Karelia provider response. Polygon,
coordinates and HTTP responses below are synthetic; no external service runs.
"""
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
import json
import time
import unittest
from unittest.mock import Mock, patch

import httpx
import candidate_grounding as grounding
import places
from destination_scope import RegionScope, canonical_region_base_query, check_region_itinerary
from schemas import Itinerary, ItineraryDay, TripRequest
from yandex_ai import GenerationResult


def scope():
    return RegionScope('Карелия, Северо-Западный федеральный округ, Россия', 'Россия',
        (60, 65, 29, 36), {'type': 'Polygon', 'coordinates': [
            [[29, 60], [36, 60], [36, 65], [29, 65], [29, 60]],
        ]}, area_type='administrative')


def record(city, *, country='ru', region='Карелия', kind=None, category='place', lat=62., lon=32.):
    kind = kind or ('city' if city == 'Петрозаводск' else 'town')
    return {'lat': lat, 'lon': lon, 'name': f'{city}, {region}, Россия',
            'namedetails': {'name': city}, 'category': category, 'type': kind, 'addresstype': kind,
            'address': {'city' if city == 'Петрозаводск' else 'town': city,
                        'state': region, 'country': 'Россия', 'country_code': country}}


class RegionAdminAliasTests(unittest.TestCase):
    def test_actual_provider_state_short_and_full_query_names_confirm_same_bound_base(self):
        for city in ('Петрозаводск', 'Сортавала'):
            for admin in ('Карелия', 'Республика Карелия'):
                with self.subTest(city=city, admin=admin):
                    query = f'{city}, {admin}, Россия'
                    actual = record(city)
                    before = deepcopy(actual)
                    canonical = canonical_region_base_query(query, scope())
                    self.assertEqual(canonical, f'{city}, Карелия, Россия')
                    self.assertTrue(places._place_matches_query(canonical, actual))
                    self.assertEqual(actual, before)
                    # No global weakening of the original strict identity API.
                    self.assertEqual(places._place_matches_query(query, actual), admin == 'Карелия')

    def test_reverse_spelling_uses_only_exact_provider_region_name(self):
        area = replace(scope(), name='Республика Карелия, Россия')
        self.assertEqual(canonical_region_base_query('Сортавала, Карелия, Россия', area),
                         'Сортавала, Республика Карелия, Россия')
        self.assertTrue(places._place_matches_query('Сортавала, Республика Карелия, Россия',
                                                   record('Сортавала', region='Республика Карелия')))

    def test_city_name_and_country_bytes_are_never_shortened(self):
        query = 'Республика Карелия,  Республика Карелия , RU'
        self.assertEqual(canonical_region_base_query(query, scope()), 'Республика Карелия,  Карелия , RU')
        self.assertFalse(places._place_matches_query(canonical_region_base_query(query, scope()), record('Карелия')))
        self.assertEqual(canonical_region_base_query('Республика Карелия, Россия', scope()),
                         'Республика Карелия, Россия')

    def test_other_province_country_admin_or_four_part_context_never_normalized(self):
        queries = ('Сортавала, Республика Калмыкия, Россия', 'Сортавала, Республика Корея, Россия',
                   'Сортавала, Республика Карелия, Finland', 'Сортавала, Алтайский край, Россия',
                   'Сортавала, Район, Республика Карелия, Россия', 'Сортавала, Республика, Россия',
                   'Сортавала, Карелия Республика, Россия')
        for query in queries:
            with self.subTest(query=query):
                self.assertEqual(canonical_region_base_query(query, scope()), query)
        altai = replace(scope(), name='Алтайский край, Россия')
        self.assertEqual(canonical_region_base_query('Город, Республика Алтай, Россия', altai),
                         'Город, Республика Алтай, Россия')

    def test_unbound_non_russian_or_non_administrative_scope_cannot_supply_alias(self):
        query = 'Сортавала, Республика Карелия, Россия'
        for area in (replace(scope(), geometry=None), replace(scope(), geometry={'type': 'Point', 'coordinates': [32, 62]}),
                     replace(scope(), area_type='lake'), replace(scope(), area_type=''),
                     replace(scope(), country='Finland'), replace(scope(), name='Карелия, Finland'),
                     replace(scope(), name='Карелия')):
            with self.subTest(area=area):
                self.assertEqual(canonical_region_base_query(query, area), query)

    def test_canonical_spelling_cannot_expand_base_query_beyond_schema_limit(self):
        name = 'А' * 108
        query = f'Б, {name}, Россия'
        self.assertLessEqual(len(query), 120)
        area = replace(scope(), name=f'Республика {name}, Россия')
        self.assertEqual(canonical_region_base_query(query, area), query)

    def run_grounding(self, queries, *, mutation=None, cache=None, max_http=42):
        payload = {'bases': [{'base_key': f'base_{i + 1}', 'base_query': query} for i, query in enumerate(queries)],
                   'pois': [{'title': f'Музей культуры {i}', 'place_query': f'Музей культуры {i}, Сортавала, Россия',
                             'locality': 'Сортавала', 'base_key': 'base_1'} for i in (1, 2)]}
        request = TripRequest(destination='Карелия, Россия', destination_kind='region', start_date='2099-06-01',
                              end_date='2099-06-01', budget=100000)
        completion = Mock(return_value=GenerationResult(json.dumps(payload, ensure_ascii=False), 1, 2))
        cache = {} if cache is None else cache
        def response(url, **kwargs):
            raw = record(kwargs['params']['q'].split(',')[0])
            if mutation:
                mutation(raw)
            raw['display_name'] = raw.pop('name')
            return httpx.Response(200, json=[raw], request=httpx.Request('GET', url))
        with patch.object(places, '_cache', cache), patch.object(places, '_save_cache') as save, \
                patch.object(places, '_wait_for_turn', side_effect=lambda: nullcontext()), \
                patch.object(places.httpx, 'get', side_effect=response) as get, \
                patch.object(places, 'geocode_nearby_named', return_value=None), \
                patch.object(grounding, 'destination_name_hints', return_value=[]):
            result = grounding.ground_region_candidates(request, scope(), completion_fn=completion,
                                                        max_http_attempts=max_http)
        self.assertEqual(completion.call_count, 1)
        return result, get, save

    def test_real_lookup_path_canonicalizes_before_negative_cache_and_keeps_budget(self):
        originals = ['Петрозаводск, Республика Карелия, Россия', 'Сортавала, Республика Карелия, Россия']
        cache = {query.lower() + '|miss': {'reason': 'mismatch', 'checked_at': time.time()} for query in originals}
        original_negatives = deepcopy(cache)
        result, get, _ = self.run_grounding(originals, cache=cache)
        self.assertEqual([fact['base_query'] for fact in result.base_facts],
                         ['Петрозаводск, Карелия, Россия', 'Сортавала, Карелия, Россия'])
        self.assertEqual([call.kwargs['params']['q'] for call in get.call_args_list],
                         ['Петрозаводск, Карелия, Россия', 'Сортавала, Карелия, Россия'])
        self.assertEqual((result.completion_calls, result.geocode_calls, result.verification_queries), (1, 4, 4))
        self.assertEqual(result.reserved_http_attempts_upper_bound, 8)
        self.assertTrue(all(cache[key] == value for key, value in original_negatives.items()))
        self.assertTrue(all(not fact['base_query'].startswith('Республика') for fact in result.base_facts))

    def test_fresh_canonical_negative_is_not_bypassed_or_flushed(self):
        query = 'Сортавала, Карелия, Россия'
        cache = {query.lower() + '|miss': {'reason': 'mismatch', 'checked_at': time.time()}}
        before = deepcopy(cache)
        result, get, save = self.run_grounding(['Сортавала, Республика Карелия, Россия'], cache=cache)
        self.assertEqual(result.base_facts, [])
        self.assertEqual(result.rejected[0]['reason'], 'cached_mismatch')
        get.assert_not_called(); save.assert_not_called()
        self.assertEqual(cache, before)

    def test_alias_cannot_duplicate_same_base_or_bypass_zero_budget(self):
        queries = ['Сортавала, Республика Карелия, Россия', 'Сортавала, Карелия, Россия']
        result, get, _ = self.run_grounding(queries)
        self.assertEqual(len(result.base_facts), 1)
        self.assertEqual(get.call_count, 1)
        self.assertIn('duplicate_query', [item['reason'] for item in result.rejected])
        blocked, get, _ = self.run_grounding(queries, max_http=0)
        self.assertEqual(blocked.base_facts, [])
        self.assertIn('http_budget_exhausted', [item['reason'] for item in blocked.rejected])
        get.assert_not_called()

    def test_wrong_country_province_name_invalid_coords_road_and_outside_polygon_stay_rejected(self):
        mutations = {
            'country': lambda row: row['address'].update(country_code='fi'),
            'province': lambda row: row['address'].update(state='Республика Калмыкия') or row.update(name='Сортавала, Республика Калмыкия, Россия'),
            'name': lambda row: row.update(name='Петрозаводск, Карелия, Россия', namedetails={'name': 'Петрозаводск'}),
            'coordinates': lambda row: row.update(lat=True),
            'road': lambda row: row.update(category='highway', type='trunk', addresstype='trunk'),
            'outside': lambda row: row.update(lat=50., lon=30.),
        }
        for name, mutation in mutations.items():
            with self.subTest(reason=name):
                result, _, _ = self.run_grounding(['Сортавала, Республика Карелия, Россия'], mutation=mutation)
                self.assertEqual(result.base_facts, [])
                self.assertEqual(result.confirmed_hints, [])
                self.assertEqual(result.completion_calls, 1)
                if name == 'road':
                    self.assertEqual(result.rejected[0]['reason'], 'not_a_local_base')
                if name == 'outside':
                    self.assertEqual(result.rejected[0]['reason'], 'outside_region')

    def test_followup_validator_uses_same_query_without_mutating_saved_display(self):
        base = 'Сортавала, Республика Карелия, Россия'
        itinerary = Itinerary(title='Тест', summary='Тест', days=[ItineraryDay(
            day=1, date='2099-06-01', theme='Тест', base_area=base, items=[])])
        before = itinerary.model_dump()
        with patch.object(places, 'geocode', return_value=record('Сортавала')) as geocode:
            problems = check_region_itinerary(itinerary, scope())
        geocode.assert_called_once_with('Сортавала, Карелия, Россия')
        self.assertEqual(problems, [])
        self.assertEqual(itinerary.model_dump(), before)


if __name__ == '__main__':
    unittest.main()
