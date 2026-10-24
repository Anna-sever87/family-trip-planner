"""Regional model names are joined by the server, not repeated by the model.

Synthetic reproductions of the observed aggregate failure, not assertions about
the unavailable original model response. All AI/map/cache boundaries are mocked.
"""
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import Mock, patch

import candidate_grounding as grounding
import places
from destination_scope import DestinationScopeError, resolve_region
from schemas import TripRequest
from test_region_admin_alias import scope, record
from yandex_ai import GenerationResult


def wire(count=14):
    return {'bases': [{'base_key': 'base_1', 'base_name': 'Петрозаводск'}],
            'pois': [{'title': f'Музей культуры {index}', 'locality': 'Петрозаводск', 'base_key': 'base_1'}
                     for index in range(count)]}


def request():
    return TripRequest(destination='Карелия, Россия', destination_kind='region',
                       start_date='2099-06-01', end_date='2099-06-07', budget=140000)


class RegionCandidateWireTests(unittest.TestCase):
    def run_payload(self, payload, *, base_error=None, mutation=None, http_budget=42):
        completion = Mock(return_value=GenerationResult(json.dumps(payload, ensure_ascii=False), 11, 13))
        cache = {}
        def geocode(query, **kwargs):
            self.assertEqual(kwargs, {'max_variants': 1})
            if base_error:
                raise base_error
            self.assertEqual(query, 'Петрозаводск, Карелия, Россия')
            return record('Петрозаводск')
        def nearby(title, country, lat, lon, radius, **kwargs):
            index = int(title.rsplit(' ', 1)[-1])
            actual = record('Петрозаводск', lat=62. + index * .002, lon=32.)
            actual.update(name=f'{title}, Петрозаводск, Карелия, Россия',
                          namedetails={'name': title}, category='tourism', type='museum',
                          addresstype='museum', osm_type='node', osm_id=index + 100,
                          place_query=f'{title}, Петрозаводск, Россия', locality='Петрозаводск')
            if mutation:
                mutation(actual)
            # Mirror the real provider filter boundary rather than bypassing it.
            return actual if kwargs['place_filter'](actual) else None
        with patch.object(grounding, 'generate_json', side_effect=AssertionError('No paid AI')), \
                patch.object(grounding, 'destination_name_hints', return_value=[]), \
                patch.object(places, '_cache', cache), \
                patch.object(places, '_save_cache', side_effect=AssertionError('No cache writes')), \
                patch.object(places.httpx, 'get', side_effect=AssertionError('No HTTP')), \
                patch.object(places, 'geocode', side_effect=geocode) as base_lookup, \
                patch.object(places, 'geocode_nearby_named', side_effect=nearby) as poi_lookup:
            result = grounding.ground_region_candidates(request(), scope(), completion_fn=completion,
                                                        max_http_attempts=http_budget)
        self.assertEqual(completion.call_count, 1)
        return result, completion, base_lookup, poi_lookup

    def test_new_wire_has_no_redundant_query_or_country_fields(self):
        schema = grounding.build_candidate_schema(request())
        base = schema['properties']['bases']['items']
        poi = schema['properties']['pois']['items']
        self.assertEqual(set(base['properties']), {'base_key', 'base_name'})
        self.assertEqual(set(poi['properties']), {'title', 'locality', 'base_key'})
        self.assertFalse(base['additionalProperties']); self.assertFalse(poi['additionalProperties'])
        self.assertEqual(set(base['required']), set(base['properties']))
        self.assertEqual(set(poi['required']), set(poi['properties']))

    def test_server_joins_exact_names_without_inventing_identity_or_mutating_wire(self):
        data = wire()
        before = deepcopy(data)
        batch = grounding._materialize_candidate_batch(data, scope())
        self.assertEqual(batch.bases[0].base_query, 'Петрозаводск, Карелия, Россия')
        self.assertEqual(batch.pois[0].place_query, 'Музей культуры 0, Петрозаводск, Россия')
        self.assertEqual(batch.pois[0].title, data['pois'][0]['title'])
        self.assertEqual(set(batch.pois[0].model_dump()), {'title', 'locality', 'place_query', 'base_key'})
        self.assertEqual(data, before)

    def test_fourteen_new_wire_places_reach_strict_map_gate_once_with_same_budgets(self):
        result, completion, base, poi = self.run_payload(wire())
        self.assertEqual((len(result.base_facts), len(result.confirmed_hints)), (1, 14))
        self.assertEqual((base.call_count, poi.call_count), (1, 14))
        self.assertEqual((result.completion_calls, result.geocode_calls, result.verification_queries), (1, 15, 15))
        self.assertEqual((result.reserved_http_attempts_upper_bound, result.http_attempt_limit), (30, 42))
        self.assertEqual(result.proposal_issue_counts, {})
        self.assertEqual(result.rejected, [])
        self.assertEqual(completion.call_args.kwargs['json_schema'], grounding.build_candidate_schema(request()))
        self.assertIn('Не пиши place_query', completion.call_args.args[0][0]['content'])

    def test_nonadministrative_region_is_not_invented_as_base_address_province(self):
        data = wire(2)
        data['bases'][0]['base_name'] = 'Листвянка'
        actual = record('Листвянка', region='Иркутская область', kind='village')
        for kind in ('lake', 'water', 'national_park', 'nature_reserve', 'island', 'park'):
            area = replace(scope(), name='Озеро Байкал, Россия', area_type=kind)
            with self.subTest(kind=kind):
                query = grounding._materialize_candidate_batch(data, area).bases[0].base_query
                self.assertEqual(query, 'Листвянка, Россия')
                self.assertTrue(places._place_matches_query(query, actual))
                self.assertFalse(places._place_matches_query('Листвянка, Озеро Байкал, Россия', actual))

    def test_new_wire_coastal_base_keeps_country_name_type_and_polygon_guards(self):
        area = replace(scope(), name='Озеро Байкал, Россия', area_type='lake')
        data = wire(2)
        data['bases'][0]['base_name'] = 'Листвянка'
        req = request().model_copy(update={'destination': 'Озеро Байкал, Россия', 'end_date': '2099-06-01'})
        mutations = {
            'valid': lambda row: None,
            'country': lambda row: row['address'].update(country_code='fi'),
            'name': lambda row: row.update(name='Другой город, Россия', namedetails={'name': 'Другой город'}),
            'road': lambda row: row.update(type='trunk', category='highway'),
            'outside': lambda row: row.update(lat=50.),
            'invalid_coords': lambda row: row.update(lat=True),
        }
        for name, mutate in mutations.items():
            actual = record('Листвянка', region='Иркутская область', kind='village')
            mutate(actual)
            completion = Mock(return_value=GenerationResult(json.dumps(data, ensure_ascii=False), 1, 1))
            with self.subTest(case=name), patch.object(grounding, 'destination_name_hints', return_value=[]), \
                    patch.object(places, '_cache', {}), patch.object(places, '_save_cache', side_effect=AssertionError('No writes')), \
                    patch.object(places.httpx, 'get', side_effect=AssertionError('No HTTP')), \
                    patch.object(places, 'geocode', return_value=actual) as base, \
                    patch.object(places, 'geocode_nearby_named', return_value=None) as poi:
                result = grounding.ground_region_candidates(req, area, completion_fn=completion)
            base.assert_called_once_with('Листвянка, Россия', max_variants=1)
            self.assertEqual(completion.call_count, 1)
            self.assertEqual(len(result.base_facts), int(name == 'valid'))
            if name != 'valid':
                poi.assert_not_called()
            self.assertLessEqual(result.reserved_http_attempts_upper_bound, result.http_attempt_limit)

    def test_synthetic_old_wire_reproduces_observed_counts_without_claiming_actual_response(self):
        batch = grounding._materialize_candidate_batch(wire(), scope()).model_dump()
        for poi in batch['pois']:
            poi['place_query'] = poi['place_query'].replace('Музей культуры', 'Музей искусства')
        failure = places.GeocodeIncomplete('synthetic base', reason='mismatch')
        result, _, base, poi = self.run_payload(batch, base_error=failure)
        self.assertEqual((len(result.base_facts), len(result.confirmed_hints), len(result.rejected)), (0, 0, 15))
        self.assertEqual((result.completion_calls, result.geocode_calls, result.verification_queries), (1, 1, 1))
        self.assertEqual(result.reserved_http_attempts_upper_bound, 2)
        self.assertEqual(result.rejected[0]['reason'], 'limited_search_incomplete')
        self.assertEqual([item['reason'] for item in result.rejected[1:]], ['invalid_poi_context'] * 14)
        self.assertEqual(result.proposal_issue_counts, {'title_query_mismatch': 14})
        self.assertEqual(result.lookup_issue_counts, {'base_mismatch': 1})
        self.assertEqual(base.call_count, 1); poi.assert_not_called()

    def test_legacy_explicit_foreign_country_is_never_rewritten_to_selected_country(self):
        batch = grounding._materialize_candidate_batch(wire(), scope()).model_dump()
        for poi in batch['pois']:
            poi['place_query'] = poi['place_query'].replace(', Россия', ', Finland')
        result, _, _, poi = self.run_payload(batch)
        self.assertEqual(len(result.base_facts), 1)
        self.assertEqual(result.confirmed_hints, [])
        self.assertEqual(result.proposal_issue_counts, {'country_mismatch': 14})
        poi.assert_not_called()

    def test_mixed_address_formats_extra_country_commas_and_overlong_names_fail_before_maps(self):
        variants = []
        for field, value in (('base_name', 'Петрозаводск, Finland'), ('base_query', 'Петрозаводск, Finland'),
                             ('country', 'Finland')):
            data = wire(); data['bases'][0][field] = value; variants.append(data)
        for field, value in (('place_query', 'Музей, Finland'), ('country', 'Finland'),
                             ('title', 'Музей, регион'), ('title', ' Музей'), ('locality', 'Город\nСтрана')):
            data = wire(); data['pois'][0][field] = value; variants.append(data)
        data = wire(); data['pois'][0].update(title='А' * 80, locality='Б' * 120); variants.append(data)
        for data in variants:
            with self.subTest(data=data['bases'][0]):
                result, _, base, poi = self.run_payload(data)
                self.assertEqual(result.base_facts, [])
                self.assertEqual(result.confirmed_hints, [])
                self.assertTrue(result.problems)
                base.assert_not_called(); poi.assert_not_called()

    def test_new_wire_does_not_bypass_actual_country_region_type_or_base_radius(self):
        mutations = (lambda row: row['address'].update(country_code='fi'),
                     lambda row: row['address'].update(state='Другая область'),
                     lambda row: row.update(type='city', category='place'),
                     lambda row: row.update(lat=40.),
                     lambda row: row.update(lat=64.))
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                result, _, _, _ = self.run_payload(wire(), mutation=mutation)
                self.assertEqual(len(result.base_facts), 1)
                self.assertEqual(result.confirmed_hints, [])
                self.assertEqual(result.completion_calls, 1)

    def test_empty_or_mismatched_base_lookup_is_distinct_without_paid_or_http_retry(self):
        for reason in ('mismatch', 'not_found'):
            with self.subTest(reason=reason):
                result, _, base, poi = self.run_payload(wire(), base_error=places.GeocodeIncomplete('private query', reason=reason))
                self.assertEqual(result.lookup_issue_counts, {'base_' + reason: 1})
                self.assertEqual(result.geocode_calls, 1)
                self.assertEqual(result.completion_calls, 1)
                self.assertEqual(result.proposal_issue_counts, {})
                self.assertEqual([item['reason'] for item in result.rejected[1:]], ['base_unconfirmed'] * 14)
                self.assertEqual(base.call_count, 1); poi.assert_not_called()

    def test_zero_http_budget_blocks_all_new_wire_lookups(self):
        result, _, base, poi = self.run_payload(wire(), http_budget=0)
        self.assertEqual(result.base_facts, [])
        self.assertEqual(result.confirmed_hints, [])
        self.assertEqual(result.reserved_http_attempts_upper_bound, 0)
        base.assert_not_called(); poi.assert_not_called()

    def test_legacy_context_diagnostics_distinguish_shape_without_raw_values(self):
        original = grounding._materialize_candidate_batch(wire(1), scope()).pois[0]
        cases = {'outer_whitespace': ' Музей культуры 0, Петрозаводск, Россия',
                 'not_printable': 'Музей культуры 0, Петро\nзаводск, Россия',
                 'empty_component': 'Музей культуры 0,,Россия',
                 'country_mismatch': 'Музей культуры 0, Петрозаводск, Finland',
                 'missing_locality': 'Музей культуры 0, Россия',
                 'title_query_mismatch': 'Другое имя, Петрозаводск, Россия'}
        for expected, query in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(grounding._proposal_context_issue(original.model_copy(update={'place_query': query}), 'Россия'), expected)

    def test_omitted_country_comes_only_from_consistent_verified_region_record(self):
        for country, code in (('Россия', 'ru'), ('France', 'fr')):
            raw = record('Карелия', kind='administrative', category='boundary')
            raw.update(name=f'Территория, {country}', boundingbox=list(scope().bounds), geojson=scope().geometry,
                       address={'country': country, 'country_code': code})
            before = deepcopy(raw)
            with patch.object(places, 'geocode', return_value=raw) as lookup:
                result = resolve_region('Территория')
            self.assertEqual(result.country, country)
            lookup.assert_called_once_with('Территория', include_geometry=True)
            self.assertEqual(raw, before)
            self.assertTrue(grounding._materialize_candidate_batch(wire(1), result).pois[0].place_query.endswith(country))

    def test_missing_conflicting_or_unverified_map_country_never_becomes_wire_country(self):
        raw = record('Карелия', kind='administrative', category='boundary')
        raw.update(name='Карелия, Россия', boundingbox=list(scope().bounds), geojson=scope().geometry)
        cases = ({}, {'country': '', 'country_code': 'ru'}, {'country': 'Россия', 'country_code': ''},
                 {'country': 'France', 'country_code': 'ru'}, {'country': 'Россия', 'country_code': 'fr'},
                 {'country': 'France', 'country_code': 'fr'}, {'country': 'private\ntext', 'country_code': 'ru'})
        for address in cases:
            with self.subTest(address=address), patch.object(places, 'geocode', return_value={**raw, 'address': address}):
                with self.assertRaises(DestinationScopeError):
                    resolve_region('Карелия')
        # Explicit user context keeps the existing strict-geocoder contract;
        # it is never replaced by an inferred provider label here.
        with patch.object(places, 'geocode', return_value=raw):
            self.assertEqual(resolve_region('Карелия, Russia').country, 'Russia')


if __name__ == '__main__':
    unittest.main()
