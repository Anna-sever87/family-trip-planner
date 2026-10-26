"""Independent regional ID-wire checks: identity binding and ordinary guards.

All places below are synthetic. No live map, AI, filesystem cache or database
is needed. A server-owned ID is a compact choice, not evidence of opening
hours, admission prices, road access or sufficient itinerary depth.
"""
from copy import deepcopy
import json
import socket
import unittest
from unittest.mock import patch

import grounded_response as wire
from grounded_schedule import constrain_grounded_schema, enforce_grounded_choices
import planner
from regional_schedule_witness import check_witness, constrain_witness_schema
from rules import check_program_depth
from schemas import Itinerary
import test_regional_map_only_schedule as regional
import test_regional_schedule_witness as witness_tests


def _project(value, schema):
    """Fixture serialization: emit only the actual model-facing fields."""
    if schema['type'] == 'object':
        return {key: _project(value[key], child)
                for key, child in schema['properties'].items()}
    if schema['type'] == 'array':
        return [_project(item, schema['items']) for item in value]
    return value


def _original_payload(route, original_schema):
    source = route.model_dump(mode='json')
    source_days = source.pop('days')
    for day in source_days:
        items = day.pop('items')
        day['attractions'] = [item for item in items if item['kind'] == 'attraction']
        for kind in ('meal', 'rest'):
            day[kind] = next(item for item in items if item['kind'] == kind)
        source[f"day_{day['day']}"] = day
    return _project(source, original_schema)


def _id_payload(route, original_schema, catalog):
    payload = _original_payload(route, original_schema)
    by_query = {row[2]: row[0] for row in catalog.candidates}
    by_base = {row[2]: row[0] for row in catalog.bases}
    for key, day in payload.items():
        if not key.startswith('day_'):
            continue
        day['base_id'] = by_base[day.pop('base_area')]
        for item in day['attractions']:
            item['candidate_id'] = by_query[item.pop('place_query')]
            item.pop('title')
            item.pop('area')
    return payload


def _contract(data, *, selected=None, witness=None):
    request, _, grounding, _, _ = data
    canonical = (planner.build_itinerary_json_schema(request) if selected is None
                 else planner._regional_day_repair_schema(request, selected, []))
    original = wire.build_grounded_response_schema(
        constrain_grounded_schema(canonical, grounding), request)
    original = constrain_witness_schema(original, witness)
    catalog = wire.freeze_grounded_catalog(grounding)
    schema = wire.build_grounded_id_schema(original, grounding, catalog=catalog)
    return original, schema, catalog


class GroundedIdContractRegressionTests(unittest.TestCase):
    def setUp(self):
        no_network = patch.object(socket.socket, 'connect',
                                  side_effect=AssertionError('No external requests'))
        no_network.start()
        self.addCleanup(no_network.stop)

    def decode(self, payload, data, contract):
        original, schema, catalog = contract
        return wire.decode_grounded_id_response(
            json.dumps(payload, ensure_ascii=False), schema, original, data[2], catalog=catalog)

    def test_full_week_roundtrip_hydrates_only_confirmed_identity(self):
        data = regional.fixture()
        before = deepcopy(data)
        contract = _contract(data)
        original, schema, catalog = contract
        payload = _id_payload(data[4], original, catalog)
        decoded = Itinerary.model_validate_json(self.decode(payload, data, contract))
        self.assertEqual(enforce_grounded_choices(decoded, data[2]), [])
        self.assertEqual(check_program_depth(decoded, data[0]), [])
        self.assertEqual(len(decoded.days), 7)
        for old, new in zip(data[4].days, decoded.days):
            self.assertEqual(new.base_area, old.base_area)
            for old_item, new_item in zip(old.items, new.items):
                self.assertEqual(
                    (new_item.title, new_item.area, new_item.place_query,
                     new_item.time, new_item.duration_min, new_item.cost, new_item.description),
                    (old_item.title, old_item.area, old_item.place_query,
                     old_item.time, old_item.duration_min, old_item.cost, old_item.description))
                self.assertFalse(new_item.verified)
        self.assertEqual(data, before)
        self.assertNotIn('place_query', schema['properties']['day_1']['properties']['attractions']['items']['properties'])
        self.assertNotIn('base_area', schema['properties']['day_1']['properties'])

    def test_known_id_cannot_silently_override_conflicting_identity_fields(self):
        data = regional.fixture(); contract = _contract(data)
        payload = _id_payload(data[4], contract[0], contract[2])
        for field, value in (('title', 'Музей другого города'),
                             ('place_query', 'Другой музей, Чужой город, Другая страна'),
                             ('area', 'Чужой город')):
            with self.subTest(field=field):
                bad = deepcopy(payload)
                bad['day_1']['attractions'][0][field] = value
                with self.assertRaises(ValueError): self.decode(bad, data, contract)
        bad = deepcopy(payload)
        bad['day_1']['base_area'] = 'Чужой город, Другая страна'
        with self.assertRaises(ValueError): self.decode(bad, data, contract)

    def test_unknown_wrong_type_and_cross_kind_ids_are_rejected(self):
        data = regional.fixture(); contract = _contract(data)
        payload = _id_payload(data[4], contract[0], contract[2])
        for value in ('p99', 'P01', '', None, 1, True, contract[2].bases[0][0]):
            with self.subTest(candidate_id=value):
                bad = deepcopy(payload); bad['day_1']['attractions'][0]['candidate_id'] = value
                with self.assertRaises(ValueError): self.decode(bad, data, contract)
        for value in ('b99', '', None, 1, contract[2].candidates[0][0]):
            with self.subTest(base_id=value):
                bad = deepcopy(payload); bad['day_1']['base_id'] = value
                with self.assertRaises(ValueError): self.decode(bad, data, contract)

    def test_catalog_rebinding_or_fact_mutation_fails_closed(self):
        data = regional.fixture(); contract = _contract(data)
        payload = _id_payload(data[4], contract[0], contract[2])
        for mode in ('reorder_candidates', 'reorder_bases', 'coordinates', 'identity', 'membership',
                     'source_id', 'provider', 'source_address', 'source_name'):
            with self.subTest(mode=mode):
                changed = deepcopy(data)
                if mode == 'reorder_candidates': changed[2].confirmed_hints.reverse()
                elif mode == 'reorder_bases': changed[2].base_facts.reverse()
                elif mode == 'coordinates': changed[2].confirmed_hints[0]['lat'] += .01
                elif mode == 'identity': changed[2].confirmed_hints[0]['locality'] = 'Иной город'
                elif mode == 'membership': changed[2].confirmed_hints[0]['allowed_base_keys'] = ['base_1', 'base_2']
                elif mode == 'source_id': changed[2].confirmed_hints[0]['osm_id'] += 1
                elif mode == 'provider': changed[2].confirmed_hints[0]['provider'] = 'another_provider'
                elif mode == 'source_address': changed[2].confirmed_hints[0]['map_address']['country_code'] = 'cn'
                else: changed[2].confirmed_hints[0]['official_name'] = 'Другое официальное имя'
                with self.assertRaises(ValueError): self.decode(payload, changed, contract)
                with self.assertRaises(ValueError):
                    wire.build_grounded_id_schema(contract[0], changed[2], catalog=contract[2])

    def test_distinct_base_keys_with_same_canonical_query_are_ambiguous(self):
        grounding = regional.fixture()[2]
        grounding.base_facts[1]['base_query'] = grounding.base_facts[0]['base_query']
        before = deepcopy(grounding)
        with self.assertRaises(ValueError):
            wire.freeze_grounded_catalog(grounding)
        self.assertEqual(grounding, before)

    def test_partial_repair_keeps_global_ids_dates_and_unselected_days(self):
        data = regional.fixture(); source = data[4]; before = source.model_dump()
        full = _contract(data)
        partial = _contract(data, selected=[4, 5])
        self.assertEqual(full[2], partial[2])
        replacement = source.model_copy(deep=True)
        replacement.days[3].theme = 'Уточнённая программа четвёртого дня'
        replacement.days[4].transfer_note = 'Уточнённое примечание переезда'
        payload = _id_payload(replacement, partial[0], partial[2])
        canonical_patch = self.decode(payload, data, partial)
        merged = planner._merge_regional_day_repair(source, canonical_patch, [4], [5])
        self.assertEqual(merged.days[3].theme, replacement.days[3].theme)
        self.assertEqual(merged.days[4].transfer_note, replacement.days[4].transfer_note)
        self.assertEqual(merged.days[4].items, source.days[4].items)
        for index in (0, 1, 2, 5, 6): self.assertEqual(merged.days[index], source.days[index])
        self.assertEqual(source.model_dump(), before)
        for mode in ('missing', 'extra', 'date'):
            bad = deepcopy(payload)
            if mode == 'missing': bad.pop('day_5')
            elif mode == 'extra': bad['day_1'] = deepcopy(bad['day_4'])
            else: bad['day_4']['date'] = '2099-07-01'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                self.decode(bad, data, partial)

    def test_binding_witness_rejects_another_days_valid_id_and_base(self):
        data = witness_tests.fixture()
        witness, route = witness_tests.witness_route(data)
        self.assertTrue(witness['binding'])
        contract = _contract(data, witness=witness)
        payload = _id_payload(route, contract[0], contract[2])
        decoded = Itinerary.model_validate_json(self.decode(payload, data, contract))
        self.assertEqual(check_witness(decoded, witness), [])
        bad = deepcopy(payload)
        bad['day_1']['attractions'][0]['candidate_id'] = payload['day_2']['attractions'][0]['candidate_id']
        with self.assertRaises(ValueError): self.decode(bad, data, contract)
        bad = deepcopy(payload)
        bad['day_1']['base_id'] = next(row[0] for row in contract[2].bases if row[0] != payload['day_1']['base_id'])
        with self.assertRaises(ValueError): self.decode(bad, data, contract)

    def test_advisory_witness_does_not_force_its_id_subset(self):
        data = witness_tests.fixture()
        witness, route = witness_tests.witness_route(data)
        witness['binding'] = False
        contract = _contract(data, witness=witness)
        payload = _id_payload(route, contract[0], contract[2])
        first, second = payload['day_1']['attractions'][0], payload['day_2']['attractions'][0]
        first['candidate_id'], second['candidate_id'] = second['candidate_id'], first['candidate_id']
        decoded = Itinerary.model_validate_json(self.decode(payload, data, contract))
        self.assertEqual(enforce_grounded_choices(decoded, data[2]), [])
        self.assertEqual(check_witness(decoded, witness), [])
        self.assertNotEqual(decoded.days[0].items[0].place_query, route.days[0].items[0].place_query)

    def test_partial_binding_uses_full_catalog_and_rejects_other_day_choice(self):
        data = witness_tests.fixture()
        witness, route = witness_tests.witness_route(data)
        full = _contract(data, witness=witness)
        partial = _contract(data, selected=[4, 5], witness=witness)
        self.assertEqual(full[2], partial[2])
        payload = _id_payload(route, partial[0], partial[2])
        merged = planner._merge_regional_day_repair(
            route, self.decode(payload, data, partial), [4], [5])
        self.assertEqual(check_witness(merged, witness), [])
        first_day = _id_payload(route, full[0], full[2])['day_1']
        bad = deepcopy(payload)
        bad['day_4']['attractions'][0]['candidate_id'] = first_day['attractions'][0]['candidate_id']
        with self.assertRaises(ValueError): self.decode(bad, data, partial)

    def test_known_ids_do_not_authorize_wrong_base_or_thin_program(self):
        data = regional.fixture(); contract = _contract(data)
        payload = _id_payload(data[4], contract[0], contract[2])
        bad = deepcopy(payload)
        bad['day_1']['base_id'] = contract[2].bases[1][0]
        # Strict decoding may reject the incompatible pair early. Otherwise
        # the existing canonical compatibility guard must reject it unchanged.
        try:
            decoded = Itinerary.model_validate_json(self.decode(bad, data, contract))
        except ValueError:
            pass
        else:
            self.assertTrue(any('несовместимо с базой' in p for p in enforce_grounded_choices(decoded, data[2])))
        thin = deepcopy(payload)
        for key, day in thin.items():
            if key.startswith('day_'):
                for item in day['attractions']: item['duration_min'] = 30
        decoded = Itinerary.model_validate_json(self.decode(thin, data, contract))
        self.assertTrue(check_program_depth(decoded, data[0]))
        self.assertTrue(all(item.duration_min == 30 for day in decoded.days
                            for item in day.items if item.kind == 'attraction'))

    def test_duplicate_json_keys_and_ids_on_meal_are_not_ignored(self):
        data = regional.fixture(); contract = _contract(data)
        payload = _id_payload(data[4], contract[0], contract[2])
        raw = json.dumps(payload, ensure_ascii=False)
        first_id = payload['day_1']['attractions'][0]['candidate_id']
        marker = json.dumps('candidate_id') + ': ' + json.dumps(first_id)
        duplicate = raw.replace(marker, marker + ', ' + marker, 1)
        with self.assertRaises(ValueError):
            wire.decode_grounded_id_response(duplicate, contract[1], contract[0], data[2], catalog=contract[2])
        bad = deepcopy(payload); bad['day_1']['meal']['candidate_id'] = first_id
        with self.assertRaises(ValueError): self.decode(bad, data, contract)

    def test_real_pipeline_still_checks_map_coordinates_after_id_hydration(self):
        data = regional.fixture(); contract = _contract(data)
        raw = json.dumps(_id_payload(data[4], contract[0], contract[2]), ensure_ascii=False)
        for outside in (False, True):
            with self.subTest(outside_region=outside):
                records = deepcopy(data[3])
                if outside:
                    query = next(item.place_query for item in data[4].days[0].items
                                 if item.kind == 'attraction')
                    records[query]['lat'] = 60.
                with patch.object(regional, '_id_wire_response_json', return_value=raw):
                    result, completions, map_calls, _, _ = (
                        regional.RegionalMapOnlyScheduleTests().run_route(data[4], records_override=records))
                self.assertEqual(completions, 1)
                self.assertGreater(map_calls, 0)
                if outside:
                    self.assertTrue(any('пределами выбранной территории' in p for p in result.problems))
                else:
                    self.assertEqual(result.problems, [])
                    self.assertTrue(all(item.verified for day in result.itinerary.days
                                        for item in day.items if item.kind == 'attraction'))


if __name__ == '__main__':
    unittest.main()
