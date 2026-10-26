"""Short model choices preserve full server identity and schedule validation."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import unittest

import grounded_response as wire
from grounded_schedule import build_grounded_messages
import planner
from schemas import Itinerary
from test_grounded_id_contract_regressions import _contract, _id_payload
from test_regional_map_only_schedule import fixture


class GroundedIdResponseTests(unittest.TestCase):
    def test_catalog_is_immutable_and_codes_are_short_ordered_choices(self):
        grounding = fixture()[2]
        catalog = wire.freeze_grounded_catalog(grounding)
        self.assertEqual([row[0] for row in catalog.bases], ['b01', 'b02'])
        self.assertEqual([row[0] for row in catalog.candidates], [f'p{i:02}' for i in range(1,15)])
        self.assertIsInstance(catalog.candidates[0][4], tuple)
        with self.assertRaises(FrozenInstanceError):
            catalog.fingerprint = 'replaced'
        forged = replace(catalog, candidates=tuple(reversed(catalog.candidates)))
        with self.assertRaises(ValueError):
            forged.assert_matches(grounding)

    def test_messages_publish_codes_and_allowed_base_ids_without_fingerprint(self):
        request,scope,grounding,_,_ = fixture()
        catalog = wire.freeze_grounded_catalog(grounding)
        before = deepcopy(grounding)
        messages = build_grounded_messages(request,scope,grounding,5,identity_catalog=catalog)
        data = json.loads(messages[1]['content'])
        self.assertEqual([row['candidate_id'] for row in data['candidates']], [row[0] for row in catalog.candidates])
        self.assertEqual(data['candidates'][0]['allowed_base_ids'], ['b01'])
        self.assertNotIn(catalog.fingerprint, json.dumps(messages))
        self.assertIn('НЕ передавай title, area и place_query', messages[0]['content'])
        self.assertEqual(grounding, before)

    def test_provider_identity_mutation_invalidates_frozen_catalog(self):
        for collection in ('base_facts', 'confirmed_hints'):
            with self.subTest(collection=collection):
                grounding = fixture()[2]
                catalog = wire.freeze_grounded_catalog(grounding)
                getattr(grounding, collection)[0]['osm_id'] = 999999
                with self.assertRaises(ValueError):
                    catalog.assert_matches(grounding)

    def test_full_id_schema_removes_redundant_long_identity_fields(self):
        original,schema,catalog = _contract(fixture())
        for number in range(1,8):
            fields = schema['properties'][f'day_{number}']['properties']
            self.assertEqual(fields['base_id']['enum'], ['b01','b02'])
            self.assertNotIn('base_area', fields)
            visits = fields['attractions']['items']['properties']
            self.assertTrue({'title','place_query','area'}.isdisjoint(visits))
            self.assertEqual(visits['candidate_id']['enum'], [row[0] for row in catalog.candidates])
            self.assertIn('title', fields['meal']['properties'])
            self.assertIn('place_query', fields['rest']['properties'])
        self.assertIn('place_query', original['properties']['day_1']['properties']['attractions']['items']['properties'])

    def test_original_query_const_is_projected_to_short_code(self):
        data = fixture(); original,_,catalog = _contract(data)
        query = catalog.candidates[0][2]
        original['properties']['day_1']['properties']['attractions']['items']['properties']['place_query']['const'] = query
        schema = wire.build_grounded_id_schema(original,data[2],catalog=catalog)
        field = schema['properties']['day_1']['properties']['attractions']['items']['properties']['candidate_id']
        self.assertEqual(field['const'],'p01')

    def test_revalidation_checks_canonical_constraints_after_hydration(self):
        data = fixture(); original,_,catalog = _contract(data)
        original['properties']['day_1']['properties']['attractions']['items']['properties']['title']['const'] = 'Другой музей'
        schema = wire.build_grounded_id_schema(original,data[2],catalog=catalog)
        raw = json.dumps(_id_payload(data[4],original,catalog),ensure_ascii=False)
        with self.assertRaisesRegex(ValueError,'title.*const'):
            wire.decode_grounded_id_response(raw,schema,original,data[2],catalog=catalog)

    def test_response_keeps_model_time_prose_cost_and_does_not_claim_verification(self):
        data=fixture(); original,schema,catalog=_contract(data)
        payload=_id_payload(data[4],original,catalog)
        first=payload['day_1']['attractions'][0]
        first.update(time='10:15',duration_min=45,cost=777,description='45 минут — плановая оценка; тариф не подтверждён.')
        canonical=wire.decode_grounded_id_response(json.dumps(payload,ensure_ascii=False),schema,original,data[2],catalog=catalog)
        route=Itinerary.model_validate_json(canonical)
        item=next(item for item in route.days[0].items if item.kind=='attraction')
        self.assertEqual((item.time,item.duration_min,item.cost,item.description),
                         ('10:15',45,777,first['description']))
        self.assertFalse(item.verified)
        self.assertNotIn('candidate_id',canonical)
        self.assertNotIn('base_id',canonical)

    def test_caller_cannot_relax_id_schema_before_decode(self):
        data=fixture(); original,schema,catalog=_contract(data)
        payload=_id_payload(data[4],original,catalog)
        schema['properties']['day_1']['properties']['attractions']['items']['properties']['candidate_id']['enum'].append('p99')
        with self.assertRaisesRegex(ValueError,'Схема идентификаторов'):
            wire.decode_grounded_id_response(json.dumps(payload),schema,original,data[2],catalog=catalog)

    def test_partial_repair_instructions_keep_ids_and_verified_day_locks(self):
        request,scope,grounding,_,route=fixture()
        messages=build_grounded_messages(request,scope,grounding,5)
        history={'first':{'day':1,'title':route.days[0].items[0].title,'place_query':route.days[0].items[0].place_query}}
        repair=planner._regional_repair_messages(messages,route,[2],[3],{2:['Недостаточно программы']},{},history)
        data=json.loads(repair[2]['content'].split('\n',1)[1].rsplit('\n',1)[0])
        self.assertIn('коды не меняются между попытками',repair[0]['content'])
        self.assertEqual(data['verified_places'][0]['candidate_id'],'p01')
        self.assertIn('p01',data['fixed_candidate_ids'])


if __name__ == '__main__':
    unittest.main()
