"""Precise retry hints must never guess a place or weaken the wire schema."""
from copy import deepcopy
import json
import unittest

from grounded_response import build_grounded_response_schema, grounded_query_feedback
from grounded_schedule import constrain_grounded_schema
import planner
from test_grounded_schedule import request_for, grounding_for, itinerary_for
from test_grounded_planner import _wire_response_json


def fixture():
    grounding=grounding_for()
    schema=build_grounded_response_schema(constrain_grounded_schema(
        planner.build_itinerary_json_schema(request_for()),grounding),request_for())
    payload=json.loads(_wire_response_json(itinerary_for()))
    payload['day_1']['attractions'][0]['place_query']='REJECTED_PRIVATE_MODEL_STRING'
    return grounding,schema,payload


class GroundedEnumFeedbackTests(unittest.TestCase):
    def test_known_exact_title_provides_only_verified_canonical_query(self):
        grounding,schema,payload=fixture();before=deepcopy((grounding,schema,payload))
        feedback=grounded_query_feedback(json.dumps(payload),schema,grounding)
        self.assertEqual(feedback,[{'response_path':'day_1.attractions[0].place_query',
            'confirmed_title':'Байкальский музей','allowed_place_query':'Байкальский музей, Листвянка, Россия'}])
        self.assertNotIn('REJECTED_PRIVATE',json.dumps(feedback))
        self.assertEqual((grounding,schema,payload),before)

    def test_partial_day_four_repair_preserves_exact_index_and_scope(self):
        grounding,schema,payload=fixture()
        schema={'properties':{'day_4':schema['properties']['day_1']}}
        day=payload['day_1'];day['attractions']=[deepcopy(day['attractions'][0]) for _ in range(3)]
        for item in day['attractions'][:2]:item['place_query']='Байкальский музей, Листвянка, Россия'
        feedback=grounded_query_feedback(json.dumps({'day_4':day}),schema,grounding)
        self.assertEqual(len(feedback),1)
        self.assertEqual(feedback[0]['response_path'],'day_4.attractions[2].place_query')

    def test_no_name_fuzzing_or_translation_or_invented_query(self):
        grounding,schema,payload=fixture()
        for title in ('байкальский музей','Baikal Museum','Байкальский музей!','Новый музей',None):
            payload['day_1']['attractions'][0]['title']=title
            self.assertEqual(grounded_query_feedback(json.dumps(payload),schema,grounding),[])

    def test_ambiguous_title_in_two_localities_cannot_choose_for_model(self):
        grounding,schema,payload=fixture()
        grounding.base_facts.append({**grounding.base_facts[0],'base_key':'base_2','base_query':'Большие Коты, Россия'})
        grounding.confirmed_hints.append({**grounding.confirmed_hints[0],'base_key':'base_2',
            'place_query':'Байкальский музей, Большие Коты, Россия','locality':'Большие Коты'})
        self.assertEqual(grounded_query_feedback(json.dumps(payload),schema,grounding),[])

    def test_candidate_excluded_from_day_enum_cannot_be_reintroduced(self):
        grounding,schema,payload=fixture()
        schema['properties']['day_1']['properties']['attractions']['items']['properties']['place_query']['enum']=['Камень Черского, Листвянка, Россия']
        self.assertEqual(grounded_query_feedback(json.dumps(payload),schema,grounding),[])

    def test_valid_query_does_not_produce_replacement(self):
        grounding,schema,payload=fixture()
        payload['day_1']['attractions'][0]['place_query']='Байкальский музей, Листвянка, Россия'
        self.assertEqual(grounded_query_feedback(json.dumps(payload),schema,grounding),[])

    def test_invalid_or_oversized_json_and_duplicate_keys_are_not_repaired(self):
        grounding,schema,_=fixture()
        for raw in ('no json','[]','{"day_1":{},"day_1":{}}','{"x":NaN}',' '*262145):
            self.assertEqual(grounded_query_feedback(raw,schema,grounding),[])

    def test_retry_hint_count_is_bounded_and_does_not_echo_model_description(self):
        grounding,schema,payload=fixture()
        item=payload['day_1']['attractions'][0]
        item['description']='DO_NOT_ECHO_MODEL_PROSE'
        payload['day_1']['attractions']=[deepcopy(item) for _ in range(8)]
        feedback=grounded_query_feedback(json.dumps(payload),schema,grounding)
        self.assertEqual(len(feedback),4)
        self.assertNotIn('DO_NOT_ECHO',json.dumps(feedback))


if __name__=='__main__':unittest.main()
