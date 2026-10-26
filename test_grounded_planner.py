"""Real candidate-to-schedule flow with mocked completions and map records only."""

import json
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import places
import planner
import candidate_grounding
import test_regional_generation_repair as regional
from yandex_ai import GenerationResult, YandexAIError


BASE_QUERY = "Листвянка, Россия"
SECOND_BASE_QUERY = "Большие Коты, Россия"


def _source():
    source = regional._valid_source()
    for day in source.days:
        day.base_area = BASE_QUERY
        for item in day.items:
            if item.kind == "attraction":
                item.area = "Листвянка"
    return source


def _batch(source):
    return {
        "bases": [
            {"base_key": "base_1", "base_query": BASE_QUERY},
            {"base_key": "base_2", "base_query": SECOND_BASE_QUERY},
        ],
        "pois": [{
            "title": regional._attraction(day).title,
            "place_query": regional._attraction(day).place_query,
            "locality": "Листвянка", "base_key": "base_1",
        } for day in source.days] + [{
            "title": f"Тестовый резервный музей {number}",
            "place_query": f"Тестовый резервный музей {number}, Листвянка, Россия",
            "locality": "Листвянка", "base_key": "base_1",
        } for number in range(len(source.days))],
    }


def _wire_response_json(source, selected=None):
    """Model fixtures emit the actual wire shape; never adapt them in a mock."""
    canonical = json.loads(regional._model_response_json(source))
    # Server-owned scheduling provenance is never part of model output.
    canonical.pop("source_schedule", None)
    canonical.pop("city_schedule", None)
    canonical.pop("program_report", None)
    days = canonical.pop("days")
    payload = canonical if selected is None else {}
    for day in days:
        day.pop("day_mode", None)
        if selected is not None and day["day"] not in selected:
            continue
        items = day.pop("items")
        for item in items:
            # Map-provider attribution is server-owned, just like coordinates.
            item.pop("provider", None)
            item.pop("source_url", None)
        day["attractions"] = [item for item in items if item["kind"] == "attraction"]
        day["meal"] = next(item for item in items if item["kind"] == "meal")
        day["rest"] = next(item for item in items if item["kind"] == "rest")
        payload[f'day_{day["day"]}'] = day
    return json.dumps(payload, ensure_ascii=False)


def _id_wire_response_json(source, selected=None, *, catalog=None):
    """Author an ID-wire fixture before the completion stub is installed.

    Unknown selections deliberately become unknown codes. Conflicting legacy
    identity fields stay as forbidden extras so negative tests cannot be
    accidentally repaired by fixture serialization.
    """
    payload = json.loads(_wire_response_json(source, selected))
    if catalog is None:
        fixture = _batch(_source())
        bases = {row['base_query']: f'b{index:02}' for index,row in enumerate(fixture['bases'],1)}
        choices = {row['place_query']: (f'p{index:02}',row['title'],row['locality'])
                   for index,row in enumerate(fixture['pois'],1)}
    else:
        bases = {row[2]:row[0] for row in catalog.bases}
        choices = {row[2]:(row[0],row[1],row[3]) for row in catalog.candidates}
    for key,day in payload.items():
        if not key.startswith('day_'):
            continue
        day['base_id'] = bases.get(day.pop('base_area'),'b99')
        for item in day['attractions']:
            choice = choices.get(item.pop('place_query'))
            item['candidate_id'] = choice[0] if choice else 'p99'
            if choice and item['title'] == choice[1]:
                item.pop('title')
            elif not choice:
                item.pop('title')
            if not choice or item['area'] in ('',choice[2]):
                item.pop('area')
    return json.dumps(payload,ensure_ascii=False)


class GroundedPlannerTests(unittest.TestCase):
    def test_wire_fixture_never_emits_server_schedule_markers(self):
        source = _source().model_copy(update={"source_schedule": True, "city_schedule": True})
        payload = json.loads(_id_wire_response_json(source))
        self.assertNotIn("source_schedule", payload)
        self.assertNotIn("city_schedule", payload)
        self.assertNotIn("program_report", payload)
        self.assertTrue(all("day_mode" not in day for key, day in payload.items() if key.startswith("day_")))

    def _context(self, stack, responses, *, payload=None, unconfirmed=()):
        # Synthetic model-only fixtures must not depend on the live catalogue.
        stack.enter_context(patch.object(candidate_grounding, "destination_name_hints", return_value=[]))
        # Source-card scheduling has separate integration coverage; this helper
        # exercises the model wire/repair contract without real routing/cache IO.
        stack.enter_context(patch.object(planner, "prepare_source_plan", return_value=None))
        source = _source()
        payload = _batch(source) if payload is None else payload
        records = {
            query: {"name": query, "lat": 51.85, "lon": 104.90 + index * 0.01,
                    "type": "village", "category": "place",
                    "address": {"village": query.split(",")[0], "county": "Иркутский муниципальный округ", "country_code": "ru"}}
            for index, query in enumerate((BASE_QUERY, SECOND_BASE_QUERY))
        }
        # Synthetic positions exercise identity/pool flow, not real POI facts.
        for index, proposal in enumerate(payload["pois"]):
            query = proposal["place_query"]
            records[query] = {"name": query, "lat": 51.86 + index * 0.002,
                              "lon": 104.95, "type": "museum", "category": "tourism",
                              "address": {"village": "Листвянка", "county": "Иркутский муниципальный округ", "country_code": "ru"}}

        def geocode(query, **kwargs):
            self.assertEqual(kwargs, {"max_variants": 1})
            self.assertIn(query, (BASE_QUERY, SECOND_BASE_QUERY))
            return records[query]

        def nearby(title, country, base_lat, base_lon, maximum, **_checks):
            self.assertEqual((country, base_lat, base_lon, maximum), ("Россия", 51.85, 104.90, 80))
            self.assertEqual(set(_checks), {"place_filter"})
            if title in unconfirmed:
                return None
            query = f"{title}, Листвянка, {country}"
            result = {**records[query], "place_query": query, "locality": "Листвянка"}
            self.assertTrue(_checks["place_filter"](result))
            self.assertFalse(_checks["place_filter"]({**result, "lat": 20, "lon": 20}))
            return result

        for name, value in (
            ("resolve_region", regional._scope()), ("get_weather", None),
            ("find_city", None), ("cached_place_hints", []),
            ("curated_place_hints", []), ("curated_blocked_place_terms", []),
        ):
            stack.enter_context(patch.object(planner, name, return_value=value))
        stack.enter_context(patch.object(places, "_cache", {}))
        stack.enter_context(patch.object(places, "_save_cache", side_effect=AssertionError("No cache writes")))
        stack.enter_context(patch.object(places, "geocode", side_effect=geocode))
        lookup = stack.enter_context(patch.object(places, "geocode_nearby_named", side_effect=nearby))
        verify = stack.enter_context(patch.object(planner, "_map_verification_problems", side_effect=lambda *args, **kwargs: []))
        generate = stack.enter_context(patch.object(planner, "generate_json", side_effect=[
            value if isinstance(value, Exception) else GenerationResult(value, 11, 13)
            for value in responses
        ]))
        return generate, lookup, verify

    def test_candidate_and_schedule_share_injected_client_and_usage(self):
        source = _source()
        with ExitStack() as stack:
            generate, lookup, _ = self._context(stack, [
                json.dumps(_batch(source), ensure_ascii=False), _id_wire_response_json(source),
            ])
            result = planner.generate_itinerary(regional._request(), max_attempts=3, ground_candidates=True)
        self.assertEqual((generate.call_count, result.attempts, result.problems), (2, 2, []))
        self.assertEqual((result.prompt_tokens, result.completion_tokens, result.total_tokens), (22, 26, 48))
        self.assertEqual(lookup.call_count, 14)
        self.assertIn("bases", generate.call_args_list[0].kwargs["json_schema"]["properties"])
        schema = generate.call_args_list[1].kwargs["json_schema"]
        self.assertEqual(set(schema["properties"]), {"title", "summary", "tips", *(f"day_{day}" for day in range(1, 8))})
        self.assertNotIn("days", schema["properties"])

    def test_schedule_receives_only_confirmed_pool_and_base_facts(self):
        source = _source()
        payload = _batch(source)
        with ExitStack() as stack:
            generate, _, _ = self._context(stack, [json.dumps(payload), _id_wire_response_json(source)])
            planner.generate_itinerary(regional._request(), max_attempts=3, ground_candidates=True, memory_context="Test preference")
        messages = generate.call_args_list[1].args[0]
        data = json.loads(messages[1]["content"])
        allowed = {poi["place_query"] for poi in payload["pois"]}
        self.assertEqual({poi["place_query"] for poi in data["candidates"]}, allowed)
        self.assertEqual({base["base_query"] for base in data["bases"]}, {BASE_QUERY, SECOND_BASE_QUERY})
        self.assertTrue(all({"lat", "lon", "base_key"}.issubset(base) for base in data["bases"]))
        self.assertEqual(data["memory_context"], "Test preference")
        schema = generate.call_args_list[1].kwargs["json_schema"]
        fields = schema["properties"]["day_1"]["properties"]
        self.assertEqual(set(fields["base_id"]["enum"]), {"b01", "b02"})
        self.assertEqual(set(fields["attractions"]["items"]["properties"]["candidate_id"]["enum"]), {f"p{i:02}" for i in range(1,15)})
        self.assertNotIn("items", fields)
        self.assertEqual(fields["attractions"]["minItems"], 1)
        self.assertTrue(messages[0]["content"].endswith(planner.GROUNDED_RESPONSE_FORMAT_RULE))

    def test_short_schedule_has_only_one_remaining_repair_and_keeps_pool_schema(self):
        source = _source()
        short = source.model_copy(deep=True)
        regional._attraction(short.days[1]).duration_min = 90
        with ExitStack() as stack:
            generate, _, _ = self._context(stack, [
                json.dumps(_batch(source)), _id_wire_response_json(short), _id_wire_response_json(source, [2, 3]),
            ])
            result = planner.generate_itinerary(regional._request(), max_attempts=3, ground_candidates=True)
        self.assertEqual((generate.call_count, result.attempts, result.problems), (3, 3, []))
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (33, 39))
        schema = generate.call_args_list[2].kwargs["json_schema"]
        self.assertEqual(set(schema["properties"]), {"day_2", "day_3"})
        allowed = {poi["place_query"] for poi in _batch(source)["pois"]}
        for number in (2, 3):
            fields = schema["properties"][f"day_{number}"]["properties"]
            self.assertEqual(set(fields["base_id"]["enum"]), {"b01", "b02"})
            self.assertEqual(set(fields["attractions"]["items"]["properties"]["candidate_id"]["enum"]), {f"p{i:02}" for i in range(1,15)})
            self.assertNotIn("items", fields)
        self.assertEqual(result.itinerary.days[0], source.days[0])
        self.assertEqual(result.itinerary.days[2], source.days[2])
        self.assertTrue(generate.call_args_list[2].args[0][0]["content"].endswith(planner.GROUNDED_RESPONSE_FORMAT_RULE))

    def test_unfixed_schedule_stops_after_three_total_calls(self):
        source = _source()
        regional._attraction(source.days[1]).duration_min = 90
        with ExitStack() as stack:
            generate, _, _ = self._context(stack, [
                json.dumps(_batch(source)), _id_wire_response_json(source), _id_wire_response_json(source, [2, 3]),
            ])
            result = planner.generate_itinerary(regional._request(), max_attempts=3, ground_candidates=True)
        self.assertEqual((generate.call_count, result.attempts), (3, 3))
        self.assertTrue(any("недостаточно содержательная программа" in problem for problem in result.problems))
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (33, 39))

    def test_insufficient_confirmed_places_stops_after_candidate_call(self):
        payload = _batch(_source())
        unconfirmed = {poi["title"] for poi in payload["pois"][6:]}
        with ExitStack() as stack:
            generate, _, verify = self._context(stack, [json.dumps(payload)], payload=payload, unconfirmed=unconfirmed)
            result = planner.generate_itinerary(regional._request(), max_attempts=3, ground_candidates=True)
        self.assertEqual((generate.call_count, result.attempts), (1, 1))
        self.assertEqual(result.itinerary.days, [])
        self.assertTrue(any("подтверждено 6 мест" in problem for problem in result.problems))
        self.assertEqual(result.generation_status, "insufficient_data")
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (11, 13))
        verify.assert_not_called()

    def test_zero_confirmed_places_is_data_failure_not_generic_validation(self):
        payload = _batch(_source())
        with ExitStack() as stack:
            generate, _, verify = self._context(
                stack, [json.dumps(payload)], payload=payload,
                unconfirmed={poi["title"] for poi in payload["pois"]},
            )
            result = planner.generate_itinerary(regional._request(), max_attempts=3, ground_candidates=True)
        self.assertEqual(result.generation_status, "insufficient_data")
        self.assertEqual(result.itinerary.days, [])
        self.assertEqual((generate.call_count, result.attempts), (1, 1))
        verify.assert_not_called()

    def test_too_few_proposals_stop_before_any_map_lookup(self):
        payload = _batch(_source())
        payload["pois"].pop()
        with ExitStack() as stack:
            generate, lookup, verify = self._context(stack, [json.dumps(payload)], payload=payload)
            base_lookup = stack.enter_context(patch.object(places, "geocode", side_effect=AssertionError("No map for invalid count")))
            result = planner.generate_itinerary(regional._request(), max_attempts=3, ground_candidates=True)
        self.assertEqual((generate.call_count, result.attempts), (1, 1))
        self.assertEqual(result.itinerary.days, [])
        self.assertTrue(result.problems)
        lookup.assert_not_called()
        base_lookup.assert_not_called()
        verify.assert_not_called()

    def test_candidate_transport_error_does_not_start_schedule_or_map(self):
        with ExitStack() as stack:
            generate, lookup, verify = self._context(stack, [YandexAIError("offline test")])
            result = planner.generate_itinerary(regional._request(), max_attempts=3, ground_candidates=True)
        self.assertEqual((generate.call_count, result.attempts), (1, 1))
        self.assertTrue(result.problems)
        self.assertEqual(result.itinerary.days, [])
        lookup.assert_not_called()
        verify.assert_not_called()

    def test_malformed_candidate_json_does_not_spend_schedule_budget(self):
        with ExitStack() as stack:
            generate, lookup, verify = self._context(stack, ["{not valid JSON}"])
            result = planner.generate_itinerary(regional._request(), max_attempts=3, ground_candidates=True)
        self.assertEqual((generate.call_count, result.attempts), (1, 1))
        self.assertTrue(result.problems)
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (11, 13))
        lookup.assert_not_called()
        verify.assert_not_called()

    def test_schedule_cannot_escape_confirmed_place_or_base(self):
        original = _source()
        for escape in ("place", "base"):
            with self.subTest(escape=escape):
                source = original.model_copy(deep=True)
                if escape == "place":
                    item = regional._attraction(source.days[1])
                    item.title = "Неподготовленный музей"
                    item.place_query = "Неподготовленный музей, Листвянка, Россия"
                else:
                    source.days[1].base_area = "Неподготовленная база, Россия"
                with ExitStack() as stack:
                    generate, _, verify = self._context(stack, [
                        json.dumps(_batch(original)), _id_wire_response_json(source),
                    ])
                    # The wire schema rejects an escaped enum even before the
                    # unchanged semantic choices/map validators can run.
                    with self.assertRaisesRegex(RuntimeError, "нужном формате"):
                        planner.generate_itinerary(regional._request(), max_attempts=2, ground_candidates=True)
                self.assertEqual(generate.call_count, 2)
                verify.assert_not_called()

    def test_default_legacy_generation_does_not_add_candidate_call(self):
        with ExitStack() as stack:
            generate, lookup, _ = self._context(stack, [regional._model_response_json(_source())])
            grounding = stack.enter_context(patch.object(planner, "ground_region_candidates", side_effect=AssertionError("Legacy path")))
            result = planner.generate_itinerary(regional._request(), max_attempts=3)
        self.assertEqual((generate.call_count, result.attempts, result.problems), (1, 1, []))
        grounding.assert_not_called()
        lookup.assert_not_called()

    def test_wire_missing_meal_rest_or_empty_attractions_retries_within_total_budget(self):
        source = _source()
        for failure in ("meal", "rest", "attractions"):
            malformed = json.loads(_id_wire_response_json(source))
            if failure == "attractions":
                malformed["day_2"][failure] = []
            else:
                malformed["day_2"].pop(failure)
            with self.subTest(failure=failure), ExitStack() as stack:
                generate, _, verify = self._context(stack, [
                    json.dumps(_batch(source)), json.dumps(malformed), _id_wire_response_json(source),
                ])
                result = planner.generate_itinerary(regional._request(), max_attempts=3, ground_candidates=True)
                self.assertEqual((generate.call_count, result.attempts, result.problems), (3, 3, []))
                self.assertEqual((result.prompt_tokens, result.completion_tokens), (33, 39))
                verify.assert_called_once()
                for call in generate.call_args_list[1:]:
                    self.assertNotIn("days", call.kwargs["json_schema"]["properties"])
                    self.assertTrue(call.args[0][0]["content"].endswith(planner.GROUNDED_RESPONSE_FORMAT_RULE))

    def test_grounded_never_falls_back_to_legacy_days_items_response(self):
        source = _source()
        legacy = regional._model_response_json(source)
        with ExitStack() as stack:
            generate, _, verify = self._context(stack, [json.dumps(_batch(source)), legacy, legacy])
            with self.assertRaisesRegex(RuntimeError, "дважды нарушил"):
                planner.generate_itinerary(regional._request(), max_attempts=3, ground_candidates=True)
        self.assertEqual(generate.call_count, 3)
        verify.assert_not_called()

    def test_invalid_id_retry_keeps_same_catalog_and_schema(self):
        source=_source()
        bad=json.loads(_id_wire_response_json(source))
        bad['day_4']['attractions'][0]['candidate_id']='REJECTED_PRIVATE_MODEL_STRING'
        with ExitStack() as stack:
            generate,_,verify=self._context(stack,[json.dumps(_batch(source)),json.dumps(bad),_id_wire_response_json(source)])
            result=planner.generate_itinerary(regional._request(),max_attempts=3,ground_candidates=True)
        self.assertEqual((generate.call_count,result.problems),(3,[]))
        feedback=generate.call_args_list[2].args[0][-1]['content']
        self.assertIn('day_4.attractions[0].candidate_id',feedback)
        self.assertNotIn('CANONICAL_QUERY_CORRECTIONS',feedback)
        self.assertNotIn('REJECTED_PRIVATE',feedback)
        self.assertEqual(generate.call_args_list[1].kwargs['json_schema'],generate.call_args_list[2].kwargs['json_schema'])
        verify.assert_called_once()

    def test_repeated_enum_failure_is_typed_and_never_calls_a_fourth_time(self):
        source=_source();bad=json.loads(_id_wire_response_json(source))
        bad['day_4']['attractions'][0]['candidate_id']='REJECTED_PRIVATE_MODEL_STRING'
        with ExitStack() as stack:
            generate,_,verify=self._context(stack,[json.dumps(_batch(source)),json.dumps(bad),json.dumps(bad)])
            with self.assertRaises(planner.PlannerFormatError) as raised:
                planner.generate_itinerary(regional._request(),max_attempts=3,ground_candidates=True)
        self.assertEqual(generate.call_count,3)
        self.assertNotIn('REJECTED_PRIVATE',str(raised.exception))
        verify.assert_not_called()

    def test_malformed_wire_repair_keeps_rejected_source_and_never_calls_a_fourth_time(self):
        source = _source()
        regional._attraction(source.days[1]).duration_min = 90
        malformed = json.loads(_id_wire_response_json(source, [2, 3]))
        malformed["day_2"].pop("meal")
        before = source.model_dump()
        with ExitStack() as stack:
            generate, _, verify = self._context(stack, [
                json.dumps(_batch(source)), _id_wire_response_json(source), json.dumps(malformed),
            ])
            result = planner.generate_itinerary(regional._request(), max_attempts=3, ground_candidates=True)
        self.assertEqual((generate.call_count, result.attempts), (3, 3))
        self.assertTrue(result.problems)
        self.assertTrue(any("недостаточно содержательная программа" in problem for problem in result.problems))
        verify.assert_called_once()
        self.assertEqual(result.itinerary.model_dump(), before)
        self.assertEqual(source.model_dump(), before)

    def test_id_wire_restores_canonical_locality_without_changing_meal_or_rest(self):
        source = _source()
        for day in source.days:
            for item in day.items:
                item.area = ""
        with ExitStack() as stack:
            generate, _, verify = self._context(stack, [json.dumps(_batch(source)), _id_wire_response_json(source)])
            result = planner.generate_itinerary(regional._request(), max_attempts=3, ground_candidates=True)
        self.assertEqual((generate.call_count, result.problems), (2, []))
        self.assertTrue(all(item.area == ("Листвянка" if item.kind == "attraction" else "")
                            for day in result.itinerary.days for item in day.items))
        verify.assert_called_once()

    def test_budget_below_two_is_rejected_before_any_completion(self):
        with ExitStack() as stack:
            generate, lookup, _ = self._context(stack, [])
            with self.assertRaises(ValueError):
                planner.generate_itinerary(regional._request(), max_attempts=1, ground_candidates=True)
        generate.assert_not_called()
        lookup.assert_not_called()

    def test_impossible_wire_meal_budget_stops_before_schedule_and_preserves_usage(self):
        source = _source()
        request = regional._request().model_copy(update={"budget": 1, "travel_cost": 0, "lodging_cost": 0})
        with ExitStack() as stack:
            generate, _, verify = self._context(stack, [json.dumps(_batch(source))])
            result = planner.generate_itinerary(request, max_attempts=3, ground_candidates=True)
        self.assertEqual((generate.call_count, result.attempts), (1, 1))
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (11, 13))
        self.assertEqual(result.itinerary.days, [])
        self.assertTrue(any("стоимость обеда" in problem and "бюджет" in problem for problem in result.problems))
        verify.assert_not_called()

    def test_local_wire_schema_error_is_rejected_without_claiming_model_failure(self):
        source = _source()
        with ExitStack() as stack:
            generate, _, verify = self._context(stack, [json.dumps(_batch(source))])
            stack.enter_context(patch.object(planner, "build_grounded_response_schema", side_effect=ValueError("Несовместимая схема")))
            result = planner.generate_itinerary(regional._request(), max_attempts=3, ground_candidates=True)
        self.assertEqual((generate.call_count, result.attempts), (1, 1))
        self.assertEqual((result.prompt_tokens, result.completion_tokens), (11, 13))
        self.assertEqual(result.itinerary.days, [])
        self.assertEqual(result.problems, ["Не удалось подготовить формат маршрута: Несовместимая схема"])
        verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
