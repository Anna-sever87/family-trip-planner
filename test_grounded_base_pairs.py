"""Schedule choices retain verified base/POI pairs and exact map localities."""

from contextlib import ExitStack
from copy import deepcopy
import json
import socket
import unittest
from unittest.mock import patch

import candidate_grounding as candidates
import grounded_schedule as scheduling
import places
import planner
import test_grounded_planner as integrated
import test_grounded_schedule as fixtures
import test_regional_generation_repair as regional


class GroundedBasePairsTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(socket.socket, "connect", side_effect=AssertionError("No network"))
        guard.start()
        self.addCleanup(guard.stop)

    def test_explicit_conflicting_counties_reject_even_shared_settlement_name(self):
        self.assertFalse(candidates._administratively_compatible(
            {"village": "Одинаковое имя", "county": "Иркутский муниципальный округ"},
            {"village": "Одинаковое имя", "county": "Слюдянский район"},
        ))

    def test_shared_settlement_or_county_or_municipality_is_required(self):
        for field in ("village", "town", "county", "municipality"):
            with self.subTest(field=field):
                self.assertTrue(candidates._administratively_compatible({field: "Тестовый"}, {field: "Тестовый"}))
        self.assertFalse(candidates._administratively_compatible({"country_code": "ru"}, {"country_code": "ru"}))
        self.assertFalse(candidates._administratively_compatible({"state": "Область"}, {"state": "Область"}))
        self.assertFalse(candidates._administratively_compatible({"county": "Район", "country_code": "ru"}, {"county": "Район", "country_code": "us"}))

    def test_slyudyanka_poi_cannot_be_attached_to_taltsy_across_county_boundary(self):
        bases = {
            "base_1": {"lat": 51.99081, "lon": 104.669983,
                       "map_address": {"village": "Тальцы", "county": "Иркутский муниципальный округ", "country_code": "ru"}},
            "base_2": {"lat": 51.66, "lon": 103.72,
                       "map_address": {"town": "Слюдянка", "county": "Слюдянский район", "country_code": "ru"}},
        }
        poi = {"lat": 51.68, "lon": 103.76,
               "address": {"town": "Слюдянка", "county": "Слюдянский район", "country_code": "ru"}}
        self.assertLess(places.distance_km(bases["base_1"]["lat"], bases["base_1"]["lon"], poi["lat"], poi["lon"]), 80)
        self.assertEqual(candidates._allowed_base_keys(poi, bases), ["base_2"])

    def test_compatible_county_does_not_override_radius(self):
        bases = {"base_1": {"lat": 51.85, "lon": 104.85, "map_address": {"county": "Район"}}}
        self.assertEqual(candidates._allowed_base_keys({"lat": 54, "lon": 104.85, "address": {"county": "Район"}}, bases), [])

    def test_first_cached_namesake_with_conflicting_county_does_not_hide_valid_match(self):
        title = "Музей народного быта"
        wrong_query, right_query = f"{title}, Слюдянка, Россия", f"{title}, Листвянка, Россия"
        records = {
            wrong_query.lower(): {"name": wrong_query, "lat": 51.87, "lon": 104.86, "type": "museum", "category": "tourism",
                                  "address": {"town": "Слюдянка", "county": "Слюдянский район", "country_code": "ru"}},
            right_query.lower(): {"name": right_query, "lat": 51.88, "lon": 104.86, "type": "museum", "category": "tourism",
                                  "address": {"village": "Листвянка", "county": "Иркутский муниципальный округ", "country_code": "ru"}},
        }
        base = {"lat": 51.86, "lon": 104.85, "map_address": {"village": "Листвянка", "county": "Иркутский муниципальный округ", "country_code": "ru"}}
        budget = candidates.GroundingResult(query_limit=1, http_attempt_limit=0)
        with patch.object(places, "_load_cache", return_value=records), patch.object(places, "_save_cache"), patch.object(places, "geocode_nearby_named", side_effect=AssertionError("cache hit only")):
            record, reason = candidates._lookup_nearby_poi(title, "Россия", base, budget, scope=fixtures.scope_for())
        self.assertEqual(reason, "")
        self.assertEqual(record["place_query"], right_query)
        self.assertEqual(budget.reserved_http_attempts_upper_bound, 0)

    def pool_with_two_bases(self):
        grounding = fixtures.grounding_for()
        grounding.base_facts.append({"base_key": "base_2", "base_query": "Тальцы, Россия", "lat": 51.99, "lon": 104.67})
        for item in grounding.confirmed_hints:
            item["allowed_base_keys"] = ["base_1"]
        return grounding

    def test_pool_preserves_pair_and_locality_facts_without_mutation(self):
        grounding = self.pool_with_two_bases()
        before = deepcopy(grounding)
        _, pool = scheduling._pool(grounding)
        self.assertEqual(pool[0]["allowed_base_keys"], ["base_1"])
        self.assertEqual(pool[0]["locality"], "Листвянка")
        pool[0]["allowed_base_keys"].append("base_2")
        self.assertEqual(grounding, before)

    def test_approved_base_alone_is_insufficient_if_poi_pair_is_forbidden(self):
        itinerary = fixtures.itinerary_for()
        itinerary.days[0].base_area = "Тальцы, Россия"
        problems = scheduling.enforce_grounded_choices(itinerary, self.pool_with_two_bases())
        self.assertTrue(any("несовместимо с базой" in problem for problem in problems))

    def test_explicit_empty_or_invalid_pair_list_does_not_use_legacy_fallback(self):
        for allowed in ([], ["unknown"], "base_1", None, ["base_2"]):
            with self.subTest(allowed=allowed):
                grounding = self.pool_with_two_bases()
                grounding.confirmed_hints[0]["allowed_base_keys"] = allowed
                self.assertTrue(any("отсутствует в подтверждённом наборе" in p for p in scheduling.enforce_grounded_choices(fixtures.itinerary_for(), grounding)))

    def test_legacy_fact_fallback_only_authorizes_its_original_base(self):
        grounding = self.pool_with_two_bases()
        for item in grounding.confirmed_hints:
            item.pop("allowed_base_keys")
            item.pop("locality")
        self.assertEqual(scheduling.enforce_grounded_choices(fixtures.itinerary_for(), grounding), [])
        itinerary = fixtures.itinerary_for()
        itinerary.days[0].base_area = "Тальцы, Россия"
        self.assertTrue(scheduling.enforce_grounded_choices(itinerary, grounding))

    def test_area_is_empty_or_selected_candidate_locality_not_any_other_pool_locality(self):
        grounding = self.pool_with_two_bases()
        grounding.confirmed_hints[1]["locality"] = "Иркутский муниципальный округ"
        for area, accepted in (("", True), ("Листвянка", True), ("посёлок Листвянка", False),
                               ("Иркутский муниципальный округ", False), ("Париж", False)):
            with self.subTest(area=area):
                itinerary = fixtures.itinerary_for()
                itinerary.days[0].items[0].area = area
                problems = scheduling.enforce_grounded_choices(itinerary, grounding)
                self.assertEqual(not problems, accepted)

    def test_schema_area_enum_applies_to_full_and_keyed_repair_schema_without_mutation(self):
        grounding = self.pool_with_two_bases()
        schema = {"type": "object", "properties": {"day_2": {"type": "object", "properties": {
            "base_area": {"type": "string"}, "items": {"type": "array", "items": {"type": "object", "properties": {
                "area": {"type": "string"}, "place_query": {"type": "string"},
            }}},
        }}}}
        before = deepcopy(schema)
        narrowed = scheduling.constrain_grounded_schema(schema, grounding)
        self.assertEqual(narrowed["properties"]["day_2"]["properties"]["items"]["items"]["properties"]["area"]["enum"], ["", "Листвянка"])
        self.assertEqual(schema, before)
        self.assertIn("allowed_base_keys", scheduling.GROUNDED_SYSTEM_PROMPT)
        self.assertIn("area — дословная locality", scheduling.GROUNDED_SYSTEM_PROMPT)

    def test_no_arbitrary_area_search_before_grounded_contract_is_satisfied(self):
        source = integrated._source()
        original = source.model_copy(deep=True)
        regional._attraction(source.days[1]).area = "Несуществующая придуманная местность"
        with ExitStack() as stack:
            helper = integrated.GroundedPlannerTests()
            generate, lookup, verify = helper._context(stack, [
                json.dumps(integrated._batch(original)), integrated._wire_response_json(source),
            ])
            # Strict wire enums reject arbitrary areas before map validation.
            with self.assertRaisesRegex(RuntimeError, "нужном формате"):
                planner.generate_itinerary(regional._request(), max_attempts=2, ground_candidates=True)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(lookup.call_count, 14)  # Candidate names only; no invented-area search.
        verify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
