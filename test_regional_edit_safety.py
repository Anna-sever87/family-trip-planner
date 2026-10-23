"""Offline regional edits retain local-base and neighbouring-transfer rules."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import places
import planner
import test_regional_generation_repair as fixture
from yandex_ai import GenerationResult


BASE_A = "База А, Россия"
BASE_B = "База Б, Россия"
BASE_C = "База В, Россия"


def _source(days=2):
    source = fixture._valid_source()
    source.days = source.days[:days]
    for index, day in enumerate(source.days):
        day.base_area = BASE_A if index == 0 else BASE_B
        day.transfer_note = "Переезд База А -> База Б; дорогу уточнить." if index == 1 else ""
        for item in day.items:
            item.area = ""
    return source


def _base_record(query, **overrides):
    return {
        "name": query, "lat": 51.852, "lon": 104.902,
        "type": "village", "category": "place",
        "address": {"country": "Россия", "country_code": "ru"},
        **overrides,
    }


class RegionalEditSafetyTests(unittest.TestCase):
    def _run(self, source, changed, target, *, base_overrides=None, place_overrides=None,
             blocked_places=None, raw=None):
        request = fixture._request().model_copy(update={"end_date": source.days[-1].date})
        records = {query: _base_record(query) for query in (BASE_A, BASE_B, BASE_C)}
        records.update(base_overrides or {})
        for index, day in enumerate(source.days):
            item = fixture._attraction(day)
            # Synthetic nearby coordinates and long durations isolate edit safety,
            # not actual geography or visit duration of the named test fixtures.
            records[item.place_query] = {
                "name": item.place_query, "lat": 51.85 + index * .006, "lon": 104.91,
                "type": "museum", "category": "tourism",
            }
        records.update(place_overrides or {})
        before = source.model_dump()
        with ExitStack() as stack:
            stack.enter_context(patch.object(planner, "resolve_region", return_value=fixture._scope()))
            stack.enter_context(patch.object(planner, "curated_blocked_place_terms", return_value=blocked_places or []))
            completion = stack.enter_context(patch.object(
                planner, "generate_json", return_value=GenerationResult(raw if raw is not None else changed.model_dump_json(), 1, 1),
            ))
            audit = stack.enter_context(patch.object(planner, "_audit_edit_fulfillment", return_value=([], 0, 0)))
            lookup = stack.enter_context(patch.object(places, "geocode", side_effect=lambda query, **_: records[query]))
            stack.enter_context(patch.object(places, "_search", side_effect=AssertionError("No network")))
            stack.enter_context(patch.object(places, "_save_cache", side_effect=AssertionError("No cache writes")))
            result = planner.edit_itinerary(request, source, target, "Измени выбранный день", max_attempts=1)
        self.assertEqual(source.model_dump(), before, "the saved source must not be mutated")
        self.assertEqual(completion.call_count, 1)
        self.assertEqual(result.attempts, 1)
        return result, lookup, audit

    def test_first_base_change_rejects_stale_next_transfer_without_mutating_neighbour(self):
        source = _source()
        changed = source.days[0].model_copy(deep=True)
        changed.base_area = BASE_C
        result, lookup, audit = self._run(source, changed, 1)
        self.assertTrue(any("переезд дня 2" in problem for problem in result.problems), result.problems)
        self.assertEqual(result.itinerary.days[1], source.days[1])
        lookup.assert_not_called()
        audit.assert_not_called()

    def test_new_transfer_to_unchanged_next_base_cannot_be_silently_added(self):
        source = _source()
        source.days[1].base_area = BASE_A
        source.days[1].transfer_note = ""
        changed = source.days[0].model_copy(deep=True)
        changed.base_area = BASE_C
        result, _, _ = self._run(source, changed, 1)
        self.assertTrue(any("переезд дня 2" in problem for problem in result.problems), result.problems)
        self.assertEqual(result.itinerary.days[1], source.days[1])

    def test_merging_with_next_base_still_rejects_obsolete_nonempty_transfer_note(self):
        source = _source()
        changed = source.days[0].model_copy(deep=True)
        changed.base_area = BASE_B
        result, _, _ = self._run(source, changed, 1)
        self.assertTrue(any("переезд дня 2" in problem for problem in result.problems), result.problems)

    def test_middle_base_change_checks_both_incoming_and_outgoing_links(self):
        source = _source(3)
        changed = source.days[1].model_copy(deep=True)
        changed.base_area = BASE_C
        result, lookup, _ = self._run(source, changed, 2)
        self.assertTrue(any("обновить transfer_note" in problem for problem in result.problems), result.problems)
        self.assertTrue(any("переезд дня 3" in problem for problem in result.problems), result.problems)
        self.assertEqual(result.itinerary.days[0], source.days[0])
        self.assertEqual(result.itinerary.days[2], source.days[2])
        lookup.assert_not_called()

    def test_last_base_change_rejects_stale_incoming_transfer(self):
        source = _source()
        changed = source.days[-1].model_copy(deep=True)
        changed.base_area = BASE_C
        result, lookup, audit = self._run(source, changed, 2)
        self.assertTrue(any("обновить transfer_note" in problem for problem in result.problems), result.problems)
        lookup.assert_not_called()
        audit.assert_not_called()

    def test_last_base_change_with_updated_transfer_is_validated_and_allowed(self):
        source = _source()
        changed = source.days[-1].model_copy(deep=True)
        changed.base_area = BASE_C
        changed.transfer_note = "Переезд База А -> База В; дорогу уточнить."
        result, lookup, audit = self._run(source, changed, 2)
        self.assertEqual(result.problems, [])
        self.assertEqual(result.itinerary.days[-1].base_area, BASE_C)
        self.assertEqual(result.itinerary.days[0].model_dump(exclude={"items"}), source.days[0].model_dump(exclude={"items"}))
        self.assertGreater(lookup.call_count, 0)
        audit.assert_called_once()

    def test_ordinary_edit_with_unchanged_bases_still_succeeds(self):
        source = _source()
        changed = source.days[0].model_copy(deep=True)
        changed.theme = "Обновлённая программа дня"
        result, _, audit = self._run(source, changed, 1)
        self.assertEqual(result.problems, [])
        self.assertEqual(result.itinerary.days[1].transfer_note, source.days[1].transfer_note)
        audit.assert_called_once()

    def test_natural_and_large_administrative_objects_are_not_edit_bases(self):
        for place_type, category in (("lake", "natural"), ("peak", "natural"), ("administrative", "boundary"), ("county", "place")):
            with self.subTest(place_type=place_type):
                source = _source()
                changed = source.days[-1].model_copy(deep=True)
                changed.base_area = BASE_C
                changed.transfer_note = "Переезд База А -> База В; дорогу уточнить."
                result, _, audit = self._run(source, changed, 2, base_overrides={
                    BASE_C: _base_record(BASE_C, type=place_type, category=category, addresstype="county"),
                })
                self.assertTrue(any("не подтверждена как населённый пункт" in problem for problem in result.problems), result.problems)
                audit.assert_not_called()

    def test_base_without_map_type_cannot_be_confirmed(self):
        source = _source()
        changed = source.days[-1].model_copy(deep=True)
        changed.theme = "Обновлённый день"
        result, _, _ = self._run(source, changed, 2, base_overrides={BASE_B: _base_record(BASE_B, type="", category="")})
        self.assertTrue(any("не подтверждена как населённый пункт" in problem for problem in result.problems), result.problems)

    def test_administrative_boundary_of_a_town_is_a_valid_local_base(self):
        source = _source()
        changed = source.days[-1].model_copy(deep=True)
        changed.theme = "Обновлённый день"
        result, _, _ = self._run(source, changed, 2, base_overrides={
            BASE_B: _base_record(BASE_B, type="administrative", category="boundary", addresstype="town"),
        })
        self.assertEqual(result.problems, [])

    def test_wrong_actual_country_is_rejected_even_with_matching_display_name(self):
        source = _source()
        changed = source.days[-1].model_copy(deep=True)
        changed.theme = "Обновлённый день"
        result, _, audit = self._run(source, changed, 2, base_overrides={
            BASE_B: _base_record(BASE_B, address={"country": "United States", "country_code": "us"}),
        })
        self.assertTrue(any("название и страну" in problem for problem in result.problems), result.problems)
        audit.assert_not_called()

    def test_valid_settlement_outside_region_still_fails(self):
        source = _source()
        changed = source.days[-1].model_copy(deep=True)
        changed.theme = "Обновлённый день"
        result, _, _ = self._run(source, changed, 2, base_overrides={
            BASE_B: _base_record(BASE_B, lat=65.0, lon=130.0),
        })
        self.assertTrue(any("база" in problem and "за пределами" in problem for problem in result.problems), result.problems)

    def test_absent_base_is_not_restored_or_accepted_by_edit(self):
        source = _source()
        changed = source.days[-1].model_copy(deep=True)
        changed.base_area = ""
        changed.transfer_note = "База не выбрана"
        result, _, _ = self._run(source, changed, 2)
        self.assertTrue(any("base_area" in problem for problem in result.problems), result.problems)

    def test_county_canonical_poi_survives_ordinary_positive_cache_recheck(self):
        record = {
            "name": "Камень Черского, Иркутский район, Россия", "lat": 51.88, "lon": 104.85,
            "type": "peak", "category": "natural",
            "address": {"county": "Иркутский район", "country": "Россия", "country_code": "ru"},
        }
        canonical = places._nearby_named_record("Камень Черского", "Россия", 51.87, 104.85, 80, record)
        self.assertIsNotNone(canonical)
        self.assertEqual(canonical["place_query"], "Камень Черского, Иркутский район, Россия")
        with (
            patch.object(places, "_cache", {canonical["place_query"].lower(): record}),
            patch.object(places, "_search", side_effect=AssertionError("No network")),
            patch.object(places, "_save_cache", side_effect=AssertionError("No cache writes")),
        ):
            self.assertEqual(places.geocode(canonical["place_query"]), record)


if __name__ == "__main__":
    unittest.main()
