"""Managed city evidence survives save/edit without authorizing model facts."""

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import city_edit_service
import city_evidence
from city_context import (attach_city_presentation, build_city_context, city_presentation_tips,
                          load_city_context)
from city_deterministic import build_deterministic_city_itinerary
from city_edit import prepare_city_edit, city_edit_prompt_data, decode_city_edit, validate_city_edit
from planner import build_itinerary_day_json_schema
from test_city_deterministic import request, candidates, CENTER, visits
from test_city_edit import wire_for


class CityEditEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name in ("httpx.get", "httpx.post", "places.geocode", "places._load_cache",
                     "places._save_cache", "sqlite3.connect", "planner.generate_json"):
            mock = self.stack.enter_context(patch(name, side_effect=AssertionError("No external I/O")))
            self.addCleanup(mock.assert_not_called)
        self.req = request(2)
        self.pool = candidates(6)
        result = build_deterministic_city_itinerary(self.req, self.pool, resolved_city=CENTER)
        self.assertEqual(result.problems, [])
        self.original = result.itinerary
        self.original.tips.insert(1, "Личное примечание, которое нельзя переписывать.")
        self.snapshot = build_city_context(self.req, self.pool, CENTER)
        self.marked = attach_city_presentation(self.snapshot, self.original, self.pool, self.req)

    def context(self, payload=None):
        managed = city_presentation_tips(payload or self.marked, self.original, self.pool, self.req)
        return prepare_city_edit(self.req, self.original, 1, candidates=self.pool, resolved_city=CENTER,
                                 canonical_day_schema=build_itinerary_day_json_schema(self.req, 1),
                                 managed_tip_values=managed)

    def replacement(self, context):
        wire = wire_for(context)
        wire["day_1"]["attractions"][0]["candidate_id"] = self.pool[4].candidate_id
        return decode_city_edit(json.dumps(wire, ensure_ascii=False), context)

    def test_manifest_is_private_bounded_copy_and_legacy_context_loads(self):
        self.assertNotIn("presentation", self.snapshot)
        self.assertIn("presentation", self.marked)
        self.assertEqual(self.marked["presentation"]["version"], 1)
        self.assertTrue(all(len(value) == 64 for value in self.marked["presentation"]["managed_tip_sha256"]))
        self.assertEqual(load_city_context(self.snapshot, self.req), load_city_context(self.marked, self.req))
        self.assertNotIn("Личное примечание", json.dumps(self.marked["presentation"], ensure_ascii=False))

    def test_replacement_rebuilds_only_marked_tips_and_preserves_other_days(self):
        context = self.context()
        result = self.replacement(context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        self.assertEqual(result.days[1:].__repr__(), self.original.days[1:].__repr__())
        self.assertEqual([day.model_dump() for day in result.days[1:]], [day.model_dump() for day in self.original.days[1:]])
        self.assertFalse(any(self.pool[0].title in note for note in result.tips))
        self.assertTrue(any(self.pool[4].title in note for note in result.tips))
        self.assertIn("Личное примечание, которое нельзя переписывать.", result.tips)
        original_unmanaged = [value for value in self.original.tips if value not in context.managed_tip_values]
        new_managed = city_evidence.city_managed_tips(result, self.pool, self.req)
        self.assertEqual([value for value in result.tips if value not in new_managed], original_unmanaged)

    def test_legacy_unmarked_evidence_has_exact_match_migration(self):
        self.original.tips = ["Ручная запись"] + city_evidence.city_managed_tips(
            self.original, self.pool, self.req, legacy=True)
        context = self.context(self.snapshot)
        result = self.replacement(context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        self.assertIn("Ручная запись", result.tips)
        self.assertFalse(any(self.pool[0].title in value for value in result.tips))

    def test_marker_prefix_alone_is_not_authority(self):
        forged = "Сведения о посещении [osm-node-101]. Неизвестный пользовательский текст."
        self.original.tips.append(forged)
        payload = attach_city_presentation(self.snapshot, self.original, self.pool, self.req)
        owned = city_presentation_tips(payload, self.original, self.pool, self.req)
        self.assertNotIn(forged, owned)
        context = self.context(payload)
        result = self.replacement(context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        self.assertIn(forged, result.tips)

    def test_model_and_direct_candidate_cannot_change_arbitrary_tips(self):
        context = self.context()
        wire = wire_for(context)
        wire["tips"] = ["Injected"]
        with self.assertRaises(ValueError):
            decode_city_edit(json.dumps(wire, ensure_ascii=False), context)
        result = self.replacement(context)
        result.tips.append("Injected")
        self.assertTrue(validate_city_edit(result, context, blocked_places=[]))

    def test_manifest_shape_missing_hash_and_duplicate_hash_fail_closed(self):
        digest = self.marked["presentation"]["managed_tip_sha256"][0]
        for presentation in ({"version": True, "managed_tip_sha256": []},
                             {"version": 1, "managed_tip_sha256": [digest, digest]},
                             {"version": 1, "managed_tip_sha256": ["X" * 64]},
                             {"version": 1, "managed_tip_sha256": [], "other": True},
                             {"version": 1, "managed_tip_sha256": [f"{i:064x}" for i in range(33)]}):
            malformed = dict(self.marked, presentation=presentation)
            with self.subTest(presentation=presentation), self.assertRaises(ValueError):
                load_city_context(malformed, self.req)
        missing = dict(self.marked, presentation={"version": 1, "managed_tip_sha256": ["0" * 64]})
        with self.assertRaises(ValueError):
            city_presentation_tips(missing, self.original, self.pool, self.req)

    def test_manifest_can_replace_old_server_text_after_sources_change(self):
        old = tuple(city_presentation_tips(self.marked, self.original, self.pool, self.req))
        current_option = city_evidence.visit_option
        def changed(candidate, req):
            return replace(current_option(candidate, req), unknown_fields=("price", "hours"))
        with patch.object(city_evidence, "visit_option", side_effect=changed):
            context = self.context()
            result = self.replacement(context)
            self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
            new = attach_city_presentation(self.marked, result, self.pool, self.req)
        self.assertNotEqual(new["presentation"], self.marked["presentation"])
        self.assertFalse(any(value in result.tips for value in old if "Сведения о посещении" in value))

    def test_changing_duration_does_not_relabel_it_as_published_fact(self):
        context = self.context()
        wire = wire_for(context)
        wire["day_1"]["attractions"][0]["duration_min"] = 100
        wire["day_1"]["attractions"][1]["time"] = "11:00"
        result = decode_city_edit(json.dumps(wire, ensure_ascii=False), context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        note = next(value for value in result.tips if "[osm-node-101]" in value)
        self.assertIn("100 мин. — оценка изменённого расписания", note)
        self.assertIn("Исходный ориентир: 90 мин.", note)

    def test_unknown_indoor_estimate_no_longer_claims_not_used_by_rain(self):
        context = self.context()
        wire = wire_for(context)
        wire["day_1"]["attractions"][0]["indoor"] = True
        wire["day_1"]["meal"]["time"] = "13:30"
        wire["day_1"]["rest"]["time"] = "14:30"
        result = decode_city_edit(json.dumps(wire, ensure_ascii=False), context)
        self.assertEqual(validate_city_edit(result, context, blocked_places=[]), [])
        self.assertIn(city_evidence.UNKNOWN_EDITED_INDOOR_NOTE, result.tips)
        self.assertNotIn(city_evidence.UNKNOWN_INDOOR_NOTE, result.tips)

    def test_successful_service_updates_manifest_and_failure_preserves_original(self):
        context = self.context()
        wire = wire_for(context)
        wire["day_1"]["attractions"][0]["candidate_id"] = self.pool[4].candidate_id
        def response(value):
            return SimpleNamespace(text=json.dumps(value, ensure_ascii=False), input_tokens=1, output_tokens=1)
        for accepted in (True, False):
            complete = Mock(side_effect=[response(wire), response({"fulfilled": accepted,
                                      "problems": [] if accepted else ["Synthetic rejection"]})])
            with patch.object(city_edit_service, "get_weather", return_value=None), \
                    patch("planner._load_blocked_places", return_value=[]):
                result = city_edit_service.edit_city_itinerary(
                    self.req, self.original, 1, "Замени первое посещение", city_map_context=self.marked,
                    completion=complete, max_attempts=1)
            self.assertEqual(complete.call_count, 2)
            if accepted:
                self.assertEqual(result.problems, [])
                self.assertNotEqual(result.city_map_context["presentation"], self.marked["presentation"])
                self.assertTrue(city_presentation_tips(result.city_map_context, result.itinerary, self.pool, self.req))
            else:
                self.assertTrue(result.problems)
                self.assertEqual(result.city_map_context, self.marked)
                self.assertEqual(result.itinerary.model_dump(), self.original.model_dump())

    def test_bad_manifest_is_rejected_before_weather_or_any_completion(self):
        malformed = deepcopy(self.marked)
        malformed["presentation"]["managed_tip_sha256"] = ["0" * 64]
        with patch.object(city_edit_service, "get_weather") as weather:
            complete = Mock()
            with self.assertRaises(ValueError):
                city_edit_service.edit_city_itinerary(self.req, self.original, 1, "Измени обед",
                    city_map_context=malformed, completion=complete)
            complete.assert_not_called()
            weather.assert_not_called()

    def test_known_windows_scope_and_sources_reach_edit_prompt_before_ai(self):
        from test_city_visit_options import FixedDate, ocean, yersin
        req = request(5, destination="Нячанг, Вьетнам", start_date="2026-10-28", end_date="2026-11-01",
                      children_ages=[5], currency="USD", budget=3000)
        pool = [ocean(), yersin()]
        center = {"name": "Нячанг", "lat": 12.25, "lon": 109.19, "country_code": "VN"}
        with patch("city_visit_options.date", FixedDate):
            original = build_deterministic_city_itinerary(req, pool, resolved_city=center).itinerary
            context = prepare_city_edit(req, original, 1, candidates=pool, resolved_city=center,
                                         canonical_day_schema=build_itinerary_day_json_schema(req, 1))
            values = city_edit_prompt_data(context)["available_candidates"]
            yersin_data = next(value["visit_conditions"] for value in values if value["candidate_id"] == yersin().candidate_id)
            self.assertEqual(yersin_data["duration_min"], 30)
            self.assertEqual(yersin_data["duration_basis"], "published")
            self.assertTrue(yersin_data["opening_windows"])
            self.assertIn("экспозиции", yersin_data["description_scope"])
            self.assertTrue(yersin_data["sources"])
            sunday = prepare_city_edit(req, original, 5, candidates=pool, resolved_city=center,
                                        canonical_day_schema=build_itinerary_day_json_schema(req, 5))
            self.assertNotIn(yersin().candidate_id, [value.candidate_id for value in sunday.allowed_candidates])


if __name__ == "__main__":
    unittest.main()
