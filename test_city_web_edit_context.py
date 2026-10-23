"""Returned edit manifests are persisted, while map evidence stays immutable."""

from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import city_context
import city_edit_service
from city_deterministic import build_deterministic_city_itinerary
from city_edit import prepare_city_edit
import db
import planner
import web_service
from test_city_deterministic import request, candidates, CENTER
from test_city_edit import wire_for


class CityWebEditContextTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="web-edit-manifest-"))
        self.stack.enter_context(patch.object(db, "DB_PATH", str(Path(temporary) / "synthetic.db")))
        for name in ("httpx.get", "httpx.post", "places.geocode", "places._load_cache",
                     "places._save_cache", "planner.generate_json", "city_context.recover_city_context"):
            self.stack.enter_context(patch(name, side_effect=AssertionError("External I/O forbidden")))
        self.stack.enter_context(patch.object(web_service, "_daily_ai_limit", return_value=30))
        self.stack.enter_context(patch.object(web_service, "city_candidates_enabled", return_value=False))
        self.req = web_service._normalized_trip_request(request(2))
        self.pool = candidates(6)
        built = build_deterministic_city_itinerary(self.req, self.pool, resolved_city=CENTER)
        self.assertEqual(built.problems, [])
        self.original = built.itinerary
        self.snapshot = city_context.attach_city_presentation(
            city_context.build_city_context(self.req, self.pool, CENTER), self.original, self.pool, self.req)
        self.assertIn("presentation", self.snapshot)
        db.init_db()
        db.save_user(100, "Synthetic", 2, [], "RUB")
        self.trip_id = db.save_trip_bundle(100, {**self.req.model_dump(), "weather": None,
                                               city_context.CONTEXT_KEY: self.snapshot},
                                           self.original.model_dump(), [("General", "Synthetic")])
        self.before = deepcopy(db.get_trip(self.trip_id, 100))
        self.edit = self.stack.enter_context(patch.object(web_service, "edit_itinerary"))

    def result(self, snapshot=None, itinerary=None):
        return planner.PlanResult(itinerary=itinerary or self.original.model_copy(deep=True), weather=None,
                                  city_map_context=deepcopy(self.snapshot if snapshot is None else snapshot), attempts=2)

    def test_actual_two_edits_persist_refreshed_manifest_and_keep_unchanged_map_evidence(self):
        returned = []
        completions = []
        def edit(req, original, target, instruction, *, city_map_context, prior_weather):
            pool, center = city_context_module.load_city_context(city_map_context, req)
            managed = city_context_module.city_presentation_tips(city_map_context, original, pool, req)
            context = prepare_city_edit(req, original, target, candidates=pool, resolved_city=center,
                canonical_day_schema=planner.build_itinerary_day_json_schema(req, target), managed_tip_values=managed)
            wire = wire_for(context)
            if not returned:
                wire["day_1"]["attractions"][0]["candidate_id"] = self.pool[4].candidate_id
            else:
                wire["day_1"]["meal"]["time"] = "13:30"
                wire["day_1"]["rest"]["time"] = "14:30"
            complete = Mock(side_effect=[SimpleNamespace(text=json.dumps(value, ensure_ascii=False),
                                  input_tokens=1, output_tokens=1) for value in (wire, {"fulfilled": True, "problems": []})])
            with patch.object(city_edit_service, "get_weather", return_value=None), \
                    patch.object(planner, "_load_blocked_places", return_value=[]):
                result = city_edit_service.edit_city_itinerary(req, original, target, instruction,
                    city_map_context=city_map_context, prior_weather=prior_weather, completion=complete, max_attempts=1)
            self.assertEqual(result.problems, [])
            returned.append(result)
            completions.append(complete.call_count)
            return result
        city_context_module = city_context  # Avoid the editor's API keyword shadowing the module.
        self.edit.side_effect = edit
        first = web_service.edit_trip_day(100, self.trip_id, 1, "Замени первое посещение")
        saved = db.get_trip(self.trip_id, 100)
        refreshed = saved["request"][city_context.CONTEXT_KEY]
        self.assertEqual(first["revision"], 2)
        self.assertNotEqual(refreshed["presentation"], self.snapshot["presentation"])
        self.assertEqual(refreshed, returned[0].city_map_context)
        self.assertEqual({k: v for k, v in refreshed.items() if k != "presentation"},
                         {k: v for k, v in self.snapshot.items() if k != "presentation"})
        with self.assertRaises(ValueError):
            city_context.city_presentation_tips(self.snapshot, returned[0].itinerary, self.pool, self.req)
        second = web_service.edit_trip_day(100, self.trip_id, 1, "Перенеси обед на 13:30, отдых на 14:30")
        final = db.get_trip(self.trip_id, 100)
        self.assertEqual(second["revision"], 3)
        self.assertEqual(completions, [2, 2])
        self.assertEqual(final["request"][city_context.CONTEXT_KEY], returned[1].city_map_context)
        self.assertEqual(final["itinerary"]["days"][1], self.before["itinerary"]["days"][1])
        self.assertNotIn(city_context.CONTEXT_KEY, second["request"])
        self.assertEqual(len(db.get_checklist(self.trip_id)), 1)

    def test_malformed_returned_context_or_stale_manifest_never_reaches_save(self):
        for change in ("missing", "shape", "unmatched_hash"):
            with self.subTest(change=change):
                result = self.result()
                if change == "missing":
                    result.city_map_context = None
                elif change == "shape":
                    result.city_map_context["presentation"] = {"version": True, "managed_tip_sha256": []}
                else:
                    result.city_map_context["presentation"]["managed_tip_sha256"] = ["0" * 64]
                self.edit.return_value = result
                with patch.object(db, "update_trip_plan", wraps=db.update_trip_plan) as save, \
                        self.assertRaises(web_service.EditRejectedError):
                    web_service.edit_trip_day(100, self.trip_id, 1, "Перенеси обед")
                save.assert_not_called()
                self.assertEqual(db.get_trip(self.trip_id, 100), self.before)

    def test_revalidated_but_changed_unselected_record_center_or_capture_is_rejected(self):
        for change in ("unselected_record", "center", "captured_at", "pool_scope"):
            with self.subTest(change=change):
                result = self.result()
                payload = result.city_map_context
                if change == "unselected_record":
                    payload["candidates"][-1]["record"]["lat"] += .001
                elif change == "center":
                    payload["resolved_city"]["lat"] += .001
                elif change == "captured_at":
                    payload["captured_at"] = "2099-01-01T00:00:00+00:00"
                else:
                    payload["pool_scope"] = "existing_visits_rechecked"
                pool, _ = city_context.load_city_context(payload, self.req)
                city_context.check_context_places(result.itinerary, pool)
                self.edit.return_value = result
                with patch.object(db, "update_trip_plan", wraps=db.update_trip_plan) as save, \
                        self.assertRaises(web_service.EditRejectedError):
                    web_service.edit_trip_day(100, self.trip_id, 1, "Перенеси обед")
                save.assert_not_called()
                self.assertEqual(db.get_trip(self.trip_id, 100), self.before)

    def test_editor_cannot_mutate_original_context_to_bypass_evidence_comparison(self):
        def edit(*args, **kwargs):
            supplied = kwargs["city_map_context"]
            supplied["candidates"][-1]["record"]["lat"] += .001
            return self.result(supplied)
        self.edit.side_effect = edit
        with patch.object(db, "update_trip_plan", wraps=db.update_trip_plan) as save, \
                self.assertRaises(web_service.EditRejectedError):
            web_service.edit_trip_day(100, self.trip_id, 1, "Перенеси обед")
        save.assert_not_called()
        self.assertEqual(db.get_trip(self.trip_id, 100), self.before)


if __name__ == "__main__":
    unittest.main()
