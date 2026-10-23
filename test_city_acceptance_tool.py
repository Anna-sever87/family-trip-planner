"""Offline saved-artifact acceptance; invented records, disposable SQLite only."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from city_candidates import CityCandidate
from program_coverage import rebuild_program_report
from test_city_limited_storage import city_partial_fixture
from tools import check_city_acceptance as acceptance


class CityAcceptanceToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="saved-city-acceptance-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.request, self.itinerary = city_partial_fixture()
        for day in self.itinerary.days:
            for item in day.items:
                if item.kind == "rest":
                    item.duration_min = 60  # Existing child-under-seven rule.
        self.center = {"name": "Тестовый город", "lat": 43.6, "lon": 39.72, "country_code": "RU", "admin1": ""}
        candidates = []
        for day in self.itinerary.days:
            for item in day.items:
                if item.kind != "attraction":
                    continue
                identity = int(item.source_url.rsplit("/", 1)[-1])
                record = {"lat": item.lat, "lon": item.lon, "name": item.place_query,
                          "provider": "osm", "osm_type": "node", "osm_id": identity,
                          "type": "museum", "category": "tourism",
                          "address": {"city": "Тестовый город", "country": "Россия", "country_code": "ru"}}
                candidates.append(asdict(CityCandidate(f"osm-node-{identity}", item.title, item.place_query, item.area, record)))
        self.itinerary.program_report = None
        rebuild_program_report(self.itinerary, self.request, eligible_unique_visits=len(candidates),
                               allow_city_limited_program=True)
        self.payloads = {
            "request.json": self.request.model_dump(), "final-itinerary.json": self.itinerary.model_dump(),
            "city-candidates.json": {"candidates": candidates, "problems": [], "map_requests": 0, "usage": {}, "rejected": []},
            "report.json": {"status": "succeeded", "pipeline": "city_candidates_first", "problems": [],
                            "private_memory_used": False, "trip_saved": False, "quota_finished": True,
                            "resolved_city": self.center, "weather": None, "weather_text": ""},
        }
        self.write_inputs()

    def write_inputs(self):
        for name, value in self.payloads.items():
            (self.source / name).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def run_check(self, **kwargs):
        return acceptance.run_acceptance(self.source, self.root / "output", **kwargs)

    def test_real_temp_storage_reload_owner_invalid_context_refusal_pdf_and_report(self):
        before = {path.name: path.read_bytes() for path in self.source.iterdir()}
        result = self.run_check()
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual((result["requested_days"], result["program_days"], result["free_days"]), (4, 2, 2))
        self.assertEqual((result["validated_candidates"], result["selected_visits"]), (4, 4))
        self.assertTrue(result["owner_isolation"] and result["reload_unchanged"]
                        and result["invalid_context_refused_before_quota"])
        self.assertEqual((result["ai_calls"], result["quota_operations"], result["network_calls"], result["guard_violations"]), (0, 0, 0, 0))
        self.assertTrue(result["edit_available"])
        self.assertFalse(result["user_trip_saved"] or result["browser_acceptance"]
                         or result["edit_acceptance"] or result["product_fully_accepted"])
        self.assertIn("city_edit_not_tested", result["limitations"])
        self.assertNotIn("city_edit_not_supported", result["limitations"])
        self.assertNotIn("edit_refused_before_quota", result)
        self.assertTrue((self.root / "output/trip.pdf").read_bytes().startswith(b"%PDF-"))
        self.assertEqual(result["input_sha256"], {name: hashlib.sha256(raw).hexdigest() for name, raw in before.items()})
        self.assertEqual({path.name: path.read_bytes() for path in self.source.iterdir()}, before)
        public = (self.root / "output/acceptance-report.json").read_text(encoding="utf-8")
        for private in (self.request.destination, self.itinerary.summary, "Учебный музей", str(self.source)):
            self.assertNotIn(private, public)
        self.assertEqual(set(path.name for path in (self.root / "output").iterdir()), {"trip.pdf", "acceptance-report.json"})

    def test_valid_private_context_roundtrips_and_only_damaged_copy_is_probed_and_deleted(self):
        import city_context
        import db
        import web_service
        from schemas import Itinerary

        original_save, original_probe = db.save_trip_bundle, db.save_trip
        original_edit, original_delete = web_service.edit_trip_day, db.delete_trip
        saved, probes, deleted = [], [], []

        def save(function, *args, **kwargs):
            trip_id = function(*args, **kwargs)
            saved.append((trip_id, deepcopy(db.get_trip(trip_id, acceptance.OWNER))))
            return trip_id

        def edit(owner, trip_id, day, instruction):
            if owner == acceptance.OWNER:
                probe = db.get_trip(trip_id, owner)
                probes.append((trip_id, day, probe["request"][city_context.CONTEXT_KEY]["version"]))
            return original_edit(owner, trip_id, day, instruction)

        def delete(trip_id, owner):
            before = next(value for identifier, value in saved if identifier == trip_id)
            self.assertEqual(db.get_trip(trip_id, owner), before)
            deleted.append((trip_id, owner))
            return original_delete(trip_id, owner)

        with patch.object(db, "save_trip_bundle", side_effect=lambda *a, **kw: save(original_save, *a, **kw)), \
                patch.object(db, "save_trip", side_effect=lambda *a, **kw: save(original_probe, *a, **kw)), \
                patch.object(web_service, "edit_trip_day", side_effect=edit), \
                patch.object(db, "delete_trip", side_effect=delete):
            result = self.run_check()
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(len(saved), 2)
        valid_id, valid = saved[0]
        probe_id, damaged = saved[1]
        self.assertNotEqual(valid_id, probe_id)
        self.assertEqual(valid["request"][city_context.CONTEXT_KEY]["version"], 1)
        candidates, center = city_context.load_city_context(valid["request"][city_context.CONTEXT_KEY], self.request)
        city_context.check_context_places(Itinerary.model_validate(valid["itinerary"]), candidates)
        self.assertEqual(len(candidates), 4)
        self.assertEqual(center["name"], self.center["name"])
        expected_damaged = deepcopy(valid["request"])
        expected_damaged[city_context.CONTEXT_KEY]["version"] = 0
        self.assertEqual(damaged["request"], expected_damaged)
        self.assertEqual(probes, [(probe_id, 1, 0), (probe_id, 2, 0)])
        self.assertEqual(deleted, [(probe_id, acceptance.OWNER)])

    def test_bypassed_context_refusal_cannot_reach_quota_or_pass_acceptance(self):
        # Even if preflight regresses, the tool must block quota before writing it.
        with patch("city_context.prepare_stored_city_context", return_value={}):
            result = self.run_check()
        self.assertEqual((result["status"], result["guard_violations"]), ("failed", 1))
        self.assertFalse(result["edit_acceptance"])
        self.assertNotIn("invalid_context_refused_before_quota", result)
        self.assertFalse((self.root / "output/trip.pdf").exists())

    def test_success_instead_of_invalid_context_refusal_is_not_accepted(self):
        import web_service

        original_edit = web_service.edit_trip_day

        def wrong_edit(owner, *args):
            if owner == acceptance.OWNER:
                return {"edit": {"accepted": True}}
            return original_edit(owner, *args)

        with patch.object(web_service, "edit_trip_day", side_effect=wrong_edit):
            result = self.run_check()
        self.assertEqual((result["status"], result["failure_stage"]), ("rejected", "invalid_context_refusal"))
        self.assertEqual(result["guard_violations"], 0)
        self.assertFalse(result["edit_acceptance"])

    def test_missing_center_rejects_without_lookup_or_storage(self):
        self.payloads["report.json"].pop("resolved_city")
        self.write_inputs()
        with patch.object(acceptance, "_stored_acceptance", side_effect=AssertionError("must not store")):
            result = self.run_check()
        self.assertEqual(result["failure_stage"], "resolved_city_coordinates")
        self.assertFalse((self.root / "output/trip.pdf").exists())

    def test_explicit_saved_center_and_label_copy_work_without_mutating_inputs(self):
        self.payloads["report.json"].pop("resolved_city")
        self.write_inputs()
        center_file = self.root / "center.json"
        center_file.write_text(json.dumps(self.center), encoding="utf-8")
        before = (self.source / "final-itinerary.json").read_bytes()
        result = self.run_check(center_json=center_file, label_estimates=True)
        self.assertEqual(result["status"], "passed", result)
        self.assertTrue(result["display_estimates_relabelled"])
        self.assertEqual((self.source / "final-itinerary.json").read_bytes(), before)

    def test_rejected_diagnostic_cannot_be_promoted(self):
        self.payloads["report.json"]["status"] = "rejected"
        self.write_inputs()
        result = self.run_check()
        self.assertEqual((result["status"], result["failure_stage"]), ("rejected", "accepted_diagnostic"))

    def test_forged_record_or_selected_identity_fails_closed(self):
        self.payloads["city-candidates.json"]["candidates"][0]["record"]["provenance"] = "model"
        self.write_inputs()
        result = self.run_check()
        self.assertEqual(result["failure_stage"], "candidate_canonical_record")

    def test_wrong_country_center_cannot_authorize_radius(self):
        self.payloads["report.json"]["resolved_city"]["country_code"] = "TR"
        self.write_inputs()
        result = self.run_check()
        self.assertEqual(result["failure_stage"], "resolved_city_country")

    def test_duplicate_visits_are_rejected_even_with_distinct_saved_titles(self):
        first = deepcopy(self.payloads["final-itinerary.json"]["days"][0]["items"][0])
        self.payloads["final-itinerary.json"]["days"][2]["items"][0] = first
        self.write_inputs()
        result = self.run_check()
        self.assertEqual(result["status"], "rejected")
        self.assertIn(result["failure_stage"], {"rules", "duplicates"})

    def test_budget_validation_runs_before_storage(self):
        self.payloads["request.json"]["budget"] = 25001
        self.write_inputs()
        result = self.run_check()
        self.assertEqual(result["failure_stage"], "rules")

    def test_visit_planning_cap_is_rechecked_before_storage(self):
        # The morning synthetic museum has plenty of free time afterwards:
        # this must fail the shared product ceiling, not overlap or depth.
        self.payloads["final-itinerary.json"]["days"][0]["items"][0]["duration_min"] = 121
        self.write_inputs()
        with patch.object(acceptance, "_stored_acceptance", side_effect=AssertionError("must not store")):
            result = self.run_check()
        self.assertEqual((result["status"], result["failure_stage"]), ("rejected", "city_visit_estimates"))
        self.assertFalse((self.root / "output/trip.pdf").exists())

    def test_null_rain_is_not_treated_as_dry(self):
        self.payloads["report.json"]["weather"] = {
            "city": self.center["name"], "source": "forecast", "days": [
                {"date": day.date, "tmin": 5, "tmax": 12, "rain_mm": None} for day in self.itinerary.days],
        }
        self.write_inputs()
        result = self.run_check()
        self.assertEqual(result["failure_stage"], "weather_values")

    def test_unknown_source_basis_does_not_authorize_city_free_days(self):
        self.payloads["final-itinerary.json"]["program_report"]["coverage_basis"] = "source"
        self.write_inputs()
        result = self.run_check()
        self.assertEqual(result["failure_stage"], "metadata")

    def test_output_must_be_new_and_outside_original_artifacts(self):
        with self.assertRaises(acceptance.AcceptanceFailure):
            acceptance.run_acceptance(self.source, self.source / "new-output")
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            acceptance.run_acceptance(self.source, existing)

    def test_guard_blocks_wrong_database_before_creating_it(self):
        forbidden = self.root / "working-database-must-not-exist.sqlite3"
        with patch.object(acceptance, "_stored_acceptance", side_effect=lambda *args: sqlite3.connect(forbidden)):
            result = self.run_check()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["guard_violations"], 1)
        self.assertFalse(forbidden.exists())
        self.assertFalse((self.root / "output/trip.pdf").exists())

    def test_guard_denies_dns_network_and_outside_writes_without_real_operations(self):
        from tools.run_offline_checks import OfflineGuardError
        scratch, output = self.root / "scratch", self.root / "guard-output"
        scratch.mkdir()
        output.mkdir()
        guard = acceptance._Guard(scratch, output)
        guard.active = True
        for event, args in (("socket.getaddrinfo", ("example.invalid", 443)),
                            ("socket.connect", (None, ("192.0.2.1", 443))),
                            ("os.mkdir", (str(self.source / "forbidden"), 511, -1))):
            with self.subTest(event=event), self.assertRaises(OfflineGuardError):
                guard.audit(event, args)
        self.assertEqual(len(guard.violations), 3)

    def test_explicit_label_cannot_change_substantive_schedule(self):
        def unsafe_label(itinerary, request):
            itinerary.days[0].items[0].cost = 0
        with patch("city_estimates.label_city_estimates", side_effect=unsafe_label):
            result = self.run_check(label_estimates=True)
        self.assertEqual(result["failure_stage"], "estimate_label_changed_program")


if __name__ == "__main__":
    unittest.main()
