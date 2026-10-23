"""Read-only preview preparation, including a cold process without credentials."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import preview_source_plan as preview
from test_program_coverage import _request, _route


class SourcePreviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="source-preview-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "output"
        self.output.mkdir()
        self.path = self.output / "report.json"
        request, itinerary = _request(), _route()
        self.report = {
            "request": request.model_dump(), "itinerary": itinerary.model_dump(),
            "route_problems": [], "validation_problems": [],
            "skeleton": {"status": "limited_program", "days": [
                {"day": day.day, "depth_credit_min": 0} for day in itinerary.days
            ]},
        }
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.report, ensure_ascii=False), encoding="utf-8")

    def test_invalid_or_outside_path_is_rejected_before_application_import(self):
        outside = self.root / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        import builtins
        actual_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name in {"web_service", "memory", "rules", "schemas"}:
                raise AssertionError("Application imported before path validation")
            return actual_import(name, *args, **kwargs)

        with patch.object(preview, "ROOT", self.root), patch("builtins.__import__", side_effect=checked_import):
            for path in (outside, self.output / "missing.json", self.output):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "inside output"):
                    preview.preview_data(path)

    def test_cold_process_never_reads_env_opens_database_or_connects(self):
        script = r'''
import json, os, sys
from pathlib import Path
violations = []
def deny(event, args):
    if event == 'open' and not isinstance(args[0], int) and os.path.basename(os.fsdecode(args[0])).lower() == '.env':
        violations.append('.env')
        raise AssertionError('Credential read forbidden')
    if event in ('sqlite3.connect', 'socket.connect', 'socket.getaddrinfo', 'subprocess.Popen'):
        violations.append(event)
        raise AssertionError('External side effect forbidden')
sys.addaudithook(deny)
import dotenv
assert 'web_service' not in sys.modules, 'The preview check must start cold'
def forbidden_dotenv(*args, **kwargs):
    violations.append('dotenv import call')
    raise AssertionError('Credential loading forbidden even without an .env file')
# The general offline runner suppresses dotenv itself. Replace that suppression
# with a sentinel so this regression still tests the preview's own boundary.
dotenv.load_dotenv = forbidden_dotenv
from tools import preview_source_plan as preview
preview.ROOT = Path(sys.argv[1])
trip, detail, summary = preview.preview_data(Path(sys.argv[2]))
assert not violations, violations
assert trip['id'] == detail['id'] == summary['id'] == 1
assert detail['itinerary']['program_report']['free_days'] == 4
print(json.dumps({'ok': True, 'violations': len(violations)}))
'''
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", "-c", script, str(self.root), str(self.path)],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {"ok": True, "violations": 0})

    def test_failed_replay_is_not_rendered_as_success(self):
        for key, value in (("route_problems", ["not verified"]), ("validation_problems", ["over budget"])):
            with self.subTest(key=key):
                original = self.report[key]
                self.report[key] = value
                self.save()
                with patch.object(preview, "ROOT", self.root), self.assertRaisesRegex(ValueError, "passed both"):
                    preview.preview_data(self.path)
                self.report[key] = original

    def test_rules_are_rechecked_without_rewriting_input(self):
        self.report["itinerary"]["days"][3]["items"][0]["cost"] = 0
        self.save()
        before = self.path.read_bytes()
        with patch.object(preview, "ROOT", self.root), self.assertRaisesRegex(ValueError, "local rules"):
            preview.preview_data(self.path)
        self.assertEqual(self.path.read_bytes(), before)

    def test_valid_preview_uses_only_fixture_storage_and_canonical_disclosure(self):
        before = self.path.read_bytes()
        with patch.object(preview, "ROOT", self.root), patch("sqlite3.connect", side_effect=AssertionError("No DB")):
            trip, detail, summary = preview.preview_data(self.path)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(trip["id"], summary["id"])
        self.assertEqual(detail["itinerary"]["program_report"]["free_days"], 4)
        self.assertIn("Дней свободного режима: 4", detail["itinerary"]["program_report_text"])
        self.assertIn("напоминания не создаются", detail["reminders_text"])


if __name__ == "__main__":
    unittest.main()
