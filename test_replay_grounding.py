"""Saved-candidate replay CLI: temporary files only, no AI, DB or network."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import socket
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

from candidate_grounding import GroundingResult
from tools import replay_grounding as replay


class ReplayGroundingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.output = self.root / "output" / "fresh" / "report.json"
        self.request_file = self.root / "request.json"
        self.raw_file = self.root / "candidates.json"
        self.request_file.write_text(json.dumps({
            "destination": "Тестовый край, Россия", "destination_kind": "region",
            "start_date": "2099-06-01", "end_date": "2099-06-01", "budget": 10000,
        }, ensure_ascii=False), encoding="utf-8")
        self.raw = '{"bases": [], "pois": []}'
        self.raw_file.write_text(self.raw, encoding="utf-8")
        self.mock(replay, "ROOT", new=self.root)
        self.mock(Path, "cwd", return_value=self.root)
        self.mock(socket.socket, "connect", side_effect=AssertionError("External network forbidden"))
        self.mock(sqlite3, "connect", side_effect=AssertionError("All database access forbidden"))
        self.get = self.mock(replay.places.httpx, "get", side_effect=AssertionError("Unmocked GET forbidden"))
        self.post = self.mock(replay.places.httpx, "post", side_effect=AssertionError("Unmocked POST forbidden"))
        self.scope = object()
        self.resolve = self.mock(replay, "resolve_region", return_value=self.scope)
        self.hints = self.mock(replay.places, "cached_place_hints", return_value=[])
        self.result = GroundingResult()
        self.ground = self.mock(replay.candidate_grounding, "ground_region_candidates", return_value=self.result)
        self.generate = self.mock(replay.candidate_grounding, "generate_json", side_effect=AssertionError("AI forbidden"))

    def mock(self, owner, name, **kwargs):
        patcher = patch.object(owner, name, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def run_cli(self, *, cached_only=True, output=None):
        argv = ["replay_grounding.py", "--raw-file", str(self.raw_file),
                "--request-file", str(self.request_file), "--output-file", str(output or self.output)]
        if cached_only:
            argv.append("--cached-only")
        with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
            return replay.main()

    def assert_http_restored(self):
        self.assertIs(replay.places.httpx.get, self.get)
        self.assertIs(replay.places.httpx.post, self.post)
        self.get.assert_not_called()
        self.post.assert_not_called()
        self.generate.assert_not_called()

    def test_cached_only_passes_zero_http_budget_and_saved_raw_instead_of_ai(self):
        self.assertEqual(self.run_cli(), 0)
        self.ground.assert_called_once()
        args, kwargs = self.ground.call_args
        self.assertEqual(args[0].destination, "Тестовый край, Россия")
        self.assertIs(args[1], self.scope)
        self.assertEqual(kwargs["max_http_attempts"], 0)
        self.assertEqual(kwargs["known_hints"], [])
        completion = kwargs["completion_fn"]([], json_schema={})
        self.assertEqual(completion.text, self.raw)
        self.assertEqual((completion.input_tokens, completion.output_tokens), (0, 0))
        self.hints.assert_called_once_with("Тестовый край, Россия", None, region_scope=self.scope)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual((report["mode"], report["paid_requests"], report["cached_only"]),
                         ("saved_candidate_replay", 0, True))
        self.assertEqual(report["source_raw"], str(self.raw_file))
        self.assertEqual(report["result"]["confirmed_hints"], [])
        self.assert_http_restored()

    def test_cached_only_blocks_get_during_region_resolution_and_grounding(self):
        for stage in ("resolve", "ground"):
            with self.subTest(stage=stage):
                self.resolve.side_effect = None
                self.ground.side_effect = None
                target = self.resolve if stage == "resolve" else self.ground
                target.side_effect = lambda *a, **kw: replay.places.httpx.get("https://example.invalid")
                with self.assertRaisesRegex(AssertionError, "HTTP forbidden in cached-only replay"):
                    self.run_cli()
                self.assertFalse(self.output.exists())
                self.assert_http_restored()

    def test_post_is_blocked_in_both_modes_and_restored_on_failure(self):
        self.ground.side_effect = lambda *a, **kw: replay.places.httpx.post("https://example.invalid")
        for cached in (False, True):
            with self.subTest(cached=cached):
                with self.assertRaisesRegex(AssertionError, "Paid requests forbidden during replay"):
                    self.run_cli(cached_only=cached)
                self.assertFalse(self.output.exists())
                self.assert_http_restored()

    def test_non_cached_mode_uses_only_candidate_stage_budget_and_still_injects_raw(self):
        self.assertEqual(self.run_cli(cached_only=False), 0)
        kwargs = self.ground.call_args.kwargs
        self.assertEqual(kwargs["max_http_attempts"], replay.candidate_grounding.MAX_HTTP_ATTEMPTS)
        self.assertEqual(kwargs["completion_fn"]([]).text, self.raw)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertFalse(report["cached_only"])
        self.assertEqual(report["paid_requests"], 0)
        self.assert_http_restored()

    def test_existing_output_is_never_overwritten_even_before_input_read(self):
        self.output.parent.mkdir(parents=True)
        self.output.write_text("existing result", encoding="utf-8")
        with patch.object(Path, "read_text", side_effect=AssertionError("Must reject before reading input")):
            with self.assertRaisesRegex(SystemExit, "Choose a new file within output"):
                self.run_cli()
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing result")
        self.resolve.assert_not_called()
        self.ground.assert_not_called()
        self.assert_http_restored()

    def test_output_must_resolve_within_exact_output_root(self):
        for output in (self.root / "escaped.json", self.root / "output-sibling" / "x.json",
                       self.root / "output" / ".." / "escaped.json"):
            with self.subTest(output=output):
                with patch.object(Path, "read_text", side_effect=AssertionError("Must reject before reading input")):
                    with self.assertRaisesRegex(SystemExit, "Choose a new file within output"):
                        self.run_cli(output=output)
                self.assertFalse(output.exists())
        self.resolve.assert_not_called()
        self.ground.assert_not_called()
        self.assert_http_restored()

    def test_wrong_working_directory_is_rejected_before_input_or_map(self):
        with patch.object(Path, "cwd", return_value=self.root.parent), \
             patch.object(Path, "read_text", side_effect=AssertionError("Must reject before reading input")):
            with self.assertRaisesRegex(SystemExit, "Run from the project directory"):
                self.run_cli()
        self.assertFalse(self.output.exists())
        self.resolve.assert_not_called()
        self.ground.assert_not_called()
        self.assert_http_restored()

    def test_malformed_saved_json_stops_before_any_map_or_completion(self):
        self.raw_file.write_text("not JSON", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            self.run_cli()
        self.assertFalse(self.output.exists())
        self.resolve.assert_not_called()
        self.ground.assert_not_called()
        self.assert_http_restored()

    def test_failed_replay_writes_diagnostic_but_returns_failure(self):
        self.result.problems = ["Недостаточно подтверждённых мест"]
        self.result.incomplete = True
        self.assertEqual(self.run_cli(), 1)
        report = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(report["result"]["problems"], self.result.problems)
        self.assertTrue(report["result"]["incomplete"])
        self.assertEqual(report["paid_requests"], 0)
        self.assert_http_restored()


if __name__ == "__main__":
    unittest.main()
