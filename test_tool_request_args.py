"""Diagnostic CLIs require explicit input before file, account or network access."""

from contextlib import ExitStack, redirect_stderr
import io
from pathlib import Path
import socket
import sqlite3
import sys
import unittest
from unittest.mock import patch

import dotenv
import httpx

from tools import diagnose_generation, replay_grounding, replay_source_plan


class ExplicitRequestArgumentTests(unittest.TestCase):
    def assert_request_required(self, module, arguments):
        error_output = io.StringIO()
        with ExitStack() as stack:
            blocked = []
            for owner, name in (
                (Path, "read_text"), (Path, "mkdir"), (sqlite3, "connect"),
                (socket.socket, "connect"), (httpx, "get"), (httpx, "post"),
                (dotenv, "load_dotenv"),
            ):
                blocked.append(stack.enter_context(patch.object(
                    owner, name, side_effect=AssertionError("Input must be explicit before side effects"),
                )))
            stack.enter_context(patch.object(sys, "argv", [module.__name__, *arguments]))
            stack.enter_context(redirect_stderr(error_output))
            with self.assertRaises(SystemExit) as stopped:
                module.main()
            self.assertEqual(stopped.exception.code, 2)
            self.assertIn("--request-file", error_output.getvalue())
            for operation in blocked:
                operation.assert_not_called()

    def test_diagnostic_requires_request_before_paid_run_or_account_access(self):
        self.assert_request_required(diagnose_generation, [
            "--source-trip", "1", "--output-dir", "output/not-created", "--run-once",
        ])

    def test_candidate_replay_requires_request_before_loading_saved_proposals(self):
        self.assert_request_required(replay_grounding, [
            "--raw-file", "not-read-candidates.json", "--output-file", "output/not-created.json",
            "--cached-only",
        ])

    def test_source_replay_requires_request_before_loading_saved_grounding(self):
        self.assert_request_required(replay_source_plan, [
            "--grounding-file", "not-read-grounding.json", "--output-dir", "output/not-created",
            "--cached-only",
        ])


if __name__ == "__main__":
    unittest.main()
