"""Portable runner configuration without launching the application or network."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import run_offline_checks as runner


class OfflineRunnerConfigTests(unittest.TestCase):
    def parser_without_node(self):
        with patch.object(runner.shutil, "which", return_value=None):
            return runner.argument_parser()

    def test_defaults_use_current_python_and_no_private_bundle(self):
        parser = self.parser_without_node()
        args = parser.parse_args([])
        self.assertEqual(args.python, Path(sys.executable))
        self.assertIsNone(args.node)
        self.assertIsNone(args.bundled_sitepackages)

    def test_node_is_discovered_from_path(self):
        executable = str(Path(sys.executable).with_name("node"))
        with patch.object(runner.shutil, "which", return_value=executable) as find:
            args = runner.argument_parser().parse_args([])
        find.assert_called_once_with("node")
        self.assertEqual(args.node, Path(executable))

    def test_python_only_categories_do_not_require_node_or_bundled_pdf(self):
        for category in ("guard", "unit", "smoke"):
            with self.subTest(category=category):
                parser = self.parser_without_node()
                args = parser.parse_args(["--only", category])
                runner.validate_runtime_options(parser, args)

    def test_all_and_node_reject_missing_node(self):
        for category in ("all", "node"):
            with self.subTest(category=category), redirect_stderr(io.StringIO()):
                parser = self.parser_without_node()
                args = parser.parse_args(["--only", category])
                with self.assertRaises(SystemExit) as error:
                    runner.validate_runtime_options(parser, args)
                self.assertEqual(error.exception.code, 2)

    def test_explicit_runtime_and_optional_bundle_paths_are_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            parser = self.parser_without_node()
            args = parser.parse_args(["--python", sys.executable, "--node", sys.executable,
                                      "--bundled-sitepackages", directory])
            runner.validate_runtime_options(parser, args)
            self.assertEqual(args.bundled_sitepackages, Path(directory))

    def test_missing_explicit_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            parser = self.parser_without_node()
            args = parser.parse_args(["--only", "guard", "--bundled-sitepackages",
                                      str(Path(directory) / "absent")])
            with self.assertRaises(SystemExit):
                runner.validate_runtime_options(parser, args)

    def test_nonpositive_timeout_is_rejected(self):
        for timeout in ("0", "-1"):
            with self.subTest(timeout=timeout), redirect_stderr(io.StringIO()):
                parser = self.parser_without_node()
                args = parser.parse_args(["--only", "guard", "--timeout", timeout])
                with self.assertRaises(SystemExit):
                    runner.validate_runtime_options(parser, args)

    def test_child_environment_has_no_implicit_bundle(self):
        root = Path(tempfile.gettempdir()) / "offline-config-test"
        environment = runner.child_environment(root, root / "source", root / "bootstrap", None)
        self.assertNotIn("OFFLINE_CHECK_BUNDLED", environment)
        self.assertEqual(environment["PYTHONPATH"], str(root / "bootstrap"))

    def test_child_environment_passes_only_explicit_optional_bundle(self):
        root = Path(tempfile.gettempdir()) / "offline-config-test"
        bundle = root / "optional-packages"
        environment = runner.child_environment(root, root / "source", root / "bootstrap", bundle)
        self.assertEqual(environment["OFFLINE_CHECK_BUNDLED"], str(bundle))

    def test_child_environment_does_not_inherit_application_values(self):
        root = Path(tempfile.gettempdir()) / "offline-config-test"
        with patch.dict(runner.os.environ, {"OFFLINE_TEST_UNTRUSTED": "synthetic"}):
            environment = runner.child_environment(root, root / "source", root / "bootstrap", None)
        self.assertNotIn("OFFLINE_TEST_UNTRUSTED", environment)


if __name__ == "__main__":
    unittest.main()
