r"""Run offline acceptance checks on a disposable source snapshot, never the app DB.

Usage: python -X utf8 -B tools/run_offline_checks.py
Install requirements-dev.txt in the active Python environment first. Node.js 24+
must be on PATH for the full run (or pass --node with its explicit path).
The default run includes unittest discover, Node UI, 11 documented smoke scripts,
and the standalone regional-request checks. No app server or paid API is started.
This is a regression-safety harness for trusted local tests, not a hostile-code sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SMOKE_SCRIPTS = (
    "try_planner_edit.py", "try_planner_memory.py", "try_route_preferences.py",
    "try_web.py", "try_db.py", "try_db_edit.py", "try_ai_limit.py",
    "try_reminders.py", "try_bot_edit.py", "try_bot_memory.py",
    "try_bot_reminders.py",
)
RESULT_PREFIX = "OFFLINE_RESULT "
_GUARD = None


class OfflineGuardError(RuntimeError):
    """A real external side effect was attempted by an offline check."""


class OfflineGuard:
    def __init__(self, temporary_root: Path):
        self.root = self.path(temporary_root)
        self.violations: list[str] = []
        self.local = threading.local()
        self.socketpair_calls = 0

    @staticmethod
    def path(value):
        return os.path.normcase(os.path.realpath(os.path.abspath(os.fsdecode(value))))

    def deny(self, operation):
        # Never include URLs, environment values, request bodies or file contents.
        self.violations.append(operation)
        raise OfflineGuardError("Offline guard blocked " + operation)

    def inside(self, path):
        candidate = self.path(path)
        try:
            return os.path.commonpath((self.root, candidate)) == self.root
        except ValueError:  # A different Windows drive is outside the sandbox.
            return False

    def file(self, path, *, write=False):
        if isinstance(path, int) or path is None:
            return
        if os.path.basename(os.fsdecode(path)).lower() == ".env":
            self.deny(".env access")
        if write and not self.inside(path):
            self.deny("write outside temporary workspace")

    def audit(self, event, args):
        if event == "open":
            path, mode, flags = args
            writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            self.file(path, write=writing)
        elif event == "sqlite3.connect":
            database = os.fsdecode(args[0])
            if database != ":memory:":
                # URI filenames can hide a second path; tests use ordinary paths.
                if database.startswith("file:"):
                    self.deny("SQLite URI connection")
                self.file(database, write=True)
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.utime", "os.truncate"}:
            self.file(args[0], write=True)
        elif event in {"os.rename", "os.link", "os.symlink"}:
            self.file(args[0], write=True)
            self.file(args[1], write=True)
        elif event in {"socket.connect", "socket.bind"}:
            address = args[1]
            loopback = isinstance(address, tuple) and address[0] in {"127.0.0.1", "::1"}
            # urllib3 probes IPv6 by binding localhost:0 without listening or
            # connecting. Listen/connect remain denied outside socketpair.
            probe = event == "socket.bind" and loopback and address[1] == 0
            if not (probe or (getattr(self.local, "socketpair", False) and loopback)):
                self.deny(event)
        elif event.startswith("socket.get") or event == "socket.sendto":
            self.deny(event)
        elif event in {"os.system", "os.posix_spawn", "os.spawn"}:
            self.deny("uncontrolled child process")
        elif event == "subprocess.Popen":
            if not getattr(self.local, "approved_child", False):
                self.deny("unguarded child process")

    def install(self):
        import socket

        original_pair = socket.socketpair
        original_listen = socket.socket.listen
        original_popen = subprocess.Popen
        guard = self

        def internal_listen(sock, *args, **kwargs):
            if not getattr(self.local, "socketpair", False):
                self.deny("socket.listen")
            return original_listen(sock, *args, **kwargs)

        class GuardedPopen(original_popen):
            def __init__(process, argv, *args, **kwargs):
                environment = kwargs.get("env")
                environment = os.environ if environment is None else environment
                if (not isinstance(argv, (list, tuple)) or not argv
                        or guard.path(argv[0]) != guard.path(sys.executable)
                        or "-B" not in argv or any(flag in argv for flag in ("-I", "-E", "-S"))
                        or kwargs.get("shell") or kwargs.get("executable")
                        or environment.get("OFFLINE_CHECK_ROOT") != guard.root
                        or environment.get("PYTHONPATH") != os.environ.get("PYTHONPATH")
                        or not guard.inside(kwargs.get("cwd") or os.getcwd())):
                    guard.deny("unguarded child process")
                guard.local.approved_child = True
                try:
                    super().__init__(argv, *args, **kwargs)
                finally:
                    guard.local.approved_child = False

        def internal_socketpair(*args, **kwargs):
            # asyncio's Windows wakeup pipe is a local socket pair, not HTTP.
            self.local.socketpair = True
            try:
                result = original_pair(*args, **kwargs)
                self.socketpair_calls += 1
                return result
            finally:
                self.local.socketpair = False

        socket.socketpair = internal_socketpair
        socket.socket.listen = internal_listen
        subprocess.Popen = GuardedPopen
        sys.addaudithook(self.audit)
        import dotenv
        import httpx

        # Tests can still mock public calls and use ASGI/TestClient transports.
        dotenv.load_dotenv = lambda *args, **kwargs: False
        httpx.HTTPTransport.handle_request = lambda *a, **kw: self.deny("real HTTP transport")

        async def deny_async(*args, **kwargs):
            self.deny("real async HTTP transport")

        httpx.AsyncHTTPTransport.handle_async_request = deny_async
        if importlib.util.find_spec("requests"):
            import requests
            requests.sessions.Session.send = lambda *a, **kw: self.deny("real requests transport")


def install_from_environment():
    """sitecustomize entry point, inherited by legacy subprocess checks."""
    global _GUARD
    if _GUARD is not None:
        return
    sys.dont_write_bytecode = True
    bundled = os.environ.get("OFFLINE_CHECK_BUNDLED")
    if bundled and bundled not in sys.path:
        sys.path.append(bundled)  # venv packages must retain priority.
    _GUARD = OfflineGuard(Path(os.environ["OFFLINE_CHECK_ROOT"]))
    tempfile.tempdir = os.environ["TEMP"]
    _GUARD.install()


def guard_self_check():
    import socket
    import sqlite3
    import httpx

    outside = Path(_GUARD.root).parent / "offline-guard-must-not-create.txt"
    def blocked_connect():
        with socket.socket() as connection:
            connection.connect(("127.0.0.1", 1))

    def blocked_listen():
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            server.listen()

    attempts = [
        lambda: socket.getaddrinfo("example.invalid", 443),
        blocked_connect,
        blocked_listen,
        lambda: httpx.get("https://example.invalid"),
        lambda: outside.write_text("forbidden", encoding="utf-8"),
        lambda: sqlite3.connect(outside.with_suffix(".db")),
        lambda: (Path(_GUARD.root) / ".env").read_text(encoding="utf-8"),
        lambda: subprocess.Popen([sys.executable, "-I", "-c", "pass"]),
    ]
    if importlib.util.find_spec("requests"):
        import requests
        attempts.append(lambda: requests.get("https://example.invalid"))
    for attempt in attempts:
        try:
            attempt()
        except OfflineGuardError:
            continue
        raise AssertionError("An offline guard probe was not blocked")
    assert len(_GUARD.violations) == len(attempts), _GUARD.violations
    _GUARD.violations.clear()
    with sqlite3.connect(Path(_GUARD.root) / "guard-test.db") as connection:
        connection.execute("CREATE TABLE synthetic(value INTEGER)")
    first, second = socket.socketpair()
    first.close()
    second.close()
    child = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c",
                            "from tools.run_offline_checks import _GUARD; "
                            "assert _GUARD is not None; print('guard inherited')"],
                           capture_output=True, text=True, timeout=20)
    assert child.returncode == 0 and child.stdout.strip() == "guard inherited", "Child guard missing"
    assert not _GUARD.violations
    return {"checks": len(attempts) + 3, "passed": len(attempts) + 3}


def python_worker(kind, name=None):
    # Import the same module as sitecustomize; running this file uses __main__.
    from tools import run_offline_checks as harness
    if harness._GUARD is None:
        raise RuntimeError("Required startup guard did not install")
    guard = harness._GUARD
    if kind == "guard":
        result = harness.guard_self_check()
        result.update(category="guard", ok=True)
    elif kind == "unit":
        import pypdf  # Missing PDF dependency is fatal, never a skipped PDF check.
        suite = unittest.defaultTestLoader.discover(str(ROOT), pattern="test_*.py", top_level_dir=str(ROOT))
        def ids(tests):
            for test in tests:
                if isinstance(test, unittest.TestSuite):
                    yield from ids(test)
                else:
                    yield test.id()
        unique = list(ids(suite))
        if len(unique) != len(set(unique)):
            raise AssertionError("Duplicate unittest identifiers")
        run = unittest.TextTestRunner(verbosity=1).run(suite)
        result = {"category": "unit", "tests": run.testsRun, "unique": len(set(unique)),
                  "failures": len(run.failures), "errors": len(run.errors), "skipped": len(run.skipped),
                  "expected_failures": len(run.expectedFailures), "unexpected_successes": len(run.unexpectedSuccesses),
                  "passed": run.testsRun - len(run.failures) - len(run.errors) - len(run.skipped),
                  "ok": bool(unique) and run.testsRun == len(unique) and run.wasSuccessful()
                  and not run.skipped and not run.expectedFailures}
        if run.failures or run.errors:
            result["failed_tests"] = [test.id() for test, _ in run.failures + run.errors]
    else:
        allowed = set(SMOKE_SCRIPTS) | {"test_region_request.py"}
        if name not in allowed:
            raise ValueError("Not an allowlisted offline script")
        from contextlib import redirect_stdout
        capture = io.StringIO()
        with redirect_stdout(capture):
            runpy.run_path(str(ROOT / name), run_name="__main__")
        output = capture.getvalue()
        print(output, end="")
        count = len(re.findall(r"(?m)^\[OK\]", output))
        result = {"category": kind, "script": name, "checks": count, "passed": count, "ok": count > 0}
    result["guard_violations"] = len(guard.violations)
    result["ok"] = result["ok"] and not guard.violations
    if guard.violations:
        result["blocked_operations"] = sorted(set(guard.violations))
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


BOOTSTRAP = '''import os, sys
sys.path.insert(0, os.environ["OFFLINE_CHECK_SOURCE"])
try:
    from tools.run_offline_checks import install_from_environment
    install_from_environment()
except BaseException:
    os.write(2, b"Offline safety bootstrap failed\\n")
    os._exit(92)
'''

NODE_GUARD = '''"use strict";
let violations = 0;
const deny = () => { violations++; throw new Error("Offline guard blocked Node external I/O"); };
process.on("exit", () => {
  if (violations) {
    process.stderr.write("Offline guard: " + violations + " blocked Node operations\\n");
    process.exitCode = 91;
  }
});
for (const [moduleName, names] of Object.entries({
  net: ["connect", "createConnection", "createServer"],
  http: ["request", "get", "createServer"], https: ["request", "get", "createServer"],
  tls: ["connect", "createServer"], dgram: ["createSocket"],
  dns: ["lookup", "resolve", "resolve4", "resolve6", "reverse"],
  child_process: ["spawn", "spawnSync", "exec", "execSync", "execFile", "execFileSync", "fork"]
})) {
  const module = require("node:" + moduleName);
  for (const name of names) module[name] = deny;
}
for (const name of Object.keys(require("node:dns").promises)) {
  if (typeof require("node:dns").promises[name] === "function") require("node:dns").promises[name] = deny;
}
require("node:net").Socket.prototype.connect = deny;
require("node:net").Server.prototype.listen = deny;
globalThis.fetch = deny;
'''


def digest(path):
    if not path.exists():
        return None
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def snapshot(destination):
    # Existing archive allowlist excludes secrets, DBs, caches and personal output.
    from tools import build_submission_archive as packaging
    manifest = {}
    for source in packaging.source_files():
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = packaging.checked_content(source)
        target.write_bytes(content)
        manifest[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    return manifest


def child_environment(temporary, source, bootstrap, bundled):
    # Do not inherit application credentials or proxy URLs from the host.
    inherited = ("SYSTEMROOT", "WINDIR", "PATH", "PATHEXT", "COMSPEC", "SYSTEMDRIVE",
                 "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "USERPROFILE")
    environment = {key: os.environ[key] for key in inherited if key in os.environ}
    environment.update({
        "TEMP": str(temporary / "tmp"), "TMP": str(temporary / "tmp"), "TMPDIR": str(temporary / "tmp"),
        "PYTHONPATH": str(bootstrap), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8",
        "OFFLINE_CHECK_ROOT": OfflineGuard.path(temporary), "OFFLINE_CHECK_SOURCE": str(source),
        "NO_COLOR": "1",
    })
    if bundled is not None:
        environment["OFFLINE_CHECK_BUNDLED"] = str(bundled)
    return environment


def execute(command, *, cwd, env, label, timeout):
    started = time.monotonic()
    try:
        process = subprocess.run(command, cwd=cwd, env=env, capture_output=True,
                                 text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        result = {"category": label, "ok": False, "error": "stage timeout"}
        print(json.dumps(result), flush=True)
        return result
    output = process.stdout + "\n" + process.stderr
    records = [line[len(RESULT_PREFIX):] for line in process.stdout.splitlines() if line.startswith(RESULT_PREFIX)]
    result = json.loads(records[-1]) if records else {"category": label, "ok": False}
    if "--script" in command:
        result.setdefault("script", command[command.index("--script") + 1])
    if label == "node":
        for field in ("tests", "pass", "fail", "skipped", "cancelled"):
            match = re.search(r"(?m)^# " + field + r" (\d+)\s*$", output)
            result[field] = int(match.group(1)) if match else None
        result["ok"] = (process.returncode == 0 and bool(result["tests"])
                        and result["fail"] == 0 and result["skipped"] == 0 and result["cancelled"] == 0)
    result["ok"] = result["ok"] and process.returncode == 0
    result["seconds"] = round(time.monotonic() - started, 2)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result["ok"]:
        # The child has no real credentials, DBs or .env. Keep diagnostics bounded.
        details = output.find("\n======================================================================")
        print(output[details:] if details >= 0 else output[-10000:], flush=True)
    return result


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    node = shutil.which("node")
    parser.add_argument("--python", type=Path, default=Path(sys.executable),
                        help="Python interpreter; defaults to the interpreter running this command")
    parser.add_argument("--node", type=Path, default=Path(node) if node else None,
                        help="Node.js 24+ executable; defaults to node on PATH (needed for all/node)")
    parser.add_argument("--bundled-sitepackages", type=Path, default=None,
                        help="Optional additional site-packages directory; installed environment packages retain priority")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout for each category/script")
    parser.add_argument("--only", choices=("all", "guard", "unit", "node", "smoke"), default="all")
    parser.add_argument("--worker", choices=("guard", "unit", "smoke", "standalone"), help=argparse.SUPPRESS)
    parser.add_argument("--script", help=argparse.SUPPRESS)
    return parser


def validate_runtime_options(parser, args):
    if not args.python.is_file():
        parser.error("Python runtime is missing; supply --python with its explicit executable path")
    if args.only in {"all", "node"} and (args.node is None or not args.node.is_file()):
        parser.error("Node.js 24+ is required for all/node checks; add node to PATH or supply --node")
    if args.bundled_sitepackages is not None and not args.bundled_sitepackages.is_dir():
        parser.error("Optional --bundled-sitepackages directory does not exist")
    if args.timeout <= 0:
        parser.error("--timeout must be a positive number of seconds")


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if args.worker:
        return python_worker(args.worker, args.script)
    validate_runtime_options(parser, args)
    before = digest(ROOT / "planner.db")
    results = []
    try:
        with tempfile.TemporaryDirectory(prefix="family-trip-offline-") as directory:
            temporary = Path(directory)
            source = temporary / "source"
            bootstrap = temporary / "bootstrap"
            bootstrap.mkdir()
            (temporary / "tmp").mkdir()
            manifest = snapshot(source)
            fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest()
            print(json.dumps({"category": "source_snapshot", "files": len(manifest), "sha256": fingerprint}), flush=True)
            (bootstrap / "sitecustomize.py").write_text(BOOTSTRAP, encoding="utf-8")
            (bootstrap / "node_guard.cjs").write_text(NODE_GUARD, encoding="utf-8")
            env = child_environment(temporary, source, bootstrap, args.bundled_sitepackages)
            base = [str(args.python.resolve()), "-X", "utf8", "-B", str(source / "tools/run_offline_checks.py")]
            guard = execute(base + ["--worker", "guard"], cwd=source, env=env, label="guard", timeout=args.timeout)
            results.append(guard)
            if not guard["ok"]:
                return 1
            if args.only in {"all", "unit"}:
                results.append(execute(base + ["--worker", "unit"], cwd=source, env=env, label="unit", timeout=args.timeout))
                results.append(execute(base + ["--worker", "standalone", "--script", "test_region_request.py"],
                                       cwd=source, env=env, label="standalone", timeout=args.timeout))
            if args.only in {"all", "node"}:
                # --test normally launches a child; isolation=none keeps the guard in the test process.
                node = [str(args.node.resolve()), "--permission", "--allow-fs-read=" + str(temporary),
                        "--require", str(bootstrap / "node_guard.cjs"),
                        "--test", "--experimental-test-isolation=none", "--test-reporter=tap", "test_web_ui.js"]
                results.append(execute(node, cwd=source, env=env, label="node", timeout=args.timeout))
            if args.only in {"all", "smoke"}:
                for script in SMOKE_SCRIPTS:
                    results.append(execute(base + ["--worker", "smoke", "--script", script],
                                           cwd=source, env=env, label="smoke", timeout=args.timeout))
    finally:
        after = digest(ROOT / "planner.db")
        print(json.dumps({"category": "working_db_integrity", "sha256_before": before,
                          "sha256_after": after, "unchanged": before == after}, ensure_ascii=False), flush=True)
    smoke = [result for result in results if result.get("category") == "smoke"]
    from tools import build_submission_archive as packaging
    current_names = {path.relative_to(ROOT).as_posix() for path in packaging.source_files()}
    changed = sorted((set(manifest) ^ current_names) |
                     {name for name, checksum in manifest.items() if digest(ROOT / name) != checksum})
    if changed:
        print(json.dumps({"category": "source_changed_during_run", "files": changed}, ensure_ascii=False), flush=True)
    summary = {"category": "summary", "ok": all(result["ok"] for result in results) and before == after and not changed,
               "stages": len(results), "smoke_scripts": len(smoke),
               "smoke_checks": sum(result.get("checks", 0) for result in smoke),
               "source_changed_during_run": bool(changed),
               "scope": "offline regression only; not live provider or seven-day trip acceptance"}
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
