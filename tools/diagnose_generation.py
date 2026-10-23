"""Explicit bounded generator diagnostic; never saves a trip or private memory.

Run from the project directory. Preview with --dry-run. A paid run requires
--run-once and a fresh output directory; rerunning that directory never calls AI.
The ordinary account quota and busy guard apply to this diagnostic too.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import db
import planner
import reminders
import web_service
import yandex_ai
from schemas import TripRequest

class DiagnosticBudgetExceeded(RuntimeError):
    """A fourth completion is blocked before entering the AI client."""


def _write_json(path: Path, payload, *, exclusive=False):
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


@contextmanager
def _model_override(model):
    if model is None:
        yield
        return
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", model):
        raise ValueError("Model must be a model name, not a URL or credentials.")
    folder = os.getenv("YANDEX_FOLDER_ID", "").strip()
    if not folder:
        raise ValueError("YANDEX_FOLDER_ID is not configured.")
    previous = os.environ.get("YANDEX_MODEL_URI")
    os.environ["YANDEX_MODEL_URI"] = f"gpt://{folder}/{model}"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("YANDEX_MODEL_URI", None)
        else:
            os.environ["YANDEX_MODEL_URI"] = previous


def run_diagnostic(request, owner, source_trip, output_dir, *, max_attempts=3, model=None,
                   city_candidates_first=False):
    """One quota operation, at most three completions; only diagnostics are saved."""
    if type(max_attempts) is not int or not 2 <= max_attempts <= 3:
        raise ValueError("max_attempts must be an integer from 2 to 3.")
    if city_candidates_first and request.destination_kind != "city":
        raise ValueError("Candidate-first city diagnostics require a city request.")
    output_dir = Path(output_dir)
    # The exclusive directory is the run-once guard, including interrupted runs.
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "prepared", "source_trip": source_trip,
        "model_override": model, "max_completion_calls": max_attempts,
        "pipeline": "city_candidates_first" if city_candidates_first else "standard",
        "private_memory_used": False, "trip_saved": False, "calls": [],
    }
    report_path = output_dir / "report.json"
    _write_json(output_dir / "request.json", request.model_dump(mode="json"), exclusive=True)
    _write_json(report_path, report, exclusive=True)
    account_key = db.get_account_key(owner)
    if not account_key:
        report.update(status="blocked", reason="account_unavailable")
        _write_json(report_path, report)
        return report
    usage_date = datetime.now(reminders.get_tz()).date()
    claim = db.claim_ai_request(
        owner, usage_date, "plan", "diagnostic", trip_id=source_trip,
        daily_limit=web_service._daily_ai_limit(), expected_account_key=account_key,
    )
    report.update(quota_used=claim.used, quota_limit=claim.limit, usage_date=str(usage_date))
    if claim.status != "ok" or not claim.claim_id:
        report.update(status="blocked", reason=claim.status)
        _write_json(report_path, report)
        return report

    original_generate = planner.generate_json
    original_ground = planner.ground_region_candidates
    original_post = yandex_ai.httpx.post
    city_module = None
    original_city_ground = None
    original_city_resolver = None
    succeeded = False

    def capture_grounding(*args, **kwargs):
        grounding = original_ground(*args, **kwargs)
        _write_json(output_dir / "grounding.json", asdict(grounding), exclusive=True)
        return grounding

    def capture_city_grounding(*args, **kwargs):
        grounding = original_city_ground(*args, **kwargs)
        _write_json(output_dir / "city-candidates.json", asdict(grounding), exclusive=True)
        report["map_requests"] = grounding.map_requests
        report["confirmed_candidates"] = len(grounding.candidates)
        return grounding

    def capture_city_resolution(*args, **kwargs):
        center = original_city_resolver(*args, **kwargs)
        if isinstance(center, dict):
            # Keep only public geographic context returned by the real resolver;
            # acceptance must not fabricate its radius anchor from POI positions.
            report["resolved_city"] = {
                key: center[key] for key in ("name", "lat", "lon", "country_code", "admin1") if key in center
            }
        return center

    def capture_http_response(*args, **kwargs):
        response = original_post(*args, **kwargs)
        if response.is_error and report["calls"]:
            try:
                body = response.json()
                message = body.get("message", "") if isinstance(body, dict) else ""
                if not message and isinstance(body, dict) and isinstance(body.get("error"), dict):
                    message = body["error"].get("message", "")
                if not isinstance(message, str):
                    message = ""
                # Never retain request/response headers or full payloads. Provider
                # diagnostics may echo identifiers, so redact all configured secrets.
                secrets = {
                    value for name, value in os.environ.items()
                    if value and len(value) > 3 and any(part in name.upper() for part in
                        ("KEY", "TOKEN", "SECRET", "FOLDER_ID", "PASSWORD", "CREDENTIAL"))
                }
                for value in sorted(secrets, key=len, reverse=True):
                    message = message.replace(value, "[redacted]")
                message = re.sub(r"(?:https?|gpt)://\S+", "[redacted-url]", message, flags=re.IGNORECASE)[:2000]
                _write_json(output_dir / f"call-{len(report['calls']):02d}-error.json",
                            {"http_status": response.status_code, "provider_message": message}, exclusive=True)
            except (ValueError, TypeError):
                pass
        return response

    def capture(messages, **kwargs):
        if len(report["calls"]) >= max_attempts:
            report["blocked_extra_completion"] = True
            raise DiagnosticBudgetExceeded("Completion budget exhausted; no extra HTTP request was sent.")
        number = len(report["calls"]) + 1
        call = {"number": number, "status": "started"}
        report["calls"].append(call)
        # Save schema, not prompts: model messages may contain untrusted text.
        _write_json(output_dir / f"call-{number:02d}-schema.json", kwargs.get("json_schema"), exclusive=True)
        _write_json(report_path, report)
        call_started = time.monotonic()
        try:
            generation = original_generate(messages, **kwargs)
            with (output_dir / f"call-{number:02d}-raw.txt").open("x", encoding="utf-8") as stream:
                stream.write(generation.text)
            call.update(
                status="received", input_tokens=int(generation.input_tokens),
                output_tokens=int(generation.output_tokens),
                raw_file=f"call-{number:02d}-raw.txt",
            )
            return generation
        except Exception as error:
            # Exception text, request URLs, headers and messages are not captured.
            call.update(status="failed", error_type=type(error).__name__)
            raise
        finally:
            call["elapsed_seconds"] = round(time.monotonic() - call_started, 3)
            _write_json(report_path, report)

    try:
        report["status"] = "running"
        _write_json(report_path, report)
        planner.generate_json = capture
        planner.ground_region_candidates = capture_grounding
        if city_candidates_first:
            import city_planner as city_module
            original_city_ground = city_module.ground_city_candidates
            original_city_resolver = city_module.find_city
            city_module.ground_city_candidates = capture_city_grounding
            city_module.find_city = capture_city_resolution
        yandex_ai.httpx.post = capture_http_response
        with _model_override(model):
            options = {"city_candidates_first": True} if city_candidates_first else {}
            result = planner.generate_itinerary(request, max_attempts=max_attempts, memory_context="",
                                                ground_candidates=request.destination_kind == "region", **options)
        accepted = not result.problems
        report.update(
            status="succeeded" if accepted else "rejected",
            problems=list(result.problems), attempts=int(result.attempts),
            prompt_tokens=int(result.prompt_tokens), completion_tokens=int(result.completion_tokens),
            total_tokens=int(result.total_tokens),
            weather=result.weather,
            weather_text=result.weather_text,
            generation_status=getattr(result, "generation_status", "unknown"),
            generation_coverage=getattr(result, "generation_coverage", {}),
        )
        _write_json(output_dir / "final-itinerary.json", result.itinerary.model_dump(mode="json"), exclusive=True)
        # A valid result is not a completed diagnostic until its artifact exists.
        succeeded = accepted
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__)
    finally:
        planner.generate_json = original_generate
        planner.ground_region_candidates = original_ground
        if city_module is not None and original_city_ground is not None:
            city_module.ground_city_candidates = original_city_ground
        if city_module is not None and original_city_resolver is not None:
            city_module.find_city = original_city_resolver
        yandex_ai.httpx.post = original_post
        if report["status"] == "running":
            report["status"] = "interrupted"
        try:
            report["quota_finished"] = bool(db.finish_ai_request(claim.claim_id, succeeded=succeeded))
        except Exception as error:
            report.update(quota_finished=False, quota_finish_error_type=type(error).__name__)
        report.update(
            finished_at=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            completion_calls_attempted=len(report["calls"]),
        )
        _write_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-trip", type=int, required=True, help="Existing trip used only to identify its owner.")
    parser.add_argument("--request-file", type=Path, required=True,
                        help="Explicit request JSON; no saved personal request is selected implicitly.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, choices=(2, 3), default=3)
    parser.add_argument("--model", help="Optional model name, e.g. yandexgpt-5.1; .env is never modified.")
    parser.add_argument("--city-candidates-first", action="store_true",
                        help="Opt-in city pipeline: verify candidate places before scheduling. Not the public UI default.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run-once", action="store_true")
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("Run this command from the project directory.")
    output_dir = args.output_dir.resolve()
    output_root = (ROOT / "output").resolve()
    if output_dir == output_root or not output_dir.is_relative_to(output_root):
        raise SystemExit("Choose a new subdirectory inside this project's output directory.")
    if output_dir.exists():
        raise SystemExit("Output directory already exists: this diagnostic will not run again.")
    if args.model and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", args.model):
        raise SystemExit("Use a model name only.")
    if args.run_once:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    request = web_service._normalized_trip_request(
        TripRequest.model_validate_json(args.request_file.read_text(encoding="utf-8"))
    )
    if args.dry_run:
        print(json.dumps({
            "status": "preview", "request": request.model_dump(mode="json"),
            "max_completion_calls": args.max_attempts, "model_override": args.model,
            "pipeline": "city_candidates_first" if args.city_candidates_first else "standard",
            "quota_reserved": False, "trip_saved": False, "output_created": False,
        }, ensure_ascii=False, indent=2))
        return 0
    database = Path(db.DB_PATH).resolve()
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        row = connection.execute("SELECT tg_id FROM trips WHERE id=?", (args.source_trip,)).fetchone()
    if row is None:
        raise SystemExit("Source trip does not exist.")
    report = run_diagnostic(request, int(row[0]), args.source_trip, output_dir,
                            max_attempts=args.max_attempts, model=args.model,
                            city_candidates_first=args.city_candidates_first)
    # Raw candidate content is left in local artifacts, not dumped to the terminal.
    print(json.dumps({key: report.get(key) for key in (
        "status", "model_override", "max_completion_calls", "completion_calls_attempted",
        "attempts", "total_tokens", "quota_used", "quota_limit", "quota_finished", "trip_saved",
    )}, ensure_ascii=False, indent=2))
    print(f"Diagnostic artifacts: {output_dir}")
    return 0 if report["status"] == "succeeded" and report.get("quota_finished") else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    raise SystemExit(main())
