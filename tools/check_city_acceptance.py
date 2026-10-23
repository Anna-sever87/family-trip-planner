"""Offline acceptance of saved city diagnostics, not a user trip or browser test.

Reads request.json, final-itinerary.json, city-candidates.json and report.json.
Revalidates exact map evidence, stores only in disposable SQLite, exercises the
public read/PDF paths and invalid-context edit refusal before quota, and exports
trip.pdf plus a non-private acceptance report. Successful editing is not tested.
No AI, network, working database, private memory, or source-artifact writes.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from datetime import date, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_offline_checks import OfflineGuard

INPUT_FILES = ("request.json", "final-itinerary.json", "city-candidates.json", "report.json")
OWNER, OTHER = 93001, 93002
MAX_INPUT_BYTES = 2 * 1024 * 1024


class AcceptanceFailure(ValueError):
    def __init__(self, stage, count=1):
        self.stage, self.count = stage, count
        super().__init__(stage)  # No raw model/private text in errors or reports.


def _require(condition, stage):
    if not condition:
        raise AcceptanceFailure(stage)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AcceptanceFailure("duplicate_json_key")
        result[key] = value
    return result


def _load(path):
    _require(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_INPUT_BYTES, "input_file")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_pairs,
                       parse_constant=lambda _: (_ for _ in ()).throw(AcceptanceFailure("nonfinite_json")))
    _require(isinstance(value, dict), "input_object")
    return value, hashlib.sha256(raw).hexdigest()


class _Guard(OfflineGuard):
    """Reversible transport patches plus an inactive-on-exit audit hook."""
    def __init__(self, scratch, output):
        super().__init__(scratch)
        self.output = self.path(output)
        self.database = self.path(Path(scratch) / "acceptance.sqlite3")
        self.active = False

    def inside(self, path):
        candidate = self.path(path)
        return any(os.path.commonpath((root, candidate)) == root for root in (self.root, self.output)
                   if os.path.splitdrive(root)[0] == os.path.splitdrive(candidate)[0])

    def audit(self, event, args):
        if not self.active:
            return
        if event == "sqlite3.connect" and self.path(args[0]) != self.database:
            self.deny("SQLite outside disposable acceptance database")
        super().audit(event, args)


@contextmanager
def _isolated(scratch, output):
    guard = _Guard(scratch, output)
    sys.addaudithook(guard.audit)
    guard.active = True
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with ExitStack() as stack:
            import dotenv
            import httpx
            stack.enter_context(patch.object(dotenv, "load_dotenv", return_value=False))
            forbidden = lambda *a, **kw: guard.deny("external service or private memory")
            stack.enter_context(patch.object(httpx.HTTPTransport, "handle_request", side_effect=forbidden))
            stack.enter_context(patch.object(httpx.AsyncHTTPTransport, "handle_async_request", side_effect=forbidden))
            import db
            import planner
            import web_service
            import yandex_ai
            import city_context
            import weather
            stack.enter_context(patch.object(db, "DB_PATH", guard.database))
            for module, name in ((db, "claim_ai_request"), (planner, "generate_json"),
                                 (planner, "generate_itinerary"), (web_service, "generate_itinerary"),
                                 (planner, "edit_itinerary"), (web_service, "edit_itinerary"),
                                 (yandex_ai, "generate_json"),
                                 (city_context, "recover_city_context"), (city_context, "_lookup"),
                                 (weather, "find_city"), (weather, "get_weather"),
                                 (web_service.vector_memory, "recall_family_memories"),
                                 (web_service, "_save_trip_memory_safely")):
                stack.enter_context(patch.object(module, name, side_effect=forbidden))
            import places
            for name in ("geocode", "_load_cache", "_save_cache", "_wait_for_turn"):
                stack.enter_context(patch.object(places, name, side_effect=forbidden))
            stack.enter_context(patch.object(web_service.reminders, "get_tz", return_value=timezone.utc))
            stack.enter_context(patch.object(web_service.reminders, "get_days_before", return_value=3))
            yield guard
    finally:
        guard.active = False
        sys.dont_write_bytecode = old_bytecode


def _center(request, value):
    from city_candidates import _country_code, _finite_coordinates
    import places
    _require(_finite_coordinates(value), "resolved_city_coordinates")
    parts = [part.strip() for part in request.destination.split(",")]
    code = _country_code(value.get("country_code"))
    _require(bool(code) and 1 <= len(parts) <= 3 and all(parts), "resolved_city_context")
    _require(places._context_identity(value.get("name", "")) == places._context_identity(parts[0]),
             "resolved_city_name")
    if len(parts) > 1:
        _require(places._country_identity(parts[-1]) == (code,), "resolved_city_country")
    if len(parts) == 3:
        _require(places._context_identity(value.get("admin1", "")) == places._context_identity(parts[1]),
                 "resolved_city_admin")
    return dict(value, country_code=code)


def _records(pool, request, center):
    import places
    from city_candidates import _canonical_record, _finite_coordinates
    candidates = pool.get("candidates")
    _require(pool.get("problems") == [] and isinstance(candidates, list)
             and 1 <= len(candidates) <= min(18, 2 * request.days_count + 4), "candidate_pool")
    records, identities, by_query = {}, set(), {}
    for candidate in candidates:
        _require(isinstance(candidate, dict) and _finite_coordinates(candidate.get("record")), "candidate_record")
        prepared = _canonical_record(candidate.get("title"), candidate["record"], center)
        _require(prepared is not None, "candidate_canonical_record")
        identity = places.map_record_identity(prepared)
        query = prepared["place_query"]
        _require(candidate.get("candidate_id") == "-".join(map(str, identity))
                 and candidate.get("place_query") == query
                 and candidate.get("title") == query.split(",", 1)[0]
                 and candidate.get("area") == prepared["locality"], "candidate_identity")
        _require(identity not in identities and query not in records, "candidate_duplicate")
        identities.add(identity)
        records[query], by_query[query] = prepared, candidate
    return records, by_query


def _rainy_dates(weather, request, center):
    import places
    from weather import RAIN_MM
    if weather is None:
        return []
    _require(isinstance(weather, dict) and weather.get("source") in {"forecast", "fact", "history"}, "weather_source")
    _require(places._context_identity(weather.get("city", "")) == places._context_identity(center["name"]), "weather_city")
    days = weather.get("days")
    expected = [(date.fromisoformat(request.start_date) + timedelta(days=index)).isoformat()
                for index in range(request.days_count)]
    _require(isinstance(days, list) and bool(days) and all(isinstance(day, dict) for day in days), "weather_days")
    dates = [day.get("date") for day in days]
    # A forecast made during a trip may cover its remaining contiguous suffix;
    # archive/history output must cover the entire original requested calendar.
    _require(dates == expected or (weather["source"] == "forecast" and len(dates) <= len(expected)
                                  and dates == expected[-len(dates):]), "weather_calendar")
    for day in days:
        values = [day.get(key) for key in ("tmin", "tmax", "rain_mm")]
        _require(all(type(value) in (int, float) and math.isfinite(value) for value in values)
                 and values[0] <= values[1] and values[2] >= 0, "weather_values")
    return [day["date"] for day in days if weather["source"] in {"forecast", "fact"} and day["rain_mm"] >= RAIN_MM]


def _validate(payloads, center_override, *, label_estimates=False):
    from schemas import Itinerary, TripRequest
    from program_coverage import limited_program_problems
    from city_quality import check_city_quality
    from city_candidates import CityCandidate
    from city_context import build_city_context
    from city_visit_estimates import check_city_visit_estimates
    from places import verify_itinerary
    from rules import check_city_radius, check_program_depth, check_rules, check_verified_place_duplicates
    report = payloads["report.json"]
    _require(report.get("status") == "succeeded" and report.get("pipeline") == "city_candidates_first"
             and report.get("problems") == [] and report.get("private_memory_used") is False
             and report.get("trip_saved") is False and report.get("quota_finished") is True, "accepted_diagnostic")
    request = TripRequest.model_validate(payloads["request.json"])
    _require(request.destination_kind == "city" and 1 <= request.days_count <= 14, "city_request")
    center = _center(request, center_override if center_override is not None else report.get("resolved_city"))
    if center_override is not None and report.get("resolved_city") is not None:
        _require(_center(request, report["resolved_city"]) == center, "center_override_conflict")
    records, candidates = _records(payloads["city-candidates.json"], request, center)
    itinerary = Itinerary.model_validate(payloads["final-itinerary.json"])
    _require(itinerary.city_schedule is True and itinerary.source_schedule is False, "city_schedule_flag")
    if label_estimates:
        from city_estimates import label_city_estimates
        # Work on a model copy; even presentation changes never touch artifacts.
        itinerary = itinerary.model_copy(deep=True)
        def substantive(value):
            values = value.model_dump()
            values.pop("tips")
            for day in values["days"]:
                for item in day["items"]:
                    if item["kind"] in {"meal", "rest"}:
                        item.pop("description")
                        item.pop("indoor")
                    if request.children_ages and item["kind"] in {"meal", "rest", "attraction"}:
                        item.pop("kids_note")
            return values
        before = substantive(itinerary)
        label_city_estimates(itinerary, request)
        _require(substantive(itinerary) == before, "estimate_label_changed_program")
    if itinerary.program_report is not None:
        _require(itinerary.program_report.eligible_unique_visits == len(records), "eligible_count")
    for day in itinerary.days:
        for item in day.items:
            if item.kind == "attraction":
                candidate = candidates.get(item.place_query)
                _require(candidate is not None and item.title == candidate["title"] and item.area == candidate["area"],
                         "selected_candidate_identity")
    rainy = _rainy_dates(report.get("weather"), request, center)
    checked_candidates = [CityCandidate(candidate_id=value["candidate_id"], title=value["title"],
                                        place_query=query, area=value["area"], record=records[query])
                          for query, value in candidates.items()]
    import planner
    stages = (
        ("metadata", lambda: limited_program_problems(itinerary, request, allow_city_limited_program=True)),
        ("rules", lambda: check_rules(itinerary, request, rainy, allow_city_limited_program=True)),
        ("exact_map_records", lambda: verify_itinerary(itinerary, verified_records=records)),
        ("radius", lambda: check_city_radius(itinerary, (center["lat"], center["lon"]))),
        ("duplicates", lambda: check_verified_place_duplicates(itinerary, verified_records=records)),
        ("curated_exclusions", lambda: planner._blocked_place_problems(itinerary, planner._load_blocked_places(request.destination))),
        ("depth", lambda: check_program_depth(itinerary, request, allow_city_limited_program=True)),
        ("city_quality", lambda: check_city_quality(itinerary, request, records, allow_city_limited_program=True)),
        ("city_visit_estimates", lambda: check_city_visit_estimates(itinerary, checked_candidates)),
    )
    for stage, check in stages:
        problems = check()
        if problems:
            raise AcceptanceFailure(stage, len(problems))
    context = build_city_context(request, checked_candidates, center)
    return request, itinerary, report, records, rainy, context


def _pdf_reader(sitepackages):
    try:
        from pypdf import PdfReader
    except ImportError:
        bundle = Path(sitepackages) if sitepackages is not None else (
            Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages")
        _require((bundle / "pypdf").is_dir(), "pdf_reader_unavailable")
        sys.path.append(str(bundle))
        try:
            from pypdf import PdfReader
        finally:
            sys.path.remove(str(bundle))
    return PdfReader


def _invalid_context_refusal(request_data, itinerary):
    """Exercise only a separate disposable trip, never mutate the valid one."""
    import city_context
    import db
    import web_service

    damaged_request = deepcopy(request_data)
    damaged_request[city_context.CONTEXT_KEY]["version"] = 0
    probe_id = db.save_trip(OWNER, damaged_request, itinerary.model_dump())
    try:
        before = deepcopy(db.get_trip(probe_id, OWNER))
        for target_day in {itinerary.days[0].day,
                           next((day.day for day in itinerary.days if day.day_mode == "free"), itinerary.days[0].day)}:
            try:
                web_service.edit_trip_day(OWNER, probe_id, target_day, "Уточни обед")
            except ValueError as error:
                _require(str(error) == city_context.CONTEXT_ERROR, "invalid_context_refusal_reason")
            else:
                raise AcceptanceFailure("invalid_context_refusal")
        _require(db.get_trip(probe_id, OWNER) == before, "invalid_context_edit_mutation")
    finally:
        # The guard binds every SQLite connection to acceptance.sqlite3.
        _require(db.delete_trip(probe_id, OWNER), "temporary_probe_cleanup")


def _stored_acceptance(request, itinerary, report, reader, city_map_context):
    import city_context
    import db
    import web_service
    from checklist import build_checklist
    from program_coverage import FREE_DAY_NOTE, program_report_text
    db.init_db()
    db.save_user(OTHER, name="Диагностический другой владелец")
    request_data = request.model_dump()
    request_data.update(weather=report.get("weather"), weather_text=report.get("weather_text", ""))
    candidates, _ = city_context.load_city_context(city_map_context, request)
    city_context.check_context_places(itinerary, candidates)
    request_data[city_context.CONTEXT_KEY] = deepcopy(city_map_context)
    checklist = build_checklist(request.adults, request.children_ages, report.get("weather"))
    trip_id = db.save_trip_bundle(OWNER, request_data, itinerary.model_dump(), checklist,
                                 name="Диагностическая временная семья", adults=request.adults,
                                 children_ages=request.children_ages, currency=request.currency, pace=request.pace)
    before = deepcopy(db.get_trip(trip_id, OWNER))
    detail = web_service.get_trip_detail(OWNER, trip_id)
    reloaded = web_service.get_trip_detail(OWNER, trip_id)
    _require(detail == reloaded and len(detail["checklist"]) == len(checklist) > 0, "reload_checklist")
    _require(detail["itinerary"]["program_report_text"] == program_report_text(itinerary), "public_report")
    _require(detail["edit_available"] is True and detail["edit_unavailable_reason"] == ""
             and detail["edit_scope_note"] == city_context.CITY_EDIT_NOTE, "edit_disclosure")
    _require(detail["request"] == request.model_dump()
             and city_context.CONTEXT_KEY not in detail
             and city_context.CONTEXT_KEY not in detail["itinerary"], "private_context_disclosure")
    for action in (lambda: web_service.get_trip_detail(OTHER, trip_id),
                   lambda: web_service.get_trip_pdf(OTHER, trip_id),
                   lambda: web_service.edit_trip_day(OTHER, trip_id, 1, "Уточни обед")):
        try:
            action()
        except web_service.TripNotFoundError:
            pass
        else:
            raise AcceptanceFailure("owner_isolation")
    _invalid_context_refusal(request_data, itinerary)
    pdf = web_service.get_trip_pdf(OWNER, trip_id)
    _require(pdf.startswith(b"%PDF-"), "pdf_format")
    parsed = reader(io.BytesIO(pdf))
    text = " ".join(" ".join(page.extract_text() or "" for page in parsed.pages).split())
    disclosure = program_report_text(itinerary)
    _require(not disclosure or " ".join(disclosure.split()) in text, "pdf_report")
    if any(day.day_mode == "free" for day in itinerary.days):
        _require(" ".join(FREE_DAY_NOTE.split()) in text, "pdf_free_day_note")
    _require(db.get_trip(trip_id, OWNER) == before
             and web_service.get_trip_detail(OWNER, trip_id) == detail, "read_edit_mutation")
    with db.connect() as connection:
        quota = connection.execute("SELECT COUNT(*) FROM ai_requests").fetchone()[0]
        trips = connection.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    _require(quota == 0 and trips == 1, "zero_quota_single_trip")
    return pdf, detail, len(parsed.pages)


def run_acceptance(input_dir, output_dir, *, center_json=None, bundled_sitepackages=None, label_estimates=False):
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    _require(source.is_dir() and source != output and source not in output.parents
             and output not in source.parents, "separate_output_directory")
    output.mkdir(parents=True, exist_ok=False)
    summary = {"kind": "offline_city_diagnostic_acceptance", "status": "failed",
               "acceptance_scope": "offline_storage_read_pdf_only", "product_fully_accepted": False,
               "user_trip_saved": False, "browser_acceptance": False, "working_db_accessed": False,
               "ai_calls": 0, "quota_operations": 0, "network_calls": 0, "private_memory_used": False,
               "edit_available": None, "edit_acceptance": False, "input_unchanged": False,
               "limitations": ["city_edit_not_tested", "not_browser_ui_acceptance", "map_only_not_visit_facts"],
               "display_estimates_relabelled": label_estimates is True}
    hashes, guard = {}, None
    try:
        with tempfile.TemporaryDirectory(prefix="city-acceptance-") as temporary:
            with _isolated(Path(temporary), output) as guard:
                payloads = {}
                for name in INPUT_FILES:
                    payloads[name], hashes[name] = _load(source / name)
                center = _load(Path(center_json).resolve())[0] if center_json is not None else None
                request, itinerary, report, records, rainy, context = _validate(payloads, center, label_estimates=label_estimates)
                pdf, detail, pages = _stored_acceptance(request, itinerary, report, _pdf_reader(bundled_sitepackages), context)
                _require(all(hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
                             for name, digest in hashes.items()), "input_changed")
                _require(not guard.violations, "guard_violation")
                with (output / "trip.pdf").open("xb") as stream:
                    stream.write(pdf)
                summary.update(status="passed", input_unchanged=True, validated_candidates=len(records),
                               requested_days=request.days_count,
                               program_days=sum(day.day_mode == "program" for day in itinerary.days),
                               free_days=sum(day.day_mode == "free" for day in itinerary.days),
                               selected_visits=sum(item.kind == "attraction" for day in itinerary.days for item in day.items),
                               rainy_dates_checked=len(rainy), checklist_items=len(detail["checklist"]),
                               budget=detail["budget"], pdf_pages=pages, pdf_sha256=hashlib.sha256(pdf).hexdigest(),
                               edit_available=detail["edit_available"], invalid_context_refused_before_quota=True,
                               owner_isolation=True, reload_unchanged=True)
                # The summary contains numbers/bools only, not request prose,
                # destination, profile, model messages, or private record payloads.
                summary["budget"] = {key: value for key, value in detail["budget"].items()
                                     if type(value) in (int, float, bool) or key == "currency"}
    except AcceptanceFailure as error:
        summary.update(status="rejected", failure_stage=error.stage, problem_count=error.count)
    except Exception as error:
        summary.update(error_type=type(error).__name__)
    finally:
        summary["input_sha256"] = hashes
        summary["guard_violations"] = len(guard.violations) if guard is not None else 0
        if hashes:
            summary["input_unchanged"] = all((source / name).is_file()
                and hashlib.sha256((source / name).read_bytes()).hexdigest() == digest for name, digest in hashes.items())
        with (output / "acceptance-report.json").open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, ensure_ascii=False, indent=2)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New separate directory; never overwritten")
    parser.add_argument("--center-json", type=Path, help="Saved confirmed city context; required if report has no resolved_city")
    parser.add_argument("--bundled-sitepackages", type=Path, help="Existing pypdf runtime, no installation")
    parser.add_argument("--label-estimates", action="store_true", help="Neutral estimate labels on an in-memory copy only")
    args = parser.parse_args(argv)
    result = run_acceptance(args.input_dir, args.output_dir, center_json=args.center_json,
                            bundled_sitepackages=args.bundled_sitepackages, label_estimates=args.label_estimates)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
