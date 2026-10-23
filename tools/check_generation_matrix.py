"""Reproducible synthetic acceptance matrix, with no network, AI or working DB.

These fixtures test scheduling contracts, not real POI availability or weather.
All places are deliberately synthetic; they must never be saved as user trips.
Run with --output output/diagnostics/<new-name>.json to retain an exclusive report.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from city_candidates import CityCandidate
from city_deterministic import build_deterministic_city_itinerary
from city_quality import check_city_quality
from city_visit_estimates import check_city_visit_estimates
from city_visit_options import visit_option, check_city_visit_conditions
from places import verify_itinerary
from rules import check_city_radius, check_program_depth, check_rules, check_verified_place_duplicates
from schemas import TripRequest


DESTINATIONS = (
    ("Магадан", "Россия", "RU", 59.56, 150.80, "RUB"),
    ("Сочи", "Россия", "RU", 43.60, 39.73, "RUB"),
    ("Nha Trang", "Việt Nam", "VN", 12.24, 109.19, "EUR"),
    ("Kemer", "Türkiye", "TR", 36.60, 30.56, "EUR"),
)
NAMES = ("Альфа", "Бета", "Гамма", "Дельта", "Эпсилон", "Зета", "Эта", "Тета")


def fixture(destination=DESTINATIONS[0], *, days=3, ages=()):
    city, country, code, lat, lon, currency = destination
    start = date(2026, 10, 28)
    request = TripRequest(destination=f"{city}, {country}", start_date=start.isoformat(),
                          end_date=(start + timedelta(days=days - 1)).isoformat(), adults=2,
                          children_ages=list(ages), budget=100000 if currency == "RUB" else 10000,
                          currency=currency, pace="средний", start_time="09:00", end_time="19:00")
    center = {"name": city, "country": country, "country_code": code, "lat": lat, "lon": lon}
    candidates = []
    for index, name in enumerate(NAMES):
        title = f"Контрольный исторический музей {name}"
        query = f"{title}, {city}, {country}"
        candidates.append(CityCandidate(f"osm-node-{900001 + index}", title, query, city, {
            "name": query, "lat": lat, "lon": lon + index * .0005,
            "provider": "osm", "osm_type": "node", "osm_id": 900001 + index,
            "type": "museum", "address": {"city": city, "country": country, "country_code": code.lower()},
        }))
    return request, center, candidates


def scenarios():
    cases = []
    for destination in DESTINATIONS:
        for days in (3, 4, 5):
            for ages in ((), (5,), (4, 10)):
                request, center, candidates = fixture(destination, days=days, ages=ages)
                cases.append({"name": f"{destination[0]}-{days}d-{len(ages)}children",
                              "request": request, "center": center, "candidates": candidates,
                              "expected": "success"})
    for days in (1, 7, 14):
        request, center, candidates = fixture(days=days)
        cases.append(dict(name=f"calendar-{days}", request=request, center=center,
                          candidates=candidates, expected="success"))
    for case, expected in (("empty", "insufficient_data"), ("rain-unknown-indoor", "insufficient_data"),
                           ("rain-confirmed-fixture", "success"), ("meal-budget", "insufficient_data"),
                           ("short-window", "invalid_request"), ("search-bound", "search_limit"),
                           ("incompatible-distance", "insufficient_data")):
        request, center, candidates = fixture()
        extra = {}
        if case == "empty":
            candidates = []
        elif case.startswith("rain-"):
            extra["rainy_dates"] = [(date.fromisoformat(request.start_date) + timedelta(days=i)).isoformat()
                                    for i in range(request.days_count)]
            if case == "rain-confirmed-fixture":
                # Test-only server-owned inputs, not published place facts.
                extra["options"] = [replace(visit_option(value, request), indoor=True,
                                            description="Синтетическая крытая экспозиция для проверки алгоритма.")
                                    for value in candidates]
        elif case == "meal-budget":
            request.budget = 100
        elif case == "short-window":
            request.end_time = "10:00"
        elif case == "search-bound":
            extra["max_states"] = 1
        elif case == "incompatible-distance":
            candidates = candidates[:2]
            candidates[1].record["lon"] += .20
        cases.append(dict(name=case, request=request, center=center, candidates=candidates,
                          expected=expected, extra=extra))
    return cases


def independent_checks(result, request, center, candidates, rainy_dates):
    if result.itinerary is None:
        return []
    itinerary = result.itinerary.model_copy(deep=True)
    records = {value.place_query: value.record for value in candidates}
    problems = check_rules(itinerary, request, rainy_dates, allow_city_limited_program=True)
    problems += verify_itinerary(itinerary, verified_records=records)
    problems += check_city_radius(itinerary, (center["lat"], center["lon"]))
    problems += check_verified_place_duplicates(itinerary, verified_records=records)
    problems += check_program_depth(itinerary, request, allow_city_limited_program=True)
    problems += check_city_quality(itinerary, request, records, allow_city_limited_program=True)
    problems += check_city_visit_estimates(itinerary, candidates)
    problems += check_city_visit_conditions(itinerary, candidates, request)
    return problems


def run_matrix():
    reports = []
    with ExitStack() as stack:
        def forbidden(*args, **kwargs):
            raise AssertionError("The synthetic matrix must never use external services or a database")
        for target in ("httpx.get", "httpx.post", "httpx.Client.send", "sqlite3.connect", "places.geocode",
                       "planner.generate_json", "city_candidates._lookup", "weather.get_weather"):
            stack.enter_context(patch(target, side_effect=forbidden))
        for case in scenarios():
            request, center, candidates = case["request"], case["center"], case["candidates"]
            extra = case.get("extra", {})
            before = deepcopy((request, center, candidates, extra))
            try:
                result = build_deterministic_city_itinerary(request, candidates, resolved_city=center, **extra)
                findings = independent_checks(result, request, center, candidates, extra.get("rainy_dates", []))
                preserved = (request, center, candidates, extra) == before
                ok = (result.status == case["expected"] and not findings and preserved
                      and (result.itinerary is not None) == (result.status == "success")
                      and bool(result.problems) == (result.status != "success"))
                reports.append({"case": case["name"], "expected": case["expected"], "status": result.status,
                                "ok": ok, "inputs_unchanged": preserved, "validator_findings": len(findings),
                                "coverage": result.coverage})
            except Exception as error:
                reports.append({"case": case["name"], "expected": case["expected"],
                                "status": "technical_failure", "ok": False, "error_type": type(error).__name__})
    counts = Counter(row["status"] for row in reports)
    return {"schema_version": 1, "scope": "synthetic offline scheduling, NOT live destination availability",
            "ai_calls": 0, "network_calls": 0, "working_db_used": False,
            "cases": len(reports), "passed": sum(row["ok"] for row in reports),
            "ok": all(row["ok"] for row in reports), "outcomes": dict(counts), "results": reports}


def replay_diagnostics(directories):
    """Rebuild from previously captured records, not a fresh place search."""
    from tools.check_city_acceptance import _center, _load, _rainy_dates, _records
    rows = []
    for directory in directories:
        path = Path(directory).resolve()
        request_data, request_sha = _load(path / "request.json")
        report, report_sha = _load(path / "report.json")
        pool, pool_sha = _load(path / "city-candidates.json")
        request = TripRequest.model_validate(request_data)
        center = _center(request, report.get("resolved_city"))
        _, by_query = _records(pool, request, center)
        candidates = [CityCandidate(**value) for value in by_query.values()]
        rainy = _rainy_dates(report.get("weather"), request, center)
        with ExitStack() as stack:
            def forbidden(*args, **kwargs):
                raise AssertionError("Replay must not call external services or a database")
            for target in ("httpx.get", "httpx.post", "httpx.Client.send", "sqlite3.connect",
                           "places.geocode", "planner.generate_json", "city_candidates._lookup"):
                stack.enter_context(patch(target, side_effect=forbidden))
            result = build_deterministic_city_itinerary(request, candidates, rainy, resolved_city=center)
            findings = independent_checks(result, request, center, candidates, rainy)
        rows.append({"case": path.name, "status": result.status, "problems": result.problems,
                     "validator_findings": findings, "coverage": result.coverage,
                     "input_hashes": {"request": request_sha, "report": report_sha, "pool": pool_sha},
                     "itinerary": result.itinerary.model_dump(mode="json") if result.itinerary else None})
    return {"schema_version": 1, "scope": "offline replay of saved map/weather records, NOT fresh live search",
            "ai_calls": 0, "network_calls": 0, "working_db_used": False,
            "cases": len(rows), "outcomes": dict(Counter(row["status"] for row in rows)),
            "ok": all(not row["validator_findings"] and row["status"] in {"success", "insufficient_data", "search_limit"}
                      for row in rows), "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--diagnostic-dir", type=Path, action="append", default=[],
                        help="Replay a saved diagnostic instead of synthetic fixtures; repeat for each city")
    args = parser.parse_args()
    report = replay_diagnostics(args.diagnostic_dir) if args.diagnostic_dir else run_matrix()
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to(ROOT / "output"):
            parser.error("Reports must stay inside the project's output directory")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False))
    for row in report["results"]:
        if not row.get("ok", row["status"] == "success"):
            print(json.dumps(row, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
