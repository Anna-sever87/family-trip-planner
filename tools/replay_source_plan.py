"""Check source scheduling against saved grounding, without any paid AI call."""
import argparse
from dataclasses import asdict
from datetime import date, timedelta
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schemas import TripRequest
from destination_scope import resolve_region
from grounded_schedule import enforce_grounded_choices
from planner import _map_verification_problems
from rules import check_program_depth, check_rules
from regional_skeleton import _visit_problem, build_regional_skeleton
from regional_routing import build_travel_graph, verify_selected_routes
from source_planning import itinerary_from_skeleton
from visit_facts import visit_facts_for_grounding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grounding-file", type=Path, required=True)
    parser.add_argument("--request-file", type=Path, required=True,
                        help="Explicit request JSON for the saved grounding.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--days", type=int)
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--allow-free-days", action="store_true",
                        help="Allow an explicit limited programme while retaining all requested dates and costs")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if Path.cwd().resolve() != ROOT or not output.is_relative_to(ROOT / "output") or output.exists():
        raise SystemExit("Choose a fresh output directory within this project's output/.")
    request = TripRequest.model_validate_json(args.request_file.read_text(encoding="utf-8"))
    if args.days is not None:
        if not 1 <= args.days <= 14:
            raise SystemExit("days must be 1..14")
        request.end_date = (date.fromisoformat(request.start_date) + timedelta(days=args.days-1)).isoformat()
    raw = json.loads(args.grounding_file.read_text(encoding="utf-8"))
    grounding = SimpleNamespace(**raw.get("result", raw))
    facts = visit_facts_for_grounding(request, grounding)
    usable_facts = [fact for fact in facts if _visit_problem(fact, request) is None]
    with patch("yandex_ai.generate_json", side_effect=AssertionError("Paid AI forbidden in replay")):
        if args.cached_only:
            with patch("regional_routing.httpx.get", side_effect=ValueError("Network forbidden in cached-only replay")):
                graph, warnings = build_travel_graph(grounding, usable_facts)
        else:
            graph, warnings = build_travel_graph(grounding, usable_facts)
        skeleton = build_regional_skeleton(request, grounding, facts, travel_facts=graph,
                                           allow_free_days=args.allow_free_days)
    report = dict(paid_requests=0, request=request.model_dump(), visit_facts=[asdict(item) for item in facts],
                  usable_visit_ids=[item.visit_id for item in usable_facts],
                  travel_facts=[asdict(item) for item in graph], warnings=warnings, skeleton=asdict(skeleton))
    if skeleton.status in ("ready", "limited_program"):
        if args.cached_only:
            with patch("regional_routing.httpx.get", side_effect=ValueError("Network forbidden in cached-only replay")):
                problems = verify_selected_routes(skeleton, grounding, usable_facts)
        else:
            problems = verify_selected_routes(skeleton, grounding, usable_facts)
        report["route_problems"] = problems
        if not problems:
            itinerary = itinerary_from_skeleton(request, grounding, skeleton)
            problems = enforce_grounded_choices(itinerary, grounding) + check_rules(
                itinerary, request, allow_limited_program=args.allow_free_days)
            if not problems:
                def check_map():
                    return _map_verification_problems(itinerary, None, resolve_region(request.destination),
                                                     start_time=request.start_time, end_time=request.end_time)
                if args.cached_only:
                    with patch("places.httpx.get", side_effect=ValueError("Network forbidden in cached-only replay")):
                        problems += check_map()
                else:
                    problems += check_map()
            problems += check_program_depth(itinerary, request, {day.day: day.depth_credit_min for day in skeleton.days},
                                             allow_limited_program=args.allow_free_days)
            report["validation_problems"] = problems
            report["itinerary"] = itinerary.model_dump()
    output.mkdir(parents=True)
    with (output / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status=skeleton.status, facts=len(facts), usable=len(usable_facts),
                          edges=len(graph), warnings=warnings, states=skeleton.search_states,
                          cost=skeleton.known_cost_total+skeleton.planning_reserve_total,
                          program_days=sum(not day.is_free_day for day in skeleton.days),
                          free_days=sum(day.is_free_day for day in skeleton.days),
                          route_problems=report.get("route_problems", []),
                          validation_problems=report.get("validation_problems", [])), ensure_ascii=False))


if __name__ == "__main__":
    main()
