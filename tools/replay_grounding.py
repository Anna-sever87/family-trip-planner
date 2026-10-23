"""Recheck saved public candidate proposals against the map, never calling AI."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import candidate_grounding
import places
from destination_scope import resolve_region
from schemas import TripRequest
from yandex_ai import GenerationResult


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, required=True)
    parser.add_argument("--request-file", type=Path, required=True,
                        help="Explicit request JSON for the saved candidate proposals.")
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--cached-only", action="store_true",
                        help="Forbid all HTTP and reuse positive map records only.")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("Run from the project directory.")
    output = args.output_file.resolve()
    if not output.is_relative_to(ROOT / "output") or output.exists():
        raise SystemExit("Choose a new file within output/.")
    request = TripRequest.model_validate_json(args.request_file.read_text(encoding="utf-8"))
    raw = args.raw_file.read_text(encoding="utf-8")
    json.loads(raw)
    # Blocking POST protects the no-AI contract even if implementation changes.
    with ExitStack() as guards:
        guards.enter_context(patch.object(places.httpx, "post", side_effect=AssertionError("Paid requests forbidden during replay")))
        if args.cached_only:
            guards.enter_context(patch.object(places.httpx, "get", side_effect=AssertionError("HTTP forbidden in cached-only replay")))
        scope = resolve_region(request.destination)
        result = candidate_grounding.ground_region_candidates(
            request, scope,
            max_http_attempts=0 if args.cached_only else candidate_grounding.MAX_HTTP_ATTEMPTS,
            completion_fn=lambda *a, **kw: GenerationResult(text=raw),
            known_hints=places.cached_place_hints(request.destination, None, region_scope=scope),
        )
    report = {"mode": "saved_candidate_replay", "paid_requests": 0,
              "cached_only": args.cached_only,
              "source_raw": str(args.raw_file), "result": asdict(result)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"paid_requests": 0, "bases": len(result.base_facts),
                      "places": len(result.confirmed_hints), "rejected": len(result.rejected),
                      "problems": result.problems}, ensure_ascii=False))
    return 0 if not result.problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
