"""Server-only map snapshots for city edits; never trust model/UI metadata.

Persist in the saved request envelope, not the public TripRequest or model
schema. Map observations establish identity/location only. Legacy recovery
rechecks existing exact places; it never guesses IDs from stored coordinates.
"""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import re

from city_candidates import CityCandidate, CityCandidateResult, _canonical_record, _lookup
from city_evidence import city_managed_tip_limit, recognize_city_managed_tips, tip_sha256
import places
from schemas import Itinerary, TripRequest
from rules import city_candidate_limit


CONTEXT_KEY = "city_map_context"
CONTEXT_ERROR = (
    "Не удалось подтвердить сохранённые данные карты для изменения маршрута. "
    "Исходная поездка сохранена без изменений."
)
CITY_EDIT_NOTE = (
    "Можно изменить один день в пределах ранее проверенного набора мест. "
    "Другие дни не меняются; новые места за пределами этого набора не добавляются. "
    "У старых городских поездок сначала перепроверяются уже включённые места."
)


def _request_digest(request: TripRequest) -> str:
    raw = json.dumps(request.model_dump(mode="json"), ensure_ascii=False,
                     sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _resolved_center(value: dict, request: TripRequest) -> dict:
    if not isinstance(value, dict) or request.destination_kind != "city":
        raise ValueError(CONTEXT_ERROR)
    center = {key: deepcopy(value[key]) for key in
              ("name", "lat", "lon", "country_code", "country", "admin1") if key in value}
    code = center.get("country_code")
    if not isinstance(code, str) or len(code) != 2 or not code.isascii() or not code.isalpha():
        raise ValueError(CONTEXT_ERROR)
    center["country_code"] = code.lower()
    provider_country = center.get("country", "")
    if (not isinstance(provider_country, str) or len(provider_country) > 240
            or provider_country and not provider_country.isprintable()):
        raise ValueError(CONTEXT_ERROR)
    for key, bound in (("lat", 90), ("lon", 180)):
        # Check bounds before math.isfinite converts integers to float. An
        # arbitrarily large persisted integer must fail closed, not overflow.
        if (type(center.get(key)) not in (int, float)
                or not -bound <= center[key] <= bound or not math.isfinite(center[key])):
            raise ValueError(CONTEXT_ERROR)
    parts = [part.strip() for part in request.destination.split(",")]
    if (not 1 <= len(parts) <= 3 or not all(parts)
            or not isinstance(center.get("name"), str)
            or places._context_identity(center["name"]) != places._context_identity(parts[0])):
        raise ValueError(CONTEXT_ERROR)
    if len(parts) == 3 and places._context_identity(center.get("admin1", "")) != places._context_identity(parts[1]):
        raise ValueError(CONTEXT_ERROR)
    if len(parts) > 1:
        expected = places._country_identity(parts[-1])
        known_code = (expected[0] if len(expected) == 1 and len(expected[0]) == 2
                      and expected[0].isascii() and expected[0].isalpha() else "")
        # Match the city resolver's exact-country fallback. Do not invent a
        # code/translation for names outside the deliberately small alias map.
        if known_code:
            if known_code != center["country_code"]:
                raise ValueError(CONTEXT_ERROR)
        elif not provider_country or places._country_identity(provider_country) != expected:
            raise ValueError(CONTEXT_ERROR)
    return center


def build_city_context(request: TripRequest, candidates, resolved_city: dict, *,
                       pool_scope: str = "generation_pool") -> dict:
    """Serialize only a bounded, individually revalidated server pool."""
    if pool_scope not in {"generation_pool", "existing_visits_rechecked"}:
        raise ValueError(CONTEXT_ERROR)
    center = _resolved_center(resolved_city, request)
    if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= city_candidate_limit(request):
        raise ValueError(CONTEXT_ERROR)
    prepared, identifiers, queries = [], set(), set()
    for candidate in candidates:
        if not isinstance(candidate, CityCandidate):
            raise ValueError(CONTEXT_ERROR)
        record = _canonical_record(candidate.title, candidate.record, center)
        identity = places.map_record_identity(record)
        identifier = "-".join(map(str, identity)) if identity is not None else ""
        if (record is None or candidate.candidate_id != identifier
                or candidate.place_query != record["place_query"] or candidate.area != record["locality"]
                or candidate.title != record["place_query"].split(",", 1)[0]
                or identifier in identifiers or candidate.place_query in queries):
            raise ValueError(CONTEXT_ERROR)
        identifiers.add(identifier)
        queries.add(candidate.place_query)
        prepared.append(asdict(CityCandidate(identifier, candidate.title, candidate.place_query,
                                             candidate.area, record)))
    payload = {"version": 1, "request_sha256": _request_digest(request),
               "captured_at": datetime.now(timezone.utc).isoformat(),
               "pool_scope": pool_scope, "resolved_city": center, "candidates": prepared}
    if len(json.dumps(payload, ensure_ascii=False, allow_nan=False)) > 200_000:
        raise ValueError(CONTEXT_ERROR)
    return payload


def load_city_context(payload: dict, request: TripRequest) -> tuple[list[CityCandidate], dict]:
    """Fail closed on foreign, truncated, duplicate or forged snapshot data."""
    try:
        if (not isinstance(payload, dict) or set(payload) - {"presentation"} != {
                "version", "request_sha256", "captured_at", "pool_scope", "resolved_city", "candidates"}
                or type(payload["version"]) is not int or payload["version"] != 1
                or payload["request_sha256"] != _request_digest(request)
                or not isinstance(payload["captured_at"], str)
                or datetime.fromisoformat(payload["captured_at"]).tzinfo is None
                or not isinstance(payload["candidates"], list)
                or not 1 <= len(payload["candidates"]) <= city_candidate_limit(request)
                or len(json.dumps(payload, ensure_ascii=False, allow_nan=False)) > 200_000):
            raise ValueError(CONTEXT_ERROR)
        if "presentation" in payload:
            _validated_presentation(payload["presentation"], request)
        candidates = [CityCandidate(**entry) for entry in payload["candidates"]]
        checked = build_city_context(request, candidates, payload["resolved_city"],
                                     pool_scope=payload["pool_scope"])
        return [CityCandidate(**entry) for entry in checked["candidates"]], checked["resolved_city"]
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
        raise ValueError(CONTEXT_ERROR) from error


def _validated_presentation(value, request):
    """Presentation ownership only; hashes never authenticate visitor facts."""
    if (not isinstance(value, dict) or set(value) != {"version", "managed_tip_sha256"}
            or type(value["version"]) is not int or value["version"] != 1
            or not isinstance(value["managed_tip_sha256"], list)
            or len(value["managed_tip_sha256"]) > city_managed_tip_limit(request)
            or any(not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
                   for digest in value["managed_tip_sha256"])
            or len(set(value["managed_tip_sha256"])) != len(value["managed_tip_sha256"])):
        raise ValueError(CONTEXT_ERROR)
    return value["managed_tip_sha256"]


def city_presentation_tips(payload: dict, itinerary: Itinerary, candidates, request: TripRequest) -> tuple[str, ...]:
    """Read a server manifest and bind every hash to the saved original text.

    The caller must have loaded this server-owned context and checked its map
    records first. An AI/public request cannot supply or extend the manifest.
    Older contexts have a conservative exact-canonical recognition fallback.
    """
    if "presentation" not in payload:
        return recognize_city_managed_tips(itinerary, candidates, request)
    hashes = _validated_presentation(payload["presentation"], request)
    indexed = {}
    for value in itinerary.tips:
        indexed.setdefault(tip_sha256(value), []).append(value)
    if any(len(indexed.get(digest, ())) != 1 for digest in hashes):
        raise ValueError(CONTEXT_ERROR)
    recognized = set(hashes)
    return tuple(value for value in itinerary.tips if tip_sha256(value) in recognized)


def attach_city_presentation(payload: dict, itinerary: Itinerary, candidates, request: TripRequest) -> dict:
    """Attach only exact server-rendered tip hashes to a validated context copy."""
    checked, _ = load_city_context(payload, request)
    if ({value.candidate_id: value.place_query for value in checked}
            != {value.candidate_id: value.place_query for value in candidates}):
        raise ValueError(CONTEXT_ERROR)
    owned = recognize_city_managed_tips(itinerary, checked, request)
    hashes = list(dict.fromkeys(tip_sha256(value) for value in owned))
    if not hashes and "presentation" not in payload:
        return deepcopy(payload)  # A legacy itinerary without managed text is unchanged.
    presentation = {"version": 1, "managed_tip_sha256": hashes}
    _validated_presentation(presentation, request)
    result = deepcopy(payload)
    result["presentation"] = presentation
    if len(json.dumps(result, ensure_ascii=False, allow_nan=False)) > 200_000:
        raise ValueError(CONTEXT_ERROR)
    # Validate the saved-string crosscheck too, without authorizing any maps.
    city_presentation_tips(result, itinerary, checked, request)
    return result


def check_context_places(itinerary: Itinerary, candidates) -> None:
    """Bind every old visit to the exact observed title, location and provider."""
    records = {candidate.place_query: candidate.record for candidate in candidates}
    for day in itinerary.days:
        for item in day.items:
            if item.kind != "attraction":
                continue
            record = records.get(item.place_query)
            if (record is None or item.verified is not True
                    or item.lat != record.get("lat") or item.lon != record.get("lon")
                    or item.official_name != record.get("name")
                    or item.provider != (record.get("provider") or "osm")
                    or item.title != item.place_query.split(",", 1)[0]
                    or not places._area_matches_map_context(item.area, record)):
                raise ValueError(CONTEXT_ERROR)
            if item.provider == "2gis" and item.source_url != record.get("source_url"):
                raise ValueError(CONTEXT_ERROR)


def validate_city_edit_context(original: dict, returned: dict, request: TripRequest,
                               itinerary: Itinerary) -> dict:
    """Validate a returned edit snapshot; only presentation may change.

    Both callers retain an independent original and pass a copy to the editor.
    Exact JSON equality outside presentation preserves record evidence, capture
    metadata and request binding. A matching tip hash is presentation ownership,
    never authority to replace map records or introduce visitor facts.
    """
    try:
        if type(original) is not dict or type(returned) is not dict:
            raise ValueError(CONTEXT_ERROR)
        def evidence(value):
            return json.dumps({key: item for key, item in value.items() if key != "presentation"},
                              sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if evidence(original) != evidence(returned):
            raise ValueError(CONTEXT_ERROR)
        candidates, _ = load_city_context(returned, request)
        check_context_places(itinerary, candidates)
        city_presentation_tips(returned, itinerary, candidates, request)
        return deepcopy(returned)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError) as error:
        raise ValueError(CONTEXT_ERROR) from error


def recover_city_context(request: TripRequest, itinerary: Itinerary, *,
                         resolved_city=None, lookup=None) -> dict:
    """Bounded map-only recovery for old trips, before claiming paid AI quota.

    Only existing exact queries are searched. Existing coordinates/names must
    equal the independently returned observation. No candidate is invented or
    inferred from a coordinate URL. The trip-bound pool and 36 map-attempt cap
    still apply; a long legacy route may fail safely when lookup runs out.
    """
    from weather import find_city

    if itinerary.city_schedule is not True or itinerary.source_schedule:
        raise ValueError(CONTEXT_ERROR)
    visits = [item for day in itinerary.days for item in day.items if item.kind == "attraction"]
    if not 1 <= len(visits) <= city_candidate_limit(request) or len({item.place_query for item in visits}) != len(visits):
        raise ValueError(CONTEXT_ERROR)
    center = _resolved_center(resolved_city if resolved_city is not None else find_city(request.destination), request)
    lookup = lookup or _lookup
    budget = CityCandidateResult()
    candidates = []
    country = request.destination.split(",")[-1].strip() if "," in request.destination else center["country_code"].upper()
    for item in visits:
        record = lookup(item.title, country, center, budget)
        if record is None or record.get("place_query") != item.place_query:
            raise ValueError(CONTEXT_ERROR)
        identity = places.map_record_identity(record)
        if identity is None:
            raise ValueError(CONTEXT_ERROR)
        candidates.append(CityCandidate("-".join(map(str, identity)), item.title, item.place_query,
                                        item.area, deepcopy(record)))
    check_context_places(itinerary, candidates)
    return build_city_context(request, candidates, center, pool_scope="existing_visits_rechecked")


def prepare_stored_city_context(saved_request: dict, request: TripRequest, itinerary: Itinerary) -> dict:
    if CONTEXT_KEY in saved_request:
        payload = deepcopy(saved_request[CONTEXT_KEY])
    else:
        payload = recover_city_context(request, itinerary)
    candidates, _ = load_city_context(payload, request)
    check_context_places(itinerary, candidates)
    return payload
