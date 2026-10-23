"""Source-backed visit cards, separate from independently verified map facts.

Published facts, editorial planning estimates and unknown-price reserves remain
distinct. A card never establishes POI identity/coordinates or proves a booking.
A sourced access point for a loop is separate and still needs map proximity and
route verification before it can be used as a transport endpoint.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
import math
import re
from pathlib import Path
from typing import Any

from destination_name_hints import _alias, _https_source, _text
from regional_skeleton import VisitFact, VisitWindow
from schemas import TripRequest

VISIT_FACTS_FILE = Path(__file__).resolve().parent / "data" / "visit_facts.json"


def _amount(value: Any) -> bool:
    return type(value) in (int, float) and 0 <= value <= 1_000_000 and math.isfinite(value)


def _closed_weekdays_valid(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, dict) and set(item) == {"weekday", "occurrence"}
        and type(item["weekday"]) is int and 0 <= item["weekday"] <= 6
        and type(item["occurrence"]) is int and 1 <= item["occurrence"] <= 5
        for item in value
    )


def _route_access_valid(card: dict) -> bool:
    kind, point, source = card.get("route_kind", "point"), card.get("access_point"), card.get("access_point_source", "")
    anchor = card.get("map_anchor_point")
    if kind == "point":
        return point is None and source == "" and anchor is None
    if kind != "loop" or not isinstance(point, list) or len(point) != 2 or not _https_source(source):
        return False
    if not all(type(value) in (int, float) and -bound <= value <= bound and math.isfinite(value)
               for value, bound in zip(point, (90, 180))):
        return False
    if anchor is None:
        return True
    from places import distance_km
    return (isinstance(anchor, list) and len(anchor) == 2
            and all(type(value) in (int, float) and -bound <= value <= bound and math.isfinite(value)
                    for value, bound in zip(anchor, (90, 180)))
            and _amount(card.get("length_km")) and 0 < card["length_km"] <= 50
            and 2 * distance_km(*point, *anchor) <= card["length_km"])


def load_visit_cards(destination: str, *, path: Path | None = None) -> list[dict]:
    """An exact destination match; malformed/untrusted records fail closed."""
    try:
        data = json.loads((path or VISIT_FACTS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return []
    if not isinstance(data, dict) or data.get("version") != 1 or not isinstance(data.get("regions"), list):
        return []
    result, seen = [], set()
    for region in data["regions"]:
        if not isinstance(region, dict) or not isinstance(region.get("aliases"), list) or _alias(destination) not in {
            _alias(item) for item in region["aliases"] if isinstance(item, str)
        }:
            continue
        if not isinstance(region.get("visits"), list):
            continue
        for card in region["visits"]:
            if not isinstance(card, dict):
                continue
            card = {"metadata_checked_at": data.get("metadata_checked_at"), **card}
            queries = card.get("place_queries")
            if (not _text(card.get("id"), 80) or card["id"] in seen
                    or not isinstance(queries, list) or not queries
                    or not all(_text(query, 180) for query in queries)
                    or not _text(card.get("description"), 1500)
                    or type(card.get("duration_min")) is not int
                    or not 15 <= card["duration_min"] <= 600
                    or card.get("duration_basis") not in ("published", "planning_estimate")
                    or not _https_source(card.get("duration_source"))
                    or type(card.get("indoor")) is not bool
                    or not isinstance(card.get("hours"), list)
                    or not isinstance(card.get("sessions", []), list)
                    or not isinstance(card.get("closed_month_days", []), list)
                    or not _closed_weekdays_valid(card.get("closed_weekdays", []))
                    or not _route_access_valid(card)
                    or ("max_group_size" in card and (type(card["max_group_size"]) is not int
                                                       or not 1 <= card["max_group_size"] <= 100))
                    or card.get("currency", "RUB") != "RUB"
                    or ("visitor_note" in card and not _text(card["visitor_note"], 1000))
                    or any(not isinstance(card.get(key, []), list) or not all(_text(item, 1500) for item in card.get(key, []))
                           for key in ("interest_categories", "restrictions"))):
                continue
            if card["duration_basis"] == "planning_estimate" and not _text(card.get("basis_note"), 1000):
                continue
            try:
                date.fromisoformat(card["metadata_checked_at"])
            except (KeyError, TypeError, ValueError):
                continue
            if any(value is not None and not _amount(value) for value in (
                card.get("adult_price"), card.get("child_price"), card.get("group_min_price"),
                card.get("planning_reserve_per_person"),
            )):
                continue
            seen.add(card["id"])
            result.append(card)
    return result[:24]


def _windows(card: dict, request: TripRequest) -> tuple[VisitWindow, ...]:
    result = []
    first = date.fromisoformat(request.start_date)
    for offset in range(request.days_count):
        current = first + timedelta(days=offset)
        if current.strftime("%m-%d") in card.get("closed_month_days", []):
            continue
        if any(current.weekday() == item["weekday"] and (current.day - 1) // 7 + 1 == item["occurrence"]
               for item in card.get("closed_weekdays", [])):
            continue
        for hours in card["hours"]:
            if not isinstance(hours, dict):
                continue
            exact = hours.get("dates")
            if exact is not None and (not isinstance(exact, list) or current.isoformat() not in exact):
                continue
            weekdays, months = hours.get("weekdays", list(range(7))), hours.get("months", list(range(1, 13)))
            if (not isinstance(weekdays, list) or not isinstance(months, list)
                    or any(type(value) is not int or not 0 <= value <= 6 for value in weekdays)
                    or any(type(value) is not int or not 1 <= value <= 12 for value in months)
                    or current.weekday() not in weekdays or current.month not in months):
                continue
            start, end = hours.get("start"), hours.get("end")
            if not all(isinstance(value, str) and re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value) for value in (start, end)):
                continue
            try:
                values = [int(value[:2]) * 60 + int(value[3:]) for value in (start, end)]
                if not (0 <= values[0] < values[1] <= 1440) or any(int(value[3:]) > 59 for value in (start, end)):
                    continue
            except ValueError:
                continue
            source = card.get("hours_source", "")
            if not _https_source(source):
                continue
            result.append(VisitWindow(current.isoformat(), start, end, evidence="source", source=source,
                                      note=card.get("hours_note", "Опубликованный график; актуальность перед визитом уточнить.")))
        for session in card.get("sessions", []):
            if not isinstance(session, dict) or not isinstance(session.get("weekdays"), list):
                continue
            start = session.get("start", "")
            if (current.weekday() not in session["weekdays"]
                    or not isinstance(start, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", start)
                    or not _https_source(card.get("hours_source"))):
                continue
            end = int(start[:2]) * 60 + int(start[3:]) + card["duration_min"]
            if end >= 1440:
                continue
            result.append(VisitWindow(current.isoformat(), start, f"{end//60:02d}:{end%60:02d}",
                                      evidence="estimate", source=card["hours_source"],
                                      note="Начало сеанса опубликовано; конец — плановая оценка по длительности карточки, не часы работы."))
    return tuple(result)


def visit_fact_from_card(request: TripRequest, card: dict, place_query: str) -> VisitFact | None:
    """Project a validated source card onto a request, without claiming map facts.

    The caller loads cards with ``load_visit_cards``. This pure projection is
    also useful for candidate priority; only ``visit_facts_for_grounding`` binds
    the returned content to an independently confirmed map identity.
    """
    if (request.destination_kind != "region" or request.currency != "RUB"
            or place_query not in card["place_queries"]
            or request.adults + len(request.children_ages) > card.get("max_group_size", 100)):
        return None
    age = (date.fromisoformat(request.end_date) - date.fromisoformat(card["metadata_checked_at"])).days
    if not 0 <= age <= 90:
        return None
    adult, minimum = card.get("adult_price"), card.get("group_min_price")
    # A single child price cannot represent free/adult age bands. Without
    # structured bands use an explicit reserve, never parse prose as a tariff.
    known = adult is not None and not request.children_ages
    group_cost = max(minimum or 0, adult * request.adults) if known else None
    reserve = card.get("planning_reserve_per_person")
    reserve = reserve * (request.adults + len(request.children_ages)) if group_cost is None and reserve is not None else None
    if group_cost is not None and not _https_source(card.get("price_source")):
        return None
    return VisitFact(
        visit_id=card["id"], place_query=place_query, description=card["description"],
        duration_min=card["duration_min"], duration_basis=card["duration_basis"],
        duration_source=card["duration_source"], basis_note=card.get("basis_note", ""),
        indoor=card["indoor"], date_windows=_windows(card, request), group_cost=group_cost, currency="RUB",
        price_source=card.get("price_source", ""), planning_reserve=reserve,
        price_note=card.get("price_note", "Резерв планирования; точная цена неизвестна." if group_cost is None
                            else "Цены по опубликованному источнику, перед поездкой уточнить."),
        children_allowed=card.get("children_allowed"), minimum_age=card.get("minimum_age"),
        interest_categories=tuple(card.get("interest_categories", [])),
        restrictions=tuple(card.get("restrictions", [])), kids_note=card.get("kids_note", ""),
        length_km=card.get("length_km"), visitor_note=card.get("visitor_note", ""),
        route_kind=card.get("route_kind", "point"),
        access_point=tuple(card["access_point"]) if card.get("access_point") is not None else None,
        access_point_source=card.get("access_point_source", ""),
        map_anchor_point=tuple(card["map_anchor_point"]) if card.get("map_anchor_point") is not None else None,
    )


def visit_facts_for_grounding(request: TripRequest, grounding, *, path: Path | None = None) -> list[VisitFact]:
    """Bind exact card query aliases only to existing grounded POI identities.

    No name-only, proximity-only, currency conversion or invented opening hours.
    Freshness is bounded by 90 days from research to the requested final date.
    """
    if request.destination_kind != "region" or request.currency != "RUB":
        return []
    hints, ambiguous = {}, set()
    for item in grounding.confirmed_hints:
        if not isinstance(item, dict) or not isinstance(item.get("place_query"), str):
            continue
        query = item["place_query"]
        if query in hints:
            ambiguous.add(query)
        hints[query] = item
    result = []
    for card in load_visit_cards(request.destination, path=path):
        matches = [query for query in card["place_queries"] if query in hints]
        if len(matches) != 1 or matches[0] in ambiguous:
            continue  # Ambiguous identity is not silently resolved by source order.
        fact = visit_fact_from_card(request, card, matches[0])
        if fact is not None:
            result.append(fact)
    return result
