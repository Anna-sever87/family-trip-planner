"""Pure, bounded planning from supplied visit cards and a directed travel graph.

``ready`` means a structurally feasible *plan*, not a booking or proof that a
venue operates on a particular date. Estimates and budget reserves remain
explicit. Coordinates never manufacture a transport connection or its time.
No network, cache, database, AI, or canonical itinerary mutation happens here.
The caller must still run the normal final itinerary validators.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
import math
import re
from typing import Any, Iterable, Literal, Mapping
from urllib.parse import urlsplit

from places import compare_map_identities, distance_km, map_record_identity, same_map_place
from pricing import minimum_meal_cost
from rules import attraction_limit, program_depth_target
from schemas import INTEREST_CATEGORIES, TripRequest


@dataclass(frozen=True)
class VisitWindow:
    """A supplied date-specific planning window, never inferred office hours."""

    date: str
    start_time: str
    end_time: str
    evidence: Literal["source", "estimate"] = "estimate"
    source: str = ""
    note: str = ""


@dataclass(frozen=True)
class VisitFact:
    visit_id: str
    place_query: str
    description: str
    duration_min: int | None
    duration_basis: Literal["published", "planning_estimate"]
    duration_source: str = ""
    basis_note: str = ""
    indoor: bool | None = None
    date_windows: tuple[VisitWindow, ...] = ()
    group_cost: float | None = None
    currency: str = "RUB"
    price_source: str = ""
    planning_reserve: float | None = None
    price_note: str = ""
    children_allowed: bool | None = None
    minimum_age: int | None = None
    interest_categories: tuple[str, ...] = ()
    restrictions: tuple[str, ...] = ()
    kids_note: str = ""
    length_km: float | None = None
    visitor_note: str = ""
    # Only a source-confirmed closed loop can use an entry/return point.
    # This is routing metadata, never a replacement for grounded POI identity.
    route_kind: Literal["point", "loop"] = "point"
    access_point: tuple[float, float] | None = None  # latitude, longitude
    access_point_source: str = ""
    # A named summit/lake may be far from the trail entrance. The same reviewed
    # route source must locate that named POI as well; routing compares this
    # anchor to the independently geocoded POI, never to an AI coordinate.
    map_anchor_point: tuple[float, float] | None = None


@dataclass(frozen=True)
class TravelFact:
    """Directed edge: base_key or visit_id -> base_key or visit_id.

    For a linear visit the incoming edge reaches its real start and the outgoing
    edge leaves its real finish. The caller supplies these connections; this
    module does not assume that a trail is circular or reverse an edge.
    Empty date_windows means the supplied edge applies to all requested dates.
    """

    from_key: str
    to_key: str
    minutes: int
    mode: str
    evidence: Literal["source", "routing", "estimate"]
    source: str = ""
    note: str = ""
    group_cost: float | None = None
    planning_reserve: float | None = None
    currency: str = "RUB"
    date_windows: tuple[VisitWindow, ...] = ()
    price_source: str = ""
    price_note: str = ""
    restrictions: tuple[str, ...] = ()
    depth_credit_min: int = 0


@dataclass(frozen=True)
class SkeletonSlot:
    kind: Literal["attraction", "meal", "rest", "travel"]
    start_time: str
    end_time: str
    duration_min: int
    title: str
    description: str
    visit_id: str = ""
    place_query: str = ""
    group_cost: float | None = None
    planning_reserve: float = 0.0
    visit_fact: VisitFact | None = None
    travel_fact: TravelFact | None = None

    @property
    def budget_amount(self) -> float:
        return (self.group_cost or 0.0) + self.planning_reserve


@dataclass(frozen=True)
class SkeletonDay:
    day: int
    date: str
    base_key: str
    base_query: str
    slots: tuple[SkeletonSlot, ...]
    visit_ids: tuple[str, ...]
    visit_minutes: int
    travel_minutes: int
    depth_credit_min: int
    known_cost_total: float
    planning_reserve_total: float
    is_free_day: bool = False


@dataclass(frozen=True)
class DateCoverage:
    date: str
    unique_candidates: int
    available_visit_minutes: int


@dataclass(frozen=True)
class SkeletonCoverage:
    """Evidence about this supplied pool, never about the entire destination.

    Visit minutes exclude travel; they alone are not an infeasibility proof.
    Per-date counts require a full visit window but do not promise routing,
    meals, weather suitability or a feasible aggregate budget.
    """

    requested_days: int
    unique_candidates: int
    target_minutes_per_day: int
    target_total_minutes: int
    available_visit_minutes: int
    by_date: tuple[DateCoverage, ...] = ()
    exclusions: tuple[tuple[str, str], ...] = ()
    missing_facts: tuple[str, ...] = ()
    reason: str = ""
    first_day_max_minutes: int | None = None


@dataclass(frozen=True)
class SkeletonResult:
    status: Literal["ready", "limited_program", "needs_facts", "insufficient_content", "search_limit"]
    days: tuple[SkeletonDay, ...] = ()
    problems: tuple[str, ...] = ()
    missing_facts: tuple[str, ...] = ()
    excluded_visit_ids: tuple[str, ...] = ()
    known_cost_total: float = 0.0
    planning_reserve_total: float = 0.0
    search_states: int = 0
    coverage: SkeletonCoverage | None = None
    search_complete: bool = True


def _minutes(value: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", value):
        raise ValueError("Время должно быть в формате ЧЧ:ММ")
    hour, minute = map(int, value.split(":"))
    return hour * 60 + minute


def _clock(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and not any(ord(char) < 32 for char in value)


def _number(value: Any) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and 0 <= value <= 1_000_000_000 and math.isfinite(value))


def _windows_valid(windows: Iterable[VisitWindow]) -> bool:
    try:
        for window in windows:
            if not isinstance(window, VisitWindow) or date.fromisoformat(window.date).isoformat() != window.date:
                return False
            if _minutes(window.start_time) >= _minutes(window.end_time):
                return False
            if window.evidence == "source":
                if not _text(window.source):
                    return False
            elif window.evidence == "estimate":
                if not _text(window.note):
                    return False
            else:
                return False
        return True
    except (TypeError, ValueError, AttributeError):
        return False


def _price_valid(fact: VisitFact | TravelFact, currency: str) -> bool:
    if fact.currency != currency:
        return False  # No implicit exchange rate, nor price in another currency.
    if fact.group_cost is not None:
        return (_number(fact.group_cost) and fact.planning_reserve is None
                and _text(fact.price_source))
    return _number(fact.planning_reserve) and _text(fact.price_note)


def _cost(fact: VisitFact | TravelFact) -> tuple[float, float]:
    return (float(fact.group_cost) if fact.group_cost is not None else 0.0,
            float(fact.planning_reserve) if fact.planning_reserve is not None else 0.0)


def _strings(values: Any) -> bool:
    return isinstance(values, (tuple, list)) and all(_text(value) for value in values)


def _visit_access_valid(fact: VisitFact) -> bool:
    """Structural source-entry guard; map proximity is checked by routing."""
    if fact.route_kind == "point":
        return fact.access_point is None and fact.access_point_source == "" and fact.map_anchor_point is None
    if fact.route_kind != "loop":
        return False  # Linear routes need separate start/finish support.
    point, source = fact.access_point, fact.access_point_source
    try:
        if (not isinstance(point, tuple) or len(point) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in point)
                or not -90 <= point[0] <= 90 or not -180 <= point[1] <= 180
                or not _text(source) or len(source) > 2048 or any(char.isspace() for char in source)):
            return False
        parsed = urlsplit(source)
        if not (parsed.scheme == "https" and bool(parsed.hostname)
                and parsed.username is None and parsed.password is None
                and (parsed.port is None or 1 <= parsed.port <= 65535)):
            return False
        anchor = fact.map_anchor_point
        if anchor is None:
            return True  # Existing loops keep the routing layer's 250-m cap.
        if (not isinstance(anchor, tuple) or len(anchor) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in anchor)
                or not -90 <= anchor[0] <= 90 or not -180 <= anchor[1] <= 180
                or not _number(fact.length_km) or not 0 < fact.length_km <= 50):
            return False
        # A closed visit must at least reach the named POI and return. This
        # sanity bound does not manufacture a trail or count its length twice.
        return 2 * distance_km(*point, *anchor) <= fact.length_km
    except (TypeError, ValueError, OverflowError):
        return False


def _visit_problem(fact: VisitFact, request: TripRequest) -> str | None:
    if not _visit_access_valid(fact):
        return "route_access_point"
    if not _text(fact.description):
        return "description"
    if type(fact.duration_min) is not int or not 1 <= fact.duration_min <= 600:
        return "duration_min"
    if fact.duration_basis == "published":
        if not _text(fact.duration_source):
            return "duration_source"
    elif fact.duration_basis == "planning_estimate":
        if not _text(fact.basis_note):
            return "basis_note"
    else:
        return "duration_basis"
    if type(fact.indoor) is not bool:
        return "indoor"
    if not fact.date_windows or not _windows_valid(fact.date_windows):
        return "date_windows"
    if not _price_valid(fact, request.currency):
        return "price_or_explicit_reserve"
    if fact.length_km is not None and (not _number(fact.length_km) or fact.length_km <= 0):
        return "length_km"
    if not _strings(fact.restrictions):
        return "restrictions"
    if fact.children_allowed is not None and type(fact.children_allowed) is not bool:
        return "children_allowed"
    if fact.minimum_age is not None and (type(fact.minimum_age) is not int or not 0 <= fact.minimum_age <= 120):
        return "minimum_age"
    if request.children_ages:
        if fact.children_allowed is not True:
            return "children_not_eligible_or_unknown"
        if fact.minimum_age is not None and min(request.children_ages) < fact.minimum_age:
            return "minimum_age"
    elif fact.minimum_age is not None and fact.minimum_age > 18:
        return "adult_age_unknown"
    if (not _strings(fact.interest_categories)
            or any(value not in INTEREST_CATEGORIES for value in fact.interest_categories)):
        return "interest_categories"
    if request.interest_categories and not set(request.interest_categories).intersection(fact.interest_categories):
        return "interests_not_matched"
    return None


def _travel_problem(fact: TravelFact, request: TripRequest) -> str | None:
    if type(fact.minutes) is not int or not 0 <= fact.minutes <= 1440 or not _text(fact.mode):
        return "minutes_or_mode"
    if fact.evidence in {"source", "routing"}:
        if not _text(fact.source):
            return "source"
    elif fact.evidence == "estimate":
        if not _text(fact.note):
            return "estimate_note"
    else:
        return "evidence"
    if (type(fact.depth_credit_min) is not int or not 0 <= fact.depth_credit_min <= fact.minutes
            or (fact.evidence == "estimate" and fact.depth_credit_min != 0)):
        return "depth_credit_min"
    if not _price_valid(fact, request.currency):
        return "price_or_explicit_reserve"
    if not _windows_valid(fact.date_windows) or not _strings(fact.restrictions):
        return "date_windows_or_restrictions"
    # An appointment/permit is a condition of a preliminary plan, not a reason
    # to pretend it is booked. Explicitly uncertain water/ice access is different:
    # the existing regional validator forbids relying on that connection.
    from destination_scope import _has_unconfirmed_water_transfer
    if _has_unconfirmed_water_transfer(" ".join((fact.mode, fact.note, *fact.restrictions))):
        return "unconfirmed_water_transfer"
    return None


def _known_visit_exclusion(fact: VisitFact, problem: str) -> bool:
    """A known mismatch is not missing evidence and must remain diagnostic."""
    return (problem in {"interests_not_matched", "minimum_age"}
            or (problem == "children_not_eligible_or_unknown" and fact.children_allowed is False))


def _known_visit_rejection(fact: VisitFact, request: TripRequest) -> str | None:
    # A documented exclusion remains an exclusion even when some unrelated
    # evidence (for example opening hours) is also absent from that card.
    if request.children_ages:
        if fact.children_allowed is False:
            return "children_not_eligible_or_unknown"
        if (type(fact.minimum_age) is int and 0 <= fact.minimum_age <= 120
                and min(request.children_ages) < fact.minimum_age):
            return "minimum_age"
    if (_strings(fact.interest_categories)
            and all(value in INTEREST_CATEGORIES for value in fact.interest_categories)
            and request.interest_categories
            and not set(request.interest_categories).intersection(fact.interest_categories)):
        return "interests_not_matched"
    return None


def _coordinates(record: Mapping[str, Any]) -> bool:
    for key, bound in (("lat", 90), ("lon", 180)):
        value = record.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or abs(value) > bound or not math.isfinite(value):
            return False
    return True


def _grounded_pools(grounding: Any) -> tuple[dict[str, dict], dict[str, dict]]:
    """Accept the existing map-confirmed result; never geocode or enlarge it."""
    bases: dict[str, dict] = {}
    hints: dict[str, dict] = {}
    for raw in grounding.base_facts:
        if (not isinstance(raw, dict) or not _text(raw.get("base_key"))
                or not _text(raw.get("base_query")) or not _coordinates(raw)):
            raise ValueError("Некорректная подтверждённая база")
        if raw["base_key"] in bases:
            raise ValueError("Повторный ключ базы")
        bases[raw["base_key"]] = dict(raw)
    for raw in grounding.confirmed_hints:
        if (not isinstance(raw, dict) or not _text(raw.get("place_query"))
                or not _text(raw.get("title")) or not _coordinates(raw)):
            raise ValueError("Некорректное подтверждённое место")
        allowed = raw.get("allowed_base_keys", [raw.get("base_key")])
        if not isinstance(allowed, (list, tuple)) or not allowed or any(key not in bases for key in allowed):
            raise ValueError("Не подтверждена совместимость места с базой")
        if raw["place_query"] in hints:
            raise ValueError("Повторный запрос подтверждённого места")
        hints[raw["place_query"]] = {**raw, "allowed_base_keys": tuple(allowed)}
    # Coordinates are used ONLY for the existing 30 m identity safeguard, never
    # to infer transport, duration, accessibility, or an allowed base pairing.
    queries = list(hints)
    identities = list(range(len(queries)))
    map_identities = [map_record_identity(hints[query]) for query in queries]
    for i, query in enumerate(queries):
        first = hints[query]
        for j in range(i):
            second = hints[queries[j]]
            # Grounded IDs take precedence over proximity. An unknown-ID point
            # must not transitively bridge two distinct authoritative objects.
            if same_map_place(first, second, first_identity=map_identities[i], second_identity=map_identities[j]):
                old, new = identities[i], identities[j]
                if any(compare_map_identities(map_identities[a], map_identities[b]) is False
                       for a, group_a in enumerate(identities) if group_a == old
                       for b, group_b in enumerate(identities) if group_b == new):
                    continue
                identities = [new if value == old else value for value in identities]
    for query, identity in zip(queries, identities):
        hints[query]["_identity"] = identity
    return bases, hints


def assess_visit_coverage(request: TripRequest, grounding: Any,
                          visit_facts: Iterable[VisitFact]) -> SkeletonCoverage:
    """Pure preflight using the same eligibility and identity rules as search."""
    if request.destination_kind != "region" or not 1 <= request.days_count <= 14:
        raise ValueError("Нужен региональный запрос от 1 до 14 дней")
    bases, hints = _grounded_pools(grounding)
    if request.stay_area.strip():
        wanted = " ".join(request.stay_area.casefold().split())
        bases = {key: base for key, base in bases.items()
                 if wanted in {" ".join(str(value).casefold().split()) for value in (
                     base["base_query"], base["base_query"].split(",", 1)[0],
                     *base.get("map_address", {}).values())}}
    first = date.fromisoformat(request.start_date)
    dates = tuple((first + timedelta(days=offset)).isoformat() for offset in range(request.days_count))
    start, end = _minutes(request.start_time), _minutes(request.end_time)
    if start >= end:
        raise ValueError("Некорректное окно дня")
    target = program_depth_target(request) or 0
    known, missing, exclusions, card_queries = [], [], [], set()
    for fact in visit_facts:
        card_queries.add(fact.place_query)
        problem = _known_visit_rejection(fact, request) or (
            "map_confirmation" if fact.place_query not in hints else _visit_problem(fact, request))
        if problem is None:
            known.append(fact)
        elif _known_visit_exclusion(fact, problem):
            exclusions.append((fact.visit_id, problem))
        else:
            missing.append(f"visit:{fact.visit_id}:{problem}")
    # Grounded names without a card are evidence gaps, not rejected interests.
    missing.extend(f"visit:{query}:card" for query in sorted(set(hints) - card_queries))
    identities, per_date = {}, []
    for on_date in dates:
        available = {}
        for fact in known:
            hint = hints[fact.place_query]
            if not set(hint["allowed_base_keys"]).intersection(bases):
                continue
            at = _fit(start, fact.duration_min, fact.date_windows, on_date)
            if at is None or at + fact.duration_min > end:
                continue
            key = hint["_identity"]
            available[key] = max(available.get(key, 0), fact.duration_min)
            identities[key] = max(identities.get(key, 0), fact.duration_min)
        per_date.append(DateCoverage(on_date, len(available), sum(available.values())))
    reason = ("dates_without_visits" if any(not row.unique_candidates for row in per_date)
              else "unique_candidates" if len(identities) < request.days_count else "")
    return SkeletonCoverage(request.days_count, len(identities), target, target * request.days_count,
                            sum(identities.values()), tuple(per_date), tuple(sorted(exclusions)),
                            tuple(sorted(missing)), reason)


def _first_day_capacity(coverage: SkeletonCoverage, request: TripRequest, bases: dict, hints: dict,
                        facts: dict[str, VisitFact], graph: dict) -> SkeletonCoverage:
    """Prove only the N unique visits / N days single-visit first-day case.

    Unknown potential return connections prevent this proof. Travel windows,
    weather and price are deliberately ignored: even their optimistic upper
    bound must fail the depth target before the explanation can be emitted.
    """
    if (coverage.reason or coverage.unique_candidates != request.days_count
            or coverage.target_minutes_per_day <= 0):
        return coverage
    dates = {row.date for row in coverage.by_date}
    search_identities = {hints[fact.place_query]["_identity"] for fact in facts.values()
                         if any(window.date in dates for window in fact.date_windows)}
    if len(search_identities) != request.days_count:
        return coverage  # The actual search must reserve exactly one per day.
    start, end = _minutes(request.start_time), _minutes(request.end_time)
    maxima = []
    for fact in facts.values():
        at = _fit(start, fact.duration_min, fact.date_windows, request.start_date)
        if at is None or at + fact.duration_min > end:
            continue
        for base in hints[fact.place_query]["allowed_base_keys"]:
            if base not in bases:
                continue
            outward, backward = graph.get((base, fact.visit_id)), graph.get((fact.visit_id, base))
            if outward is None or backward is None:
                missing = set(coverage.missing_facts)
                if outward is None:
                    missing.add(f"travel:{base}->{fact.visit_id}")
                if backward is None:
                    missing.add(f"travel:{fact.visit_id}->{base}")
                return replace(coverage, missing_facts=tuple(sorted(missing)))
            maxima.append(fact.duration_min + outward.depth_credit_min + backward.depth_credit_min)
    if maxima and max(maxima) < coverage.target_minutes_per_day:
        return replace(coverage, reason="first_day_depth_capacity", first_day_max_minutes=max(maxima))
    return coverage


class _SearchLimit(Exception):
    pass


class _Counter:
    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0

    def tick(self) -> None:
        if self.used >= self.limit:
            raise _SearchLimit
        self.used += 1


def _fit(current: int, duration: int, windows: tuple[VisitWindow, ...], on_date: str) -> int | None:
    if not windows:
        return current
    starts = [max(current, _minutes(window.start_time)) for window in windows
              if window.date == on_date
              and max(current, _minutes(window.start_time)) + duration <= _minutes(window.end_time)]
    return min(starts) if starts else None


def _slot(fact: VisitFact | TravelFact, at: int, hints: dict[str, dict]) -> SkeletonSlot:
    if isinstance(fact, VisitFact):
        return SkeletonSlot("attraction", _clock(at), _clock(at + fact.duration_min), fact.duration_min,
                            hints[fact.place_query]["title"], fact.description,
                            visit_id=fact.visit_id, place_query=fact.place_query,
                            group_cost=fact.group_cost, planning_reserve=fact.planning_reserve or 0.0,
                            visit_fact=fact)
    return SkeletonSlot("travel", _clock(at), _clock(at + fact.minutes), fact.minutes,
                        fact.mode, fact.note or f"Переезд; источник: {fact.source}",
                        group_cost=fact.group_cost, planning_reserve=fact.planning_reserve or 0.0,
                        travel_fact=fact)


def _timeline(
    steps: tuple[VisitFact | TravelFact, ...], on_date: str, start: int, end: int,
    meal_minutes: int, rest_minutes: int, meal_reserve: float, hints: dict[str, dict], counter: _Counter,
) -> tuple[SkeletonSlot, ...] | None:
    """Insert genuine pauses between complete visits/edges, never inside them.

    Try finite placements near ordinary lunch time first, then other feasible
    positions. A published continuous excursion is never sliced into fake POIs.
    """
    boundaries = [start]
    for step in steps:
        boundaries.append(boundaries[-1] + (step.duration_min if isinstance(step, VisitFact) else step.minutes))
    layouts = sorted(((meal, rest) for meal in range(len(steps) + 1) for rest in range(len(steps) + 1)),
                     key=lambda pair: (abs(boundaries[pair[0]] - 13 * 60)
                                       + abs(boundaries[pair[1]] - 14 * 60 - 30)
                                       + (120 if pair[1] < pair[0] else 0), pair))
    for meal_position, rest_position in layouts:
        for prefer_lunch_time in (True, False):
            counter.tick()
            current, slots, valid = start, [], True
            for position in range(len(steps) + 1):
                if position == meal_position:
                    at = max(current, min(13 * 60, end - meal_minutes)) if prefer_lunch_time else current
                    # A full ordinary lunch window must not silently turn into
                    # an evening meal. Partial/late requested days retain their
                    # existing flexibility rather than acquiring extra hours.
                    if start <= 12 * 60 and end >= 15 * 60:
                        at = max(at, 12 * 60)
                        if at + meal_minutes > 15 * 60:
                            valid = False
                            break
                    slots.append(SkeletonSlot("meal", _clock(at), _clock(at + meal_minutes), meal_minutes,
                                              "Обед", "Плановая пауза на обед; сумма — резерв бюджета, не цена выбранного кафе.",
                                              planning_reserve=meal_reserve))
                    current = at + meal_minutes
                if position == rest_position:
                    slots.append(SkeletonSlot("rest", _clock(current), _clock(current + rest_minutes), rest_minutes,
                                              "Отдых", "Спокойный перерыв для отдыха и восстановления.", group_cost=0.0))
                    current += rest_minutes
                if position < len(steps):
                    step = steps[position]
                    duration = step.duration_min if isinstance(step, VisitFact) else step.minutes
                    at = _fit(current, duration, step.date_windows, on_date)
                    if at is None:
                        valid = False
                        break
                    current = at + duration
                    slots.append(_slot(step, at, hints))
                if current > end:
                    valid = False
                    break
            if valid:
                return tuple(slots)
    return None


def build_regional_skeleton(
    request: TripRequest, grounding: Any, visit_facts: Iterable[VisitFact], *,
    travel_facts: Iterable[TravelFact] = (), meal_minutes: int = 60,
    rest_minutes: int | None = None, max_search_states: int = 20000,
    rainy_dates: Iterable[str] = (), allow_free_days: bool = False,
) -> SkeletonResult:
    """Find a complete, unique, budgeted local plan, or return an atomic refusal.

    Only explicit cards and directed edges are used; estimates retain their
    provenance. Source/routing ``depth_credit_min`` is supplied by the caller,
    never inferred here; estimate travel gets no programme-depth credit. The
    bounded, deterministic search prefers keeping the same base, meaningful
    visit groups and shorter supplied connections. It is not a global optimiser.
    By default a failed/limited search returns no partial calendar. Explicit
    ``allow_free_days`` may instead return a complete ``limited_program``
    calendar containing genuine full programme days and marked free days.
    Free days keep a confirmed base and meal/rest reserves, never invented
    visits or hidden transfers. All searches share one state budget; a bounded
    result does not claim that the number of programme days is optimal.
    """
    if request.destination_kind != "region":
        raise ValueError("Планировщик предназначен для регионального запроса")
    if not 1 <= request.days_count <= 14:
        raise ValueError("Нужны даты поездки от 1 до 14 дней")
    start, end = _minutes(request.start_time), _minutes(request.end_time)
    if start >= end or type(meal_minutes) is not int or not 15 <= meal_minutes <= 180:
        raise ValueError("Некорректное окно дня или длительность обеда")
    if type(max_search_states) is not int or not 1 <= max_search_states <= 1_000_000:
        raise ValueError("Некорректный предел поиска")
    if type(allow_free_days) is not bool:
        raise ValueError("Режим свободных дней должен быть логическим значением")
    youngest = min(request.children_ages) if request.children_ages else None
    minimum_rest = 60 if youngest is not None and youngest < 7 else 15
    rest_minutes = minimum_rest if rest_minutes is None else rest_minutes
    if type(rest_minutes) is not int or not minimum_rest <= rest_minutes <= 180:
        raise ValueError("Длительность отдыха не соответствует составу группы")
    limit = attraction_limit(youngest) if youngest is not None else 5
    if request.pace.strip().casefold() == "спокойный":
        limit = max(1, limit - 1)
    meal_reserve = minimum_meal_cost(request)
    target = program_depth_target(request) or 0
    first = date.fromisoformat(request.start_date)
    dates = tuple((first + timedelta(days=offset)).isoformat() for offset in range(request.days_count))
    rainy = set(rainy_dates)
    bases, hints = _grounded_pools(grounding)
    if not bases or not hints:
        return SkeletonResult("needs_facts", problems=("Нет подтверждённых баз или мест.",))
    if request.stay_area.strip():
        wanted = " ".join(request.stay_area.casefold().split())
        bases = {key: base for key, base in bases.items()
                 if wanted in {" ".join(str(value).casefold().split()) for value in (
                     base["base_query"], base["base_query"].split(",", 1)[0],
                     *base.get("map_address", {}).values())}}
        if not bases:
            return SkeletonResult("needs_facts", problems=("Нет подтверждённой базы в указанном месте проживания.",))
    missing: set[str] = set()
    excluded: set[str] = set()
    facts: dict[str, VisitFact] = {}
    all_visit_ids: set[str] = set()
    visit_facts = tuple(visit_facts)
    for fact in visit_facts:
        if not isinstance(fact, VisitFact) or not _text(fact.visit_id) or not _text(fact.place_query):
            raise ValueError("Нужны карточки VisitFact с непустыми точными идентификаторами")
        if fact.visit_id in all_visit_ids or fact.visit_id in bases:
            raise ValueError("Повторный/конфликтующий visit_id")
        all_visit_ids.add(fact.visit_id)
        problem = _known_visit_rejection(fact, request) or (
            "map_confirmation" if fact.place_query not in hints else _visit_problem(fact, request))
        if problem is not None:
            excluded.add(fact.visit_id)
            if not _known_visit_exclusion(fact, problem):
                missing.add(f"visit:{fact.visit_id}:{problem}")
        else:
            facts[fact.visit_id] = fact
    coverage = assess_visit_coverage(request, grounding, visit_facts)
    missing.update(coverage.missing_facts)
    graph: dict[tuple[str, str], TravelFact] = {}
    all_nodes = all_visit_ids | set(bases)
    for edge in travel_facts:
        if not isinstance(edge, TravelFact) or not _text(edge.from_key) or not _text(edge.to_key):
            raise ValueError("Нужны направленные связи TravelFact")
        pair = edge.from_key, edge.to_key
        if pair in graph:
            raise ValueError("Для пары узлов нужна одна явно выбранная связь")
        if edge.from_key not in all_nodes or edge.to_key not in all_nodes or edge.from_key == edge.to_key:
            missing.add(f"travel:{edge.from_key}->{edge.to_key}:unknown_or_same_node")
            continue
        problem = _travel_problem(edge, request)
        if problem:
            missing.add(f"travel:{edge.from_key}->{edge.to_key}:{problem}")
        else:
            graph[pair] = edge
    counter = _Counter(max_search_states)
    absent_edges: set[str] = set()
    eligible = {on_date: tuple(fact for fact in facts.values() if any(window.date == on_date for window in fact.date_windows))
                for on_date in dates}

    def identity(fact: VisitFact) -> int:
        return hints[fact.place_query]["_identity"]

    def edge_between(origin: str, destination: str) -> TravelFact | None:
        edge = graph.get((origin, destination))
        if edge is None:
            absent_edges.add(f"travel:{origin}->{destination}")
        return edge

    def day_options(index: int, used: frozenset[int], previous_base: str | None, remaining_budget: float,
                    required_active_days: int | None = None):
        on_date = dates[index]
        remaining_dates = dates[index:]
        available = [fact for fact in eligible[on_date] if identity(fact) not in used]
        remaining = {identity(fact) for day in remaining_dates for fact in eligible[day]} - used
        # Every later active day needs a distinct real place. Free days need
        # no POI, but still reserve their ordinary meal below.
        needed = len(remaining_dates) if required_active_days is None else required_active_days
        selected_limit = min(limit, len(remaining) - (needed - 1))
        future_meals = meal_reserve * (len(remaining_dates) - 1)
        for base_key in sorted(bases, key=lambda key: (key != previous_base, key)):
            initial: tuple[TravelFact, ...] = ()
            if previous_base is not None and previous_base != base_key:
                transfer = edge_between(previous_base, base_key)
                if transfer is None:
                    continue
                initial = (transfer,)
            candidates = [fact for fact in available if base_key in hints[fact.place_query]["allowed_base_keys"]]

            def routes(selected: tuple[VisitFact, ...], steps: tuple[VisitFact | TravelFact, ...], node: str):
                counter.tick()
                used_here = {identity(fact) for fact in selected}
                duration = sum(fact.duration_min for fact in selected)
                elapsed = sum(step.duration_min if isinstance(step, VisitFact) else step.minutes for step in steps)
                if elapsed + meal_minutes + rest_minutes > end - start:
                    return
                # Nonnegative costs make this a lower bound. Match the stored
                # day's cent rounding so fractions cannot prune a valid plan.
                known = sum(_cost(step)[0] for step in steps)
                reserved = meal_reserve + sum(_cost(step)[1] for step in steps)
                if round(known, 2) + round(reserved, 2) + future_meals > remaining_budget + 0.001:
                    return

                def finish():
                    if not selected:
                        return
                    back = edge_between(node, base_key)
                    if back is None:
                        return
                    completed = (*steps, back)
                    credit = sum(step.depth_credit_min for step in completed if isinstance(step, TravelFact))
                    if duration + credit < target:
                        return
                    if on_date in rainy and sum(fact.indoor for fact in selected) * 2 < len(selected):
                        return
                    known = sum(_cost(step)[0] for step in completed)
                    reserved = meal_reserve + sum(_cost(step)[1] for step in completed)
                    if known + reserved > remaining_budget + 0.001:
                        return
                    if round(known, 2) + round(reserved, 2) + future_meals > remaining_budget + 0.001:
                        return
                    timeline = _timeline(completed, on_date, start, end, meal_minutes, rest_minutes,
                                         meal_reserve, hints, counter)
                    if timeline is not None:
                        yield SkeletonDay(index + 1, on_date, base_key, bases[base_key]["base_query"], timeline,
                                          tuple(fact.visit_id for fact in selected), duration,
                                          sum(step.minutes for step in completed if isinstance(step, TravelFact)),
                                          credit, round(known, 2), round(reserved, 2))

                # A complete sufficient day comes before packing more of the
                # candidate pool. Expand only if later days cannot be completed.
                yield from finish()
                if len(selected) < selected_limit:
                    choices = [fact for fact in candidates if identity(fact) not in used_here]
                    choices.sort(key=lambda fact: (
                        sum(any(window.date == day for window in fact.date_windows) for day in remaining_dates),
                        -min(fact.duration_min, max(0, target - duration)),
                        graph.get((node, fact.visit_id)).minutes if (node, fact.visit_id) in graph else 10**9,
                        sum(_cost(fact)), fact.visit_id))
                    for fact in choices:
                        leg = edge_between(node, fact.visit_id)
                        if leg is not None:
                            yield from routes((*selected, fact), (*steps, leg, fact), fact.visit_id)

            yield from routes((), initial, base_key)

    failed_budgets: dict[tuple[int, frozenset[int], str | None], float] = {}

    def search(index: int, used: frozenset[int], previous_base: str | None, budget: float):
        counter.tick()
        if index == len(dates):
            return ()
        state = (index, used, previous_base)
        if budget <= failed_budgets.get(state, -math.inf):
            return None

        def failed():
            # Only a proved/exhausted failure is memoised. _SearchLimit unwinds
            # without reaching here; a later path with MORE budget is retried.
            failed_budgets[state] = max(budget, failed_budgets.get(state, -math.inf))
            return None

        if meal_reserve * (len(dates) - index) > budget + 0.001:
            return failed()
        remaining = {identity(fact) for day in dates[index:] for fact in eligible[day]} - used
        if len(remaining) < len(dates) - index:
            return failed()
        # This is an intentionally generous upper bound, not invented travel
        # credit: it can prove insufficiency, never make an actual day pass.
        duration_upper = sum(max(fact.duration_min for fact in facts.values() if identity(fact) == key)
                             for key in remaining)
        credit_upper = (len(dates) - index) * (limit + 2) * max(
            (edge.depth_credit_min for edge in graph.values()), default=0)
        if duration_upper + credit_upper < target * (len(dates) - index):
            return failed()
        for day in day_options(index, used, previous_base, budget):
            identities = frozenset(identity(facts[visit_id]) for visit_id in day.visit_ids)
            later = search(index + 1, used | identities, day.base_key,
                           budget - day.known_cost_total - day.planning_reserve_total)
            if later is not None:
                return (day, *later)
        return failed()

    try:
        days = search(0, frozenset(), None, request.program_budget)
    except _SearchLimit:
        return SkeletonResult("search_limit", problems=("Достигнут безопасный предел локального поиска; план не сохранён.",),
                              missing_facts=tuple(sorted(missing | absent_edges)),
                              excluded_visit_ids=tuple(sorted(excluded)), search_states=counter.used,
                              coverage=replace(coverage, missing_facts=tuple(sorted(missing | absent_edges))),
                              search_complete=False)
    if days is None and allow_free_days and len(dates) > 1:
        # Exact-count searches build only complete calendars. Keeping a best
        # completed result lets a later cutoff return useful, honest output,
        # never the partially explored prefix of a route.
        limited_failed: dict[tuple[int, frozenset[int], str | None, int], float] = {}

        def free_day(index: int, base_key: str) -> SkeletonDay | None:
            timeline = _timeline((), dates[index], start, end, meal_minutes, rest_minutes,
                                 meal_reserve, hints, counter)
            if timeline is None:
                return None
            return SkeletonDay(index + 1, dates[index], base_key, bases[base_key]["base_query"],
                               timeline, (), 0, 0, 0, 0.0, round(meal_reserve, 2), is_free_day=True)

        def limited_search(index: int, used: frozenset[int], previous_base: str | None,
                           budget: float, active_left: int):
            counter.tick()
            if index == len(dates):
                return () if active_left == 0 else None
            state = index, used, previous_base, active_left
            if budget <= limited_failed.get(state, -math.inf):
                return None

            def failed():
                limited_failed[state] = max(budget, limited_failed.get(state, -math.inf))
                return None

            remaining_days = len(dates) - index
            if not 0 <= active_left <= remaining_days or meal_reserve * remaining_days > budget + 0.001:
                return failed()
            remaining = {identity(fact) for day in dates[index:] for fact in eligible[day]} - used
            if len(remaining) < active_left:
                return failed()
            duration_upper = sum(max(fact.duration_min for fact in facts.values() if identity(fact) == key)
                                 for key in remaining)
            credit_upper = active_left * (limit + 2) * max(
                (edge.depth_credit_min for edge in graph.values()), default=0)
            if duration_upper + credit_upper < target * active_left:
                return failed()
            if active_left:
                for day in day_options(index, used, previous_base, budget, active_left):
                    identities = frozenset(identity(facts[visit_id]) for visit_id in day.visit_ids)
                    later = limited_search(index + 1, used | identities, day.base_key,
                                           budget - day.known_cost_total - day.planning_reserve_total,
                                           active_left - 1)
                    if later is not None:
                        return (day, *later)
            if active_left < remaining_days:
                # Before the first active day the base is unresolved. Resolve
                # it from the completed suffix, so early free days inherit
                # that first real base without manufacturing a transfer.
                later = limited_search(index + 1, used, previous_base,
                                       budget - round(meal_reserve, 2), active_left)
                if later is not None:
                    base_key = previous_base or (later[0].base_key if later else None)
                    if base_key is not None:
                        day = free_day(index, base_key)
                        if day is not None:
                            return (day, *later)
            return failed()

        all_identities = {identity(fact) for candidates in eligible.values() for fact in candidates}
        maximum_active = min(len(dates) - 1, len(all_identities))
        if target > 0 and not any(edge.depth_credit_min for edge in graph.values()):
            maximum_active = min(maximum_active, sum(
                max(fact.duration_min for fact in facts.values() if identity(fact) == key)
                for key in all_identities) // target)
        best_days = None
        complete = True
        try:
            for active_count in range(1, maximum_active + 1):
                candidate = limited_search(0, frozenset(), None, request.program_budget, active_count)
                if candidate is not None:
                    best_days = candidate
        except _SearchLimit:
            complete = False
        if best_days is not None:
            active_count = sum(not day.is_free_day for day in best_days)
            problems = (f"Составлена программа на {active_count} из {len(dates)} дней; остальные дни свободные.",)
            if not complete:
                problems += ("Поиск ограничен; большее число программных дней не исключено.",)
            return SkeletonResult("limited_program", days=best_days, problems=problems,
                                  missing_facts=tuple(sorted(missing | absent_edges)),
                                  excluded_visit_ids=tuple(sorted(excluded)),
                                  known_cost_total=round(sum(day.known_cost_total for day in best_days), 2),
                                  planning_reserve_total=round(sum(day.planning_reserve_total for day in best_days), 2),
                                  search_states=counter.used,
                                  coverage=replace(coverage, missing_facts=tuple(sorted(missing | absent_edges))),
                                  search_complete=complete)
        if not complete:
            return SkeletonResult("search_limit", problems=("Достигнут безопасный предел локального поиска; план не сохранён.",),
                                  missing_facts=tuple(sorted(missing | absent_edges)),
                                  excluded_visit_ids=tuple(sorted(excluded)), search_states=counter.used,
                                  coverage=replace(coverage, missing_facts=tuple(sorted(missing | absent_edges))),
                                  search_complete=False)
    if days is None:
        coverage = _first_day_capacity(coverage, request, bases, hints, facts, graph)
        missing.update(coverage.missing_facts)
        status = ("insufficient_content" if coverage.reason == "first_day_depth_capacity"
                  else "needs_facts" if missing or absent_edges else "insufficient_content")
        return SkeletonResult(status,
                              problems=("Из переданных карточек и связей не удалось собрать все дни без повторов, пересечений и превышения бюджета.",),
                              missing_facts=tuple(sorted(missing | absent_edges)),
                              excluded_visit_ids=tuple(sorted(excluded)), search_states=counter.used,
                              coverage=replace(coverage, missing_facts=tuple(sorted(missing | absent_edges))))
    return SkeletonResult("ready", days=days, missing_facts=tuple(sorted(missing)),
                          excluded_visit_ids=tuple(sorted(excluded)),
                          known_cost_total=round(sum(day.known_cost_total for day in days), 2),
                          planning_reserve_total=round(sum(day.planning_reserve_total for day in days), 2),
                           search_states=counter.used,
                           coverage=replace(coverage, missing_facts=tuple(sorted(missing))))

