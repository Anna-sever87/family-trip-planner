"""Bounded scheduling witness over map identities, not visitor/road facts.

Fixed product planning blocks are independent of the requested day depth.
The witness does not create an itinerary or grant validation exemptions. It
can only constrain a model response and supply a complete local timetable;
the caller must still run all normal geography, budget and transport checks.
No AI, HTTP, catalogue, cache or persistence access occurs here.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from itertools import combinations
import math
import re

import places
from rules import _CHILD_WORD_RE, program_depth_target


_DURATIONS = {
    "museum": 90, "gallery": 90, "park": 90, "garden": 90,
    "mall": 90, "shopping_centre": 90, "zoo": 120, "aquarium": 90,
    "theme_park": 120, "archaeological_site": 60, "nature_reserve": 60,
    "beach": 60, "place_of_worship": 30, "artwork": 30,
    "monument": 30, "memorial": 30, "attraction": 45,
}
MAX_SEARCH_STATES = 16000
MAX_OPTION_CHECKS = 120000
_INTEREST_TYPES = {
    "main_sights": {"attraction", "place_of_worship", "artwork", "monument", "memorial", "archaeological_site"},
    "museums": {"museum", "gallery"}, "parks": {"park", "garden", "nature_reserve", "beach"},
    "shopping_malls": {"mall", "shopping_centre"}, "zoos": {"zoo", "aquarium"},
    "amusement_parks": {"theme_park"},
}


def requested_minimum_bases(request):
    """Recognise explicit overnight-base requests, never infer from 'region'.

    This finite Russian-language rule is intentionally conservative: current
    wishes/goal only, no memory, no single-city or fixed-stay interpretation.
    Unrecognised prose remains the model's task; it is not silently classified.
    """
    if request.destination_kind != 'region' or request.days_count < 2:
        return 1
    text = (request.wishes + ';' + request.trip_goal).casefold().replace('ё', 'е')
    if re.search(r'\b(?:без\s+смен\w*\s+(?:жилья|баз\w*|отел\w*|гостиниц\w*)|'
                 r'(?:одн\w*|единственн\w*)\s+баз\w*|не\s+(?:менять|меняем)\s+(?:жилье|баз\w*|отел\w*))\b', text):
        return 1
    required = 1
    for clause in re.split(r'[.!?;\n]', text):
        if re.search(r'\b(?:не\s+(?:нужн\w*|хочу|планиру\w*)\s+(?:нескольк\w*|разн\w*|две|двух|2|три|трех|3|смен\w*)|'
                     r'необязательн\w*\s+смен\w*|смен\w*\s+жилья\s+не\s+нужн\w*|баз\w*\s+не\s+нужн\w*|'
                     r'возможн\w*|при\s+необходимости|если\s+понадобится|допускаю|не\s+против|можно)\b', clause):
            continue
        if re.search(r'\b(?:смен\w*\s+(?:жилья|баз\w*|отел\w*|гостиниц\w*)|'
                     r'(?:нескольк\w*|разн\w*|две|двух|2|три|трех|3)\s+(?:разн\w*\s+)?баз\w*)\b', clause):
            required = max(required, 3 if re.search(r'\b(?:три|трех|3)\s+(?:разн\w*\s+)?баз\w*', clause) else 2)
    return min(required, request.days_count)


def regional_coverage(request, itinerary, grounding=None):
    """Observable base coverage; missing bases are a partial outcome, not fake geography."""
    from grounded_schedule import _pool
    bases = _pool(grounding)[0] if grounding is not None else []
    by_query = {base['base_query']: base for base in bases}
    selected = {day.base_area for day in itinerary.days if day.base_area
                and (grounding is None or day.base_area in by_query)
                and any(item.kind == 'attraction' for item in day.items)}
    required = requested_minimum_bases(request)
    missing = len(selected) < required
    return {'quality_status': 'partial' if missing else 'complete',
            'quality_reasons': ['regional_scope_narrowed'] if missing else [],
            'requested_bases': required, 'selected_bases': len(selected),
            **({'confirmed_bases': len(bases)} if grounding is not None else {})}


def supports_witness(request, bases):
    """Distinguish an unsupported profile from an insufficient verified pool."""
    return (request.destination_kind == "region" and not request.children_ages
            and program_depth_target(request) is not None and 1 <= len(bases) <= 3
            and all(value in _INTEREST_TYPES for value in request.interest_categories))


def _minutes(value):
    hour, minute = map(int, value.split(":"))
    return hour * 60 + minute


def _clock(value):
    return f"{value // 60:02d}:{value % 60:02d}"


def _distance(first, second):
    return places.distance_km(first["lat"], first["lon"], second["lat"], second["lon"])


def _reserve(first, second):
    """An explicit planning buffer over the existing geometric lower bound."""
    distance = _distance(first, second)
    lower = math.ceil(distance + 10) if distance > 3 else 0
    return max(15, math.ceil(lower * 1.5) + 15)


def _timeline(request, base, ordered, previous_base=None, *, lunch_override=None):
    start, end = _minutes(request.start_time), _minutes(request.end_time)
    if lunch_override is None:
        # The requested day window is fixed, lunch is not. A 13:00 lunch can
        # permit an honest morning regional visit where 12:00 creates a gap.
        for lunch in dict.fromkeys(max(start + 150, min(start + offset, 14 * 60)) for offset in (180, 240)):
            result = _timeline(request, base, ordered, previous_base, lunch_override=lunch)
            if result is not None:
                return result
        return None
    lunch = lunch_override
    pause_end = lunch + 80  # a 60-minute lunch and a 20-minute adult rest
    def travel_after(cursor, duration):
        # A travel buffer is free time, not elapsed wall-clock time. The
        # validators subtract meal/rest, so a leg crossing them must pause.
        if cursor >= pause_end or cursor + duration <= lunch:
            return cursor + duration
        before_pause = max(0, lunch - cursor)
        return pause_end + duration - before_pause

    cursor = travel_after(start, _reserve(previous_base, base)) if previous_base and previous_base != base else start
    previous = base
    slots = []
    for candidate in ordered:
        cursor = travel_after(cursor, _reserve(previous, candidate))
        duration = candidate["planning_duration_min"]
        if cursor < pause_end and cursor + duration > lunch:
            cursor = pause_end
        if cursor + duration > end:
            return None
        slots.append({"place_query": candidate["place_query"], "time": _clock(cursor), "duration_min": duration})
        cursor += duration
        previous = candidate
    return_minutes = _reserve(previous, base)
    returned = travel_after(cursor, return_minutes)
    if returned > end or pause_end > end:
        return None
    return {"visits": slots, "meal_time": _clock(lunch), "meal_duration_min": 60,
            "rest_time": _clock(lunch + 60), "rest_duration_min": 20,
            "return_to_base_by": _clock(returned),
            "transfer_reserve_min": (_reserve(previous_base, base)
                                     if previous_base and previous_base != base else 0)}


def build_witness(request, bases, candidates, max_attractions, *, minimum_bases=1):
    """Find a complete disjoint adult schedule, or return None without guessing.

    Pure set packing is bounded at 18 candidates/3 bases/16000 search states.
    Missing map subtype is not inferred from a title. Every planned day reaches
    the existing depth target from visits alone; travel never pads the target.
    """
    target = program_depth_target(request)
    if not supports_witness(request, bases) or not 1 <= len(candidates) <= 18:
        return None
    if type(minimum_bases) is not int or not 1 <= minimum_bases <= min(request.days_count, len(bases)):
        return None
    eligible = []
    allowed_types = set().union(*(_INTEREST_TYPES.get(value, set()) for value in request.interest_categories))
    for candidate in candidates:
        kind = candidate.get("map_poi_type")
        if (not isinstance(kind, str) or kind not in _DURATIONS or _CHILD_WORD_RE.search(candidate["title"])
                or allowed_types and kind not in allowed_types):
            continue
        row = dict(candidate, planning_duration_min=_DURATIONS[kind])
        eligible.append(row)
    if sum(row["planning_duration_min"] for row in eligible) < target * request.days_count:
        return None
    cap = min(max_attractions, 5)
    options = []
    nearest = {index: min((base_index for base_index, base in enumerate(bases)
                          if base['base_key'] in candidate['allowed_base_keys']),
                         key=lambda base_index: _distance(bases[base_index], candidate))
               for index, candidate in enumerate(eligible)}
    local_counts = {base_index: sum(value == base_index for value in nearest.values()) for base_index in range(len(bases))}
    for base_index, base in enumerate(bases):
        indices = [index for index, candidate in enumerate(eligible) if base["base_key"] in candidate["allowed_base_keys"]]
        for size in range(1, min(cap, len(indices)) + 1):
            for selected in combinations(indices, size):
                duration = sum(eligible[index]["planning_duration_min"] for index in selected)
                # Minimal sufficient sets preserve scarce long visits for later dates.
                if duration < target or not any(nearest[index] == base_index for index in selected):
                    continue
                if any(duration - eligible[index]['planning_duration_min'] >= target
                       and any(nearest[other] == base_index for other in selected if other != index)
                       for index in selected):
                    continue
                if any(_distance(eligible[left], eligible[right]) > 120 for left, right in combinations(selected, 2)):
                    continue
                remaining, ordered, current = list(selected), [], base
                while remaining:
                    index = min(remaining, key=lambda index: (_distance(current, eligible[index]), index))
                    ordered.append(index); remaining.remove(index); current = eligible[index]
                if _timeline(request, base, [eligible[index] for index in ordered]) is None:
                    continue
                mask = sum(1 << index for index in selected)
                distance = sum(_distance(left, right) for left, right in zip(
                    [base] + [eligible[index] for index in ordered], [eligible[index] for index in ordered] + [base]))
                options.append((mask, base_index, tuple(ordered), duration, distance))
    if not options:
        return None
    options.sort(key=lambda item: (item[3] - target, item[4], len(item[2]), item[1], item[0]))
    states = option_checks = 0
    total_mask = (1 << len(eligible)) - 1

    @lru_cache(maxsize=MAX_SEARCH_STATES)
    def find(used, previous, visited_bases, days_left):
        nonlocal states, option_checks
        states += 1
        if days_left == 0:
            return () if visited_bases.bit_count() >= minimum_bases else None
        if states > MAX_SEARCH_STATES:
            return None
        remaining = total_mask ^ used
        if sum(row["planning_duration_min"] for index, row in enumerate(eligible) if remaining & (1 << index)) < days_left * target:
            return None
        # Stay at a productive base where possible. A remote base with one
        # short stop does not receive a whole empty day just because it exists.
        ranked = sorted(options, key=lambda item: (
            previous not in (-1, item[1]),
            local_counts[item[1]] if previous == -1 and minimum_bases > 1 else 0,
            item[3] - target, item[4], item[0]))
        for mask, base_index, ordered, duration, _ in ranked:
            if states >= MAX_SEARCH_STATES or option_checks >= MAX_OPTION_CHECKS:
                return None
            option_checks += 1
            if mask & used:
                continue
            timeline = _timeline(request, bases[base_index], [eligible[index] for index in ordered],
                                 bases[previous] if previous >= 0 else None)
            if timeline is None:
                continue
            tail = find(used | mask, base_index, visited_bases | (1 << base_index), days_left - 1)
            if tail is not None:
                return ((base_index, ordered, timeline, duration),) + tail
        return None

    selected = find(0, -1, 0, request.days_count)
    if selected is None:
        return None
    days = []
    for number, (base_index, ordered, timeline, duration) in enumerate(selected, 1):
        days.append({"day": number, "base_area": bases[base_index]["base_query"],
                     "visit_minutes": duration, **timeline})
    return {"basis": "planning_estimates_only", "road_access_verified": False,
            "hours_and_prices_verified": False, "program_depth_target_minutes": target,
            "binding": not any((request.wishes.strip(), request.trip_goal.strip(), request.stay_area.strip()))
                       and request.location_preference == "anywhere",
            "days": days}


def constrain_witness_schema(schema, witness):
    """Bind complete day sets; untouched days in a keyed repair stay untouched."""
    result = deepcopy(schema)
    if witness is None or not witness.get("binding", False):
        return result
    for day in witness["days"]:
        field = result["properties"].get(f'day_{day["day"]}')
        if field is None:
            continue
        properties = field["properties"]
        properties["base_area"].update(const=day["base_area"], enum=[day["base_area"]])
        attractions = properties["attractions"]
        queries = [slot["place_query"] for slot in day["visits"]]
        if len(queries) > attractions["maxItems"]:
            raise ValueError("План посещений превышает лимит дня")
        attractions.update(minItems=len(queries), maxItems=len(queries))
        attractions["items"]["properties"]["place_query"]["enum"] = queries
        attractions["items"]["properties"]["duration_min"]["enum"] = list(dict.fromkeys(slot["duration_min"] for slot in day["visits"]))
        attractions["items"]["properties"]["time"]["enum"] = list(dict.fromkeys(slot["time"] for slot in day["visits"]))
        for kind in ("meal", "rest"):
            properties[kind]["properties"]["time"]["enum"] = [day[kind + "_time"]]
            properties[kind]["properties"]["duration_min"]["enum"] = [day[kind + "_duration_min"]]
    return result


def check_witness(itinerary, witness):
    """Validate a binding contract without rewriting clocks, costs or prose."""
    if witness is None or not witness.get("binding", False) or len(itinerary.days) != len(witness["days"]):
        return []
    by_day = {day.day: day for day in itinerary.days}
    if len(by_day) != len(witness["days"]):
        return []  # The normal calendar validator explains invalid day sets.
    problems = []
    for schedule in witness["days"]:
        day = by_day.get(schedule["day"])
        if day is None or day.base_area != schedule["base_area"]:
            problems.append(f'День {schedule["day"]}: база не соответствует принятому плановому расписанию.')
            continue
        queries = [item.place_query for item in day.items if item.kind == "attraction"]
        expected = [slot["place_query"] for slot in schedule["visits"]]
        if len(queries) != len(set(queries)) or set(queries) != set(expected):
            problems.append(f'День {day.day}: набор мест не соответствует полному плановому расписанию.')
            continue
        if sum(item.kind == "meal" for item in day.items) != 1 or sum(item.kind == "rest" for item in day.items) != 1:
            continue  # The normal structure validator explains meal/rest.
        slots = {slot["place_query"]: slot for slot in schedule["visits"]}
        for item in day.items:
            slot = slots.get(item.place_query) if item.kind == "attraction" else None
            if slot:
                expected_time, duration = slot["time"], slot["duration_min"]
            elif item.kind in {"meal", "rest"}:
                expected_time, duration = schedule[item.kind + "_time"], schedule[item.kind + "_duration_min"]
            else:
                continue
            if (item.time, item.duration_min) != (expected_time, duration):
                problems.append(f'День {day.day}: время/длительность «{item.title}» не совпадают с плановыми блоками; '
                                f'нужны {expected_time} и {duration} минут без противоречий в описании.')
    return problems
