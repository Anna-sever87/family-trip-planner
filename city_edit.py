"""Record-only one-day city edits; no completion, geocoder or storage calls.

The caller must authenticate and bind its server-owned candidate snapshot and
resolved city to the saved request before preparing an edit. A model, public
itinerary, coordinate URL or caller-supplied JSON is never a record source.
Semantic fulfillment, quota, weather retrieval and optimistic saving belong
to orchestration. This module does not declare an edit fulfilled by the user.
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
import json

from city_candidates import CityCandidate, _canonical_record, _country_code, _finite_coordinates
from city_estimates import CHILD_VISIT_NOTE, label_city_estimates
from city_evidence import recognize_city_managed_tips, replace_city_managed_tips
from city_quality import check_city_quality
from city_schedule import build_city_schedule_schema, decode_city_schedule
from city_visit_estimates import candidate_planning_data, check_city_visit_estimates
from city_visit_options import check_city_visit_conditions, visit_option
from grounded_response import _check_schema, _validate
from places import MAX_DAY_SPREAD_KM, distance_km, map_record_identity, verify_itinerary
from program_coverage import authorized_free_days, limited_program_problems, rebuild_program_report
from rules import check_city_radius, check_program_depth, check_rules, check_verified_place_duplicates
from schemas import Itinerary, ItineraryDay, TripRequest


_MAP_FIELDS = {"verified", "lat", "lon", "official_name", "provider", "source_url"}


@dataclass(frozen=True)
class CityEditContext:
    request: TripRequest
    original: Itinerary
    target_day: int
    resolved_city: dict
    rainy_dates: tuple[str, ...]
    candidates: list[CityCandidate]
    records: dict[str, dict]
    allowed_candidates: list[CityCandidate]
    before_day: ItineraryDay
    was_free: bool
    canonical_day_schema: dict
    managed_tip_values: tuple[str, ...] = ()


def _day_content(day: ItineraryDay) -> dict:
    value = day.model_dump(warnings=False)
    value.pop("day_mode", None)
    for item in value["items"]:
        for field in _MAP_FIELDS:
            item.pop(field, None)
    return value


def expected_root_metadata(context: CityEditContext, candidate: Itinerary) -> dict:
    """Only retire exact legacy server wording after the last free date fills.

    Arbitrary titles/summaries remain immutable. This is not model editing of
    route prose or a claim that map-only visitor conditions are now verified.
    """
    original = context.original
    result = {"title": original.title, "summary": original.summary}
    if context.managed_tip_values:
        result["tips"] = replace_city_managed_tips(original, candidate, context.managed_tip_values,
                                                   context.candidates, context.request)
    if (context.was_free and original.program_report is not None
            and candidate.program_report is None
            and all(day.day_mode == "program" for day in candidate.days)):
        if original.title == f"{context.request.destination}: поездка со свободными днями":
            result["title"] = f"{context.request.destination}: предварительная городская программа"
        if original.summary == (
            "Предварительная программа по местам, подтверждённым на карте. "
            "На часть дат предусмотрен свободный режим. Условия посещений требуют уточнения."
        ):
            result["summary"] = (
                "Предварительная программа по местам, подтверждённым на карте. "
                "Условия посещений требуют уточнения."
            )
    return result


def _scope_problems(candidate: Itinerary, context: CityEditContext) -> list[str]:
    original = context.original
    if len(candidate.days) != len(original.days):
        return ["Нельзя изменять количество дней городской поездки."]
    problems = []
    expected = expected_root_metadata(context, candidate)
    for field in ("title", "summary", "tips", "city_schedule", "source_schedule"):
        if getattr(candidate, field) != expected.get(field, getattr(original, field)):
            problems.append("Нельзя изменять общие сведения и признаки городской поездки.")
            break
    for number, (before, after) in enumerate(zip(original.days, candidate.days), 1):
        if number != context.target_day:
            if before.model_dump() != after.model_dump():
                problems.append(f"День {number} не является целевым и должен остаться без изменений.")
        elif after.day != before.day or after.date != before.date:
            problems.append("Нельзя изменять номер или дату целевого дня.")
        elif after.day_mode == "free" and not context.was_free:
            problems.append("Модель не может превратить программный день в свободный.")
    return problems


def prepare_city_edit(
    request: TripRequest, original: Itinerary, target_day: int, *,
    candidates: list[CityCandidate], resolved_city: dict,
    canonical_day_schema: dict, rainy_dates=(), managed_tip_values=None,
) -> CityEditContext:
    """Prepare a defensive snapshot after the caller's destination binding.

    Every saved visit must still match an authoritative record exactly. A
    selected-only recovered pool permits rearrangement of those visits, not
    invention of the missing historical pool or new visitor facts.
    """
    if (request.destination_kind != "city" or original.city_schedule is not True
            or original.source_schedule is not False or not 1 <= request.days_count <= 14):
        raise ValueError("Нужен сохранённый городской план по проверенным кандидатам.")
    if type(target_day) is not int or not 1 <= target_day <= request.days_count:
        raise ValueError("Неверный номер изменяемого дня.")
    calendar = [(date.fromisoformat(request.start_date) + timedelta(days=i)).isoformat()
                for i in range(request.days_count)]
    if (len(original.days) != request.days_count
            or any(type(day.day) is not int or day.day != i + 1 or day.date != calendar[i]
                   for i, day in enumerate(original.days))):
        raise ValueError("Сохранённый маршрут должен содержать все исходные даты по порядку.")
    problems = limited_program_problems(original, request, allow_city_limited_program=True)
    if problems:
        raise ValueError(" ".join(problems))
    if (not isinstance(resolved_city, dict) or not _finite_coordinates(resolved_city)
            or not _country_code(resolved_city.get("country_code"))):
        raise ValueError("Нужен подтверждённый центр города с контекстом страны.")
    if isinstance(rainy_dates, (str, bytes)):
        raise ValueError("Дождливые даты должны быть отдельными датами поездки.")
    rain = tuple(rainy_dates)
    if any(type(value) is not str or value not in calendar for value in rain) or len(set(rain)) != len(rain):
        raise ValueError("Дождливые даты должны быть уникальными датами поездки.")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 18:
        raise ValueError("Нужен ограниченный серверный набор из 1–18 кандидатов.")
    center = deepcopy(resolved_city)
    center["country_code"] = _country_code(center["country_code"])
    prepared, records, by_query, identifiers = [], {}, {}, set()
    for candidate in candidates:
        if not isinstance(candidate, CityCandidate) or not _finite_coordinates(candidate.record):
            raise ValueError("Некорректная запись серверного кандидата.")
        record = _canonical_record(candidate.title, candidate.record, center)
        identity = map_record_identity(record)
        if (record is None or identity is None or candidate.candidate_id != "-".join(map(str, identity))
                or candidate.title != record["place_query"].split(",", 1)[0]
                or candidate.place_query != record["place_query"] or candidate.area != record["locality"]
                or candidate.candidate_id in identifiers or candidate.place_query in records):
            raise ValueError("Кандидат не соответствует своей проверенной идентичности карты.")
        value = CityCandidate(candidate.candidate_id, candidate.title, candidate.place_query,
                              candidate.area, deepcopy(record))
        prepared.append(value)
        identifiers.add(value.candidate_id)
        records[value.place_query] = value.record
        by_query[value.place_query] = value
    used, elsewhere = set(), set()
    for day in original.days:
        for item in day.items:
            if item.kind != "attraction":
                continue
            candidate = by_query.get(item.place_query)
            if (candidate is None or item.title != candidate.title or item.area != candidate.area
                    or candidate.candidate_id in used or item.verified is not True):
                raise ValueError("Сохранённое посещение не связано с уникальной подтверждённой записью карты.")
            used.add(candidate.candidate_id)
            if day.day != target_day:
                elsewhere.add(candidate.candidate_id)
    original_copy = original.model_copy(deep=True)
    map_problems = verify_itinerary(original_copy, verified_records=records)
    if map_problems or original_copy.model_dump() != original.model_dump():
        raise ValueError("Данные карты сохранённого маршрута изменились или требуют повторной проверки.")
    if managed_tip_values is None:
        managed_tip_values = recognize_city_managed_tips(original, prepared, request)
    if (not isinstance(managed_tip_values, (tuple, list)) or len(managed_tip_values) > 32
            or any(not isinstance(value, str) or value not in original.tips for value in managed_tip_values)
            or len(set(managed_tip_values)) != len(managed_tip_values)):
        raise ValueError("Некорректная серверная маркировка пояснений маршрута.")
    # A known closed date/age restriction must be visible before a completion,
    # not discovered only by the final validator after a paid attempt.
    available = []
    for value in prepared:
        if value.candidate_id in elsewhere:
            continue
        option = visit_option(value, request)
        if (option is None or option.allowed_dates is not None and calendar[target_day - 1] not in option.allowed_dates
                or request.children_ages and (option.children_allowed is False or
                    option.min_child_age is not None and min(request.children_ages) < option.min_child_age)):
            continue
        available.append(value)
    was_free = target_day in authorized_free_days(original, request, allow_city_limited_program=True)
    canonical = deepcopy(canonical_day_schema)
    _check_schema(canonical)
    _validate(target_day, canonical["properties"]["day"], "schema.day")
    _validate(calendar[target_day - 1], canonical["properties"]["date"], "schema.date")
    items = canonical["properties"]["items"]
    if was_free:
        items["minItems"] = 2
        # Free pauses have no asserted venue/locality. Attraction identity is
        # still restored from the nonempty, validated server candidate area.
        items["items"]["properties"]["area"]["minLength"] = 0
    # Server-owned neutral labels can exceed the compact original model limit.
    # Preserve existing accepted text without loosening identity or quantities.
    for field in ("description", "kids_note"):
        rule = items["items"]["properties"][field]
        old_maximum = max((len(getattr(item, field)) for item in original.days[target_day - 1].items), default=0)
        rule["maxLength"] = max(rule.get("maxLength", 0), min(1000, old_maximum))
        if field == "kids_note" and request.children_ages:
            rule["maxLength"] = max(rule["maxLength"], len(CHILD_VISIT_NOTE))
    return CityEditContext(request.model_copy(deep=True), original.model_copy(deep=True), target_day,
                           center, rain, prepared, records, available,
                           original.days[target_day - 1].model_copy(deep=True), was_free, canonical,
                           tuple(managed_tip_values))


def _wire_parts(context: CityEditContext) -> tuple[dict, dict, list[CityCandidate]]:
    key = f"day_{context.target_day}"
    canonical = {"type": "object", "properties": {key: deepcopy(context.canonical_day_schema)},
                 "required": [key], "additionalProperties": False}
    # When every ID belongs to other days, only the empty attraction array is
    # allowed. A nonempty pool is needed by the shared strict schema adapter.
    pool = context.allowed_candidates or context.candidates
    wire = build_city_schedule_schema(canonical, context.request, pool)
    visits = wire["properties"][key]["properties"]["attractions"]
    if context.was_free:
        visits["minItems"] = 0
        if not context.allowed_candidates:
            visits["maxItems"] = 0
    _check_schema(wire)
    return canonical, wire, pool


def build_city_edit_schema(context: CityEditContext) -> dict:
    return _wire_parts(context)[1]


def city_edit_prompt_data(context: CityEditContext) -> dict:
    """Public scheduling inputs; full provider records never enter the prompt."""
    pairs = []
    for index, first in enumerate(context.allowed_candidates):
        for second in context.allowed_candidates[index + 1:]:
            km = distance_km(first.record["lat"], first.record["lon"], second.record["lat"], second.record["lon"])
            if km > MAX_DAY_SPREAD_KM:
                pairs.append({"candidate_ids": [first.candidate_id, second.candidate_id], "distance_km": round(km, 2)})
    by_query = {value.place_query: value.candidate_id for value in context.candidates}
    before = _day_content(context.before_day)
    for item in before["items"]:
        if item["kind"] == "attraction":
            item["candidate_id"] = by_query[item["place_query"]]
    def known_conditions(candidate):
        option = visit_option(candidate, context.request)
        if option is None:
            return {"status": "unavailable"}
        return {
            "status": "available", "duration_min": option.duration_min, "duration_basis": option.basis,
            "indoor": option.indoor, "estimated_cost": option.estimated_cost,
            "description_scope": option.description, "unknown_fields": list(option.unknown_fields),
            "allowed_dates": deepcopy(option.allowed_dates), "opening_windows": deepcopy(option.opening_windows),
            "children_allowed": option.children_allowed, "min_child_age": option.min_child_age,
            "restrictions": list(option.restrictions), "sources": deepcopy(option.sources),
            "note": "Источники относятся только к указанной программе/экспозиции. Не переноси эти свойства на другие части объекта; неизвестное не является подтверждением.",
        }
    return {
        "target_day": context.target_day, "target_date": context.before_day.date,
        "existing_free_day": context.was_free, "before_day": before,
        "rainy_dates": list(context.rainy_dates),
        "unavailable_candidate_ids": [value.candidate_id for value in context.candidates
                                      if value not in context.allowed_candidates],
        "available_candidates": [dict(candidate_id=value.candidate_id, title=value.title,
                                      place_query=value.place_query, area=value.area,
                                      map_type=value.record.get("type", ""), lat=value.record["lat"], lon=value.record["lon"],
                                      planning_limits=candidate_planning_data(value),
                                      visit_conditions=known_conditions(value))
                                 for value in context.allowed_candidates],
        "incompatible_same_day_pairs": pairs,
        "maximum_same_day_spread_km": MAX_DAY_SPREAD_KM,
    }


def decode_city_edit(raw: str, context: CityEditContext) -> Itinerary:
    canonical, wire, pool = _wire_parts(context)
    decoded = decode_city_schedule(raw, wire, pool, canonical_schema=canonical)
    changed = ItineraryDay.model_validate(json.loads(decoded)[f"day_{context.target_day}"])
    changed.day_mode = "free" if context.was_free and not any(item.kind == "attraction" for item in changed.items) else "program"
    # Never relabel unchanged days. Only independently marked server evidence
    # can change at route level; arbitrary original tips remain immutable.
    target_only = Itinerary(title="", summary="", days=[changed], city_schedule=True)
    label_city_estimates(target_only, context.request)
    result = context.original.model_copy(deep=True)
    result.days[context.target_day - 1] = target_only.days[0]
    rebuild_program_report(result, context.request, eligible_unique_visits=len(context.candidates),
                           allow_city_limited_program=True)
    for field, value in expected_root_metadata(context, result).items():
        setattr(result, field, value)
    return result


def validate_city_edit(candidate: Itinerary, context: CityEditContext, *, blocked_places: list[dict]) -> list[str]:
    """Validate a full copy; publish the checked target only on complete success.

    No other day/root text is repaired, and no IO fallback is permitted. The
    existing pure blocked-place checker is reused; its catalogue is supplied
    by orchestration. Semantic fulfillment must still be audited afterwards.
    """
    from planner import _blocked_place_problems

    scope = _scope_problems(candidate, context)
    if scope:
        return scope
    metadata = limited_program_problems(candidate, context.request, allow_city_limited_program=True)
    if metadata:
        return metadata
    try:
        # Recheck primitive types/bounds before Pydantic can coerce an object
        # created via assignment or model_copy. The ordinary decoder already
        # enforces this boundary; direct callers get the same protection.
        raw_target = candidate.days[context.target_day - 1]
        _validate(_day_content(raw_target), context.canonical_day_schema)
        if raw_target.base_area or raw_target.transfer_note or raw_target.rainy_alternative:
            return ["Городской день не может добавлять базу, переезд или неподтверждённое запасное посещение."]
        if any(item.place_query for item in raw_target.items if item.kind in ("meal", "rest")):
            return ["Паузы городского дня не могут объявлять неподтверждённые места."]
        # Metadata is server-owned even for direct callers bypassing the decoder.
        expected = candidate.model_copy(deep=True)
        expected.program_report = deepcopy(context.original.program_report)
        rebuild_program_report(expected, context.request, eligible_unique_visits=len(context.candidates),
                               allow_city_limited_program=True)
        if expected.program_report != candidate.program_report:
            return ["Отчёт об ограниченной программе должен быть пересчитан сервером."]
        checked = Itinerary.model_validate(candidate.model_dump(warnings=False))
    except ValueError:
        return ["Изменённая городская программа содержит некорректные значения."]
    target = checked.days[context.target_day - 1]
    if _day_content(target) == _day_content(context.before_day):
        return ["Целевой день остался без изменений. Нужно выполнить пожелание пользователя."]
    allowed = {value.place_query: value for value in context.allowed_candidates}
    for item in target.items:
        if item.kind == "attraction":
            identity = allowed.get(item.place_query)
            if identity is None or item.title != identity.title or item.area != identity.area:
                return ["Посещение целевого дня не соответствует разрешённому серверному кандидату."]
    problems = limited_program_problems(checked, context.request, allow_city_limited_program=True)
    problems += _blocked_place_problems(checked, blocked_places)
    problems += check_rules(checked, context.request, context.rainy_dates, allow_city_limited_program=True)
    problems += verify_itinerary(checked, verified_records=context.records)
    problems += _scope_problems(checked, context)
    problems += _blocked_place_problems(checked, blocked_places)
    problems += check_city_radius(checked, (context.resolved_city["lat"], context.resolved_city["lon"]))
    problems += check_verified_place_duplicates(checked, verified_records=context.records)
    problems += check_program_depth(checked, context.request, allow_city_limited_program=True)
    problems += check_city_quality(checked, context.request, context.records, allow_city_limited_program=True)
    problems += check_city_visit_estimates(checked, context.candidates)
    problems += check_city_visit_conditions(checked, context.candidates, context.request)
    if not problems:
        candidate.days[context.target_day - 1] = target.model_copy(deep=True)
    return problems
