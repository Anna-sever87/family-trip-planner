"""Bounded, deterministic city timetables over server-owned visit options.

No completion, geocoder, HTTP or database operation occurs here. Map records
remain observations, fixed durations/reserves remain labelled planning inputs,
and unknown indoor/price information is never promoted to a visitor fact.
Only already feasible days enter the calendar search; an invalid model day is
never repaired into a free day. The unchanged final validators still run.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from itertools import combinations, permutations
import logging
import math

from city_candidates import CityCandidate
from city_context import build_city_context
from city_coverage import build_city_free_day
from city_estimates import label_city_estimates
from city_evidence import UNKNOWN_INDOOR_NOTE, city_managed_tips
from city_quality import (_family_minimum, apply_city_day_coverage, check_city_quality,
                          city_break_minutes, city_candidate_may_have_shelter,
                          desired_city_program_minutes)
from city_visit_estimates import check_city_visit_estimates, get_city_planning_cap
from destination_name_hints import _https_source
from places import MAX_DAY_SPREAD_KM, distance_km, verify_itinerary
from pricing import minimum_meal_cost
from program_coverage import rebuild_program_report
from rules import check_city_radius, check_program_depth, check_rules, check_verified_place_duplicates, program_depth_target
from schemas import INTEREST_LABELS, Itinerary, ItineraryDay, ItineraryItem, TripRequest


MAX_SEARCH_STATES = 20_000
MAX_GROUP_SEARCH_STATES = 128
INVALID_INPUT_STAGES = frozenset({
    "search_budget", "request", "rain_dates", "candidate_pool", "city_context",
    "interest_groups", "candidate_priority", "visit_options", "blocked_places",
    "visit_conditions", "meal_budget", "visit_limit",
})
_LOGGER = logging.getLogger(__name__)
TRANSFER_RESERVE_MINUTES = 30
TRANSFER_RESERVE_NOTE = "На перемещения оставлен плановый резерв; время дороги и транспорт не проверены."
MAP_ESTIMATE_NOTE = (
    "Карта подтверждает названия и местоположение. Длительность и резервы — плановые оценки, "
    "если отдельный источник не указан; это не подтверждение доступа, часов работы или фактических цен."
)


@dataclass
class DeterministicCityResult:
    itinerary: Itinerary | None = None
    problems: list[str] = field(default_factory=list)
    status: str = "insufficient_data"
    coverage: dict = field(default_factory=dict)

    @property
    def reasons(self) -> list[str]:
        return list(self.coverage.get("reason_codes", ()))


@dataclass(frozen=True)
class _Visit:
    candidate: CityCandidate
    option: object
    duration: int
    cost_cents: int
    windows: dict[str, tuple[tuple[int, int], ...]]


@dataclass(frozen=True)
class _Template:
    mask: int
    order: tuple[int, ...]
    starts: tuple[int, ...]
    meal_start: int
    cost_cents: int


class _Budget:
    def __init__(self, maximum):
        self.maximum = maximum
        self.used = 0
        self.limited = False
        self.group_limits = 0

    def take(self, limit=None):
        if self.used >= (self.maximum if limit is None else min(self.maximum, limit)):
            self.limited = True
            return False
        self.used += 1
        return True


def _minutes(value, *, end=False):
    if not isinstance(value, str) or len(value) != 5 or value[2] != ":":
        raise ValueError("Некорректное время.")
    hour, minute = value.split(":")
    if not hour.isascii() or not minute.isascii() or not hour.isdigit() or not minute.isdigit():
        raise ValueError("Некорректное время.")
    hour, minute = int(hour), int(minute)
    if end and hour == 24 and minute == 0:
        return 1440
    if not 0 <= hour < 24 or not 0 <= minute < 60:
        raise ValueError("Некорректное время.")
    return hour * 60 + minute


def _clock(value):
    return f"{value // 60:02d}:{value % 60:02d}"


def _cents(value, *, budget=False):
    if type(value) not in (int, float) or not 0 <= value <= 1_000_000_000 or not math.isfinite(value):
        raise ValueError("Нужна конечная неотрицательная стоимость.")
    # Never create extra budget by rounding a budget up or a known reserve down.
    return int((Decimal(str(value)) * 100).to_integral_value(
        rounding=ROUND_FLOOR if budget else ROUND_CEILING))


def _checked_request(request):
    if (not isinstance(request, TripRequest) or request.destination_kind != "city"
            or not 1 <= request.days_count <= 14
            or type(request.adults) is not int or not 1 <= request.adults <= 10
            or not isinstance(request.children_ages, list) or len(request.children_ages) > 6
            or any(type(age) is not int or not 0 <= age <= 17 for age in request.children_ages)):
        raise ValueError("Нужна корректная городская анкета на 1–14 дней.")
    if any(date.fromisoformat(value).isoformat() != value for value in (request.start_date, request.end_date)):
        raise ValueError("Даты должны иметь формат ГГГГ-ММ-ДД.")
    start, end = _minutes(request.start_time), _minutes(request.end_time)
    if end - start < 120:
        raise ValueError("Для обеда и отдыха нужно окно не короче двух часов.")
    for name in ("budget", "travel_cost", "lodging_cost"):
        _cents(getattr(request, name), budget=name == "budget")
    if request.budget <= 0:
        raise ValueError("Нужен положительный общий бюджет.")
    if request.program_budget < 0:
        raise ValueError("Дорога и жильё превышают общий бюджет.")
    minimum_meal_cost(request)  # Reject unsupported currency instead of a zero reserve.
    return start, end


def _checked_option(candidate, option, dates):
    from city_visit_options import CityVisitOption

    if not isinstance(option, CityVisitOption) or option.candidate_id != candidate.candidate_id:
        raise ValueError("Условия посещения не связаны с серверным ID.")
    if (type(option.duration_min) is not int or not 1 <= option.duration_min <= get_city_planning_cap(candidate)
            or option.indoor is not None and type(option.indoor) is not bool
            or option.children_allowed is not None and type(option.children_allowed) is not bool
            or option.min_child_age is not None and (type(option.min_child_age) is not int
                                                    or not 0 <= option.min_child_age <= 18)
            or not isinstance(option.description, str) or not option.description.strip()
            or len(option.description) > 2000
            or option.basis not in {"planning_estimate", "published"}
            or not isinstance(option.sources, tuple) or len(option.sources) > 12
            or not isinstance(option.unknown_fields, tuple) or len(option.unknown_fields) > 12
            or any(not isinstance(value, str) or len(value) > 60 for value in option.unknown_fields)
            or not isinstance(option.restrictions, tuple) or len(option.restrictions) > 12
            or any(not isinstance(value, str) or len(value) > 600 for value in option.restrictions)):
        raise ValueError("Некорректные структурированные условия посещения.")
    for source in option.sources:
        if (not isinstance(source, dict) or not _https_source(source.get("source_url"))
                or source.get("field") not in {"duration", "indoor", "hours", "price", "children", "minimum_age"}
                or source.get("basis") not in {"published", "planning_estimate"}
                or date.fromisoformat(source.get("checked_at")).isoformat() != source["checked_at"]):
            raise ValueError("Некорректные метаданные источника условий посещения.")
    if option.allowed_dates is not None:
        if (not isinstance(option.allowed_dates, tuple) or len(set(option.allowed_dates)) != len(option.allowed_dates)
                or any(type(value) is not str or value not in dates for value in option.allowed_dates)):
            raise ValueError("Некорректные разрешённые даты посещения.")
    if not isinstance(option.opening_windows, tuple) or len(option.opening_windows) > 14 * 8:
        raise ValueError("Слишком много или некорректные окна посещения.")
    windows = {}
    for window in option.opening_windows:
        if (not isinstance(window, dict) or window.get("date") not in dates
                or window.get("basis") not in {"source", "estimate"}):
            raise ValueError("Некорректное окно посещения.")
        start, end = _minutes(window.get("start")), _minutes(window.get("end"), end=True)
        if start >= end:
            raise ValueError("Некорректное окно посещения.")
        windows.setdefault(window["date"], []).append((start, end))
    return _Visit(candidate, deepcopy(option), option.duration_min,
                  0 if option.estimated_cost is None else _cents(option.estimated_cost),
                  {key: tuple(sorted(set(values))) for key, values in windows.items()})


def _free_minutes_after(start, reserve, meal_start, pause_duration=120):
    """Advance a transfer reserve through free time, never through meal/rest."""
    pause_end = meal_start + pause_duration
    if start >= pause_end or start + reserve <= meal_start:
        return start + reserve
    before = max(0, meal_start - start)
    return pause_end + max(0, reserve - before)


def _place_visit(start, duration, windows, meal_start, end, pause_duration=120):
    for opening, closing in windows:
        value = max(start, opening)
        if value < meal_start + pause_duration and value + duration > meal_start:
            value = meal_start + pause_duration
        if value + duration <= min(closing, end):
            return value
    return None


def _template(indices, visits, windows, start, end, distances, counter, group_limit,
              *, attempt_limit=None, pause_duration=120, split_after=None, late_finish=None):
    # Necessary bounds only: prove impossible pairs before enumerating lunch
    # and permutations. Ignoring lunch here is optimistic, never a relaxation
    # of the actual schedule or a source of visitor conditions.
    if sum(visits[index].duration for index in indices) + pause_duration > end - start:
        return None
    # A spanning tree is a lower bound on any visiting order's total reserved
    # transfers, even with the non-metric zero-distance threshold. This rules
    # out an overfull group without enumerating every lunch/permutation.
    connected, remaining = {indices[0]}, set(indices[1:])
    minimum_transfer = 0
    while remaining:
        edges = ((0 if distances[tuple(sorted((left, right)))] <= .03 else
                  max(TRANSFER_RESERVE_MINUTES, math.ceil(distances[tuple(sorted((left, right)))])), right)
                 for left in connected for right in remaining)
        cost, reached = min(edges)
        minimum_transfer += cost
        connected.add(reached)
        remaining.remove(reached)
    if sum(visits[index].duration for index in indices) + pause_duration + minimum_transfer > end - start:
        return None
    by_index = dict(zip(indices, windows))
    earliest_finish = {}
    for index in indices:
        finishes = [max(start, opening) + visits[index].duration
                    for opening, closing in by_index[index]
                    if max(start, opening) + visits[index].duration <= min(end, closing)]
        if not finishes:
            return None
        earliest_finish[index] = min(finishes)
    for left, right in combinations(indices, 2):
        km = distances[tuple(sorted((left, right)))]
        # In larger groups these two visits need not be adjacent. The existing
        # co-location threshold is non-metric: intermediate nearby stops may
        # have zero transfer reserve even when this direct pair would not.
        # Zero is the safe optimistic bound; the actual order still reserves
        # every transfer below.
        reserve = 0 if len(indices) > 2 or km <= .03 else max(TRANSFER_RESERVE_MINUTES, math.ceil(km))
        if not any(any(max(start, opening, earliest_finish[first] + reserve) + visits[second].duration
                       <= min(end, closing) for opening, closing in by_index[second])
                   for first, second in ((left, right), (right, left))):
            return None
    # Half-hour lunch alternatives plus source-window boundaries. The fallback
    # slots stay inside the requested day; they are reservations, not venues.
    preferred = min(max(13 * 60, start), end - pause_duration)
    meal_slots = {preferred, start, end - pause_duration}
    meal_slots.update(range(((start + 29) // 30) * 30, end - pause_duration + 1, 30))
    for index, allowed in zip(indices, windows):
        for opening, closing in allowed:
            meal_slots.update((max(start, min(end - pause_duration, opening - pause_duration)),
                               max(start, min(end - pause_duration, closing)),
                               max(start, min(end - pause_duration, max(start, opening) + visits[index].duration)),
                               max(start, min(end - pause_duration, min(end, closing) - visits[index].duration - pause_duration))))
    convenient = sorted(meal_slots, key=lambda value: (abs(value - preferred), value))
    # A half-hour grid alone can miss a feasible source window (e.g. lunch
    # starting 10:37). Try convenient slots first, then every other minute;
    # all attempts share the same strict global search budget.
    meal_slots = convenient + sorted(set(range(start, end - pause_duration + 1)) - meal_slots,
                                    key=lambda value: (abs(value - preferred), value))
    group_started = counter.used
    for meal_start in meal_slots:
        for order in permutations(indices):
            if attempt_limit is not None and counter.used - group_started >= attempt_limit:
                counter.limited = True
                counter.group_limits += 1
                return None
            if not counter.take(group_limit):
                return None
            cursor, starts, previous = start, [], None
            for position, index in enumerate(order):
                if previous is not None:
                    km = distances[tuple(sorted((previous, index)))]
                    gap = 0 if km <= .03 else max(TRANSFER_RESERVE_MINUTES, math.ceil(km))
                    cursor = _free_minutes_after(cursor, gap, meal_start, pause_duration)
                if position == split_after:
                    if previous is not None and distances[tuple(sorted((previous, index)))] <= .03:
                        break  # Keep co-located visits together, not a fake return trip.
                    cursor = max(cursor, meal_start + pause_duration)
                    if late_finish is not None:
                        suffix = order[position:]
                        reserve = sum(0 if distances[tuple(sorted(pair))] <= .03 else
                                      max(TRANSFER_RESERVE_MINUTES, math.ceil(distances[tuple(sorted(pair))]))
                                      for pair in zip(suffix, suffix[1:]))
                        desired_start = late_finish - sum(visits[value].duration for value in suffix) - reserve
                        # At most one extra hour of open time, never a fake visit
                        # or an inflated duration just to reach the end of a day.
                        cursor = max(cursor, min(desired_start, cursor + 60))
                value = _place_visit(cursor, visits[index].duration, by_index[index], meal_start, end, pause_duration)
                if value is None:
                    break
                starts.append(value)
                cursor = value + visits[index].duration
                previous = index
            if len(starts) == len(order):
                return _Template(sum(1 << index for index in indices), tuple(order), tuple(starts),
                                 meal_start, sum(visits[index].cost_cents for index in indices))
    return None


def _make_day(number, day_date, template, visits, meal_cost, rest_minutes=60):
    items = []
    for index, minute in zip(template.order, template.starts):
        visit = visits[index]
        item = visit.candidate
        items.append(ItineraryItem(
            time=_clock(minute), title=item.title, kind="attraction", area=item.area,
            place_query=item.place_query, description=visit.option.description,
            duration_min=visit.duration, cost=visit.cost_cents / 100,
            indoor=visit.option.indoor is True,
        ))
    pause_area = visits[template.order[0]].candidate.area
    items.extend((
        ItineraryItem(time=_clock(template.meal_start), kind="meal", title="Обед", area=pause_area,
                      place_query="", description="Плановый резерв на обед.", duration_min=60, cost=meal_cost),
        ItineraryItem(time=_clock(template.meal_start + 60), kind="rest", title="Отдых", area=pause_area,
                      place_query="", description=("Пауза для отдыха." if rest_minutes >= 60 else
                                                   "Короткая пауза по желанию; дневной сон не предполагается."),
                      duration_min=rest_minutes, cost=0),
    ))
    return ItineraryDay(day=number, date=day_date, theme="Городская программа",
                        items=sorted(items, key=lambda item: item.time))


def build_deterministic_city_itinerary(
    request: TripRequest, candidates, rainy_dates=(), *, resolved_city: dict,
    options=None, blocked_places=None, candidate_priority=None, required_any_of_ids=(),
    required_category_groups=None,
    max_states: int = MAX_SEARCH_STATES,
) -> DeterministicCityResult:
    """Construct one fully checked calendar or return no itinerary.

    ``candidates``, ``resolved_city`` and optional ``options`` must be supplied
    by the server, never by the model or public request. Unknown prices retain
    zero *unpriced* reserves under the existing disclosure contract, not an
    assertion of free admission or proof of actual total travel cost.
    """
    from planner import _blocked_place_problems, _max_attractions_for_request
    from city_visit_options import check_city_visit_conditions, visit_option

    coverage = {"schema_version": 1, "reason_codes": [], "exclusions": [],
                "quality_status": "insufficient_data", "quality_reasons": []}

    def failure(status, message, code):
        coverage["reason_codes"] = list(dict.fromkeys([*coverage["reason_codes"], code]))
        coverage["quality_status"] = "insufficient_data"
        coverage["quality_reasons"] = list(dict.fromkeys([*coverage["quality_reasons"], code]))
        return DeterministicCityResult(None, [message], status, deepcopy(coverage))

    # Only server-owned finite stage names may enter diagnostics. Never log
    # validation exception text: it can contain provider or user input.
    input_stage = "search_budget"
    try:
        if type(max_states) is not int or not 1 <= max_states <= MAX_SEARCH_STATES:
            raise ValueError("Лимит поиска должен быть от 1 до 20000 состояний.")
        input_stage = "request"
        start, end = _checked_request(request)
        dates = [(date.fromisoformat(request.start_date) + timedelta(days=index)).isoformat()
                 for index in range(request.days_count)]
        input_stage = "rain_dates"
        if (not isinstance(rainy_dates, (list, tuple)) or len(rainy_dates) > len(dates)
                or any(type(value) is not str or value not in dates for value in rainy_dates)
                or len(set(rainy_dates)) != len(rainy_dates)):
            raise ValueError("Дождливые даты должны быть уникальными датами поездки.")
        rainy = set(rainy_dates)
        input_stage = "candidate_pool"
        if not isinstance(candidates, (list, tuple)) or len(candidates) > 24:
            raise ValueError("Нужен ограниченный серверный набор кандидатов.")
        coverage.update(requested_days=len(dates), confirmed_candidates=len(candidates))
        if not candidates:
            return failure("insufficient_data", "В проверенном наборе нет мест для городской программы. "
                           "Это не означает, что других мест в городе нет.", "empty_pool")
        input_stage = "city_context"
        snapshot = build_city_context(request, candidates, resolved_city)
        prepared = [CityCandidate(**entry) for entry in snapshot["candidates"]]
        input_stage = "interest_groups"
        if (not isinstance(required_any_of_ids, (tuple, list))
                or any(type(value) is not str for value in required_any_of_ids)
                or len(set(required_any_of_ids)) != len(required_any_of_ids)
                or set(required_any_of_ids) - {candidate.candidate_id for candidate in prepared}):
            raise ValueError("Обязательная группа должна содержать уникальные проверенные ID.")
        required_ids = set(required_any_of_ids)
        category_groups = {} if required_category_groups is None else required_category_groups
        known_ids = {candidate.candidate_id for candidate in prepared}
        if (not isinstance(category_groups, dict) or len(category_groups) > len(INTEREST_LABELS)
                or any(key not in INTEREST_LABELS or not isinstance(values, (tuple, list))
                       or any(type(value) is not str for value in values)
                       or len(set(values)) != len(values) or set(values) - known_ids
                       for key, values in category_groups.items())):
            raise ValueError("Группы интересов должны ссылаться только на проверенные уникальные ID.")
        input_stage = "candidate_priority"
        priority = () if candidate_priority is None else candidate_priority
        if (not isinstance(priority, (tuple, list)) or len(priority) > len(prepared)
                or any(type(value) is not str for value in priority)
                or len(set(priority)) != len(priority)
                or set(priority) - {candidate.candidate_id for candidate in prepared}):
            raise ValueError("Приоритет должен содержать уникальные ID из серверного набора.")
        rank = {identifier: index for index, identifier in enumerate(priority)}
        prepared.sort(key=lambda candidate: (rank.get(candidate.candidate_id, len(rank)), candidate.candidate_id))
        center = snapshot["resolved_city"]
        records = {candidate.place_query: candidate.record for candidate in prepared}
        input_stage = "visit_options"
        if options is None:
            option_map = {candidate.candidate_id: visit_option(candidate, request) for candidate in prepared}
        else:
            if not isinstance(options, (list, tuple)) or len(options) > len(prepared):
                raise ValueError("Некорректный набор условий посещений.")
            option_map = {option.candidate_id: option for option in options}
            if len(option_map) != len(options) or set(option_map) - {candidate.candidate_id for candidate in prepared}:
                raise ValueError("Условия посещения содержат неизвестные или повторные ID.")
        input_stage = "blocked_places"
        if blocked_places is not None and not isinstance(blocked_places, (list, tuple)):
            raise ValueError("Некорректный серверный список исключений.")
        blocked = list(blocked_places or ())
        visits = []
        input_stage = "visit_conditions"
        for candidate in prepared:
            option = option_map.get(candidate.candidate_id)
            reason = "visit_conditions_unavailable" if option is None else None
            if option is not None:
                visit = _checked_option(candidate, option, dates)
                if (request.children_ages and (option.children_allowed is False or
                        option.min_child_age is not None and min(request.children_ages) < option.min_child_age)):
                    reason = "age_restriction"
                elif option.allowed_dates == ():
                    reason = "no_eligible_dates"
                elif option.indoor is True and not city_candidate_may_have_shelter(candidate.record):
                    reason = "conflicting_indoor_evidence"
                probe = Itinerary(title="", summary="", days=[ItineraryDay(day=1, date=dates[0], theme="", items=[
                    ItineraryItem(time=request.start_time, title=candidate.title, place_query=candidate.place_query,
                                  description="", official_name=candidate.record.get("name", "")),
                ])])
                if _blocked_place_problems(probe, blocked):
                    reason = "blocked_place"
            if reason:
                coverage["exclusions"].append({"candidate_id": candidate.candidate_id, "reason": reason})
            else:
                visits.append(visit)
        input_stage = "meal_budget"
        meal_cents = _cents(minimum_meal_cost(request))
        base_cost = meal_cents * len(dates)
        extra_budget = _cents(request.program_budget, budget=True) - base_cost
        coverage.update(eligible_candidates=len(visits), meal_reserve_total=base_cost / 100)
        if extra_budget < 0:
            return failure("insufficient_data", "Бюджета программы недостаточно для резервов на обеды "
                           "на все исходные даты поездки.", "meal_budget")
        if not visits:
            return failure("insufficient_data", "В проверенном наборе не осталось посещений, подходящих "
                           "по известным условиям. Поездка только из свободных дней не создаётся.", "no_visit_options")
        input_stage = "visit_limit"
        max_visits = _max_attractions_for_request(request)
        if type(max_visits) is not int or not 1 <= max_visits <= 5:
            raise ValueError("Некорректный предел посещений.")
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        coverage["invalid_input_stage"] = input_stage
        _LOGGER.warning("city_schedule_input_rejected stage=%s", input_stage)
        return failure("invalid_request", "Не удалось проверить анкету, карту или структурированные "
                       "условия посещения для серверного расписания.", "invalid_inputs")

    target = _family_minimum(request) or program_depth_target(request) or 1
    coverage["daily_program_target_minutes"] = target
    desired = desired_city_program_minutes(request) or target
    rest_minutes = city_break_minutes(request)
    pause_duration = 60 + rest_minutes
    distances = {(left, right): distance_km(visits[left].candidate.record["lat"], visits[left].candidate.record["lon"],
                                           visits[right].candidate.record["lat"], visits[right].candidate.record["lon"])
                 for left, right in combinations(range(len(visits)), 2)}
    counter = _Budget(max_states)
    balance_possible = (len(visits) >= 2 and end - start >= 6 * 60
                        and start < 12 * 60 and end >= 16 * 60)
    # Packing cannot prove that a thin reserve fills a long trip. Keep a small
    # part of the SAME global budget for improving its feasible incumbent:
    # another useful visit, then the actual time distribution. Otherwise a
    # difficult day-count search starves both stages, even with open windows.
    # Tiny diagnostic budgets retain the original behaviour.
    balance_reserve = min(max_states // 10, len(dates) * MAX_GROUP_SEARCH_STATES * 2) if balance_possible else 0
    enrichment_reserve = min(max_states // 20, len(visits) * MAX_GROUP_SEARCH_STATES) if len(visits) >= 2 else 0
    if max_states < MAX_GROUP_SEARCH_STATES * 2:
        balance_reserve = enrichment_reserve = 0
    enrichment_limit = max_states - balance_reserve
    packing_limit = enrichment_limit - enrichment_reserve
    required_mask = sum(1 << index for index, visit in enumerate(visits)
                        if visit.candidate.candidate_id in required_ids)
    preference_masks = {sum(1 << index for index, visit in enumerate(visits)
                            if visit.candidate.candidate_id in identifiers)
                        for identifiers in category_groups.values()}
    preference_masks.add(required_mask)
    preference_masks.discard(0)
    preference_masks = tuple(sorted(preference_masks))
    # Keep an explicit portion for choosing a calendar from already found days.
    group_limit = min(packing_limit, max(1, max_states * 3 // 4))
    templates = [[] for _ in dates]
    template_masks = [set() for _ in dates]
    template_cache = {}
    budget_groups_rejected = 0

    def capacity_upper_bound(day_index, used):
        # Every program day needs this request's actual family/adult target
        # and at least one unique visit. Ignore weather, windows, transfers and
        # price here deliberately: relaxing them is optimistic, never a reason
        # to discard a feasible calendar or invent an indoor/date condition.
        unused = [visit for index, visit in enumerate(visits) if not used & (1 << index)]
        return min(len(dates) - day_index, len(unused), sum(visit.duration for visit in unused) // target)

    def diverse_incumbent():
        """Find a small checked incumbent before exhaustive minimal groups.

        Shorter visits from rare categories may need four-visit days, while
        thousands of museum-only triples occur earlier in combinations(). A
        bounded constructive pass gives those categories a fair chance without
        enlarging the state budget or making this heuristic a feasibility proof.
        """
        if len(preference_masks) < 2:
            return ()
        seed_limit = min(group_limit, max(1, max_states // 20))
        chosen, remaining, calendar = 0, extra_budget, []
        for day_index, day_date in enumerate(dates):
            available = [index for index, visit in enumerate(visits)
                         if not chosen & (1 << index)
                         and visit.cost_cents <= remaining
                         and (visit.option.allowed_dates is None or day_date in visit.option.allowed_dates)
                         and (not visit.windows or visit.windows.get(day_date))]
            best = None
            seen_groups = set()
            for offset in range(min(8, len(available))):
                group, group_mask, duration, cost = [], chosen, 0, 0
                rotated = available[offset:] + available[:offset]
                while len(group) < max_visits and duration < target:
                    eligible = [index for index in rotated if index not in group
                                and cost + visits[index].cost_cents <= remaining
                                and all(distances[tuple(sorted((index, previous)))] <= MAX_DAY_SPREAD_KM
                                        for previous in group)]
                    if not eligible:
                        break
                    index = min(eligible, key=lambda value: (
                        -bool(visits[value].option.indoor is True) if day_date in rainy else 0,
                        -bool((1 << value) & required_mask) if not group_mask & required_mask else 0,
                        -sum(bool((1 << value) & mask) for mask in preference_masks if not group_mask & mask),
                        -visits[value].duration, rotated.index(value)))
                    group.append(index)
                    group_mask |= 1 << index
                    duration += visits[index].duration
                    cost += visits[index].cost_cents
                indices = tuple(sorted(group))
                if (not indices or duration < target or indices in seen_groups
                        or day_date in rainy
                        and sum(visits[index].option.indoor is True for index in indices) * 2 < len(indices)):
                    continue
                seen_groups.add(indices)
                if not counter.take(seed_limit):
                    return tuple(calendar)
                windows = tuple(visits[index].windows.get(day_date, ()) if visits[index].windows
                                else ((start, end),) for index in indices)
                template = _template(indices, visits, windows, start, end, distances, counter,
                                     seed_limit, attempt_limit=MAX_GROUP_SEARCH_STATES,
                                     pause_duration=pause_duration)
                if template is None:
                    continue
                def seed_score(value):
                    return (bool(value.mask & required_mask) if not chosen & required_mask else False,
                            sum(bool(value.mask & mask) for mask in preference_masks if not chosen & mask),
                            -len(value.order), -value.cost_cents)
                if best is None or seed_score(template) > seed_score(best):
                    best = template
            if best is not None:
                calendar.append((day_index, best))
                chosen |= best.mask
                remaining -= best.cost_cents
        return tuple(calendar)

    seeded_calendar = diverse_incumbent()
    seeded_mask = 0
    for day_index, template in seeded_calendar:
        seeded_mask |= template.mask
        templates[day_index].append(template)
        template_masks[day_index].add(template.mask)
    # A checked calendar reaching the optimistic day bound and all requested
    # feasible categories already meets the objective. The bound can be below
    # the requested duration: that result remains partial, never called complete.
    seed_optimal = (len(seeded_calendar) == capacity_upper_bound(0, 0)
                    and all(seeded_mask & mask for mask in preference_masks))

    def dominated(day_index, mask):
        # Groups contain at most five visits: inspecting their <=31 subsets
        # is much cheaper than scanning thousands of existing templates.
        subset = mask
        while subset:
            if (subset in template_masks[day_index]
                    and all(not mask & preference or subset & preference for preference in preference_masks)):
                return True
            subset = (subset - 1) & mask
        return False
    for size in range(1, min(max_visits, len(visits)) + 1):
        if seed_optimal:
            break
        if counter.used >= group_limit:
            counter.limited = True
            break
        for indices in combinations(range(len(visits)), size):
            if not counter.take(group_limit):
                break
            if sum(visits[index].cost_cents for index in indices) > extra_budget:
                budget_groups_rejected += 1
                continue
            if (sum(visits[index].duration for index in indices) < target
                    or any(distances[pair] > MAX_DAY_SPREAD_KM for pair in combinations(indices, 2))):
                continue
            mask = sum(1 << index for index in indices)
            indoors = sum(visits[index].option.indoor is True for index in indices)
            for day_index, day_date in enumerate(dates):
                if (day_date in rainy and indoors * 2 < size
                        or any(visits[index].option.allowed_dates is not None
                               and day_date not in visits[index].option.allowed_dates for index in indices)
                        or dominated(day_index, mask)):
                    continue
                windows = tuple(visits[index].windows.get(day_date, ()) if visits[index].windows
                                else ((start, end),) for index in indices)
                if any(not allowed for allowed in windows):
                    continue
                key = (indices, windows)
                if key not in template_cache:
                    template_cache[key] = _template(indices, visits, windows, start, end,
                                                    distances, counter, group_limit,
                                                    attempt_limit=MAX_GROUP_SEARCH_STATES,
                                                    pause_duration=pause_duration)
                if template_cache[key] is not None:
                    templates[day_index].append(template_cache[key])
                    template_masks[day_index].add(mask)
                if counter.used >= group_limit:
                    break

    def order_key(template, rainy_day):
        indoor_count = sum(visits[index].option.indoor is True for index in template.order)
        # Reserve capacity for other dates before optimizing soft preference.
        # Equal-depth groups still preserve the supplied AI ordering;
        # extra useful visits are added only after packing the maximum days.
        compact = (sum(visits[index].duration for index in template.order) - target, len(template.order))
        if priority:
            # Ranking is only an optimization among feasible groups. It never
            # expands a pool, creates visitor facts or defeats the budget/rain.
            return (0 if rainy_day else indoor_count, compact[0],
                    tuple(sorted(template.order)), compact[1], template.cost_cents)
        return (0 if rainy_day else indoor_count, *compact, template.cost_cents, template.mask)

    for index in range(len(dates)):
        templates[index].sort(key=lambda value: order_key(value, dates[index] in rainy))
    memo = {}
    packing_required_mask = 0
    feasible_visit_mask = 0
    for day_templates in templates:
        for template in day_templates:
            feasible_visit_mask |= template.mask
    feasible_preference_masks = tuple(mask & feasible_visit_mask for mask in preference_masks if mask & feasible_visit_mask)
    feasible_required_mask = 0
    for mask in feasible_preference_masks:
        feasible_required_mask |= mask

    def score(calendar, used):
        selected_mask = used
        for _, template in calendar:
            selected_mask |= template.mask
        # Never trade away a program day just to satisfy an interest. Among
        # equally complete calendars, retain an explicitly requested category.
        return (len(calendar), bool(required_mask and selected_mask & required_mask),
                sum(bool(selected_mask & mask) for mask in preference_masks))

    def pack(day_index, used, remaining):
        if day_index == len(dates):
            return ()
        key = (day_index, used)  # Nonnegative one-use costs make remaining a function of used.
        if key in memo:
            return memo[key]
        upper_bound = capacity_upper_bound(day_index, used)
        if not upper_bound:
            return ()
        if not counter.take(packing_limit):
            return ()
        best = ()
        ordered = templates[day_index]
        if packing_required_mask:
            ordered = sorted(ordered, key=lambda template: (
                -bool(template.mask & required_mask) if not used & required_mask else 0,
                -sum(bool(template.mask & mask) for mask in feasible_preference_masks if not used & mask)))
        for template in ordered:
            if template.mask & used or template.cost_cents > remaining:
                continue
            if 1 + capacity_upper_bound(day_index + 1, used | template.mask) < len(best):
                continue
            rest = pack(day_index + 1, used | template.mask, remaining - template.cost_cents)
            proposal = ((day_index, template),) + rest
            if score(proposal, used) > score(best, used):
                best = proposal
            if len(best) == upper_bound and (not packing_required_mask
                    or (not required_mask & feasible_visit_mask or score(best, used)[1])
                    and score(best, used)[2] >= len(feasible_preference_masks)):
                memo[key] = best
                return best
            if counter.used >= packing_limit:
                break
        if counter.used < packing_limit and capacity_upper_bound(day_index + 1, used) >= len(best):
            skipped = pack(day_index + 1, used, remaining)
            if score(skipped, used) > score(best, used):
                best = skipped
        elif counter.used >= packing_limit:
            counter.limited = True
        memo[key] = best
        return best

    # First retain the ordinary calendar as an incumbent. A difficult/expensive
    # requested category must not spend the bounded search on a worse calendar.
    # A second, category-first pass shares the same global counter and may only
    # replace that incumbent with an equal-or-better number of program days.
    calendar = seeded_calendar if seed_optimal else pack(0, 0, extra_budget)
    if score(seeded_calendar, 0) > score(calendar, 0):
        calendar = seeded_calendar
    if (feasible_required_mask and (score(calendar, 0)[2] < len(feasible_preference_masks)
            or required_mask & feasible_visit_mask and not score(calendar, 0)[1])
            and counter.used < packing_limit):
        packing_required_mask = feasible_required_mask
        memo.clear()
        alternative = pack(0, 0, extra_budget)
        if score(alternative, 0) > score(calendar, 0):
            calendar = alternative
    selected = dict(calendar)
    # Minimal groups preserve calendar capacity during packing. Once the day
    # count is fixed, retain additional useful unique visits when they fit;
    # never stretch a duration, fill a free day below target, or remove a day.
    chosen = {index for template in selected.values() for index in template.order}
    spent = sum(template.cost_cents for template in selected.values())
    enrichment_count = 0
    remaining_indices = set(range(len(visits))) - chosen
    while remaining_indices:
        chosen_mask = sum(1 << index for index in chosen)
        extra = min(remaining_indices, key=lambda index: (
            -sum(bool((1 << index) & mask) for mask in preference_masks if not chosen_mask & mask), index))
        remaining_indices.remove(extra)
        if counter.used >= enrichment_limit:
            counter.limited = True
            break
        if spent + visits[extra].cost_cents > extra_budget:
            continue
        # Enrich the lightest eligible day first instead of packing all spare
        # visits into the beginning of the trip. Never remove a program day.
        ordered_days = sorted(selected.items(), key=lambda pair: (
            sum(visits[index].duration for index in pair[1].order) >= desired,
            sum(visits[index].duration for index in pair[1].order), pair[0]))
        for day_index, original in ordered_days:
            indices = tuple(sorted((*original.order, extra)))
            if len(indices) > max_visits or any(distances[pair] > MAX_DAY_SPREAD_KM
                                               for pair in combinations(indices, 2)):
                continue
            day_date = dates[day_index]
            if (day_date in rainy and sum(visits[index].option.indoor is True for index in indices) * 2 < len(indices)
                    or any(visits[index].option.allowed_dates is not None
                           and day_date not in visits[index].option.allowed_dates for index in indices)):
                continue
            windows = tuple(visits[index].windows.get(day_date, ()) if visits[index].windows
                            else ((start, end),) for index in indices)
            if any(not allowed for allowed in windows):
                continue
            expanded = _template(indices, visits, windows, start, end, distances, counter,
                                 enrichment_limit, attempt_limit=MAX_GROUP_SEARCH_STATES,
                                 pause_duration=pause_duration)
            if expanded is not None:
                selected[day_index] = expanded
                chosen.add(extra)
                spent += visits[extra].cost_cents
                enrichment_count += 1
                break
    # Place the SAME visits on both sides of lunch where sourced windows allow.
    # This does not create capacity: sparse days stay partial below the desired
    # volume, even if a visit can be moved to the afternoon.
    if balance_possible:
        late_finish = end - (120 if rest_minutes == 60 else 90)
        balance_days = [(day_index, original) for day_index, original in sorted(selected.items())
                        if len(original.order) >= 2]
        for position, (day_index, original) in enumerate(balance_days):
            indices = tuple(original.order)
            if all(distances[tuple(sorted(pair))] <= .03 for pair in combinations(indices, 2)):
                continue
            day_date = dates[day_index]
            windows = tuple(visits[index].windows.get(day_date, ()) if visits[index].windows
                            else ((start, end),) for index in indices)
            def balance_score(value):
                return (any(minute < value.meal_start for minute in value.starts)
                        and any(minute >= value.meal_start + pause_duration for minute in value.starts),
                        min(late_finish, max(minute + visits[index].duration
                                            for index, minute in zip(value.order, value.starts))))
            best = original
            # A hard early group must not consume every later day's share.
            # Unused shares remain available; only checked replacements win.
            day_limit = min(max_states, counter.used + max(1, (max_states - counter.used)
                                                          // (len(balance_days) - position)))
            # Existing fixed-size templates have at most five visits. Share the
            # original bounded counter; exhausted search yields honest partial.
            splits = sorted(range(1, len(indices)), key=lambda value: (abs(2 * value - len(indices)), -value))
            for split in splits:
                if counter.used >= day_limit:
                    counter.limited = True
                    break
                balanced = _template(indices, visits, windows, start, end, distances, counter, day_limit,
                                     attempt_limit=MAX_GROUP_SEARCH_STATES, pause_duration=pause_duration,
                                     split_after=split, late_finish=late_finish)
                if balanced is not None and balance_score(balanced) > balance_score(best):
                    best = balanced
                if balance_score(best) == (True, late_finish):
                    break
            selected[day_index] = best
    coverage.update(states_explored=counter.used, search_complete=not counter.limited,
                    program_days=len(selected), free_days=len(dates) - len(selected),
                    available_day_templates=sum(map(len, templates)),
                    group_search_limits=counter.group_limits,
                    enrichment_visits=enrichment_count)
    if counter.limited:
        coverage["reason_codes"].append("search_limit")
    if not selected:
        return failure("search_limit" if counter.limited else "insufficient_data",
                       "В пределах проверки не найдено ни одного полного дня с программой при исходных "
                       "датах, времени, бюджете и известных условиях. Посещения не растягивались; "
                       "это не утверждение, что других мест в городе нет.", "no_feasible_program_day")

    coverage["quality_status"] = "complete" if len(selected) == len(dates) else "partial"
    if len(selected) < len(dates):
        quality_reasons = ["incomplete_calendar"]
        if sum(visit.duration for visit in visits) < target * len(dates):
            quality_reasons.append("insufficient_confirmed_places")
        if (any(index not in selected and value in rainy for index, value in enumerate(dates))
                and not any(visit.option.indoor is True for visit in visits)):
            quality_reasons.append("indoor_evidence_missing")
        if coverage["exclusions"] or any(visit.windows or visit.option.allowed_dates is not None for visit in visits):
            quality_reasons.append("visit_constraints")
        if budget_groups_rejected:
            quality_reasons.append("budget_constraint")
        if counter.limited:
            quality_reasons.append("search_limit")
        if len(quality_reasons) == 1:
            quality_reasons.append("schedule_constraints")
        coverage["quality_reasons"] = quality_reasons

    days = [_make_day(index + 1, value, selected[index], visits, meal_cents / 100, rest_minutes) if index in selected
            else build_city_free_day(request, index + 1) for index, value in enumerate(dates)]
    itinerary = Itinerary(title=f"{request.destination}: предварительная городская программа",
                          summary="Расписание составлено сервером по проверенным местам и отдельно указанным "
                                  "условиям или плановым оценкам посещений.",
                          days=days, city_schedule=True, tips=[MAP_ESTIMATE_NOTE, TRANSFER_RESERVE_NOTE])
    chosen = {index for template in selected.values() for index in template.order}
    required_satisfied = bool(required_mask & sum(1 << index for index in chosen))
    coverage.update(requested_category_candidates=len(required_ids),
                    requested_category_satisfied=required_satisfied if required_ids else None)
    if required_ids and not required_satisfied:
        coverage["reason_codes"].append("requested_shopping_not_scheduled")
        coverage["quality_status"] = "partial"
        if "requested_interest_missing" not in coverage["quality_reasons"]:
            coverage["quality_reasons"].append("requested_interest_missing")
        # The current coverage report carries this mutable outcome. A stored
        # summary/tip would otherwise keep saying "missing" after a valid edit.
    itinerary.tips.extend(city_managed_tips(itinerary, prepared, request,
                                          options_by_id={visit.candidate.candidate_id: visit.option for visit in visits}))
    if counter.limited:
        itinerary.tips.append("Поиск расписания ограничен: показана найденная проверенная комбинация, "
                              "а не доказательство максимального числа дней с программой.")
    label_city_estimates(itinerary, request)
    try:
        rebuild_program_report(itinerary, request, eligible_unique_visits=len(prepared),
                               allow_city_limited_program=True)
        problems = check_rules(itinerary, request, rainy, allow_city_limited_program=True)
        problems += verify_itinerary(itinerary, verified_records=records)
        problems += check_city_radius(itinerary, (center["lat"], center["lon"]))
        problems += check_verified_place_duplicates(itinerary, verified_records=records)
        problems += _blocked_place_problems(itinerary, blocked)
        problems += check_program_depth(itinerary, request, allow_city_limited_program=True)
        problems += check_city_quality(itinerary, request, records, allow_city_limited_program=True)
        problems += check_city_visit_estimates(itinerary, prepared)
        problems += check_city_visit_conditions(itinerary, prepared, request)
    except (TypeError, ValueError, OverflowError):
        problems = ["Некорректное серверное расписание."]
    if problems:
        coverage["reason_codes"].append("schedule_validation_failed")
        coverage["quality_status"] = "insufficient_data"
        coverage["quality_reasons"] = ["schedule_validation_failed"]
        return DeterministicCityResult(None, list(problems[:12]), "validation_error", coverage)
    apply_city_day_coverage(request, itinerary, coverage)
    coverage.update(program_day_numbers=[index + 1 for index in sorted(selected)],
                    selected_unique_visits=len(chosen),
                    selected_ids_by_day={str(index + 1): [visits[value].candidate.candidate_id for value in template.order]
                                         for index, template in sorted(selected.items())},
                    admission_reserve=sum(visits[index].cost_cents for index in chosen) / 100,
                    unknown_price_visits=sum(visits[index].option.estimated_cost is None for index in chosen),
                    unknown_indoor_visits=sum(visits[index].option.indoor is None for index in chosen))
    return DeterministicCityResult(itinerary, [], "success", coverage)
