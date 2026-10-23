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
import math

from city_candidates import CityCandidate
from city_context import build_city_context
from city_coverage import build_city_free_day
from city_estimates import label_city_estimates
from city_evidence import UNKNOWN_INDOOR_NOTE, city_managed_tips
from city_quality import _family_minimum, check_city_quality, city_candidate_may_have_shelter
from city_visit_estimates import check_city_visit_estimates, get_city_planning_cap
from destination_name_hints import _https_source
from places import MAX_DAY_SPREAD_KM, distance_km, verify_itinerary
from pricing import minimum_meal_cost
from program_coverage import rebuild_program_report
from rules import check_city_radius, check_program_depth, check_rules, check_verified_place_duplicates, program_depth_target
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


MAX_SEARCH_STATES = 20_000
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


def _free_minutes_after(start, reserve, meal_start):
    """Advance a transfer reserve through free time, never through meal/rest."""
    pause_end = meal_start + 120
    if start >= pause_end or start + reserve <= meal_start:
        return start + reserve
    before = max(0, meal_start - start)
    return pause_end + max(0, reserve - before)


def _place_visit(start, duration, windows, meal_start, end):
    for opening, closing in windows:
        value = max(start, opening)
        if value < meal_start + 120 and value + duration > meal_start:
            value = meal_start + 120
        if value + duration <= min(closing, end):
            return value
    return None


def _template(indices, visits, windows, start, end, distances, counter, group_limit):
    # Half-hour lunch alternatives plus source-window boundaries. The fallback
    # slots stay inside the requested day; they are reservations, not venues.
    preferred = min(max(13 * 60, start), end - 120)
    meal_slots = {preferred, start, end - 120}
    meal_slots.update(range(((start + 29) // 30) * 30, end - 119, 30))
    for allowed in windows:
        for opening, closing in allowed:
            meal_slots.update((max(start, min(end - 120, opening - 120)),
                               max(start, min(end - 120, closing))))
    convenient = sorted(meal_slots, key=lambda value: (abs(value - preferred), value))
    # A half-hour grid alone can miss a feasible source window (e.g. lunch
    # starting 10:37). Try convenient slots first, then every other minute;
    # all attempts share the same strict global search budget.
    meal_slots = convenient + sorted(set(range(start, end - 119)) - meal_slots,
                                    key=lambda value: (abs(value - preferred), value))
    by_index = dict(zip(indices, windows))
    for meal_start in meal_slots:
        for order in permutations(indices):
            if not counter.take(group_limit):
                return None
            cursor, starts, previous = start, [], None
            for index in order:
                if previous is not None:
                    km = distances[tuple(sorted((previous, index)))]
                    gap = 0 if km <= .03 else max(TRANSFER_RESERVE_MINUTES, math.ceil(km))
                    cursor = _free_minutes_after(cursor, gap, meal_start)
                value = _place_visit(cursor, visits[index].duration, by_index[index], meal_start, end)
                if value is None:
                    break
                starts.append(value)
                cursor = value + visits[index].duration
                previous = index
            if len(starts) == len(order):
                return _Template(sum(1 << index for index in indices), tuple(order), tuple(starts),
                                 meal_start, sum(visits[index].cost_cents for index in indices))
    return None


def _make_day(number, day_date, template, visits, meal_cost):
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
                      place_query="", description="Пауза для отдыха.", duration_min=60, cost=0),
    ))
    return ItineraryDay(day=number, date=day_date, theme="Городская программа",
                        items=sorted(items, key=lambda item: item.time))


def build_deterministic_city_itinerary(
    request: TripRequest, candidates, rainy_dates=(), *, resolved_city: dict,
    options=None, blocked_places=None, candidate_priority=None, max_states: int = MAX_SEARCH_STATES,
) -> DeterministicCityResult:
    """Construct one fully checked calendar or return no itinerary.

    ``candidates``, ``resolved_city`` and optional ``options`` must be supplied
    by the server, never by the model or public request. Unknown prices retain
    zero *unpriced* reserves under the existing disclosure contract, not an
    assertion of free admission or proof of actual total travel cost.
    """
    from planner import _blocked_place_problems, _max_attractions_for_request
    from city_visit_options import check_city_visit_conditions, visit_option

    coverage = {"schema_version": 1, "reason_codes": [], "exclusions": []}

    def failure(status, message, code):
        coverage["reason_codes"] = list(dict.fromkeys([*coverage["reason_codes"], code]))
        return DeterministicCityResult(None, [message], status, deepcopy(coverage))

    try:
        if type(max_states) is not int or not 1 <= max_states <= MAX_SEARCH_STATES:
            raise ValueError("Лимит поиска должен быть от 1 до 20000 состояний.")
        start, end = _checked_request(request)
        dates = [(date.fromisoformat(request.start_date) + timedelta(days=index)).isoformat()
                 for index in range(request.days_count)]
        if (not isinstance(rainy_dates, (list, tuple)) or len(rainy_dates) > len(dates)
                or any(type(value) is not str or value not in dates for value in rainy_dates)
                or len(set(rainy_dates)) != len(rainy_dates)):
            raise ValueError("Дождливые даты должны быть уникальными датами поездки.")
        rainy = set(rainy_dates)
        if not isinstance(candidates, (list, tuple)) or len(candidates) > 18:
            raise ValueError("Нужен ограниченный серверный набор кандидатов.")
        coverage.update(requested_days=len(dates), confirmed_candidates=len(candidates))
        if not candidates:
            return failure("insufficient_data", "В проверенном наборе нет мест для городской программы. "
                           "Это не означает, что других мест в городе нет.", "empty_pool")
        snapshot = build_city_context(request, candidates, resolved_city)
        prepared = [CityCandidate(**entry) for entry in snapshot["candidates"]]
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
        if options is None:
            option_map = {candidate.candidate_id: visit_option(candidate, request) for candidate in prepared}
        else:
            if not isinstance(options, (list, tuple)) or len(options) > len(prepared):
                raise ValueError("Некорректный набор условий посещений.")
            option_map = {option.candidate_id: option for option in options}
            if len(option_map) != len(options) or set(option_map) - {candidate.candidate_id for candidate in prepared}:
                raise ValueError("Условия посещения содержат неизвестные или повторные ID.")
        if blocked_places is not None and not isinstance(blocked_places, (list, tuple)):
            raise ValueError("Некорректный серверный список исключений.")
        blocked = list(blocked_places or ())
        visits = []
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
        max_visits = _max_attractions_for_request(request)
        if type(max_visits) is not int or not 1 <= max_visits <= 5:
            raise ValueError("Некорректный предел посещений.")
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return failure("invalid_request", "Не удалось проверить анкету, карту или структурированные "
                       "условия посещения для серверного расписания.", "invalid_inputs")

    target = _family_minimum(request) or program_depth_target(request) or 1
    coverage["daily_program_target_minutes"] = target
    distances = {(left, right): distance_km(visits[left].candidate.record["lat"], visits[left].candidate.record["lon"],
                                           visits[right].candidate.record["lat"], visits[right].candidate.record["lon"])
                 for left, right in combinations(range(len(visits)), 2)}
    counter = _Budget(max_states)
    # Keep an explicit portion for choosing a calendar from already found days.
    group_limit = max(1, max_states * 3 // 4)
    templates = [[] for _ in dates]
    template_cache = {}
    for size in range(1, min(max_visits, len(visits)) + 1):
        if counter.used >= group_limit:
            counter.limited = True
            break
        for indices in combinations(range(len(visits)), size):
            if not counter.take(group_limit):
                break
            if (sum(visits[index].duration for index in indices) < target
                    or sum(visits[index].cost_cents for index in indices) > extra_budget
                    or any(distances[pair] > MAX_DAY_SPREAD_KM for pair in combinations(indices, 2))):
                continue
            mask = sum(1 << index for index in indices)
            indoors = sum(visits[index].option.indoor is True for index in indices)
            for day_index, day_date in enumerate(dates):
                if (day_date in rainy and indoors * 2 < size
                        or any(visits[index].option.allowed_dates is not None
                               and day_date not in visits[index].option.allowed_dates for index in indices)
                        or any(template.mask & mask == template.mask for template in templates[day_index])):
                    continue
                windows = tuple(visits[index].windows.get(day_date, ()) if visits[index].windows
                                else ((start, end),) for index in indices)
                if any(not allowed for allowed in windows):
                    continue
                key = (indices, windows)
                if key not in template_cache:
                    template_cache[key] = _template(indices, visits, windows, start, end,
                                                    distances, counter, group_limit)
                if template_cache[key] is not None:
                    templates[day_index].append(template_cache[key])
                if counter.used >= group_limit:
                    break

    def order_key(template, rainy_day):
        indoor_count = sum(visits[index].option.indoor is True for index in template.order)
        if priority:
            # Ranking is only an optimization among feasible groups. It never
            # expands a pool, creates visitor facts or defeats the budget/rain.
            return (0 if rainy_day else indoor_count, tuple(sorted(template.order)),
                    template.cost_cents, len(template.order))
        return (0 if rainy_day else indoor_count, template.cost_cents, len(template.order), template.mask)

    for index in range(len(dates)):
        templates[index].sort(key=lambda value: order_key(value, dates[index] in rainy))
    memo = {}

    def pack(day_index, used, remaining):
        if day_index == len(dates):
            return ()
        key = (day_index, used)  # Nonnegative one-use costs make remaining a function of used.
        if key in memo:
            return memo[key]
        if not counter.take():
            return ()
        best = ()
        for template in templates[day_index]:
            if template.mask & used or template.cost_cents > remaining:
                continue
            rest = pack(day_index + 1, used | template.mask, remaining - template.cost_cents)
            proposal = ((day_index, template),) + rest
            if len(proposal) > len(best):
                best = proposal
            if len(best) == len(dates) - day_index:
                memo[key] = best
                return best
            if counter.used >= max_states:
                break
        if counter.used < max_states:
            skipped = pack(day_index + 1, used, remaining)
            if len(skipped) > len(best):
                best = skipped
        else:
            counter.limited = True
        memo[key] = best
        return best

    selected = dict(pack(0, 0, extra_budget))
    coverage.update(states_explored=counter.used, search_complete=not counter.limited,
                    program_days=len(selected), free_days=len(dates) - len(selected),
                    available_day_templates=sum(map(len, templates)))
    if counter.limited:
        coverage["reason_codes"].append("search_limit")
    if not selected:
        return failure("search_limit" if counter.limited else "insufficient_data",
                       "В пределах проверки не найдено ни одного полного дня с программой при исходных "
                       "датах, времени, бюджете и известных условиях. Посещения не растягивались; "
                       "это не утверждение, что других мест в городе нет.", "no_feasible_program_day")

    days = [_make_day(index + 1, value, selected[index], visits, meal_cents / 100) if index in selected
            else build_city_free_day(request, index + 1) for index, value in enumerate(dates)]
    itinerary = Itinerary(title=f"{request.destination}: предварительная городская программа",
                          summary="Расписание составлено сервером по проверенным местам и отдельно указанным "
                                  "условиям или плановым оценкам посещений.",
                          days=days, city_schedule=True, tips=[MAP_ESTIMATE_NOTE, TRANSFER_RESERVE_NOTE])
    chosen = {index for template in selected.values() for index in template.order}
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
        return DeterministicCityResult(None, list(problems[:12]), "validation_error", coverage)
    coverage.update(program_day_numbers=[index + 1 for index in sorted(selected)],
                    selected_unique_visits=len(chosen),
                    selected_ids_by_day={str(index + 1): [visits[value].candidate.candidate_id for value in template.order]
                                         for index, template in sorted(selected.items())},
                    admission_reserve=sum(visits[index].cost_cents for index in chosen) / 100,
                    unknown_price_visits=sum(visits[index].option.estimated_cost is None for index in chosen),
                    unknown_indoor_visits=sum(visits[index].option.indoor is None for index in chosen))
    return DeterministicCityResult(itinerary, [], "success", coverage)
