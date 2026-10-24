"""Server-owned city calendar capacity, not evidence of visit conditions.

The bounded map pool can limit the number of unique program days. It cannot
prove that a destination has no other sights, or that a museum is indoors.
Invalid model programs are never silently converted into free days here.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from itertools import combinations

from city_quality import _family_minimum, city_break_minutes, city_candidate_may_have_shelter
from city_visit_estimates import get_city_planning_cap
from places import MAX_DAY_SPREAD_KM, distance_km
from pricing import minimum_meal_cost
from rules import program_depth_target
from schemas import ItineraryDay, ItineraryItem, TripRequest


CITY_PACKING_MAX_STATES = 20_000


@dataclass(frozen=True)
class CityProgramWitness:
    """A packing of optimistic planning limits, NOT a verified itinerary.

    Groups have no assigned duration, indoor flag, price or opening window.
    Even an unexhausted search establishes only combinatorial feasibility of
    these limits, never actual availability or the destination's full coverage.
    """

    program_day_numbers: tuple[int, ...]
    groups_by_day: dict[int, tuple[str, ...]]
    states_explored: int
    groups_examined: int
    search_exhausted: bool


def city_program_day_capacity(request: TripRequest, candidates) -> int:
    """An upper bound using indivisible visits, never invented duration facts.

    Per-candidate product caps do not prove a visit actually fills its target.
    Reviewed estimates can further tighten those caps. Bounded
    visits are grouped optimistically, respecting the minimum number needed
    and the existing visit-count cap. This is not a feasible route solver.
    Leftover minutes cannot split one visit across dates. Geography, weather
    and the actual selected durations still require all final checks.
    """
    # Reuse the planner's existing cap rather than maintain a second age table.
    # Lazy import avoids a module-initialization cycle in the opt-in planner.
    from planner import _max_attractions_for_request

    target = _family_minimum(request)
    if target is None:
        target = program_depth_target(request)
    if target is None:
        return min(request.days_count, len(candidates))
    long_count, bounded = 0, []
    for candidate in candidates:
        cap = get_city_planning_cap(candidate)
        if cap >= target:
            long_count += 1
        else:
            bounded.append(cap)
    bounded.sort(reverse=True)
    total, needed = 0, None
    for number, cap in enumerate(bounded[:_max_attractions_for_request(request)], start=1):
        total += cap
        if total >= target:
            needed = number
            break
    # Both are necessary upper bounds, not proof the different-sized caps can
    # really be packed into this many days. Work is O(N log N), N <= pool cap.
    bounded_days = min(sum(bounded) // target, len(bounded) // needed) if needed is not None else 0
    return min(request.days_count, long_count + bounded_days)


def build_city_program_witness(request: TripRequest, candidates, rainy_dates=(), *,
                               max_states: int = CITY_PACKING_MAX_STATES) -> CityProgramWitness:
    """Jointly pack unique visits, depth caps, rain counts and pairwise spread.

    The old independent upper bounds could reserve three days even when only
    two disjoint groups satisfy them together. Enumerate minimal groups, then
    search their disjoint packings. At most 24 candidates / five visits per
    group are inspected; subset checks AND packing states share max_states.
    If it expires, return only a packing actually found, possibly empty.

    Rain uses the existing half-of-visits rule, not minutes. A non-excluded
    map candidate is merely potential shelter. Caps are product limits, not
    durations we assign to fill a day. Final schedule/map/rain/budget/quality
    validation remains mandatory and can reject every suggested grouping.
    """
    from planner import _max_attractions_for_request

    if request.destination_kind != "city" or not 1 <= request.days_count <= 14:
        raise ValueError("Нужна городская поездка на 1–14 дней.")
    if type(max_states) is not int or not 1 <= max_states <= CITY_PACKING_MAX_STATES:
        raise ValueError("Лимит поиска должен быть целым числом от 1 до 20000.")
    candidates = list(candidates)
    if len(candidates) > 24:
        raise ValueError("Для ограниченного поиска допускается не более 24 кандидатов.")
    first = date.fromisoformat(request.start_date)
    dates = [(first + timedelta(days=index)).isoformat() for index in range(request.days_count)]
    rainy = set(rainy_dates)
    if not rainy.issubset(dates):
        raise ValueError("Дождливые даты должны принадлежать поездке.")
    identifiers = [candidate.candidate_id for candidate in candidates]
    if any(not isinstance(value, str) or not value for value in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Нужны уникальные серверные ID кандидатов.")
    dry = [index + 1 for index, value in enumerate(dates) if value not in rainy]
    wet = [index + 1 for index, value in enumerate(dates) if value in rainy]
    # Stable even when the same grounded pool arrives in a different order.
    candidates.sort(key=lambda candidate: candidate.candidate_id)
    identifiers = [candidate.candidate_id for candidate in candidates]
    coordinates, caps, shelter = [], [], []
    for candidate in candidates:
        record = candidate.record
        point = (record.get("lat"), record.get("lon")) if isinstance(record, dict) else (None, None)
        if any(type(value) not in (int, float) for value in point):
            raise ValueError("Для группировки нужны проверенные конечные координаты всех кандидатов.")
        # Bounds also reject NaN/infinity without converting arbitrarily large
        # integers to float (which could itself raise OverflowError).
        if not -90 <= point[0] <= 90 or not -180 <= point[1] <= 180:
            raise ValueError("Координаты кандидата выходят за допустимые пределы.")
        cap = get_city_planning_cap(candidate)
        if type(cap) is not int or cap <= 0:
            raise ValueError("Некорректный предел планирования кандидата.")
        coordinates.append(point)
        caps.append(cap)
        shelter.append(city_candidate_may_have_shelter(record))
    target = _family_minimum(request)
    if target is None:
        target = program_depth_target(request)
    target = target or 1
    max_visits = _max_attractions_for_request(request)
    if type(max_visits) is not int or not 1 <= max_visits <= 5:
        raise ValueError("Некорректный предел числа посещений.")
    compatible = {
        (left, right): distance_km(*coordinates[left], *coordinates[right]) <= MAX_DAY_SPREAD_KM
        for left, right in combinations(range(len(candidates)), 2)
    }
    dry_groups, wet_groups = [], []
    states, examined, exhausted = 0, 0, False
    group_limit = min(max_states - 1, max(1, max_states * 3 // 4))
    for size in range(1, min(max_visits, len(candidates)) + 1):
        for indices in combinations(range(len(candidates)), size):
            # Reserve a quarter of the unchanged global budget for packing;
            # a larger bounded pool must not leave only one calendar state.
            if states >= group_limit:
                exhausted = True
                break
            states += 1
            examined += 1
            if sum(caps[index] for index in indices) < target:
                continue
            if any(not compatible[pair] for pair in combinations(indices, 2)):
                continue
            mask = sum(1 << index for index in indices)
            # A strict feasible subset always dominates its superset: it uses
            # fewer unique IDs and leaves at least as many choices for others.
            if not any(group & mask == group for group in dry_groups):
                dry_groups.append(mask)
            if (sum(shelter[index] for index in indices) * 2 >= size
                    and not any(group & mask == group for group in wet_groups)):
                wet_groups.append(mask)
        if exhausted:
            break

    def group_order(mask):
        # Preserve scarce potential shelter for wet dates when choosing dry
        # groups. This is traversal order only, never positive indoor evidence.
        return (sum(shelter[index] for index in range(len(candidates)) if mask & (1 << index)),
                mask.bit_count(), mask)

    dry_groups.sort(key=group_order)
    wet_groups.sort(key=lambda mask: (mask.bit_count(), mask))
    by_anchor = [
        ([mask for mask in dry_groups if mask & (1 << index)],
         [mask for mask in wet_groups if mask & (1 << index)])
        for index in range(len(candidates))
    ]
    memo = {}

    def score(groups):
        return (len(groups[0]) + len(groups[1]), len(groups[0]))

    def pack(available, dry_left, wet_left):
        nonlocal states, exhausted
        if not available or not dry_left + wet_left:
            return (), ()
        key = (available, dry_left, wet_left)
        if key in memo:
            return memo[key]
        if states >= max_states:
            exhausted = True
            return (), ()
        states += 1
        anchor = (available & -available).bit_length() - 1
        best = ((), ())
        for mode, left in enumerate((dry_left, wet_left)):
            if not left:
                continue
            for mask in by_anchor[anchor][mode]:
                if mask & available != mask:
                    continue
                rest = pack(available ^ mask, dry_left - (mode == 0), wet_left - (mode == 1))
                proposal = ((mask,) + rest[0], rest[1]) if mode == 0 else (rest[0], (mask,) + rest[1])
                if score(proposal) > score(best):
                    best = proposal
                if score(best) == (dry_left + wet_left, dry_left):
                    memo[key] = best
                    return best
                if states >= max_states:
                    break
            if states >= max_states:
                break
        if states < max_states:
            skipped = pack(available ^ (1 << anchor), dry_left, wet_left)
            if score(skipped) > score(best):
                best = skipped
        else:
            exhausted = True
        memo[key] = best
        return best

    dry_packing, wet_packing = pack((1 << len(candidates)) - 1, len(dry), len(wet))
    groups_by_day = {}
    for numbers, packing in ((dry, dry_packing), (wet, wet_packing)):
        for number, mask in zip(numbers, packing):
            groups_by_day[number] = tuple(identifiers[index] for index in range(len(candidates))
                                          if mask & (1 << index))
    groups_by_day = dict(sorted(groups_by_day.items()))
    return CityProgramWitness(tuple(groups_by_day), groups_by_day, states, examined, exhausted)


def select_program_days(request: TripRequest, candidates, rainy_dates=()) -> list[int]:
    """Compatibility API: dates backed by one bounded optimistic packing."""
    return list(build_city_program_witness(request, candidates, rainy_dates).program_day_numbers)


def build_city_free_day(request: TripRequest, number: int) -> ItineraryDay:
    """Unlocated meal/rest reservations; no hotel, attraction or fake map data."""
    if type(number) is not int or not 1 <= number <= request.days_count:
        raise ValueError("Неверный номер свободного дня.")
    def minutes(value):
        hour, minute = map(int, value.split(":"))
        if not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError("Некорректное время дня.")
        return hour * 60 + minute
    start, end = minutes(request.start_time), minutes(request.end_time)
    if end - start < 120:
        raise ValueError("Для резервов обеда и отдыха нужно окно не короче двух часов.")
    meal_time = max(start, min(13 * 60, end - 120))
    def clock(value):
        return f"{value // 60:02d}:{value % 60:02d}"
    return ItineraryDay(
        day=number, date=(date.fromisoformat(request.start_date) + timedelta(days=number - 1)).isoformat(),
        theme="Свободный день", day_mode="free", base_area="", items=[
            ItineraryItem(time=clock(meal_time), kind="meal", title="Обед", duration_min=60,
                          description="Плановый резерв на обед для всей группы; заведение и цена не подтверждены.",
                          cost=minimum_meal_cost(request)),
            ItineraryItem(time=clock(meal_time + 60), kind="rest", title="Отдых",
                          duration_min=city_break_minutes(request),
                          description="Пауза по желанию. Остальное время свободного дня не заполнено посещениями.", cost=0),
        ],
    )
