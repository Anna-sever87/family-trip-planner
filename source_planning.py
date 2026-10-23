"""Source-backed regional scheduling and explicit provenance for public output.

The model proposes candidates; this path never asks it to invent missing hours,
prices, durations or road connections. Unknown regions retain the normal path.
"""
from dataclasses import dataclass
import re
from types import SimpleNamespace

from pricing import minimum_meal_cost
from program_coverage import rebuild_program_report
from regional_routing import ATTRIBUTION, build_travel_graph, verify_route_edges, verify_selected_routes
from regional_skeleton import SkeletonCoverage, _visit_problem, assess_visit_coverage, build_regional_skeleton
from rules import _CHILD_WORD_RE
from schemas import Itinerary, ItineraryDay, ItineraryItem
from visit_facts import load_visit_cards, visit_facts_for_grounding


@dataclass
class SourcePlan:
    itinerary: Itinerary
    problems: list[str]
    depth_by_day: dict[int, int]
    status: str
    coverage: SkeletonCoverage | None = None


def _source_failure(marker, request, usable_cards, status, diagnostics=(), *, coverage=None):
    """Stable public category plus bounded coverage; details stay diagnostic."""
    allowed = {"SOURCE_VISIT_DATA_INCOMPLETE", "SOURCE_SEARCH_LIMIT",
               "SOURCE_CONSTRAINTS_INCOMPATIBLE", "SOURCE_ROUTES_UNCONFIRMED", "SOURCE_CANDIDATE_CAPACITY"}
    header = "SOURCE_PLAN_UNAVAILABLE:" + (f" {marker}" if marker in allowed else "")
    for key, value in (("requestedDays", request.days_count), ("usableCards", usable_cards)):
        if type(value) is int and 0 <= value <= 100:
            header += f" {key}={value}"
    if isinstance(coverage, SkeletonCoverage):
        for key, value, bound in (
            ("uniqueCandidates", coverage.unique_candidates, 100),
            ("targetMinutes", coverage.target_minutes_per_day, 1440),
            ("availableMinutes", coverage.available_visit_minutes, 60000),
            ("uncoveredDays", sum(row.unique_candidates == 0 for row in coverage.by_date), 14),
            ("excludedVisits", len(coverage.exclusions), 100),
            ("firstDayMaxMinutes", coverage.first_day_max_minutes, 1440),
        ):
            if type(value) is int and 0 <= value <= bound:
                header += f" {key}={value}"
        if coverage.reason in {"unique_candidates", "dates_without_visits", "first_day_depth_capacity"}:
            header += f" coverageReason={coverage.reason}"
    return SourcePlan(Itinerary(title="Маршрут пока не составлен", summary="", days=[], tips=[]),
                      [header, *diagnostics], {}, status, coverage)


def _skeleton_failure_marker(result):
    if result.status == "search_limit":
        return "SOURCE_SEARCH_LIMIT"
    if result.status == "insufficient_content":
        if result.coverage is not None and result.coverage.reason in {
            "unique_candidates", "dates_without_visits", "first_day_depth_capacity",
        }:
            return "SOURCE_CANDIDATE_CAPACITY"
        return "SOURCE_CONSTRAINTS_INCOMPATIBLE"
    if result.status == "needs_facts":
        if any(item.startswith("visit:") for item in result.missing_facts):
            return "SOURCE_VISIT_DATA_INCOMPLETE"
        if any(item.startswith("travel:") for item in result.missing_facts):
            return "SOURCE_ROUTES_UNCONFIRMED"
        return "SOURCE_VISIT_DATA_INCOMPLETE"
    return ""  # Unknown future states must not be mislabelled as missing data.


def _visit_description(fact, request=None):
    duration = ("Опубликованная длительность" if fact.duration_basis == "published"
                else "Плановая оценка длительности, не обещание организатора")
    price = (f"Опубликованная стоимость для группы: {fact.group_cost:g} ₽."
             if fact.group_cost is not None else
             f"Резерв на посещение: {fact.planning_reserve:g} ₽; точная цена неизвестна.")
    notes = [fact.visitor_note] if fact.visitor_note else [fact.basis_note, fact.price_note, *fact.restrictions]
    urls = list(dict.fromkeys(url for url in (fact.duration_source, fact.price_source) if url))
    parts = [part for part in (
        fact.description, f"{duration}: {fact.duration_min} мин.",
        price, *notes,
        "Часы и доступность на дату нужно перепроверить перед поездкой.",
        "Источники: " + " ".join(urls),
    ) if part]
    if request is not None and not request.children_ages:
        parts = [sentence for part in parts for sentence in re.split(r"(?<=[.!?;])\s+", part)
                 if not _CHILD_WORD_RE.search(sentence)]
    return " ".join(parts)


def _travel_note(slots, labels):
    lines = []
    for slot in slots:
        edge = slot.travel_fact
        if edge is None:
            continue
        cost = ((edge.group_cost if edge.group_cost is not None else edge.planning_reserve) or 0)
        lines.append(f"{slot.start_time}–{slot.end_time}: {labels.get(edge.from_key, edge.from_key)} → "
                     f"{labels.get(edge.to_key, edge.to_key)}, {edge.mode}, {edge.minutes} мин; "
                     f"{'резерв' if edge.group_cost is None else 'расход'} {cost:g} ₽.")
    return " ".join(lines) + " Время — оценка дорожного графа с запасом, не расписание перевозчика."


def itinerary_from_skeleton(request, grounding, skeleton):
    if skeleton.status not in ("ready", "limited_program") or len(skeleton.days) != request.days_count:
        raise ValueError("Нельзя публиковать частичное расписание")
    free_days = [day for day in skeleton.days if day.is_free_day]
    if (bool(free_days) != (skeleton.status == "limited_program")
            or len(free_days) == len(skeleton.days)):
        raise ValueError("Свободные дни должны быть явно обозначены в полном календаре с программой")
    labels = {base["base_key"]: base["base_query"] for base in grounding.base_facts}
    for day in skeleton.days:
        labels.update({slot.visit_id: slot.title for slot in day.slots if slot.visit_id})
    days = []
    for day in skeleton.days:
        travel_total = sum(slot.budget_amount for slot in day.slots if slot.kind == "travel")
        items = []
        for slot in day.slots:
            if slot.kind == "travel":
                continue  # Genuine free intervals, never disguised as a visit.
            fact = slot.visit_fact
            description = _visit_description(fact, request) if fact else slot.description
            cost = slot.budget_amount
            if slot.kind == "meal":
                cost += travel_total
                description += (f" В сумме этого блока отдельно учтены: обед — {slot.budget_amount:g} ₽; "
                                f"дорожные расходы/резерв дня — {travel_total:g} ₽. Это не цена обеда в кафе.")
            items.append(ItineraryItem(time=slot.start_time, kind=slot.kind,
                                       title="Обед и транспортный резерв" if slot.kind == "meal" and travel_total else slot.title,
                                       place_query=slot.place_query, description=description,
                                       duration_min=slot.duration_min, cost=round(cost, 2),
                                       indoor=fact.indoor if fact else slot.kind == "meal",
                                       kids_note=fact.kids_note if fact and request.children_ages else ""))
        days.append(ItineraryDay(day=day.day, date=day.date,
                                 theme="Свободный день" if day.is_free_day else " и ".join(
                                     slot.title for slot in day.slots if slot.kind == "attraction"),
                                 items=items, base_area=day.base_query,
                                 day_mode="free" if day.is_free_day else "program",
                                 transfer_note="" if day.is_free_day else _travel_note(day.slots, labels)))
    itinerary = Itinerary(
        title=f"{request.destination}: план поездки по проверенным сведениям",
        summary=("Предварительный план, не бронирование. Время самостоятельных посещений — плановая оценка; "
                 "известные цены и резервы указаны отдельно. "
                 f"Программа: {skeleton.known_cost_total:g} ₽ по опубликованным тарифам и "
                 f"{skeleton.planning_reserve_total:g} ₽ резервов. "
                 f"Введённые дорога и проживание: {request.fixed_costs:g} ₽."),
        days=days, source_schedule=True,
        tips=["Перед поездкой перепроверьте график, билеты, разрешения и прогноз по каждой базе. "
              "Запись и свободные места не подтверждены автоматически.",
              "Для автомобиля/такси заложен плановый резерв 45 ₽/км, минимум 500 ₽ за переезд; "
              "это не тариф. Пешие переходы не требуют транспортного билета.", ATTRIBUTION],
    )
    if free_days:
        if skeleton.coverage is None:
            raise ValueError("Для ограниченной программы нужен отчёт о проверенном наборе")
        rebuild_program_report(itinerary, request,
                               eligible_unique_visits=skeleton.coverage.unique_candidates)
        itinerary.tips.append("Если в маршруте есть свободные дни, в них учтён резерв на обед. Самостоятельные развлечения, "
                              "покупки и другие личные расходы не запланированы; оплачивайте их из остатка бюджета.")
    return itinerary


def prepare_source_plan(request, grounding, *, rainy_dates=()):
    cards = load_visit_cards(request.destination)
    if not cards:
        return None
    facts = visit_facts_for_grounding(request, grounding)
    # Synthetic/other pools and currencies without matching evidence retain
    # the legacy path. A partially supported actual pool must not fabricate the
    # rest of its schedule with another paid model retry.
    catalog_queries = {query for card in cards for query in card["place_queries"]}
    if not any(hint.get("place_query") in catalog_queries for hint in grounding.confirmed_hints):
        return None
    coverage = assess_visit_coverage(request, grounding, facts)
    if not facts:
        return _source_failure("SOURCE_VISIT_DATA_INCOMPLETE", request, 0, "needs_facts", coverage=coverage)
    usable_facts = [fact for fact in facts if _visit_problem(fact, request) is None]
    if not usable_facts:
        marker, status = (("SOURCE_VISIT_DATA_INCOMPLETE", "needs_facts") if coverage.missing_facts
                          else ("SOURCE_CANDIDATE_CAPACITY", "insufficient_content"))
        return _source_failure(marker, request, len(usable_facts), status, coverage=coverage)
    graph, warnings = build_travel_graph(grounding, usable_facts)
    result = build_regional_skeleton(request, grounding, facts, travel_facts=graph,
                                    rainy_dates=rainy_dates, allow_free_days=True)
    if result.status not in ("ready", "limited_program"):
        return _source_failure(_skeleton_failure_marker(result), request, len(usable_facts), result.status,
                               warnings, coverage=result.coverage)
    route_problems = verify_selected_routes(result, grounding, usable_facts)
    if route_problems:
        return _source_failure("SOURCE_ROUTES_UNCONFIRMED", request, len(usable_facts), "needs_facts",
                               route_problems, coverage=result.coverage)
    return SourcePlan(itinerary_from_skeleton(request, grounding, result), [],
                      {day.day: day.depth_credit_min for day in result.days}, result.status, result.coverage)


def _minute(value):
    hours, minutes = map(int, value.split(":"))
    return hours * 60 + minutes


def _free_start(at, until, items, duration):
    intervals = [(at, until)]
    for pause in (part for part in items if part.kind in ("meal", "rest")):
        left, right = _minute(pause.time), _minute(pause.time) + pause.duration_min
        intervals = [(a, min(b, left)) for a, b in intervals if a < min(b, left)] + [
            (max(a, right), b) for a, b in intervals if max(a, right) < b]
    return min((a for a, b in intervals if b - a >= duration), default=None)


def validate_source_edit(request, itinerary):
    """Recheck an edited source schedule using new map-verified identities.

    Runs after normal map/rule validation. No cached itinerary flag or model
    wording can grant source/routing evidence to a different place or time.
    """
    import places
    bases, hints, keys = [], [], {}
    for day in itinerary.days:
        if day.base_area not in keys:
            record = places.geocode(day.base_area)
            if not record:
                return ["Не подтверждена база изменённого расписания."], {}
            key = f"base_{len(keys) + 1}"
            keys[day.base_area] = key
            bases.append(dict(base_key=key, base_query=day.base_area, lat=record["lat"], lon=record["lon"]))
        for item in day.items:
            if item.kind == "attraction":
                if not item.verified:
                    return ["Не подтверждено место изменённого расписания."], {}
                hints.append(dict(title=item.title, place_query=item.place_query, lat=item.lat, lon=item.lon,
                                  base_key=keys[day.base_area], allowed_base_keys=[keys[day.base_area]]))
    grounding = SimpleNamespace(base_facts=bases, confirmed_hints=hints)
    facts = visit_facts_for_grounding(request, grounding)
    by_query = {fact.place_query: fact for fact in facts}
    for day in itinerary.days:
        for item in day.items:
            if item.kind != "attraction":
                continue
            fact = by_query.get(item.place_query)
            if (fact is None or _visit_problem(fact, request) is not None
                    or item.duration_min != fact.duration_min or item.indoor != fact.indoor
                    or not any(window.date == day.date and window.start_time <= item.time
                               and _minute(item.time) + item.duration_min <= _minute(window.end_time)
                               for window in fact.date_windows)):
                return [f"День {day.day}: изменение не согласовано с длительностью и часами посещения из источника."], {}
            expected_cost = fact.group_cost if fact.group_cost is not None else fact.planning_reserve
            if expected_cost is None or abs(item.cost - expected_cost) > .001:
                return [f"День {day.day}: стоимость посещения не соответствует тарифу или явному резерву."], {}
    graph, warnings = build_travel_graph(grounding, facts)
    edges = {(edge.from_key, edge.to_key): edge for edge in graph}
    credits, problems, previous, notes, selected_edges = {}, [], None, {}, []
    labels = {base["base_key"]: base["base_query"] for base in bases}
    labels.update({fact.visit_id: fact.place_query.split(",")[0] for fact in facts})
    for day in itinerary.days:
        base = keys[day.base_area]
        visits = sorted((item for item in day.items if item.kind == "attraction"), key=lambda item: item.time)
        if day.day_mode == "free":
            # Only the server's explicitly reported free days may have no
            # visits. They stay at their base and still pay the daily meal floor.
            if (visits or itinerary.program_report is None
                    or (previous is not None and previous != base)
                    or any(item.kind not in ("meal", "rest") for item in day.items)
                    or sum(item.cost for item in day.items) + 0.001 < minimum_meal_cost(request)):
                problems.append(f"День {day.day}: свободный режим не согласован с базой, программой или бюджетом.")
                continue
            credits[day.day] = 0
            notes[day.day] = ""
            previous = base
            continue
        if not visits:
            problems.append(f"День {day.day}: в дне с программой нет подтверждённых посещений.")
            continue
        legs, slots, node, at = [], [], base, _minute(request.start_time)
        def record_leg(edge, begin):
            legs.append(edge)
            slots.append(SimpleNamespace(travel_fact=edge, start_time=f"{begin//60:02d}:{begin%60:02d}",
                                         end_time=f"{(begin+edge.minutes)//60:02d}:{(begin+edge.minutes)%60:02d}"))
        if previous and previous != base:
            edge = edges.get((previous, base))
            begin = None if edge is None else _free_start(at, _minute(visits[0].time), day.items, edge.minutes)
            if edge is None or begin is None:
                problems.append(f"День {day.day}: не подтверждён межбазовый переезд.")
                continue
            record_leg(edge, begin)
            at = begin + edge.minutes
        for item in [*visits, None]:
            destination = by_query[item.place_query].visit_id if item else base
            edge = edges.get((node, destination))
            until = _minute(item.time) if item else _minute(request.end_time)
            begin = None if edge is None else _free_start(at, until, day.items, edge.minutes)
            if edge is None or begin is None:
                problems.append(f"День {day.day}: не оставлено непрерывного времени для подтверждённого переезда.")
                break
            record_leg(edge, begin)
            node = destination
            if item:
                at = _minute(item.time) + item.duration_min
        known_visit_cost = sum((by_query[item.place_query].group_cost if by_query[item.place_query].group_cost is not None
                                else by_query[item.place_query].planning_reserve) or 0 for item in visits)
        required = known_visit_cost + minimum_meal_cost(request) + sum(
            (edge.group_cost if edge.group_cost is not None else edge.planning_reserve) or 0 for edge in legs)
        if sum(item.cost for item in day.items) + 0.001 < required:
            problems.append(f"День {day.day}: в бюджете не учтены цены или дорожные резервы.")
        credits[day.day] = sum(edge.depth_credit_min for edge in legs)
        selected_edges.extend(legs)
        notes[day.day] = _travel_note(slots, labels)
        previous = base
    if not problems:
        problems += verify_route_edges(selected_edges, grounding, facts)
    if not problems:
        # The model may reword the visit but cannot remove source limitations or
        # retain stale road instructions after moving its start time.
        for day in itinerary.days:
            day.transfer_note = notes[day.day]
            for item in day.items:
                if item.kind == "attraction":
                    item.description = _visit_description(by_query[item.place_query], request)
        itinerary.summary = ("Предварительный план, не бронирование. Часы, цены и доступность требуют повторной проверки. "
                             "Стоимость посещений и транспортные резервы указаны в пунктах; "
                             "актуальный итог рассчитан в бюджете поездки.")
    return problems, {} if problems else credits
