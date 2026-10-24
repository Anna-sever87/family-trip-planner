"""Server-owned visit choices; map observations never supply visit conditions.

The fixed planning durations below are estimates, not provider facts or the
upper caps from city_visit_estimates. Unknown indoor/price remain None. The
caller must enforce dates/windows/ages and run the normal final validators.
No HTTP, AI, persistence, currency conversion, or source-catalogue reads.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import date
import math
import re
import unicodedata

import places
from candidate_grounding import _eligible_poi_type
from city_quality import city_candidate_may_have_shelter
from city_visit_estimates import get_city_planning_cap, lookup_city_visit_estimate
from destination_name_hints import _https_source
from schemas import TripRequest
from visit_facts import _windows


@dataclass(frozen=True)
class CityVisitOption:
    candidate_id: str
    duration_min: int
    indoor: bool | None
    estimated_cost: float | None
    description: str
    basis: str
    unknown_fields: tuple[str, ...]
    sources: tuple[dict, ...] = ()
    allowed_dates: tuple[str, ...] | None = None
    opening_windows: tuple[dict, ...] = ()
    children_allowed: bool | None = None
    min_child_age: int | None = None
    restrictions: tuple[str, ...] = ()


_LONG_PLANNING_TYPES = frozenset({"museum", "gallery", "park", "garden"})
_SHORT_PLANNING_TYPES = frozenset({"place_of_worship", "artwork", "monument", "memorial"})
_MEDIUM_PLANNING_TYPES = frozenset({"archaeological_site", "nature_reserve", "beach"})
_LOCALITY_FIELDS = ("city", "town", "village", "hamlet", "suburb", "neighbourhood", "quarter",
                    "locality", "municipality", "county")
_CONDITIONS_CHECKED = "2026-10-22"
_YERSIN_SOURCE = "https://nhatrang.khanhhoa.gov.vn/tin-bai/diem-den/bao-tang-alexandre-yersin-nha-trang-1625"
_OCEAN_ROOMS_SOURCE = "https://baotanghdh.vn/khu-trung-bay-hdh/"
_OCEAN_HOURS_SOURCE = "https://baotanghdh.vn/en/contact-us/"


def _planning_duration(record: dict) -> int:
    """Fixed product blocks, never inferred visit facts or a day-filling target."""
    category, kind = record.get("category"), record.get("type")
    if category == "tourism" and kind == "zoo":
        # Reuse the already reviewed observed subtype/name classifier, without
        # introducing broader name heuristics or any new provider/cache lookup.
        # Lazy import avoids the discovery -> visit options dependency cycle.
        from city_candidates import _specialized_animal_place
        return 60 if _specialized_animal_place(record) else 120
    if category == "tourism" and kind == "theme_park":
        return 120
    if category == "tourism" and kind == "aquarium":
        return 90
    shopping = category == "shop" and kind in {"mall", "shopping_centre"}
    return (90 if kind in _LONG_PLANNING_TYPES or shopping else 30 if kind in _SHORT_PLANNING_TYPES
            else 60 if kind in _MEDIUM_PLANNING_TYPES else 45)


def _fold(value) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split()) if isinstance(value, str) else ""


def _fresh(checked_at, request: TripRequest) -> bool:
    try:
        checked = date.fromisoformat(checked_at)
        return (checked.isoformat() == checked_at and checked <= date.today()
                and 0 <= (date.fromisoformat(request.end_date) - checked).days <= 90)
    except (TypeError, ValueError):
        return False


def _bound(candidate, request: TripRequest) -> bool:
    try:
        record, query = candidate.record, candidate.place_query
        identity = places.map_record_identity(record)
        parts = query.split(", ")
        destination = [part.strip() for part in request.destination.split(",")]
        return (request.destination_kind == "city" and identity is not None
                and candidate.candidate_id == "-".join(map(str, identity))
                and len(parts) == 3 and parts[0] == candidate.title and parts[1] == candidate.area
                and record.get("name", "").split(",", 1)[0].strip() == candidate.title
                and places._valid_record_only_place(query, record)
                and _eligible_poi_type(record) and record.get("type") != "peak"
                and places._area_matches_map_context(candidate.area, {"address": record.get("address")})
                and (len(destination) < 2 or places._country_identity(destination[-1]) == places._country_identity(parts[-1])))
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def _source(field: str, url: str, checked_at: str, basis: str = "published") -> dict:
    return {"field": field, "source_url": url, "checked_at": checked_at, "basis": basis}


def _nha_conditions(candidate, request: TripRequest) -> dict | None:
    """Two reviewed primary sources, exact existing map IDs, never name-only."""
    record = candidate.record
    address = record["address"]
    if (not _fresh(_CONDITIONS_CHECKED, request) or places._country_identity(request.destination.split(",")[-1]) != ("vn",)
            or _fold(request.destination.split(",")[0]) not in {"нячанг", "nha trang"}
            or _fold(address.get("country_code")) != "vn" or record.get("type") != "museum"
            or not {_fold(address.get(key)) for key in _LOCALITY_FIELDS}.intersection({"nha trang", "phường nha trang"})):
        return None
    names = {_fold(candidate.title)}
    details = record.get("namedetails")
    if isinstance(details, dict):
        names.update(_fold(details.get(key)) for key in ("name", "name:vi", "official_name", "official_name:vi"))
    if candidate.candidate_id == "osm-way-1193057654" and names.intersection({
        "bảo tàng alexandre yersin", "bảo tàng yersin", "bào tàng alexandre yersin",
    }):
        return {
            "description": "Осмотр постоянной экспозиции и рабочей комнаты в музее Alexandre Yersin. "
                           "Перед визитом уточните праздничные исключения из обычного графика.",
            "indoor": True, "indoor_source": _YERSIN_SOURCE,
            "hours_source": _YERSIN_SOURCE,
            "hours": [{"weekdays": [0, 1, 2, 3, 4, 5], "start": "07:30", "end": "11:30"},
                      {"weekdays": [0, 1, 2, 3, 4], "start": "14:00", "end": "17:00"}],
            "duration_min": 30, "duration_basis": "published", "duration_source": _YERSIN_SOURCE,
            "cost": 26000 * (request.adults + len(request.children_ages)) if request.currency == "VND" else None,
            "price_source": _YERSIN_SOURCE, "checked_at": _CONDITIONS_CHECKED,
            "restrictions": ("Воскресенье и праздники — закрыто; календарь праздников отдельно не проверен. "
                             "Льготы и условия посещения для состава группы уточнить.",),
        }
    if candidate.candidate_id == "osm-relation-15662995" and names.intersection({
        "bảo tàng hải dương học nha trang", "bảo tàng hải dương học",
    }):
        return {
            "description": "Осмотр коллекций в выставочных залах Музея океанографии Нячанга. "
                           "Плановый блок относится к залам основной площадки, не ко всему кампусу и не к филиалу Hòn Chồng.",
            "indoor": True, "indoor_source": _OCEAN_ROOMS_SOURCE,
            "hours_source": _OCEAN_HOURS_SOURCE,
            "hours": [{"weekdays": list(range(7)), "start": "06:00", "end": "18:00"}],
            "duration_min": 90, "duration_basis": "planning_estimate", "duration_source": _OCEAN_ROOMS_SOURCE,
            "cost": None, "price_source": "", "checked_at": _CONDITIONS_CHECKED,
            "restrictions": ("90 минут — оценка планирования, не опубликованная длительность. "
                             "Часы и доступность выбранных залов перед визитом уточнить.",),
        }
    return None


def _option_from_conditions(candidate, request: TripRequest, values: dict) -> CityVisitOption | None:
    duration = values["duration_min"]
    if type(duration) is not int or not 15 <= duration <= get_city_planning_cap(candidate):
        return None  # Do not shorten a real/sourced visit merely to satisfy a cap.
    indoor = values.get("indoor")
    if indoor is True and not city_candidate_may_have_shelter(candidate.record):
        return None  # A source/map contradiction needs review, not silent preference.
    children, age = values.get("children_allowed"), values.get("min_child_age")
    if children is not None and type(children) is not bool:
        return None
    if age is not None and (type(age) is not int or not 0 <= age <= 18):
        return None
    if request.children_ages and (children is False or age is not None and min(request.children_ages) < age):
        return None
    checked = values["checked_at"]
    windows = tuple({"date": window.date, "start": window.start_time, "end": window.end_time,
                     "basis": window.evidence, "source_url": window.source, "checked_at": checked}
                    for window in _windows(values, request))
    hours_known = bool(values.get("hours") or values.get("sessions")) and _https_source(values.get("hours_source"))
    allowed_dates = tuple(dict.fromkeys(window["date"] for window in windows)) if hours_known else None
    cost = values.get("cost")
    if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
        return None
    sources = [_source("duration", values["duration_source"], checked, values["duration_basis"])]
    if indoor is not None:
        sources.append(_source("indoor", values["indoor_source"], checked))
    if hours_known:
        sources.append(_source("hours", values["hours_source"], checked))
    if cost is not None:
        sources.append(_source("price", values["price_source"], checked))
    unknown = tuple(field for field, known in (("indoor", indoor is not None), ("price", cost is not None),
                    ("hours", hours_known), ("children", children is not None), ("minimum_age", age is not None)) if not known)
    return CityVisitOption(candidate.candidate_id, duration, indoor, cost, values["description"],
                           values["duration_basis"], unknown, tuple(sources), allowed_dates, windows,
                           children, age, tuple(values.get("restrictions", ())))


def _card_conditions(card: dict, request: TripRequest) -> dict | None:
    """Adapt only simple, already loaded source cards; regional rules stay intact."""
    if (not _fresh(card.get("metadata_checked_at"), request) or card.get("route_kind", "point") != "point"
            or card.get("restrictions") or card.get("closed_month_days") or card.get("closed_weekdays")
            or request.adults + len(request.children_ages) > card.get("max_group_size", 100)):
        return None  # Unsupported prose/route/closure semantics are not claimed as enforced.
    if (not _https_source(card.get("duration_source"))
            or card.get("duration_basis") not in {"published", "planning_estimate"}
            or not isinstance(card.get("description"), str) or not isinstance(card.get("hours"), list)):
        return None
    indoor = card.get("indoor") if _https_source(card.get("indoor_source")) else None
    if indoor is not None and type(indoor) is not bool:
        return None
    adult, minimum = card.get("adult_price"), card.get("group_min_price")
    known_cost = None
    if (request.currency == card.get("currency", "RUB") and not request.children_ages
            and _https_source(card.get("price_source")) and type(adult) in (int, float)
            and math.isfinite(adult) and adult >= 0):
        if minimum is not None and (type(minimum) not in (int, float) or not math.isfinite(minimum) or minimum < 0):
            return None
        known_cost = max(minimum or 0, adult * request.adults)
    values = deepcopy(card)
    values.update(checked_at=card["metadata_checked_at"], indoor=indoor, cost=known_cost,
                  min_child_age=card.get("minimum_age"))
    return values


def visit_option(candidate, request: TripRequest, *, visit_cards=()) -> CityVisitOption | None:
    """Choose one server option, not an AI assertion or a guarantee of admission.

    ``visit_cards`` may only be supplied by the server from load_visit_cards;
    exact query binding, freshness and supported semantics are checked again.
    Empty allowed_dates means no sourced date; None means hours are unknown.
    """
    if not _bound(candidate, request):
        return None
    try:
        matches = [card for card in visit_cards if isinstance(card, dict)
                   and candidate.place_query in card.get("place_queries", [])]
        if matches:
            values = _card_conditions(matches[0], request) if len(matches) == 1 else None
            return _option_from_conditions(candidate, request, values) if values is not None else None
        conditions = _nha_conditions(candidate, request)
        if conditions is not None:
            return _option_from_conditions(candidate, request, conditions)
        kind = candidate.record.get("type", "")
        shopping = candidate.record.get("category") == "shop" and kind in {"mall", "shopping_centre"}
        duration = _planning_duration(candidate.record)
        basis, sources = "planning_estimate", ()
        published = lookup_city_visit_estimate(candidate)
        if published is not None and _fresh(published.checked_at, request):
            duration, basis = published.typical_min_minutes, "published"
            sources = (_source("duration", published.source_url, published.checked_at),)
        if duration > get_city_planning_cap(candidate):
            return None  # A cap is not a source for selecting a shorter visit.
        indoor = None if city_candidate_may_have_shelter(candidate.record) else False
        unknown = ("price", "hours", "children", "minimum_age") + (("indoor",) if indoor is None else ())
        activity = "Прогулка и покупки по желанию в" if shopping else "Осмотр"
        description = (f"{activity} «{candidate.title}». Плановый блок {duration} минут — "
                       + ("нижняя граница опубликованной типичной длительности." if basis == "published" else
                          "оценка планирования, не подтверждённая длительность посещения.")
                       + " Часы работы, стоимость и условия посещения уточните заранее.")
        if shopping:
            description += " Личные покупки не включены в расчёт стоимости маршрута."
        if kind in {"theatre", "cinema"}:
            description += " Спектакль или киносеанс в этот блок не включён: расписание не подтверждено."
        return CityVisitOption(candidate.candidate_id, duration, indoor, None, description, basis, unknown, sources)
    except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
        return None


def check_city_visit_conditions(itinerary, candidates, request: TripRequest, *, visit_cards=()) -> list[str]:
    """Recheck confirmed restrictions, not the selected planning durations.

    An indoor source describes the precise option's room programme. This guard
    never upgrades an arbitrary itinerary description/indoor flag into a fact;
    existing model estimates still need the independent quality/rain checks.
    """
    if (getattr(itinerary, "city_schedule", False) is not True
            or getattr(itinerary, "source_schedule", False) is True or request.destination_kind != "city"):
        return []  # No change to the regional/legacy contract.
    by_query, ambiguous = {}, set()
    for candidate in candidates:
        query = getattr(candidate, "place_query", None)
        if query in by_query:
            ambiguous.add(query)
        by_query[query] = candidate
    problems = []
    for day in itinerary.days:
        for item in day.items:
            if item.kind != "attraction":
                continue
            candidate = by_query.get(item.place_query)
            if (candidate is None or item.place_query in ambiguous or not _bound(candidate, request)
                    or item.title != candidate.title or item.area != candidate.area):
                problems.append(f"День {day.day}: условия посещения «{item.title}» не привязаны к точному объекту карты.")
                continue
            option = visit_option(candidate, request, visit_cards=visit_cards)
            if option is None:
                problems.append(f"День {day.day}: подтверждённые ограничения «{item.title}» требуют проверки.")
                continue
            if request.children_ages and (option.children_allowed is False or option.min_child_age is not None
                                         and min(request.children_ages) < option.min_child_age):
                problems.append(f"День {day.day}: возрастные ограничения «{item.title}» не допускают эту программу.")
            if option.allowed_dates is not None and day.date not in option.allowed_dates:
                problems.append(f"День {day.day}: «{item.title}» не имеет опубликованного окна посещения на {day.date}.")
                continue
            if option.allowed_dates is not None:
                valid_time = isinstance(item.time, str) and re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", item.time)
                if not valid_time or type(item.duration_min) is not int or item.duration_min <= 0:
                    problems.append(f"День {day.day}: время посещения «{item.title}» невозможно сверить с опубликованным графиком.")
                    continue
                minute = int(item.time[:2]) * 60 + int(item.time[3:])
                def fits(window):
                    start = int(window["start"][:2]) * 60 + int(window["start"][3:])
                    end = int(window["end"][:2]) * 60 + int(window["end"][3:])
                    return window["date"] == day.date and start <= minute and minute + item.duration_min <= end
                if not any(fits(window) for window in option.opening_windows):
                    problems.append(f"День {day.day}: посещение «{item.title}» целиком не помещается в опубликованное окно работы.")
            if option.indoor is False and item.indoor is True:
                problems.append(f"День {day.day}: «{item.title}» не подтверждает запрошенный осмотр в помещении.")
    return problems
