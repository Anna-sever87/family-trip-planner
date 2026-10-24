"""Bounded product constraints for the opt-in, map-only city planner.

These checks do NOT verify admission prices, opening hours, indoor access,
medical suitability, visit duration or actual road/walking travel. A museum
record is not evidence of shelter. The caller must separately run schema,
map-record verification, duplicate, ordinary schedule/budget and rain checks.
No lookup, cache access, mutation or automatic metadata correction occurs here.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Mapping

from places import _context_identity, _country_identity, distance_km, map_record_identity
from program_coverage import authorized_free_days
from rules import _schedule_is_explicit, _time_minutes
from schemas import Itinerary, TripRequest


CITY_QUALITY_RULE = """ОГРАНИЧЕННЫЕ ПРОВЕРКИ КАЧЕСТВА ГОРОДСКОГО MAP-ONLY ПЛАНА:
Кандидат типа peak не допускается в этом режиме: координата вершины не подтверждает доступное посещение, путь подъёма, подъёмник или его стоимость. Нужен отдельно проверенный посетительский объект и сведения о доступе; не подменяй их модельными оценками. Это ограничение map-only продукта, а не утверждение о недоступности всех вершин и не VisitFacts.
Тип карты map_type/type не доказывает помещение, часы, цены или пригодность для ребёнка. Для кандидатов park, garden, beach, viewpoint, nature_reserve, waterfall, peak, lake, artwork, monument, memorial, archaeological_site нельзя ставить indoor=true: для крытого посещения нужен отдельный подтверждённый на карте крытый объект. Не переименовывай кандидата и не выдумывай крытую часть. Тип museum также не подтверждает indoor; это по-прежнему неподтверждённая оценка.
Это отрицательные ограничения, а не общий запрет indoor=true. Для отдельного объекта без этих запретов обоснованная программа осмотра внутренних залов может иметь indoor=true как модельную оценку, не как проверенный факт карты. Не выводи эту оценку только из названия или типа museum; при неизвестной программе не выдумывай крытое посещение ради дождя.
Явное название карты «open-air/open air», «Açık Hava», «под открытым небом» или «ngoài trời» также исключает indoor=true, даже при типе museum. Не игнорируй указание открытого воздуха в названии объекта или его официальных вариантах названия.
В этом режиме остановки artwork, monument, memorial не должны длиться больше 60 минут. Это ограничение продукта против искусственного растягивания, а не установленная длительность посещения.
Если есть дети, темп средний и явно заданное окно дня не короче 8 часов, суммарные оценки посещений должны составлять не меньше 90 минут при младшем ребёнке до 7 лет, иначе 120 минут. Обед, отдых и дорога в этот минимум не входят. Это ориентир содержательности, не медицинская рекомендация и не подтверждение длительностей.
Между последовательными проверенными посещениями оставляй свободные минуты не меньше округлённого вверх расстояния по прямой в километрах; точки до 30 метров считаются совместно расположенными. Из промежутка исключаются все минуты обеда и отдыха, включая пересечения пауз. Это только крайне оптимистичная геометрическая нижняя оценка при 60 км/ч, не расчёт пешей дороги, маршрута или доступного транспорта. Прохождение этой проверки не доказывает выполнимость переезда. Не используй отель/базу без проверенных координат и не добавляй выдуманные источники."""


_NO_INDOOR_EVIDENCE_TYPES = frozenset({
    "park", "garden", "beach", "viewpoint", "nature_reserve", "waterfall",
    "peak", "lake", "artwork", "monument", "memorial", "archaeological_site",
})
_SHORT_STOP_TYPES = frozenset({"artwork", "monument", "memorial"})

# A narrow hard constraint, not a replacement for the semantic wish selector.
# No place names, guessed translations or radius-as-boundary rule belong here.
_CITY_ONLY = re.compile(
    r"\b(?:в\s+(?:пределах|границах|черте)\s+(?:(?:выбранного|этого|самого)\s+)?города|"
    r"(?:только|строго|исключительно)\s+(?:в\s+(?:самом\s+)?городе|по\s+городу)|"
    r"(?:не\s+выезжать|без\s+выездов)\s+за\s+(?:пределы\s+города|город)|"
    r"(?:only|strictly)\s+(?:within|in)\s+(?:the\s+)?city|"
    r"within\s+(?:the\s+)?city\s+(?:limits|boundaries)|"
    r"(?:do\s+not|don't)\s+leave\s+(?:the\s+)?city)\b")
_NEGATED_CITY_ONLY = re.compile(
    r"\b(?:не|not)(?:\s+(?:только|хочу|хотим|нужно|надо|обязательно|оставаться|ограничива\w*|маршрут|"
    r"only|want|need|limited|restricted|staying|to)){0,3}\s*$")
CITY_ONLY_WARNING = (
    "Запрошена программа только в выбранном городе. Принадлежность части мест этому населённому пункту "
    "не подтверждена адресом карты; они не включены. Радиус поиска не считается границей города. "
    "Это не означает, что подходящих мест в городе нет."
)


def city_only_requested(request: TripRequest) -> bool:
    """Recognize explicit current RU/EN scope, never exclusions from memory."""
    if request.destination_kind != "city":
        return False
    for raw in (request.trip_goal, request.wishes):
        text = unicodedata.normalize("NFKC", raw).casefold().replace("ё", "е").replace("’", "'")
        for clause in re.split(r"[.!?;\n]+", text):
            for match in _CITY_ONLY.finditer(clause):
                if _NEGATED_CITY_ONLY.search(clause[:match.start()]):
                    continue
                if re.match(r"\s+не\s+(?:обязательно|нужно|надо|хочу|хотим)\b", clause[match.end():]):
                    continue
                return True
    return False


def city_only_record_allowed(request: TripRequest, record: Mapping | None) -> bool:
    """Necessary source-address evidence, NOT an inferred municipal polygon.

    Callers still independently verify the provider record, identity, country,
    coordinates and radius. Only existing geographic aliases are recognized;
    a POI's title/name:ru, county, district or proximity cannot prove its city.
    Missing or differently written locality is unconfirmed, not 'outside'.
    """
    if not city_only_requested(request):
        return True
    if not isinstance(record, dict) or map_record_identity(record) is None:
        return False
    parts = [part.strip() for part in request.destination.split(",")]
    address = record.get("address")
    if len(parts) < 2 or any(not part for part in parts) or not isinstance(address, dict):
        return False
    expected_country = _country_identity(parts[-1])
    code = address.get("country_code")
    country = address.get("country")
    if (not isinstance(code, str) or not re.fullmatch(r"[a-zA-Z]{2}", code)
            or not isinstance(country, str) or not country.strip()):
        return False
    actual_country = _country_identity(country)
    expected_code = expected_country[0] if len(expected_country) == 1 and len(expected_country[0]) == 2 else ""
    actual_code = actual_country[0] if len(actual_country) == 1 and len(actual_country[0]) == 2 else ""
    if (expected_code and code.lower() != expected_code
            or actual_code and code.lower() != actual_code
            or not expected_code and expected_country != actual_country):
        return False
    target = _context_identity(parts[0])
    localities = [address.get(key) for key in ("city", "town") if address.get(key)]
    if not target or not localities or any(not isinstance(value, str) or _context_identity(value) != target
                                           for value in localities):
        return False
    # A smaller separately named settlement is not promoted to the city merely
    # because a broader address field contains the requested municipality.
    for key in ("village", "hamlet"):
        value = address.get(key)
        if value and (not isinstance(value, str) or _context_identity(value) != target):
            return False
    return True


def _name_words(value: str) -> tuple[str, ...]:
    # Match adjacent complete words, including Unicode hyphens/case variants.
    # Turkish dotted/dotless I differs under ordinary Unicode casefold.
    normalized = unicodedata.normalize("NFKC", value).casefold().replace("ı", "i").replace("\u0307", "")
    return tuple("".join(character if unicodedata.category(character)[0] in {"L", "N"} else " "
                         for character in normalized).split())


_OPEN_AIR_PHRASES = tuple(_name_words(value) for value in
                          ("open air", "Açık Hava", "под открытым небом", "ngoài trời"))


def _map_type(record: Mapping) -> str:
    value = record.get("type")
    return value.strip().casefold() if isinstance(value, str) else ""


def city_candidate_may_have_shelter(record: Mapping) -> bool:
    """Negative filter only: True is NOT evidence of an indoor visit.

    The caller supplies a grounded provider observation, never model text.
    Only POI type and names are inspected, not addresses or descriptions.
    Explicit open-air names defeat a generic museum/gallery type. Missing or
    other names do not prove shelter; all ordinary rain checks still apply.
    """
    if not isinstance(record, Mapping) or _map_type(record) in _NO_INDOOR_EVIDENCE_TYPES:
        return False
    names = []
    name = record.get("name") or record.get("display_name")
    if isinstance(name, str):
        names.append(name.split(",", 1)[0])  # The rest of a display name is address context.
    details = record.get("namedetails")
    if isinstance(details, Mapping):
        names.extend(value for key, value in details.items()
                     if isinstance(key, str) and isinstance(value, str)
                     and (key in {"name", "official_name"} or key.startswith(("name:", "official_name:"))))
    for name in names:
        words = _name_words(name)
        if any(words[index:index + len(phrase)] == phrase
               for phrase in _OPEN_AIR_PHRASES for index in range(len(words) - len(phrase) + 1)):
            return False
    return True


def city_pool_weather_problems(candidates, rainy_dates) -> list[str]:
    """A necessary uniqueness bound, never positive evidence of shelter.

    Each rainy program day needs at least one indoor visit. A unique POI cannot
    cover two dates. Excluding outdoor/mixed types and explicit open-air names
    can prove an inadequate pool; remaining candidates are still unverified.
    """
    required = len(set(rainy_dates))
    potential = sum(city_candidate_may_have_shelter(candidate.record) for candidate in candidates)
    if required and potential < required:
        return [f"Для {required} дождливых дней в проверенном наборе только {potential} мест, "
                "не исключённых по уличному/смешанному типу или явному названию карты. Их крытая программа также не подтверждена. "
                "Расписание не запрашивалось: повторять одно место или выдумывать помещения нельзя. "
                "Ограниченный поиск не означает, что других мест в городе нет."]
    return []


def _coordinates(item) -> tuple[float, float] | None:
    values = (item.lat, item.lon)
    if any(type(value) not in (int, float) or not -bound <= value <= bound
           or not math.isfinite(value) for value, bound in zip(values, (90, 180))):
        return None
    return values


def _bound_record(item, records: Mapping[str, dict]) -> dict | None:
    """Use only the caller's exact observation bound to this verified item.

    This binding is deliberately not a substitute for full map verification.
    No catalogue reload or geocoder call is permitted in this quality layer.
    """
    if item.verified is not True or item.kind != "attraction":
        return None
    record = records.get(item.place_query)
    coordinates = _coordinates(item)
    if not isinstance(record, dict) or coordinates is None:
        return None
    record_coordinates = (record.get("lat"), record.get("lon"))
    if any(type(value) not in (int, float) for value in record_coordinates):
        return None
    provider = record.get("provider") or "osm"
    if (provider not in {"osm", "2gis"} or item.provider != provider
            or coordinates != record_coordinates
            or (provider == "2gis" and item.source_url != record.get("source_url"))):
        return None
    return record


def _duration(item) -> int:
    value = item.duration_min
    return value if type(value) is int and 0 < value <= 24 * 60 else 0


def _interval(item) -> tuple[int, int] | None:
    start, duration = _time_minutes(item.time), _duration(item)
    if start is None or duration == 0 or start + duration > 24 * 60:
        return None  # Ordinary schedule validation rejects malformed intervals.
    return start, start + duration


def _family_minimum(request: TripRequest) -> int | None:
    if (not request.children_ages or request.pace.strip().casefold() != "средний"
            or not _schedule_is_explicit(request)):
        return None
    start, end = _time_minutes(request.start_time), _time_minutes(request.end_time)
    if start is None or end is None or end - start < 8 * 60:
        return None
    return 90 if min(request.children_ages) < 7 else 120


def city_break_minutes(request: TripRequest) -> int:
    """A planning pause, not a medical rule or an imposed afternoon nap."""
    return 60 if request.children_ages and min(request.children_ages) < 7 else 20


def desired_city_program_minutes(request: TripRequest) -> int | None:
    """Soft visit-volume goal, deliberately separate from acceptance minima.

    Adolescents use the adult preference. Do not inflate an individual visit,
    reject an otherwise useful partial day or manufacture POIs to reach it.
    Meals, breaks and transfer reserves do not count as visits.
    """
    if request.destination_kind != "city" or not _schedule_is_explicit(request):
        return None
    start, end = _time_minutes(request.start_time), _time_minutes(request.end_time)
    if start is None or end is None or end <= start:
        return None
    pace = request.pace.strip().casefold()
    percent = {"спокойный": 35, "средний": 50, "насыщенный": 65}.get(pace)
    if percent is None:
        return None
    youngest = min(request.children_ages) if request.children_ages else 18
    percent = max(20, percent - (20 if youngest < 7 else 10 if youngest < 13 else 0))
    rounded = ((end - start) * percent + 1499) // 1500 * 15
    cap = {"спокойный": 210, "средний": 300, "насыщенный": 360}[pace]
    return max(0, min(rounded, cap, end - start - 60 - city_break_minutes(request)))


def city_day_coverage(request: TripRequest, itinerary: Itinerary) -> dict:
    """Observable soft quality; never an exception or a reason to erase a day."""
    desired = desired_city_program_minutes(request)
    if desired is None:
        return {}
    start, end = _time_minutes(request.start_time), _time_minutes(request.end_time)
    # Do not demand a morning/afternoon split for a short or evening-only trip.
    split_day = end - start >= 6 * 60 and start < 12 * 60 and end >= 16 * 60
    tail = 120 if city_break_minutes(request) == 60 else 90
    light = unbalanced = 0
    for day in itinerary.days:
        if day.day_mode == "free":
            continue  # Already reported as an incomplete calendar.
        visits = [item for item in day.items if item.kind == "attraction"]
        light += sum(_duration(item) for item in visits) < desired
        intervals = [value for item in visits if (value := _interval(item)) is not None]
        meals = [value for item in day.items if item.kind == "meal"
                 and (value := _interval(item)) is not None]
        if split_day:
            meal = meals[0] if len(meals) == 1 else (13 * 60, 14 * 60)
            balanced = (any(left < meal[0] for left, _ in intervals)
                        and any(left >= meal[1] for left, _ in intervals)
                        and max((right for _, right in intervals), default=start) >= end - tail)
            unbalanced += not balanced
    reasons = (["day_program_light"] if light else []) + (["day_window_unbalanced"] if unbalanced else [])
    return {"desired_program_minutes": desired, "light_program_days": light,
            "unbalanced_program_days": unbalanced, "quality_reasons": reasons}


def apply_city_day_coverage(request: TripRequest, itinerary: Itinerary, coverage: dict) -> None:
    """Recompute after generation/edit; stale light-day reasons must disappear."""
    observed = city_day_coverage(request, itinerary)
    if not observed:
        return  # Regional/legacy requests have no city-window preference.
    previous_reasons = list(coverage.get("quality_reasons", []))
    prior = [code for code in previous_reasons
             if code not in {"day_program_light", "day_window_unbalanced"}]
    current = observed.pop("quality_reasons", [])
    coverage.update(observed)
    coverage["quality_reasons"] = list(dict.fromkeys([*prior, *current]))
    if current and coverage.get("quality_status") != "insufficient_data":
        coverage["quality_status"] = "partial"
    elif (coverage.get("quality_status") == "partial" and previous_reasons and not prior
          and not current and all(day.day_mode != "free" for day in itinerary.days)):
        coverage["quality_status"] = "complete"


def _busy_minutes(items, start: int, end: int) -> int:
    """Union of meal/rest intervals clipped to the gap, never double-counted."""
    intervals = []
    for item in items:
        if item.kind not in {"meal", "rest"}:
            continue
        interval = _interval(item)
        if interval is not None:
            left, right = max(start, interval[0]), min(end, interval[1])
            if left < right:
                intervals.append((left, right))
    total, current_end = 0, start
    for left, right in sorted(intervals):
        if right > current_end:
            total += right - max(left, current_end)
            current_end = right
    return total


def check_city_quality(itinerary: Itinerary, request: TripRequest,
                       records: Mapping[str, dict], *, allow_city_limited_program: bool = False) -> list[str]:
    """Check explicit opt-in constraints, not factual completeness or safety.

    Only server-owned exact-query records belong in ``records``. Empty output
    means these bounded constraints passed, NOT that metadata was verified.
    Legacy city generation does not call this function; regions are a no-op.
    """
    if request.destination_kind != "city":
        return []
    problems = []
    free_days = authorized_free_days(itinerary, request,
                                    allow_city_limited_program=allow_city_limited_program)
    family_minimum = _family_minimum(request)
    for day in itinerary.days:
        if day.day in free_days:
            continue
        visits = [item for item in day.items if item.kind == "attraction"]
        for item in visits:
            record = _bound_record(item, records)
            if not city_only_record_allowed(request, record):
                problems.append(f"День {day.day}: принадлежность посещения выбранному городу не подтверждена "
                                "адресом карты при явном ограничении «только в городе». "
                                "Радиус, название объекта и соседний район не подтверждают это условие.")
            map_type = _map_type(record) if record is not None else ""
            if map_type == "peak":
                problems.append(
                    f"День {day.day}: «{item.title}» имеет тип карты peak; координата вершины "
                    "не подтверждает доступное посещение, путь подъёма, подъёмник или его стоимость. "
                    "В городском map-only режиме такой кандидат не допускается: нужен отдельно "
                    "проверенный посетительский объект и сведения о доступе, не модельная оценка или VisitFacts из карты."
                )
            if item.indoor is True and record is not None and not city_candidate_may_have_shelter(record):
                problems.append(
                    f"День {day.day}: «{item.title}» исключён по уличному/смешанному типу или явному названию карты; такой кандидат "
                    "не обосновывает indoor=true. Для крытого посещения нужен отдельный проверенный "
                    "на карте крытый объект; не выдумывай помещение и не меняй свойства этого места."
                )
            if map_type in _SHORT_STOP_TYPES and _duration(item) > 60:
                problems.append(
                    f"День {day.day}: остановка «{item.title}» ({map_type}) растянута до "
                    f"{item.duration_min} мин. В map-only режиме предел — 60 мин.; это ограничение "
                    "продукта против необоснованного растягивания, не факт о часах или длительности посещения."
                )
        if family_minimum is not None:
            minutes = sum(_duration(item) for item in visits)
            if minutes < family_minimum:
                problems.append(
                    f"День {day.day}: семейная программа содержит {minutes} мин. посещений, "
                    f"нужно не менее {family_minimum} мин. для среднего темпа и явно заданного окна "
                    "от 8 часов. Обед, отдых и дорога не засчитываются. Это ориентир продукта, "
                    "а длительности — оценки, не медицинская рекомендация или проверенные факты."
                )
        known = []
        for item in visits:
            interval, coordinates = _interval(item), _coordinates(item)
            if item.verified is True and interval is not None and coordinates is not None:
                known.append((interval, coordinates, item))
        known.sort(key=lambda entry: entry[0][0])
        for first, second in zip(known, known[1:]):
            km = distance_km(*first[1], *second[1])
            if km <= 0.03:
                continue
            minimum = math.ceil(km)
            start, end = first[0][1], second[0][0]
            available = max(0, end - start - _busy_minutes(day.items, start, end))
            if available < minimum:
                problems.append(
                    f"День {day.day}: между «{first[2].title}» и «{second[2].title}» свободно "
                    f"{available} мин. после исключения обеда/отдыха; по прямой {km:.2f} км, "
                    f"нужно хотя бы {minimum} мин. Это очень оптимистичная геометрическая нижняя "
                    "оценка при 60 км/ч, не пеший маршрут, дорога или подтверждение транспорта."
                )
    return problems
