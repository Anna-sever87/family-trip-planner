"""Правила, которые программа проверяет после ответа модели."""

import math
import re
import unicodedata
from typing import Iterable, List, Mapping, Optional, Tuple

from places import (
    _valid_record_only_place, distance_km, map_record_identity,
    resolved_map_identity, same_map_place,
)
from pricing import minimum_meal_cost
from program_coverage import authorized_free_days, limited_program_problems
from schemas import INTEREST_LABELS, Itinerary, TripRequest


MAX_CITY_RADIUS_KM = 40.0

PAID_INTEREST_CATEGORIES = {
    "museums",
    "theatre",
    "cinema",
    "water_trips",
    "zoos",
    "amusement_parks",
}

_GENERIC_ATTRACTION_NAMES = {
    "центральная улица",
    "главная улица",
    "центральная площадь",
    "главная площадь",
    "центр города",
    "исторический центр",
    "местный парк",
    "интерактивный музей",
    "детский парк",
    "торговый центр",
    "универмаг",
    "набережная",
    "музей",
    "зоопарк",
    "кинотеатр",
    "театр",
    "парк",
    "cinema",
    "museum",
    "park",
    "theatre",
    "zoo",
    "shopping mall",
    "shopping center",
    "shopping centre",
    "department store",
    "waterfront",
    "embankment",
    "promenade",
    # Observed category placeholders are not named places. Keep exact labels:
    # named museums, trails and waterfronts must still reach map verification.
    "природные достопримечательности",
    "сувенирные лавки",
    "природные тропы",
    "достопримечательности",
    "музеи",
    "выставки",
    "парки",
    "природа",
    "зоопарки",
    "аквариумы",
    "торговые центры",
    "шопинг",
    "морские прогулки",
    "речные прогулки",
}

_CHILD_WORD_RE = re.compile(
    r"\b(?:дети|детей|детям|детьми|детях|детск\w*|"
    r"реб[её]н\w*|коляск\w*|малыш\w*|семейн\w*)\b",
    re.IGNORECASE,
)

_NEUTRAL_REST_TITLE = "Отдых"
_NEUTRAL_REST_DESCRIPTION = (
    "Спокойный перерыв для отдыха и восстановления."
)
_NON_VISIT_SLOT_NAMES = frozenset({"обед", "отдых"})


def _explicit_outdoor_description(description: str, title: str) -> bool:
    """Recognize a small grammar of complete, unqualified outdoor assertions.

    This is a consistency check, not a classifier of places or an evidence
    source. Match the entire description so negations, conditions, quotations
    and mixed/covered-part visits stay outside this deliberately narrow rule.
    Only the item's exact title may follow a museum template; no free-text
    suffix can accidentally hide an exception or a second activity.
    """
    def normalize(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())

    text = normalize(description)
    if text.endswith((".", "!")):
        text = text[:-1].rstrip()
    standalone = {
        "посещение проходит под открытым небом",
        "осмотр проходит под открытым небом",
        "экскурсия проходит под открытым небом",
        "прогулка проходит под открытым небом",
        "прогулка под открытым небом",
        "экскурсия под открытым небом",
    }
    museum_templates = (
        "посещение музея под открытым небом",
        "осмотр музея под открытым небом",
        "экскурсия по музею под открытым небом",
    )
    standalone.update(museum_templates)
    name = normalize(title)
    # An exact copied title is still untrusted text. A qualifying clause or
    # explicit covered part inside it must not enter this atomic-name grammar.
    qualified_name = re.search(
        r"[,;:()!?]|\b(?:не|без|вместо|если|при|только|лишь|но|и|"
        r"крыт\w*|внутренн\w*|помещени\w*|павильон\w*|зал\w*)\b",
        name,
    )
    if name and not qualified_name:
        standalone.update(f"{prefix} {name}" for prefix in museum_templates)
    return text in standalone


def attraction_limit(
    youngest_age: int,
) -> int:
    """Допустимое число достопримечательностей."""

    if youngest_age < 3:
        return 2

    if youngest_age < 7:
        return 3

    if youngest_age < 13:
        return 4

    return 5


def _normalize_place_name(
    text: str,
) -> str:
    """Упрощает название для эвристик и общих формулировок."""

    main_part = str(text or "").split(
        "(",
        1,
    )[0]

    decomposed = unicodedata.normalize(
        "NFKD",
        main_part.casefold(),
    )
    # В Unicode русская «й» раскладывается как «и» + breve.
    # Простое удаление всех combining-знаков превращало, например,
    # «торговый центр» в «торговыи центр», тогда как справочный
    # список общих названий содержал исходную «й». Сначала собираем
    # эту букву обратно, а остальные диакритики по-прежнему убираем.
    decomposed = decomposed.replace(
        "и\N{COMBINING BREVE}",
        "й",
    )
    without_accents = "".join(
        char
        for char in decomposed
        if not unicodedata.combining(char)
    )

    words = re.findall(
        r"[0-9a-zа-яё]+",
        without_accents,
    )

    return " ".join(words)


_GENERIC_ATTRACTION_KEYS = frozenset(
    _normalize_place_name(name)
    # A copied questionnaire interest is a category, never a concrete POI.
    for name in _GENERIC_ATTRACTION_NAMES | set(INTEREST_LABELS.values())
)


def is_generic_attraction(
    title: str,
    place_query: str,
) -> bool:
    """Проверяет, что название описывает тип места, а не конкретный POI."""

    title_key = _normalize_place_name(title)
    query_title_key = (
        _normalize_place_name(
            str(place_query or "").split(",", 1)[0]
        )
        if place_query
        else ""
    )

    return bool(
        # A specific map query cannot turn a meal/rest slot into a visit.
        # Match the complete title: do not reject named places such as
        # «Дом отдыха» or strip a meaningful parenthetical qualification.
        _exact_place_identity(title) in _NON_VISIT_SLOT_NAMES
        or title_key in _GENERIC_ATTRACTION_KEYS
        or query_title_key in _GENERIC_ATTRACTION_KEYS
        or re.fullmatch(
            r"прогулка по (?:центру|городу|району|улицам)",
            title_key,
        )
    )


def _exact_place_identity(
    text: str,
) -> str:
    """Игнорирует пунктуацию, сохраняя весь Unicode и уточнения."""

    normalized = unicodedata.normalize(
        "NFKC",
        str(text or ""),
    ).casefold()
    identity = "".join(
        (
            char
            if unicodedata.category(char)[0]
            in {"L", "M", "N", "S"}
            else " "
        )
        for char in normalized
    )

    return " ".join(identity.split())


def _reference_name(text: str) -> str:
    """Убирает служебные слова из названия запасного места."""

    normalized = _normalize_place_name(text)
    prefixes = (
        "посещение ",
        "экскурсия в ",
        "экскурсия по ",
        "поход в ",
        "визит в ",
    )

    for prefix in prefixes:
        if normalized.startswith(prefix):
            return normalized[len(prefix):].strip()

    return normalized


def sanitize_rainy_alternatives(
    itinerary: Itinerary,
    target_day: Optional[int] = None,
) -> List[int]:
    """Очищает необязательные и конфликтующие запасные места.

    При полном маршруте сохраняется первое уникальное место. При
    изменении одного дня остальные дни остаются неизменными.
    """

    attraction_names = {
        _reference_name(value)
        for day in itinerary.days
        for item in day.items
        if item.kind == "attraction"
        for value in (
            item.title,
            item.place_query.split(",", 1)[0],
        )
        if _reference_name(value)
    }

    seen_alternatives = {
        _reference_name(day.rainy_alternative)
        for position, day in enumerate(
            itinerary.days,
            start=1,
        )
        if (
            target_day is not None
            and position != target_day
            and _reference_name(
                day.rainy_alternative
            )
        )
    }
    changed_days: List[int] = []

    for position, day in enumerate(
        itinerary.days,
        start=1,
    ):
        if (
            target_day is not None
            and position != target_day
        ):
            continue

        cleaned = day.rainy_alternative.strip()
        alternative = _reference_name(cleaned)
        invalid = (
            not alternative
            or alternative in attraction_names
            or alternative in seen_alternatives
        )
        replacement = "" if invalid else cleaned

        if day.rainy_alternative != replacement:
            day.rainy_alternative = replacement
            changed_days.append(day.day)

        if not invalid:
            seen_alternatives.add(alternative)

    return changed_days


def sanitize_rest_items(
    itinerary: Itinerary,
    target_day: Optional[int] = None,
) -> List[int]:
    """Приводит отдых к нейтральной паузе без ссылки на место.

    Это консервативное правило не зависит от языка и склонений: любой
    ``rest`` получает единые заголовок и описание, пустой map-запрос и
    нулевую стоимость. При изменении маршрута затрагивается только
    целевой день; время, длительность, район и заметки сохраняются.
    """

    changed_days: List[int] = []

    for day_plan in itinerary.days:
        if (
            target_day is not None
            and day_plan.day != target_day
        ):
            continue

        day_changed = False

        for item in day_plan.items:
            if item.kind != "rest":
                continue

            before = item.model_dump()
            item.title = _NEUTRAL_REST_TITLE
            item.description = (
                _NEUTRAL_REST_DESCRIPTION
            )
            item.place_query = ""
            item.cost = 0
            item.verified = False
            item.lat = None
            item.lon = None
            item.official_name = ""

            if item.model_dump() != before:
                day_changed = True

        if day_changed:
            changed_days.append(day_plan.day)

    return changed_days


def _time_minutes(value: str) -> Optional[int]:
    """Преобразует ЧЧ:ММ в минуты или возвращает None."""

    match = re.fullmatch(
        r"(\d{2}):(\d{2})",
        str(value or "").strip(),
    )

    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))

    if hour > 23 or minute > 59:
        return None

    return hour * 60 + minute


def _schedule_is_explicit(
    request: TripRequest,
) -> bool:
    """Не ужесточает старые анкеты без полей режима дня."""

    return {
        "start_time",
        "end_time",
    }.issubset(request.model_fields_set)


def program_depth_target(
    request: TripRequest,
) -> Optional[int]:
    """Ориентир продукта для длинного взрослого дня, не требование заполнить его целиком."""

    if (
        request.children_ages
        or request.pace.strip().casefold() != "средний"
        or not _schedule_is_explicit(request)
    ):
        return None

    start = _time_minutes(request.start_time)
    end = _time_minutes(request.end_time)
    if start is None or end is None or end - start < 8 * 60:
        return None

    # 30% окна, округлённые вверх до 15 минут, без ошибок float.
    rounded_minutes = ((end - start + 49) // 50) * 15
    return min(240, max(180, rounded_minutes))


def _program_minutes(value: object) -> float:
    """Учитывает только конечные неотрицательные числовые минуты."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return 0.0
    return numeric if math.isfinite(numeric) and numeric >= 0 else 0.0


def check_program_depth(
    itinerary: Itinerary,
    request: TripRequest,
    travel_minutes_by_day: Optional[Mapping[int, float]] = None,
    *,
    only_day: Optional[int] = None,
    allow_limited_program: bool = False,
    allow_city_limited_program: bool = False,
) -> List[str]:
    """Проверяет измеримую содержательность, отдельно от check_rules.

    Длительности посещений остаются оценками модели. Минуты дороги должен
    передать вызывающий код из геопроверки как доказанную нижнюю оценку;
    transfer_note, название базы и пустые интервалы доказательством не служат.
    Эта эвристика продукта не подтверждает фактическое время посещений/пути.
    """

    context = dict(allow_limited_program=allow_limited_program,
                   allow_city_limited_program=allow_city_limited_program)
    problems = limited_program_problems(itinerary, request, **context)
    free_days = authorized_free_days(itinerary, request, **context)
    target = program_depth_target(request)
    if target is None:
        return problems

    travel_minutes_by_day = travel_minutes_by_day or {}
    for day in itinerary.days:
        if only_day is not None and day.day != only_day:
            continue
        if day.day in free_days:
            continue

        visits = sum(
            _program_minutes(item.duration_min)
            for item in day.items
            if item.kind == "attraction"
        )
        travel = _program_minutes(travel_minutes_by_day.get(day.day, 0))
        if visits + travel < target:
            problems.append(
                f"День {day.day}: недостаточно содержательная программа: "
                f"посещения {visits:g} мин., доказанная нижняя оценка дороги "
                f"{travel:g} мин., ориентир {target} мин. Добавь реальное "
                "уникальное посещение или действительно длительный маршрут; "
                "не растягивай короткие остановки искусственно и не выдумывай места."
            )

    return problems


def _adult_wording_problems(
    itinerary: Itinerary,
) -> List[str]:
    """Не допускает детские формулировки в маршруте для взрослых."""

    fields: List[Tuple[str, str]] = [
        ("название поездки", itinerary.title),
        ("описание поездки", itinerary.summary),
    ]

    for day in itinerary.days:
        fields.extend(
            [
                (f"тема дня {day.day}", day.theme),
                (
                    f"запасной план дня {day.day}",
                    day.rainy_alternative,
                ),
            ]
        )

        for item in day.items:
            fields.extend(
                [
                    (
                        f"название пункта дня {day.day}",
                        item.title,
                    ),
                    (
                        f"описание пункта «{item.title}» "
                        f"дня {day.day}",
                        item.description,
                    ),
                    (
                        f"kids_note пункта «{item.title}» "
                        f"дня {day.day}",
                        item.kids_note,
                    ),
                ]
            )

    fields.extend(
        (f"совет {index}", tip)
        for index, tip in enumerate(
            itinerary.tips,
            start=1,
        )
    )

    problems: List[str] = []

    for label, value in fields:
        has_forbidden_wording = bool(
            value
            and _CHILD_WORD_RE.search(value)
        )
        has_adult_kids_note = bool(
            value
            and label.startswith("kids_note ")
        )

        if (
            has_forbidden_wording
            or has_adult_kids_note
        ):
            problems.append(
                "Поездка только для взрослых, но "
                f"{label} содержит детскую формулировку: "
                f"«{value}»"
            )

    return problems


def _schedule_problems(
    itinerary: Itinerary,
    request: TripRequest,
) -> List[str]:
    """Проверяет границы и пересечения программы."""

    if not _schedule_is_explicit(request):
        return []

    start_bound = _time_minutes(
        request.start_time
    )
    end_bound = _time_minutes(
        request.end_time
    )

    if (
        start_bound is None
        or end_bound is None
        or start_bound >= end_bound
    ):
        return [
            "В анкете указан некорректный "
            "интервал времени дня"
        ]

    problems: List[str] = []

    for day in itinerary.days:
        timed_items: List[
            Tuple[int, int, object]
        ] = []

        for item in day.items:
            start = _time_minutes(item.time)

            if start is None:
                problems.append(
                    f"День {day.day}: у пункта "
                    f"«{item.title}» неверное время"
                )
                continue

            finish = start + item.duration_min
            timed_items.append(
                (start, finish, item)
            )

            if start < start_bound:
                problems.append(
                    f"День {day.day}: «{item.title}» "
                    f"начинается раньше {request.start_time}"
                )

            if finish > end_bound:
                problems.append(
                    f"День {day.day}: «{item.title}» "
                    f"заканчивается позже {request.end_time}"
                )

        timed_items.sort(
            key=lambda entry: entry[0]
        )

        for previous, current in zip(
            timed_items,
            timed_items[1:],
        ):
            if current[0] < previous[1]:
                problems.append(
                    f"День {day.day}: пункты "
                    f"«{previous[2].title}» и "
                    f"«{current[2].title}» "
                    "пересекаются по времени"
                )

    return problems


def check_rules(
    itinerary: Itinerary,
    request: TripRequest,
    rainy_dates: Optional[
        Iterable[str]
    ] = None,
    *,
    allow_limited_program: bool = False,
    allow_city_limited_program: bool = False,
) -> List[str]:
    """Проверяет маршрут, бюджет и нагрузку."""

    rainy = set(
        rainy_dates or []
    )

    context = dict(allow_limited_program=allow_limited_program,
                   allow_city_limited_program=allow_city_limited_program)
    problems = limited_program_problems(itinerary, request, **context)
    free_days = authorized_free_days(itinerary, request, **context)

    if (
        len(itinerary.days)
        != request.days_count
    ):
        problems.append(
            "В маршруте "
            f"{len(itinerary.days)} дн., "
            "а нужно ровно "
            f"{request.days_count}"
        )

    # Assignments, model_copy(update=...) and legacy objects can bypass model
    # validation. Reject their prices before either summing or comparing them.
    invalid_prices = [
        f"День {day.day}: стоимость пункта «{item.title}» должна быть конечным "
        "неотрицательным числом не больше 1 000 000 000."
        for day in itinerary.days
        for item in day.items
        if not (type(item.cost) in (int, float)
                and 0 <= item.cost <= 1_000_000_000 and math.isfinite(item.cost))
    ]
    if invalid_prices:
        return problems + invalid_prices

    total = sum(
        item.cost
        for day in itinerary.days
        for item in day.items
    )

    if round(total, 2) > request.program_budget:
        problems.append(
            "Расходы на программу "
            f"{total:g} {request.currency} "
            "превышают доступный бюджет программы после дороги и жилья "
            f"{request.program_budget:g} "
            f"{request.currency}"
        )

    try:
        meal_floor = minimum_meal_cost(
            request
        )
    except ValueError:
        meal_floor = 0.0

    if not request.children_ages:
        problems.extend(
            _adult_wording_problems(
                itinerary
            )
        )

    problems.extend(
        _schedule_problems(
            itinerary,
            request,
        )
    )

    youngest = (
        min(request.children_ages)
        if request.children_ages
        else None
    )

    seen_places: dict[
        str,
        Tuple[int, str],
    ] = {}

    for day in itinerary.days:
        kinds = [
            item.kind
            for item in day.items
        ]

        attraction_count = (
            kinds.count("attraction")
        )

        for item in day.items:
            if item.kind != "attraction":
                continue

            if item.indoor is True and _explicit_outdoor_description(item.description, item.title):
                problems.append(
                    f"День {day.day}: у «{item.title}» описание прямо указывает посещение "
                    "под открытым небом, но indoor = true. Уточните фактическую программу: "
                    "признак помещения противоречит этому описанию."
                )

            if is_generic_attraction(
                item.title,
                item.place_query,
            ):
                problems.append(
                    f"День {day.day}: «{item.title}» — "
                    "слишком общее название. Нужно "
                    "конкретное реально существующее место"
                )

            place_key = _exact_place_identity(
                item.place_query
            )

            if not place_key:
                continue

            previous = seen_places.get(
                place_key
            )

            if previous is not None:
                problems.append(
                    f"Дни {previous[0]} и {day.day}: "
                    "повторяется место "
                    f"«{item.title}» "
                    f"(ранее «{previous[1]}»)"
                )
            else:
                seen_places[
                    place_key
                ] = (
                    day.day,
                    item.title,
                )

        if attraction_count == 0 and day.day not in free_days:
            problems.append(
                f"День {day.day}: "
                "нет ни одной "
                "достопримечательности"
            )

        if (
            day.date in rainy
            and attraction_count
        ):
            outdoor = [
                item.title
                for item in day.items
                if (
                    item.kind
                    == "attraction"
                    and not item.indoor
                )
            ]

            indoor_count = (
                attraction_count
                - len(outdoor)
            )

            if (
                indoor_count * 2
                < attraction_count
            ):
                problems.append(
                    f"День {day.day} "
                    f"({day.date}) дождливый, "
                    "но под открытым небом: "
                    f"{', '.join(outdoor)}. "
                    "Не менее половины "
                    "достопримечательностей "
                    "должны быть в помещении "
                    "с indoor = true. "
                    "Если достопримечательность "
                    "одна, она обязана быть "
                    "в помещении."
                )

        if kinds.count("meal") != 1:
            problems.append(
                f"День {day.day}: "
                "должен быть ровно один обед"
            )

        for meal in (
            item
            for item in day.items
            if item.kind == "meal"
        ):
            if meal.cost < meal_floor:
                problems.append(
                    f"День {day.day}: стоимость обеда "
                    f"{meal.cost:g} {request.currency} "
                    "нереалистично мала для состава "
                    "группы; нужно не меньше "
                    f"{meal_floor:g} {request.currency}"
                )

        if kinds.count("rest") != 1:
            problems.append(
                f"День {day.day}: "
                "должен быть ровно один отдых"
            )

        if youngest is None:
            continue

        limit = attraction_limit(
            youngest
        )

        if attraction_count > limit:
            problems.append(
                f"День {day.day}: "
                f"{attraction_count} "
                "достопримечательностей, "
                f"а для ребёнка {youngest} "
                f"лет не больше {limit}"
            )

        has_sufficient_rest = any(
            (
                item.kind == "rest"
                and item.duration_min >= 60
            )
            for item in day.items
        )

        if (
            youngest < 7
            and not has_sufficient_rest
        ):
            problems.append(
                f"День {day.day}: "
                "нет отдыха не короче "
                "60 минут для ребёнка "
                f"{youngest} лет"
            )

    return problems


def _record_only_duplicate_identity(item, records: Mapping[str, dict]) -> Optional[tuple]:
    """Bind a server observation to this verified item, never to model IDs.

    A supplied record is not permission to move/relabel an item. Invalid or
    mismatching evidence leaves the ordinary conservative duplicate check in
    force. This lookup neither populates the map cache nor changes the item.
    """
    if item.kind != "attraction" or item.verified is not True:
        return None
    query = item.place_query
    record = records.get(query)
    if not _valid_record_only_place(query, record):
        return None
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or not -bound <= value <= bound
           for value, bound in ((item.lat, 90), (item.lon, 180))):
        return None
    provider = "2gis" if record.get("provider") == "2gis" else "osm"
    if (item.lat != record["lat"] or item.lon != record["lon"]
            or item.provider != provider):
        return None
    if provider == "2gis" and item.source_url != record["source_url"]:
        return None
    return map_record_identity(record)


def check_verified_place_duplicates(
    itinerary: Itinerary, *, verified_records: Mapping[str, dict] | None = None,
) -> List[str]:
    """Находит повторно использованную проверенную геоточку.

    Optional records are exact-query server observations already used by map
    verification, not itinerary/model fields. The default cache-based path is
    unchanged; missing or invalid observations fall back to that same path.
    """

    verified: List[
        Tuple[int, object]
    ] = [
        (day.day, item)
        for day in itinerary.days
        for item in day.items
        if (
            item.kind == "attraction"
            and item.verified
            and item.lat is not None
            and item.lon is not None
        )
    ]
    problems: List[str] = []

    identities = {}
    for _, item in verified:
        identity = (_record_only_duplicate_identity(item, verified_records)
                    if verified_records is not None else None)
        identities[id(item)] = identity if identity is not None else resolved_map_identity(item)
    for index, (
        first_day,
        first,
    ) in enumerate(verified):
        for second_day, second in (
            verified[index + 1:]
        ):
            if same_map_place(first, second, first_identity=identities[id(first)],
                              second_identity=identities[id(second)]):
                problems.append(
                    f"Дни {first_day} и {second_day}: "
                    "карта показала, что "
                    f"«{first.title}» и «{second.title}» "
                    "являются одним и тем же местом"
                )

    return problems


# Короткое имя сохранено для возможных внешних проверок.
check_verified_duplicates = (
    check_verified_place_duplicates
)


def check_city_radius(
    itinerary: Itinerary,
    center: Optional[
        Tuple[float, float]
    ],
) -> List[str]:
    """Проверяет расположение мест относительно города."""

    if center is None:
        return []

    problems: List[str] = []

    for day in itinerary.days:
        for item in day.items:
            if (
                item.verified
                and item.lat is not None
                and item.lon is not None
            ):
                km = distance_km(
                    center[0],
                    center[1],
                    item.lat,
                    item.lon,
                )

                if km > MAX_CITY_RADIUS_KM:
                    problems.append(
                        f"День {day.day}: "
                        f"«{item.title}» найдено "
                        f"в {km:.0f} км от центра "
                        "города, нужны места "
                        "в самом городе или рядом. "
                        f"place_query «{item.place_query}»"
                    )

    return problems
