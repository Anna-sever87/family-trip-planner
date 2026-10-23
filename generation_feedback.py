"""Public generation feedback: fixed copy, never raw model-repair instructions."""

from collections.abc import Iterable
import re


_SOURCE_REASONS = {
    "SOURCE_SEARCH_LIMIT": (
        "Локальный поиск достиг безопасного предела и не завершил составление плана. "
        "Это не доказывает нехватку мест или невозможность поездки."
    ),
    "SOURCE_ROUTES_UNCONFIRMED": (
        "Не удалось подтвердить необходимые дорожные связи для выбранной программы."
    ),
    "SOURCE_VISIT_DATA_INCOMPLETE": (
        "В текущих данных недостаточно подходящих карточек посещений для этих дат и условий. "
        "Это не означает, что в регионе нет подходящих мест."
    ),
    "SOURCE_CANDIDATE_CAPACITY": (
        "Текущего проверенного набора посещений недостаточно для полной программы. "
        "Это не доказывает невозможность поездки по региону."
    ),
    "SOURCE_CONSTRAINTS_INCOMPATIBLE": (
        "Из проверенных вариантов пока не удалось собрать полную программу "
        "с заданными датами, временем и бюджетом."
    ),
}
_SOURCE_PREFIX = re.compile(r"^\s*SOURCE_PLAN_UNAVAILABLE\s*(?::|$)", re.IGNORECASE)
_RAIN_PROGRAM_REASON = "Не удалось подобрать подходящие посещения в помещении для дождливых дней."
_RAIN_DAY_PROBLEM = re.compile(
    r"день (?:[1-9]|1[0-4]) \([0-9]{4}-[0-9]{2}-[0-9]{2}\) дождливый, но под открытым небом: .+\. "
    r"не менее половины достопримечательностей должны быть в помещении с indoor = true\. "
    r"если достопримечательность одна, она обязана быть в помещении\."
)
_RAIN_POOL_PROBLEM = re.compile(
    r"для (?P<days>[1-9]|1[0-4]) дождливых дней в проверенном наборе только (?P<places>[0-9]|1[0-8]) мест, "
    r"не исключенных по уличному/смешанному типу(?: или явному названию)? карты\. "
    r"их крытая программа также не подтверждена\. "
    r"расписание не запрашивалось: повторять одно место или выдумывать помещения нельзя\. "
    r"ограниченный поиск не означает, что других мест в городе нет\."
)


def _is_rain_program_failure(problem: str) -> bool:
    """Recognize whole validator templates, not arbitrary rain/indoor words."""
    normalized = " ".join(problem.casefold().replace("ё", "е").split())
    if _RAIN_DAY_PROBLEM.fullmatch(normalized):
        return True
    pool = _RAIN_POOL_PROBLEM.fullmatch(normalized)
    return pool is not None and int(pool["places"]) < int(pool["days"])


def _bounded_count(text: str, key: str, bound: int) -> int | None:
    matches = re.findall(rf"(?<![\w]){key}=([^\s;]+)", text, re.IGNORECASE)
    if len(matches) != 1 or not re.fullmatch(r"0|[1-9][0-9]{0,4}", matches[0]):
        return None
    value = int(matches[0])
    return value if value <= bound else None


def _coverage(text: str) -> str:
    """Only bounded decimal counts, never arbitrary diagnostic fragments."""
    days, cards = (_bounded_count(text, key, 100) for key in ("requestedDays", "usableCards"))
    if days is None or cards is None:
        return ""
    result = f" Дней в запросе — {days}; посещений с подходящими данными — {cards}."
    unique = _bounded_count(text, "uniqueCandidates", 100)
    minutes = _bounded_count(text, "availableMinutes", 60000)
    target = _bounded_count(text, "targetMinutes", 1440)
    if unique is not None and minutes is not None and target is not None:
        result += (f" Уникальных мест — {unique}; длительность самих посещений в этом наборе — {minutes} мин; "
                   f"ориентир на день с учётом дороги — {target} мин.")
    excluded = _bounded_count(text, "excludedVisits", 100)
    if excluded:
        result += f" Отклонено по известным условиям — {excluded}."
    return result


def _capacity_explanation(text: str) -> str:
    reasons = re.findall(r"(?<![\w])coverageReason=([^\s;]+)", text)
    if len(reasons) != 1:
        return ""
    days = _bounded_count(text, "requestedDays", 14)
    unique = _bounded_count(text, "uniqueCandidates", 100)
    if days is None or days < 1 or unique is None:
        return ""
    if reasons[0] == "first_day_depth_capacity":
        maximum = _bounded_count(text, "firstDayMaxMinutes", 1440)
        target = _bounded_count(text, "targetMinutes", 1440)
        if unique == days and maximum is not None and target is not None and maximum < target:
            return (" Без повторов на каждый день остаётся по одному посещению; "
                    f"в первый день даже с дорогой получается не больше {maximum} мин при ориентире {target} мин.")
    elif reasons[0] == "dates_without_visits":
        uncovered = _bounded_count(text, "uncoveredDays", 14)
        if uncovered is not None and 0 < uncovered <= days:
            return f" Для {uncovered} дней в этом наборе нет посещений с подходящим временным окном."
    elif reasons[0] == "unique_candidates" and unique < days:
        return " Уникальных посещений в этом наборе меньше, чем дней поездки."
    return ""


def _source_failure_reason(problems: list[str]) -> str | None:
    diagnostics = [problem for problem in problems if _SOURCE_PREFIX.match(problem)]
    if not diagnostics:
        return None
    # A search cutoff is not proof of bad data or incompatible constraints.
    # Deterministic precedence also prevents generic road/budget text elsewhere
    # from overriding an explicit machine category.
    for code, reason in _SOURCE_REASONS.items():
        pattern = re.compile(rf"^\s*SOURCE_PLAN_UNAVAILABLE\s*:\s*{code}(?=$|[\s;])", re.IGNORECASE)
        matching = [problem for problem in diagnostics if pattern.match(problem)]
        if matching:
            # Conflicting duplicate headers should not choose an arbitrary count.
            coverage = {(_capacity_explanation(problem) if code == "SOURCE_CANDIDATE_CAPACITY" else "")
                        + _coverage(problem) for problem in matching}
            return reason + (next(iter(coverage)) if len(coverage) == 1 else "")
    return "План по проверенным сведениям пока не составлен. Причину отказа не удалось уточнить."


def generation_failure_reason(problems: Iterable[str]) -> str:
    """Summarize known validation categories without echoing diagnostic content."""
    safe_problems = [str(problem) for problem in problems]
    source_reason = _source_failure_reason(safe_problems)
    if source_reason is not None:
        return source_reason
    text = "\n".join(problem.casefold().replace("ё", "е") for problem in safe_problems)
    # A failed lookup does not establish an outage. Reserve this wording for
    # explicit availability failures, not an unconfirmed or missing place.
    if any(marker in text for marker in (
        "карта временно недоступна", "карта недоступна", "сервис карты недоступен",
        "сервис карты временно недоступен", "геокодер недоступен",
    )):
        return "Сервис карты временно недоступен, поэтому проверить маршрут не удалось."
    if any(_is_rain_program_failure(problem) for problem in safe_problems):
        return _RAIN_PROGRAM_REASON
    if any(marker in text for marker in (
        "place_query", "base_area", "подтверд", "подтвержд", "проверку карты",
        "не найдено на карте", "не найден на карте", "не найдены на карте",
        "повтор места", "повторяется место", "слишком общее название",
        "конкретных мест", "реально существующее место", "достопримечательност",
        "за пределами выбранной территории", "слишком далеко от базы",
    )):
        return "Не удалось подобрать достаточно подходящих мест, подтверждённых картой."
    if any(marker in text for marker in (
        "бюджет", "стоимость обеда", "расходы на программу", "времени",
        "время ", "временные", "интервал", "duration_min", "пересека",
        "недостаточно содержательная программа",
    )):
        return "Не удалось согласовать программу с доступным временем и бюджетом."
    return "Не удалось согласовать маршрут с условиями поездки."


def generation_failure_message(problems: Iterable[str], *, editing: bool = False) -> str:
    reason = generation_failure_reason(problems)
    if editing:
        return f"Изменения не сохранены. {reason} Старый маршрут сохранён без изменений."
    return (
        f"Подходящий маршрут пока не составлен. {reason} "
        "Новая поездка не сохранена, существующие поездки не изменялись."
    )
