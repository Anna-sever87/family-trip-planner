"""Разбор ответов пользователя в анкете: даты, состав семьи, бюджет."""
import re
import math
from datetime import date
from typing import List, Tuple

MAX_TRIP_DAYS = 14
PACES = ["спокойный", "средний", "насыщенный"]
DESTINATION_KIND_LABELS = {
    "city": "Город / место",
    "region": "Регион / озеро",
}

_DESTINATION_KIND_ALIASES = {
    "city": {
        "city",
        "город",
        "город / место",
        "город или место",
        "место",
    },
    "region": {
        "region",
        "регион",
        "регион / озеро",
        "регион или озеро",
        "озеро",
    },
}

CURRENCY_WORDS = {
    "EUR": ["eur", "euro", "евро", "€"],
    "USD": ["usd", "dollar", "доллар", "$"],
    "RUB": ["rub", "руб", "₽", "р."],
}

DATE_RE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})|(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?")


def parse_destination_kind(text: str) -> str:
    """Возвращает city или region из ответа анкеты."""

    normalized = " ".join(
        (text or "").strip().casefold().split()
    )

    for kind, aliases in (
        _DESTINATION_KIND_ALIASES.items()
    ):
        if normalized in aliases:
            return kind

    raise ValueError(
        "Выберите кнопкой: "
        "«Город / место» или "
        "«Регион / озеро»."
    )


def _to_date(match: re.Match, today: date) -> date:
    if match.group(1):
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    day, month = int(match.group(4)), int(match.group(5))
    year_text = match.group(6)
    if year_text:
        year = int(year_text)
        if year < 100:
            year += 2000
        return date(year, month, day)
    # год не указан: берём ближайшую будущую дату
    candidate = date(today.year, month, day)
    return candidate if candidate >= today else date(today.year + 1, month, day)


def parse_dates(text: str, today: date = None) -> Tuple[str, str]:
    """Возвращает (начало, конец) в формате ГГГГ-ММ-ДД. При ошибке бросает ValueError с понятным текстом."""
    today = today or date.today()
    found = list(DATE_RE.finditer(text))
    if len(found) < 2:
        raise ValueError("Нужны две даты: начало и конец. Например: 10.11.2026 - 15.11.2026")
    try:
        start, end = _to_date(found[0], today), _to_date(found[1], today)
    except ValueError:
        raise ValueError("Не получилось разобрать дату. Пример: 10.11.2026 - 15.11.2026")
    if start < today:
        raise ValueError("Дата начала уже прошла. Напишите даты поездки ещё раз.")
    if end < start:
        raise ValueError("Дата окончания раньше даты начала. Напишите даты ещё раз.")
    if (end - start).days + 1 > MAX_TRIP_DAYS:
        raise ValueError(f"Пока составляю маршруты не длиннее {MAX_TRIP_DAYS} дней. Сократите поездку.")
    return start.isoformat(), end.isoformat()


def parse_family(text: str) -> Tuple[int, List[int]]:
    """Возвращает (число взрослых, возрасты детей). Пример ответа: «2 взрослых, дети 4 и 9»."""
    lowered = text.lower()
    adults_match = re.search(r"(\d+)\s*(?:взр|человек|чел)", lowered)
    kids_split = re.split(r"дет|ребен|ребён", lowered, maxsplit=1)
    if adults_match:
        adults = int(adults_match.group(1))
        kids_part = kids_split[1] if len(kids_split) > 1 else lowered[adults_match.end():]
    else:
        numbers = re.findall(r"\d+", lowered)
        if not numbers:
            raise ValueError("Напишите состав семьи. Пример: 2 взрослых, дети 4 и 9")
        adults = int(numbers[0])
        kids_part = kids_split[1] if len(kids_split) > 1 else " ".join(numbers[1:])
    if "без дет" in lowered:
        kids_part = ""
    ages = [int(n) for n in re.findall(r"\d+", kids_part)]
    if not 1 <= adults <= 10:
        raise ValueError("Число взрослых должно быть от 1 до 10.")
    if len(ages) > 6:
        raise ValueError("Укажите не больше 6 детей.")
    if any(age > 17 for age in ages):
        raise ValueError("Возраст ребёнка должен быть от 0 до 17 лет. Напишите состав семьи ещё раз.")
    return adults, ages


def parse_budget(text: str, default_currency: str = "EUR") -> Tuple[float, str]:
    """Возвращает (сумма, валюта). Пример ответа: «600 EUR» или «80000 руб»."""
    lowered = text.lower().replace(" ", "").replace("\u00a0", "")
    number = re.fullmatch(r"(\d+(?:[.,]\d{1,2})?)([a-zа-яё€$₽.]*)", lowered)
    if not number:
        raise ValueError("Напишите бюджет числом и валютой. Пример: 600 EUR")
    amount = float(number.group(1).replace(",", "."))
    if not math.isfinite(amount) or not 0 < amount <= 1_000_000_000:
        raise ValueError("Бюджет должен быть больше нуля и не превышать 1 000 000 000.")
    currency = default_currency
    suffix = number.group(2)
    for code, words in CURRENCY_WORDS.items():
        if suffix in words:
            currency = code
            break
    else:
        if suffix:
            raise ValueError("Поддерживаются валюты RUB, EUR и USD.")
    if currency not in CURRENCY_WORDS:
        raise ValueError("Поддерживаются валюты RUB, EUR и USD.")
    return amount, currency


def parse_fixed_cost(text: str, currency: str) -> float:
    """Разбирает общую стоимость дороги/жилья в валюте бюджета."""
    compact = text.lower().replace(" ", "").replace("\u00a0", "")
    if compact in {"-", "—", "нет", "неизвестно"}:
        return 0.0
    match = re.fullmatch(r"(\d+(?:[.,]\d{1,2})?)([a-zа-яё€$₽.]*)", compact)
    if not match:
        raise ValueError("Введите одну неотрицательную сумму, например 15000, или «-», если она неизвестна.")
    amount = float(match.group(1).replace(",", "."))
    suffix = match.group(2)
    if suffix and suffix not in CURRENCY_WORDS.get(currency, []):
        raise ValueError(f"Укажите сумму в валюте бюджета {currency}; пересчитайте её заранее.")
    if not math.isfinite(amount) or amount > 1_000_000_000:
        raise ValueError("Сумма не должна превышать 1 000 000 000.")
    return round(amount, 2)
