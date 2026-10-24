"""Напоминания о поездке: расписание по датам поездки и тексты сообщений.

Расписание:
- заранее, по умолчанию за 3 дня (в 10:00): состояние чек-листа;
- накануне (в 18:00): погода, чек-лист и напоминание про PDF;
- каждое утро поездки (в 08:00): программа дня и запасной план на дождь.
Число дней задаётся REMINDER_DAYS_BEFORE, часовой пояс — REMINDER_TZ.
"""

import logging
import os
import re
from collections.abc import Mapping
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from place_labels import build_place_labels, display_place_title


HOUR_BEFORE = 10

# Старое имя оставлено для совместимости
# с ранней версией модуля.
HOUR_3_DAYS = HOUR_BEFORE

HOUR_EVE = 18
HOUR_MORNING = 8
WINDOW_HOURS = 12
DEFAULT_DAYS_BEFORE = 3
MIN_DAYS_BEFORE = 1
MAX_DAYS_BEFORE = 30

logger = logging.getLogger(
    "reminders"
)


def get_days_before() -> int:
    """За сколько дней присылать первое напоминание."""

    raw_value = os.getenv(
        "REMINDER_DAYS_BEFORE",
        str(DEFAULT_DAYS_BEFORE),
    ).strip()

    try:
        value = int(
            raw_value
        )
    except ValueError as error:
        raise RuntimeError(
            "REMINDER_DAYS_BEFORE должен быть "
            "целым числом от 1 до 30."
        ) from error

    if not (
        MIN_DAYS_BEFORE
        <= value
        <= MAX_DAYS_BEFORE
    ):
        raise RuntimeError(
            "REMINDER_DAYS_BEFORE должен быть "
            "целым числом от 1 до 30."
        )

    return value


def get_tz() -> ZoneInfo:
    """Возвращает проверенный часовой пояс напоминаний."""

    name = os.getenv(
        "REMINDER_TZ",
        "Europe/Moscow",
    ).strip()

    if not name:
        raise RuntimeError(
            "REMINDER_TZ не должен быть пустым. "
            "Пример: Asia/Novosibirsk."
        )

    try:
        return ZoneInfo(
            name
        )
    except (
        ZoneInfoNotFoundError,
        ValueError,
    ) as error:
        raise RuntimeError(
            "В REMINDER_TZ указан неизвестный "
            f"часовой пояс «{name}». "
            "Пример: Asia/Novosibirsk."
        ) from error


def normalize_datetime(
    value: datetime,
    tz: Optional[ZoneInfo] = None,
) -> datetime:
    """Приводит время к REMINDER_TZ."""

    if not isinstance(
        value,
        datetime,
    ):
        raise TypeError(
            "Ожидались дата и время datetime."
        )

    target_tz = (
        tz
        or get_tz()
    )

    if value.tzinfo is None:
        return value.replace(
            tzinfo=target_tz
        )

    return value.astimezone(
        target_tz
    )


def _value(
    source: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Читает поле dict или sqlite3.Row."""

    if isinstance(
        source,
        Mapping,
    ):
        return source.get(
            key,
            default,
        )

    try:
        return source[key]
    except (
        KeyError,
        TypeError,
        IndexError,
    ):
        return default


def _at(
    day: date,
    hour: int,
    tz: Optional[ZoneInfo] = None,
) -> datetime:
    return datetime.combine(
        day,
        time(
            hour,
            0,
        ),
        tzinfo=(
            tz
            or get_tz()
        ),
    )


def reminder_plan(
    trip: Dict[str, Any],
) -> List[Tuple[str, datetime]]:
    """Возвращает виды и время напоминаний."""

    try:
        start = date.fromisoformat(
            str(
                _value(
                    trip,
                    "start_date",
                    "",
                )
            )
        )

        end = date.fromisoformat(
            str(
                _value(
                    trip,
                    "end_date",
                    "",
                )
            )
        )
    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            "У поездки неверные "
            "start_date или end_date."
        ) from error

    if end < start:
        raise ValueError(
            "Дата окончания поездки "
            "раньше даты начала."
        )

    days_before = (
        get_days_before()
    )
    tz = get_tz()

    try:
        plan = [
            (
                "before",
                _at(
                    start - timedelta(
                        days=days_before
                    ),
                    HOUR_BEFORE,
                    tz,
                ),
            ),
            (
                "eve",
                _at(
                    start - timedelta(
                        days=1
                    ),
                    HOUR_EVE,
                    tz,
                ),
            ),
        ]
    except OverflowError as error:
        raise ValueError(
            "Дата начала поездки "
            "слишком ранняя."
        ) from error

    for number in range(
        (end - start).days + 1
    ):
        plan.append(
            (
                f"day{number + 1}",
                _at(
                    start + timedelta(
                        days=number
                    ),
                    HOUR_MORNING,
                    tz,
                ),
            )
        )

    return plan


def _created_at(
    trip: Dict[str, Any],
    tz: ZoneInfo,
) -> datetime:
    raw_value = _value(
        trip,
        "created_at",
        "",
    )

    if not raw_value:
        return datetime.min.replace(
            tzinfo=tz
        )

    try:
        created = datetime.fromisoformat(
            str(
                raw_value
            )
        )
    except ValueError as error:
        raise ValueError(
            "У поездки неверное "
            "поле created_at."
        ) from error

    return normalize_datetime(
        created,
        tz,
    )


def _canonical_kind(
    kind: Any,
) -> str:
    """Поддерживает старые виды d3 и d1."""

    value = str(
        kind
    ).strip()

    return {
        "d3": "before",
        "d1": "eve",
    }.get(
        value,
        value,
    )


def _sent_aliases(
    kind: str,
) -> Tuple[str, ...]:
    return {
        "before": (
            "before",
            "d3",
        ),
        "eve": (
            "eve",
            "d1",
        ),
    }.get(
        kind,
        (
            kind,
        ),
    )


def due_reminders(
    trips: List[Dict[str, Any]],
    is_sent: Callable[
        [
            int,
            str,
        ],
        bool,
    ],
    now: datetime,
) -> List[Tuple[Dict[str, Any], str]]:
    """Возвращает напоминания из текущего окна."""

    tz = get_tz()

    # Ошибка общей настройки должна
    # быть видна сразу при запуске.
    get_days_before()

    current = normalize_datetime(
        now,
        tz,
    )

    due = []

    for trip in trips:
        try:
            trip_id = int(
                _value(
                    trip,
                    "id",
                )
            )

            created = _created_at(
                trip,
                tz,
            )

            plan = reminder_plan(
                trip
            )
        except (
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            logger.warning(
                "Поездка пропущена при "
                "проверке напоминаний: %s",
                error,
            )
            continue

        for kind, when in plan:
            # Поездку создали уже после
            # времени этого напоминания.
            if when < created:
                continue

            if not (
                when
                <= current
                < when + timedelta(
                    hours=WINDOW_HOURS
                )
            ):
                continue

            if any(
                is_sent(
                    trip_id,
                    alias,
                )
                for alias in _sent_aliases(
                    kind
                )
            ):
                continue

            due.append(
                (
                    trip,
                    kind,
                )
            )

    return due


def _status_map(
    statuses: Any,
) -> Dict[str, str]:
    """Преобразует set или строки БД в словарь."""

    if statuses is None:
        return {}

    if isinstance(
        statuses,
        (
            set,
            frozenset,
        ),
    ):
        return {
            _canonical_kind(
                kind
            ): "sent"
            for kind in statuses
        }

    if isinstance(
        statuses,
        str,
    ):
        return {
            _canonical_kind(
                statuses
            ): "sent"
        }

    if isinstance(
        statuses,
        Mapping,
    ):
        if "kind" not in statuses:
            return {
                _canonical_kind(
                    kind
                ): str(
                    status
                )
                for kind, status
                in statuses.items()
            }

        rows: Iterable[Any] = [
            statuses
        ]

    elif (
        hasattr(
            statuses,
            "keys",
        )
        and "kind" in statuses.keys()
    ):
        rows = [
            statuses
        ]

    else:
        try:
            rows = list(
                statuses
            )
        except TypeError:
            return {}

    result = {}

    for row in rows:
        if isinstance(
            row,
            str,
        ):
            result[
                _canonical_kind(
                    row
                )
            ] = "sent"
            continue

        kind = _canonical_kind(
            _value(
                row,
                "kind",
                "",
            )
        )

        if kind:
            result[kind] = str(
                _value(
                    row,
                    "status",
                    "sent",
                )
            ).strip()

    return result


def _days_word(
    value: int,
) -> str:
    last_two = (
        value % 100
    )
    last = (
        value % 10
    )

    if 11 <= last_two <= 14:
        word = "дней"
    elif last == 1:
        word = "день"
    elif 2 <= last <= 4:
        word = "дня"
    else:
        word = "дней"

    return (
        f"{value} {word}"
    )


def _kind_name(
    kind: str,
    days_before: int,
) -> str:
    if kind == "before":
        return (
            f"за {_days_word(days_before)} "
            "до поездки"
        )

    if kind == "eve":
        return (
            "накануне вечером"
        )

    if kind.startswith(
        "day"
    ):
        return (
            "утро дня "
            f"{kind[3:]}"
        )

    return kind


def describe_plan(
    trip: Dict[str, Any],
    sent_kinds: Any,
    now: Optional[datetime] = None,
) -> str:
    """Строит текст для команды /reminders."""

    tz = get_tz()

    current = normalize_datetime(
        (
            now
            or datetime.now(
                tz
            )
        ),
        tz,
    )

    created = _created_at(
        trip,
        tz,
    )

    statuses = _status_map(
        sent_kinds
    )

    days_before = (
        get_days_before()
    )

    trip_id = _value(
        trip,
        "id",
        "?",
    )

    destination = str(
        _value(
            trip,
            "destination",
            "",
        )
        or "направление не указано"
    )

    lines = [
        "Напоминания по поездке "
        f"№{trip_id} ({destination}):"
    ]

    for kind, when in reminder_plan(
        trip
    ):
        delivery_status = statuses.get(
            kind
        )

        if delivery_status == "sent":
            status = "отправлено"

        elif delivery_status == "sending":
            status = "отправляется"

        elif delivery_status == "failed":
            status = "ошибка отправки"

        elif when < created:
            status = (
                "пропущено "
                "(поездка создана позже)"
            )

        elif current >= (
            when
            + timedelta(
                hours=WINDOW_HOURS
            )
        ):
            status = "срок прошёл"

        else:
            status = "ожидает"

        lines.append(
            f"{when:%d.%m %H:%M} - "
            f"{_kind_name(kind, days_before)}: "
            f"{status}"
        )

    return "\n".join(
        lines
    )


def _safe_dict(
    value: Any,
) -> Dict[str, Any]:
    if isinstance(
        value,
        Mapping,
    ):
        return dict(
            value
        )

    return {}


def _safe_list(
    value: Any,
) -> List[Any]:
    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return list(
            value
        )

    return []


def _unchecked(
    checklist: Optional[
        List[
            Dict[
                str,
                Any,
            ]
        ]
    ],
) -> List[str]:
    result = []

    for item in (
        checklist
        or []
    ):
        text = str(
            _value(
                item,
                "text",
                "",
            )
            or ""
        ).strip()

        if (
            text
            and not bool(
                _value(
                    item,
                    "done",
                    False,
                )
            )
        ):
            result.append(
                text
            )

    return result


def _number_text(
    value: Any,
) -> Optional[str]:
    try:
        return (
            f"{float(value):g}"
        )
    except (
        TypeError,
        ValueError,
    ):
        return None


def _command(
    name: str,
    trip_id: Any,
) -> str:
    try:
        return (
            f"/{name} "
            f"{int(trip_id)}"
        )
    except (
        TypeError,
        ValueError,
    ):
        return ""


def build_reminder_text(
    kind: str,
    trip: Dict[str, Any],
    checklist: List[Dict[str, Any]],
) -> str:
    """Строит текст даже при неполном старом JSON."""

    kind = _canonical_kind(
        kind
    )

    trip_id = _value(
        trip,
        "id",
    )

    destination = str(
        _value(
            trip,
            "destination",
            "",
        )
        or "ваше путешествие"
    )

    start_date = str(
        _value(
            trip,
            "start_date",
            "",
        )
        or "дата уточняется"
    )

    request = _safe_dict(
        _value(
            trip,
            "request",
            {},
        )
    )

    itinerary = _safe_dict(
        _value(
            trip,
            "itinerary",
            {},
        )
    )

    checklist_items = (
        _safe_list(
            checklist
        )
    )

    left = _unchecked(
        checklist_items
    )

    if kind == "before":
        text = (
            "🧳 Через "
            f"{_days_word(get_days_before())} "
            f"поездка: {destination} "
            f"({start_date})."
        )

        if (
            checklist_items
            and left
        ):
            text += (
                "\nВ чек-листе собрано "
                f"{len(checklist_items) - len(left)} "
                f"из {len(checklist_items)}. "
                "Ещё не отмечено, например: "
                + "; ".join(
                    left[:3]
                )
                + "."
            )

        elif checklist_items:
            text += (
                "\nВсе вещи из чек-листа "
                "отмечены, молодцы!"
            )

        else:
            text += (
                "\nЧек-лист пока пуст."
            )

        command = _command(
            "checklist",
            trip_id,
        )

        if command:
            text += (
                "\nЧек-лист: "
                f"{command}"
            )

        return text

    if kind == "eve":
        text = (
            "✈️ Завтра в путь: "
            f"{destination}!"
        )

        weather = _safe_dict(
            request.get(
                "weather"
            )
        )

        days = _safe_list(
            weather.get(
                "days"
            )
        )

        first = (
            _safe_dict(
                days[0]
            )
            if days
            else {}
        )

        tmin = _number_text(
            first.get(
                "tmin"
            )
        )

        tmax = _number_text(
            first.get(
                "tmax"
            )
        )

        rain = _number_text(
            first.get(
                "rain_mm",
                first.get(
                    "precipitation_mm"
                ),
            )
        )

        if (
            tmin is not None
            and tmax is not None
        ):
            text += (
                "\nПогода в первый день: "
                f"от {tmin} до {tmax} °C"
            )

            if rain is not None:
                text += (
                    f", осадки {rain} мм"
                )

            text += "."

        if left:
            text += (
                "\nНе отмечено в чек-листе "
                f"({len(left)}): "
                + "; ".join(
                    left[:5]
                )
                + (
                    "..."
                    if len(left) > 5
                    else ""
                )
            )

        command = _command(
            "pdf",
            trip_id,
        )

        if command:
            text += (
                "\nСкачайте PDF с маршрутом, "
                "он пригодится без интернета: "
                f"{command}"
            )

        return text

    match = re.fullmatch(
        r"day(\d+)",
        kind,
    )

    if match is None:
        return (
            "🔔 Напоминание о поездке: "
            f"{destination}."
        )

    number = int(
        match.group(
            1
        )
    )

    days = _safe_list(
        itinerary.get(
            "days"
        )
    )

    if not (
        1
        <= number
        <= len(days)
    ):
        return (
            "Доброе утро! "
            "Сегодня поездка "
            f"{destination}."
        )

    day = _safe_dict(
        days[
            number - 1
        ]
    )

    day_date = str(
        day.get(
            "date"
        )
        or "дата не указана"
    )

    theme = str(
        day.get(
            "theme"
        )
        or "план на сегодня"
    )

    lines = [
        "☀️ Доброе утро! "
        f"Программа на день {number} "
        f"({day_date}): {theme}"
    ]

    labels = build_place_labels({"request": request, "itinerary": itinerary})
    for item in _safe_list(
        day.get(
            "items"
        )
    ):
        item_data = _safe_dict(
            item
        )

        item_time = str(
            item_data.get(
                "time"
            )
            or ""
        ).strip()

        title = display_place_title(item_data, labels).strip()

        if title:
            lines.append(
                " ".join(
                    part
                    for part in (
                        item_time,
                        title,
                    )
                    if part
                )
            )

    rainy_alternative = str(
        day.get(
            "rainy_alternative"
        )
        or ""
    ).strip()

    if rainy_alternative:
        lines.append(
            "☔ Если дождь: "
            f"{rainy_alternative}"
        )

    command = _command(
        "trip",
        trip_id,
    )

    if command:
        lines.append(
            "Подробнее: "
            f"{command}"
        )

    return "\n".join(
        lines
    )
