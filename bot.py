"""Telegram-бот «Семейный планировщик путешествий».

Запуск: python bot.py
"""

from __future__ import annotations

import io
import logging
import os
import re
import threading
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Optional

import telebot
from apscheduler.schedulers.background import (
    BackgroundScheduler,
)
from dotenv import load_dotenv
from telebot import types

import anketa
import access_control
import db
import memory as vector_memory
import privacy
import reminders
import web_auth
from checklist import build_checklist
from city_context import (
    CITY_EDIT_NOTE, CONTEXT_ERROR, CONTEXT_KEY, check_context_places,
    load_city_context, prepare_stored_city_context, validate_city_edit_context,
)
from city_rollout import CITY_MVP_DISCLOSURE, city_candidates_enabled
from formatting import format_itinerary, split_text
from generation_feedback import generation_failure_message
from pdf_export import build_trip_pdf
from pricing import minimum_meal_cost
from planner import (
    edit_itinerary,
    generate_itinerary,
)
from schemas import Itinerary, TripRequest
from weather import CityResolutionError, get_weather, resolve_city_for_planning


load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s "
        "%(name)s - %(message)s"
    ),
)

for noisy_logger in (
    "httpx",
    "httpcore",
    "urllib3",
):
    logging.getLogger(
        noisy_logger
    ).setLevel(
        logging.WARNING
    )

logger = logging.getLogger("bot")

TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN"
)

if not TOKEN:
    raise SystemExit(
        "В файле .env не задан "
        "TELEGRAM_BOT_TOKEN"
    )

DAILY_LIMIT = int(
    os.getenv(
        "DAILY_LIMIT",
        "3",
    )
)

MAX_MANUAL_MEMORY_CHARS = 600
MAX_FEEDBACK_CHARS = 600
MAX_EDIT_REQUEST_CHARS = 600
REMINDER_CHECK_MINUTES = 5

bot = telebot.TeleBot(
    TOKEN
)

db.init_db()

# Состояния анкеты.
states: dict[int, dict] = {}

# Пользователи, которым сейчас
# составляется маршрут.
generating: set[int] = set()
generation_jobs_guard = threading.Lock()

# Ручные операции с памятью.
memory_jobs: set[int] = set()
memory_jobs_guard = threading.Lock()


HELP_TEXT = (
    "Я составляю маршруты путешествий "
    "для семей с детьми.\n\n"
    "/plan - составить новый маршрут\n"
    "/trips - мои сохранённые поездки\n"
    "/trip 3 - показать поездку №3\n"
    "/edit 3 1 текст - изменить день маршрута\n"
    "/checklist 3 - чек-лист поездки №3\n"
    "/pdf 3 - скачать поездку №3 в PDF\n"
    "/reminders 3 - расписание напоминаний\n"
    "/web - открыть веб-кабинет\n"
    "/delete 3 - удалить поездку №3\n\n"
    "/remember текст - запомнить предпочтение\n"
    "/feedback 3 текст - сохранить отзыв "
    "о поездке №3\n"
    "/memory - показать память\n"
    "/forget_memory - очистить память\n\n"
    "/delete_data - удалить все данные аккаунта\n\n"
    "/cancel - отменить текущую анкету\n"
    "/help - эта подсказка"
)


QUESTIONS = {
    "destination_kind": (
        "Что планируем: поездку в один "
        "город/место или по региону/озеру?"
    ),
    "destination": (
        "Куда поедете? Напишите город, "
        "регион или озеро и страну. "
        "Например: Париж, Франция или "
        "Байкал, Россия"
    ),
    "dates": (
        "Даты поездки: начало и конец. "
        "Например: 10.11.2026 - 15.11.2026"
    ),
    "family": (
        "Состав семьи. Например: "
        "2 взрослых, дети 4 и 9"
    ),
    "budget": (
        "Общий бюджет всей поездки "
        "(дорога, жильё и программа) на всех участников. "
        "Например: 600 EUR или 80000 руб"
    ),
    "travel_cost": "Стоимость дороги туда и обратно на всех участников в валюте бюджета. Например: 15000. Если пока неизвестна, напишите «-».",
    "lodging_cost": "Стоимость жилья за все ночи в валюте бюджета. Например: 20000. Если пока неизвестна, напишите «-».",
    "pace": "Какой темп поездки?",
    "wishes": (
        "Пожелания: что любите, "
        "чего избегаете? Все обязательные ограничения укажите сейчас: память прошлых поездок помогает с приоритетами, но не заменяет текущую анкету. "
        "Если нет, напишите «-»"
    ),
}


MEMORY_KIND_LABELS = {
    "family": "Состав семьи",
    "pace": "Темп поездок",
    "currency": "Валюта",
    "destination_wishes": (
        "Пожелания к направлению"
    ),
    "custom": "Личная заметка",
    "feedback": "Отзыв о поездке",
}


# ---------- вспомогательные функции ----------
def send_parts(
    chat_id: int,
    text: str,
) -> None:
    for part in split_text(
        text
    ):
        bot.send_message(
            chat_id,
            part,
        )


def _message_user_id(
    message,
) -> int:
    if message.from_user:
        return int(
            message.from_user.id
        )

    return int(
        message.chat.id
    )


def _clean_memory_text(
    value: str,
    max_chars: int,
) -> str:
    return " ".join(
        (value or "").split()
    )[:max_chars]


def _clean_edit_request(
    value: str,
) -> str:
    text = " ".join(
        (value or "").split()
    )

    if len(text) > MAX_EDIT_REQUEST_CHARS:
        raise ValueError(
            "Пожелание к изменению должно быть "
            f"не длиннее {MAX_EDIT_REQUEST_CHARS} "
            "символов."
        )

    if len(text) < 3:
        raise ValueError(
            "Пожелание к изменению слишком короткое."
        )

    return text


def _is_generation_active(
    chat_id: int,
) -> bool:
    with generation_jobs_guard:
        return chat_id in generating


def _reserve_generation_job(
    chat_id: int,
) -> bool:
    with generation_jobs_guard:
        if chat_id in generating:
            return False

        generating.add(
            chat_id
        )

    return True


def _finish_generation_job(
    chat_id: int,
) -> None:
    with generation_jobs_guard:
        generating.discard(
            chat_id
        )


def _claim_route_ai_request(
    chat_id: int,
    kind: str,
    trip_id: Optional[int] = None,
):
    return db.claim_ai_request(
        chat_id,
        datetime.now(
            reminders.get_tz()
        ).date(),
        kind,
        "telegram",
        trip_id=trip_id,
        daily_limit=DAILY_LIMIT,
    )


def _ai_claim_error_text(
    claim,
) -> str:
    if claim.status == "busy":
        return (
            "Другая операция с маршрутом уже "
            "выполняется. Подождите немного."
        )
    return (
        "На сегодня лимит "
        f"{claim.limit} запросов на создание "
        "и изменение маршрутов исчерпан. "
        "Попробуйте завтра."
    )


def _finish_ai_claim(
    claim_id: Optional[str],
    succeeded: bool,
) -> None:
    if not claim_id:
        return
    try:
        db.finish_ai_request(
            claim_id,
            succeeded=succeeded,
        )
    except Exception:
        logger.exception(
            "Не удалось завершить журнал "
            "AI-запроса %s",
            claim_id,
        )


def _reserve_memory_job(
    chat_id: int,
) -> bool:
    with memory_jobs_guard:
        if chat_id in memory_jobs:
            return False

        memory_jobs.add(
            chat_id
        )

    return True


def _finish_memory_job(
    chat_id: int,
) -> None:
    with memory_jobs_guard:
        memory_jobs.discard(
            chat_id
        )


def pace_keyboard() -> (
    types.ReplyKeyboardMarkup
):
    keyboard = types.ReplyKeyboardMarkup(
        resize_keyboard=True,
        one_time_keyboard=True,
    )

    keyboard.add(
        *[
            types.KeyboardButton(pace)
            for pace in anketa.PACES
        ]
    )

    return keyboard


def destination_kind_keyboard() -> (
    types.ReplyKeyboardMarkup
):
    keyboard = types.ReplyKeyboardMarkup(
        resize_keyboard=True,
        one_time_keyboard=True,
    )

    keyboard.add(
        *[
            types.KeyboardButton(label)
            for label in (
                anketa.DESTINATION_KIND_LABELS.values()
            )
        ]
    )

    return keyboard


def ask(
    chat_id: int,
    step: str,
) -> None:
    if step == "destination_kind":
        bot.send_message(
            chat_id,
            QUESTIONS[step],
            reply_markup=(
                destination_kind_keyboard()
            ),
        )
        return

    if step == "pace":
        bot.send_message(
            chat_id,
            QUESTIONS[step],
            reply_markup=(
                pace_keyboard()
            ),
        )
        return

    if step == "family":
        profile = db.get_user(
            chat_id
        )

        hint = ""

        if (
            profile
            and profile["children_ages"]
        ):
            ages = ", ".join(
                str(age)
                for age
                in profile[
                    "children_ages"
                ]
            )

            hint = (
                "\n\nВ прошлый раз: "
                f"{profile['adults']} взрослых, "
                f"дети {ages}. "
                "Напишите «да», "
                "чтобы оставить так же."
            )

        bot.send_message(
            chat_id,
            QUESTIONS[step] + hint,
        )
        return

    question = QUESTIONS[step]
    if (step == "wishes" and city_candidates_enabled()
            and states.get(chat_id, {}).get("data", {}).get("destination_kind", "city") == "city"):
        question += ("\n\n" + CITY_MVP_DISCLOSURE
                     + "\nВаш ответ на этот последний вопрос запустит создание. Для отмены: /cancel.")
    bot.send_message(chat_id, question)


def used_today(
    tg_id: int,
) -> int:
    tz = reminders.get_tz()
    today = datetime.now(
        tz
    ).date()

    used = 0

    for trip in db.list_trips(
        tg_id,
        limit=50,
    ):
        raw_created = str(
            trip.get(
                "created_at",
                "",
            )
        )

        try:
            created = datetime.fromisoformat(
                raw_created
            )
        except ValueError:
            continue

        if (
            reminders.normalize_datetime(
                created,
                tz,
            ).date()
            == today
        ):
            used += 1

    return max(
        used,
        db.get_ai_request_usage(
            tg_id,
            today,
        ),
    )


def _trip_with_fresh_eve_weather(
    trip: dict,
) -> tuple[dict, bool]:
    """Возвращает копию поездки со свежей погодой для eve."""

    prepared_trip = dict(
        trip
    )
    request = dict(
        trip.get("request")
        if isinstance(
            trip.get("request"),
            dict,
        )
        else {}
    )

    try:
        fresh_weather = get_weather(
            str(
                trip.get(
                    "destination",
                    "",
                )
            ),
            str(
                trip.get(
                    "start_date",
                    "",
                )
            ),
            str(
                trip.get(
                    "start_date",
                    "",
                )
            ),
        )
    except Exception:
        logger.warning(
            "Не удалось обновить погоду "
            "для поездки %s",
            trip.get("id"),
            exc_info=True,
        )
        fresh_weather = None

    # Старую историческую погоду накануне
    # не показываем как актуальный прогноз.
    request["weather"] = (
        fresh_weather
        if fresh_weather
        else None
    )
    prepared_trip["request"] = request

    return (
        prepared_trip,
        bool(fresh_weather),
    )


def dispatch_reminders(
    now: Optional[datetime] = None,
) -> None:
    """Проверяет сроки и отправляет готовые напоминания."""

    try:
        tz = reminders.get_tz()
        current = reminders.normalize_datetime(
            now or datetime.now(tz),
            tz,
        )
        days_before = (
            reminders.get_days_before()
        )
        trips = (
            db.list_reminder_candidate_trips(
                current.date(),
                current.date()
                + timedelta(
                    days=days_before
                ),
            )
        )
        due = reminders.due_reminders(
            trips,
            db.is_reminder_sent,
            current,
        )
    except Exception:
        logger.exception(
            "Не удалось проверить сроки "
            "напоминаний"
        )
        return

    for trip, kind in due:
        trip_id = int(
            trip["id"]
        )
        delivery_now = (
            current
            if now is not None
            else datetime.now(tz)
        )

        try:
            claimed = db.claim_reminder(
                trip_id,
                kind,
                delivery_now,
            )
        except Exception:
            logger.exception(
                "Не удалось зарезервировать "
                "напоминание %s поездки %s",
                kind,
                trip_id,
            )
            continue

        if not claimed:
            continue

        sent_to_telegram = False

        try:
            # The due-list snapshot may predate a completed day swap. Read the
            # owner-bound current plan only AFTER claiming: the sending lease
            # now prevents a concurrent swap/delete from changing this plan.
            reminder_trip = db.get_trip(trip_id, int(trip["tg_id"]))
            if reminder_trip is None:
                raise ValueError("Поездка для напоминания больше недоступна.")
            fresh_weather_available = True

            if kind in (
                "eve",
                "d1",
            ):
                (
                    reminder_trip,
                    fresh_weather_available,
                ) = _trip_with_fresh_eve_weather(
                    reminder_trip
                )

            text = (
                reminders.build_reminder_text(
                    kind,
                    reminder_trip,
                    db.get_checklist(
                        trip_id
                    ),
                )
            )

            if (
                kind in (
                    "eve",
                    "d1",
                )
                and not fresh_weather_available
            ):
                text += (
                    "\nСвежий прогноз сейчас "
                    "недоступен."
                )

            bot.send_message(
                int(trip["tg_id"]),
                text,
            )
            sent_to_telegram = True

            if not db.mark_reminder_sent(
                trip_id,
                kind,
                (
                    delivery_now
                    if now is not None
                    else datetime.now(tz)
                ),
            ):
                logger.error(
                    "Telegram принял напоминание "
                    "%s поездки %s, но статус "
                    "не удалось подтвердить",
                    kind,
                    trip_id,
                )
                continue

            logger.info(
                "Напоминание %s поездки %s "
                "отправлено пользователю %s",
                kind,
                trip_id,
                trip["tg_id"],
            )

        except Exception as error:
            logger.exception(
                "Ошибка напоминания %s "
                "поездки %s",
                kind,
                trip_id,
            )

            # Если Telegram уже принял сообщение,
            # повторная отправка опаснее пропуска.
            if sent_to_telegram:
                continue

            try:
                db.mark_reminder_failed(
                    trip_id,
                    kind,
                    error,
                )
            except Exception:
                logger.exception(
                    "Не удалось записать ошибку "
                    "напоминания %s поездки %s",
                    kind,
                    trip_id,
                )


def build_reminder_scheduler(
    tz=None,
):
    """Создаёт, но не запускает фоновый планировщик."""

    reminder_tz = (
        tz
        or reminders.get_tz()
    )
    scheduler = BackgroundScheduler(
        timezone=reminder_tz,
        daemon=True,
    )

    scheduler.add_job(
        dispatch_reminders,
        trigger="interval",
        minutes=REMINDER_CHECK_MINUTES,
        id="trip-reminders",
        name="Напоминания о поездках",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
        misfire_grace_time=300,
        next_run_time=datetime.now(
            reminder_tz
        ),
    )

    return scheduler


def category_keyboard(
    trip_id: int,
    category: str,
) -> types.InlineKeyboardMarkup:
    keyboard = types.InlineKeyboardMarkup(
        row_width=1
    )

    for item in db.get_checklist(
        trip_id
    ):
        if item["category"] != category:
            continue

        mark = (
            "✅"
            if item["done"]
            else "⬜"
        )

        keyboard.add(
            types.InlineKeyboardButton(
                f"{mark} {item['text']}",
                callback_data=(
                    f"ck:{item['id']}"
                ),
            )
        )

    return keyboard


def send_checklist(
    chat_id: int,
    trip_id: int,
) -> None:
    items = db.get_checklist(
        trip_id
    )

    bot.send_message(
        chat_id,
        (
            "🧳 Чек-лист для поездки "
            f"№{trip_id}. Нажимайте "
            "на вещи, чтобы отмечать "
            "собранное."
        ),
    )

    categories = list(
        dict.fromkeys(
            item["category"]
            for item in items
        )
    )

    for category in categories:
        bot.send_message(
            chat_id,
            category,
            reply_markup=(
                category_keyboard(
                    trip_id,
                    category,
                )
            ),
        )


def trip_title(
    trip: dict,
) -> str:
    return (
        f"№{trip['id']}: "
        f"{trip['destination']}, "
        f"{trip['start_date']} - "
        f"{trip['end_date']}"
    )


def _trip_number(
    message,
) -> Optional[int]:
    parts = (
        message.text or ""
    ).split()

    if (
        len(parts) < 2
        or not parts[1].isdigit()
    ):
        return None

    return int(
        parts[1]
    )


def _memory_query(
    request: TripRequest,
) -> str:
    if request.children_ages:
        ages = ", ".join(
            str(age)
            for age
            in request.children_ages
        )

        family = (
            f"{request.adults} взрослых, "
            f"дети {ages} лет"
        )
    else:
        family = (
            f"{request.adults} взрослых, "
            "без детей"
        )

    wishes = (
        request.wishes
        if request.wishes.strip()
        else "нет дополнительных пожеланий"
    )

    return (
        "Какие сохранённые семейные "
        "предпочтения и результаты прошлых "
        "поездок полезны для новой поездки? "
        f"Направление: {request.destination}. "
        f"Состав: {family}. "
        f"Темп: {request.pace}. "
        f"Пожелания текущей анкеты: {wishes}."
    )


def _remember_trip_safely(
    chat_id: int,
    user_id: int,
    request: TripRequest,
    trip_id: int,
) -> int:
    """Сбой памяти не должен сломать маршрут."""

    try:
        memory_result = (
            vector_memory.save_trip_memory(
                chat_id,
                user_id,
                chat_id,
                request,
                trip_id=trip_id,
            )
        )

        if not memory_result.success:
            logger.warning(
                "Память поездки %s не записана "
                "полностью: requested=%s, "
                "vector=%s, journal=%s, error=%s",
                trip_id,
                memory_result.requested,
                memory_result.vector_saved,
                memory_result.journal_saved,
                memory_result.error_type,
            )
            return 0

        logger.info(
            "Память поездки %s: "
            "сохранено записей %s",
            trip_id,
            memory_result.journal_saved,
        )

        return (
            memory_result.journal_saved
        )

    except Exception:
        logger.exception(
            "Не удалось сохранить "
            "память поездки %s",
            trip_id,
        )
        return 0


def _save_custom_memory_job(
    chat_id: int,
    user_id: int,
    text: str,
    *,
    access_grant: Optional[str] = None,
) -> None:
    import access_control
    from access_guard import operation
    from contextlib import ExitStack
    access_scope = ExitStack()
    try:
        access_scope.enter_context(operation(user_id, expected_grant=access_grant))
        access_control.reserve_aux(user_id, kind="memory")
        memory_result = (
            vector_memory.save_custom_memory(
                chat_id,
                user_id,
                chat_id,
                text,
            )
        )

        if not memory_result.success:
            logger.warning(
                "Ручная память не записана: "
                "requested=%s, vector=%s, "
                "journal=%s, error=%s",
                memory_result.requested,
                memory_result.vector_saved,
                memory_result.journal_saved,
                memory_result.error_type,
            )

            bot.send_message(
                chat_id,
                "Память сейчас недоступна. "
                "Запись не была подтверждена. "
                "Попробуйте немного позже.",
            )
            return

        bot.send_message(
            chat_id,
            "Запомнила. "
            "Посмотреть память: /memory",
        )

    except (access_control.AccessDeniedError, access_control.AccessQuotaError) as error:
        bot.send_message(chat_id, str(error))
    except Exception:
        logger.exception(
            "Ошибка ручного "
            "сохранения памяти"
        )

        bot.send_message(
            chat_id,
            "Не получилось сохранить "
            "запись памяти. "
            "Попробуйте немного позже.",
        )

    finally:
        access_scope.close()
        _finish_memory_job(
            chat_id
        )


def _save_feedback_memory_job(
    chat_id: int,
    user_id: int,
    trip_id: int,
    destination: str,
    text: str,
    *,
    access_grant: Optional[str] = None,
) -> None:
    import access_control
    from access_guard import operation
    from contextlib import ExitStack
    access_scope = ExitStack()
    try:
        access_scope.enter_context(operation(user_id, expected_grant=access_grant))
        access_control.reserve_aux(user_id, kind="memory")
        memory_result = (
            vector_memory.save_feedback_memory(
                chat_id,
                user_id,
                chat_id,
                trip_id=trip_id,
                destination=destination,
                text=text,
            )
        )

        if not memory_result.success:
            logger.warning(
                "Отзыв не записан в память: "
                "requested=%s, vector=%s, "
                "journal=%s, error=%s",
                memory_result.requested,
                memory_result.vector_saved,
                memory_result.journal_saved,
                memory_result.error_type,
            )

            bot.send_message(
                chat_id,
                "Память сейчас недоступна. "
                "Отзыв не был подтверждён. "
                "Попробуйте немного позже.",
            )
            return

        bot.send_message(
            chat_id,
            "Отзыв сохранён. "
            "Он будет учитываться "
            "в следующих поездках.",
        )

    except (access_control.AccessDeniedError, access_control.AccessQuotaError) as error:
        bot.send_message(chat_id, str(error))
    except Exception:
        logger.exception(
            "Ошибка сохранения отзыва"
        )

        bot.send_message(
            chat_id,
            "Не получилось сохранить отзыв. "
            "Попробуйте немного позже.",
        )

    finally:
        access_scope.close()
        _finish_memory_job(
            chat_id
        )


def _clear_memory_job(
    chat_id: int,
    user_id: int,
) -> None:
    try:
        cleared = (
            vector_memory.clear_saved_memories(
                chat_id,
                user_id,
                chat_id,
                confirm=True,
            )
        )

        if not cleared:
            bot.send_message(
                chat_id,
                "Не удалось подтвердить "
                "полную очистку памяти. "
                "Попробуйте немного позже.",
            )
            return

        bot.send_message(
            chat_id,
            "Память полностью очищена. "
            "Сохранённые поездки, PDF "
            "и чек-листы не удалялись.",
        )

    except Exception:
        logger.exception(
            "Ошибка полной очистки памяти"
        )

        bot.send_message(
            chat_id,
            "Не получилось полностью "
            "очистить память. "
            "Попробуйте немного позже.",
        )

    finally:
        _finish_memory_job(
            chat_id
        )


def forget_memory_keyboard() -> (
    types.InlineKeyboardMarkup
):
    keyboard = types.InlineKeyboardMarkup(
        row_width=2
    )

    keyboard.add(
        types.InlineKeyboardButton(
            "Да, очистить",
            callback_data=(
                "memory_clear:confirm"
            ),
        ),
        types.InlineKeyboardButton(
            "Отмена",
            callback_data=(
                "memory_clear:cancel"
            ),
        ),
    )

    return keyboard


# ---------- ограничение private-chat MVP ----------
# Этот обработчик зарегистрирован раньше
# обработчиков команд. Поэтому память
# не становится общей для участников группы.
@bot.message_handler(
    func=lambda message:
    getattr(
        message.chat,
        "type",
        "",
    ) != "private",
    content_types=["text"],
)
def handle_non_private(
    message,
):
    bot.send_message(
        message.chat.id,
        "Сейчас бот работает только "
        "в личном чате. "
        "Откройте диалог с ботом напрямую.",
    )


ACCESS_SAFE_COMMANDS = frozenset({"start", "help", "cancel", "delete_data", "forget_memory", "memory", "delete"})
ACCESS_DENIED_TEXT = "Доступ к планировщику предоставляется по приглашению. Попросите владельца прислать действующую ссылку. Команды /help, /cancel и удаления своих данных остаются доступны."


def _private_actor(message):
    actor = getattr(message, "from_user", None)
    chat = getattr(message, "chat", None)
    return bool(actor and chat and getattr(chat, "type", "") == "private"
                and getattr(actor, "id", None) == getattr(chat, "id", None))


def _message_access_blocked(message):
    if not _private_actor(message):
        return True
    text = (getattr(message, "text", "") or "").split(maxsplit=1)
    command = text[0].split("@", 1)[0].lstrip("/").lower() if text and text[0].startswith("/") else ""
    if command in ACCESS_SAFE_COMMANDS:
        return False
    try:
        access_control.require_access(message.from_user.id)
    except (access_control.AccessDeniedError, access_control.AccessQuotaError):
        return True
    except Exception:
        # A failed permission read must not turn into public access.
        return True
    return False


@bot.message_handler(func=_message_access_blocked, content_types=["text"])
def handle_access_denied(message):
    states.pop(message.chat.id, None)
    bot.send_message(message.chat.id, ACCESS_DENIED_TEXT, reply_markup=types.ReplyKeyboardRemove())


def _callback_access_blocked(call):
    message = getattr(call, "message", None)
    actor = getattr(call, "from_user", None)
    chat = getattr(message, "chat", None)
    if not actor or not chat or getattr(chat, "type", "") != "private" or actor.id != chat.id:
        return True
    data = getattr(call, "data", "") or ""
    if data.startswith(("account_delete:", "memory_clear:")):
        return False
    try:
        access_control.require_access(actor.id)
    except Exception:
        return True
    return False


@bot.callback_query_handler(func=_callback_access_blocked)
def handle_callback_access_denied(call):
    bot.answer_callback_query(call.id, "Доступ закончился или не предоставлен. Нужна действующая ссылка-приглашение.", show_alert=True)


def _invite_expiry(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).strftime("%d.%m.%Y %H:%M UTC")
    except (ValueError, TypeError):
        return "указанного владельцем срока"


def _owner_invite_chat(message):
    try:
        owner = _private_actor(message) and access_control.is_owner(message.from_user.id)
    except Exception:
        owner = False
    if not owner:
        bot.send_message(message.chat.id, "Управлять приглашениями может только владелец в личном чате.")
        return False
    return True


def _send_owner_invitation(owner_id, chat_id, *, days=7, max_users=1):
    """Shared private-only delivery; never log bearer links or provider exceptions."""
    try:
        if (chat_id != owner_id or not access_control.is_owner(owner_id)
                or type(max_users) is not int or max_users != 1
                or type(days) is not int or not 1 <= days <= 30):
            raise ValueError("Invalid invitation request")
        username = str(bot.get_me().username or "")
        if not re.fullmatch(r"[A-Za-z0-9_]{5,32}", username):
            raise ValueError("Invalid bot username")
        invitation = access_control.create_invite(owner_id, days=days, max_users=1, route_limit=5)
        token = invitation.get("token", "")
        if not isinstance(token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,57}", token):
            raise ValueError("Invalid invite token")
        url = f"https://t.me/{username}?start=invite_{token}"
        capacity = "Ссылка для 1 аккаунта."
        bot.send_message(chat_id,
            f"Приглашение до {_invite_expiry(invitation.get('expires_at'))}:\n{url}\n\n"
            f"Номер приглашения: {invitation.get('invite_id', '')}\n"
            f"{capacity} Доступно до 5 попыток создания или ИИ-изменения поездки за весь срок доступа, не ежедневно. "
            "Пересылайте ссылку только выбранным участникам: её может использовать получивший ссылку человек.\n"
            "Посмотреть и отозвать приглашения: /invites", disable_web_page_preview=True)
    except Exception:
        # Telegram/API exceptions may contain credentials or the invite URL.
        bot.send_message(chat_id, "Не удалось подготовить приглашение. Проверьте /invites перед повторной командой.")


def _handle_invite_command(message, *, max_users):
    if not _owner_invite_chat(message):
        return
    parts = (message.text or "").split()
    command = "/invite_one" if max_users == 1 else "/invite"
    if len(parts) > 2 or (len(parts) == 2 and not re.fullmatch(r"(?:[1-9]|[12][0-9]|30)", parts[1])):
        bot.send_message(message.chat.id, f"Напишите {command} или {command} 7. Срок — от 1 до 30 дней.")
        return
    days = int(parts[1]) if len(parts) == 2 else 7
    _send_owner_invitation(message.from_user.id, message.chat.id, days=days, max_users=max_users)


@bot.message_handler(commands=["invite"])
def handle_invite(message):
    _handle_invite_command(message, max_users=1)


@bot.message_handler(commands=["invite_one"])
def handle_invite_one(message):
    _handle_invite_command(message, max_users=1)


@bot.message_handler(commands=["invites"])
def handle_invites(message):
    if not _owner_invite_chat(message):
        return
    try:
        invitations = access_control.list_invites(message.from_user.id)
        keyboard = types.InlineKeyboardMarkup(row_width=1)
        lines = ["Последние 20 приглашений (без секретных ссылок). Отозвать по полному номеру: /revoke номер"
                 if invitations else "Приглашений пока нет."]
        for invitation in invitations[:20]:
            invite_id = str(invitation.get("invite_id", ""))
            revoked = bool(invitation.get("revoked_at"))
            used = invitation.get("joined_users", 0)
            lines.append(f"• {invite_id} · до {_invite_expiry(invitation.get('expires_at'))} · "
                         f"участников {used}/{invitation.get('max_users', 1)} · "
                         f"попыток на участника {invitation.get('route_limit', '?')} · "
                         + ("отозвано" if revoked else "не отозвано"))
            if not revoked and re.fullmatch(r"[a-f0-9]{32}", invite_id):
                keyboard.add(types.InlineKeyboardButton(f"Отозвать {invite_id[:8]}", callback_data=f"invite_revoke:{invite_id}"))
        lines.append("Новая ссылка: один проверяющий, 5 попыток на весь срок, по умолчанию 7 дней. Другой срок: /invite_one [дни] или /invite [дни].")
        keyboard.add(types.InlineKeyboardButton("Пригласить одного", callback_data="invite_create:one"))
        bot.send_message(message.chat.id, "\n".join(lines), reply_markup=keyboard)
    except Exception:
        bot.send_message(message.chat.id, "Не удалось получить список приглашений. Попробуйте позже.")


@bot.message_handler(commands=["revoke"])
def handle_revoke(message):
    if not _owner_invite_chat(message):
        return
    parts = (message.text or "").split()
    if len(parts) != 2 or not re.fullmatch(r"[a-f0-9]{32}", parts[1]):
        bot.send_message(message.chat.id, "Напишите /revoke и полный номер приглашения из /invites.")
        return
    try:
        revoked = access_control.revoke_invite(message.from_user.id, parts[1])
    except Exception:
        bot.send_message(message.chat.id, "Не удалось отозвать приглашение. Проверьте /invites.")
        return
    bot.send_message(message.chat.id, "Приглашение отозвано. Уже отправленный провайдеру запрос отменить нельзя."
                     if revoked else "Приглашение не найдено.")


def _owner_invite_callback(call):
    message = getattr(call, "message", None)
    actor = getattr(call, "from_user", None)
    chat = getattr(message, "chat", None)
    try:
        owner = bool(actor and chat and getattr(chat, "type", "") == "private"
                     and actor.id == chat.id and access_control.is_owner(actor.id))
    except Exception:
        owner = False
    if not owner:
        bot.answer_callback_query(call.id, "Управление доступно только владельцу в личном чате.", show_alert=True)
        return None
    return actor.id


@bot.callback_query_handler(func=lambda call: bool(call.data) and call.data.startswith("invite_create:"))
def handle_invite_create(call):
    owner_id = _owner_invite_callback(call)
    if owner_id is None:
        return
    max_users = {"invite_create:one": 1}.get(call.data)
    if max_users is None:
        bot.answer_callback_query(call.id, "Некорректный вид приглашения.")
        return
    bot.answer_callback_query(call.id, "Подготавливаю приглашение на 7 дней.")
    _send_owner_invitation(owner_id, owner_id, days=7, max_users=max_users)


@bot.callback_query_handler(func=lambda call: bool(call.data) and call.data.startswith("invite_revoke:"))
def handle_invite_revoke(call):
    owner_id = _owner_invite_callback(call)
    if owner_id is None:
        return
    invite_id = (call.data or "").split(":", 1)[-1]
    if not re.fullmatch(r"[a-f0-9]{32}", invite_id):
        bot.answer_callback_query(call.id, "Некорректное приглашение.")
        return
    try:
        revoked = access_control.revoke_invite(owner_id, invite_id)
    except Exception:
        bot.answer_callback_query(call.id, "Не удалось отозвать приглашение. Проверьте /invites.", show_alert=True)
        return
    if not revoked:
        bot.answer_callback_query(call.id, "Приглашение не найдено.")
        return
    bot.answer_callback_query(call.id, "Приглашение отозвано.")
    bot.send_message(owner_id, "Приглашение отозвано. Новые платные операции по нему запрещены. Уже отправленный провайдеру запрос отменить нельзя.")


# ---------- команды ----------
@bot.message_handler(
    commands=["start", "help"]
)
def handle_start(
    message,
):
    if not _private_actor(message):
        bot.send_message(message.chat.id, "Откройте личный чат с ботом.")
        return
    parts = (message.text or "").split(maxsplit=1)
    command = parts[0].split("@", 1)[0].lower() if parts else ""
    if command == "/start" and len(parts) == 2 and parts[1].startswith("invite_"):
        token = parts[1][7:]
        try:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,57}", token):
                raise ValueError("Invalid invite")
            access_control.redeem_invite(message.from_user.id, token)
            grant = access_control.status(message.from_user.id)
        except Exception:
            bot.send_message(message.chat.id, "Приглашение недействительно, истекло, отозвано или уже заполнено. Попросите владельца проверить доступ.")
            return
        bot.send_message(message.chat.id, f"Доступ открыт до {_invite_expiry(grant.get('expires_at'))}. "
                         f"Осталось попыток создания и изменения маршрутов: {grant.get('route_remaining', 0)}. "
                         "Лимит действует за весь срок доступа, не ежедневно. Начать: /plan. Веб-кабинет: /web.")
        return
    try:
        owner_help = access_control.is_owner(message.from_user.id)
    except Exception:
        owner_help = False
    bot.send_message(
        message.chat.id,
        HELP_TEXT + ("\n\nДля владельца: /invite_one [дни] или /invite [дни] — ссылка для одного проверяющего. "
                     "Доступно 5 попыток на весь срок приглашения, по умолчанию 7 дней. /invites — список и кнопка нового приглашения; /revoke номер — отзыв."
                     if owner_help else ""),
        reply_markup=(
            types.ReplyKeyboardRemove()
        ),
    )


@bot.message_handler(
    commands=["cancel"]
)
def handle_cancel(
    message,
):
    states.pop(
        message.chat.id,
        None,
    )

    bot.send_message(
        message.chat.id,
        "Анкету отменила. "
        "Когда будете готовы, "
        "напишите /plan.",
        reply_markup=(
            types.ReplyKeyboardRemove()
        ),
    )


@bot.message_handler(
    commands=["plan"]
)
def handle_plan(
    message,
):
    chat_id = message.chat.id

    if _is_generation_active(
        chat_id
    ):
        bot.send_message(
            chat_id,
            "Ваш маршрут ещё составляется, "
            "подождите минутку.",
        )
        return

    try:
        grant = access_control.require_access(message.from_user.id)
    except access_control.AccessDeniedError as error:
        bot.send_message(chat_id, str(error))
        return
    guest = grant.get("owner") is not True and grant.get("grant_id") != "open"
    if guest and grant.get("route_remaining", 0) <= 0:
        bot.send_message(chat_id, "Попытки создания и ИИ-изменения поездки по приглашению исчерпаны. Попросите владельца выдать новое приглашение.")
        return

    if (not guest and
        used_today(chat_id)
        >= DAILY_LIMIT
    ):
        bot.send_message(
            chat_id,
            "На сегодня лимит "
            f"{DAILY_LIMIT} запросов на создание "
            "и изменение маршрутов "
            "исчерпан. Попробуйте завтра.",
        )
        return

    states[chat_id] = {
        "step": "destination_kind",
        "data": {},
    }

    bot.send_message(
        chat_id,
        "Давайте составим маршрут. "
        "Отменить можно командой /cancel.",
    )

    ask(
        chat_id,
        "destination_kind",
    )


@bot.message_handler(
    commands=["trips"]
)
def handle_trips(
    message,
):
    trips = db.list_trips(
        message.chat.id
    )

    if not trips:
        bot.send_message(
            message.chat.id,
            "Сохранённых поездок "
            "пока нет. Начните с /plan.",
        )
        return

    lines = [
        "Ваши поездки (последние 10):"
    ]

    lines.extend(
        trip_title(trip)
        for trip in trips
    )

    lines.append(
        "\nОткрыть: /trip номер. "
        "Удалить: /delete номер."
    )

    bot.send_message(
        message.chat.id,
        "\n".join(lines),
    )


@bot.message_handler(
    commands=["web"]
)
def handle_web(
    message,
):
    """Выдаёт одноразовую ссылку без передачи tg_id браузером."""

    chat_id = _message_user_id(
        message
    )

    try:
        url = web_auth.build_login_url(
            chat_id
        )
    except web_auth.WebAuthConfigError:
        logger.exception(
            "Веб-кабинет не настроен"
        )
        bot.send_message(
            message.chat.id,
            "Веб-кабинет пока не запущен. "
            "Попробуйте немного позже.",
        )
        return

    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(
        types.InlineKeyboardButton(
            "Открыть мои поездки",
            url=url,
        )
    )

    bot.send_message(
        message.chat.id,
        "Ссылка действует ограниченное время "
        "и открывает только ваши поездки.",
        reply_markup=keyboard,
    )


@bot.message_handler(
    commands=["reminders"]
)
def handle_reminders(
    message,
):
    chat_id = message.chat.id

    if getattr(
        message.chat,
        "type",
        "",
    ) != "private":
        bot.send_message(
            chat_id,
            "Напоминания доступны только "
            "в личном чате с ботом.",
        )
        return

    number = _trip_number(
        message
    )

    if number is None:
        trips = db.list_trips(
            chat_id,
            limit=1,
        )

        if not trips:
            bot.send_message(
                chat_id,
                "Сначала составьте маршрут: "
                "/plan",
            )
            return

        number = trips[0]["id"]

    trip = db.get_trip(
        number,
        chat_id,
    )

    if not trip:
        bot.send_message(
            chat_id,
            "Такой поездки у вас нет. "
            "Список: /trips",
        )
        return

    try:
        text = reminders.describe_plan(
            trip,
            db.get_reminder_statuses(
                number,
                chat_id,
            ),
            now=datetime.now(
                reminders.get_tz()
            ),
        )
    except (
        RuntimeError,
        ValueError,
    ):
        logger.exception(
            "Не удалось показать "
            "напоминания поездки %s",
            number,
        )

        bot.send_message(
            chat_id,
            "Не получилось открыть "
            "расписание напоминаний. "
            "Попробуйте немного позже.",
        )
        return

    bot.send_message(
        chat_id,
        text,
    )


@bot.message_handler(
    commands=["edit"]
)
def handle_edit(
    message,
):
    import access_control
    from access_guard import operation, snapshot
    chat_id = message.chat.id
    try:
        access_control.require_access(chat_id)
        access_grant = snapshot(chat_id) if access_control.enabled() else None
    except (access_control.AccessDeniedError, access_control.AccessQuotaError) as error:
        bot.send_message(chat_id, str(error))
        return
    parts = (
        message.text or ""
    ).split(
        maxsplit=3
    )

    if (
        len(parts) < 4
        or not parts[1].isdigit()
        or not parts[2].isdigit()
    ):
        bot.send_message(
            chat_id,
            "Напишите номер поездки, номер дня "
            "и пожелание. Например:\n"
            "/edit 2 1 Убери музей, добавь парк",
        )
        return

    trip_id = int(
        parts[1]
    )
    target_day = int(
        parts[2]
    )

    try:
        edit_request = _clean_edit_request(
            parts[3]
        )
    except ValueError as error:
        bot.send_message(
            chat_id,
            str(error),
        )
        return

    trip = db.get_trip(
        trip_id,
        chat_id,
    )

    if not trip:
        bot.send_message(
            chat_id,
            "Такой поездки у вас нет. "
            "Список: /trips",
        )
        return

    try:
        saved_itinerary = (
            Itinerary.model_validate(
                trip["itinerary"]
            )
        )
    except Exception:
        logger.exception(
            "Повреждён сохранённый маршрут "
            "поездки %s",
            trip_id,
        )
        bot.send_message(
            chat_id,
            "Сохранённый маршрут повреждён, "
            "его нельзя безопасно изменить.",
        )
        return

    if not 1 <= target_day <= len(
        saved_itinerary.days
    ):
        bot.send_message(
            chat_id,
            "В этой поездке нет такого дня. "
            "Допустимые номера: от 1 до "
            f"{len(saved_itinerary.days)}.",
        )
        return

    worker_kwargs = {}
    if getattr(saved_itinerary, "city_schedule", False) is True:
        # A legacy city snapshot may need bounded read-only recovery. Do it
        # before claiming paid AI, not implicitly inside the editor worker.
        try:
            saved_request = TripRequest.model_validate(trip["request"])
            revision = trip.get("revision", 1)
            if type(revision) is not int or revision < 1:
                raise ValueError(CONTEXT_ERROR)
            context = prepare_stored_city_context(trip["request"], saved_request, saved_itinerary)
            if type(context) is not dict:
                raise ValueError(CONTEXT_ERROR)
            worker_kwargs = {"city_map_context": context, "expected_revision": revision}
        except Exception:
            # Provider exceptions may contain URLs or credentials. Expose only
            # the fixed safe message and never log the exception/response.
            logger.warning("Данные карты не подтверждены перед правкой поездки %s", trip_id)
            bot.send_message(chat_id, CONTEXT_ERROR)
            return
        bot.send_message(chat_id, CITY_EDIT_NOTE)

    if not _reserve_generation_job(
        chat_id
    ):
        bot.send_message(
            chat_id,
            "Другая операция с маршрутом "
            "уже выполняется. Подождите немного.",
        )
        return

    try:
        if access_grant is not None:
            with operation(chat_id, expected_grant=access_grant):
                claim = _claim_route_ai_request(chat_id, "edit", trip_id)
        else:
            claim = _claim_route_ai_request(chat_id, "edit", trip_id)
    except (access_control.AccessDeniedError, access_control.AccessQuotaError) as error:
        _finish_generation_job(chat_id)
        bot.send_message(chat_id, str(error))
        return
    if claim.status != "ok":
        _finish_generation_job(
            chat_id
        )
        bot.send_message(
            chat_id,
            _ai_claim_error_text(claim),
        )
        return

    if access_grant is not None:
        worker_kwargs["access_grant"] = access_grant
    try:
        bot.send_message(
            chat_id,
            "Изменяю день "
            f"{target_day} поездки №{trip_id}. "
            "Остальные дни останутся без изменений. "
            "Проверяю семейные правила, бюджет "
            "и места на карте.",
        )
        threading.Thread(
            target=edit_generation_job,
            args=(
                chat_id,
                trip_id,
                target_day,
                edit_request,
                claim.claim_id,
            ),
            **({"kwargs": worker_kwargs} if worker_kwargs else {}),
            daemon=True,
            name=(
                f"route-edit-{chat_id}-{trip_id}"
            ),
        ).start()
    except Exception:
        _finish_ai_claim(
            claim.claim_id,
            False,
        )
        _finish_generation_job(
            chat_id
        )
        raise


@bot.message_handler(
    commands=["trip"]
)
def handle_trip(
    message,
):
    number = _trip_number(
        message
    )

    if number is None:
        bot.send_message(
            message.chat.id,
            "Напишите номер поездки, "
            "например: /trip 3. "
            "Список: /trips",
        )
        return

    trip = db.get_trip(
        number,
        message.chat.id,
    )

    if not trip:
        bot.send_message(
            message.chat.id,
            "Такой поездки у вас нет. "
            "Список: /trips",
        )
        return

    request = trip["request"]

    text = format_itinerary(
        Itinerary.model_validate(
            trip["itinerary"]
        ),
        request.get(
            "currency",
            "EUR",
        ),
        float(
            request.get(
                "budget",
                0,
            )
        ),
        request.get(
            "weather_text",
            "",
        ),
        children_ages=request.get(
            "children_ages",
            [],
        ),
        travel_cost=float(request.get("travel_cost", 0)),
        lodging_cost=float(request.get("lodging_cost", 0)),
        saved_request=request,
    )

    send_parts(
        message.chat.id,
        text,
    )


@bot.message_handler(
    commands=["checklist"]
)
def handle_checklist(
    message,
):
    chat_id = message.chat.id
    number = _trip_number(
        message
    )

    if number is None:
        trips = db.list_trips(
            chat_id,
            limit=1,
        )

        if not trips:
            bot.send_message(
                chat_id,
                "Сначала составьте "
                "маршрут: /plan",
            )
            return

        number = trips[0]["id"]

    trip = db.get_trip(
        number,
        chat_id,
    )

    if not trip:
        bot.send_message(
            chat_id,
            "Такой поездки у вас нет. "
            "Список: /trips",
        )
        return

    if not db.get_checklist(
        number
    ):
        request = trip["request"]

        db.save_checklist(
            number,
            build_checklist(
                int(
                    request.get(
                        "adults",
                        2,
                    )
                ),
                request.get(
                    "children_ages",
                    [],
                ),
                request.get(
                    "weather"
                ),
            ),
        )

    send_checklist(
        chat_id,
        number,
    )


@bot.callback_query_handler(
    func=lambda call:
    bool(call.data)
    and call.data.startswith("ck:")
)
def handle_check(
    call,
):
    if (
        not call.message
        or getattr(
            call.message.chat,
            "type",
            "",
        ) != "private"
        or call.from_user.id
        != call.message.chat.id
    ):
        bot.answer_callback_query(
            call.id,
            "Доступно только "
            "в личном чате",
        )
        return

    try:
        item_id = int(
            call.data[3:]
        )
    except (
        TypeError,
        ValueError,
    ):
        bot.answer_callback_query(
            call.id,
            "Некорректная кнопка",
        )
        return

    item = db.get_checklist_item(
        item_id,
        call.from_user.id,
    )

    if not item:
        bot.answer_callback_query(
            call.id,
            "Пункт не найден",
        )
        return

    db.set_item_done(
        item_id,
        not item["done"],
    )

    bot.edit_message_reply_markup(
        call.message.chat.id,
        call.message.message_id,
        reply_markup=(
            category_keyboard(
                item["trip_id"],
                item["category"],
            )
        ),
    )

    bot.answer_callback_query(
        call.id
    )


@bot.message_handler(
    commands=["pdf"]
)
def handle_pdf(
    message,
):
    chat_id = message.chat.id
    number = _trip_number(
        message
    )

    if number is None:
        trips = db.list_trips(
            chat_id,
            limit=1,
        )

        if not trips:
            bot.send_message(
                chat_id,
                "Сначала составьте "
                "маршрут: /plan",
            )
            return

        number = trips[0]["id"]

    trip = db.get_trip(
        number,
        chat_id,
    )

    if not trip:
        bot.send_message(
            chat_id,
            "Такой поездки у вас нет. "
            "Список: /trips",
        )
        return

    try:
        bot.send_chat_action(
            chat_id,
            "upload_document",
        )

        pdf_bytes = build_trip_pdf(
            trip,
            db.get_checklist(
                number
            ),
        )

        document = io.BytesIO(
            pdf_bytes
        )
        document.name = (
            f"trip_{number}.pdf"
        )

        bot.send_document(
            chat_id,
            document,
            caption=(
                "Маршрут и чек-лист: "
                f"{trip_title(trip)}"
            ),
        )

    except Exception:
        logger.exception(
            "Ошибка при создании PDF"
        )

        bot.send_message(
            chat_id,
            "Не получилось собрать PDF. "
            "Попробуйте ещё раз "
            "чуть позже.",
        )


@bot.message_handler(
    commands=["delete"]
)
def handle_delete(
    message,
):
    chat_id = message.chat.id
    if not _private_actor(message):
        bot.send_message(chat_id, "Удаление поездки доступно только в личном чате с ботом.")
        return

    try:
        number = _trip_number(message)
    except (ValueError, OverflowError):
        number = None

    if number is None or number <= 0:
        bot.send_message(
            chat_id,
            "Напишите номер поездки, "
            "например: /delete 3. "
            "Список: /trips",
        )
        return

    # Privacy actions stay available after an invitation expires. The shared
    # service checks ownership/account lifetime, active jobs and exact files.
    import trip_deletion
    import web_service

    try:
        account_key = db.get_account_key(message.from_user.id)
        if not account_key:
            bot.send_message(chat_id,
                             "Профиль недоступен или его удаление ещё выполняется. "
                             "Удаление поездки не запущено.")
            return
        result = web_service.delete_trip(
            message.from_user.id, number, expected_account_key=account_key,
        )
    except web_service.TripNotFoundError:
        bot.send_message(
            chat_id,
            "Такой поездки у вас нет. "
            "Список: /trips",
        )
        return
    except trip_deletion.TripDeleteBusyError:
        bot.send_message(chat_id,
                         "Сейчас выполняется операция с маршрутом, обложкой или напоминанием. "
                         f"Дождитесь её завершения и повторите /delete {number}. Поездка сохранена.")
        return
    except trip_deletion.TripDeleteStorageError:
        bot.send_message(chat_id,
                         "Не удалось безопасно проверить файлы поездки. Поездка сохранена. "
                         "Попробуйте позже или обратитесь к владельцу приложения.")
        return
    except Exception:
        logger.warning("trip_delete_result_unconfirmed")
        bot.send_message(chat_id,
                         "Не удалось подтвердить результат удаления. Проверьте список /trips; "
                         f"если поездка осталась, повторите /delete {number}.")
        return

    text = result["message"]
    if result["cleanup_pending"]:
        text += f" Повторить очистку файлов можно командой /delete {number}."
    bot.send_message(chat_id, text)


@bot.message_handler(
    commands=["memory"]
)
def handle_memory(
    message,
):
    items = db.list_memory_items(
        message.chat.id,
        limit=20,
    )

    if not items:
        bot.send_message(
            message.chat.id,
            "Память пока пуста.\n\n"
            "Добавить предпочтение: "
            "/remember текст",
        )
        return

    lines = [
        "Что я помню о вашей семье:"
    ]

    for number, item in enumerate(
        items,
        start=1,
    ):
        label = (
            MEMORY_KIND_LABELS.get(
                item["kind"],
                "Запись",
            )
        )

        destination = str(
            item.get(
                "destination",
                "",
            )
        ).strip()

        suffix = (
            f" — {destination}"
            if destination
            else ""
        )

        lines.append(
            f"\n{number}. "
            f"{label}{suffix}\n"
            f"{item['text']}"
        )

    lines.append(
        "\nОчистить всю память: "
        "/forget_memory"
    )

    send_parts(
        message.chat.id,
        "\n".join(lines),
    )


@bot.message_handler(
    commands=["remember"]
)
def handle_remember(
    message,
):
    import access_control
    from access_guard import snapshot
    try:
        access_control.require_access(message.chat.id)
        access_grant = snapshot(message.chat.id) if access_control.enabled() else None
    except (access_control.AccessDeniedError, access_control.AccessQuotaError) as error:
        bot.send_message(message.chat.id, str(error))
        return
    parts = (
        message.text or ""
    ).split(
        maxsplit=1
    )

    if len(parts) < 2:
        bot.send_message(
            message.chat.id,
            "После команды напишите, "
            "что нужно запомнить.\n"
            "Например:\n"
            "/remember ребёнку нужен "
            "дневной отдых",
        )
        return

    text = _clean_memory_text(
        parts[1],
        MAX_MANUAL_MEMORY_CHARS,
    )

    if len(text) < 3:
        bot.send_message(
            message.chat.id,
            "Заметка слишком короткая.",
        )
        return

    chat_id = message.chat.id

    if not _reserve_memory_job(
        chat_id
    ):
        bot.send_message(
            chat_id,
            "Предыдущая операция с памятью "
            "ещё выполняется. "
            "Подождите немного.",
        )
        return

    bot.send_message(
        chat_id,
        "Сохраняю запись в память...",
    )

    threading.Thread(
        target=_save_custom_memory_job,
        **({"kwargs": {"access_grant": access_grant}} if access_grant is not None else {}),
        args=(
            chat_id,
            _message_user_id(
                message
            ),
            text,
        ),
        daemon=True,
        name=(
            f"memory-save-{chat_id}"
        ),
    ).start()


@bot.message_handler(
    commands=["feedback"]
)
def handle_feedback(
    message,
):
    import access_control
    from access_guard import snapshot
    try:
        access_control.require_access(message.chat.id)
        access_grant = snapshot(message.chat.id) if access_control.enabled() else None
    except (access_control.AccessDeniedError, access_control.AccessQuotaError) as error:
        bot.send_message(message.chat.id, str(error))
        return
    parts = (
        message.text or ""
    ).split(
        maxsplit=2
    )

    if (
        len(parts) < 3
        or not parts[1].isdigit()
    ):
        bot.send_message(
            message.chat.id,
            "Напишите номер поездки "
            "и отзыв.\n"
            "Например:\n"
            "/feedback 3 детям понравились "
            "музеи, но было мало отдыха",
        )
        return

    trip_id = int(
        parts[1]
    )

    trip = db.get_trip(
        trip_id,
        message.chat.id,
    )

    if not trip:
        bot.send_message(
            message.chat.id,
            "Такой поездки у вас нет. "
            "Список: /trips",
        )
        return

    feedback = _clean_memory_text(
        parts[2],
        MAX_FEEDBACK_CHARS,
    )

    if len(feedback) < 3:
        bot.send_message(
            message.chat.id,
            "Отзыв слишком короткий.",
        )
        return

    chat_id = message.chat.id

    if not _reserve_memory_job(
        chat_id
    ):
        bot.send_message(
            chat_id,
            "Предыдущая операция с памятью "
            "ещё выполняется. "
            "Подождите немного.",
        )
        return

    bot.send_message(
        chat_id,
        "Сохраняю отзыв в память...",
    )

    threading.Thread(
        target=(
            _save_feedback_memory_job
        ),
        **({"kwargs": {"access_grant": access_grant}} if access_grant is not None else {}),
        args=(
            chat_id,
            _message_user_id(
                message
            ),
            trip_id,
            str(
                trip.get(
                    "destination",
                    "",
                )
            ).strip(),
            feedback,
        ),
        daemon=True,
        name=(
            f"feedback-save-{chat_id}"
        ),
    ).start()


@bot.message_handler(
    commands=["forget_memory"]
)
def handle_forget_memory(
    message,
):
    if _is_generation_active(
        message.chat.id
    ):
        bot.send_message(
            message.chat.id,
            "Сейчас составляется маршрут. "
            "Дождитесь его завершения, "
            "а затем очистите память.",
        )
        return

    bot.send_message(
        message.chat.id,
        "Удалить все сохранённые "
        "предпочтения и отзывы?\n\n"
        "Сами поездки, PDF и чек-листы "
        "останутся.",
        reply_markup=(
            forget_memory_keyboard()
        ),
    )


@bot.callback_query_handler(
    func=lambda call:
    bool(call.data)
    and call.data.startswith(
        "memory_clear:"
    )
)
def handle_memory_clear(
    call,
):
    if (
        not call.message
        or getattr(
            call.message.chat,
            "type",
            "",
        ) != "private"
        or call.from_user.id
        != call.message.chat.id
    ):
        bot.answer_callback_query(
            call.id,
            "Доступно только "
            "в личном чате",
        )
        return

    chat_id = (
        call.message.chat.id
    )

    action = call.data.split(
        ":",
        maxsplit=1,
    )[1]

    if action == "cancel":
        bot.answer_callback_query(
            call.id,
            "Отменено",
        )

        try:
            bot.edit_message_text(
                "Очистка памяти отменена.",
                chat_id,
                call.message.message_id,
            )
        except Exception:
            logger.warning(
                "Не удалось изменить "
                "сообщение отмены",
                exc_info=True,
            )

        return

    if action != "confirm":
        bot.answer_callback_query(
            call.id,
            "Некорректная команда",
        )
        return

    if _is_generation_active(
        chat_id
    ):
        bot.answer_callback_query(
            call.id,
            "Дождитесь готового маршрута",
        )

        bot.send_message(
            chat_id,
            "Маршрут ещё составляется. "
            "После завершения снова "
            "вызовите /forget_memory.",
        )
        return

    if not _reserve_memory_job(
        chat_id
    ):
        bot.answer_callback_query(
            call.id,
            "Другая операция ещё выполняется",
        )
        return

    bot.answer_callback_query(
        call.id,
        "Начинаю очистку",
    )

    try:
        bot.edit_message_text(
            "Удаляю сохранённую память...",
            chat_id,
            call.message.message_id,
        )
    except Exception:
        logger.warning(
            "Не удалось изменить "
            "сообщение очистки",
            exc_info=True,
        )

    threading.Thread(
        target=_clear_memory_job,
        args=(
            chat_id,
            call.from_user.id,
        ),
        daemon=True,
        name=(
            f"memory-clear-{chat_id}"
        ),
    ).start()


# ---------- анкета ----------
@bot.message_handler(
    func=lambda message:
    message.content_type == "text"
    and message.chat.id in states
    and not (
        message.text or ""
    ).startswith("/"),
)
def handle_answer(
    message,
):
    chat_id = message.chat.id
    state = states[chat_id]
    step = state["step"]
    data = state["data"]
    text = (
        message.text or ""
    ).strip()

    try:
        if step == "destination_kind":
            data["destination_kind"] = (
                anketa.parse_destination_kind(
                    text
                )
            )
            next_step = "destination"

        elif step == "destination":
            if not 2 <= len(text) <= 100:
                raise ValueError(
                    "Напишите город, регион "
                    "или озеро и страну, "
                    "например: Париж, Франция "
                    "или Байкал, Россия"
                )

            data["destination"] = text
            next_step = "dates"

        elif step == "dates":
            (
                data["start_date"],
                data["end_date"],
            ) = anketa.parse_dates(
                text
            )

            next_step = "family"

        elif step == "family":
            profile = db.get_user(
                chat_id
            )

            if (
                text.lower()
                in (
                    "да",
                    "yes",
                    "ок",
                )
                and profile
            ):
                data["adults"] = (
                    profile["adults"]
                )

                data[
                    "children_ages"
                ] = profile[
                    "children_ages"
                ]
            else:
                (
                    data["adults"],
                    data[
                        "children_ages"
                    ],
                ) = anketa.parse_family(
                    text
                )

            next_step = "budget"

        elif step == "budget":
            profile = db.get_user(
                chat_id
            )

            default_currency = (
                profile["currency"]
                if profile
                else "EUR"
            )

            (
                data["budget"],
                data["currency"],
            ) = anketa.parse_budget(
                text,
                default_currency,
            )

            next_step = "travel_cost"

        elif step == "travel_cost":
            data["travel_cost"] = anketa.parse_fixed_cost(text, data["currency"])
            if data["travel_cost"] >= data["budget"]:
                raise ValueError("Дорога занимает весь бюджет. Укажите меньшую сумму или начните анкету заново с большим бюджетом: /plan.")
            next_step = "lodging_cost"

        elif step == "lodging_cost":
            lodging_cost = anketa.parse_fixed_cost(text, data["currency"])
            candidate = TripRequest(**{**data, "lodging_cost": lodging_cost})
            minimum_program = minimum_meal_cost(candidate) * candidate.days_count
            if candidate.program_budget < minimum_program:
                raise ValueError(f"После дороги и жилья на программу нужно оставить хотя бы {minimum_program:g} {candidate.currency} на обеды за все дни. Укажите меньшую сумму или начните заново: /plan.")
            data["lodging_cost"] = lodging_cost
            next_step = "pace"

        elif step == "pace":
            if (
                text.lower()
                not in anketa.PACES
            ):
                raise ValueError(
                    "Выберите темп кнопкой "
                    "или напишите: спокойный, "
                    "средний или насыщенный"
                )

            data["pace"] = (
                text.lower()
            )

            next_step = "wishes"

        else:
            data["wishes"] = (
                ""
                if text in (
                    "-",
                    "—",
                    "нет",
                )
                else text[:500]
            )

            states.pop(
                chat_id,
                None,
            )

            start_generation(
                message,
                data,
            )
            return

    except ValueError as error:
        bot.send_message(
            chat_id,
            str(error),
        )
        return

    state["step"] = next_step

    if step in {
        "destination_kind",
        "pace",
    }:
        bot.send_message(
            chat_id,
            "Принято.",
            reply_markup=(
                types.ReplyKeyboardRemove()
            ),
        )

    ask(
        chat_id,
        next_step,
    )


@bot.message_handler(commands=["delete_data"])
def handle_delete_data(message):
    chat_id = message.chat.id
    if getattr(message.chat, "type", "") != "private" or _message_user_id(message) != chat_id:
        bot.send_message(chat_id, "Удаление данных доступно только в личном чате с ботом.")
        return
    keyboard = types.InlineKeyboardMarkup()
    keyboard.add(types.InlineKeyboardButton("Удалить все мои данные", callback_data="account_delete:confirm"))
    keyboard.add(types.InlineKeyboardButton("Отмена", callback_data="account_delete:cancel"))
    bot.send_message(chat_id,
                     "Удалить профиль, все поездки, чек-листы, напоминания, память и доступ к кабинету? "
                     "Это действие нельзя отменить. Скачанные вами файлы и резервные копии оператора не удаляются автоматически. "
                     "Чтобы удаление не сбрасывало лимиты, сохраняются хешированные идентификаторы и служебные записи "
                     "о выдаче доступа и расходе попыток. Эти записи псевдонимизированы, но не анонимны.",
                     reply_markup=keyboard)


@bot.callback_query_handler(func=lambda call: bool(call.data) and call.data.startswith("account_delete:"))
def handle_account_delete(call):
    if (not call.message or getattr(call.message.chat, "type", "") != "private"
            or call.from_user.id != call.message.chat.id):
        bot.answer_callback_query(call.id, "Доступно только владельцу в личном чате")
        return
    chat_id = call.message.chat.id
    action = call.data.split(":", 1)[1]
    if action == "cancel":
        bot.answer_callback_query(call.id, "Отменено")
        bot.edit_message_text("Удаление данных отменено.", chat_id, call.message.message_id)
        return
    if action != "confirm":
        bot.answer_callback_query(call.id, "Некорректная команда")
        return
    if not _reserve_generation_job(chat_id):
        bot.answer_callback_query(call.id, "Дождитесь завершения работы с маршрутом")
        return
    if not _reserve_memory_job(chat_id):
        _finish_generation_job(chat_id)
        bot.answer_callback_query(call.id, "Дождитесь завершения работы с памятью")
        return
    bot.answer_callback_query(call.id, "Начинаю удаление")
    try:
        bot.edit_message_text("Удаляю ваши данные...", chat_id, call.message.message_id)
    except Exception:
        logger.warning("Не удалось обновить сообщение удаления аккаунта")
    try:
        threading.Thread(target=_delete_account_job, args=(chat_id, call.from_user.id),
                         daemon=True, name=f"account-delete-{chat_id}").start()
    except Exception:
        _finish_memory_job(chat_id)
        _finish_generation_job(chat_id)
        logger.exception("Не удалось запустить удаление аккаунта")
        bot.send_message(chat_id, "Не удалось начать удаление. Попробуйте /delete_data ещё раз.")


def _delete_account_job(chat_id, user_id):
    try:
        privacy.delete_all_user_data(chat_id, actor_tg_id=user_id, confirm=True)
        states.pop(chat_id, None)
        bot.send_message(chat_id, "Профиль, поездки, чек-листы, напоминания и память удалены из активного хранилища. "
                                 "Старые ссылки и сессии кабинета больше не действуют. "
                                 "Для защиты от сброса лимитов сохранены хешированные идентификаторы и служебные записи "
                                 "о выдаче доступа и расходе попыток. Эти записи псевдонимизированы, но не анонимны. "
                                 "Для новой поездки при действующем доступе и оставшихся попытках: /plan.")
    except privacy.AccountDeletionBusyError:
        bot.send_message(chat_id, "Ещё выполняется операция с поездкой. Дождитесь её завершения и повторите /delete_data.")
    except Exception:
        logger.exception("Удаление аккаунта не завершено")
        bot.send_message(chat_id, "Не удалось завершить удаление всех данных. Повторите /delete_data; если ошибка сохранится, обратитесь к оператору.")
    finally:
        _finish_memory_job(chat_id)
        _finish_generation_job(chat_id)


# ---------- генерация ----------
def start_generation(
    message,
    data: dict,
) -> None:
    import access_control
    from access_guard import operation, snapshot
    chat_id = message.chat.id
    user_id = _message_user_id(
        message
    )
    try:
        access_control.require_access(user_id)
        access_grant = snapshot(user_id) if access_control.enabled() else None
    except (access_control.AccessDeniedError, access_control.AccessQuotaError) as error:
        bot.send_message(chat_id, str(error))
        return

    name = (
        message.from_user.first_name
        if message.from_user
        else ""
    ) or ""

    request = TripRequest(
        **data
    )

    if request.program_budget < minimum_meal_cost(request) * request.days_count:
        bot.send_message(chat_id, "После дороги и жилья бюджета не хватает даже на обеды. Начните анкету заново: /plan.")
        return

    # Keep the disclosure before any quota claim, including direct callers.
    if request.destination_kind == "city" and city_candidates_enabled():
        from city_rollout import city_deterministic_enabled
        from city_preferences import location_preference_problems
        if city_deterministic_enabled():
            location_problems = location_preference_problems(request)
            if location_problems:
                bot.send_message(chat_id, location_problems[0] + " Начните анкету заново: /plan.")
                return
        bot.send_message(chat_id, CITY_MVP_DISCLOSURE)

    # Resolve the destination before quota or memory work. This server-owned
    # result is carried into the worker; a map ambiguity never spends a turn.
    resolved_city = None
    if request.destination_kind == "city":
        try:
            resolved_city = resolve_city_for_planning(request.destination)
        except CityResolutionError as error:
            next_step = ("Повторите заполнение анкеты позже: /plan. "
                         if error.reason in {"unavailable", "invalid_response"} else
                         "Уточните направление в новой анкете: /plan. ")
            bot.send_message(
                chat_id,
                f"{error}\nПопытка создания поездки не списана. "
                "Новая поездка не сохранена. " + next_step
                + "Платная генерация автоматически не запускается.",
            )
            return

    db.save_user(
        chat_id,
        name,
        request.adults,
        request.children_ages,
        request.currency,
        pace=request.pace,
        wishes=request.wishes,
    )

    if not _reserve_generation_job(
        chat_id
    ):
        bot.send_message(
            chat_id,
            "Другая операция с маршрутом "
            "уже выполняется. Подождите немного.",
        )
        return

    try:
        if access_grant is not None:
            with operation(chat_id, expected_grant=access_grant):
                claim = _claim_route_ai_request(chat_id, "plan")
        else:
            claim = _claim_route_ai_request(chat_id, "plan")
    except (access_control.AccessDeniedError, access_control.AccessQuotaError) as error:
        _finish_generation_job(chat_id)
        bot.send_message(chat_id, str(error))
        return
    if claim.status != "ok":
        _finish_generation_job(
            chat_id
        )
        bot.send_message(
            chat_id,
            _ai_claim_error_text(claim),
        )
        return

    try:
        bot.send_message(
            chat_id,
            (
                "Составляю маршрут: "
                f"{request.destination}, "
                f"{request.days_count} дн. "
                "Проверяю места на карте, "
                "учитываю возраст детей, "
                "погоду и сохранённую память. "
                "Это займёт около минуты."
            ),
            reply_markup=(
                types.ReplyKeyboardRemove()
            ),
        )

        threading.Thread(
            target=generation_job,
            kwargs={
                **({"access_grant": access_grant} if access_grant is not None else {}),
                **({"resolved_city": resolved_city} if resolved_city is not None else {}),
            },
            args=(
                chat_id,
                user_id,
                request,
                claim.claim_id,
            ),
            daemon=True,
            name=(
                f"route-{chat_id}"
            ),
        ).start()
    except Exception:
        _finish_ai_claim(
            claim.claim_id,
            False,
        )
        _finish_generation_job(
            chat_id
        )
        raise


def generation_job(
    chat_id: int,
    user_id: int,
    request: TripRequest,
    claim_id: Optional[str] = None,
    *,
    access_grant: Optional[str] = None,
    resolved_city=None,
) -> None:
    import access_control
    from access_guard import operation
    from contextlib import ExitStack
    access_scope = ExitStack()
    ai_completed = False
    saved_trip_id: Optional[int] = None
    try:
        access_scope.enter_context(operation(user_id, expected_grant=access_grant))
        access_control.require_access(user_id)
        bot.send_chat_action(
            chat_id,
            "typing",
        )

        matches = (
            vector_memory
            .recall_family_memories(
                chat_id,
                user_id,
                _memory_query(
                    request
                ),
                top_k=8,
                min_score=0.35,
                destination=request.destination,
            )
        )

        memory_context = (
            vector_memory
            .format_memory_context(
                matches
            )
        )

        logger.info(
            "Для маршрута пользователя %s "
            "найдено записей памяти: %s",
            chat_id,
            len(matches),
        )

        result = generate_itinerary(
            request,
            memory_context=(
                memory_context
            ),
            ground_candidates=request.destination_kind == "region",
            **({"city_candidates_first": True}
               if request.destination_kind == "city" and city_candidates_enabled() else {}),
            **({"resolved_city": resolved_city}
               if request.destination_kind == "city" and resolved_city is not None else {}),
        )
        if result.problems:
            bot.send_message(
                chat_id,
                generation_failure_message(result.problems),
            )
            return

        request_dict = (
            request.model_dump()
        )

        request_dict[
            "weather_text"
        ] = result.weather_text

        request_dict[
            "weather"
        ] = result.weather

        from generation_quality import KEY as quality_key, safe_coverage
        request_dict[quality_key] = safe_coverage(getattr(result, "generation_coverage", None))

        if getattr(result.itinerary, "city_schedule", False) is True:
            context = getattr(result, "city_map_context", None)
            try:
                if type(context) is not dict:
                    raise ValueError(CONTEXT_ERROR)
                candidates, _ = load_city_context(context, request)
                check_context_places(result.itinerary, candidates)
            except (ValueError, TypeError):
                bot.send_message(chat_id, "Не удалось сохранить подтверждённые данные карты. Новая поездка не сохранена.")
                return
            request_dict[CONTEXT_KEY] = deepcopy(context)

        trip_id = db.save_trip(
            chat_id,
            request_dict,
            result.itinerary.model_dump(),
        )
        saved_trip_id = trip_id
        ai_completed = True

        db.save_checklist(
            trip_id,
            build_checklist(
                request.adults,
                request.children_ages,
                result.weather,
            ),
        )

        # Ошибка памяти не отменяет
        # уже созданную поездку.
        remembered_count = (
            _remember_trip_safely(
                chat_id,
                user_id,
                request,
                trip_id,
            )
        )

        text = format_itinerary(
            result.itinerary,
            request.currency,
            request.budget,
            result.weather_text,
            children_ages=request.children_ages,
            travel_cost=request.travel_cost,
            lodging_cost=request.lodging_cost,
            saved_request=request_dict,
        )

        send_parts(
            chat_id,
            text,
        )

        if request_dict[quality_key].get("quality_status") == "partial":
            bot.send_message(chat_id, "Программа составлена частично: не все дни или выбранные интересы удалось заполнить. "
                             "Это ограничение подбора, а не доказательство отсутствия других мест в городе.")

        from generation_quality import memory_note
        if note := memory_note(request_dict[quality_key]):
            bot.send_message(chat_id, note)

        edit_hint = f"Изменить день: /edit {trip_id} 1 текст. "
        if getattr(result.itinerary, "city_schedule", False) is True:
            edit_hint += CITY_EDIT_NOTE + " "
        footer = (
            "Поездка сохранена "
            f"под номером {trip_id}. "
            "Чек-лист вещей: "
            f"/checklist {trip_id}. "
            f"PDF: /pdf {trip_id}. "
            + edit_hint +
            "Напоминания: "
            f"/reminders {trip_id}. "
            "Все поездки: /trips. "
            "Память: /memory"
        )

        bot.send_message(
            chat_id,
            footer,
        )

        logger.info(
            "Маршрут готов: "
            "пользователь %s, "
            "поездка %s, попыток %s, "
            "токены: вход %s, выход %s, "
            "записей памяти %s",
            chat_id,
            trip_id,
            result.attempts,
            result.prompt_tokens,
            result.completion_tokens,
            remembered_count,
        )

    except (access_control.AccessDeniedError, access_control.AccessQuotaError) as error:
        bot.send_message(chat_id, str(error))
    except CityResolutionError as error:
        bot.send_message(
            chat_id,
            (f"Поездка №{saved_trip_id} сохранена, но её показ не завершился. "
             f"Откройте /trip {saved_trip_id}. Не запускайте создание повторно."
             if saved_trip_id is not None else
             f"{error} Новая поездка не сохранена. Уточните направление в новой анкете: /plan"),
        )
    except Exception:
        logger.exception(
            "Ошибка при составлении маршрута"
        )

        bot.send_message(
            chat_id,
            (f"Поездка №{saved_trip_id} сохранена, но чек-лист или показ результата не завершился. "
             f"Откройте /trip {saved_trip_id}. Не запускайте создание повторно."
             if saved_trip_id is not None else
             "Не получилось составить маршрут. Попробуйте ещё раз чуть позже: /plan"),
        )

    finally:
        access_scope.close()
        _finish_ai_claim(
            claim_id,
            ai_completed,
        )
        _finish_generation_job(
            chat_id
        )


def edit_generation_job(
    chat_id: int,
    trip_id: int,
    target_day: int,
    edit_request: str,
    claim_id: Optional[str] = None,
    *,
    city_map_context: Optional[dict] = None,
    expected_revision: Optional[int] = None,
    access_grant: Optional[str] = None,
) -> None:
    """Изменяет один день и сохраняет только проверенный результат."""

    import access_control
    from access_guard import operation
    from contextlib import ExitStack
    access_scope = ExitStack()
    ai_completed = False
    try:
        access_scope.enter_context(operation(chat_id, expected_grant=access_grant))
        access_control.require_access(chat_id)
        bot.send_chat_action(
            chat_id,
            "typing",
        )

        # Повторно читаем поездку в рабочем потоке:
        # пользователь мог удалить её после команды.
        trip = db.get_trip(
            trip_id,
            chat_id,
        )

        if not trip:
            bot.send_message(
                chat_id,
                "Поездка больше не найдена. "
                "Старые данные не изменялись.",
            )
            return

        request = TripRequest.model_validate(
            trip["request"]
        )
        current_itinerary = (
            Itinerary.model_validate(
                trip["itinerary"]
            )
        )

        revision = trip.get("revision", 1)
        if expected_revision is not None and (
            type(expected_revision) is not int or type(revision) is not int
            or expected_revision != revision
        ):
            bot.send_message(chat_id, "Маршрут уже изменился после команды. Откройте поездку и повторите запрос; AI не запускался.")
            return

        edit_kwargs = {}
        is_city = getattr(current_itinerary, "city_schedule", False) is True
        if is_city:
            # Rebind the preflight snapshot to the freshly loaded request and
            # itinerary. No repeat lookup and no recovery after quota claim.
            try:
                if type(city_map_context) is not dict or expected_revision is None:
                    raise ValueError(CONTEXT_ERROR)
                candidates, _ = load_city_context(city_map_context, request)
                check_context_places(current_itinerary, candidates)
            except (ValueError, TypeError):
                bot.send_message(chat_id, CONTEXT_ERROR)
                return
            edit_kwargs["city_map_context"] = deepcopy(city_map_context)
            edit_kwargs["prior_weather"] = deepcopy(trip["request"].get("weather"))
        elif city_map_context is not None:
            bot.send_message(chat_id, CONTEXT_ERROR)
            return

        result = edit_itinerary(
            request,
            current_itinerary,
            target_day,
            edit_request,
            **edit_kwargs,
        )

        if result.problems:
            bot.send_message(
                chat_id,
                generation_failure_message(result.problems, editing=True),
            )
            return

        request_dict = dict(
            trip["request"]
        )
        request_dict["weather_text"] = (
            result.weather_text
        )
        request_dict["weather"] = (
            result.weather
        )

        from generation_quality import KEY as quality_key, safe_coverage
        request_dict[quality_key] = safe_coverage(getattr(result, "generation_coverage", None))

        if is_city or getattr(result.itinerary, "city_schedule", False) is True:
            context = getattr(result, "city_map_context", None)
            try:
                if getattr(result.itinerary, "city_schedule", False) is not True or type(context) is not dict:
                    raise ValueError(CONTEXT_ERROR)
                context = validate_city_edit_context(city_map_context, context, request, result.itinerary)
                from city_planner import evaluate_saved_city_quality
                updated_candidates, _ = load_city_context(context, request)
                request_dict[quality_key] = safe_coverage(evaluate_saved_city_quality(
                    request, updated_candidates, result.itinerary, trip["request"].get(quality_key)))
            except (ValueError, TypeError):
                bot.send_message(chat_id, "Изменения не сохранены: не удалось проверить данные карты. Старый маршрут сохранён без изменений.")
                return
            request_dict[CONTEXT_KEY] = deepcopy(context)

        saved = db.update_trip_plan(
            trip_id,
            chat_id,
            request_dict,
            result.itinerary.model_dump(),
            expected_revision=int(revision),
        )

        if not saved:
            bot.send_message(
                chat_id,
                "Поездка была удалена или маршрут "
                "уже изменился в другом окне. "
                "Обновите поездку и повторите запрос. "
                "Новые данные не перезаписывались.",
            )
            return
        ai_completed = True

        text = format_itinerary(
            result.itinerary,
            request.currency,
            request.budget,
            result.weather_text,
            children_ages=request.children_ages,
            travel_cost=request.travel_cost,
            lodging_cost=request.lodging_cost,
            saved_request=request_dict,
        )

        bot.send_message(
            chat_id,
            "✅ День "
            f"{target_day} поездки №{trip_id} "
            "обновлён. Остальные дни сохранены.",
        )
        send_parts(
            chat_id,
            text,
        )
        bot.send_message(
            chat_id,
            "Обновлённый PDF: "
            f"/pdf {trip_id}. "
            "Чек-лист: "
            f"/checklist {trip_id}. "
            "Напоминания: "
            f"/reminders {trip_id}.",
        )

        logger.info(
            "Маршрут изменён: пользователь %s, "
            "поездка %s, день %s, попыток %s, "
            "токены: вход %s, выход %s",
            chat_id,
            trip_id,
            target_day,
            result.attempts,
            result.prompt_tokens,
            result.completion_tokens,
        )

    except (access_control.AccessDeniedError, access_control.AccessQuotaError) as error:
        bot.send_message(chat_id, str(error))
    except CityResolutionError as error:
        bot.send_message(
            chat_id,
            f"{error} Изменения не сохранены. Старый маршрут сохранён без изменений. "
            "Для уточнённого направления начните новую анкету: /plan",
        )
    except Exception:
        logger.exception(
            "Ошибка изменения поездки %s "
            "пользователя %s",
            trip_id,
            chat_id,
        )
        bot.send_message(
            chat_id,
            "Не получилось безопасно изменить "
            "маршрут. Старый маршрут сохранён "
            "без изменений. Попробуйте позже.",
        )

    finally:
        access_scope.close()
        _finish_ai_claim(
            claim_id,
            ai_completed,
        )
        _finish_generation_job(
            chat_id
        )


# ---------- остальные сообщения ----------
@bot.message_handler(
    func=lambda message: True,
    content_types=["text"],
)
def handle_other(
    message,
):
    bot.send_message(
        message.chat.id,
        "Чтобы составить маршрут, "
        "напишите /plan. "
        "Все команды: /help",
    )


def setup_commands() -> None:
    try:
        bot.set_my_commands(
            [
                types.BotCommand(
                    "plan",
                    "Составить маршрут",
                ),
                types.BotCommand(
                    "trips",
                    "Мои поездки",
                ),
                types.BotCommand(
                    "trip",
                    "Показать поездку",
                ),
                types.BotCommand(
                    "edit",
                    "Изменить день маршрута",
                ),
                types.BotCommand(
                    "checklist",
                    "Чек-лист вещей",
                ),
                types.BotCommand(
                    "pdf",
                    "Скачать PDF",
                ),
                types.BotCommand(
                    "reminders",
                    "Напоминания о поездке",
                ),
                types.BotCommand(
                    "web",
                    "Открыть веб-кабинет",
                ),
                types.BotCommand(
                    "delete",
                    "Удалить поездку",
                ),
                types.BotCommand(
                    "remember",
                    "Запомнить предпочтение",
                ),
                types.BotCommand(
                    "feedback",
                    "Отзыв о поездке",
                ),
                types.BotCommand(
                    "memory",
                    "Показать память",
                ),
                types.BotCommand(
                    "forget_memory",
                    "Очистить память",
                ),
                types.BotCommand("delete_data", "Удалить все мои данные"),
                types.BotCommand("invite_one", "Владельцу: пригласить один аккаунт"),
                types.BotCommand("invite", "Владельцу: приглашение на 5 попыток"),
                types.BotCommand("invites", "Владельцу: приглашения и отзыв"),
                types.BotCommand(
                    "cancel",
                    "Отменить анкету",
                ),
                types.BotCommand(
                    "help",
                    "Помощь",
                ),
            ]
        )

    except Exception:
        logger.warning(
            "Не удалось обновить "
            "список команд",
            exc_info=True,
        )


if __name__ == "__main__":
    reminder_tz = reminders.get_tz()
    reminders.get_days_before()

    reminder_scheduler = None
    scheduler_started = False

    try:
        # Сначала проверяем токен. При ошибке
        # фоновая отправка не должна запускаться.
        me = bot.get_me()

        logger.info(
            "Бот @%s запущен",
            me.username,
        )

        setup_commands()

        reminder_scheduler = (
            build_reminder_scheduler(
                reminder_tz
            )
        )
        reminder_scheduler.start()
        scheduler_started = True

        bot.infinity_polling(
            skip_pending=True,
        )
    finally:
        if (
            scheduler_started
            and reminder_scheduler
            is not None
        ):
            reminder_scheduler.shutdown(
                wait=False
            )
