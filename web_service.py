"""Сценарии веб-кабинета поверх проверенной логики проекта."""

from __future__ import annotations

import logging
import math
import os
from copy import deepcopy
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

import db
import memory as vector_memory
import access_guard
import reminders
import city_context
import generation_quality
from anketa import MAX_TRIP_DAYS, PACES
from checklist import build_checklist
from city_rollout import CITY_MVP_DISCLOSURE, city_candidates_enabled, city_deterministic_enabled
from city_estimates import city_unpriced_visit_note, unpriced_city_visit_count
from destination_scope import DestinationScopeError
from generation_feedback import generation_failure_message, generation_failure_reason
from verified_map_places import is_public_map_point, item_map_attribution
from pdf_export import build_trip_pdf
from pricing import minimum_meal_cost
from program_coverage import limited_program_problems, program_report_text
from planner import (
    CITY_SCHEDULE_EDIT_UNAVAILABLE,
    edit_itinerary,
    generate_itinerary,
    paid_interest_cost_warning,
)
from schemas import Itinerary, TripRequest
from weather import CityResolutionError, resolve_city_for_planning, safe_city_suggestions


logger = logging.getLogger("web.service")

INTEREST_LABELS = {
    "main_sights": "основные достопримечательности",
    "museums": "музеи и выставки",
    "theatre": "театр",
    "cinema": "кино",
    "water_trips": "морские и речные прогулки",
    "parks": "парки и природа",
    "zoos": "зоопарки и аквариумы",
    "amusement_parks": "аттракционы",
    "shopping_malls": "торговые центры и шопинг",
}
DEFAULT_TRIP_GOAL = "Познакомиться с городом"
DEFAULT_INTERESTS = ["main_sights"]


def planning_capabilities() -> Dict[str, Any]:
    """Public mode/disclosure only; never expose environment or credentials."""
    enabled = city_candidates_enabled()
    return {"city_candidates_first": enabled,
            "city_mvp_disclosure": CITY_MVP_DISCLOSURE if enabled else ""}


class TripNotFoundError(LookupError):
    pass


class TripDataError(ValueError):
    pass


CITY_PREFLIGHT_CODES = frozenset({"city_ambiguous", "city_not_found", "city_unavailable", "city_invalid"})


class TripCreateValidationError(ValueError):
    def __init__(self, message: str, *, diagnostic_problems: Optional[List[str]] = None,
                 code: str = "validation", destination_options=()) -> None:
        self.diagnostic_problems = list(diagnostic_problems or [])
        self.code = code if code in {"insufficient_data", "search_limit", "invalid_request", "validation_error",
                                    "unsupported_preferences", *CITY_PREFLIGHT_CODES} else "validation"
        self.destination_options = (safe_city_suggestions(destination_options)
                                    if self.code == "city_ambiguous" else ())
        super().__init__(message)


class TripSavedResponseError(RuntimeError):
    """Запись уже завершена, но ответ для кабинета собрать не удалось."""

    def __init__(self, trip_id: int, message: str) -> None:
        self.trip_id = trip_id
        super().__init__(message)


class EditRejectedError(ValueError):
    def __init__(self, problems: List[str]) -> None:
        self.diagnostic_problems = list(problems)
        # This field is part of the public API; raw repair instructions are not.
        self.problems = [generation_failure_reason(self.diagnostic_problems)]
        super().__init__(generation_failure_message(self.diagnostic_problems, editing=True))


class EditConflictError(RuntimeError):
    pass


class EditBusyError(RuntimeError):
    pass


class AIDailyLimitError(RuntimeError):
    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        super().__init__(f"Суточный лимит AI-запросов: {self.limit}.")


def _daily_ai_limit() -> int:
    try:
        value = int(os.getenv("DAILY_LIMIT", "3"))
    except ValueError as error:
        raise RuntimeError("DAILY_LIMIT должен быть целым числом.") from error
    if value < 1:
        raise RuntimeError("DAILY_LIMIT должен быть положительным.")
    return value


def _finish_ai_request_safely(claim_id: str, *, succeeded: bool) -> None:
    """Завершает AI-запрос с ограниченным повтором при сбое SQLite."""

    for attempt in range(1, 4):
        try:
            finished = db.finish_ai_request(
                claim_id,
                succeeded=succeeded,
            )
        except Exception as error:
            logger.warning(
                "Попытка %s/3 завершить AI-запрос %s не удалась: %s",
                attempt,
                claim_id,
                type(error).__name__,
            )
            continue

        if not finished:
            logger.error(
                "AI-запрос %s уже не находится в статусе running",
                claim_id,
            )
        return

    logger.error(
        "Не удалось завершить AI-запрос %s после трёх попыток",
        claim_id,
    )


def _normalized_trip_request(request: TripRequest) -> TripRequest:
    """Проверяет веб-анкету и возвращает её нормализованную копию."""

    try:
        value = TripRequest.model_validate(request)
    except ValidationError as error:
        raise ValueError("Проверьте заполнение анкеты поездки.") from error

    destination_kind = getattr(value, "destination_kind", "city")
    destination = " ".join(value.destination.split())
    if not 2 <= len(destination) <= 100:
        destination_label = (
            "Название города"
            if destination_kind == "city"
            else "Название территории"
        )
        raise ValueError(
            f"{destination_label} должно содержать от 2 до 100 символов."
        )

    start_text = str(value.start_date).strip()
    end_text = str(value.end_date).strip()
    try:
        start = date.fromisoformat(start_text)
        end = date.fromisoformat(end_text)
    except ValueError as error:
        raise ValueError("Даты должны быть в формате ГГГГ-ММ-ДД.") from error

    if start_text != start.isoformat() or end_text != end.isoformat():
        raise ValueError("Даты должны быть в формате ГГГГ-ММ-ДД.")

    today = datetime.now(reminders.get_tz()).date()
    if start < today:
        raise ValueError("Дата начала уже прошла.")
    if end < start:
        raise ValueError("Дата окончания не может быть раньше даты начала.")
    if (end - start).days + 1 > MAX_TRIP_DAYS:
        raise ValueError(
            f"Маршрут можно составить не более чем на {MAX_TRIP_DAYS} дней."
        )

    adults = value.adults
    if isinstance(adults, bool) or not 1 <= adults <= 10:
        raise ValueError("Число взрослых должно быть от 1 до 10.")

    children_ages = list(value.children_ages)
    if len(children_ages) > 6:
        raise ValueError("Укажите не больше 6 детей.")
    if any(
        isinstance(age, bool) or not isinstance(age, int) or not 0 <= age <= 17
        for age in children_ages
    ):
        raise ValueError("Возраст каждого ребёнка должен быть от 0 до 17 лет.")

    budget = float(value.budget)
    if not math.isfinite(budget) or budget <= 0 or budget > 1_000_000_000:
        raise ValueError(
            "Бюджет должен быть конечным числом от 1 до 1 000 000 000."
        )

    fixed_amounts = {}
    for field, label in (("travel_cost", "дороги"), ("lodging_cost", "жилья")):
        amount = float(getattr(value, field, 0))
        if not math.isfinite(amount) or not 0 <= amount <= 1_000_000_000:
            raise ValueError(
                f"Стоимость {label} должна быть конечным числом "
                "от 0 до 1 000 000 000."
            )
        fixed_amounts[field] = round(amount, 2)
    budget = round(budget, 2)
    program_budget = round(budget - sum(fixed_amounts.values()), 2)
    if program_budget <= 0:
        raise ValueError(
            "После дороги и жилья должен остаться бюджет на программу. "
            "Увеличьте общий бюджет или уменьшите эти суммы."
        )

    currency = str(value.currency).strip().upper()
    if currency not in {"EUR", "USD", "RUB"}:
        raise ValueError("Поддерживаются валюты EUR, USD и RUB.")

    meal_minimum = minimum_meal_cost(currency, adults, children_ages) * (
        (end - start).days + 1
    )
    if program_budget < meal_minimum:
        raise ValueError(
            f"После дороги и жилья на программу остаётся {program_budget:g} "
            f"{currency}. Для одного обеда в день на всю группу нужно "
            f"хотя бы {meal_minimum:g} {currency} за поездку."
        )

    pace = " ".join(str(value.pace).split()).casefold()
    if pace not in PACES:
        raise ValueError("Выберите спокойный, средний или насыщенный темп.")

    def normalize_time(raw_value: object, fallback: str) -> tuple[str, int]:
        text = str(raw_value or fallback).strip()
        try:
            hour_text, minute_text = text.split(":", 1)
            hour = int(hour_text)
            minute = int(minute_text)
        except (TypeError, ValueError) as error:
            raise ValueError("Время должно быть в формате ЧЧ:ММ.") from error
        if (
            len(text) != 5
            or len(hour_text) != 2
            or len(minute_text) != 2
            or not 0 <= hour <= 23
            or not 0 <= minute <= 59
        ):
            raise ValueError("Время должно быть в формате ЧЧ:ММ.")
        minutes = hour * 60 + minute
        if not 6 * 60 <= minutes <= 23 * 60 + 30:
            raise ValueError("Время маршрута должно быть между 06:00 и 23:30.")
        return text, minutes

    start_time, start_minutes = normalize_time(
        getattr(value, "start_time", "09:00"),
        "09:00",
    )
    end_time, end_minutes = normalize_time(
        getattr(value, "end_time", "20:00"),
        "20:00",
    )
    if start_minutes >= end_minutes:
        raise ValueError("Окончание дня должно быть позже начала.")
    if end_minutes - start_minutes < 4 * 60:
        raise ValueError("Для маршрута нужно выделить не меньше 4 часов в день.")

    trip_goal = " ".join(
        str(getattr(value, "trip_goal", "") or "").split()
    ) or (
        "Познакомиться с территорией"
        if destination_kind == "region"
        else DEFAULT_TRIP_GOAL
    )
    if not 2 <= len(trip_goal) <= 160:
        raise ValueError("Цель поездки должна содержать от 2 до 160 символов.")

    raw_interests = getattr(value, "interest_categories", None)
    if not raw_interests:
        raw_interests = DEFAULT_INTERESTS
    if isinstance(raw_interests, (str, bytes)):
        raise ValueError("Выберите варианты отдыха из списка.")
    interests = []
    for raw_interest in raw_interests:
        interest = str(raw_interest).strip().casefold()
        if interest not in INTEREST_LABELS:
            raise ValueError("Выбран неизвестный вариант отдыха.")
        if interest not in interests:
            interests.append(interest)
    if not interests:
        interests = list(DEFAULT_INTERESTS)
    if len(interests) > len(INTEREST_LABELS):
        raise ValueError("Выбрано слишком много вариантов отдыха.")

    location_preference = " ".join(
        str(getattr(value, "location_preference", "anywhere") or "anywhere").split()
    ).casefold()
    if location_preference not in {"anywhere", "nearby"}:
        raise ValueError("Выберите допустимый способ планирования по району.")
    stay_area = " ".join(
        str(getattr(value, "stay_area", "") or "").split()
    )
    if len(stay_area) > 120:
        raise ValueError("Район проживания должен быть не длиннее 120 символов.")
    if location_preference == "nearby" and len(stay_area) < 2:
        raise ValueError("Укажите район проживания или ближайшую станцию.")
    if location_preference == "anywhere":
        stay_area = ""

    wishes = " ".join(str(value.wishes or "").split())
    if len(wishes) > 500:
        raise ValueError("Пожелания должны быть не длиннее 500 символов.")

    normalized = TripRequest(
        destination=destination,
        destination_kind=destination_kind,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        adults=adults,
        children_ages=children_ages,
        budget=budget,
        **fixed_amounts,
        currency=currency,
        pace=pace,
        start_time=start_time,
        end_time=end_time,
        trip_goal=trip_goal,
        interest_categories=interests,
        location_preference=location_preference,
        stay_area=stay_area,
        wishes=wishes,
    )
    if destination_kind == "city" and city_candidates_enabled() and city_deterministic_enabled():
        from city_preferences import location_preference_problems
        problems = location_preference_problems(normalized)
        if problems:
            raise TripCreateValidationError(" ".join(problems), code="unsupported_preferences")
    return normalized


def _memory_query(request: TripRequest) -> str:
    if request.children_ages:
        ages = ", ".join(str(age) for age in request.children_ages)
        family = f"{request.adults} взрослых, дети {ages} лет"
    else:
        family = f"{request.adults} взрослых, без детей"

    wishes = request.wishes or "нет дополнительных пожеланий"
    trip_goal = getattr(request, "trip_goal", "") or DEFAULT_TRIP_GOAL
    interest_ids = getattr(request, "interest_categories", None) or DEFAULT_INTERESTS
    interests = ", ".join(
        INTEREST_LABELS.get(str(item), str(item))
        for item in interest_ids
    )
    location_mode = (
        "Основную программу желательно строить рядом с жильём."
        if getattr(request, "location_preference", "anywhere") == "nearby"
        else "Район проживания не ограничивает программу."
    )
    return (
        "Какие сохранённые семейные предпочтения и результаты прошлых "
        "поездок полезны для новой поездки? "
        f"Направление: {request.destination}. Состав: {family}. "
        f"Темп: {request.pace}. Цель: {trip_goal}. "
        f"Интересы: {interests}. {location_mode} "
        f"Пожелания текущей анкеты: {wishes}."
    )


def _save_trip_memory_safely(
    tg_id: int,
    request: TripRequest,
    trip_id: int,
) -> int:
    """Сбой Pinecone или журнала памяти не отменяет готовую поездку."""

    try:
        result = vector_memory.save_trip_memory(
            tg_id,
            tg_id,
            tg_id,
            request,
            trip_id=trip_id,
        )
    except Exception as error:
        logger.warning(
            "Память новой поездки %s не сохранена: %s",
            trip_id,
            type(error).__name__,
        )
        return 0

    if not result.success:
        logger.warning(
            "Память новой поездки %s сохранена не полностью: "
            "requested=%s, vector=%s, journal=%s, error=%s",
            trip_id,
            result.requested,
            result.vector_saved,
            result.journal_saved,
            result.error_type or "unknown",
        )
        return 0

    return result.journal_saved


def _city_schedule_metadata_problems(itinerary: Itinerary, request: TripRequest) -> List[str]:
    """Validate new city metadata without changing legacy/source admission."""
    if (getattr(itinerary, "city_schedule", False) is False
            and getattr(getattr(itinerary, "program_report", None), "coverage_basis", None) != "map"):
        return []
    try:
        return limited_program_problems(
            itinerary, request, allow_city_limited_program=request.destination_kind == "city",
        )
    except (TypeError, ValueError):
        return ["Не удалось проверить сведения о городской программе и её датах."]


def _validated_trip(tg_id: int, trip_id: int) -> tuple[Dict[str, Any], TripRequest, Itinerary]:
    trip = db.get_trip(trip_id, tg_id)
    if trip is None:
        raise TripNotFoundError("Поездка не найдена.")

    try:
        request = TripRequest.model_validate(trip.get("request", {}))
        itinerary = Itinerary.model_validate(trip.get("itinerary", {}))
    except ValidationError as error:
        raise TripDataError(
            "Сохранённые данные поездки повреждены. Откройте её в Telegram-боте."
        ) from error

    if _city_schedule_metadata_problems(itinerary, request):
        raise TripDataError("Сведения о городской программе повреждены и требуют перепроверки.")

    return trip, request, itinerary


def _program_total(itinerary: Itinerary) -> float:
    return round(
        sum(float(item.cost) for day in itinerary.days for item in day.items),
        2,
    )


def _destination_scope_note(request: TripRequest) -> str:
    if getattr(request, "destination_kind", "city") != "region":
        return ""
    return (
        "Для озёр и водных территорий используется сам контур и ближайшая береговая зона до 5 км; "
        "для остальных регионов — допуск до 40 км. Это технический допуск, не официальная граница. "
        "Маршрут может включать несколько зон выбранной территории. Базы дней и переезды "
        "ориентировочные: время, сезонную доступность и стоимость нужно "
        "проверить отдельно — маршрут не подтверждает бронирование."
    )


def _budget_scope_note(
    request: TripRequest,
    itinerary: Itinerary,
) -> str:
    note = (
        "Плановые суммы, не фактические траты. "
        "Дорога и жильё — введённые суммы; цены программы ориентировочные."
    )
    missing = []
    if request.travel_cost == 0:
        missing.append("дорога")
    if request.lodging_cost == 0:
        missing.append("жильё")
    if missing:
        note += (
            " Не указаны и не учтены: " + ", ".join(missing)
            + ". Итог может быть неполным."
        )

    cost_warning = paid_interest_cost_warning(
        request,
        itinerary,
    )
    if cost_warning:
        note += " " + cost_warning

    unpriced_note = city_unpriced_visit_note(itinerary)
    if unpriced_note:
        note += " " + unpriced_note

    return note


def _trip_status(start_date: str, end_date: str) -> str:
    today = date.today().isoformat()
    if end_date < today:
        return "past"
    if start_date <= today <= end_date:
        return "active"
    return "upcoming"


def _checklist_progress(items: List[Dict[str, Any]]) -> Dict[str, int]:
    total = len(items)
    done = sum(1 for item in items if bool(item.get("done")))
    return {"done": done, "total": total}


def _summary_from_trip(trip: Dict[str, Any]) -> Dict[str, Any]:
    try:
        request = TripRequest.model_validate(trip.get("request", {}))
        itinerary = Itinerary.model_validate(trip.get("itinerary", {}))
    except ValidationError as error:
        raise TripDataError(
            f"Поездка №{trip.get('id', '?')} содержит повреждённые данные."
        ) from error

    checklist = db.get_checklist(int(trip["id"]))
    total = _program_total(itinerary)
    planned_total = round(request.fixed_costs + total, 2)
    return {
        "id": int(trip["id"]),
        "destination": request.destination,
        "destination_kind": request.destination_kind,
        "destination_scope_note": _destination_scope_note(request),
        "start_date": request.start_date,
        "end_date": request.end_date,
        "title": itinerary.title,
        "days_count": len(itinerary.days),
        "status": _trip_status(request.start_date, request.end_date),
        "currency": request.currency,
        "budget": request.budget,
        "travel_cost": request.travel_cost,
        "lodging_cost": request.lodging_cost,
        "fixed_costs": request.fixed_costs,
        "program_budget": request.program_budget,
        "program_total": total,
        "planned_total": planned_total,
        "remaining": round(request.budget - planned_total, 2),
        "checklist": _checklist_progress(checklist),
        "revision": int(trip.get("revision", 1)),
    }


def list_user_trips(tg_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    safe_limit = max(1, min(int(limit), 100))
    return [_summary_from_trip(trip) for trip in db.list_trips(tg_id, safe_limit)]


def get_trip_detail(tg_id: int, trip_id: int) -> Dict[str, Any]:
    trip, request, itinerary = _validated_trip(tg_id, trip_id)
    checklist = db.get_checklist(trip_id)
    statuses = db.get_reminder_statuses(trip_id, tg_id)
    total = _program_total(itinerary)
    planned_total = round(request.fixed_costs + total, 2)
    spent_through_day = 0.0
    budget_days = []
    for day in itinerary.days:
        day_total = round(
            sum(float(item.cost) for item in day.items),
            2,
        )
        spent_through_day = round(spent_through_day + day_total, 2)
        planned_through_day = round(request.fixed_costs + spent_through_day, 2)
        budget_days.append(
            {
                "day": day.day,
                "day_total": day_total,
                "program_spent_through_day": spent_through_day,
                "spent_through_day": planned_through_day,
                "planned_through_day": planned_through_day,
                "remaining_after_day": round(
                    request.budget - planned_through_day,
                    2,
                ),
            }
        )

    points = []
    public_itinerary = itinerary.model_dump()
    public_itinerary["program_report_text"] = program_report_text(itinerary)
    public_itinerary["unpriced_visit_note"] = city_unpriced_visit_note(itinerary)
    for day, public_day in zip(itinerary.days, public_itinerary["days"]):
        for item, public_item in zip(day.items, public_day["items"]):
            attribution = item_map_attribution(item)
            public_item.update(attribution)
            public_verified = is_public_map_point(item)
            public_item["verified"] = public_verified
            if public_verified:
                points.append(
                    {
                        "day": day.day,
                        "time": item.time,
                        "title": item.title,
                        "official_name": item.official_name,
                        "verified": True,
                        "lat": item.lat,
                        "lon": item.lon,
                        **attribution,
                    }
                )

    return {
        "id": int(trip["id"]),
        "destination": request.destination,
        "destination_kind": request.destination_kind,
        "destination_scope_note": _destination_scope_note(request),
        "start_date": request.start_date,
        "end_date": request.end_date,
        "created_at": trip.get("created_at"),
        "revision": int(trip.get("revision", 1)),
        "request": request.model_dump(),
        "edit_available": True,
        "edit_unavailable_reason": "",
        "edit_scope_note": city_context.CITY_EDIT_NOTE if itinerary.city_schedule else "",
        "weather_text": str(trip.get("request", {}).get("weather_text", "")),
        "itinerary": public_itinerary,
        "generation_quality": generation_quality.public_quality(
            trip.get("request", {}).get(generation_quality.KEY), itinerary),
        "budget": {
            "currency": request.currency,
            "limit": request.budget,
            "travel_cost": request.travel_cost,
            "lodging_cost": request.lodging_cost,
            "fixed_costs": request.fixed_costs,
            "program_budget": request.program_budget,
            "program_total": total,
            "planned_total": planned_total,
            "remaining": round(request.budget - planned_total, 2),
            "over_budget": planned_total > request.budget,
            "unpriced_visit_count": unpriced_city_visit_count(itinerary),
            "days": budget_days,
            "scope_note": _budget_scope_note(
                request,
                itinerary
            ),
        },
        "checklist": checklist,
        "checklist_progress": _checklist_progress(checklist),
        "map_points": points,
        "reminders_text": reminders.describe_plan(trip, statuses),
    }


def get_trip_pdf(tg_id: int, trip_id: int) -> bytes:
    trip, _, _ = _validated_trip(tg_id, trip_id)
    return build_trip_pdf(trip, db.get_checklist(trip_id))


def set_checklist_done(tg_id: int, item_id: int, done: bool) -> Dict[str, Any]:
    changed = db.set_checklist_item_done(item_id, tg_id, done)
    if not changed:
        raise TripNotFoundError("Пункт чек-листа не найден.")

    item = db.get_checklist_item(item_id, tg_id)
    if item is None:
        raise TripNotFoundError("Пункт чек-листа не найден.")
    return item


def _enqueue_trip_cover_safely(tg_id: int, account_key: str, trip_id: int) -> None:
    """Decorative work is optional and can never invalidate a committed route.

    Import locally so the bot and offline callers do not initialize a worker.
    Only a running web application's manager may accept automatic cover work.
    Existing-trip reads, edits and Telegram creation do not call this hook.
    """
    try:
        import trip_covers

        if (not trip_covers.enabled() or trip_covers.manager.executor is None
                or trip_covers.manager.closing):
            return
        trip_covers.manager.submit(tg_id, account_key, trip_id)
    except Exception:
        # No provider exception, prompt, identity or destination in diagnostics.
        logger.warning("optional_trip_cover_not_enqueued")


@access_guard.guard_user_operation
def create_trip(tg_id: int, request: TripRequest, *, expected_account_key: Optional[str] = None,
                generation_job_id: Optional[str] = None) -> Dict[str, Any]:
    """Создаёт новую поездку из веб-анкеты и возвращает её карточку."""

    if isinstance(tg_id, bool) or not isinstance(tg_id, int):
        raise ValueError("Некорректный идентификатор пользователя.")

    account_key = expected_account_key or db.get_account_key(tg_id)
    if not account_key:
        raise EditBusyError("Аккаунт удалён или удаление ещё выполняется. Войдите заново через бота.")

    try:
        normalized = _normalized_trip_request(request)
    except ValueError as error:
        if isinstance(error, TripCreateValidationError):
            raise
        raise TripCreateValidationError(str(error)) from error
    reminders.get_days_before()
    # Resolve before claiming an attempt or paid memory lookup. Never do HTTP
    # in the queue's SQLite transaction. This server-only context is reused.
    resolved_city = None
    if normalized.destination_kind == "city":
        try:
            resolved_city = resolve_city_for_planning(normalized.destination)
        except CityResolutionError as error:
            code = {"ambiguous": "city_ambiguous", "not_found": "city_not_found",
                    "context_mismatch": "city_not_found", "unavailable": "city_unavailable",
                    "invalid_response": "city_unavailable"}.get(error.reason, "city_invalid")
            raise TripCreateValidationError(
                f"{error} Новая поездка не сохранена. Создание маршрута не запускалось; попытка не списана.",
                code=code, destination_options=error.suggestions,
            ) from error
    claim = db.claim_ai_request(
        tg_id,
        datetime.now(reminders.get_tz()).date(),
        "plan",
        "web",
        daily_limit=_daily_ai_limit(),
        expected_account_key=account_key,
    )
    if claim.status == "busy":
        raise EditBusyError("Другая AI-операция уже выполняется.")
    if claim.status == "limit":
        raise AIDailyLimitError(claim.limit)
    if claim.claim_id is None:
        raise RuntimeError("Не удалось зарегистрировать AI-запрос.")

    ai_completed = False
    try:
        matches = vector_memory.recall_family_memories(
            tg_id,
            tg_id,
            _memory_query(normalized),
            top_k=8,
            min_score=0.35,
            destination=normalized.destination,
        )
        memory_context = vector_memory.format_memory_context(matches)
        logger.info(
            "Для нового веб-маршрута пользователя %s найдено записей памяти: %s",
            tg_id,
            len(matches),
        )

        try:
            result = generate_itinerary(
                normalized,
                memory_context=memory_context,
                ground_candidates=normalized.destination_kind == "region",
                **({"resolved_city": resolved_city} if resolved_city is not None else {}),
                **({"city_candidates_first": True}
                   if normalized.destination_kind == "city" and city_candidates_enabled() else {}),
            )
        except CityResolutionError as error:
            raise TripCreateValidationError(
                f"{error} Новая поездка не сохранена."
            ) from error
        except DestinationScopeError as error:
            raise TripCreateValidationError(str(error)) from error
        if result.problems:
            raise TripCreateValidationError(
                generation_failure_message(result.problems),
                diagnostic_problems=result.problems,
                code=getattr(result, "generation_status", "validation"),
            )
        metadata_problems = _city_schedule_metadata_problems(result.itinerary, normalized)
        if metadata_problems:
            raise TripCreateValidationError(
                generation_failure_message(metadata_problems),
                diagnostic_problems=metadata_problems,
            )
        profile = db.get_user(tg_id) or {}
        request_dict = normalized.model_dump()
        request_dict["weather_text"] = result.weather_text
        request_dict["weather"] = result.weather
        request_dict[generation_quality.KEY] = generation_quality.safe_coverage(
            getattr(result, "generation_coverage", None))
        if getattr(result.itinerary, "city_schedule", False) is True:
            context = getattr(result, "city_map_context", None)
            try:
                candidates, _ = city_context.load_city_context(context, normalized)
                city_context.check_context_places(result.itinerary, candidates)
            except ValueError as error:
                raise TripCreateValidationError(
                    "Не удалось сохранить проверенный набор карты. Новая поездка не сохранена."
                ) from error
            request_dict[city_context.CONTEXT_KEY] = context
        checklist_items = build_checklist(
            normalized.adults,
            normalized.children_ages,
            result.weather,
        )
        trip_id = db.save_trip_bundle(
            tg_id,
            request_dict,
            result.itinerary.model_dump(),
            checklist_items,
            name=str(profile.get("name") or ""),
            adults=normalized.adults,
            children_ages=normalized.children_ages,
            currency=normalized.currency,
            pace=normalized.pace,
            wishes=normalized.wishes,
            **({"generation_job_id": generation_job_id, "expected_account_key": account_key}
               if generation_job_id is not None else {}),
        )
        ai_completed = True
        _enqueue_trip_cover_safely(tg_id, account_key, trip_id)
        try:
            detail = get_trip_detail(tg_id, trip_id)
            remembered_count = _save_trip_memory_safely(
                tg_id,
                normalized,
                trip_id,
            )

            detail["generation"] = {
                "attempts": int(result.attempts),
                "prompt_tokens": int(result.prompt_tokens),
                "completion_tokens": int(result.completion_tokens),
                "total_tokens": int(result.total_tokens),
                "memory_matches": len(matches),
                "memory_saved": remembered_count,
            }
            return detail
        except Exception as error:
            raise TripSavedResponseError(
                trip_id,
                f"Маршрут №{trip_id} создан и сохранён, но не удалось показать его в кабинете. "
                "Обновите список поездок и откройте маршрут. Не запускайте создание повторно.",
            ) from error
    finally:
        _finish_ai_request_safely(
            claim.claim_id,
            succeeded=ai_completed,
        )


@access_guard.guard_user_operation
def edit_trip_day(
    tg_id: int,
    trip_id: int,
    target_day: int,
    instruction: str,
) -> Dict[str, Any]:
    """Меняет ровно один день и сохраняет только полностью проверенный результат."""

    trip, request, current_itinerary = _validated_trip(tg_id, trip_id)
    instruction = " ".join((instruction or "").split())
    if len(instruction) < 3:
        raise ValueError("Пожелание к изменению слишком короткое.")
    if len(instruction) > 600:
        raise ValueError("Пожелание должно быть не длиннее 600 символов.")

    day_numbers = {day.day for day in current_itinerary.days}
    if target_day not in day_numbers:
        raise ValueError("В этой поездке нет выбранного дня.")

    # Exact map binding is checked before paid quota. Legacy city plans recover
    # only their existing independently rechecked visits, not guessed IDs.
    city_map_context = None
    if getattr(current_itinerary, "city_schedule", False) is True:
        city_map_context = city_context.prepare_stored_city_context(trip["request"], request, current_itinerary)

    claim = db.claim_ai_request(
        tg_id,
        datetime.now(reminders.get_tz()).date(),
        "edit",
        "web",
        trip_id=trip_id,
        daily_limit=_daily_ai_limit(),
    )
    if claim.status == "busy":
        raise EditBusyError("Другая AI-операция уже выполняется.")
    if claim.status == "limit":
        raise AIDailyLimitError(claim.limit)
    if claim.claim_id is None:
        raise RuntimeError("Не удалось зарегистрировать AI-запрос.")

    ai_completed = False
    try:
        try:
            result = edit_itinerary(
                request,
                current_itinerary,
                target_day,
                instruction,
                **({"city_map_context": deepcopy(city_map_context),
                    "prior_weather": trip["request"].get("weather")}
                   if city_map_context is not None else {}),
            )
        except CityResolutionError as error:
            raise ValueError(
                f"{error} Изменения не сохранены. Старый маршрут сохранён без изменений."
            ) from error
        if result.problems:
            raise EditRejectedError(result.problems)

        if getattr(current_itinerary, "city_schedule", False) is True:
            if (getattr(result.itinerary, "city_schedule", False) is not True
                    or getattr(result.itinerary, "source_schedule", False) is not False
                    or _city_schedule_metadata_problems(result.itinerary, request)):
                raise EditRejectedError(["Нельзя изменить тип или проверенный календарь городского маршрута."])

        request_dict = dict(trip["request"])
        request_dict["weather_text"] = result.weather_text
        request_dict["weather"] = result.weather

        # An edit must not retain stale completeness from the previous plan.
        request_dict[generation_quality.KEY] = generation_quality.safe_coverage(
            getattr(result, "generation_coverage", None))

        if getattr(current_itinerary, "city_schedule", False) is True:
            try:
                request_dict[city_context.CONTEXT_KEY] = city_context.validate_city_edit_context(
                    city_map_context, getattr(result, "city_map_context", None), request, result.itinerary)
                from city_planner import evaluate_saved_city_quality
                updated_candidates, _ = city_context.load_city_context(request_dict[city_context.CONTEXT_KEY], request)
                request_dict[generation_quality.KEY] = generation_quality.safe_coverage(evaluate_saved_city_quality(
                    request, updated_candidates, result.itinerary,
                    trip["request"].get(generation_quality.KEY)))
            except ValueError as error:
                raise EditRejectedError([city_context.CONTEXT_ERROR]) from error

        saved = db.update_trip_plan(
            trip_id,
            tg_id,
            request_dict,
            result.itinerary.model_dump(),
            expected_revision=int(trip.get("revision", 1)),
        )
        if not saved:
            if db.get_trip(trip_id, tg_id) is None:
                raise TripNotFoundError("Поездка не найдена.")
            raise EditConflictError(
                "Маршрут уже изменился в другом окне. Обновите страницу и повторите запрос."
            )

        ai_completed = True
        try:
            detail = get_trip_detail(tg_id, trip_id)
            detail["edit"] = {
                "day": target_day,
                "attempts": result.attempts,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            }
            return detail
        except Exception as error:
            raise TripSavedResponseError(
                trip_id,
                f"Изменения дня {target_day} поездки №{trip_id} сохранены, "
                "но не удалось показать обновлённый маршрут. Откройте поездку заново "
                "перед повторным изменением.",
            ) from error
    finally:
        _finish_ai_request_safely(
            claim.claim_id,
            succeeded=ai_completed,
        )
