"""Neutral server-owned wording for the opt-in map-only city plan.

This is not a fact check of attraction descriptions or a repair of identity,
prices, time, duration, indoor suitability, map evidence or invalid items.
The caller keeps original model responses as diagnostics and still runs every
ordinary validation after this narrow presentation step. No I/O occurs here.
"""

import math

from schemas import Itinerary, TripRequest


UNPRICED_VISIT_LABEL = "Цена не уточнена"


def is_unpriced_city_visit(itinerary: Itinerary, item) -> bool:
    """Presentation only: a zero map-only reserve is not a verified free ticket."""
    return (
        getattr(itinerary, "city_schedule", False) is True
        and getattr(itinerary, "source_schedule", False) is not True
        and getattr(item, "kind", "") == "attraction"
        and getattr(item, "cost", None) == 0
    )


def unpriced_city_visit_count(itinerary: Itinerary) -> int:
    return sum(is_unpriced_city_visit(itinerary, item) for day in itinerary.days for item in day.items)


def city_unpriced_visit_note(itinerary: Itinerary) -> str:
    """Disclose unknown admission without changing costs or claiming a price."""
    count = unpriced_city_visit_count(itinerary)
    if not count:
        return ""
    return (
        f"Посещений с неуточнённой ценой: {count}. Итог предварительный и не включает "
        "неизвестную стоимость входа; нулевой резерв не означает бесплатный вход."
    )


def budget_coverage(request, itinerary: Itinerary) -> dict:
    """Describe known omissions, never repair prices or certify a complete budget.

    Accept saved legacy request dictionaries as well as validated requests. Only
    the explicit map-only city mode supports the narrower meal/transfer claims.
    No saved values are changed and no provider is queried.
    """
    data = request.model_dump() if isinstance(request, TripRequest) else request
    data = data if isinstance(data, dict) else {}
    missing, notes = [], []

    def positive(value):
        try:
            return not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0
        except (TypeError, ValueError, OverflowError):
            return False

    for key, code, label in (("travel_cost", "travel", "Дорога туда и обратно"),
                             ("lodging_cost", "lodging", "Жильё")):
        if not positive(data.get(key)):
            missing.append({"code": code, "message": f"{label}: сумма не указана и не включена в итог."})
    unpriced = city_unpriced_visit_note(itinerary)
    if unpriced:
        missing.append({"code": "admission", "message": unpriced})
    if itinerary.city_schedule is True and itinerary.source_schedule is not True:
        program_days = [day for day in itinerary.days if any(item.kind == "attraction" for item in day.items)]
        no_transport = sum(not any(item.kind == "transport" and positive(item.cost) for item in day.items)
                           for day in program_days)
        if no_transport:
            missing.append({"code": "local_transport", "message":
                f"Денежный резерв на местный транспорт отдельно не рассчитан в {no_transport} днях программы. "
                "Временной запас на переходы и переезды не является оплатой проезда; пешие переходы могут быть бесплатными."})
        lunch_only = sum(
            bool(meals := [item for item in day.items if item.kind == "meal"])
            and all(item.title.strip().casefold() in {"обед", "lunch"} for item in meals)
            for day in itinerary.days
        )
        if lunch_only:
            missing.append({"code": "other_meals", "message":
                f"В {lunch_only} днях на питание предусмотрен только обед. Завтраки, ужины и перекусы отдельно не рассчитаны."})
        if any(item.kind == "meal" for day in itinerary.days for item in day.items):
            notes.append("Питание — плановый резерв на всех участников, а не подтверждённая цена конкретного заведения.")
        raw_interests = data.get("interest_categories")
        interests = {value for value in raw_interests if isinstance(value, str)} if isinstance(raw_interests, (list, tuple, set)) else set()
        if "shopping_malls" in interests:
            notes.append("Покупки в торговых центрах и личные расходы в смету не включены.")
        if {"theatre", "cinema"}.intersection(interests):
            notes.append("Точки театров и кинотеатров на карте не подтверждают спектакль или сеанс: афишу, время и билеты нужно проверить отдельно.")
    return {"status": "incomplete" if missing else "estimate", "missing": missing, "notes": notes}


MEAL_ESTIMATE_DESCRIPTION = (
    "Плановый резерв на обед; конкретное заведение, меню и фактическая цена не подтверждены."
)
REST_ESTIMATE_DESCRIPTION = (
    "Пауза для отдыха без привязки к конкретному месту; условия для отдыха не подтверждены."
)
CHILD_VISIT_NOTE = (
    "Заранее уточните возрастные ограничения и условия посещения; предусмотрите перерыв по потребностям ребёнка."
)
CHILD_MEAL_NOTE = "Заранее уточните меню и условия питания для ребёнка."
CHILD_REST_NOTE = "Подберите место для спокойного перерыва с учётом потребностей ребёнка."
CHILD_CONDITIONS_WARNING = (
    "Возрастные ограничения, детское меню, интерактивные занятия и другие условия для детей не проверены; "
    "уточните их у площадки заранее."
)


def label_city_estimates(itinerary: Itinerary, request: TripRequest) -> None:
    """Label estimates without making any itinerary pass its substantive guards.

    Adult-only nonempty kids_note is deliberately retained for rejection by
    the existing wording guard, including meal/rest notes. Unknown item kinds
    and all identity/map fields are also untouched, never cleaned into validity.
    """
    if request.destination_kind != "city" or itinerary.source_schedule is True:
        raise ValueError("Нейтральные оценки предназначены только для городского плана по карте.")
    children = bool(request.children_ages)
    for day in itinerary.days:
        for item in day.items:
            if item.kind == "meal":
                item.description = MEAL_ESTIMATE_DESCRIPTION
                item.indoor = False
                if children:
                    item.kids_note = CHILD_MEAL_NOTE
            elif item.kind == "rest":
                item.description = REST_ESTIMATE_DESCRIPTION
                item.indoor = False
                if children:
                    item.kids_note = CHILD_REST_NOTE
            elif item.kind == "attraction" and children:
                item.kids_note = CHILD_VISIT_NOTE
    if children and CHILD_CONDITIONS_WARNING not in itinerary.tips:
        itinerary.tips.append(CHILD_CONDITIONS_WARNING)
