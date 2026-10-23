"""Neutral server-owned wording for the opt-in map-only city plan.

This is not a fact check of attraction descriptions or a repair of identity,
prices, time, duration, indoor suitability, map evidence or invalid items.
The caller keeps original model responses as diagnostics and still runs every
ordinary validation after this narrow presentation step. No I/O occurs here.
"""

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
