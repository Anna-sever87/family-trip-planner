"""Оформление готового маршрута в текст для Telegram."""
from typing import List, Optional

from schemas import Itinerary
from verified_map_places import is_public_map_point
from program_coverage import FREE_DAY_NOTE, program_report_text
from city_estimates import UNPRICED_VISIT_LABEL, city_unpriced_visit_note, is_unpriced_city_visit
from place_labels import build_place_labels, display_place_title

KIND_ICONS = {"attraction": "🏛", "meal": "🍽", "rest": "😴"}


def format_itinerary(
    itinerary: Itinerary,
    currency: str,
    budget: float,
    weather_text: str = "",
    children_ages: Optional[List[int]] = None,
    travel_cost: float = 0,
    lodging_cost: float = 0,
    saved_request: Optional[dict] = None,
) -> str:
    """Собирает читаемый текст маршрута без разметки (чтобы Telegram не ругался на спецсимволы)."""
    # The display helper also accepts legacy attribute-based itinerary objects;
    # serialization is not needed to render their original titles safely.
    labels = build_place_labels({"request": saved_request or {}, "itinerary": itinerary})
    lines: List[str] = [f"✈️ {itinerary.title}", "", itinerary.summary]
    coverage = program_report_text(itinerary)
    if coverage:
        lines += ["", "Программа и свободные дни", coverage]
    if weather_text:
        lines += ["", f"🌦 {weather_text}"]

    total = 0.0
    for day in itinerary.days:
        lines += ["", f"📅 День {day.day} ({day.date}): {day.theme}"]
        if getattr(day, "day_mode", "program") == "free":
            lines += ["Свободный режим", FREE_DAY_NOTE]
        base_area = str(
            getattr(
                day,
                "base_area",
                "",
            )
            or ""
        ).strip()
        transfer_note = str(
            getattr(
                day,
                "transfer_note",
                "",
            )
            or ""
        ).strip()
        if base_area:
            lines.append(
                f"📍 База дня: {base_area}"
            )
        if transfer_note:
            lines.append(
                f"🚗 Переезд: {transfer_note}"
            )
        for item in day.items:
            total += item.cost
            icon = KIND_ICONS.get(item.kind, "•")
            cost = f", ~{item.cost:g} {currency}" if item.cost else ""
            if is_unpriced_city_visit(itinerary, item):
                cost = f", {UNPRICED_VISIT_LABEL}"
            place = f" ({item.area})" if item.area else ""
            lines.append(f"{item.time} {icon} {display_place_title(item, labels)}{place}{cost}")
            if item.description:
                lines.append(f"   {item.description}")
            if item.kids_note and (children_ages is None or children_ages):
                lines.append(f"   👶 {item.kids_note}")
            if item.kind == "attraction" and not is_public_map_point(item):
                lines.append("   ⚠️ место не подтверждено на карте, проверьте перед поездкой")
        if day.rainy_alternative:
            lines.append(f"☔ Если дождь: {day.rainy_alternative}")

    if itinerary.tips:
        lines += ["", "💡 Советы:"] + [f"• {tip}" for tip in itinerary.tips]
    total = round(total, 2)
    planned_total = round(total + travel_cost + lodging_cost, 2)
    remaining = round(budget - planned_total, 2)
    lines += [
        "",
        f"💰 Общий бюджет: {budget:g} {currency}",
        f"Дорога туда и обратно на всех: {travel_cost:g} {currency}" if travel_cost else "Дорога: не указана и пока не включена в итог.",
        f"Жильё за все ночи: {lodging_cost:g} {currency}" if lodging_cost else "Жильё: не указано и пока не включено в итог.",
        f"Расходы на программу: около {total:g} {currency}",
        f"Итого запланировано: около {planned_total:g} {currency}",
        f"Остаток бюджета: {remaining:g} {currency}" if remaining >= 0 else f"Превышение бюджета: {-remaining:g} {currency}",
    ]
    unpriced_note = city_unpriced_visit_note(itinerary)
    if unpriced_note:
        lines.append(unpriced_note)
    return "\n".join(lines)


def split_text(text: str, limit: int = 4000) -> List[str]:
    """Делит длинный текст на части для Telegram (лимит сообщения 4096 символов), стараясь резать по строкам."""
    parts: List[str] = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        cut = cut if cut > 0 else limit
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts
