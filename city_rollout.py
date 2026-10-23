"""Explicit, off-by-default city MVP switch; no dotenv or service startup."""

import os


CITY_MVP_DISCLOSURE = (
    "Предварительная городская программа: часть дат может остаться свободной. "
    "Карта подтверждает название и местоположение, но не условия посещения, цены или наличие помещения. "
    "Проезд к местам и транспортные расходы отдельно не рассчитаны. "
    "Изменить один день можно в пределах ранее проверенного набора мест; остальные дни сохраняются."
)


def city_candidates_enabled() -> bool:
    """Only literal 'true' or '1' enable the mode; all other values fail off."""
    return os.getenv("CITY_CANDIDATES_FIRST", "0") in {"true", "1"}


def city_deterministic_enabled() -> bool:
    """Separate rollout switch; never silently fall back to an AI timetable."""
    return os.getenv("CITY_DETERMINISTIC_SCHEDULE", "0") in {"true", "1"}
