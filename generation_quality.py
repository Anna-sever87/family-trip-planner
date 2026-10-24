"""Bounded server-owned quality metadata, separate from technical completion.

No raw prompts, titles, memory, provider errors, IDs or credentials are stored
here. Older trips retain their itinerary and get only observable day counts.
"""

from schemas import INTEREST_LABELS

KEY = "generation_coverage"
STATUSES = {"complete", "partial", "insufficient_data"}
REASONS = {
    "incomplete_calendar": "Не все даты заполнены программой.",
    "insufficient_confirmed_places": "Найденного подтверждённого набора недостаточно для всех дней; это не означает, что в городе мало мест.",
    "indoor_evidence_missing": "Для части погодных условий не хватает подтверждённых сведений о помещениях.",
    "visit_constraints": "Часть посещений ограничена датами, временем или условиями поездки.",
    "budget_constraint": "На подбор программы повлиял доступный бюджет.",
    "search_limit": "Достигнут установленный предел поиска; полнота подбора не гарантируется.",
    "schedule_constraints": "Не удалось совместить все посещения в допустимое расписание.",
    "requested_interest_missing": "Не все выбранные интересы удалось включить.",
    "day_program_light": "На части дней запланирован сокращённый объём посещений; остальное время остаётся свободным.",
    "day_window_unbalanced": "На части дней посещения сосредоточены в одной части заданного окна; оставшееся время не заполнено программой.",
    "selection_depth_shortfall": "После отбора осталось недостаточно подходящих посещений для всех дней; исходный поиск нашёл больше мест.",
    "selection_repair_failed": "Дополнительная проверка отбора не завершилась; сохранён ранее проверенный результат без автоматического повтора.",
}
_COUNTS = {
    "requested_days", "program_days", "free_days", "eligible_candidates", "input_candidates",
    "selected_unique_visits", "eligible_unique_visits", "states_explored", "available_day_templates",
    "group_search_limits", "enrichment_visits", "confirmed_candidates", "rejected_count",
    "warning_count", "map_requests", "attempted_names", "name_limit", "usable_limit", "pool_target",
    "planning_minutes", "required_planning_minutes",
    "selection_calls", "preference_planning_minutes", "pool_planning_minutes", "discovered_candidates", "preference_candidates",
    "source_alternatives_confirmed",
    "desired_program_minutes", "light_program_days", "unbalanced_program_days",
}
_BOOLS = {"search_complete", "name_limit_reached", "map_limit_reached", "usable_limit_reached",
          "replenishment_attempted", "selection_repair_attempted", "selection_depth_shortfall"}
_REJECTIONS = {
    "generic_name_or_non_visit", "limited_search_unconfirmed", "map_request_limit",
    "map_unavailable_or_invalid_response", "nonlocal_cyrillic_proposal", "shared_map_clock_unavailable",
    "unsupported_leading_punctuation", "canonical_record_mismatch", "country_mismatch_or_missing",
    "invalid_coordinates", "invalid_record", "missing_locality", "missing_provider_identity",
    "name_mismatch", "name_type_conflict", "outside_city_radius", "unsupported_poi_type", "category_mismatch", "other",
    "amusement_interest_not_requested", "specialized_animal_reserve_limit",
    "duplicate_canonical_query",
    "city_only_membership_unconfirmed",
}


def _count(value):
    return type(value) is int and 0 <= value <= 100000


def safe_coverage(raw):
    """Allowlist each field; never serialize arbitrary nested diagnostics."""
    if not isinstance(raw, dict):
        return {}
    clean = {key: raw[key] for key in _COUNTS if _count(raw.get(key))}
    clean.update({key: raw[key] for key in _BOOLS if type(raw.get(key)) is bool})
    if isinstance(raw.get("quality_status"), str) and raw["quality_status"] in STATUSES:
        clean["quality_status"] = raw["quality_status"]
    for key in ("quality_reasons", "reason_codes"):
        if isinstance(raw.get(key), (list, tuple)):
            clean[key] = sorted({v for v in raw[key][:100] if isinstance(v, str) and v in REASONS})
    for key in ("requested_categories", "missing_categories"):
        if isinstance(raw.get(key), (list, tuple)):
            clean[key] = sorted({v for v in raw[key][:100] if isinstance(v, str) and v in INTEREST_LABELS})
    for key, allowed in (("category_counts", INTEREST_LABELS), ("rejection_reasons", _REJECTIONS),
                         ("detail_rejection_reasons", _REJECTIONS)):
        if isinstance(raw.get(key), dict):
            clean[key] = {k: v for k, v in raw[key].items() if k in allowed and _count(v)}
    for key, allowed in (("pool_quality", {"adequate", "limited", "empty"}),
                         ("map_discovery_status", {"success", "empty", "unavailable", "not_requested"}),
                         ("category_fallback_status", {"success", "empty", "unavailable", "not_requested"}),
                         ("map_discovery_error", {"none", "http_error", "timeout", "invalid_response",
                                                  "oversized_response", "incomplete_response"}),
                         ("selection_repair_status", {"not_needed", "not_attempted", "improved", "unchanged", "failed"}),
                         ("selection_mode", {"structured_preferences", "semantic_preferences", "memory_ranking"}),
                         ("memory_ranking_status", {"applied", "unavailable"})):
        if isinstance(raw.get(key), str) and raw[key] in allowed:
            clean[key] = raw[key]
    interests = raw.get("interest_coverage")
    if isinstance(interests, dict):
        clean["interest_coverage"] = {}
        for key in INTEREST_LABELS:
            item = interests.get(key)
            if (isinstance(item, dict) and isinstance(item.get("status"), str)
                    and item["status"] in {"covered", "not_confirmed", "not_scheduled", "not_verifiable"}
                    and _count(item.get("confirmed_count")) and _count(item.get("selected_count"))
                    and item["selected_count"] <= item["confirmed_count"]):
                clean["interest_coverage"][key] = {k: item[k] for k in ("status", "confirmed_count", "selected_count")}
    if isinstance(raw.get("discovery"), dict):
        # No recursion: reject nested envelopes inside discovery.
        clean["discovery"] = safe_coverage({k: v for k, v in raw["discovery"].items() if k != "discovery"})
    return clean


def memory_note(stored):
    quality = safe_coverage(stored)
    if quality.get("selection_mode") != "memory_ranking":
        return ""
    return (
        "Память прошлых поездок используется для приоритетов. Обязательные ограничения укажите в текущей анкете."
        if quality.get("memory_ranking_status") == "applied" else
        "Не удалось применить память прошлых поездок к приоритетам; использована текущая анкета. "
        "Обязательные ограничения укажите в ней.")


def public_quality(stored, itinerary):
    """Observable calendar always overrides stale saved totals after editing."""
    quality = safe_coverage(stored)
    days = itinerary.days
    if not quality and not getattr(itinerary, "city_schedule", False):
        return {}
    free = sum(day.day_mode == "free" for day in days)
    quality.update(requested_days=len(days), program_days=len(days) - free, free_days=free)
    if free:
        quality["quality_status"] = "partial" if free < len(days) else "insufficient_data"
        quality["quality_reasons"] = sorted(set(quality.get("quality_reasons", [])) | {"incomplete_calendar"})
    # Do not invent a successful category audit for old trips.
    title = {"complete": "Программа составлена на все дни",
             "partial": "Программа составлена частично",
             "insufficient_data": "Недостаточно подтверждённых данных"}.get(quality.get("quality_status"), "Полнота программы")
    quality["title"] = title
    if note := memory_note(quality):
        quality["memory_note"] = note
    quality["reason_texts"] = [REASONS[key] for key in quality.get("quality_reasons", [])]
    status_text = {"covered": "включено", "not_confirmed": "не удалось подтвердить места",
                   "not_scheduled": "найдено, но не включено в расписание",
                   "not_verifiable": "соответствие этому интересу картой не подтверждается"}
    quality["interest_texts"] = [
        f"{INTEREST_LABELS[key]}: {status_text[item['status']]}. Найдено: {item['confirmed_count']}, включено: {item['selected_count']}."
        for key, item in quality.get("interest_coverage", {}).items()
        if item["status"] != "not_verifiable"
    ]
    return quality
