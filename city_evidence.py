"""Server-owned presentation for selected city visits, not new visitor facts.

Only exact regenerated values or a separately persisted server manifest may
authorize replacing these strings during an edit. A marker/prefix is never
authority. Hashes track presentation ownership; they do not verify evidence.
"""

from copy import deepcopy
import hashlib

from city_estimates import city_unpriced_visit_note
from city_visit_options import visit_option


UNKNOWN_INDOOR_NOTE = (
    "Для части посещений наличие помещения неизвестно. Они не использованы как крытые посещения "
    "для дождевого правила; это не утверждение, что объекты находятся на улице."
)
UNKNOWN_EDITED_INDOOR_NOTE = (
    "Для части посещений помещение отдельно не подтверждено. Отметка помещения в изменённой программе "
    "остаётся оценкой, а не фактом карты или источника; уточните условия перед визитом."
)
_LABELS = {"duration": "длительность", "indoor": "помещение выбранной экспозиции", "hours": "график",
           "price": "цена", "children": "посещение с детьми", "minimum_age": "возраст"}


def tip_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _visit_tip(candidate, option, item, day_date, request, *, legacy=False):
    basis = "опубликованный ориентир" if option.basis == "published" else "плановая оценка, не факт источника"
    prefix = "" if legacy else f"Сведения о посещении [{candidate.candidate_id}]. "
    if item.duration_min == option.duration_min or legacy:
        parts = [prefix + f"«{candidate.title}»: {item.duration_min} мин. — {basis}."]
    else:
        parts = [prefix + f"«{candidate.title}»: {item.duration_min} мин. — оценка изменённого расписания. "
                 f"Исходный ориентир: {option.duration_min} мин. — {basis}."]
    for source in option.sources:
        field = source.get("field")
        if field not in _LABELS or not request.children_ages and field in {"children", "minimum_age"}:
            continue
        source_basis = "плановая оценка" if source["basis"] == "planning_estimate" else "опубликованные сведения"
        parts.append(f"{_LABELS[field]} ({source_basis}; проверено {source['checked_at']}): {source['source_url']}.")
    if option.sources and not legacy:
        parts.append("Область сведений источников: " + option.description)
        if item.description != option.description:
            parts.append("Изменённое описание программы отдельно по этим источникам не подтверждалось.")
    windows = sorted({(value["start"], value["end"]) for value in option.opening_windows
                      if value["date"] == day_date})
    if windows:
        parts.append("Учтённые окна на " + day_date + ": "
                     + ", ".join(f"{left}–{right}" for left, right in windows) + ".")
    unknown = [_LABELS[value] for value in option.unknown_fields if value in _LABELS
               and (request.children_ages or value not in {"children", "minimum_age"})]
    if unknown:
        parts.append("Не подтверждены: " + ", ".join(unknown) + ".")
    parts.extend(option.restrictions)
    return " ".join(parts)


def city_managed_tips(itinerary, candidates, request, *, options_by_id=None, legacy=False) -> list[str]:
    """Render selected visits only; do not modify descriptions, maps or timing.

    ``options_by_id`` is for the server constructor's already validated
    options. Editing re-resolves conditions from the current server module.
    Legacy rendering is only an exact-recognition migration, never a prefix
    match or a way for model text to assert source/map authority.
    """
    by_query = {candidate.place_query: candidate for candidate in candidates}
    rendered, unknown_indoor, estimated_indoor = [], False, False
    for day in itinerary.days:
        for item in day.items:
            if item.kind != "attraction":
                continue
            candidate = by_query.get(item.place_query)
            if candidate is None:
                continue  # The ordinary identity validator rejects this item.
            option = (options_by_id.get(candidate.candidate_id) if options_by_id is not None
                      else visit_option(candidate, request))
            if option is None:
                continue  # Missing conditions are never invented for display.
            rendered.append(_visit_tip(candidate, option, item, day.date, request, legacy=legacy))
            unknown_indoor = unknown_indoor or option.indoor is None
            estimated_indoor = estimated_indoor or option.indoor is None and item.indoor is True
    if unknown_indoor:
        rendered.append(UNKNOWN_EDITED_INDOOR_NOTE if estimated_indoor and not legacy else UNKNOWN_INDOOR_NOTE)
    unpriced = city_unpriced_visit_note(itinerary)
    if unpriced:
        rendered.append(unpriced)
    return rendered


def recognize_city_managed_tips(itinerary, candidates, request) -> tuple[str, ...]:
    """Conservative fallback for old contexts without a server manifest."""
    expected = set(city_managed_tips(itinerary, candidates, request))
    expected.update(city_managed_tips(itinerary, candidates, request, legacy=True))
    return tuple(value for value in itinerary.tips if value in expected)


def replace_city_managed_tips(original, candidate, managed_values, candidates, request) -> list[str]:
    """Keep every other original tip verbatim and in its original order."""
    if not managed_values:
        return deepcopy(original.tips)
    managed = set(managed_values)
    replacement = city_managed_tips(candidate, candidates, request)
    result, inserted = [], False
    for value in original.tips:
        if value not in managed:
            result.append(value)
        elif not inserted:
            result.extend(replacement)
            inserted = True
    return result
