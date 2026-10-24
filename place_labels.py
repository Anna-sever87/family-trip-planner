"""Read-only Russian explanations for original map place names.

Display text is not a place identity and must never be saved over ``title``,
``official_name`` or ``place_query``. No translation service, geocoder, cache,
database or recovery is consulted. A saved city snapshot is checked once per
trip; without it we describe the language, not an unproved kind of place.
"""

from collections.abc import Mapping
import unicodedata


_GENERIC_EXPLANATION = "название на языке источника"
_RUSSIAN_LETTERS = frozenset("АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя")
# A type is a map observation, not proof of tickets, opening hours, indoor
# access, popularity, a scheduled show or suitability for a particular family.
_TYPE_EXPLANATIONS = {
    "museum": ("музей", {"tourism"}),
    "gallery": ("галерея", {"tourism"}),
    "zoo": ("зоопарк", {"tourism"}),
    "aquarium": ("аквариум", {"tourism"}),
    "theme_park": ("парк аттракционов", {"tourism"}),
    "attraction": ("достопримечательность", {"tourism"}),
    "viewpoint": ("смотровая площадка", {"tourism"}),
    "artwork": ("арт-объект", {"tourism", "historic"}),
    "park": ("парк", {"leisure"}),
    "garden": ("сад", {"leisure"}),
    "nature_reserve": ("природная территория", {"leisure"}),
    "beach": ("пляж", {"natural"}),
    "peak": ("горная вершина", {"natural"}),
    "volcano": ("вулкан", {"natural"}),
    "cave_entrance": ("вход в пещеру", {"natural"}),
    "waterfall": ("водопад", {"waterway", "natural"}),
    "spring": ("родник", {"natural"}),
    "cliff": ("скальный обрыв", {"natural"}),
    "cape": ("мыс", {"natural"}),
    "ridge": ("горный хребет", {"natural"}),
    "rock": ("скала", {"natural"}),
    "stone": ("камень", {"natural"}),
    "lake": ("озеро", {"natural", "water"}),
    "memorial": ("мемориал", {"historic"}),
    "monument": ("памятник", {"historic"}),
    "archaeological_site": ("археологический объект", {"historic"}),
    "castle": ("замок", {"historic"}),
    "ruins": ("руины", {"historic"}),
    "manor": ("усадьба", {"historic"}),
    "fort": ("форт", {"historic"}),
    "palace": ("дворец", {"historic"}),
    "monastery": ("монастырь", {"historic", "amenity", "building"}),
    "church": ("церковь", {"historic", "amenity", "building"}),
    "cathedral": ("собор", {"historic", "amenity", "building"}),
    "place_of_worship": ("религиозный объект", {"amenity"}),
    "theatre": ("театр", {"amenity"}),
    "cinema": ("кинотеатр", {"amenity"}),
    "observatory": ("обсерватория", {"man_made", "tourism"}),
    "mall": ("торговый центр", {"shop"}),
    "shopping_centre": ("торговый центр", {"shop"}),
}


def _value(value, key, default=None):
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _needs_explanation(title):
    # An already Russian/mixed Russian label is readable; do not append a
    # second explanation to e.g. "Louvre (Лувр)" or "Музей LEGO".
    return (isinstance(title, str) and 0 < len(title) <= 240
            and any(unicodedata.category(char).startswith("L") for char in title)
            and not any(char in _RUSSIAN_LETTERS for char in title))


def _russian_source_name(record):
    details = record.get("namedetails")
    value = details.get("name:ru") if isinstance(details, Mapping) else None
    if (not isinstance(value, str) or not 1 <= len(value) <= 160 or value != value.strip()
            or not any(char in _RUSSIAN_LETTERS for char in value)):
        return ""
    # Plain, bounded source text only: no HTML, URL, control/bidi characters
    # or pseudo-JSON. It is rendered as text, never a prompt or executable HTML.
    if any(unicodedata.category(char)[0] not in "LMN" and char not in " .,'’()-\"«»–—"
           for char in value):
        return ""
    return value


def _record_explanation(record):
    russian_name = _russian_source_name(record)
    if russian_name:
        return russian_name
    category, kind = record.get("category", ""), record.get("type")
    entry = _TYPE_EXPLANATIONS.get(kind) if isinstance(kind, str) else None
    # Older checked snapshots can lack category but preserve a known type.
    # Shopping centres require their dedicated shop tag, just like discovery.
    if entry and (category in entry[1] or category in (None, "") and kind not in {"mall", "shopping_centre"}):
        return entry[0]
    tags = record.get("extratags")
    if ((category, kind) in {("amenity", "university"), ("office", "research")}
            and isinstance(tags, Mapping) and tags.get("observatory:type") == "astronomical"):
        return "астрономическая обсерватория"
    return _GENERIC_EXPLANATION


def display_place_title(item, labels: Mapping[str, str] | None = None) -> str:
    """Return display-only text for a dict/model item; never change identity.

    ``labels`` must be made by :func:`build_place_labels` for the same saved
    itinerary. Without trusted evidence a foreign name gets an honest generic
    explanation. Meals, breaks, transfers and existing Russian text stay intact.
    Consumers must use their ordinary HTML/PDF/Telegram text escaping.
    """
    title = _value(item, "title", "")
    if not isinstance(title, str):
        return ""
    if _value(item, "kind", "attraction") != "attraction" or not _needs_explanation(title):
        return title
    query = _value(item, "place_query", "")
    label = labels.get(query) if isinstance(labels, Mapping) and isinstance(query, str) else None
    if (isinstance(label, str) and label.startswith(title + " (") and label.endswith(")")
            and len(label) <= len(title) + 163):
        return label
    return f"{title} ({_GENERIC_EXPLANATION})"


def build_place_labels(trip: dict) -> dict[str, str]:
    """Build exact-query display labels from an already owner-authorized trip.

    Accept the ``db.get_trip`` shape: ``request`` and ``itinerary`` are decoded
    dictionaries (an Itinerary model is also accepted). This is a read-only
    projection, not an authorization boundary. Missing, stale or invalid city
    evidence falls back to generic text; no legacy network recovery is run.
    The existing 2GIS validation may read its application-owned local catalogue;
    that safety check is not replaced by trusting a saved provenance marker.
    """
    if not isinstance(trip, Mapping):
        return {}
    raw_itinerary = trip.get("itinerary")
    days = _value(raw_itinerary, "days", ())
    if not isinstance(days, (list, tuple)):
        return {}
    items = [item for day in days if isinstance(_value(day, "items"), (list, tuple))
             for item in _value(day, "items")
             if _value(item, "kind", "attraction") == "attraction"]
    labels = {query: display_place_title(item) for item in items
              if isinstance(query := _value(item, "place_query"), str) and query}
    raw_request = trip.get("request")
    if not isinstance(raw_request, Mapping) or "city_map_context" not in raw_request:
        return labels
    # Import only on the checked-city path. The context loader revalidates the
    # existing record snapshot; unlike prepare/recover it performs no network,
    # geocoder/cache recovery or database calls. The 2GIS authority remains its
    # exact current local catalogue, not a client-supplied provider marker.
    from city_context import check_context_places, load_city_context
    from schemas import Itinerary, TripRequest
    try:
        request = TripRequest.model_validate(raw_request)
        itinerary = Itinerary.model_validate(raw_itinerary)
        candidates, _ = load_city_context(raw_request["city_map_context"], request)
        check_context_places(itinerary, candidates)
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
        return labels
    records = {candidate.place_query: candidate.record for candidate in candidates}
    for item in items:
        title, query = _value(item, "title"), _value(item, "place_query")
        if _needs_explanation(title) and query in records:
            labels[query] = f"{title} ({_record_explanation(records[query])})"
    return labels
