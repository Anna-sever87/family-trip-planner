"""Проверка мест: Nominatim и отдельный локальный каталог с явным источником."""
import json
import logging
import math
import sqlite3
import time
import unicodedata
from collections.abc import Mapping
from contextlib import closing, contextmanager
from copy import deepcopy
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from schemas import Itinerary
import verified_map_places

logger = logging.getLogger("places")

USER_AGENT = "family-trip-planner/0.1 (student project)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_CLOCK_FILE = Path(__file__).resolve().parent / "output" / "nominatim_clock.db"
CACHE_FILE = Path("output/geocode_cache.json")
CURATED_PLACES_FILE = (
    Path(__file__).resolve().parent
    / "data"
    / "verified_places.json"
)
MIN_INTERVAL = 1.1            # правило сервиса: не чаще одного запроса в секунду
MAX_DAY_SPREAD_KM = 8.0       # максимальное расстояние между местами одного дня
MAX_AREA_DISTANCE_KM = 3.0    # место должно соответствовать указанному району
NEGATIVE_CACHE_TTL_SECONDS = 3600
MAX_QUERY_VARIANTS = 6

_QUOTE_CHARACTERS = "\"'«»„“”‘’"
_OPTIONAL_NAME_PREFIXES = (
    "памятник ",
    "монумент ",
    "скульптура ",
)
_GEOGRAPHIC_NAME_PREFIX_TYPES = {"озеро ": "lake", "lake ": "lake"}
# Алиасы написания, не каталог POI и не источник координат.
_COUNTRY_NAME_CODES = {
    "vn": ("Вьетнам", "Vietnam", "Viet Nam", "Việt Nam"),
    "tr": ("Турция", "Turkey", "Türkiye", "Turkiye"),
    "ru": ("Россия", "Russia", "Российская Федерация"),
    "fr": ("Франция", "France"),
    "jp": ("Япония", "Japan", "日本"),
    "us": ("США", "USA", "United States", "United States of America", "Соединённые Штаты Америки"),
    "gb": ("Великобритания", "UK", "United Kingdom", "Соединённое Королевство"),
    "de": ("Германия", "Germany", "Deutschland"),
}
_GEOGRAPHIC_CONTEXT_ALIASES = (
    ("Paris", "Париж"),
    ("Tokyo", "Токио", "東京"),
)
# Тип подтверждён в ответе Nominatim; координаты по-прежнему берутся из API.
_KNOWN_GEOGRAPHIC_QUERY_TYPES = {"байкал": "lake", "baikal": "lake"}
_GENERIC_NAME_WORDS = {
    "памятник", "монумент", "скульптура", "музей", "парк",
    "кинотеатр", "театр", "центр", "город", "район", "округ",
    "museum", "park", "cinema", "theatre", "theater", "monument",
    "озеро", "lake",
}

GENERIC_AREA_LABELS = {
    "город",
    "центр",
    "центр города",
    "центральный район",
    "район города",
    "окрестности",
    "рядом с отелем",
}

_cache: Optional[Dict[str, Any]] = None
_curated_cache: Optional[List[Dict[str, Any]]] = None
_blocked_cache: Optional[List[Dict[str, Any]]] = None


def _load_cache() -> Dict[str, Any]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _save_cache() -> None:
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")


@contextmanager
def _wait_for_turn():
    """Serialize Nominatim attempts across bot/web threads and processes.

    Hold the shared lock through HTTP so a delayed caller cannot send after a
    later reservation. Persist failed attempts too; a broken clock fails closed.
    The absolute default path shares one clock regardless of the launch folder.
    """
    try:
        path = Path(NOMINATIM_CLOCK_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(path, timeout=40)) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS request_clock (id INTEGER PRIMARY KEY CHECK(id=1), requested REAL NOT NULL)")
            connection.commit()
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT requested FROM request_clock WHERE id=1").fetchone()
            if previous:
                if not isinstance(previous[0], (int, float)) or not math.isfinite(previous[0]):
                    raise GeocodeUnavailable("Повреждён общий ограничитель запросов карты.")
                # A wall-clock correction must not create an unbounded wait.
                pause = min(MIN_INTERVAL, max(0, MIN_INTERVAL - (time.time() - previous[0])))
                if pause:
                    time.sleep(pause)
            connection.execute("INSERT OR REPLACE INTO request_clock VALUES(1,?)", (time.time(),))
            try:
                yield
            finally:
                connection.commit()
    except (sqlite3.Error, OSError) as error:
        raise GeocodeUnavailable("Не удалось обеспечить общий интервал запросов карты.") from error


class GeocodeUnavailable(Exception):
    """Сервис карты не ответил (сбой сети или ограничение). Это не значит, что места не существует."""


class GeocodeMismatch(Exception):
    """Карта ответила, но найденный объект не соответствует запросу."""


class GeocodeIncomplete(GeocodeUnavailable):
    """Ограниченная проверка не подтвердила место; полный поиск не завершён."""

    def __init__(self, query: str, *, reason: str):
        super().__init__(query)
        self.reason = reason


def _fold(text: str) -> str:
    """Убирает диакритику: Musée -> Musee."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _variants(query: str) -> List[str]:
    """Небольшой набор написаний одного имени, всегда с исходным городом."""
    parts = [part.strip() for part in query.split(",") if part.strip()]
    if not parts:
        return []
    variants: List[str] = []

    def add(value: str) -> None:
        value = value.strip()
        if value and value not in variants and len(variants) < MAX_QUERY_VARIANTS:
            variants.append(value)

    add(query)
    unquoted = "".join(char for char in parts[0] if char not in _QUOTE_CHARACTERS)
    unquoted = " ".join(unquoted.split())
    names = [parts[0], unquoted]
    lowered = unquoted.casefold()
    for prefix in _OPTIONAL_NAME_PREFIXES:
        if lowered.startswith(prefix):
            remainder = unquoted[len(prefix):].strip()
            words = set(_identity_words(remainder))
            if words and not words.issubset(_GENERIC_NAME_WORDS):
                names.append(remainder)
            break
    for prefix in _GEOGRAPHIC_NAME_PREFIX_TYPES:
        if lowered.startswith(prefix):
            remainder = unquoted[len(prefix):].strip()
            words = set(_identity_words(remainder))
            if words and not words.issubset(_GENERIC_NAME_WORDS):
                names.append(remainder)
            break
    for name in names:
        add(", ".join([name] + parts[1:]))
    # Удаляется только последнее поле страны; адресные части и город остаются.
    if len(parts) >= 3:
        for name in names:
            add(", ".join([name] + parts[1:-1]))
    add(_fold(query.strip()))
    return variants


def _identity_words(value: str) -> tuple[str, ...]:
    text = _fold(unicodedata.normalize("NFKC", str(value))).casefold()
    text = "".join(
        char if unicodedata.category(char)[0] in {"L", "N"} else " "
        for char in text
    )
    return tuple(text.split())


def _name_identities(value: str) -> set[tuple[str, ...]]:
    words = _identity_words(value)
    identities = {tuple(sorted(words))} if words else set()
    for prefix in _OPTIONAL_NAME_PREFIXES:
        prefix_words = _identity_words(prefix)
        if words[:len(prefix_words)] == prefix_words:
            remainder = words[len(prefix_words):]
            if remainder and not set(remainder).issubset(_GENERIC_NAME_WORDS):
                identities.add(tuple(sorted(remainder)))
    return identities


def _geographic_name_identities(value: str) -> set[tuple[str, ...]]:
    identities = _name_identities(value)
    words = _identity_words(value)
    for prefix in _GEOGRAPHIC_NAME_PREFIX_TYPES:
        prefix_words = _identity_words(prefix)
        if words[:len(prefix_words)] == prefix_words:
            remainder = words[len(prefix_words):]
            if remainder and not set(remainder).issubset(_GENERIC_NAME_WORDS):
                identities.add(tuple(sorted(remainder)))
    return identities


def _context_identity(value: str) -> tuple[str, ...]:
    identity = _identity_words(value)
    for aliases in _GEOGRAPHIC_CONTEXT_ALIASES:
        if identity in {_identity_words(alias) for alias in aliases}:
            return _identity_words(aliases[0])
    return identity


def _country_identity(value: str) -> tuple[str, ...]:
    identity = _identity_words(value)
    for code, aliases in _COUNTRY_NAME_CODES.items():
        if identity == (code,) or identity in {_identity_words(alias) for alias in aliases}:
            return (code,)
    return identity


def _place_matches_query(query: str, place: Any) -> bool:
    """Координаты подтверждают только совпавшие имя и географический контекст."""
    if not isinstance(place, dict):
        return False
    if (place.get("provider") == "2gis" or place.get("provenance") == "official_linked_map") and not verified_map_places.is_current_verified_map_record(place):
        return False
    if isinstance(place.get("lat"), bool) or isinstance(place.get("lon"), bool):
        return False
    try:
        lat, lon = float(place["lat"]), float(place["lon"])
    except (KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
        return False
    parts = [part.strip() for part in str(query).split(",") if part.strip()]
    display_parts = [part.strip() for part in str(place.get("name") or "").split(",") if part.strip()]
    if not parts or not display_parts:
        return False
    address = place.get("address")
    address = address if isinstance(address, dict) else {}

    # Двухкомпонентный «объект, страна» также ограничен страной.
    if len(parts) >= 2:
        expected_country = _country_identity(parts[-1])
        actual_code = str(address.get("country_code") or "").strip().casefold()
        known_expected_code = expected_country[0] if len(expected_country) == 1 and len(expected_country[0]) == 2 else ""
        if known_expected_code and actual_code:
            if known_expected_code != actual_code:
                return False
        elif expected_country not in {
            _country_identity(display_parts[-1]),
            _country_identity(str(address.get("country") or "")),
        }:
            return False

    # Every explicitly supplied locality/admin component constrains identity.
    # In «POI, city, province, country» checking only province would silently
    # discard the city. An explicit house number is validated separately below.
    if len(parts) >= 3:
        context_labels = display_parts[1:] + [
            str(address.get(field) or "")
            for field in ("city", "town", "village", "hamlet", "municipality", "county", "state", "region",
                          "suburb", "neighbourhood", "quarter", "borough", "city_district", "district")
        ]
        expected_context = parts[2:-1] if len(parts) >= 4 and parts[1][0].isdigit() else parts[1:-1]
        actual_context = {_context_identity(value) for value in context_labels}
        if any(_context_identity(value) not in actual_context for value in expected_context):
            return False

    names = [display_parts[0]]
    details = place.get("namedetails")
    if isinstance(details, dict):
        for key, value in details.items():
            if key == "name" or key.startswith(("name:", "official_name", "alt_name", "short_name")):
                names.extend(str(value or "").split(";"))
    query_name = parts[0]
    query_identities = _name_identities(query_name)
    expected_geographic_type = next((expected_type for name, expected_type in _KNOWN_GEOGRAPHIC_QUERY_TYPES.items() if _identity_words(name) == _identity_words(query_name)), "")
    for prefix, expected_type in _GEOGRAPHIC_NAME_PREFIX_TYPES.items():
        if query_name.casefold().startswith(prefix):
            expected_geographic_type = expected_type
            break
    if expected_geographic_type:
        if str(place.get("type") or "").casefold() != expected_geographic_type:
            return False
    if str(place.get("type") or "").casefold() == "lake":
        query_identities = _geographic_name_identities(query_name)
        if any(query_identities & _geographic_name_identities(value) for value in names):
            return True
    if any(query_name.casefold().startswith(prefix) for prefix in _OPTIONAL_NAME_PREFIXES):
        # Совпавшее имя парка/музея не превращает этот объект в памятник.
        conflicting_types = {"museum", "park", "theatre", "cinema", "administrative", "city", "town", "village", "water", "lake", "river", "residential", "pedestrian", "restaurant", "school"}
        if str(place.get("type") or "").casefold() in conflicting_types:
            return False
    if len(parts) >= 4 and parts[1][0].isdigit():
        actual_house = str(address.get("house_number") or "")
        if not actual_house:
            actual_house = next((value for value in display_parts[:3] if value[0].isdigit()), "")
        if not actual_house or _identity_words(parts[1]) != _identity_words(actual_house):
            return False
    if any(query_identities & _name_identities(value) for value in names):
        return True

    # Точный адрес может находиться после названия здания в display_name.
    # Это не разрешение искать имя POI среди произвольных районов/улиц.
    name_words = _identity_words(query_name)
    address_markers = {"улица", "проспект", "площадь", "переулок", "шоссе", "street", "road", "avenue", "square"}
    if set(name_words) & address_markers:
        address_names = display_parts[:3] + [str(address.get("road") or "")]
        if any(query_identities & _name_identities(value) for value in address_names):
            return True
    return False


def _fresh_negative_cache(value: Any, now: float) -> bool:
    """Старые boolean-промахи не имеют даты и должны проверяться заново."""
    if not isinstance(value, dict):
        return False
    checked_at = value.get("checked_at")
    if isinstance(checked_at, bool) or not isinstance(checked_at, (int, float)):
        return False
    age = now - checked_at
    return math.isfinite(age) and 0 <= age < NEGATIVE_CACHE_TTL_SECONDS


def _normalized_label(value: str) -> str:
    """Нормализует короткое название для безопасного сравнения."""

    return " ".join(
        _fold(value).casefold().split()
    )


_NORMALIZED_GENERIC_AREA_LABELS = frozenset(
    _normalized_label(value) for value in GENERIC_AREA_LABELS
)


def _destination_matches(
    requested: str,
    catalog_value: str,
) -> bool:
    """Сопоставляет полное направление и безопасный ввод одного города."""

    requested_key = _normalized_label(requested)
    catalog_key = _normalized_label(catalog_value)

    if not requested_key or not catalog_key:
        return False

    if requested_key == catalog_key:
        return True

    requested_parts = [
        part.strip()
        for part in str(requested).split(",")
        if part.strip()
    ]
    catalog_parts = [
        part.strip()
        for part in str(catalog_value).split(",")
        if part.strip()
    ]

    return (
        len(requested_parts) == 1
        and bool(catalog_parts)
        and _normalized_label(requested_parts[0])
        == _normalized_label(catalog_parts[0])
    )


def _area_search_query(
    area: str,
    place_query: str,
) -> Optional[str]:
    """Собирает запрос района в контексте города и страны."""

    clean_area = area.strip()

    if (
        not clean_area
        or _normalized_label(clean_area)
        in _NORMALIZED_GENERIC_AREA_LABELS
    ):
        return None

    parts = [
        part.strip()
        for part in place_query.split(",")
        if part.strip()
    ]

    context = ", ".join(parts[1:])

    if context:
        return f"{clean_area}, {context}"

    return clean_area


def _area_matches_map_context(area: str, place: Dict[str, Any]) -> bool:
    """Reuse an exact map locality after the POI's identity/country check.

    Administrative localities need not appear in display_name; their centroid
    is not a second proof of the already supplied address. Neither a substring
    of a locality, the POI name itself, nor the country is an area match.
    """
    wanted = _normalized_label(area)
    if not wanted:
        return False
    address = place.get("address")
    labels = [
        value for field in (
            "city", "town", "village", "hamlet", "suburb", "neighbourhood", "quarter", "locality",
            "municipality", "county", "state", "region",
        )
        if isinstance(address, dict) and isinstance((value := address.get(field)), str)
    ]
    # Retain legacy display-address evidence, but only whole components and
    # without its first (POI) or final (country) component.
    display = [part.strip() for part in str(place.get("name") or "").split(",") if part.strip()]
    labels.extend(display[1:-1])
    return any(_normalized_label(value) == wanted for value in labels)


def _osm_identity(value: Any) -> Optional[tuple[str, int]]:
    """An optional provider identity, never evidence of type or visitability."""
    if not isinstance(value, dict):
        return None
    osm_type, osm_id = value.get("osm_type"), value.get("osm_id")
    if (
        isinstance(osm_type, str) and osm_type in {"node", "way", "relation"}
        and isinstance(osm_id, int) and not isinstance(osm_id, bool) and osm_id > 0
    ):
        return osm_type, osm_id
    return None


def map_record_identity(record: Any) -> Optional[tuple]:
    """Identity of an ALREADY trusted map record, not arbitrary model JSON."""
    if not isinstance(record, dict):
        return None
    if _independent_map_record(record):
        identifier = record.get("provider_id")
        if (record.get("provider") == "2gis" and _osm_identity(record) is None
                and isinstance(identifier, str) and 1 <= len(identifier) <= 32
                and identifier.isascii() and identifier.isdigit() and not identifier.startswith("0")):
            return "2gis", identifier
        return None
    identity = _osm_identity(record)
    if identity is not None and record.get("provider", "osm") in ("", "osm"):
        return ("osm", *identity)
    return None


def compare_map_identities(first: Optional[tuple], second: Optional[tuple]) -> Optional[bool]:
    """True=same object, False=distinct comparable IDs, None=unknown relation."""
    if first is not None and second is not None and first[0] == second[0]:
        return first == second
    return None


def same_map_place(first: Any, second: Any, *, first_identity: Optional[tuple] = None,
                   second_identity: Optional[tuple] = None) -> bool:
    """Use authoritative IDs first; proximity is only an unknown-ID fallback."""
    relation = compare_map_identities(first_identity, second_identity)
    if relation is not None:
        return relation
    coordinates = []
    for record in (first, second):
        get = record.get if isinstance(record, dict) else lambda key: getattr(record, key, None)
        lat, lon = get("lat"), get("lon")
        if any(type(value) not in (int, float) or not -bound <= value <= bound or not math.isfinite(value)
               for value, bound in ((lat, 90), (lon, 180))):
            return False
        coordinates.extend((lat, lon))
    return distance_km(*coordinates) <= 0.03


def resolved_map_identity(item: Any, *, require_verified: bool = True) -> Optional[tuple]:
    """Read-only identity lookup: never trust an item's IDs or URL alone.

    Only an exact query and exact coordinates in a valid existing OSM cache
    record or the current independent-provider catalogue can supply identity.
    This does not mark an item verified, geocode, refresh or write the cache.
    """
    get = item.get if isinstance(item, dict) else lambda key, default=None: getattr(item, key, default)
    if require_verified and (get("kind", "attraction") != "attraction" or get("verified") is not True):
        return None
    query, lat, lon = get("place_query", ""), get("lat"), get("lon")
    if (not isinstance(query, str) or not query or "|" in query
            or any(type(value) not in (int, float) or not -bound <= value <= bound or not math.isfinite(value)
                   for value, bound in ((lat, 90), (lon, 180)))):
        return None
    provider = get("provider", "")
    if provider == "2gis":
        record = verified_map_places.exact_verified_map_place(query)
        if (record is not None and verified_map_places.is_current_verified_map_record(record)
                and lat == record["lat"] and lon == record["lon"]
                and get("source_url") == record["source_url"]):
            return map_record_identity(record)
        return None
    if provider not in ("", "osm") or (not provider and get("source_url", "")):
        return None
    cache = _load_cache()
    if not isinstance(cache, dict):
        return None
    record = cache.get(query.lower())
    if (_independent_map_record(record) or not _place_matches_query(query, record)
            or _fresh_negative_cache(cache.get(query.lower() + "|miss"), time.time())):
        return None
    if lat != float(record["lat"]) or lon != float(record["lon"]):
        return None
    return map_record_identity(record)


def same_cached_map_place(first: Any, second: Any, *, require_verified: bool = True) -> bool:
    """Compare app items/hints only with independently resolved identities."""
    return same_map_place(first, second,
                          first_identity=resolved_map_identity(first, require_verified=require_verified),
                          second_identity=resolved_map_identity(second, require_verified=require_verified))


def _semantic_place_metadata(value: Any) -> Dict[str, str]:
    """Keep only provider tags used for a narrow semantic POI classification.

    A tag describes the facility, not opening hours or permission to visit it.
    Other extra tags (including website/operator/free text) are not evidence.
    """
    if isinstance(value, dict) and value.get("observatory:type") == "astronomical":
        return {"observatory:type": "astronomical"}
    return {}


def _search(
    query: str, *, validation_query: Optional[str] = None, include_geometry: bool = False,
    viewbox: Optional[tuple[float, float, float, float]] = None,
    within_radius: Optional[tuple[float, float, float]] = None,
    place_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Optional[Dict[str, Any]]:
    """Один поиск в Nominatim с одним повтором при сбое сети."""
    for attempt in range(2):
        try:
            query_parts = [part.strip() for part in (validation_query or query).split(",") if part.strip()]
            context = query_parts[-2] if len(query_parts) >= 3 else query
            language = "ru" if any("а" <= char.casefold() <= "я" or char.casefold() == "ё" for char in context) else "en"
            if any("\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff" for char in context):
                language = "ja"
            params = {"q": query, "format": "json", "limit": 5, "accept-language": language, "namedetails": 1, "addressdetails": 1}
            if viewbox is not None:
                params["viewbox"] = ",".join(str(value) for value in viewbox)
                params["bounded"] = 1
            if within_radius is not None:
                params["extratags"] = 1
            if include_geometry:
                params["polygon_geojson"] = 1
                # Региональная проверка допускает ближайшие окрестности,
                # поэтому субметровый береговой контур не нужен. Ответ API
                # упрощается примерно до сотен метров, без ручных координат.
                params["polygon_threshold"] = 0.002
            with _wait_for_turn():
                response = httpx.get(
                    NOMINATIM_URL,
                    params=params,
                    headers={"User-Agent": USER_AGENT},
                    timeout=15,
                )
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                raise ValueError("Некорректный ответ карты")
            for result in data:
                if not isinstance(result, dict):
                    continue
                place = {
                    "lat": result.get("lat"),
                    "lon": result.get("lon"),
                    "name": result.get("display_name"),
                    "namedetails": result.get("namedetails", {}),
                    "address": result.get("address", {}),
                    "boundingbox": result.get("boundingbox", []),
                    "category": result.get("category", result.get("class", "")),
                    "type": result.get("type", ""),
                    "addresstype": result.get("addresstype", ""),
                }
                identity = _osm_identity(result)
                if identity is not None:
                    place["osm_type"], place["osm_id"] = identity
                if include_geometry or "geojson" in result:
                    place["geojson"] = result.get("geojson")
                semantic_tags = _semantic_place_metadata(result.get("extratags"))
                if semantic_tags:
                    place["extratags"] = semantic_tags
                if _place_matches_query(validation_query or query, place):
                    place["lat"], place["lon"] = float(place["lat"]), float(place["lon"])
                    if within_radius is not None:
                        base_lat, base_lon, maximum = within_radius
                        if distance_km(base_lat, base_lon, place["lat"], place["lon"]) > maximum:
                            continue
                        if _nearby_named_record(query_parts[0], query_parts[-1], base_lat, base_lon, maximum, place) is None:
                            continue
                    if place_filter is not None and not place_filter(place):
                        continue
                    return place
            if data:
                raise GeocodeMismatch(query)
            return None
        except (httpx.HTTPError, ValueError, KeyError):
            logger.warning("Сбой поиска на карте (%s), попытка %s", query, attempt + 1)
            time.sleep(2)
    raise GeocodeUnavailable(query)


def geocode(query: str, *, include_geometry: bool = False, max_variants: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Ищет место на карте. Возвращает {lat, lon, name} или None, если места нет.

    Если сервис карты недоступен, бросает GeocodeUnavailable (сбой не запоминаем).
    max_variants ограничивает число написаний (до двух HTTP-попыток на каждое).
    Неуспех ограниченного поиска — GeocodeIncomplete, без записи полного промаха.
    """
    if max_variants is not None and (
        isinstance(max_variants, bool) or not isinstance(max_variants, int)
        or not 1 <= max_variants <= MAX_QUERY_VARIANTS
    ):
        raise ValueError("max_variants должен быть целым числом от 1 до MAX_QUERY_VARIANTS")
    if not include_geometry:
        local = verified_map_places.exact_verified_map_place(query)
        if local is not None and _place_matches_query(query, local):
            return local
    cache = _load_cache()
    key = query.strip().lower()
    if include_geometry:
        key += "|geometry"
    if not _independent_map_record(cache.get(key)) and _place_matches_query(query, cache.get(key)):
        return cache[key]
    negative = cache.get(key + "|miss")
    if _fresh_negative_cache(negative, time.time()):
        if negative.get("reason") == "mismatch":
            raise GeocodeMismatch(query)
        return None
    mismatched = isinstance(cache.get(key), dict) and bool(cache[key].get("name"))
    variants = _variants(query)
    for variant in variants if max_variants is None else variants[:max_variants]:
        try:
            options: Dict[str, Any] = {"validation_query": query}
            if include_geometry:
                options["include_geometry"] = True
            place = _search(variant, **options)
        except GeocodeMismatch:
            mismatched = True
            continue
        if _place_matches_query(query, place):
            cache[key] = place
            cache.pop(key + "|miss", None)
            _save_cache()
            return place
        if place is not None:
            mismatched = True
    if max_variants is not None:
        raise GeocodeIncomplete(query, reason="mismatch" if mismatched else "not_found")
    cache.pop(key, None)
    cache[key + "|miss"] = {
        "checked_at": time.time(),
        "reason": "mismatch" if mismatched else "not_found",
    }
    _save_cache()
    if mismatched:
        raise GeocodeMismatch(query)
    return None


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Расстояние между двумя точками по прямой (формула гаверсинусов)."""
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    d_lat = p2 - p1
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _nearby_named_record(
    title: str, country: str, base_lat: float, base_lon: float,
    max_distance_km: float, place: Any,
) -> Optional[Dict[str, Any]]:
    """A canonical alias uses only locality actually supplied by the map."""
    query = f"{title}, {country}"
    if not _place_matches_query(query, place) or not str(place.get("type") or "").strip():
        return None
    if distance_km(base_lat, base_lon, float(place["lat"]), float(place["lon"])) > max_distance_km:
        return None
    address = place.get("address")
    if not isinstance(address, dict):
        return None
    for field in ("village", "town", "city", "municipality", "county", "state", "region"):
        locality = address.get(field)
        if not isinstance(locality, str) or not locality.strip() or not locality.isprintable() or "," in locality:
            continue
        locality = locality.strip()
        canonical = f"{title}, {locality}, {country}"
        if _place_matches_query(canonical, place):
            result = deepcopy(place)
            result.update(lat=float(place["lat"]), lon=float(place["lon"]),
                          place_query=canonical, locality=locality)
            return result
    return None


def _cached_nearby_named(
    title: str, country: str, base_lat: float, base_lon: float, max_distance_km: float = 80,
    *, place_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Optional[Dict[str, Any]]:
    """Read-only positive cache reuse; missing type/locality is not evidence."""
    for record in verified_map_places.same_title_verified_map_places(title, country):
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (base_lat, base_lon, max_distance_km)):
            continue
        if not (-90 <= base_lat <= 90 and -180 <= base_lon <= 180 and 0 < max_distance_km <= 80):
            continue
        place = _nearby_named_record(title, country, base_lat, base_lon, max_distance_km, record)
        if place is not None and (place_filter is None or place_filter(place)):
            return place
    for key, record in list(_load_cache().items()):
        if not isinstance(key, str) or "|" in key or _independent_map_record(record) or not _place_matches_query(key, record):
            continue
        place = _nearby_named_record(title, country, base_lat, base_lon, max_distance_km, record)
        if place is not None and (place_filter is None or place_filter(place)):
            return place
    return None


def _independent_map_record(record: Any) -> bool:
    """Independent-provider facts must never become timeless OSM cache hits."""
    return isinstance(record, dict) and (
        record.get("provider") == "2gis" or record.get("provenance") == "official_linked_map"
    )


def _remember_nearby_alias(place: Dict[str, Any]) -> None:
    """Persist the validated canonical alias, never a country-wide search key."""
    if _independent_map_record(place):
        return
    cache = _load_cache()
    key = place["place_query"].lower()
    record = {name: deepcopy(value) for name, value in place.items() if name not in {"place_query", "locality"}}
    if cache.get(key) != record or key + "|miss" in cache:
        cache[key] = record
        cache.pop(key + "|miss", None)
        _save_cache()


def geocode_nearby_named(
    title: str, country: str, base_lat: float, base_lon: float, max_distance_km: float = 80,
    *, place_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Optional[Dict[str, Any]]:
    """One exact named POI search near a base, at most two HTTP attempts.

    The square viewbox bounds the provider search; the circular radius is
    checked for every result. No variants, category search, negative cache, or
    assumed village: returned place_query/locality come from map address.
    Region boundary and allowed POI type remain the caller's responsibility.
    """
    if any(not isinstance(value, str) or not value or value != value.strip()
           or not value.isprintable() or any(char in value for char in ",[]")
           for value in (title, country)):
        raise ValueError("Нужны точное название одного места и страна")
    generic_words = {token for word in _GENERIC_NAME_WORDS for token in _identity_words(word)}
    if not _identity_words(title) or set(_identity_words(title)).issubset(generic_words):
        raise ValueError("Нужен именованный объект, не категория мест")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in (base_lat, base_lon, max_distance_km)):
        raise ValueError("Координаты базы и радиус должны быть конечными числами")
    if not (-90 <= base_lat <= 90 and -180 <= base_lon <= 180 and 0 < max_distance_km <= 80):
        raise ValueError("Некорректные координаты базы или радиус больше 80 км")
    if place_filter is not None and not callable(place_filter):
        raise ValueError("Фильтр территории должен быть вызываемой функцией")
    place = _cached_nearby_named(title, country, base_lat, base_lon, max_distance_km, place_filter=place_filter)
    if place is None:
        angular = max_distance_km / 6371.0
        latitude_delta = math.degrees(angular)
        south, north = max(-90.0, base_lat - latitude_delta), min(90.0, base_lat + latitude_delta)
        if south <= -90 or north >= 90:
            west, east = -180.0, 180.0
        else:
            longitude_delta = math.degrees(math.asin(min(1.0, math.sin(angular) / math.cos(math.radians(base_lat)))))
            west, east = base_lon - longitude_delta, base_lon + longitude_delta
            # A single box crossing the date line needs a full longitude span;
            # strict circular filtering still prevents remote matches.
            if west < -180 or east > 180:
                west, east = -180.0, 180.0
        query = f"{title}, {country}"
        raw = _search(query, validation_query=query, viewbox=(west, north, east, south),
                      within_radius=(base_lat, base_lon, max_distance_km), place_filter=place_filter)
        place = _nearby_named_record(title, country, base_lat, base_lon, max_distance_km, raw)
    if place is None:
        return None
    _remember_nearby_alias(place)
    return place


def _load_curated_places() -> List[Dict[str, Any]]:
    """Читает небольшой отслеживаемый каталог проверенных POI."""

    global _curated_cache

    if _curated_cache is not None:
        return _curated_cache

    try:
        payload = json.loads(
            CURATED_PLACES_FILE.read_text(
                encoding="utf-8"
            )
        )
        raw_places = payload.get("places", [])

        if not isinstance(raw_places, list):
            raise ValueError("places должен быть списком")

        _curated_cache = [
            item
            for item in raw_places
            if isinstance(item, dict)
        ]
    except (OSError, ValueError, TypeError) as error:
        logger.warning(
            "Не удалось прочитать каталог проверенных мест: %s",
            type(error).__name__,
        )
        _curated_cache = []

    return _curated_cache


def curated_place_hints(
    destination: str,
    center: Optional[tuple[float, float]],
    *,
    max_distance_km: float = 40.0,
    limit: int = 40,
) -> List[Dict[str, Any]]:
    """Возвращает POI с проверенным назначением и режимом доступа.

    В отличие от обычного geocode-cache этот каталог хранит не только
    координаты, но и безопасный сценарий посещения. Сетевых запросов нет.
    """

    if center is None or limit < 1:
        return []

    if not _normalized_label(
        str(destination or "")
    ):
        return []

    candidates: List[Dict[str, Any]] = []
    seen_queries: set[str] = set()

    for raw_place in _load_curated_places():
        if (
            not _destination_matches(
                str(destination or ""),
                str(raw_place.get("destination") or ""),
            )
            or str(raw_place.get("status") or "active")
            .strip()
            .casefold()
            != "active"
        ):
            continue

        title = str(
            raw_place.get("title") or ""
        ).strip()
        place_query = str(
            raw_place.get("place_query") or ""
        ).strip()
        official_name = str(
            raw_place.get("official_name") or ""
        ).strip()
        access_mode = str(
            raw_place.get("access_mode") or ""
        ).strip().casefold()
        verified_at = str(
            raw_place.get("verified_at") or ""
        ).strip()
        source = str(
            raw_place.get("source") or ""
        ).strip()

        try:
            date.fromisoformat(verified_at)
        except ValueError:
            continue

        try:
            lat = float(raw_place["lat"])
            lon = float(raw_place["lon"])
        except (KeyError, TypeError, ValueError):
            continue

        query_key = _normalized_label(place_query)

        if (
            not title
            or not query_key
            or not official_name
            or not math.isfinite(lat)
            or not math.isfinite(lon)
            or not -90 <= lat <= 90
            or not -180 <= lon <= 180
            or not source.startswith("https://")
            or query_key in seen_queries
            or distance_km(
                float(center[0]),
                float(center[1]),
                lat,
                lon,
            )
            > max_distance_km
        ):
            continue

        seen_queries.add(query_key)
        candidates.append(
            {
                "title": title[:120],
                "place_query": place_query[:220],
                "lat": lat,
                "lon": lon,
                "official_name": official_name[:220],
                "description": str(
                    raw_place.get("description") or ""
                ).strip()[:180],
                "indoor": raw_place.get("indoor") is True,
                "access_mode": access_mode,
                "trusted_late": (
                    raw_place.get("trusted_late") is True
                    and access_mode
                    in {
                        "exterior",
                        "public_exterior",
                        "public_24h",
                    }
                ),
                "catalog_entry": True,
                "verified_at": verified_at[:20],
                "source": source[:500],
            }
        )

        if len(candidates) >= limit:
            break

    return candidates


def curated_blocked_place_terms(
    destination: str,
) -> List[Dict[str, Any]]:
    """Возвращает проверенные исключения для конкретного направления."""

    global _blocked_cache

    if _blocked_cache is None:
        try:
            payload = json.loads(
                CURATED_PLACES_FILE.read_text(
                    encoding="utf-8"
                )
            )
            raw_places = payload.get(
                "blocked_places",
                [],
            )

            if not isinstance(raw_places, list):
                raise ValueError(
                    "blocked_places должен быть списком"
                )

            _blocked_cache = [
                item
                for item in raw_places
                if isinstance(item, dict)
            ]
        except (OSError, ValueError, TypeError) as error:
            logger.warning(
                "Не удалось прочитать исключения мест: %s",
                type(error).__name__,
            )
            _blocked_cache = []

    blocked: List[Dict[str, Any]] = []

    for raw_place in _blocked_cache:
        if (
            not _destination_matches(
                str(destination or ""),
                str(raw_place.get("destination") or ""),
            )
            or str(raw_place.get("status") or "")
            .strip()
            .casefold()
            != "blocked"
        ):
            continue

        verified_at = str(
            raw_place.get("verified_at") or ""
        ).strip()
        source = str(
            raw_place.get("source") or ""
        ).strip()
        aliases = raw_place.get("aliases", [])
        queries = raw_place.get(
            "place_queries",
            [],
        )
        official_names = raw_place.get(
            "official_names",
            [],
        )

        try:
            date.fromisoformat(verified_at)
        except ValueError:
            continue

        if (
            not source.startswith("https://")
            or not isinstance(aliases, list)
            or not isinstance(queries, list)
            or not isinstance(official_names, list)
        ):
            continue

        prepared_aliases = [
            str(value).strip()[:120]
            for value in aliases
            if str(value).strip()
        ]
        prepared_queries = [
            str(value).strip()[:220]
            for value in queries
            if str(value).strip()
        ]
        prepared_official_names = [
            str(value).strip()[:300]
            for value in official_names
            if str(value).strip()
        ]

        if (
            not prepared_aliases
            and not prepared_queries
            and not prepared_official_names
        ):
            continue

        blocked.append(
            {
                "title": str(
                    raw_place.get("title") or ""
                ).strip()[:120],
                "aliases": prepared_aliases,
                "place_queries": prepared_queries,
                "official_names": prepared_official_names,
                "reason": str(
                    raw_place.get("reason") or ""
                ).strip()[:300],
                "verified_at": verified_at[:20],
                "source": source[:500],
            }
        )

    return blocked


def _cached_region_place_hints(region_scope: Any, *, limit: int) -> List[Dict[str, Any]]:
    """Выбирает реальные POI внутри исходного контура, сохраняя ключ запроса.

    Географические модули зависят от places, поэтому импорт отложен до вызова.
    Старые записи без типа объекта не доказывают, что это POI, а не база/город.
    Положительный кеш не имеет отдельного TTL; его имя, контекст и координаты
    повторно проверяются тем же matcher, что и в geocode. Промахи не обновляем.
    """
    from destination_scope import region_margin_km, _distance_from_bounds, _valid_geometry
    from rules import is_generic_attraction

    if limit < 1 or not _valid_geometry(getattr(region_scope, "geometry", None)):
        return []
    poi_types = {
        "museum", "gallery", "attraction", "viewpoint", "artwork", "zoo", "aquarium",
        "theme_park", "park", "garden", "nature_reserve", "beach", "peak", "volcano",
        "cave_entrance", "waterfall", "spring", "cliff", "memorial", "monument",
        "archaeological_site", "castle", "ruins", "manor", "fort", "palace",
        "monastery", "church", "cathedral", "place_of_worship", "theatre", "cinema",
    }
    cache = _load_cache()
    now = time.time()
    candidates = []
    for query, raw_place in cache.items():
        if (
            not isinstance(query, str)
            or query.endswith(("|geometry", "|miss"))
            or not query or len(query) > 220 or query != query.strip()
            or not query.isprintable()
            or not isinstance(raw_place, dict)
            or raw_place.get("reason") in {"mismatch", "not_found"}
            or _fresh_negative_cache(cache.get(query + "|miss"), now)
            or not _place_matches_query(query, raw_place)
        ):
            continue
        query_parts = [part.strip() for part in query.split(",") if part.strip()]
        if len(query_parts) < 3:
            continue
        title = query_parts[0]
        if len(title) > 120 or is_generic_attraction(title, query):
            continue
        if (
            str(raw_place.get("type") or "").casefold() not in poi_types
            or str(raw_place.get("category") or "").casefold() in {"place", "boundary"}
            or (region_scope.country and _country_identity(query_parts[-1]) != _country_identity(region_scope.country))
        ):
            continue
        lat, lon = float(raw_place["lat"]), float(raw_place["lon"])
        distance = _distance_from_bounds(lat, lon, region_scope)
        if not math.isfinite(distance) or distance > region_margin_km(region_scope):
            continue
        candidates.append((distance, query, {
            "title": title,
            "place_query": query,
            "lat": lat,
            "lon": lon,
            "official_name": str(raw_place["name"]).strip()[:220],
        }))

    hints: List[Dict[str, Any]] = []
    seen_queries: set[str] = set()
    for _, query, hint in sorted(candidates, key=lambda row: (row[0], row[1].casefold(), row[1])):
        key = _normalized_label(query)
        if key in seen_queries or any(
            same_cached_map_place(hint, other, require_verified=False)
            for other in hints
        ):
            continue
        hints.append(hint)
        seen_queries.add(key)
        if len(hints) >= limit:
            break
    return hints


def cached_place_hints(
    destination: str,
    center: Optional[tuple[float, float]],
    *,
    max_distance_km: float = 40.0,
    limit: int = 12,
    region_scope: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Возвращает короткий список ранее подтверждённых картой мест.

    Функция не обращается к сети и не считает промахи карты подсказками.
    Координаты защищают от попадания одноимённых мест из другого города.
    """

    if region_scope is not None:
        return _cached_region_place_hints(region_scope, limit=limit)

    if center is None or limit < 1:
        return []

    destination_parts = [
        part.strip()
        for part in str(destination or "").split(",")
        if part.strip()
    ]

    if not destination_parts:
        return []

    city_key = _normalized_label(
        destination_parts[0]
    )
    generic_labels = {
        "cinema",
        "museum",
        "park",
        "theatre",
        "zoo",
        "кинотеатр",
        "музей",
        "парк",
        "театр",
        "зоопарк",
        "центр",
        "центр города",
        "центральный район",
        "городской округ",
        "муниципальный округ",
        "район",
        "округ",
        "district",
    }
    candidates: List[
        tuple[
            float,
            str,
            str,
            float,
            float,
            str,
        ]
    ] = []
    seen_names: set[str] = set()

    for cache_key, raw_place in _load_cache().items():
        if not isinstance(raw_place, dict):
            continue
        if str(cache_key).endswith("|geometry"):
            continue
        if not _place_matches_query(str(cache_key), raw_place):
            continue

        query_parts = [
            part.strip()
            for part in str(cache_key).split(",")
            if part.strip()
        ]

        if len(query_parts) < 3:
            continue

        requested_name = _normalized_label(
            query_parts[0]
        )

        display_name = str(
            raw_place.get("name") or ""
        ).strip()

        if not display_name:
            continue

        haystack = _normalized_label(
            f"{cache_key} {display_name}"
        )

        if city_key and city_key not in haystack:
            continue

        try:
            lat = float(raw_place["lat"])
            lon = float(raw_place["lon"])
        except (KeyError, TypeError, ValueError):
            continue

        distance = distance_km(
            float(center[0]),
            float(center[1]),
            lat,
            lon,
        )

        if distance > max_distance_km:
            continue

        official_name = display_name.split(",", 1)[0].strip()
        official_key = _normalized_label(
            official_name
        )

        if (
            not official_key
            or requested_name != official_key
            or official_key in generic_labels
            or official_key in seen_names
        ):
            continue

        seen_names.add(official_key)
        place_query = ", ".join(
            [official_name]
            + destination_parts
        )
        candidates.append(
            (
                distance,
                official_name[:120],
                place_query[:220],
                lat,
                lon,
                display_name[:220],
            )
        )

    candidates.sort(
        key=lambda row: (
            row[0],
            _normalized_label(row[1]),
        )
    )

    return [
        {
            "title": title,
            "place_query": place_query,
            # Эти поля используются только локальным fallback.
            # _map_hints_block передаёт модели лишь title/query.
            "lat": lat,
            "lon": lon,
            "official_name": official_name,
        }
        for (
            _,
            title,
            place_query,
            lat,
            lon,
            official_name,
        ) in candidates[:limit]
    ]


def _valid_record_only_place(query: str, place: Any) -> bool:
    """Revalidate a server-owned record; model JSON is never a record source.

    This checks an already obtained map observation, not public accessibility,
    opening hours, indoor suitability or the truth of a model's metadata.
    """
    if not isinstance(query, str) or not query.isprintable() or query != query.strip() or len(query) > 220:
        return False
    parts = [part.strip() for part in query.split(",")]
    if len(parts) < 3 or not all(parts) or not isinstance(place, dict):
        return False
    if not isinstance(place.get("name"), str) or not place["name"].strip():
        return False
    if any(type(place.get(field)) not in (int, float)
           or not math.isfinite(place[field]) or not -bound <= place[field] <= bound
           for field, bound in (("lat", 90), ("lon", 180))):
        return False
    provider = place.get("provider", "")
    if not isinstance(provider, str) or provider not in ("", "osm", "2gis"):
        return False
    if provider == "2gis":
        if not verified_map_places.is_current_verified_map_record(place):
            return False
        exact = verified_map_places.exact_verified_map_place(query)
        if exact is None or map_record_identity(exact) != map_record_identity(place):
            return False
    elif place.get("provenance") or place.get("source_url"):
        # Independent or unrecognised provenance must never become OSM.
        return False
    return map_record_identity(place) is not None and _place_matches_query(query, place)


def verify_itinerary(
    itinerary: Itinerary, max_day_spread_km: float = MAX_DAY_SPREAD_KM, *,
    verified_records: Mapping[str, dict] | None = None,
) -> List[str]:
    """Находит координаты достопримечательностей и проверяет маршрут.

    Заполняет поля verified, lat, lon, official_name. Возвращает список замечаний на русском.
    Если передан server-owned verified_records, точный ключ place_query должен
    присутствовать в нём. Этот режим не вызывает геокодер, включая проверку
    района, и не использует app/model map-поля как доказательство.
    """
    if verified_records is not None and not isinstance(verified_records, Mapping):
        raise TypeError("verified_records должен быть отображением точных запросов на записи карты")
    problems: List[str] = []
    for day in itinerary.days:
        found = []
        for item in day.items:
            item.provider, item.source_url = "", ""
            item.verified = False
            item.lat, item.lon, item.official_name = None, None, ""
            if item.kind != "attraction":
                continue
            if not isinstance(item.place_query, str) or not item.place_query.strip():
                if verified_records is not None:
                    problems.append(f"День {day.day}: у места «{item.title}» отсутствует точный запрос подтверждённой записи карты.")
                continue
            try:
                if verified_records is None:
                    place = geocode(item.place_query)
                else:
                    place = verified_records.get(item.place_query)
                    if not _valid_record_only_place(item.place_query, place):
                        problems.append(
                            f"День {day.day}: место «{item.title}» не подтверждено подготовленной "
                            f"записью карты ({item.place_query}); повторный поиск не выполнялся."
                        )
                        continue
                    # The city candidate supplies an actual provider address.
                    # Generic labels and display-name-only areas are not proof.
                    if not _area_matches_map_context(item.area, {"address": place.get("address")}):
                        problems.append(
                            f"День {day.day}: район «{item.area}» у места «{item.title}» "
                            "не соответствует адресу подтверждённой записи карты; повторный поиск не выполнялся."
                        )
                        continue
            except GeocodeUnavailable:
                logger.warning("Карта недоступна, место «%s» не проверено", item.title)
                problems.append(
                    f"День {day.day}: карта временно недоступна, место «{item.title}» не проверено ({item.place_query}). Повторите проверку позже."
                )
                continue  # Сбой блокирует успех, но не объявляет место отсутствующим.
            except GeocodeMismatch:
                problems.append(
                    f"День {day.day}: результат карты не соответствует месту «{item.title}» ({item.place_query})"
                )
                continue
            if place is None:
                problems.append(f"День {day.day}: место «{item.title}» не найдено на карте ({item.place_query})")
                continue
            independent = place.get("provider") == "2gis" or place.get("provenance") == "official_linked_map"
            if independent and not verified_map_places.is_current_verified_map_record(place):
                problems.append(f"День {day.day}: источник координат места «{item.title}» требует повторной проверки.")
                continue
            item.verified = True
            item.lat, item.lon, item.official_name = place["lat"], place["lon"], place["name"]
            item.provider = "2gis" if place.get("provider") == "2gis" else "osm"
            item.source_url = (place["source_url"] if item.provider == "2gis" else
                               f"https://www.openstreetmap.org/?mlat={float(item.lat):.6f}&mlon={float(item.lon):.6f}"
                               f"#map=17/{float(item.lat):.6f}/{float(item.lon):.6f}")
            found.append(item)

            area_query = _area_search_query(
                item.area,
                item.place_query,
            )

            if (
                verified_records is None
                and area_query
                and not _area_matches_map_context(item.area, place)
            ):
                try:
                    area_place = geocode(area_query)
                except GeocodeUnavailable:
                    area_place = None
                    logger.warning(
                        "Карта недоступна при проверке района «%s»",
                        item.area,
                    )
                    problems.append(
                        f"День {day.day}: карта временно недоступна, район «{item.area}» у места «{item.title}» не проверен ({area_query}). Повторите проверку позже."
                    )
                except GeocodeMismatch:
                    area_place = None
                    problems.append(
                        f"День {day.day}: район «{item.area}» у места «{item.title}» не подтверждён: карта нашла другой объект ({area_query}). Уточните название района."
                    )
                else:
                    if area_place is None:
                        problems.append(
                            f"День {day.day}: район «{item.area}» у места «{item.title}» не найден на карте ({area_query})."
                        )

                if area_place:
                    area_km = distance_km(
                        item.lat,
                        item.lon,
                        area_place["lat"],
                        area_place["lon"],
                    )

                    if area_km > MAX_AREA_DISTANCE_KM:
                        problems.append(
                            f"День {day.day}: место "
                            f"«{item.title}» находится примерно "
                            f"в {area_km:.1f} км от указанного "
                            f"района «{item.area}». Выбери место "
                            "в этом районе или согласованно "
                            "измени тему и районы всех пунктов "
                            "целевого дня. "
                            f"place_query «{item.place_query}»"
                        )

        for index, first in enumerate(found):
            for second in found[index + 1:]:
                km = distance_km(first.lat, first.lon, second.lat, second.lon)
                if km > max_day_spread_km:
                    problems.append(
                        f"День {day.day}: «{first.title}» и «{second.title}» далеко друг от друга ({km:.1f} км)"
                    )
    return problems
