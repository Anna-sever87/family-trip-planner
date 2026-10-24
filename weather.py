"""Погода для поездки через Open-Meteo: прогноз, а для дальних дат ориентир по прошлому году."""
from datetime import date, timedelta
from dataclasses import dataclass, field
import math
import time
import unicodedata
from typing import Any, Dict, Optional

import httpx

USER_AGENT = "family-trip-planner/0.1 (student project)"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum"
FORECAST_LIMIT_DAYS = 15      # дальше этого срока точного прогноза нет
RAIN_MM = 1.0                 # от этого количества осадков день считаем дождливым
MAX_CITY_RESULTS = 10
MAX_HTTP_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 0.25
MAX_RETRY_AFTER_SECONDS = 2.0
TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})
UNKNOWN_WEATHER_NOTE = (
    "Погода неизвестна: достоверные данные на даты поездки не получены. "
    "Отсутствие данных не означает отсутствие дождя; перед поездкой проверьте прогноз."
)


def safe_city_suggestions(value) -> tuple[str, ...]:
    """Small plain-text geographic labels, never arbitrary provider messages."""
    if not isinstance(value, (list, tuple)):
        return ()
    result = []
    for label in value[:MAX_CITY_RESULTS]:
        if not isinstance(label, str) or not 1 <= len(label) <= 100 or not label.isprintable():
            continue
        parts = label.split(",")
        if len(parts) != 3:
            continue
        parts = [part.strip() for part in parts]
        if any(not part or not any(char.isalnum() for char in part) or any(
                not (unicodedata.category(char)[0] in "LMN" or char == " " or char in "-'’().–—")
                for char in part) for part in parts):
            continue
        clean = ", ".join(parts)
        if clean not in result and len(clean) <= 100:
            result.append(clean)
        if len(result) == 5:
            break
    return tuple(result)


class CityResolutionError(ValueError):
    """A destination is not sufficiently unambiguous to anchor a city trip."""

    def __init__(self, destination: str, *, reason: str, suggestions=()):
        self.reason = reason
        self.suggestions = safe_city_suggestions(suggestions) if reason == "ambiguous" else ()
        explanation = {
            "ambiguous": "Найдено несколько одноимённых населённых пунктов.",
            "not_found": "Указанный город не найден в географическом справочнике.",
            "unavailable": "Сервис определения города временно недоступен. Попробуйте позже.",
            "search_limit": "Поиск вернул ограниченную выборку; однозначность города не подтверждена.",
            "invalid_response": "Сервис определения города вернул неподтверждённые данные.",
        }.get(reason, "Не удалось однозначно определить город.")
        super().__init__(
            explanation + " Уточните город, область/провинцию и страну; "
            "другой одноимённый город не подставляется."
            + ("\nВарианты для уточнения:\n" + "\n".join(self.suggestions) if self.suggestions else "")
        )


def _get_json(url: str, **params) -> Dict[str, Any]:
    """At most one retry for a transient transport/HTTP failure, never bad JSON.

    A long Retry-After ends this bounded operation rather than retrying before
    the provider permits. No persistent weather cache or stale data fallback.
    """
    for attempt in range(MAX_HTTP_ATTEMPTS):
        delay = RETRY_DELAY_SECONDS
        try:
            response = httpx.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            if attempt + 1 >= MAX_HTTP_ATTEMPTS or error.response.status_code not in TRANSIENT_STATUS_CODES:
                raise
            retry_after = error.response.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    delay = float(retry_after)
                except (TypeError, ValueError):
                    raise error  # An HTTP-date is not guessed or ignored.
                if not math.isfinite(delay) or not 0 <= delay <= MAX_RETRY_AFTER_SECONDS:
                    raise error
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
            if attempt + 1 >= MAX_HTTP_ATTEMPTS:
                raise
        else:
            return response.json()
        time.sleep(delay)
    raise RuntimeError("Weather request bound exhausted")


def find_city(destination: str) -> Optional[Dict[str, Any]]:
    """Resolve city[, admin1], country without discarding geographic context.

    A bare city remains supported, but only an unambiguous exact-name result
    is usable. Provider ranking/population never chooses among homonyms.
    This is a bounded read-only search, not a catalogue of city coordinates.
    """
    from places import _context_identity, _country_identity

    if not isinstance(destination, str) or not destination.strip() or not destination.isprintable() or len(destination) > 240:
        raise CityResolutionError(str(destination), reason="invalid_context")
    parts = [part.strip() for part in destination.split(",")]
    if not 1 <= len(parts) <= 3 or any(not part for part in parts):
        raise CityResolutionError(destination, reason="invalid_context")
    city_name = parts[0]
    country = parts[-1] if len(parts) > 1 else ""
    admin = parts[1] if len(parts) == 3 else ""
    country_identity = _country_identity(country) if country else ()
    code = (country_identity[0] if len(country_identity) == 1
            and len(country_identity[0]) == 2 and country_identity[0].isascii()
            and country_identity[0].isalpha() else "")
    language = "ru" if any("а" <= char.casefold() <= "я" or char.casefold() == "ё" for char in city_name) else "en"
    # The provider accepts one comma qualifier. Known countries have their own
    # ISO filter, so the qualifier can retain the explicitly supplied admin1.
    query = f"{city_name}, {admin or country}" if admin or (country and not code) else city_name
    params = dict(name=query, count=MAX_CITY_RESULTS, language=language)
    if code:
        params["countryCode"] = code.upper()
    data = _get_json(GEOCODING_URL, **params)
    if not isinstance(data, dict):
        raise CityResolutionError(destination, reason="invalid_response")
    results = data.get("results")
    if not results:
        return None
    if not isinstance(results, list):
        raise CityResolutionError(destination, reason="invalid_response")
    matches = {}
    for found in results:
        if not isinstance(found, dict) or _context_identity(found.get("name", "")) != _context_identity(city_name):
            continue
        lat, lon = found.get("latitude"), found.get("longitude")
        if (type(lat) not in (int, float) or type(lon) not in (int, float)
                or not math.isfinite(lat) or not math.isfinite(lon)
                or not -90 <= lat <= 90 or not -180 <= lon <= 180):
            continue
        feature = found.get("feature_code", "")
        if feature and (not isinstance(feature, str) or not feature.startswith("PPL")):
            continue
        actual_code = str(found.get("country_code", "")).strip().casefold()
        if country:
            if code and actual_code:
                if actual_code != code:
                    continue
            elif _country_identity(found.get("country", "")) != country_identity:
                continue
        if admin and _context_identity(found.get("admin1", "")) != _context_identity(admin):
            continue
        # Repeated copies of one provider result are not different cities;
        # conflicting coordinates still remain an ambiguity, even for one ID.
        key = (str(found.get("id", "")), float(lat), float(lon), actual_code,
               _context_identity(found.get("admin1", "")))
        matches[key] = found
    if not matches:
        raise CityResolutionError(destination, reason="context_mismatch")
    if len(matches) > 1:
        labels = {}
        for found in matches.values():
            parts = [found.get(key) for key in ("name", "admin1", "country")]
            if not all(isinstance(part, str) and part.strip() for part in parts):
                continue
            suggestions = safe_city_suggestions([", ".join(parts)])
            if suggestions:
                label = suggestions[0]
                # Same-admin homonyms cannot be resolved by this label alone.
                identity = tuple(_context_identity(part) for part in label.split(","))
                labels.setdefault(identity, []).append(label)
        raise CityResolutionError(destination, reason="ambiguous", suggestions=[
            values[0] for values in labels.values() if len(values) == 1
        ])
    if len(results) >= MAX_CITY_RESULTS:
        raise CityResolutionError(destination, reason="search_limit")
    found = next(iter(matches.values()))
    return {"lat": found["latitude"], "lon": found["longitude"], "name": found["name"],
            "country": found.get("country", ""),
            "country_code": found.get("country_code", ""), "admin1": found.get("admin1", ""),
            "geoname_id": found.get("id")}


_CITY_CONTEXT_AUTHORITY = object()


@dataclass(frozen=True, slots=True)
class ResolvedPlanningCity:
    """Opaque in-process preflight snapshot; never a client/persisted payload.

    The immutable proof binds the exact snapshot and destination. Replacing
    fields or supplying a dict cannot turn unverified coordinates into a city.
    """
    _destination: str
    _record: tuple
    _proof: tuple = field(repr=False, compare=False)


def resolve_city_for_planning(destination: str) -> ResolvedPlanningCity:
    """Free preflight, before any memory embedding, quota claim or generation."""
    try:
        city = find_city(destination)
    except CityResolutionError:
        raise
    except httpx.HTTPError as error:
        raise CityResolutionError(destination, reason="unavailable") from error
    except (TypeError, ValueError) as error:
        raise CityResolutionError(destination, reason="invalid_response") from error
    if city is None:
        raise CityResolutionError(destination, reason="not_found")
    # find_city already checked provider identity, country, admin and bounds.
    # Keep only scalar fields used downstream, never raw provider metadata.
    keys = ("lat", "lon", "name", "country", "country_code", "admin1", "geoname_id")
    record = tuple((key, city.get(key)) for key in keys)
    if (any(type(city.get(key)) not in (int, float) or not math.isfinite(city[key])
            or not -limit <= city[key] <= limit for key, limit in (("lat", 90), ("lon", 180)))
            or not isinstance(city.get("name"), str) or not city["name"].strip()
            or any(value is not None and type(value) not in (str, int, float) for _, value in record)):
        raise CityResolutionError(destination, reason="invalid_response")
    bound = destination.strip()
    return ResolvedPlanningCity(bound, record, (_CITY_CONTEXT_AUTHORITY, bound, record))


def validated_planning_city(context: ResolvedPlanningCity, destination: str) -> Dict[str, Any]:
    """Return a fresh dict only for the original server-resolved destination."""
    if (type(context) is not ResolvedPlanningCity or not isinstance(destination, str)
            or type(context._proof) is not tuple or len(context._proof) != 3
            or context._proof[0] is not _CITY_CONTEXT_AUTHORITY
            or context._proof[1] != context._destination
            or context._proof[2] is not context._record
            or context._destination != destination.strip()):
        raise CityResolutionError(destination, reason="invalid_context")
    return dict(context._record)


def _years_back(day: date, years: int = 1) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:                   # 29 февраля
        return day.replace(year=day.year - years, day=28)


def _validated_daily(data: Any, start: date, end: date) -> Dict[str, Dict[str, Any]]:
    """Validate the exact requested provider calendar before mapping trip dates."""
    if not isinstance(data, dict) or not isinstance(data.get("daily"), dict):
        raise ValueError("Некорректная структура погоды")
    daily = data["daily"]
    expected = [(start + timedelta(days=index)).isoformat() for index in range((end - start).days + 1)]
    fields = ("time", "temperature_2m_min", "temperature_2m_max", "precipitation_sum")
    if (not expected or any(not isinstance(daily.get(field), list) or len(daily[field]) != len(expected)
                            for field in fields) or daily["time"] != expected):
        raise ValueError("Даты или длины рядов погоды не соответствуют запросу")
    result = {}
    for index, provider_day in enumerate(expected):
        minimum, maximum, rain = (daily[field][index] for field in fields[1:])
        if (any(type(value) not in (int, float) or not math.isfinite(value)
                for value in (minimum, maximum, rain)) or minimum > maximum or rain < 0):
            raise ValueError("Неполные или некорректные числовые значения погоды")
        result[provider_day] = {"tmin": minimum, "tmax": maximum, "rain_mm": rain}
    return result


def get_weather(destination: str, start_date: str, end_date: str, *,
                resolved_city: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Возвращает погоду по дням поездки или None, если получить не удалось.

    source: forecast (прогноз), fact (прошедшие даты) или history (те же даты прошлого года).
    """
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
        if end < start:
            return None
        # Server-owned resolution can be reused by candidate-first planning;
        # weather and route geography must not resolve different homonyms.
        city = resolved_city if resolved_city is not None else find_city(destination)
        if city is None:
            return None
        if not isinstance(city, dict) or any(type(city.get(key)) not in (int, float) or not math.isfinite(city[key])
               or not low <= city[key] <= high for key, low, high in
               (("lat", -90, 90), ("lon", -180, 180))):
            raise CityResolutionError(destination, reason="invalid_coordinates")
        today = date.today()
        base = {"latitude": city["lat"], "longitude": city["lon"], "daily": DAILY, "timezone": "auto"}
        if end < today:
            source = "fact"
            provider_start, provider_end, url = start, end, ARCHIVE_URL
        elif end <= today + timedelta(days=FORECAST_LIMIT_DAYS):
            source = "forecast"
            provider_start, provider_end, url = max(start, today), end, FORECAST_URL
        else:
            source = "history"
            provider_start, provider_end, url = _years_back(start), _years_back(end), ARCHIVE_URL
        data = _get_json(url, start_date=provider_start.isoformat(), end_date=provider_end.isoformat(), **base)
        daily = _validated_daily(data, provider_start, provider_end)
        trip_days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        if source == "forecast":
            trip_days = [d for d in trip_days if d >= today]
        days = []
        for trip_day in trip_days:
            # Historical evidence is matched by month/day, never array index:
            # a provider leap day cannot shift March. The existing explicit
            # analogue for a target Feb 29 in a non-leap prior year is Feb 28.
            provider_day = _years_back(trip_day) if source == "history" else trip_day
            days.append({"date": trip_day.isoformat(), **daily[provider_day.isoformat()]})
        return {"city": city["name"], "source": source, "days": days}
    except CityResolutionError:
        raise  # A wrong/ambiguous city must not silently disable city geography.
    except (httpx.HTTPError, KeyError, ValueError, IndexError, TypeError, OverflowError):
        return None


def weather_summary(weather: Optional[Dict[str, Any]]) -> str:
    """Текст для промпта и для показа пользователю."""
    if not weather or not weather["days"]:
        return UNKNOWN_WEATHER_NOTE
    titles = {
        "forecast": "Прогноз погоды",
        "fact": "Фактическая погода",
        "history": "Прогноза на эти даты ещё нет. Ориентир: погода в эти же даты прошлого года",
    }
    lines = [titles[weather["source"]] + f" ({weather['city']}):"]
    for day in weather["days"]:
        rain = "дождливый день" if day["rain_mm"] >= RAIN_MM else "без заметных осадков"
        lines.append(f"{day['date']}: от {day['tmin']:g} до {day['tmax']:g} °C, осадки {day['rain_mm']:g} мм ({rain})")
    return "\n".join(lines)
