"""Small, manually reviewed map facts; no search, HTTP, or OSM-cache writes.

These are independent provider records, not OpenStreetMap observations. Keep
their attribution links when presenting them. Public hosting/bulk reuse needs
a separate review of the provider's terms; this catalogue grants no licence.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import Any
from urllib.parse import urlsplit


CATALOG_FILE = Path(__file__).resolve().parent / "data" / "verified_map_places.json"
MAX_PLACES = 32
MAX_AGE_DAYS = 90
MAX_BYTES = 100_000
PROVENANCE_FIELDS = (
    "provider", "provider_id", "provenance", "source_url", "official_url",
    "official_link_url", "checked_at",
)
_ENTRY_FIELDS = {
    "title", "place_queries", "lat", "lon", "address", "type",
    "provider", "provider_id", "source_url", "official_url", "official_link_url", "checked_at",
}
_ADDRESS_FIELDS = {"road", "house_number", "city", "county", "state", "country", "country_code"}


def _label(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _text(value: Any, maximum: int, *, commas: bool = False) -> str:
    if (not isinstance(value, str) or not 1 <= len(value) <= maximum
            or value != value.strip() or not value.isprintable()
            or (not commas and "," in value)):
        raise ValueError("Invalid catalogue text")
    return value


def _https(value: Any) -> str:
    url = _text(value, 600, commas=True)
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname or "." not in parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.port is not None or parsed.fragment or "\\" in url
            or any(char.isspace() for char in url)):
        raise ValueError("Invalid public HTTPS source")
    return url


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate catalogue key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _record(raw: Any, today: date) -> dict[str, Any]:
    if not isinstance(raw, dict) or set(raw) != _ENTRY_FIELDS:
        raise ValueError("Invalid catalogue entry fields")
    title = _text(raw["title"], 120)
    provider_id = _text(raw["provider_id"], 32)
    if raw["provider"] != "2gis" or not re.fullmatch(r"[1-9][0-9]{0,31}", provider_id):
        raise ValueError("Invalid provider identity")
    if raw["type"] != "museum":
        raise ValueError("Unreviewed provider POI type")
    checked_at = _text(raw["checked_at"], 10)
    checked = date.fromisoformat(checked_at)
    if checked.isoformat() != checked_at or not 0 <= (today - checked).days <= MAX_AGE_DAYS:
        raise ValueError("Map evidence is expired or future dated")
    coordinates = []
    for key, maximum in (("lat", 90), ("lon", 180)):
        value = raw[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not -maximum <= value <= maximum):
            raise ValueError("Invalid map coordinates")
        coordinates.append(float(value))
    address = raw["address"]
    if not isinstance(address, dict) or set(address) != _ADDRESS_FIELDS:
        raise ValueError("Incomplete map address")
    address = {key: _text(value, 180) for key, value in address.items()}
    if not re.fullmatch(r"[a-z]{2}", address["country_code"]):
        raise ValueError("Invalid country code")
    # This reviewed catalogue currently has only Russian sources. Adding a new
    # jurisdiction requires an explicit country-name/code mapping, not a guess.
    if address["country_code"] != "ru" or address["country"] != "Россия":
        raise ValueError("Unreviewed country mapping")
    queries = raw["place_queries"]
    if not isinstance(queries, list) or not 1 <= len(queries) <= 8:
        raise ValueError("Invalid exact queries")
    prepared_queries = []
    for query in queries:
        query = _text(query, 220, commas=True)
        parts = [part.strip() for part in query.split(",")]
        if (len(parts) != 3 or parts[0] != title
                or parts[1] not in {address["city"], address["county"]}
                or parts[2] != address["country"]):
            raise ValueError("Query contradicts the named map record")
        prepared_queries.append(query)
    if len({_label(query) for query in prepared_queries}) != len(prepared_queries):
        raise ValueError("Duplicate exact query")
    source_url = _https(raw["source_url"])
    official_url = _https(raw["official_url"])
    official_link_url = _https(raw["official_link_url"])
    source = urlsplit(source_url)
    if (source.hostname != "2gis.ru" or source.query
            or not re.fullmatch(r"/[a-z-]+/firm/" + re.escape(provider_id), source.path)):
        raise ValueError("Provider URL does not identify this organisation")
    if (urlsplit(official_url).hostname in {"2gis.ru", "go.2gis.com"}
            or urlsplit(official_link_url).hostname != "go.2gis.com"
            or not re.fullmatch(r"/[A-Za-z0-9]+", urlsplit(official_link_url).path)
            or urlsplit(official_link_url).query):
        raise ValueError("Missing independent official-link provenance")
    return {
        "name": ", ".join((title, address["road"], address["house_number"],
                           address["city"], address["county"], address["state"], address["country"])),
        "namedetails": {"name": title}, "address": address,
        "lat": coordinates[0], "lon": coordinates[1],
        "category": "tourism", "type": "museum", "addresstype": "museum",
        "provider": "2gis", "provider_id": provider_id,
        "provenance": "official_linked_map", "source_url": source_url,
        "official_url": official_url, "official_link_url": official_link_url,
        "checked_at": checked_at, "catalog_queries": prepared_queries,
    }


def load_verified_map_places(*, path: Path | None = None, today: date | None = None) -> list[dict[str, Any]]:
    """Read fresh local facts on every call; stale facts cannot survive a cache.

Malformed individual entries are unusable. An oversized/ambiguous catalogue
is rejected entirely. Nothing is inferred from a model or from address text.
"""
    source = CATALOG_FILE if path is None else Path(path)
    current = date.today() if today is None else today
    try:
        if source.stat().st_size > MAX_BYTES:
            return []
        payload = json.loads(source.read_text(encoding="utf-8"),
                             object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if (not isinstance(payload, dict) or set(payload) != {"version", "places"}
                or type(payload["version"]) is not int or payload["version"] != 1
                or not isinstance(payload["places"], list) or len(payload["places"]) > MAX_PLACES):
            return []
        result, seen_queries, seen_ids = [], set(), set()
        for raw in payload["places"]:
            try:
                record = _record(raw, current)
            except (TypeError, ValueError, OverflowError):
                continue
            keys = {_label(query) for query in record["catalog_queries"]}
            if seen_queries & keys or record["provider_id"] in seen_ids:
                return []
            seen_queries.update(keys)
            seen_ids.add(record["provider_id"])
            result.append(record)
        return result
    except (OSError, UnicodeError, TypeError, ValueError, OverflowError, RecursionError):
        return []


def exact_verified_map_place(query: str) -> dict[str, Any] | None:
    """Only explicit complete catalogue queries, not an address-only fallback."""
    if not isinstance(query, str) or not query.isprintable() or len(query) > 220:
        return None
    wanted = _label(query)
    matches = [record for record in load_verified_map_places()
               if wanted in {_label(value) for value in record["catalog_queries"]}]
    return deepcopy(matches[0]) if len(matches) == 1 else None


def same_title_verified_map_places(title: str, country: str) -> list[dict[str, Any]]:
    """Candidates only: the caller must still check radius, region and base."""
    if not isinstance(title, str) or not isinstance(country, str):
        return []
    return [record for record in load_verified_map_places()
            if _label(title) == _label(record["namedetails"]["name"])
            and _label(country) in {_label(record["address"]["country"]), "ru", "russia", "российская федерация"}]


def is_current_verified_map_record(record: Any) -> bool:
    """Provenance markers alone cannot turn arbitrary/cache data into evidence."""
    if not isinstance(record, dict) or any(key in record for key in ("osm_type", "osm_id", "geojson")):
        return False
    return any(all(record.get(key) == value for key, value in trusted.items())
               for trusted in load_verified_map_places())


def item_map_attribution(item: Any) -> dict[str, str]:
    """Safe public attribution for saved app data; no network or DB access.

    An independent-provider organisation URL needs the same exact current query,
    coordinates and URL. Expired/forged 2GIS data never turns into an OSM source.
    Legacy app items without provenance predate independent map support and keep
    the previous OSM coordinate-view link, not an inferred OSM organisation id.
    """
    get = item.get if isinstance(item, dict) else lambda key, default=None: getattr(item, key, default)
    empty = {"provider": "", "source_url": ""}
    if get("kind", "attraction") != "attraction" or get("verified") is not True:
        return empty
    lat, lon = get("lat"), get("lon")
    if any(type(value) not in (int, float) or not -bound <= value <= bound or not math.isfinite(value)
           for value, bound in ((lat, 90), (lon, 180))):
        return empty
    provider = get("provider", "")
    if provider == "2gis":
        record = exact_verified_map_place(get("place_query", ""))
        if (record is not None and lat == record["lat"] and lon == record["lon"]
                and get("source_url") == record["source_url"]):
            return {"provider": "2gis", "source_url": record["source_url"]}
        return empty
    if provider not in ("", "osm") or (provider == "" and get("source_url", "")):
        return empty
    return {"provider": "osm", "source_url": (
        f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=17/{lat:.6f}/{lon:.6f}"
    )}


def is_public_map_point(item: Any) -> bool:
    """One read-only publication gate for web, PDF and Telegram.

    Attribution already validates the item kind, strict verification flag,
    coordinates and current provider evidence, including legacy OSM support.
    A saved ``verified`` flag alone cannot authorize a public map point.
    """
    attribution = item_map_attribution(item)
    return attribution["provider"] in {"osm", "2gis"} and bool(attribution["source_url"])
