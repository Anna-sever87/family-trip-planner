"""Local source-backed names, never coordinates or proof of visitability.

The catalogue is an input to candidate selection only. Every selected name
still requires independent map identity and regional geometry verification.
"""

from __future__ import annotations

import json
from pathlib import Path
import unicodedata
from urllib.parse import urlsplit
from typing import Any


NAME_HINTS_FILE = Path(__file__).resolve().parent / "data" / "destination_name_hints.json"
MAX_NAME_HINTS = 24


def _alias(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 240 or not value.isprintable():
        return ""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().replace("ё", "е").split())


def _text(value: Any, maximum: int) -> bool:
    return isinstance(value, str) and 0 < len(value) <= maximum and value == value.strip() and value.isprintable()


def _https_source(value: Any) -> bool:
    if not _text(value, 2048) or any(char.isspace() for char in value):
        return False
    try:
        parsed = urlsplit(value)
        return (parsed.scheme == "https" and bool(parsed.hostname)
                and parsed.username is None and parsed.password is None
                and (parsed.port is None or 1 <= parsed.port <= 65535))
    except ValueError:
        return False


def destination_name_hints(destination: str, *, limit: int = MAX_NAME_HINTS) -> list[dict[str, str]]:
    """Return at most 24 whitelisted names for an exact normalized region alias.

    No network, imports from the coordinate catalogue, or persistent writes.
    Missing/malformed files and invalid records produce no trusted fallback.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("Лимит подсказок должен быть неотрицательным целым числом")
    maximum = min(limit, MAX_NAME_HINTS)
    requested = _alias(destination)
    if not requested or not maximum:
        return []
    try:
        payload = json.loads(NAME_HINTS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError):
        return []
    if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(payload.get("regions"), list):
        return []
    results: list[dict[str, str]] = []
    seen = set()
    for region in payload["regions"]:
        if not isinstance(region, dict) or not isinstance(region.get("aliases"), list):
            continue
        if requested not in {_alias(alias) for alias in region["aliases"]}:
            continue
        entries = region.get("places")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            title, query, locality, source = (entry.get(key) for key in ("title", "place_query", "locality", "source_url"))
            if not (_text(title, 80) and _text(query, 140) and _text(locality, 120) and _https_source(source)):
                continue
            parts = [part.strip() for part in query.split(",")]
            if len(parts) < 3 or not all(parts) or title != parts[0] or locality != parts[-2]:
                continue
            base_query = entry.get("base_query")
            if base_query is not None:
                if not _text(base_query, 120):
                    continue
                base_parts = [part.strip() for part in base_query.split(",")]
                if len(base_parts) < 2 or not all(base_parts):
                    continue
            key = unicodedata.normalize("NFKC", query).casefold()
            if key in seen:
                continue
            seen.add(key)
            hint = {"title": title, "place_query": query, "locality": locality, "source_url": source}
            if base_query is not None:
                hint["base_query"] = base_query
            results.append(hint)
            if len(results) == maximum:
                return results
    return results
