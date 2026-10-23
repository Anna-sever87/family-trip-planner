"""Offline cover subjects from server-owned, request-bound map snapshots.

Map verification proves identity, city and category, not iconic status, facade
details or the pixels that an image model will return. No discovery/API calls.
"""

import json

import city_context
from cover_prompts import CoverSubject, SUBJECT_KINDS, safe_subject_title
import places
from schemas import Itinerary, TripRequest


_PRIORITY = {kind: rank for rank, kind in enumerate((
    "castle", "palace", "cathedral", "monastery", "theatre", "monument",
    "memorial", "church", "place_of_worship",
))}


def select_saved_cover_subject(request_json, itinerary_json, *, destination, country,
                               start_date, end_date):
    """Choose one interesting actual visit; malformed/legacy data fail closed.

    Inputs must be the owner-scoped stored JSON, never HTTP request fields.
    A valid but unused candidate, generic museum, mall or road is not enough.
    """
    try:
        if (not isinstance(request_json, str) or len(request_json) > 250_000
                or not isinstance(itinerary_json, str) or len(itinerary_json) > 1_000_000):
            return None
        envelope = json.loads(request_json)
        request = TripRequest.model_validate(envelope)
        parts = [part.strip() for part in request.destination.rsplit(",", 1)]
        if (parts != [destination, country] or request.start_date != start_date
                or request.end_date != end_date):
            return None
        pool, center = city_context.load_city_context(envelope.get(city_context.CONTEXT_KEY), request)
        itinerary = Itinerary.model_validate(json.loads(itinerary_json))
        city_context.check_context_places(itinerary, pool)
        visits = {item.place_query for day in itinerary.days for item in day.items
                  if item.kind == "attraction"}
        eligible = []
        for candidate in pool:
            kind = candidate.record.get("type")
            if kind not in SUBJECT_KINDS or candidate.place_query not in visits:
                continue
            address = candidate.record.get("address", {})
            # The route pool can include nearby settlements (up to 40 km).
            # Never turn their landmark into an illustration of the named city.
            locality = next((address.get(key) for key in ("city", "town", "village", "municipality")
                             if address.get(key)), "")
            if places._context_identity(locality) != places._context_identity(center["name"]):
                continue
            try:
                title = safe_subject_title(candidate.title)
            except ValueError:
                continue
            eligible.append((_PRIORITY[kind], title.casefold(), candidate.candidate_id,
                             CoverSubject(title, kind, candidate.candidate_id, destination, country)))
        return min(eligible, key=lambda entry: entry[:3])[3] if eligible else None
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        return None
