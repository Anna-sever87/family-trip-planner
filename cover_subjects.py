"""Offline cover subjects from server-owned, request-bound map snapshots.

Map verification proves identity, city and category, not iconic status, facade
details or the pixels that an image model will return. No discovery/API calls.
"""

import json
import re

import city_context
from cover_prompts import CoverSubject, SUBJECT_KINDS, safe_subject_title
from schemas import Itinerary, TripRequest


_PRIORITY = {kind: rank for rank, kind in enumerate((
    "castle", "palace", "cathedral", "monastery", "monument", "memorial",
    "church", "place_of_worship", "museum", "gallery", "theatre", "attraction",
))}
# A generic tourism tag is not evidence that a minor ride is a city landmark.
# Named observation wheels remain eligible as visits, not as claimed city icons.
_MINOR_RIDE_TYPES = frozenset({
    "roller_coaster", "carousel", "swing_carousel", "drop_tower", "water_slide",
    "bumper_car", "bumper_cars", "miniature_train", "train", "dark_ride", "boat_ride",
})


def select_saved_cover_subject(request_json, itinerary_json, *, destination, country,
                               start_date, end_date):
    """Choose one interesting actual visit; malformed/legacy data fail closed.

    Inputs must be the owner-scoped stored JSON, never HTTP request fields.
    Only actual visits are eligible. Their observed locality remains explicit:
    a nearby settlement is never relabelled as the trip's destination city.
    Category priority is visual suitability, not a claim of fame or popularity.
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
            tags = candidate.record.get("extratags", {})
            if kind == "attraction" and tags.get("attraction") in _MINOR_RIDE_TYPES:
                continue
            address = candidate.record.get("address", {})
            # The canonical record already binds locality to the provider's
            # address/query, and checks country/identity/radius. Radius alone
            # does not establish membership in the destination municipality.
            locality = candidate.area
            local_country = address.get("country", "")
            country_code = address.get("country_code", "")
            if (not local_country or not isinstance(country_code, str)
                    or country_code.lower() != center["country_code"]):
                continue
            try:
                title = safe_subject_title(candidate.title)
            except ValueError:
                continue
            # Only an exact alias from this already checked map record may
            # explain a foreign name in the image prompt and frozen caption.
            russian_title = ""
            details = candidate.record.get("namedetails", {})
            alias = details.get("name:ru") if isinstance(details, dict) else None
            if isinstance(alias, str) and re.search(r"[А-Яа-яЁё]", alias):
                try:
                    russian_title = safe_subject_title(alias)
                except ValueError:
                    pass
            eligible.append((_PRIORITY[kind], title.casefold(), candidate.candidate_id,
                             CoverSubject(title, kind, candidate.candidate_id, destination, country,
                                          locality, local_country, country_code.lower(), russian_title)))
        return min(eligible, key=lambda entry: entry[:3])[3] if eligible else None
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError):
        return None
