"""Conservative city planning policy and separately reviewed visit estimates.

A published typical duration is NOT a physical maximum, opening window,
proof of indoor access or a guaranteed visit. Its upper end is used here as
a conservative product planning cap. Call only with server-owned candidates
AFTER grounding; ordinary map/currentness and itinerary checks remain required.
No network, catalogue/cache reads, persistence or itinerary mutation occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

from places import _country_identity, map_record_identity


@dataclass(frozen=True)
class PublishedVisitEstimate:
    typical_min_minutes: int
    typical_max_minutes: int
    planning_cap_minutes: int
    source_url: str
    checked_at: str

    def as_prompt_data(self) -> dict:
        """A fresh data-only view; the model cannot create trusted estimates."""
        return {
            "basis": "published_typical_duration",
            "typical_min_minutes": self.typical_min_minutes,
            "typical_max_minutes": self.typical_max_minutes,
            "planning_cap_minutes": self.planning_cap_minutes,
            "source_url": self.source_url,
            "checked_at": self.checked_at,
            "note": "Типичная длительность из источника; предел планирования — правило продукта, "
                    "не физический максимум, не подтверждение часов, доступа или помещения.",
        }


# Municipal Nha Trang page, section "Tham quan ... mất bao lâu?", reviewed
# 2026-10-21: typical visit 30–45 minutes. No extrapolation to 90/180 minutes.
# The institution's other visit conditions are deliberately NOT encoded here.
_YERSIN_ESTIMATE = PublishedVisitEstimate(
    typical_min_minutes=30, typical_max_minutes=45, planning_cap_minutes=45,
    source_url="https://nhatrang.khanhhoa.gov.vn/tin-bai/diem-den/bao-tang-alexandre-yersin-nha-trang-1625",
    checked_at="2026-10-21",
)
_ADDRESS_LOCALITIES = (
    "city", "town", "village", "hamlet", "suburb", "neighbourhood", "quarter",
    "locality", "municipality", "county",
)
_SHORT_POLICY_TYPES = frozenset({"place_of_worship", "artwork", "monument", "memorial"})


def _label(value) -> str:
    if not isinstance(value, str) or not value or len(value) > 240 or not value.isprintable():
        return ""
    # Keep accents, word order and punctuation. No substring/global aliases.
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


_YERSIN_NAMES = frozenset(map(_label, ("Bảo tàng Alexandre Yersin", "Bảo tàng Yersin")))
# Observed OSM spelling, not a spelling asserted by the municipal source:
# output/diagnostics/nha-trang-children4-10-days5-policy4/city-candidates.json
# records way/1193057654 as "Bào Tàng Alexandre Yersin" on 2026-10-21.
# Keep this one exact provider alias separate; never fold accents globally.
# The existing VN + Nha Trang + canonical query/title + provider ID binding
# still applies before either source or observed-provider names are consulted.
_YERSIN_PROVIDER_NAMES = frozenset(map(_label, ("Bào Tàng Alexandre Yersin",)))
_NHA_TRANG = frozenset(map(_label, ("Nha Trang", "Phường Nha Trang")))


def lookup_city_visit_estimate(candidate) -> PublishedVisitEstimate | None:
    """Bind an estimate to an exact already-grounded name/city/country.

    The source's native name may be a provider-owned name:vi when the canonical
    display name is international. Model descriptions, duration fields and
    arbitrary estimate/source metadata are never consulted. This binding is
    not an alternative map verifier and cannot elevate a forged model record.
    """
    title = getattr(candidate, "title", None)
    query = getattr(candidate, "place_query", None)
    area = getattr(candidate, "area", None)
    record = getattr(candidate, "record", None)
    if not isinstance(record, dict) or not isinstance(query, str) or len(query) > 140:
        return None
    address = record.get("address")
    if not isinstance(address, dict) or _label(record.get("type")) != "museum":
        return None
    identity = map_record_identity(record)
    if identity is None or getattr(candidate, "candidate_id", None) != "-".join(map(str, identity)):
        return None
    parts = [part.strip() for part in query.split(",")]
    if (len(parts) != 3 or not all(parts) or _label(parts[0]) != _label(title)
            or _label(parts[1]) != _label(area)
            or _country_identity(parts[2]) != ("vn",)
            or _label(address.get("country_code")) != "vn"):
        return None
    if address.get("country") and _country_identity(address["country"]) != ("vn",):
        return None
    display = record.get("name")
    if not isinstance(display, str) or _label(display.split(",", 1)[0]) != _label(title):
        return None
    localities = {_label(address.get(field)) for field in _ADDRESS_LOCALITIES} - {""}
    if _label(area) not in localities or not localities.intersection(_NHA_TRANG):
        return None
    names = {_label(title)}
    details = record.get("namedetails")
    if isinstance(details, dict):
        names.update(_label(details.get(field)) for field in ("name", "name:vi", "official_name", "official_name:vi"))
    return _YERSIN_ESTIMATE if names.intersection(_YERSIN_NAMES | _YERSIN_PROVIDER_NAMES) else None


def _policy_cap(candidate) -> int:
    record = getattr(candidate, "record", None)
    kind = _label(record.get("type")) if isinstance(record, dict) else ""
    # Museum, park, garden, ruins and all other types use the same conservative
    # default. A map type is never evidence that a visit lasts this long.
    return 60 if kind in _SHORT_POLICY_TYPES else 120


def get_city_planning_cap(candidate) -> int:
    """A product limit, tightened by a reviewed estimate when one is bound."""
    cap = _policy_cap(candidate)
    estimate = lookup_city_visit_estimate(candidate)
    return min(cap, estimate.planning_cap_minutes) if estimate is not None else cap


def candidate_planning_data(candidate) -> dict:
    """Separate policy from published evidence in model/public diagnostics."""
    data = {
        "basis": "product_policy",
        "policy_cap_minutes": _policy_cap(candidate),
        "planning_cap_minutes": get_city_planning_cap(candidate),
        "note": "Консервативный предел планирования продукта, не длительность из карты, "
                "не физический максимум и не рекомендация источника. Не растягивать посещение ради минимума дня.",
    }
    estimate = lookup_city_visit_estimate(candidate)
    if estimate is not None:
        data["published_estimate"] = estimate.as_prompt_data()
    return data


def check_city_visit_estimates(itinerary, candidates) -> list[str]:
    """Enforce the composed policy/source cap; never shrink/repair a route.

    Exact canonical queries bind items to the caller's server pool. Unknown
    queries still require the separate mandatory record-only map check.
    """
    limits = {}
    for candidate in candidates:
        cap = get_city_planning_cap(candidate)
        previous = limits.get(candidate.place_query)
        if previous is None or cap < previous[0]:
            limits[candidate.place_query] = (cap, lookup_city_visit_estimate(candidate))
    problems = []
    for day in itinerary.days:
        for item in day.items:
            if item.kind != "attraction":
                continue
            limit = limits.get(item.place_query)
            if limit is None:
                continue
            cap, estimate = limit
            duration = item.duration_min
            if type(duration) is not int or duration <= 0 or duration > cap:
                detail = (
                    f"Опубликованная типичная длительность — {estimate.typical_min_minutes}–"
                    f"{estimate.typical_max_minutes} мин. Источник: {estimate.source_url}. "
                    if estimate is not None else "Это правило продукта, не длительность из карты или источника. "
                )
                problems.append(f"День {day.day}: «{item.title}» — предел планирования продукта {cap} мин. "
                                + detail + "Не растягивай посещение ради наполненности дня. "
                                "Это не физический максимум и не подтверждение условий посещения.")
    return problems
