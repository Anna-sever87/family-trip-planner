"""Географические ограничения поездок по большой территории.

Геометрия карты проверяет территорию и ближайшие окрестности. Это не
подтверждение реальной дороги, парома, доступности или бронирования.
"""

from dataclasses import dataclass
import math
import re

import places
from schemas import Itinerary


REGION_MARGIN_KM = 40.0
WATER_BODY_MARGIN_KM = 5.0
REGION_DAY_SPREAD_KM = 120.0
REGION_BASE_RADIUS_KM = 80.0
LOCAL_BASE_TYPES = {
    "city", "town", "village", "hamlet", "suburb", "neighbourhood", "quarter", "locality",
}
REGION_AREA_TYPES = {
    "lake", "water", "national_park", "nature_reserve", "protected_area",
    "administrative", "region", "state", "county", "island", "archipelago",
    "forest", "bay", "peninsula", "valley", "mountain_range", "park",
}


class DestinationScopeError(ValueError):
    """Выбранную территорию нельзя безопасно определить по карте."""


@dataclass(frozen=True)
class RegionScope:
    name: str
    country: str
    bounds: tuple[float, float, float, float]
    geometry: dict | None = None
    area_type: str = ""


def region_margin_km(scope: RegionScope) -> float:
    """Planning convention, not an official geographic boundary.

    A mapped lake/water body means that body plus its nearby coastal zone,
    not all attractions in a broad surrounding region. Measure from its full
    polygon, never its centroid or bounding box. Legacy manually constructed
    scopes and other geographic feature types keep the regional allowance.
    """
    return WATER_BODY_MARGIN_KM if scope.area_type in ("lake", "water") else REGION_MARGIN_KM


def resolve_region(destination: str) -> RegionScope:
    """Берёт рамку именно территории, а не одноимённого посёлка."""
    try:
        record = places.geocode(destination, include_geometry=True)
        if not record:
            raise ValueError("empty")
        bounds = tuple(float(value) for value in record.get("boundingbox", []))
        if (
            len(bounds) != 4
            or not all(math.isfinite(value) for value in bounds)
            or not -90 <= bounds[0] < bounds[1] <= 90
            or not -180 <= bounds[2] < bounds[3] <= 180
            or record.get("type") not in REGION_AREA_TYPES
            or not _valid_geometry(record.get("geojson"))
        ):
            raise ValueError("not a territory")
    except (places.GeocodeUnavailable, places.GeocodeMismatch, ValueError, TypeError, KeyError) as error:
        raise DestinationScopeError(
            f"Не удалось определить территорию «{destination}» на карте. "
            "Уточните её название и страну, например «озеро Байкал, Россия», "
            "или выберите более конкретную зону отдыха."
        ) from error
    parts = [part.strip() for part in destination.split(",") if part.strip()]
    return RegionScope(str(record["name"]), parts[-1] if len(parts) > 1 else "", bounds,
                       record["geojson"], area_type=record["type"])


def build_region_context(scope: RegionScope) -> dict:
    """Краткие факты карты для prompt; не создаёт базы и не обращается к сети.

    Preview содержит только выбранные исходные вершины. Он намеренно не
    объявлен GeoJSON-полигоном: пропуск вершин или колец не должен выглядеть
    как новая разрешающая граница. Валидатор продолжает использовать scope.
    """
    south, north, west, east = scope.bounds
    margin = region_margin_km(scope)
    rings = [
        (polygon_index, ring_index, ring)
        for polygon_index, polygon in enumerate(_polygons(scope.geometry or {}))
        for ring_index, ring in enumerate(polygon)
    ]
    selected_rings = rings[:12]
    points_per_ring = 96 // len(selected_rings) if selected_rings else 0
    previews = []
    for polygon_index, ring_index, ring in selected_rings:
        count = min(len(ring), points_per_ring)
        indices = (
            [round(index * (len(ring) - 1) / (count - 1)) for index in range(count)]
            if count > 1 else list(range(count))
        )
        previews.append({
            "polygon_index": polygon_index,
            "ring_index": ring_index,
            "role": "outer" if ring_index == 0 else "hole",
            "source_vertex_count": len(ring),
            "vertices_lon_lat": [list(ring[index][:2]) for index in indices],
        })

    return {
        "canonical_name": scope.name,
        "country": scope.country,
        "area_type": scope.area_type,
        "bounding_box": {"south": south, "north": north, "west": west, "east": east},
        "boundary_preview": {
            "source_geometry_type": (scope.geometry or {}).get("type"),
            "coordinate_order": "longitude, latitude",
            "source_ring_count": len(rings),
            "omitted_ring_count": len(rings) - len(previews),
            "sampled": len(rings) != len(previews) or any(
                len(preview["vertices_lon_lat"]) < preview["source_vertex_count"]
                for preview in previews
            ),
            "rings": previews,
            "is_membership_test": False,
        },
        "limits_km": {
            "outside_territory_margin": margin,
            "attraction_from_daily_base": REGION_BASE_RADIUS_KM,
            "attractions_within_day": REGION_DAY_SPREAD_KM,
        },
        "planning_rules": [
            "Рамка и выборка вершин показывают только расположение и форму территории. "
            "Попадание в рамку или между показанными вершинами не подтверждает допустимость места: "
            "проверка использует исходный полный контур карты, включая отверстия и острова.",
            f"Каждая база дня и каждый POI должны находиться внутри самой выбранной территории "
            f"или не дальше {margin:g} км от её настоящего контура. "
            "Это допуск планирования, не официальная географическая граница. "
            "Для озёр и водоёмов он задаёт ближнюю береговую зону. "
            "Соседний крупный город или транспортный узел не считается допустимой базой автоматически.",
            "base_area — конкретный населённый пункт или локальная база, которую можно подтвердить "
            "на карте; данный блок не содержит списка подтверждённых баз. Не придумывай названия.",
            "У POI сохраняй его собственный ближайший населённый пункт или конкретную местность "
            "в place_query. Не подставляй вместо них выбранную базу, соседний крупный город "
            "или название всей территории, если это не географический контекст объекта.",
            "При смене base_area заполни transfer_note направлением, ориентировочной длительностью "
            "и ограничениями переезда. После изменения базы дня проверь также переход к следующему дню. "
            "Не обещай неизвестные расписания, дороги или сезонные переправы.",
            "Оставляй свободное время от базы к первому POI, между POI и обратно. "
            "При смене базы в начале дня дополнительно включи в свободное время до первого POI "
            "дорогу от предыдущей базы к новой; одного текста transfer_note недостаточно. "
            "Обед, отдых и посещения не являются временем дороги. Для расстояния больше 3 км "
            "проверка требует минимум округлённое вверх число километров по прямой + 10 минут; "
            "это нижняя оценка, а не подтверждение существования дороги или фактического времени пути.",
        ],
    }


def _polygons(geometry: dict) -> list:
    if geometry.get("type") == "Polygon":
        return [geometry.get("coordinates", [])]
    if geometry.get("type") == "MultiPolygon":
        return geometry.get("coordinates", [])
    return []


def _valid_geometry(geometry) -> bool:
    if not isinstance(geometry, dict):
        return False
    polygons = _polygons(geometry)
    if not polygons:
        return False
    count = 0
    try:
        for polygon in polygons:
            if not polygon:
                return False
            for ring in polygon:
                if len(ring) < 4 or ring[0][:2] != ring[-1][:2]:
                    return False
                for point in ring:
                    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point[:2]):
                        return False
                    lon, lat = float(point[0]), float(point[1])
                    if not math.isfinite(lat) or not math.isfinite(lon) or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                        return False
                    count += 1
        return count <= 100_000
    except (TypeError, ValueError, IndexError):
        return False


def _inside_ring(lat: float, lon: float, ring: list) -> bool:
    inside = False
    for first, second in zip(ring, ring[1:]):
        x1, y1 = first[:2]
        x2, y2 = second[:2]
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _edge_distance(lat: float, lon: float, first: list, second: list) -> float:
    # Локальная проекция на ребро; затем расстояние в километрах по сфере.
    x_scale = max(0.001, math.cos(math.radians(lat)))
    x1, y1 = (first[0] - lon) * x_scale, first[1] - lat
    x2, y2 = (second[0] - lon) * x_scale, second[1] - lat
    dx, dy = x2 - x1, y2 - y1
    length = dx * dx + dy * dy
    fraction = min(1, max(0, -(x1 * dx + y1 * dy) / length)) if length else 0
    nearest_lon = first[0] + fraction * (second[0] - first[0])
    nearest_lat = first[1] + fraction * (second[1] - first[1])
    return places.distance_km(lat, lon, nearest_lat, nearest_lon)


def _distance_from_bounds(lat: float, lon: float, scope: RegionScope) -> float:
    if scope.geometry:
        distances = []
        for polygon in _polygons(scope.geometry):
            if _inside_ring(lat, lon, polygon[0]) and not any(_inside_ring(lat, lon, ring) for ring in polygon[1:]):
                return 0.0
            distances.extend(_edge_distance(lat, lon, first, second) for ring in polygon for first, second in zip(ring, ring[1:]))
        return min(distances) if distances else math.inf
    south, north, west, east = scope.bounds
    nearest_lat = min(north, max(south, lat))
    nearest_lon = min(east, max(west, lon))
    return places.distance_km(lat, lon, nearest_lat, nearest_lon)


def _minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


_WATER_TRANSFER_MODE = (
    r"(?:ледов\w*\s+(?:дорог\w*|маршрут\w*|переезд\w*|переправ\w*)|"
    r"по\s+льду|паром\w*|переправ\w*|катер(?:а|е|ом|у|ов|ами|ах|ы)?|"
    r"лодк\w*|теплоход\w*)"
)
_TRANSFER_AVAILABILITY = (
    r"(?:расписан\w*|доступност\w*|работ\w*|движени\w*|рейс\w*|"
    r"открыти\w*|навигаци\w*|состоян\w*\s+льда|прочност\w*\s+льда|"
    + _WATER_TRANSFER_MODE + r")"
)


def _has_unconfirmed_water_transfer(note: str) -> bool:
    """Узкий запрет явно условного переезда, не доказательство проходимости.

    Не выводит водный путь из координат и не считает молчание о сезонности
    подтверждением. Стоимость/билеты сами по себе не описывают доступность.
    """
    text = " ".join(note.casefold().replace("ё", "е").split())
    # Сначала исключаем отрицание необходимости, а не отрицание доступности.
    text = re.sub(
        r"\bбез\s+(?:использования\s+)?(?:(?:водн|сезонн|речн|морск)\w*\s+){0,2}"
        + _WATER_TRANSFER_MODE + r"(?:\s+и\s+" + _WATER_TRANSFER_MODE + r")?",
        " ", text,
    )
    unnecessary = r"(?:треб\w*|нуж\w*|понадоб\w*|потреб\w*|предусмотр\w*|планиру\w*|использу\w*)"
    text = re.sub(_WATER_TRANSFER_MODE + r"\s+не\s+" + unnecessary, " ", text)
    text = re.sub(r"\bне\s+" + unnecessary + r"\s+" + _WATER_TRANSFER_MODE, " ", text)
    if not re.search(r"\b" + _WATER_TRANSFER_MODE + r"\b", text):
        return False

    unknown = r"(?:не\s+(?:подтвержден\w*|проверен\w*|извест\w*|определен\w*|гарантирован\w*)|неподтвержден\w*|непроверен\w*|неизвест\w*|под\s+вопросом)"
    check = r"(?:уточнить|уточните|проверить|проверьте)"
    # Ограниченная связка слов не перескакивает через знаки препинания
    # от расписания переправы к неизвестной цене такси в соседней фразе.
    bridge = r"(?:\s+[а-я-]+){0,5}\s+"
    uncertain_availability = (
        _TRANSFER_AVAILABILITY + bridge + unknown
        + r"|" + unknown + r"\s+" + _TRANSFER_AVAILABILITY
        + r"|" + check + r"\s+" + _TRANSFER_AVAILABILITY
        + r"|" + _TRANSFER_AVAILABILITY + bridge + check
    )
    if re.search(uncertain_availability, text):
        return True
    # Условие рассматривается только в той же фразе, что и активный режим.
    conditional = (
        r"\bесли\b|\bусловн\w*\b|\bпри\s+(?:наличии|условии|разрешении|доступности)\b|"
        r"\bпри\s+(?:хорошей|подходящей|благоприятной)\s+погоде\b|"
        r"\bзавис\w*\s+от\s+(?:погод\w*|льда|ледов\w*|навигаци\w*)|"
        r"\bв\s+зависимости\s+от\b"
    )
    return any(
        re.search(r"\b" + _WATER_TRANSFER_MODE + r"\b", phrase)
        and re.search(conditional, phrase)
        for phrase in re.split(r"[.!?;]+", text)
    )


def check_region_itinerary(
    itinerary: Itinerary,
    scope: RegionScope,
    *,
    start_time: str = "09:00",
    end_time: str = "20:00",
    travel_minutes_by_day: dict[int, int] | None = None,
) -> list[str]:
    """Проверяет географию и собирает только подтверждённый минимум дороги.

    Выходной словарь заполняется заново. Вызывающий код должен исключить из
    него дни с дополнительными ошибками карты, которых этот валидатор не видит.
    """
    problems: list[str] = []
    previous_base = ""
    previous_base_place = None
    if travel_minutes_by_day is not None:
        travel_minutes_by_day.clear()
    for day in itinerary.days:
        day_problem_start = len(problems)
        confirmed_travel_minutes = 0
        confirmed_gaps = True
        base = day.base_area.strip()
        base_in_scope = False
        base_changed = bool(previous_base and base and base.casefold() != previous_base.casefold())
        if not base:
            problems.append(f"День {day.day}: у регионального маршрута не указана локальная база дня (base_area).")
        elif previous_base and base.casefold() != previous_base.casefold() and not day.transfer_note.strip():
            problems.append(f"День {day.day}: при смене базы «{previous_base}» на «{base}» нужно описать переезд (transfer_note).")
        if _has_unconfirmed_water_transfer(day.transfer_note):
            problems.append(
                f"День {day.day}: маршрут зависит от явно неподтверждённого или условного "
                "водного/ледового переезда (transfer_note). Исключи этот переезд и перестрой "
                "программу без него; координаты и свободное время не подтверждают доступность переправы."
            )
        if base:
            base_country = places._country_identity(base.rsplit(",", 1)[-1].strip())
            query = (
                f"{base}, {scope.country}"
                if scope.country and base_country != places._country_identity(scope.country)
                else base
            )
            try:
                base_place = places.geocode(query)
            except (places.GeocodeUnavailable, places.GeocodeMismatch):
                base_place = None
            if base_place is None:
                problems.append(f"День {day.day}: не удалось подтвердить локальную базу «{base}» на карте.")
            elif not places._place_matches_query(query, base_place):
                problems.append(f"День {day.day}: не удалось подтвердить название и страну локальной базы «{base}» на карте.")
                base_place = None
            elif base_place.get("type") not in LOCAL_BASE_TYPES and not (
                base_place.get("category") in {"place", "boundary"}
                and base_place.get("addresstype") in LOCAL_BASE_TYPES
            ):
                problems.append(
                    f"День {day.day}: база «{base}» не подтверждена как населённый пункт или его локальный район; "
                    "озеро, природный объект и целый регион не подходят для base_area."
                )
                base_place = None
            elif _distance_from_bounds(base_place["lat"], base_place["lon"], scope) > region_margin_km(scope):
                problems.append(f"День {day.day}: база «{base}» находится за пределами выбранной территории и её ближайших окрестностей.")
            else:
                base_in_scope = True
        else:
            base_place = None

        interbase_minimum = 0
        if base_changed and previous_base_place and base_place:
            km = places.distance_km(
                previous_base_place["lat"], previous_base_place["lon"],
                base_place["lat"], base_place["lon"],
            )
            interbase_minimum = math.ceil(km + 10) if km > 3 else 0

        attractions = sorted(
            (item for item in day.items if item.kind == "attraction" and item.verified and item.lat is not None and item.lon is not None),
            key=lambda item: item.time,
        )
        for item in attractions:
            if _distance_from_bounds(item.lat, item.lon, scope) > region_margin_km(scope):
                problems.append(f"День {day.day}: «{item.title}» находится за пределами выбранной территории и её ближайших окрестностей. place_query «{item.place_query}»")
            if base_place and places.distance_km(base_place["lat"], base_place["lon"], item.lat, item.lon) > REGION_BASE_RADIUS_KM:
                problems.append(f"День {day.day}: «{item.title}» слишком далеко от базы «{base}»; выбери соседние места или другую базу. place_query «{item.place_query}»")

        if base_place and attractions:
            for item, outbound in ((attractions[0], True), (attractions[-1], False)):
                km = places.distance_km(base_place["lat"], base_place["lon"], item.lat, item.lon)
                minimum = math.ceil(km + 10) if km > 3 else 0
                if outbound:
                    minimum += interbase_minimum
                try:
                    start = _minutes(start_time) if outbound else _minutes(item.time) + item.duration_min
                    end = _minutes(item.time) if outbound else _minutes(end_time)
                    occupied = sum(
                        max(0, min(end, _minutes(other.time) + other.duration_min) - max(start, _minutes(other.time)))
                        for other in day.items if other.kind != "attraction"
                    )
                except (ValueError, TypeError):
                    confirmed_gaps = False
                    continue
                if end - start - occupied < minimum:
                    direction = "от базы к первому месту" if outbound else "от последнего места обратно к базе"
                    if outbound and interbase_minimum:
                        direction = f"между базами «{previous_base}» → «{base}» и от новой базы к первому месту"
                    problems.append(f"День {day.day}: не оставлено достаточно времени на дорогу {direction} «{base}» (нужно хотя бы {minimum} мин. по нижней оценке).")
                else:
                    confirmed_travel_minutes += minimum

        for first, second in zip(attractions, attractions[1:]):
            km = places.distance_km(first.lat, first.lon, second.lat, second.lon)
            if km <= 3:
                continue
            # Оптимистическая нижняя оценка: 60 км/ч по прямой + 10 минут.
            # Не является расчётом реального автомобильного/водного маршрута.
            minimum = math.ceil(km + 10)
            try:
                finish = _minutes(first.time) + first.duration_min
                start = _minutes(second.time)
                occupied = sum(
                    max(0, min(start, _minutes(item.time) + item.duration_min) - max(finish, _minutes(item.time)))
                    for item in day.items if item.kind != "attraction"
                )
            except (ValueError, TypeError):
                confirmed_gaps = False
                continue  # Формат и пересечения проверяются check_rules.
            available = max(0, start - finish - occupied)
            if available < minimum:
                problems.append(f"День {day.day}: между «{first.title}» и «{second.title}» не оставлено достаточно времени на переезд (около {km:.0f} км по прямой; свободно {available} мин., нужно хотя бы {minimum} мин.).")
            else:
                confirmed_travel_minutes += minimum
        if (
            travel_minutes_by_day is not None
            and len(problems) == day_problem_start
            and confirmed_gaps and base_place and attractions
            and (not base_changed or previous_base_place is not None)
            and len(attractions) == sum(item.kind == "attraction" for item in day.items)
        ):
            travel_minutes_by_day[day.day] = confirmed_travel_minutes
        if base:
            previous_base = base
        previous_base_place = base_place if base_in_scope else None
    return problems
