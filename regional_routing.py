"""Bounded OSRM road/foot graph for a small, already verified candidate set.

At most one provisional table request per profile; selected edges require the
separate route-step validator before publication. No retry or aerial fallback.
These are routing estimates, not live traffic or a booking/date availability.
SQLite serializes cache misses across web/bot processes (max 1 request/second).
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlsplit

import httpx

from places import USER_AGENT, distance_km
from regional_skeleton import TravelFact, _visit_access_valid

ROUTING_CACHE_FILE = Path("output/routing_cache.db")
DEFAULT_ROUTER = "https://routing.openstreetmap.de"
ATTRIBUTION = ("Маршруты: OSRM / FOSSGIS; данные © OpenStreetMap contributors (ODbL). "
               "https://www.openstreetmap.org/copyright Исправить карту: https://www.openstreetmap.org/fixthemap")
CACHE_TTL = 7 * 86400
MAX_SELECTED_ROUTES = 32
MAX_LOOP_ACCESS_DISTANCE_METRES = 250


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def _coordinates(lat, lon):
    try:
        return (type(lat) in (int, float) and type(lon) in (int, float)
                and math.isfinite(lat) and math.isfinite(lon)
                and -90 <= lat <= 90 and -180 <= lon <= 180)
    except OverflowError:
        return False


def _validate_table(data, n):
    """Only complete, directed road matrices enter or leave the cache."""
    if not isinstance(data, dict) or data.get("code") != "Ok":
        raise ValueError("Маршрутизатор не подтвердил дорожную сеть")
    if "fallback_speed_cells" in data and data["fallback_speed_cells"] != []:
        raise ValueError("Прямые линии вместо дорог запрещены")
    if any(not isinstance(data.get(key), list) or len(data[key]) != n
           for key in ("durations", "distances", "sources", "destinations")):
        raise ValueError("Неполная матрица маршрутов")
    if any(not isinstance(row, list) or len(row) != n
           for key in ("durations", "distances") for row in data[key]):
        raise ValueError("Неполные строки матрицы")
    for i in range(n):
        for j in range(n):
            seconds, metres = data["durations"][i][j], data["distances"][i][j]
            if seconds is None and metres is None:
                continue  # Unreachable is not a zero-distance edge.
            if not _number(seconds) or not _number(metres):
                raise ValueError("Некорректная ячейка дорожной матрицы")


def _waypoint_matches(waypoint, point, limit):
    if not isinstance(waypoint, dict) or not _number(waypoint.get("distance")) or waypoint["distance"] > limit:
        return False
    location = waypoint.get("location")
    if not isinstance(location, list) or len(location) != 2 or not _coordinates(location[1], location[0]):
        return False
    return distance_km(point[0], point[1], location[1], location[0]) * 1000 <= limit


def _routing_url(points, profile, service):
    if (profile not in ("foot", "car") or not isinstance(points, (list, tuple)) or not 2 <= len(points) <= 21
            or any(not isinstance(point, (list, tuple)) or len(point) != 2 or not _coordinates(*point) for point in points)):
        raise ValueError("Неверный ограниченный запрос маршрутизации")
    origin = os.getenv("OSRM_ROUTING_BASE_URL", DEFAULT_ROUTER).rstrip("/")
    parsed = urlsplit(origin)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Для маршрутизации нужен HTTPS-адрес без учётных данных")
    coordinates = ";".join(f"{lon:.7f},{lat:.7f}" for lat, lon in points)
    return f"{origin}/routed-{profile}/{service}/v1/driving/{coordinates}"


def _cached_request(url, params, validate, *, cache_file=None, get=None):
    # Table estimates and checked route steps share one cross-process clock,
    # but never a cache key. Old table-only no-ferry claims cannot be reused.
    key = hashlib.sha256((url + "|route-steps-v3|" + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    path = Path(cache_file or ROUTING_CACHE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=40)) as connection, connection:
        connection.execute("CREATE TABLE IF NOT EXISTS routing_cache (key TEXT PRIMARY KEY, checked REAL NOT NULL, payload TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS routing_clock (id INTEGER PRIMARY KEY CHECK(id=1), requested REAL NOT NULL)")
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute("SELECT checked,payload FROM routing_cache WHERE key=?", (key,)).fetchone()
        if row and 0 <= time.time() - row[0] <= CACHE_TTL:
            try:
                cached = json.loads(row[1])
                validate(cached)
            except (ValueError, TypeError, OverflowError, RecursionError):
                connection.execute("DELETE FROM routing_cache WHERE key=?", (key,))
            else:
                return cached, url
        previous = connection.execute("SELECT requested FROM routing_clock WHERE id=1").fetchone()
        if previous:
            time.sleep(min(1.1, max(0, 1.1 - (time.time() - previous[0]))))
        connection.execute("INSERT OR REPLACE INTO routing_clock VALUES(1,?)", (time.time(),))
        try:
            response = (get or httpx.get)(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30.0)
            response.raise_for_status()
            if len(response.content) > 2_000_000:
                raise ValueError("Слишком большой ответ маршрутизатора")
            data = response.json()
            validate(data)
            connection.execute("INSERT OR REPLACE INTO routing_cache VALUES(?,?,?)", (key, time.time(), json.dumps(data, allow_nan=False)))
            return data, url
        finally:
            # Keep the process-wide SQLite lock until the attempt finishes, but
            # persist its start even on fast HTTP/JSON failures. A rollback here
            # would let the next profile/process evade the 1 request/sec limit.
            connection.commit()


def _table(points, profile, *, cache_file=None, get=None):
    url = _routing_url(points, profile, "table")
    # Public services do not necessarily support exclude=ferry. A table does
    # not expose modes and is only a search estimate, never a ferry check.
    params = {"annotations": "duration,distance", "generate_hints": "false"}
    return _cached_request(url, params, lambda data: _validate_table(data, len(points)),
                           cache_file=cache_file, get=get)


def _validate_route(data, points, profile):
    if not isinstance(data, dict) or data.get("code") != "Ok":
        raise ValueError("Маршрутизатор не подтвердил выбранный переход")
    routes, waypoints = data.get("routes"), data.get("waypoints")
    limit = 100 if profile == "foot" else 250
    if (not isinstance(routes, list) or len(routes) != 1
            or not isinstance(waypoints, list) or len(waypoints) != 2
            or not all(_waypoint_matches(wp, point, limit) for wp, point in zip(waypoints, points))):
        raise ValueError("Выбранный переход не привязан к подтверждённым точкам")
    route = routes[0]
    if not isinstance(route, dict):
        raise ValueError("Неполный ответ маршрутизатора")
    legs = route.get("legs")
    if not isinstance(legs, list) or len(legs) != 1 or not isinstance(legs[0], dict):
        raise ValueError("Неполные участки выбранного перехода")
    leg = legs[0]
    steps = leg.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("Не подтверждён способ передвижения")
    expected_mode = "walking" if profile == "foot" else "driving"
    if any(not isinstance(step, dict) or step.get("mode") != expected_mode for step in steps):
        raise ValueError("Паромный или неизвестный способ передвижения не подтверждён")
    if any(not _number(item.get(key)) for item in [route, leg, *steps] for key in ("duration", "distance")):
        raise ValueError("Некорректное время или расстояние выбранного перехода")
    seconds = max(route["duration"], leg["duration"], sum(step["duration"] for step in steps))
    metres = max(route["distance"], leg["distance"], sum(step["distance"] for step in steps))
    if not _number(seconds) or not _number(metres):
        raise ValueError("Некорректная сумма участков выбранного перехода")
    if metres + 2 * limit + 10 < distance_km(*points[0], *points[1]) * 1000:
        raise ValueError("Не подтверждена дорожная связь между точками")
    if profile == "foot" and (metres > 5000 or seconds > 5400):
        raise ValueError("Пеший переход превышает допустимую длину или время")
    return seconds, metres


def _route(points, profile, *, cache_file=None, get=None):
    if len(points) != 2:
        raise ValueError("Проверяется ровно один направленный переход")
    url = _routing_url(points, profile, "route")
    params = {"steps": "true", "overview": "false", "alternatives": "false", "generate_hints": "false"}
    return _cached_request(url, params, lambda data: _validate_route(data, points, profile),
                           cache_file=cache_file, get=get)


def _routing_nodes(grounding, visits):
    hints = {}
    for item in grounding.confirmed_hints:
        query = item["place_query"]
        if not isinstance(query, str) or not query or query in hints:
            raise ValueError("Неоднозначные координаты места")
        hints[query] = item
    nodes = [(base["base_key"], base) for base in grounding.base_facts]
    for visit in visits:
        point = hints[visit.place_query]
        if not _visit_access_valid(visit) or not _coordinates(point.get("lat"), point.get("lon")):
            raise ValueError("Неподтверждённая точка доступа к посещению")
        if visit.route_kind == "loop":
            lat, lon = visit.access_point
            anchor_lat, anchor_lon = visit.map_anchor_point or visit.access_point
            if distance_km(point["lat"], point["lon"], anchor_lat, anchor_lon) * 1000 > MAX_LOOP_ACCESS_DISTANCE_METRES:
                raise ValueError("Точка доступа слишком далеко от подтверждённого места")
            # Both incoming and outgoing edges use this same source entry.
            # Never mutate the map hint, substitute a summit, add an inferred
            # connector, or credit the loop duration again as travel time.
            point = {**point, "lat": lat, "lon": lon}
        nodes.append((visit.visit_id, point))
    if (not 2 <= len(nodes) <= 21 or any(not isinstance(key, str) or not key for key, _ in nodes)
            or len({key for key, _ in nodes}) != len(nodes)
            or any(not _coordinates(item.get("lat"), item.get("lon")) for _, item in nodes)):
        raise ValueError("Неоднозначные точки маршрутизации")
    return nodes


def verify_route_edges(edges, grounding, facts, *, cache_file=None, get=None):
    """Fail closed on selected directed routes; at most 32 GETs, no retries.

    An empty result confirms modes/time/reserve only in the cached OSRM graph,
    not access, traffic, weather or opening conditions on the trip date. Facts
    and edges are not mutated; table credit is usable only after this passes.
    """
    if os.getenv("ROUTING_ENABLED", "true").casefold() in ("false", "0", "no"):
        return ["Проверка выбранных дорожных переходов отключена."]
    try:
        nodes = dict(_routing_nodes(grounding, facts))
        selected = {}
        for edge in edges:
            if (not isinstance(edge, TravelFact) or edge.evidence != "routing"
                    or edge.mode not in ("пешком", "автомобиль/такси")
                    or edge.from_key not in nodes or edge.to_key not in nodes or edge.from_key == edge.to_key
                    or type(edge.minutes) is not int or not 1 <= edge.minutes <= 600
                    or type(edge.depth_credit_min) is not int or not 0 <= edge.depth_credit_min <= edge.minutes):
                raise ValueError("Неоднозначный выбранный переход")
            profile = "foot" if edge.mode == "пешком" else "car"
            if edge.currency != "RUB" or (profile == "foot" and (edge.group_cost != 0 or type(edge.group_cost) is bool or edge.planning_reserve is not None)):
                raise ValueError("Не подтверждена стоимость выбранного перехода")
            if profile == "car" and (edge.group_cost is not None or not _number(edge.planning_reserve)):
                raise ValueError("Не подтверждён резерв выбранного перехода")
            selected.setdefault((profile, edge.from_key, edge.to_key), []).append(edge)
            if len(selected) > MAX_SELECTED_ROUTES:
                raise ValueError("Превышен лимит проверки выбранных переходов")
    except (AttributeError, KeyError, TypeError, ValueError):
        return ["Не удалось однозначно проверить выбранные переходы в установленном лимите."]
    problems = []
    for (profile, start, end), occurrences in selected.items():
        points = [(nodes[key]["lat"], nodes[key]["lon"]) for key in (start, end)]
        try:
            data, _ = _route(points, profile, cache_file=cache_file, get=get)
            seconds, metres = _validate_route(data, points, profile)
            reserve = max(500, math.ceil(metres / 1000 * 45 / 100) * 100)
            for edge in occurrences:
                if seconds > edge.minutes * 60 or edge.depth_credit_min > math.floor(seconds / 60):
                    raise ValueError("Время перехода изменилось")
                if profile == "car" and reserve > edge.planning_reserve:
                    raise ValueError("Резерв на переход недостаточен")
        except (httpx.HTTPError, ValueError, TypeError, KeyError, OverflowError, RecursionError, sqlite3.Error, OSError):
            # Never echo provider bodies/URLs or a private request into feedback.
            problems.append("Не удалось подтвердить выбранный переход по дороге или пешком в пределах времени и бюджета; маршрут не принят.")
    return problems


def verify_selected_routes(skeleton, grounding, facts, *, cache_file=None, get=None):
    """Validate travel slots of a ready skeleton before canonical conversion."""
    try:
        if skeleton.status not in ("ready", "limited_program") or not skeleton.days:
            raise ValueError("Не готовый каркас маршрута")
        edges = []
        for day in skeleton.days:
            if getattr(day, "is_free_day", False) and any(slot.kind == "travel" for slot in day.slots):
                raise ValueError("Свободный день не может скрывать переезд")
            for slot in day.slots:
                if slot.kind == "travel":
                    if slot.travel_fact is None or slot.duration_min < slot.travel_fact.minutes:
                        raise ValueError("Неполный интервал перехода")
                    edges.append(slot.travel_fact)
    except (AttributeError, TypeError, ValueError):
        return ["Не удалось проверить переходы неполного расписания."]
    return verify_route_edges(edges, grounding, facts, cache_file=cache_file, get=get)


def build_travel_graph(grounding, visits, *, cache_file=None, get=None):
    """Return provisional edges; selected routes MUST pass step verification."""
    if os.getenv("ROUTING_ENABLED", "true").casefold() in ("false", "0", "no"):
        return [], ["Проверка дорожных связей отключена."]
    try:
        nodes = _routing_nodes(grounding, visits)
    except (AttributeError, KeyError, TypeError, ValueError):
        return [], ["Недостаточно однозначных точек для маршрутизации."]
    for _, item in nodes:
        if not _coordinates(item.get("lat"), item.get("lon")):
            return [], ["Некорректные координаты маршрутизации."]
    points = [(item["lat"], item["lon"]) for _, item in nodes]
    matrices, warnings = {}, []
    for profile in ("foot", "car"):
        try:
            data, source = _table(points, profile, cache_file=cache_file, get=get)
            matrices[profile] = (data, source)
        except (httpx.HTTPError, ValueError, TypeError, KeyError, sqlite3.Error, OSError):
            warnings.append(f"Не удалось подтвердить дорожный граф ({profile}); прямые линии не подставлены.")
    edges = []
    for i, (start, first) in enumerate(nodes):
        for j, (end, second) in enumerate(nodes):
            if i == j:
                continue
            candidates = []
            for profile, (data, source) in matrices.items():
                seconds, metres = data["durations"][i][j], data["distances"][i][j]
                if not _number(seconds) or not _number(metres):
                    continue
                # Do not snap an unconnected attraction onto a distant highway.
                waypoints = (data["sources"][i], data["destinations"][j])
                limit = 100 if profile == "foot" else 250
                if not all(_waypoint_matches(wp, point, limit) for wp, point in zip(waypoints, (points[i], points[j]))):
                    continue
                if profile == "foot" and (metres > 5000 or seconds > 5400):
                    continue
                raw_minutes = math.ceil(seconds / 60)
                km = distance_km(first["lat"], first["lon"], second["lat"], second["lon"])
                if metres + 2 * limit + 10 < km * 1000:
                    continue  # Impossible road distance, even allowing snapping.
                geographic_floor = math.ceil(km + 10) if km > 3 else 0
                # Reserve a transparent time buffer; it is not depth credit.
                minutes = max(5, geographic_floor, math.ceil(raw_minutes * 1.2) + (10 if profile == "car" else 5))
                if minutes > 600:
                    continue
                walking = profile == "foot"
                reserve = 0.0 if walking else float(max(500, math.ceil(metres / 1000 * 45 / 100) * 100))
                edge = TravelFact(
                    from_key=start, to_key=end, minutes=minutes,
                    mode="пешком" if walking else "автомобиль/такси", evidence="routing", source=source,
                    note=("Предварительная оценка графа OSRM; способ передвижения требует отдельной проверки участков. Учтён запас времени. "
                          "Погода, закрытия и фактическая доступность на дату не подтверждены."),
                    group_cost=0.0 if walking else None, planning_reserve=None if walking else reserve,
                    currency="RUB", price_source="https://www.openstreetmap.org/copyright" if walking else "",
                    price_note=("Пеший переход без транспортного билета." if walking else
                                "Плановый резерв на автомобиль/такси: 45 ₽/км, минимум 500 ₽ за переезд; это не тариф перевозчика, цену нужно уточнить."),
                    # Buffer rounds up; credited occupied minutes round down
                    # so several fractional legs cannot manufacture depth.
                    depth_credit_min=min(minutes, math.floor(seconds / 60)),
                )
                candidates.append(edge)
            if candidates:
                # Walk short connected routes; use driving only for longer legs.
                edges.append(next((edge for edge in candidates if edge.mode == "пешком"), candidates[0]))
    if not edges and not warnings:
        warnings.append("Связи между точками не подтверждены дорожным графом; прямые линии не подставлены.")
    return edges, warnings
