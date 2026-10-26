"""Strict wire format for grounded schedules, with no AI/map/storage calls.

The wire separates one meal, one rest and a nonempty attraction list. The
decoder only validates and rearranges supplied data; normal itinerary rules,
map verification, costs and program-depth checks remain the caller's job.
This is deliberately a small, fail-closed JSON Schema subset, not a general
JSON Schema implementation. New schema keywords need explicit support here.
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
import math
import re
from typing import Any

from pricing import minimum_meal_cost
from schemas import TripRequest


_DAY_KEY = re.compile(r"day_([1-9][0-9]*)\Z")
_FULL_FIELDS = {"title", "summary", "tips"}
_COMMON = {"type", "description", "title", "enum", "const"}
_KEYWORDS = {
    "object": {"properties", "required", "additionalProperties"},
    "array": {"items", "minItems", "maxItems"},
    "string": {"minLength", "maxLength", "pattern"},
    "integer": {"minimum", "maximum"},
    "number": {"minimum", "maximum"},
    "boolean": set(),
}


def _grounded_fingerprint(grounding):
    from grounded_schedule import _pool
    bases,candidates=_pool(grounding)
    # Provider identity stays server-owned. Include it in the immutability
    # check even though the prompt-facing pool deliberately omits these fields.
    fields = ('provider', 'osm_type', 'osm_id', 'official_name', 'map_address')
    evidence = [[{key: row[key] for key in fields if key in row}
                 for row in rows if isinstance(row, dict)]
                for rows in (grounding.base_facts, grounding.confirmed_hints)]
    try:
        encoded=json.dumps([bases,candidates,evidence],ensure_ascii=False,sort_keys=True,
                           allow_nan=False,separators=(',',':'))
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError('Некорректная идентичность подтверждённого набора') from error
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest(),bases,candidates


@dataclass(frozen=True)
class FrozenGroundedCatalog:
    """Per-operation immutable map identity choices; never returned by AI."""
    fingerprint: str
    bases: tuple
    candidates: tuple

    def assert_matches(self,grounding):
        if freeze_grounded_catalog(grounding) != self:
            raise ValueError('Подтверждённый набор изменился после назначения идентификаторов')


def freeze_grounded_catalog(grounding):
    fingerprint,bases,candidates=_grounded_fingerprint(grounding)
    if not bases or not candidates or len(bases)>3 or len(candidates)>18:
        raise ValueError('Неподдерживаемый размер подтверждённого набора')
    if len({row['base_query'] for row in bases}) != len(bases):
        raise ValueError('Подтверждённый набор содержит неоднозначный запрос базы')
    return FrozenGroundedCatalog(fingerprint,
        tuple((f'b{index:02}',row['base_key'],row['base_query']) for index,row in enumerate(bases,1)),
        tuple((f'p{index:02}',row['title'],row['place_query'],row['locality'],tuple(row['allowed_base_keys']))
              for index,row in enumerate(candidates,1)))


def build_grounded_id_schema(original_wire: dict, grounding, *, catalog: FrozenGroundedCatalog) -> dict:
    """Project the already constrained full/partial wire onto short choices."""
    catalog.assert_matches(grounding)
    _check_schema(original_wire)
    result=deepcopy(original_wire)
    candidate_by_query={row[2]:row[0] for row in catalog.candidates}
    base_by_query={row[2]:row[0] for row in catalog.bases}
    for key,day in result['properties'].items():
        if not _DAY_KEY.fullmatch(key):
            continue
        props=day['properties'];base=props.pop('base_area')
        queries=base.get('enum')
        if not isinstance(queries,list) or not queries or any(query not in base_by_query for query in queries):
            raise ValueError('Схема базы не связана с подтверждённым набором')
        props['base_id']={'type':'string','enum':[base_by_query[query] for query in queries],
                          'minLength':3,'maxLength':3}
        if 'const' in base:
            if base['const'] not in queries:
                raise ValueError('Ограничение базы не соответствует подтверждённому набору')
            props['base_id']['const']=base_by_query[base['const']]
        day['required']=[field if field!='base_area' else 'base_id' for field in day['required']]
        attraction=props['attractions']['items'];fields=attraction['properties']
        query_schema=fields['place_query']
        queries=query_schema.get('enum')
        if not isinstance(queries,list) or not queries or any(query not in candidate_by_query for query in queries):
            raise ValueError('Схема посещений не связана с подтверждённым набором')
        for field in ('title','area','place_query'):
            fields.pop(field)
        fields['candidate_id']={'type':'string','enum':[candidate_by_query[query] for query in queries],
                                'minLength':3,'maxLength':3}
        if 'const' in query_schema:
            if query_schema['const'] not in queries:
                raise ValueError('Ограничение посещения не соответствует подтверждённому набору')
            fields['candidate_id']['const']=candidate_by_query[query_schema['const']]
        attraction['required']=[field for field in attraction['required'] if field not in {'title','area','place_query'}]+['candidate_id']
    _check_schema(result)
    return result


def decode_grounded_id_response(raw: str, id_schema: dict, original_wire: dict, grounding,
                                *, catalog: FrozenGroundedCatalog) -> str:
    """Strict ID wire → exact frozen identities → original wire → canonical.

    Unknown IDs and conflicting legacy fields fail before hydration. Time,
    duration, prose, prices, day/date and transport statements stay untouched.
    """
    catalog.assert_matches(grounding)
    if id_schema != build_grounded_id_schema(original_wire,grounding,catalog=catalog):
        raise ValueError('Схема идентификаторов не соответствует исходному расписанию')
    if not isinstance(raw,str):
        raise ValueError('Ответ должен быть строкой JSON')
    try:
        payload=json.loads(raw,object_pairs_hook=_unique_object,parse_constant=_reject_constant)
    except (TypeError,RecursionError) as error:
        raise ValueError('Некорректный JSON-ответ') from error
    _validate(payload,id_schema)
    candidates={row[0]:row for row in catalog.candidates};bases={row[0]:row for row in catalog.bases}
    for key,day in payload.items():
        if not _DAY_KEY.fullmatch(key):
            continue
        day['base_area']=bases[day.pop('base_id')][2]
        for item in day['attractions']:
            candidate=candidates[item.pop('candidate_id')]
            item.update(title=candidate[1],place_query=candidate[2],area=candidate[3])
    return decode_grounded_response(json.dumps(payload,ensure_ascii=False,allow_nan=False),original_wire)


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _same_value(first: Any, second: Any) -> bool:
    # Python's True == 1 must not accept a JSON boolean as a numeric enum.
    if isinstance(first, bool) != isinstance(second, bool):
        return False
    return first == second


def _check_schema(schema: Any, path: str = "schema") -> None:
    if not isinstance(schema, dict) or schema.get("type") not in _KEYWORDS:
        raise ValueError(f"{path}: неподдерживаемый тип схемы")
    kind = schema["type"]
    if set(schema) - (_COMMON | _KEYWORDS[kind]):
        raise ValueError(f"{path}: неподдерживаемое ограничение схемы")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ValueError(f"{path}: enum должен быть непустым массивом")
    for low, high in (("minimum", "maximum"), ("minLength", "maxLength"), ("minItems", "maxItems")):
        for key in (low, high):
            if key in schema and (
                not _finite_number(schema[key])
                or (key not in {"minimum", "maximum"} and (type(schema[key]) is not int or schema[key] < 0))
            ):
                raise ValueError(f"{path}.{key}: неверная граница")
        if low in schema and high in schema and schema[low] > schema[high]:
            raise ValueError(f"{path}: несовместимые границы")
    if "pattern" in schema:
        try:
            re.compile(schema["pattern"])
        except (TypeError, re.error) as error:
            raise ValueError(f"{path}: неверный pattern") from error
    if kind == "object":
        props, required = schema.get("properties"), schema.get("required")
        if (not isinstance(props, dict) or not all(isinstance(key, str) for key in props)
                or not isinstance(required, list) or not all(isinstance(key, str) for key in required)
                or len(required) != len(set(required)) or set(required) != set(props)
                or schema.get("additionalProperties") is not False):
            raise ValueError(f"{path}: нужны ровно все обязательные поля и additionalProperties=false")
        for key, child in props.items():
            _check_schema(child, f"{path}.{key}")
    elif kind == "array":
        _check_schema(schema.get("items"), f"{path}[]")


def _validate(value: Any, schema: dict, path: str = "response") -> None:
    kind = schema["type"]
    valid_type = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": type(value) is int and _finite_number(value),
        "number": _finite_number(value),
        "boolean": type(value) is bool,
    }[kind]
    if not valid_type:
        raise ValueError(f"{path}: требуется {kind}, преобразование типов запрещено")
    if "const" in schema and not _same_value(value, schema["const"]):
        raise ValueError(f"{path}: значение должно точно соответствовать const")
    if "enum" in schema and not any(_same_value(value, choice) for choice in schema["enum"]):
        raise ValueError(f"{path}: значение отсутствует в enum")
    if kind == "object":
        props = schema["properties"]
        if set(value) != set(props):
            missing, extra = sorted(set(props) - set(value)), sorted(set(value) - set(props))
            raise ValueError(f"{path}: отсутствующие поля {missing}; лишние поля {extra}")
        for key, child in props.items():
            _validate(value[key], child, f"{path}.{key}")
    elif kind in {"array", "string"}:
        low, high = ("minItems", "maxItems") if kind == "array" else ("minLength", "maxLength")
        if len(value) < schema.get(low, 0) or len(value) > schema.get(high, math.inf):
            raise ValueError(f"{path}: нарушены границы {low}/{high}")
        if kind == "array":
            for index, item in enumerate(value):
                _validate(item, schema["items"], f"{path}[{index}]")
        elif "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ValueError(f"{path}: нарушен pattern")
    elif kind in {"integer", "number"}:
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError(f"{path}: нарушены minimum/maximum")


def _fix_value(schema: dict, value: Any) -> None:
    _validate(value, schema, "schema.fixed_value")
    schema["const"] = value
    if "enum" in schema:
        schema["enum"] = [value]


def _wire_day(canonical: dict, request: TripRequest, number: int) -> dict:
    result = deepcopy(canonical)
    props = result.get("properties", {})
    if result.get("type") != "object" or not {"day", "date", "items"}.issubset(props):
        raise ValueError("Схема дня должна содержать day, date и items")
    if {"attractions", "meal", "rest"} & set(props):
        raise ValueError("Каноническая схема уже содержит поля wire-формата")
    _fix_value(props["day"], number)
    _fix_value(props["date"], (date.fromisoformat(request.start_date) + timedelta(days=number - 1)).isoformat())
    # Explicit singleton enums keep day/date constraints visible to providers.
    props["day"]["enum"] = [number]
    props["date"]["enum"] = [props["date"]["const"]]
    items = props.pop("items")
    if items.get("type") != "array" or type(items.get("maxItems")) is not int or items["maxItems"] < 3:
        raise ValueError("Канонический день должен допускать посещение, обед и отдых")
    item = items["items"]
    if item.get("type") != "object" or not {"kind", "title", "time", "place_query", "cost"}.issubset(item.get("properties", {})):
        raise ValueError("Неполная каноническая схема пункта")
    attraction, meal, rest = (deepcopy(item) for _ in range(3))
    for schema, kind in ((attraction, "attraction"), (meal, "meal"), (rest, "rest")):
        _fix_value(schema["properties"]["kind"], kind)
    for schema in (meal, rest):
        _fix_value(schema["properties"]["place_query"], "")
    _fix_value(rest["properties"]["title"], "Отдых")
    query = attraction["properties"]["place_query"]
    query["minLength"] = max(1, query.get("minLength", 0))
    if "enum" in query:
        query["enum"] = [value for value in query["enum"] if value != ""]
        if not query["enum"]:
            raise ValueError("В схеме нет непустых place_query для посещений")
    if "const" in query:
        _validate(query["const"], query, "schema.attraction.place_query")
    cost = meal["properties"]["cost"]
    cost["minimum"] = max(cost.get("minimum", 0), minimum_meal_cost(request))
    if cost["minimum"] > cost.get("maximum", math.inf):
        raise ValueError("Минимальная стоимость обеда превышает разрешённый бюджет")
    if "enum" in cost:
        cost["enum"] = [value for value in cost["enum"] if _finite_number(value) and value >= cost["minimum"]]
        if not cost["enum"]:
            raise ValueError("Enum стоимости не допускает минимальную стоимость обеда")
    if "const" in cost:
        _validate(cost["const"], cost, "schema.meal.cost")
    props["attractions"] = {
        "type": "array", "minItems": max(1, items.get("minItems", 0) - 2),
        "maxItems": items["maxItems"] - 2, "items": attraction,
    }
    props["meal"], props["rest"] = meal, rest
    result["required"] = [key for key in result["required"] if key != "items"] + ["attractions", "meal", "rest"]
    return result


def build_grounded_response_schema(canonical_schema: dict, request: TripRequest) -> dict:
    """Return an independent strict full-route or keyed-repair wire schema."""
    _check_schema(canonical_schema)
    if canonical_schema["type"] != "object" or request.days_count < 1:
        raise ValueError("Нужны объект схемы и положительное число дней")
    result = deepcopy(canonical_schema)
    props = result["properties"]
    if "days" in props:
        if set(props) != _FULL_FIELDS | {"days"}:
            raise ValueError("Неизвестный формат полной канонической схемы")
        days = props.pop("days")
        if (days.get("type") != "array" or days.get("minItems") != request.days_count
                or days.get("maxItems") != request.days_count):
            raise ValueError("Число дней схемы должно совпасть с запросом")
        for number in range(1, request.days_count + 1):
            props[f"day_{number}"] = _wire_day(days["items"], request, number)
    else:
        if not props or any(not _DAY_KEY.fullmatch(key) for key in props):
            raise ValueError("Схема исправления должна содержать только day_N")
        for key, day in list(props.items()):
            number = int(_DAY_KEY.fullmatch(key).group(1))
            if number > request.days_count:
                raise ValueError("Номер дня исправления за пределами поездки")
            props[key] = _wire_day(day, request, number)
    result["required"] = list(props)
    _check_schema(result)
    return result


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Повторяющиеся JSON-ключи запрещены")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Нечисловая JSON-константа {value} запрещена")


def decode_grounded_response(raw: str, wire_schema: dict) -> str:
    """Validate wire locally and return canonical JSON, without filling data."""
    if not isinstance(raw, str):
        raise ValueError("Ответ должен быть строкой JSON")
    _check_schema(wire_schema)
    props = wire_schema.get("properties", {})
    day_keys = [key for key in props if _DAY_KEY.fullmatch(key)]
    other = set(props) - set(day_keys)
    full = other == _FULL_FIELDS
    if wire_schema["type"] != "object" or not day_keys or (other and not full):
        raise ValueError("Ожидалась схема wire-ответа с day_N")
    numbers = sorted(int(_DAY_KEY.fullmatch(key).group(1)) for key in day_keys)
    if full and numbers != list(range(1, len(numbers) + 1)):
        raise ValueError("Полная wire-схема должна содержать последовательные day_N")
    for key in day_keys:
        day_props = props[key].get("properties", {})
        if "items" in day_props or not {"day", "date", "attractions", "meal", "rest"}.issubset(day_props):
            raise ValueError("Только wire-день с attractions, meal и rest допустим")
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (TypeError, RecursionError) as error:
        raise ValueError("Некорректный JSON-ответ") from error
    _validate(payload, wire_schema)
    days = {}
    for number in numbers:
        key = f"day_{number}"
        wire_day = payload[key]
        day = {name: value for name, value in wire_day.items() if name not in {"attractions", "meal", "rest"}}
        day["items"] = sorted([*wire_day["attractions"], wire_day["meal"], wire_day["rest"]], key=lambda item: item["time"])
        days[key] = day
    result = ({**{key: payload[key] for key in props if key in _FULL_FIELDS}, "days": list(days.values())}
              if full else days)
    return json.dumps(result, ensure_ascii=False, allow_nan=False)


def grounded_query_feedback(raw: str, wire_schema: dict, grounding) -> list[dict]:
    """Retry data for one unambiguous verified title; never rewrite a reply.

    No alias inference or enum relaxation. Rejected model strings never enter
    the result. Both full responses and scoped day_N repairs are supported.
    """
    if not isinstance(raw, str) or len(raw) > 262144 or not isinstance(wire_schema, dict):
        return []
    from grounded_schedule import _pool
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if not isinstance(payload, dict):
            return []
        _, candidates = _pool(grounding)
        by_title = {}
        for candidate in candidates:
            by_title.setdefault(candidate['title'], []).append(candidate)
        corrections = []
        for key, schema in wire_schema.get('properties', {}).items():
            if not _DAY_KEY.fullmatch(key):
                continue
            day = payload.get(key)
            if not isinstance(day, dict) or not isinstance(day.get('attractions'), list):
                continue
            allowed = schema['properties']['attractions']['items']['properties']['place_query'].get('enum')
            if not isinstance(allowed, list):
                continue
            for index, item in enumerate(day['attractions'][:5]):
                if not isinstance(item, dict) or not isinstance(item.get('title'), str):
                    continue
                if item.get('place_query') in allowed:
                    continue
                matches = by_title.get(item['title'], [])
                if len(matches) != 1 or matches[0]['place_query'] not in allowed:
                    continue
                candidate = matches[0]
                corrections.append({'response_path': f'{key}.attractions[{index}].place_query',
                                    'confirmed_title': candidate['title'],
                                    'allowed_place_query': candidate['place_query']})
                if len(corrections) == 4:
                    return corrections
        return corrections
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return []
