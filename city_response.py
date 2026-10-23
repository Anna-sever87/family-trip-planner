"""Pure city-repair wire schema; structure is not evidence about places.

Use grounded_response.decode_grounded_response to restore canonical JSON.
Normal itinerary, map, budget and program-depth checks remain mandatory.
"""

from grounded_response import (
    _check_schema,
    _validate,
    build_grounded_response_schema,
)
from schemas import TripRequest


CITY_RESPONSE_FORMAT_RULE = """ФОРМАТ ОТВЕТА ДЛЯ ИСПРАВЛЕНИЯ ГОРОДСКОГО МАРШРУТА:
Верни только JSON по текущей response schema. Для полной поездки верхний объект содержит title, summary, tips и отдельный обязательный ключ day_1, day_2, ... для каждой даты анкеты. Если схема запрашивает только отдельные day_N, верни только эти ключи.
Внутри day_N сохрани обязательные day, date, theme, base_area, transfer_note и rainy_alternative по схеме. Вместо items используй три обязательных поля: attractions — непустой массив посещений с kind=attraction; meal — ровно один объект с kind=meal; rest — ровно один объект с kind=rest. Число посещений не превышает максимум схемы; второе посещение само по себе не обязательно. У каждого посещения place_query должен содержать конкретный объект, город и страну, а не пустую строку или одни пробелы.
У meal и rest place_query пуст; title отдыха — «Отдых». Стоимость обеда не ниже минимума схемы для всей группы. Если младшему ребёнку меньше семи лет, отдых длится не менее 60 минут. Соблюдай остальные ограничения времени, длительности, состава группы и бюджета.
В новом ответе запрещены days и items, пропущенные дни, пустой attractions, массив вместо meal/rest и неверные номера или даты дней. Предыдущие ответы и current_route могут содержать канонические days/items: это входные данные, а не формат нового ответа. Не меняй содержание без необходимости исправления замечаний.
Формат не подтверждает существование мест, цены, часы работы или пригодность для погоды и детей. Не выдумывай посещения и свойства ради заполнения схемы. Все прежние проверки мест, уникальности, расписания, бюджета, погодной пригодности и содержательности программы остаются обязательными."""


def _retain_valid_choices(field: dict, label: str) -> None:
    """Keep existing enum/const constraints compatible with the narrower field."""
    if "enum" in field:
        choices = []
        for value in field["enum"]:
            try:
                _validate(value, field, label)
            except ValueError:
                continue
            choices.append(value)
        if not choices:
            raise ValueError(f"{label}: нет допустимых значений после уточнения схемы")
        field["enum"] = choices
    if "const" in field:
        _validate(field["const"], field, label)


def build_city_response_schema(canonical_schema: dict, request: TripRequest) -> dict:
    """Narrow the existing strict adapter without mutating canonical schema."""
    if request.destination_kind != "city":
        raise ValueError("Городской wire-формат предназначен только для city")
    schema = build_grounded_response_schema(canonical_schema, request)
    young_child = bool(request.children_ages) and min(request.children_ages) < 7
    for key, day in schema["properties"].items():
        if not key.startswith("day_"):
            continue
        properties = day["properties"]
        query = properties["attractions"]["items"]["properties"]["place_query"]
        # Never silently replace a different, potentially stronger constraint.
        if "pattern" in query and query["pattern"] != r"\S":
            raise ValueError("Нельзя заменить существующий pattern запроса места")
        query["pattern"] = r"\S"
        _retain_valid_choices(query, f"{key}.attractions.place_query")
        if young_child:
            duration = properties["rest"]["properties"]["duration_min"]
            duration["minimum"] = max(60, duration.get("minimum", 0))
            _retain_valid_choices(duration, f"{key}.rest.duration_min")
    _check_schema(schema)
    return schema
