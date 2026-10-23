"""Ориентиры стоимости для проверки бюджета поездки."""

from typing import Iterable, Optional, Protocol, Union, overload


MEAL_COST_PER_ADULT = {
    "RUB": 700.0,
    "EUR": 12.0,
    "USD": 12.0,
}
MEAL_COST_PER_CHILD = {
    "RUB": 400.0,
    "EUR": 7.0,
    "USD": 7.0,
}


class _TripComposition(Protocol):
    currency: str
    adults: int
    children_ages: Iterable[int]


@overload
def minimum_meal_cost(currency: _TripComposition) -> float:
    ...


@overload
def minimum_meal_cost(
    currency: str,
    adults: int,
    children_ages: Iterable[int] = (),
) -> float:
    ...


def minimum_meal_cost(
    currency: Union[str, _TripComposition],
    adults: Optional[int] = None,
    children_ages: Iterable[int] = (),
) -> float:
    """Возвращает минимальную стоимость одного обеда на всю группу.

    Для удобства принимает как ``TripRequest``, так и отдельные значения
    валюты, числа взрослых и возрастов детей. Это сохраняет совместимость
    с кодом, который уже работает непосредственно с полями анкеты.
    """

    if isinstance(currency, str):
        currency_code = currency.strip().upper()
        if adults is None:
            raise TypeError("Нужно указать число взрослых.")
        adult_count = int(adults)
        ages = children_ages
    else:
        if adults is not None or children_ages:
            raise TypeError(
                "При передаче TripRequest состав группы указывать не нужно."
            )
        currency_code = str(currency.currency).strip().upper()
        adult_count = int(currency.adults)
        ages = currency.children_ages

    if adult_count < 0:
        raise ValueError("Число взрослых не может быть отрицательным.")

    try:
        adult_cost = MEAL_COST_PER_ADULT[currency_code]
        child_cost = MEAL_COST_PER_CHILD[currency_code]
    except KeyError as error:
        raise ValueError(f"Неподдерживаемая валюта: {currency_code!r}.") from error

    child_count = len(tuple(ages or ()))
    return adult_cost * adult_count + child_cost * child_count
