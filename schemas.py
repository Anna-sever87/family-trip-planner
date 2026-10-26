"""Модели данных: запрос на поездку и готовый маршрут."""
from datetime import date
import re
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


INTEREST_LABELS = {
    "main_sights": "основные достопримечательности",
    "museums": "музеи и выставки",
    "theatre": "театр",
    "cinema": "кино",
    "water_trips": "морские и речные прогулки",
    "parks": "парки и природа",
    "zoos": "зоопарки и аквариумы",
    "amusement_parks": "аттракционы",
    "shopping_malls": "торговые центры и шопинг",
}
INTEREST_CATEGORIES = tuple(INTEREST_LABELS)


class TripRequest(BaseModel):
    """Что пользователь указал в анкете."""
    destination: str
    destination_kind: Literal["city", "region"] = "city"
    start_date: str                      # формат ГГГГ-ММ-ДД
    end_date: str                        # формат ГГГГ-ММ-ДД
    adults: int = 2
    children_ages: List[int] = Field(default_factory=list)
    budget: float = Field(gt=0, le=1_000_000_000, allow_inf_nan=False)
    travel_cost: float = Field(default=0, ge=0, le=1_000_000_000, allow_inf_nan=False)
    lodging_cost: float = Field(default=0, ge=0, le=1_000_000_000, allow_inf_nan=False)
    currency: str = "EUR"
    pace: str = "средний"                # спокойный, средний или насыщенный
    start_time: str = "09:00"
    end_time: str = "20:00"
    trip_goal: str = ""
    interest_categories: List[str] = Field(default_factory=list)
    location_preference: str = "anywhere"
    stay_area: str = ""
    wishes: str = ""

    @property
    def fixed_costs(self) -> float:
        """Введённые пользователем суммы за дорогу и всё проживание."""
        return round(self.travel_cost + self.lodging_cost, 2)

    @property
    def program_budget(self) -> float:
        """Сколько из общего бюджета доступно для программы поездки."""
        return round(self.budget - self.fixed_costs, 2)

    @property
    def days_count(self) -> int:
        start = date.fromisoformat(self.start_date)
        end = date.fromisoformat(self.end_date)
        return (end - start).days + 1


class ItineraryItem(BaseModel):
    """Один пункт программы дня."""
    time: str                            # ЧЧ:ММ
    kind: str = "attraction"             # attraction (место), meal (обед) или rest (отдых)
    title: str
    area: str = ""                       # район города
    place_query: str = ""                # для поиска координат: «Название, Город, Страна»
    description: str
    duration_min: int = 60
    # JSON schema sent to the model is not a server-side validation boundary.
    cost: float = Field(default=0, ge=0, le=1_000_000_000, allow_inf_nan=False, strict=True)
    kids_note: str = ""
    indoor: bool = False
    # заполняется программой после проверки места на карте (не моделью)
    verified: bool = False
    lat: Optional[float] = None
    lon: Optional[float] = None
    official_name: str = ""
    provider: str = ""                   # app-only map provenance; never trusted from AI
    source_url: str = ""

    @field_validator("description")
    @classmethod
    def remove_repeated_visit_clock(cls, value, info):
        """Keep planned visit times in the structured timeline, including old trips.

        Only the generated visit-time clause is removed. Opening hours and
        transfer descriptions stay intact; parsing does not rewrite storage.
        """
        if info.data.get("kind", "attraction") != "attraction":
            return value
        clock = r"(?:[01]\d|2[0-3]):[0-5]\d"
        cleaned = re.sub(r"(?:^|,\s*)посещение\s+с\s+" + clock +
                         r"\s+до\s+" + clock + r"(?=\s*(?:[.;]|$))", "", value,
                         flags=re.IGNORECASE)
        if cleaned == value:
            return value
        cleaned = cleaned.strip(" ,;.")
        return (cleaned + ".") if cleaned else "Время посещения указано в расписании."


class ItineraryDay(BaseModel):
    day: int
    date: str
    theme: str
    items: List[ItineraryItem]
    base_area: str = ""                  # населённый пункт или район базы дня
    transfer_note: str = ""              # ориентир по межрайонному переезду
    rainy_alternative: str = ""
    # Server-owned. A model response cannot authorize an empty program day.
    day_mode: Literal["program", "free"] = "program"


class LimitedProgramReport(BaseModel):
    """Bounded server-owned disclosure; not proof of map/source validation."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=1, ge=1, le=1, strict=True)
    reason: Literal["limited_verified_program"] = "limited_verified_program"
    # Old persisted source reports have no basis field and retain their meaning.
    coverage_basis: Literal["source", "map"] = "source"
    requested_days: int = Field(ge=1, le=14, strict=True)
    program_days: int = Field(ge=1, le=14, strict=True)
    free_days: int = Field(ge=1, le=13, strict=True)
    eligible_unique_visits: int = Field(ge=1, le=100, strict=True)
    selected_unique_visits: int = Field(ge=1, le=100, strict=True)


class Itinerary(BaseModel):
    """Готовый маршрут, который возвращает модель."""
    title: str
    summary: str
    days: List[ItineraryDay]
    tips: List[str] = Field(default_factory=list)
    # Set by the local source scheduler, never trusted from model output.
    source_schedule: bool = False
    # Separate server-owned map-candidate pipeline; never claims VisitFacts.
    city_schedule: bool = Field(default=False, strict=True)
    program_report: Optional[LimitedProgramReport] = None
