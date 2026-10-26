"""Офлайн-регрессии веб-запроса для города и территории."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

import web_service
from destination_scope import DestinationScopeError
from schemas import Itinerary, ItineraryDay, TripRequest
from web_app import TripCreateRequest


BASE_DIR = Path(__file__).resolve().parent


def _request_data(**overrides):
    data = {
        "destination": "Байкал, Россия",
        "start_date": "2099-06-01",
        "end_date": "2099-06-01",
        "adults": 2,
        "children_ages": [],
        "budget": 10_000.0,
        "travel_cost": 0.0,
        "lodging_cost": 0.0,
        "currency": "EUR",
        "pace": "спокойный",
        "start_time": "09:00",
        "end_time": "19:00",
        "trip_goal": "Увидеть природу Байкала",
        "interest_categories": ["parks"],
        "location_preference": "anywhere",
        "stay_area": "",
        "wishes": "",
    }
    data.update(overrides)
    return data


def _itinerary() -> Itinerary:
    return Itinerary(
        title="Байкал за один день",
        summary="Знакомство с выбранной территорией.",
        days=[
            ItineraryDay(
                day=1,
                date="2099-06-01",
                theme="Берег Байкала",
                items=[],
                base_area="Листвянка",
                transfer_note=(
                    "Переезд из Иркутска около 1,5 часа; зимой проверьте дорогу."
                ),
            )
        ],
    )


def test_legacy_request_defaults_to_city() -> None:
    request = TripRequest.model_validate(_request_data(destination="Казань, Россия"))
    payload = TripCreateRequest.model_validate(
        _request_data(destination="Казань, Россия")
    )
    assert request.destination_kind == "city"
    assert payload.destination_kind == "city"


def test_region_kind_survives_api_and_service_normalization() -> None:
    payload = TripCreateRequest.model_validate(
        _request_data(destination_kind="region")
    )
    request = TripRequest.model_validate(payload.model_dump())
    normalized = web_service._normalized_trip_request(request)
    assert normalized.destination == "Байкал, Россия"
    assert normalized.destination_kind == "region"


def test_unknown_destination_kind_is_rejected() -> None:
    for model in (TripRequest, TripCreateRequest):
        try:
            model.model_validate(_request_data(destination_kind="country"))
        except ValidationError:
            continue
        raise AssertionError(f"{model.__name__} приняла неизвестный тип направления")


def test_legacy_day_gets_empty_regional_fields() -> None:
    day = ItineraryDay.model_validate(
        {
            "day": 1,
            "date": "2099-06-01",
            "theme": "Прогулка",
            "items": [],
        }
    )
    assert day.base_area == ""
    assert day.transfer_note == ""


def test_region_detail_roundtrip_includes_kind_base_and_transfer() -> None:
    request = TripRequest.model_validate(_request_data(destination_kind="region"))
    itinerary = _itinerary()
    trip = {
        "id": 77,
        "request": request.model_dump(),
        "itinerary": itinerary.model_dump(),
        "revision": 1,
        "created_at": "2099-01-01T00:00:00+00:00",
    }
    with (
        patch.object(
            web_service,
            "_validated_trip",
            return_value=(trip, request, itinerary),
        ),
        patch.object(web_service.db, "get_checklist", return_value=[]),
        patch.object(web_service.db, "get_reminder_statuses", return_value={}),
        patch.object(web_service.reminders, "describe_plan", return_value=""),
    ):
        detail = web_service.get_trip_detail(1, 77)

    assert detail["destination_kind"] == "region"
    assert detail["request"]["destination_kind"] == "region"
    day = detail["itinerary"]["days"][0]
    assert day["base_area"] == "Листвянка"
    assert "Начало программы на базе: Листвянка" in day["transfer_note"]
    assert "заселение планируются отдельно" in day["transfer_note"]
    assert "Иркутска" in itinerary.days[0].transfer_note
    assert trip["itinerary"] == itinerary.model_dump()
    note = detail["destination_scope_note"]
    assert "несколько зон выбранной территории" in note
    assert "береговая зона до 5 км" in note
    assert "для остальных регионов — допуск до 40 км" in note
    assert "технический допуск, не официальная граница" in note
    assert "Базы дней и переезды" in note
    assert "сезонную доступность" in note
    assert "стоимость" in note
    assert "проверить отдельно" in note
    assert "не подтверждает бронирование" in note


def test_old_saved_request_is_city_in_list_summary() -> None:
    old_request = _request_data(destination="Казань, Россия")
    old_request.pop("destination_kind", None)
    trip = {
        "id": 78,
        "request": old_request,
        "itinerary": _itinerary().model_dump(),
        "revision": 1,
    }
    with patch.object(web_service.db, "get_checklist", return_value=[]):
        summary = web_service._summary_from_trip(trip)
    assert summary["destination_kind"] == "city"
    assert summary["destination_scope_note"] == ""


def test_web_form_submits_and_renders_regional_fields() -> None:
    html = (BASE_DIR / "web" / "index.html").read_text(encoding="utf-8")
    javascript = (BASE_DIR / "web" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert 'id="tripDestinationKind"' in html
    assert '<option value="city">Город</option>' in html
    assert '<option value="region">Регион, озеро или территория</option>' in html
    assert "destination_kind: destinationKind" in javascript
    assert "request.destination_kind" in javascript
    assert "day.base_area" in javascript
    assert "day.transfer_note" in javascript


def test_region_scope_error_becomes_web_validation_error() -> None:
    request = TripRequest.model_validate(_request_data(destination_kind="region"))
    with (
        patch.object(web_service.reminders, "get_days_before", return_value=1),
        patch.object(
            web_service.db,
            "claim_ai_request",
            return_value=SimpleNamespace(
                status="claimed",
                claim_id="region-scope-test",
                limit=10,
            ),
        ),
        patch.object(web_service.vector_memory, "recall_family_memories", return_value=[]),
        patch.object(web_service.vector_memory, "format_memory_context", return_value=""),
        patch.object(
            web_service,
            "generate_itinerary",
            side_effect=DestinationScopeError("Уточните название территории."),
        ),
        patch.object(web_service, "_finish_ai_request_safely") as finish,
    ):
        try:
            web_service.create_trip(1, request, expected_account_key="account")
        except web_service.TripCreateValidationError as error:
            assert str(error) == "Уточните название территории."
        else:
            raise AssertionError("Ошибка территории не преобразована в веб-ошибку")

    finish.assert_called_once_with("region-scope-test", succeeded=False)


if __name__ == "__main__":
    tests = [
        test_legacy_request_defaults_to_city,
        test_region_kind_survives_api_and_service_normalization,
        test_unknown_destination_kind_is_rejected,
        test_legacy_day_gets_empty_regional_fields,
        test_region_detail_roundtrip_includes_kind_base_and_transfer,
        test_old_saved_request_is_city_in_list_summary,
        test_web_form_submits_and_renders_regional_fields,
        test_region_scope_error_becomes_web_validation_error,
    ]
    for test in tests:
        test()
        print(f"[OK] {test.__name__}")
