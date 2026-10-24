"""Автономная проверка безопасности и API веб-кабинета."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import db
import planner


TEST_DB = Path(__file__).with_name("test_web.db")
USER_ONE = 92001
USER_TWO = 92002
USER_CREATE = 92003
USER_FAIL = 92004
USER_GUARD = 92005
USER_REJECT = 92006
USER_FREE_CREATE = 92007
USER_EXACT_CREATE = 92008


def _ok(text: str) -> None:
    print(f"[OK] {text}")


def _fingerprint(path: Path):
    if not path.exists():
        return None
    content = path.read_bytes()
    return len(content), hashlib.sha256(content).hexdigest()


def _request(destination: str) -> dict:
    return {
        "destination": destination,
        "start_date": "2026-11-10",
        "end_date": "2026-11-10",
        "adults": 2,
        "children_ages": [5],
        "budget": 300,
        "currency": "EUR",
        "pace": "спокойный",
        "wishes": "интерактивный музей и дневной отдых",
        "weather_text": "Без заметных осадков.",
    }


def _itinerary(title: str, place: str) -> dict:
    return {
        "title": title,
        "summary": "Спокойный семейный маршрут на один день.",
        "days": [
            {
                "day": 1,
                "date": "2026-11-10",
                "theme": "Музей и прогулка",
                "items": [
                    {
                        "time": "09:00",
                        "kind": "attraction",
                        "title": place,
                        "area": "Центр",
                        "place_query": f"{place}, Europe",
                        "description": "Семейное посещение.",
                        "duration_min": 120,
                        "cost": 40,
                        "kids_note": "Есть занятия для ребёнка.",
                        "indoor": True,
                        "verified": True,
                        "lat": 48.86,
                        "lon": 2.34,
                        "official_name": place,
                    },
                    {
                        "time": "12:30",
                        "kind": "meal",
                        "title": "Обед",
                        "area": "Центр",
                        "place_query": "",
                        "description": "Обед всей семьёй.",
                        "duration_min": 60,
                        "cost": 50,
                        "kids_note": "",
                        "indoor": True,
                    },
                    {
                        "time": "13:30",
                        "kind": "rest",
                        "title": "Дневной отдых",
                        "area": "Центр",
                        "place_query": "",
                        "description": "Спокойный отдых после обеда.",
                        "duration_min": 60,
                        "cost": 0,
                        "kids_note": "",
                        "indoor": True,
                    },
                ],
                "rainy_alternative": "Крытый детский центр",
            }
        ],
        "tips": ["Купить билеты заранее."],
    }


def _new_trip_payload(destination: str = "Лион, Франция") -> dict:
    start = datetime.now(timezone.utc).date() + timedelta(days=30)
    return {
        "destination": destination,
        "start_date": start.isoformat(),
        "end_date": start.isoformat(),
        "adults": 2,
        "children_ages": [5],
        "budget": 300,
        "currency": "EUR",
        "pace": "спокойный",
        "start_time": "10:00",
        "end_time": "21:00",
        "trip_goal": "Познакомиться с городом и отдохнуть",
        "interest_categories": ["main_sights", "museums"],
        "location_preference": "nearby",
        "stay_area": "Presqu'île",
        "wishes": "интерактивный музей и дневной отдых",
    }


def _generation_result(payload: dict, itinerary_cls):
    itinerary_data = _itinerary(
        "Семейный Лион",
        "Musée des Confluences",
    )
    itinerary_data["days"][0]["date"] = payload["start_date"]
    return SimpleNamespace(
        problems=[],
        itinerary=itinerary_cls.model_validate(itinerary_data),
        weather_text="Свежая тестовая погода.",
        weather={"source": "test", "days": []},
        attempts=1,
        prompt_tokens=12,
        completion_tokens=34,
        total_tokens=46,
    )


def _vorkuta_candidate(payload: dict) -> dict:
    start = datetime.fromisoformat(
        payload["start_date"]
    ).date()
    places = (
        (
            "Монумент шахтёрской славы",
            "Монумент шахтёрской славы, Воркута, Россия",
            "Воркутинский краеведческий музей",
            "Воркутинский краеведческий музей, Воркута, Россия",
        ),
        (
            "Воркутинский выставочный центр",
            "Воркутинский выставочный центр, Воркута, Россия",
            "Дворец культуры шахтёров",
            "Дворец культуры шахтёров, Воркута, Россия",
        ),
        (
            "Парк Победы Воркуты",
            "Парк Победы, Воркута, Россия",
            "Полярный геофизический институт",
            "Полярный геофизический институт, Воркута, Россия",
        ),
        (
            "Воркутинский драматический театр",
            "Воркутинский драматический театр, Воркута, Россия",
            "Стела «Воркута — город трудовой доблести»",
            "МО ГО Воркута, Центральная площадь, 7, Воркута, Россия",
        ),
    )
    rainy = (
        "Посещение Монумент шахтёрской славы",
        "Посещение библиотеки",
        "Посещение библиотеки",
        "",
    )
    days = []

    for index, place_data in enumerate(
        places,
        start=1,
    ):
        first_title, first_query, second_title, second_query = (
            place_data
        )
        second_time = "18:00" if index == 4 else "15:00"
        days.append(
            {
                "day": index,
                "date": (
                    start
                    + timedelta(days=index - 1)
                ).isoformat(),
                "theme": f"Знакомство с Воркутой — день {index}",
                "items": [
                    {
                        "time": "10:00",
                        "kind": "attraction",
                        "title": first_title,
                        "area": "Центр",
                        "place_query": first_query,
                        "description": "Знакомство с реальным городским объектом.",
                        "duration_min": 120,
                        "cost": 0,
                        "kids_note": "",
                        "indoor": index != 3,
                    },
                    {
                        "time": "12:15",
                        "kind": "meal",
                        "title": "Обед",
                        "area": "Центр",
                        "place_query": "",
                        "description": "Обед в городском кафе.",
                        "duration_min": 60,
                        "cost": 1400,
                        "kids_note": "",
                        "indoor": True,
                    },
                    {
                        "time": "13:30",
                        "kind": "rest",
                        "title": "Отдых",
                        "area": "Центр",
                        "place_query": "",
                        "description": "Спокойная пауза после обеда.",
                        "duration_min": 60,
                        "cost": 0,
                        "kids_note": "",
                        "indoor": True,
                    },
                    {
                        "time": second_time,
                        "kind": "attraction",
                        "title": second_title,
                        "area": "Центр",
                        "place_query": second_query,
                        "description": "Посещение другого реального объекта.",
                        "duration_min": 90,
                        "cost": 0,
                        "kids_note": "",
                        "indoor": True,
                    },
                ],
                "rainy_alternative": rainy[index - 1],
            }
        )

    return {
        "title": "Воркута: четыре дня знакомства",
        "summary": "Компактный маршрут без выдуманных мест.",
        "days": days,
        "tips": [
            "Тёплая одежда обязательна.",
            "Проверяйте расписание заранее.",
        ],
    }


def _login(client, tg_id: int, web_auth) -> tuple[str, str]:
    login_url = web_auth.build_login_url(tg_id)
    parsed = urlsplit(login_url)
    token = parse_qs(parsed.fragment)["login"][0]
    response = client.post("/api/login", json={"token": token})
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert web_auth.SESSION_COOKIE in client.cookies
    csrf = client.cookies.get(web_auth.CSRF_COOKIE)
    assert csrf
    return token, csrf


def _csrf_headers(token: str) -> dict:
    return {"X-CSRF-Token": token}


def _remove_test_files() -> None:
    for path in (
        TEST_DB,
        TEST_DB.with_name(TEST_DB.name + "-wal"),
        TEST_DB.with_name(TEST_DB.name + "-shm"),
    ):
        path.unlink(missing_ok=True)


def main() -> None:
    original_db_path = db.DB_PATH
    real_db_path = Path(original_db_path).resolve()
    real_db_before = _fingerprint(real_db_path)
    env_names = (
        "WEB_SESSION_SECRET",
        "WEB_BASE_URL",
        "WEB_COOKIE_SECURE",
        "WEB_LOGIN_TOKEN_MINUTES",
        "WEB_SESSION_HOURS",
        "DAILY_LIMIT",
        "CITY_CANDIDATES_FIRST",
    )
    old_env = {name: os.environ.get(name) for name in env_names}

    try:
        _remove_test_files()
        db.DB_PATH = str(TEST_DB)
        os.environ["WEB_SESSION_SECRET"] = (
            "web-test-secret-that-is-longer-than-thirty-two-characters"
        )
        os.environ["WEB_BASE_URL"] = "http://127.0.0.1:8000"
        os.environ["WEB_COOKIE_SECURE"] = "false"
        os.environ["WEB_LOGIN_TOKEN_MINUTES"] = "15"
        os.environ["WEB_SESSION_HOURS"] = "1"
        os.environ["DAILY_LIMIT"] = "5"
        os.environ["CITY_CANDIDATES_FIRST"] = "0"

        import web_auth
        import web_app
        import web_service
        from fastapi.testclient import TestClient
        from schemas import Itinerary
        from web_app import app

        warning_itinerary = Itinerary.model_validate(
            _itinerary(
                "Честный бесплатный маршрут",
                "Бесплатный музей",
            )
        )
        warning_request = web_service.TripRequest.model_validate(
            _new_trip_payload(
                "Небольшой город"
            )
        )
        priced_note = web_service._budget_scope_note(
            warning_request,
            warning_itinerary,
        )
        assert "нет подтверждённой стоимости билетов" not in (
            priced_note
        )
        for warning_day in warning_itinerary.days:
            for warning_item in warning_day.items:
                if warning_item.kind == "attraction":
                    warning_item.cost = 0
        warning_note = web_service._budget_scope_note(
            warning_request,
            warning_itinerary
        )
        assert "нет подтверждённой стоимости билетов" in (
            warning_note
        )
        assert "Не указаны и не учтены: дорога, жильё" in (
            warning_note
        )
        _ok("веб-бюджет показывает предупреждение о неподтверждённой цене")

        db.init_db()
        db.save_user(
            USER_ONE,
            name="Анна",
            adults=2,
            children_ages=[5],
            currency="EUR",
            pace="спокойный",
            wishes="интерактивные музеи",
        )
        db.save_user(
            USER_TWO,
            name="Другой пользователь",
            adults=1,
            children_ages=[],
            currency="RUB",
            pace="средний",
            wishes="",
        )
        db.save_user(
            USER_CREATE,
            name="Веб-пользователь",
            adults=1,
            children_ages=[],
            currency="RUB",
            pace="средний",
            wishes="старые пожелания",
        )
        db.save_user(
            USER_FREE_CREATE,
            name="Путешественник",
            adults=2,
            children_ages=[5],
            currency="EUR",
            pace="спокойный",
            wishes="",
        )
        db.save_user(
            USER_EXACT_CREATE,
            name="Путешественник Воркуты",
            adults=2,
            children_ages=[],
            currency="RUB",
            pace="насыщенный",
            wishes="",
        )
        db.save_user(
            USER_FAIL,
            name="Пользователь ошибки",
            adults=1,
            children_ages=[],
            currency="USD",
            pace="насыщенный",
            wishes="профиль до ошибки",
        )
        db.save_user(
            USER_GUARD,
            name="Пользователь лимита",
            adults=1,
            children_ages=[],
            currency="RUB",
            pace="средний",
            wishes="профиль до ограничения",
        )
        db.save_user(
            USER_REJECT,
            name="Пользователь отклонённого маршрута",
            adults=1,
            children_ages=[],
            currency="RUB",
            pace="средний",
            wishes="профиль до отклонения",
        )

        trip_one = db.save_trip(
            USER_ONE,
            _request("Париж, Франция"),
            _itinerary("Семейный Париж", "Musée de Montmartre"),
        )
        trip_two = db.save_trip(
            USER_TWO,
            _request("Рим, Италия"),
            _itinerary("Семейный Рим", "Museo Explora"),
        )
        db.save_checklist(
            trip_one,
            [("Документы", "Паспорта"), ("Для ребёнка", "Любимая книга")],
        )
        db.save_checklist(
            trip_two,
            [("Документы", "Чужой паспорт")],
        )
        own_item = db.get_checklist(trip_one)[0]["id"]
        foreign_item = db.get_checklist(trip_two)[0]["id"]

        reminder_time = datetime(2026, 11, 7, 3, tzinfo=timezone.utc)
        assert db.claim_reminder(trip_one, "before", reminder_time)
        assert db.mark_reminder_sent(trip_one, "before", reminder_time)

        with TestClient(app) as owner_client:
            health = owner_client.get("/health")
            page = owner_client.get("/")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}
            assert page.status_code == 200
            assert "Семейный планировщик" in page.text
            assert "text/html" in page.headers["content-type"]
            assert "/static/vendor/leaflet.js" in page.text
            assert "unpkg.com" not in page.text
            assert page.headers["cache-control"] == "no-store, private"
            assert page.headers["referrer-policy"] == (
                "strict-origin-when-cross-origin"
            )
            content_security_policy = page.headers["content-security-policy"]
            assert "default-src 'self'" in content_security_policy
            assert (
                "img-src 'self' data: https://tile.openstreetmap.org;"
                in content_security_policy
            )
            assert "https://*.tile.openstreetmap.org" not in (
                content_security_policy
            )
            assert owner_client.get(
                "/static/vendor/leaflet.js"
            ).status_code == 200
            app_js = owner_client.get("/static/app.js")
            assert app_js.status_code == 200
            styles = owner_client.get("/static/styles.css")
            assert styles.status_code == 200
            assert "https://tile.openstreetmap.org/{z}/{x}/{y}.png" in (
                app_js.text
            )
            assert "{s}.tile.openstreetmap.org" not in app_js.text
            assert "openstreetmap.org/copyright" in app_js.text
            assert "contributors" in app_js.text
            for element_id in (
                "newTripForm",
                "tripDestination",
                "tripStartDate",
                "tripEndDate",
                "tripAdults",
                "tripChildrenAges",
                "tripNoChildren",
                "tripBudget",
                "tripTravelCost",
                "tripLodgingCost",
                "tripProgramBudget",
                "tripCurrency",
                "tripPace",
                "tripStartTime",
                "tripEndTime",
                "tripGoal",
                "tripLocationPreference",
                "tripStayArea",
                "tripWishes",
                "newTripMessage",
                "submitNewTrip",
                "closeInfoDialog",
                "cancelNewTrip",
                "tripsGrid",
                "tripResultsStatus",
                "tripSearch",
                "tripHero",
                "tripCoverControls",
            ):
                assert f'id="{element_id}"' in page.text
            assert "Веб-анкета — следующий шаг проекта" not in page.text
            for frontend_hook in (
                "pageSize: 8",
                "yearPages: new Map()",
                'api("/api/trips?limit=100")',
                "groupTripsByYear(sorted)",
                "function groupTripsByYear",
                "function createTripYearSection",
                "function createTripYearPagination",
                '$("tripResultsStatus").textContent = tripCountLabel(filtered.length)',
                "const expandMatchingGroups = Boolean(query);",
                'window.matchMedia("(prefers-reduced-motion: reduce)").matches',
                'other.setAttribute("aria-pressed", String(isActive))',
                "function submitNewTrip",
                'api("/api/trips",',
                "state.yearPages.clear()",
                "interest_categories",
                "remaining_after_day",
            ):
                assert frontend_hook in app_js.text
            for css_hook in (
                ".trip-list-tools",
                ".trip-search",
                ".trip-years",
                ".trip-year-group",
                ".trip-year-summary",
                '.trip-year-summary::marker { content: ""; }',
                ".trip-year-grid",
                ".trip-year-pagination",
                ".travel-hero",
                ".hero-route",
                ".trip-card-cover",
                ".trip-card-content",
                ".trip-detail-cover",
                ".cover-caption",
                ".trip-explainer",
                ".trip-card",
                ".pagination",
                ".trip-form-dialog",
                ".form-grid",
                ".interest-chip",
                "prefers-reduced-motion",
            ):
                assert css_hook in styles.text
            assert 'id="tripsGrid" class="trip-years" aria-live=' not in page.text
            assert 'id="tripResultsStatus" class="visually-hidden" aria-live="polite"' in page.text
            assert page.text.count('aria-pressed="false"') >= 2
            assert 'aria-pressed="true"' in page.text
            assert "/static/styles.css?v=20261024-day-order-quality" in page.text
            assert "/static/app.js?v=20261024-day-order-quality" in page.text
            assert 'class="hero travel-hero"' in page.text
            assert "ИИ-иллюстрация путешествия" in page.text
            assert page.text.count("/static/favicon.svg?v=20261022-premium-travel") == 3
            assert owner_client.get("/api/me").status_code == 401
            _ok(
                "health, защищённый HTML, локальная карта, веб-анкета, "
                "компактный список и пагинация работают"
            )

            login_token, owner_csrf = _login(owner_client, USER_ONE, web_auth)
            replay = owner_client.post(
                "/api/login",
                json={"token": login_token},
            )
            assert replay.status_code == 401
            assert "уже использована или устарела" in replay.text
            _ok("одноразовая ссылка создаёт cookie и повторно не принимается")

            profile = owner_client.get("/api/me")
            assert profile.status_code == 200
            assert profile.json() == {
                "name": "Анна",
                "adults": 2,
                "children_ages": [5],
                "currency": "EUR",
                "pace": "спокойный",
                "wishes": "интерактивные музеи",
                "capabilities": {
                    "city_candidates_first": False,
                    "city_mvp_disclosure": "",
                    "cover_images": True,
                },
                "access": {
                    "owner": False,
                    "allowed": True,
                    "expires_at": None,
                    "route_remaining": None,
                    "route_limit": None,
                    "grant_id": "open",
                },
                "generation_jobs": {
                    "enabled": True,
                    "owner": __import__("generation_jobs").owner_token(db.get_account_key(USER_ONE)),
                },
            }

            trips = owner_client.get("/api/trips")
            assert trips.status_code == 200
            assert [item["id"] for item in trips.json()["trips"]] == [trip_one]
            detail = owner_client.get(f"/api/trips/{trip_one}")
            assert detail.status_code == 200
            assert detail.json()["destination"] == "Париж, Франция"
            assert detail.json()["checklist_progress"] == {"done": 0, "total": 2}
            assert detail.json()["request"]["travel_cost"] == 0
            assert detail.json()["request"]["lodging_cost"] == 0
            assert detail.json()["budget"]["planned_total"] == 90
            assert detail.json()["budget"]["remaining"] == 210
            assert "не учтены: дорога, жильё" in detail.json()["budget"]["scope_note"]

            foreign_detail = owner_client.get(f"/api/trips/{trip_two}")
            missing_detail = owner_client.get("/api/trips/99999999")
            assert foreign_detail.status_code == missing_detail.status_code == 404
            assert foreign_detail.json() == missing_detail.json() == {
                "detail": "Поездка не найдена."
            }
            _ok("профиль, список и детали показывают только поездки владельца")

            pdf = owner_client.get(f"/api/trips/{trip_one}/pdf")
            foreign_pdf = owner_client.get(f"/api/trips/{trip_two}/pdf")
            assert pdf.status_code == 200
            assert pdf.content.startswith(b"%PDF")
            assert pdf.headers["content-type"].startswith("application/pdf")
            assert foreign_pdf.status_code == 404

            reminder = owner_client.get(f"/api/trips/{trip_one}/reminders")
            foreign_reminder = owner_client.get(f"/api/trips/{trip_two}/reminders")
            assert reminder.status_code == 200
            assert "Напоминания по поездке" in reminder.json()["text"]
            assert "отправлено" in reminder.json()["text"]
            assert foreign_reminder.status_code == 404
            _ok("PDF и напоминания доступны владельцу и закрыты от чужого чтения")

            no_csrf = owner_client.patch(
                f"/api/checklist-items/{own_item}",
                json={"done": True},
            )
            wrong_csrf = owner_client.patch(
                f"/api/checklist-items/{own_item}",
                json={"done": True},
                headers=_csrf_headers("wrong-csrf-token"),
            )
            assert no_csrf.status_code == wrong_csrf.status_code == 403
            assert not db.get_checklist_item(own_item, USER_ONE)["done"]

            changed = owner_client.patch(
                f"/api/checklist-items/{own_item}",
                json={"done": True},
                headers=_csrf_headers(owner_csrf),
            )
            assert changed.status_code == 200
            assert bool(changed.json()["item"]["done"])
            assert bool(db.get_checklist_item(own_item, USER_ONE)["done"])

            foreign_change = owner_client.patch(
                f"/api/checklist-items/{foreign_item}",
                json={"done": True},
                headers=_csrf_headers(owner_csrf),
            )
            assert foreign_change.status_code == 404
            assert not db.get_checklist_item(foreign_item, USER_TWO)["done"]
            _ok("чек-лист требует CSRF и изменяется только владельцем")

            with TestClient(app) as other_client:
                _, other_csrf = _login(other_client, USER_TWO, web_auth)
                assert other_client.get(f"/api/trips/{trip_one}").status_code == 404
                other_change = other_client.patch(
                    f"/api/checklist-items/{own_item}",
                    json={"done": False},
                    headers=_csrf_headers(other_csrf),
                )
                assert other_change.status_code == 404
                assert bool(db.get_checklist_item(own_item, USER_ONE)["done"])
            _ok("второй пользователь не видит и не меняет данные первого")

            before_success = db.get_trip(trip_one, USER_ONE)
            successful_itinerary = Itinerary.model_validate(
                before_success["itinerary"]
            ).model_copy(deep=True)
            successful_itinerary.days[0].theme = "Интерактивный музей и отдых"
            successful_result = SimpleNamespace(
                problems=[],
                itinerary=successful_itinerary,
                weather_text="Свежая тестовая погода.",
                weather={"source": "test"},
                attempts=1,
                prompt_tokens=12,
                completion_tokens=34,
            )
            with patch.object(
                web_service,
                "edit_itinerary",
                return_value=successful_result,
            ):
                edited = owner_client.post(
                    f"/api/trips/{trip_one}/days/1/edit",
                    json={"instruction": "Добавь интерактивный музей"},
                    headers=_csrf_headers(owner_csrf),
                )

            assert edited.status_code == 200, edited.text
            assert edited.json()["itinerary"]["days"][0]["theme"] == (
                "Интерактивный музей и отдых"
            )
            stored_success = db.get_trip(trip_one, USER_ONE)
            assert stored_success["itinerary"]["days"][0]["theme"] == (
                "Интерактивный музей и отдых"
            )
            assert stored_success["revision"] == before_success["revision"] + 1
            _ok("редактирование дня сохраняет проверенный результат без внешних вызовов")

            before_rejection = copy.deepcopy(db.get_trip(trip_one, USER_ONE))
            rejected_result = SimpleNamespace(
                problems=["Место не прошло проверку карты."],
            )
            with patch.object(
                web_service,
                "edit_itinerary",
                return_value=rejected_result,
            ):
                rejected = owner_client.post(
                    f"/api/trips/{trip_one}/days/1/edit",
                    json={"instruction": "Замени место другим"},
                    headers=_csrf_headers(owner_csrf),
                )
            assert rejected.status_code == 422
            assert "Старый маршрут сохранён" in rejected.text
            assert db.get_trip(trip_one, USER_ONE) == before_rejection
            _ok("маршрут с замечаниями не сохраняется")

            def concurrent_edit(request, current, target_day, instruction):
                current_trip = db.get_trip(trip_one, USER_ONE)
                concurrent_plan = copy.deepcopy(current_trip["itinerary"])
                concurrent_plan["title"] = "Изменение из другого окна"
                assert db.update_trip_plan(
                    trip_one,
                    USER_ONE,
                    current_trip["request"],
                    concurrent_plan,
                    expected_revision=current_trip["revision"],
                )
                candidate = current.model_copy(deep=True)
                candidate.days[0].theme = "Конфликтующий ответ AI"
                return SimpleNamespace(
                    problems=[],
                    itinerary=candidate,
                    weather_text="Погода из конфликтующего ответа.",
                    weather={"source": "test"},
                    attempts=1,
                    prompt_tokens=1,
                    completion_tokens=1,
                )

            with patch.object(
                web_service,
                "edit_itinerary",
                side_effect=concurrent_edit,
            ):
                conflict = owner_client.post(
                    f"/api/trips/{trip_one}/days/1/edit",
                    json={"instruction": "Обнови день ещё раз"},
                    headers=_csrf_headers(owner_csrf),
                )
            assert conflict.status_code == 409
            after_conflict = db.get_trip(trip_one, USER_ONE)
            assert after_conflict["itinerary"]["title"] == (
                "Изменение из другого окна"
            )
            assert after_conflict["itinerary"]["days"][0]["theme"] != (
                "Конфликтующий ответ AI"
            )
            _ok("конфликт версий не перезаписывает более свежий маршрут")

            usage_date = datetime.now(
                web_service.reminders.get_tz()
            ).date()
            assert db.get_ai_request_usage(USER_ONE, usage_date) == 3

            active_claim = db.claim_ai_request(
                USER_ONE,
                usage_date,
                "edit",
                "test",
                trip_id=trip_one,
                daily_limit=5,
            )
            assert active_claim.status == "ok"
            with patch.object(
                web_service,
                "edit_itinerary",
                side_effect=AssertionError("AI не должен вызываться при busy"),
            ):
                busy = owner_client.post(
                    f"/api/trips/{trip_one}/days/1/edit",
                    json={"instruction": "Параллельное изменение"},
                    headers=_csrf_headers(owner_csrf),
                )
            assert busy.status_code == 409
            assert "уже выполняется" in busy.text
            assert db.finish_ai_request(
                active_claim.claim_id,
                succeeded=False,
            )

            last_claim = db.claim_ai_request(
                USER_ONE,
                usage_date,
                "edit",
                "test",
                trip_id=trip_one,
                daily_limit=5,
            )
            assert last_claim.status == "ok"
            assert db.finish_ai_request(
                last_claim.claim_id,
                succeeded=True,
            )
            with patch.object(
                web_service,
                "edit_itinerary",
                side_effect=AssertionError("AI не должен вызываться при лимите"),
            ):
                limited = owner_client.post(
                    f"/api/trips/{trip_one}/days/1/edit",
                    json={"instruction": "Ещё одно изменение"},
                    headers=_csrf_headers(owner_csrf),
                )
            assert limited.status_code == 429
            assert "лимит 5" in limited.text
            assert db.get_ai_request_usage(USER_ONE, usage_date) == 5
            _ok("веб не запускает параллельный AI и соблюдает суточный лимит")

        create_payload = _new_trip_payload()
        create_payload.update(travel_cost=40, lodging_cost=130)
        create_usage_date = datetime.now(
            web_service.reminders.get_tz()
        ).date()

        with TestClient(app) as anonymous_client:
            anonymous = anonymous_client.post(
                "/api/trips",
                json=create_payload,
                headers=_csrf_headers("not-a-session-token"),
            )
            assert anonymous.status_code == 401
            assert anonymous.json() == {
                "detail": "Сессия истекла. Откройте новую ссылку в Telegram-боте."
            }

        with TestClient(app) as create_client:
            _, create_csrf = _login(create_client, USER_CREATE, web_auth)
            profile_before_validation = copy.deepcopy(db.get_user(USER_CREATE))

            no_csrf_create = create_client.post(
                "/api/trips",
                json=create_payload,
            )
            assert no_csrf_create.status_code == 403
            assert no_csrf_create.json() == {
                "detail": "Защитный токен страницы устарел. Обновите страницу."
            }

            today = datetime.now(web_service.reminders.get_tz()).date()
            invalid_payloads = []

            invalid_destination = dict(create_payload)
            invalid_destination["destination"] = "   "
            invalid_payloads.append(
                (invalid_destination, "Укажите город или территорию и страну")
            )

            invalid_date = dict(create_payload)
            invalid_date["start_date"] = "2026-13-01"
            invalid_payloads.append(
                (invalid_date, "Даты должны быть в формате")
            )

            past_date = (today - timedelta(days=1)).isoformat()
            invalid_past = dict(create_payload)
            invalid_past["start_date"] = past_date
            invalid_past["end_date"] = past_date
            invalid_payloads.append((invalid_past, "Дата начала уже прошла"))

            invalid_order = dict(create_payload)
            invalid_order["end_date"] = (
                datetime.fromisoformat(create_payload["start_date"])
                - timedelta(days=1)
            ).date().isoformat()
            invalid_payloads.append(
                (invalid_order, "Дата окончания не может быть раньше")
            )

            invalid_duration = dict(create_payload)
            invalid_duration["end_date"] = (
                datetime.fromisoformat(create_payload["start_date"])
                + timedelta(days=14)
            ).date().isoformat()
            invalid_payloads.append(
                (invalid_duration, "не более чем на 14 дней")
            )

            invalid_child_low = dict(create_payload)
            invalid_child_low["children_ages"] = [-1]
            invalid_payloads.append((invalid_child_low, None))

            invalid_child_high = dict(create_payload)
            invalid_child_high["children_ages"] = [18]
            invalid_payloads.append((invalid_child_high, None))

            invalid_children_count = dict(create_payload)
            invalid_children_count["children_ages"] = [1, 2, 3, 4, 5, 6, 7]
            invalid_payloads.append((invalid_children_count, None))

            invalid_budget = dict(create_payload)
            invalid_budget["budget"] = 0
            invalid_payloads.append((invalid_budget, None))

            for expense_field in ("travel_cost", "lodging_cost"):
                for invalid_amount in (-1, True, "NaN", 1_000_000_001):
                    invalid_expense = dict(create_payload)
                    invalid_expense[expense_field] = invalid_amount
                    invalid_payloads.append((invalid_expense, None))

            exhausted_program = dict(create_payload)
            exhausted_program["travel_cost"] = 170
            invalid_payloads.append((exhausted_program, "должен остаться бюджет"))

            insufficient_meals = dict(create_payload)
            insufficient_meals["budget"] = 200
            invalid_payloads.append((insufficient_meals, "Для одного обеда в день"))

            insufficient_multiday_meals = dict(create_payload)
            insufficient_multiday_meals["budget"] = 234
            insufficient_multiday_meals["end_date"] = (
                datetime.fromisoformat(create_payload["start_date"])
                + timedelta(days=2)
            ).date().isoformat()
            invalid_payloads.append((insufficient_multiday_meals, "хотя бы 93 EUR"))

            invalid_time_order = dict(create_payload)
            invalid_time_order["start_time"] = "20:00"
            invalid_time_order["end_time"] = "10:00"
            invalid_payloads.append(
                (invalid_time_order, "Окончание дня должно быть позже начала")
            )

            invalid_short_day = dict(create_payload)
            invalid_short_day["start_time"] = "10:00"
            invalid_short_day["end_time"] = "13:30"
            invalid_payloads.append(
                (invalid_short_day, "не меньше 4 часов")
            )

            invalid_interests = dict(create_payload)
            invalid_interests["interest_categories"] = []
            invalid_payloads.append((invalid_interests, None))

            duplicate_interests = dict(create_payload)
            duplicate_interests["interest_categories"] = [
                "museums",
                "museums",
            ]
            invalid_payloads.append(
                (duplicate_interests, "Не повторяйте выбранные варианты")
            )

            invalid_nearby = dict(create_payload)
            invalid_nearby["stay_area"] = " "
            invalid_payloads.append(
                (invalid_nearby, "Укажите район проживания")
            )

            with (
                patch.object(
                    web_service.vector_memory,
                    "recall_family_memories",
                    side_effect=AssertionError(
                        "Память не должна вызываться до успешной валидации"
                    ),
                ) as invalid_recall,
                patch.object(
                    web_service,
                    "generate_itinerary",
                    side_effect=AssertionError(
                        "AI не должен вызываться до успешной валидации"
                    ),
                ) as invalid_generate,
                patch.object(
                    web_service,
                    "build_checklist",
                    side_effect=AssertionError(
                        "Чек-лист не должен создаваться до успешной валидации"
                    ),
                ) as invalid_checklist,
                patch.object(
                    web_service.vector_memory,
                    "save_trip_memory",
                    side_effect=AssertionError(
                        "Память не должна сохраняться до успешной валидации"
                    ),
                ) as invalid_memory_save,
            ):
                for invalid_payload, expected_text in invalid_payloads:
                    response = create_client.post(
                        "/api/trips",
                        json=invalid_payload,
                        headers=_csrf_headers(create_csrf),
                    )
                    assert response.status_code == 422, response.text
                    if expected_text is not None:
                        assert expected_text in response.text

            invalid_recall.assert_not_called()
            invalid_generate.assert_not_called()
            invalid_checklist.assert_not_called()
            invalid_memory_save.assert_not_called()
            assert db.get_user(USER_CREATE) == profile_before_validation
            assert db.list_trips(USER_CREATE, 100) == []
            assert db.list_memory_items(USER_CREATE, 100) == []
            assert db.get_ai_request_usage(USER_CREATE, create_usage_date) == 0
            infant_payload = dict(create_payload)
            infant_payload["children_ages"] = [0]
            assert web_app.TripCreateRequest.model_validate(
                infant_payload
            ).children_ages == [0]
            _ok("веб-анкета отклоняет ошибки до AI и не меняет данные")

            matches = [
                SimpleNamespace(
                    memory_id="family-pref#web-source",
                    text="Нужен дневной отдых.",
                    kind="custom",
                    score=0.9,
                    updated_at="",
                    destination="",
                )
            ]
            generated = _generation_result(create_payload, Itinerary)

            def fake_save_trip_memory(
                chat_id,
                user_id,
                journal_tg_id,
                request,
                *,
                trip_id,
            ):
                assert (chat_id, user_id, journal_tg_id) == (
                    USER_CREATE,
                    USER_CREATE,
                    USER_CREATE,
                )
                assert request.destination == create_payload["destination"]
                db.save_memory_item(
                    USER_CREATE,
                    "family-pref#web-created",
                    "profile",
                    "Семья предпочитает спокойный темп.",
                    destination=request.destination,
                    source_trip_id=trip_id,
                )
                return SimpleNamespace(
                    success=True,
                    requested=1,
                    vector_saved=1,
                    journal_saved=1,
                    error_type="",
                )

            with (
                patch.object(
                    web_service.vector_memory,
                    "recall_family_memories",
                    return_value=matches,
                ) as recall_mock,
                patch.object(
                    web_service.vector_memory,
                    "format_memory_context",
                    return_value="КОНТЕКСТ ПАМЯТИ",
                ) as format_memory_mock,
                patch.object(
                    web_service,
                    "generate_itinerary",
                    return_value=generated,
                ) as generate_mock,
                patch.object(web_service, "resolve_city_for_planning", return_value=None),
                patch.object(
                    web_service,
                    "build_checklist",
                    return_value=[("Документы", "Проверить паспорта")],
                ) as checklist_mock,
                patch.object(
                    web_service.vector_memory,
                    "save_trip_memory",
                    side_effect=fake_save_trip_memory,
                ) as save_memory_mock,
            ):
                created = create_client.post(
                    "/api/trips",
                    json=create_payload,
                    headers=_csrf_headers(create_csrf),
                )

            assert created.status_code == 201, created.text
            created_body = created.json()
            created_trip_id = int(created_body["id"])
            assert created_body["destination"] == "Лион, Франция"
            assert created_body["request"] == {
                **create_payload,
                "destination_kind": "city",
                "budget": 300.0,
            }
            assert created_body["weather_text"] == "Свежая тестовая погода."
            assert created_body["checklist_progress"] == {"done": 0, "total": 1}
            assert created_body["budget"] == {
                "currency": "EUR",
                "limit": 300.0,
                "travel_cost": 40.0,
                "lodging_cost": 130.0,
                "fixed_costs": 170.0,
                "program_budget": 130.0,
                "program_total": 90.0,
                "planned_total": 260.0,
                "remaining": 40.0,
                "over_budget": False,
                "unpriced_visit_count": 0,
                "coverage": {"status": "estimate", "missing": [], "notes": []},
                "days": [
                    {
                        "day": 1,
                        "day_total": 90.0,
                        "program_spent_through_day": 90.0,
                        "spent_through_day": 260.0,
                        "planned_through_day": 260.0,
                        "remaining_after_day": 40.0,
                    }
                ],
                "scope_note": (
                    "Плановые суммы, не фактические траты. "
                    "Дорога и жильё — введённые суммы; цены программы ориентировочные."
                ),
            }
            assert created_body["generation"] == {
                "attempts": 1,
                "prompt_tokens": 12,
                "completion_tokens": 34,
                "total_tokens": 46,
                "memory_matches": 1,
                "memory_saved": 1,
            }
            assert "problems" not in created_body

            multi_trip = copy.deepcopy(db.get_trip(created_trip_id, USER_CREATE))
            multi_request = web_service.TripRequest.model_validate(multi_trip["request"])
            multi_request.end_date = (
                datetime.fromisoformat(multi_request.start_date) + timedelta(days=1)
            ).date().isoformat()
            multi_itinerary = generated.itinerary.model_copy(deep=True)
            second_day = multi_itinerary.days[0].model_copy(deep=True)
            second_day.day = 2
            second_day.date = multi_request.end_date
            multi_itinerary.days.append(second_day)
            multi_trip["request"] = multi_request.model_dump()
            multi_trip["itinerary"] = multi_itinerary.model_dump()
            with patch.object(
                web_service,
                "_validated_trip",
                return_value=(multi_trip, multi_request, multi_itinerary),
            ):
                multi_budget = web_service.get_trip_detail(
                    USER_CREATE, created_trip_id
                )["budget"]
            assert multi_budget["program_total"] == 180
            assert multi_budget["planned_total"] == 350
            assert multi_budget["remaining"] == -50
            assert multi_budget["over_budget"] is True
            assert [day["day_total"] for day in multi_budget["days"]] == [90, 90]
            assert [day["planned_through_day"] for day in multi_budget["days"]] == [260, 350]
            assert [day["remaining_after_day"] for day in multi_budget["days"]] == [40, -50]
            multi_summary = web_service._summary_from_trip(multi_trip)
            assert multi_summary["fixed_costs"] == 170
            assert multi_summary["planned_total"] == 350
            assert multi_summary["remaining"] == -50
            assert db.get_trip(created_trip_id, USER_CREATE)["request"]["end_date"] == create_payload["end_date"]
            _ok("дорога и жильё учитываются один раз; общий план, дни и превышение согласованы")

            recall_args = recall_mock.call_args
            assert recall_args.args[:2] == (USER_CREATE, USER_CREATE)
            assert "Лион, Франция" in recall_args.args[2]
            assert "2 взрослых, дети 5 лет" in recall_args.args[2]
            assert "Темп: спокойный" in recall_args.args[2]
            assert "Цель: Познакомиться с городом и отдохнуть" in (
                recall_args.args[2]
            )
            assert "основные достопримечательности" in recall_args.args[2]
            assert "музеи и выставки" in recall_args.args[2]
            assert "интерактивный музей" in recall_args.args[2]
            assert recall_args.kwargs == {"top_k": 8, "min_score": 0.35,
                                          "destination": create_payload["destination"]}
            format_memory_mock.assert_called_once_with(matches)
            generated_request = generate_mock.call_args.args[0]
            assert generated_request.model_dump() == {
                **create_payload,
                "destination_kind": "city",
                "budget": 300.0,
            }
            assert generate_mock.call_args.kwargs == {
                "memory_context": "КОНТЕКСТ ПАМЯТИ",
                "ground_candidates": False,
            }
            checklist_mock.assert_called_once_with(
                2,
                [5],
                {"source": "test", "days": []},
            )
            save_memory_mock.assert_called_once()

            stored_created = db.get_trip(created_trip_id, USER_CREATE)
            assert stored_created is not None
            assert stored_created["request"]["weather_text"] == (
                "Свежая тестовая погода."
            )
            assert stored_created["request"]["weather"] == {
                "source": "test",
                "days": [],
            }
            stored_checklist = db.get_checklist(created_trip_id)
            assert len(stored_checklist) == 1
            assert stored_checklist[0]["text"] == "Проверить паспорта"
            assert not bool(stored_checklist[0]["done"])
            stored_memory = db.list_memory_items(USER_CREATE, 100)
            assert len(stored_memory) == 1
            assert stored_memory[0]["source_trip_id"] == created_trip_id
            created_profile = db.get_user(USER_CREATE)
            assert created_profile is not None
            assert created_profile["name"] == "Веб-пользователь"
            assert created_profile["adults"] == 2
            assert created_profile["children_ages"] == [5]
            assert created_profile["currency"] == "EUR"
            assert created_profile["pace"] == "спокойный"
            assert created_profile["wishes"] == (
                "интерактивный музей и дневной отдых"
            )
            assert db.get_ai_request_usage(USER_CREATE, create_usage_date) == 1
            with db.connect() as conn:
                ai_row = conn.execute(
                    """
                    SELECT kind, source, status, trip_id
                    FROM ai_requests
                    WHERE tg_id = ?
                    ORDER BY claimed_at DESC
                    LIMIT 1
                    """,
                    (USER_CREATE,),
                ).fetchone()
            assert dict(ai_row) == {
                "kind": "plan",
                "source": "web",
                "status": "succeeded",
                "trip_id": None,
            }
            _ok("создание поездки выполняет memory → AI → БД → checklist → memory")

        with TestClient(app) as other_after_create:
            _login(other_after_create, USER_TWO, web_auth)
            assert other_after_create.get(
                f"/api/trips/{created_trip_id}"
            ).status_code == 404
            other_trip_ids = {
                item["id"]
                for item in other_after_create.get(
                    "/api/trips?limit=100"
                ).json()["trips"]
            }
            assert created_trip_id not in other_trip_ids
        assert db.list_memory_items(USER_TWO, 100) == []
        _ok("новая поездка и её память изолированы от другого пользователя")

        with TestClient(app) as free_create_client:
            _, free_create_csrf = _login(
                free_create_client,
                USER_FREE_CREATE,
                web_auth,
            )
            free_create_payload = _new_trip_payload(
                "Воркута, Россия"
            )
            free_generated = _generation_result(
                free_create_payload,
                Itinerary,
            )
            for free_day in free_generated.itinerary.days:
                for free_item in free_day.items:
                    if free_item.kind == "attraction":
                        free_item.cost = 0

            with (
                patch.object(
                    web_service.vector_memory,
                    "recall_family_memories",
                    return_value=[],
                ),
                patch.object(
                    web_service.vector_memory,
                    "format_memory_context",
                    return_value="",
                ),
                patch.object(
                    web_service,
                    "generate_itinerary",
                    return_value=free_generated,
                ) as free_generate_mock,
                patch.object(web_service, "resolve_city_for_planning", return_value=None),
                patch.object(
                    web_service,
                    "build_checklist",
                    return_value=[
                        ("Документы", "Проверить паспорта")
                    ],
                ),
                patch.object(
                    web_service.vector_memory,
                    "save_trip_memory",
                    return_value=SimpleNamespace(
                        success=True,
                        requested=0,
                        vector_saved=0,
                        journal_saved=0,
                        error_type="",
                    ),
                ),
            ):
                free_created = free_create_client.post(
                    "/api/trips",
                    json=free_create_payload,
                    headers=_csrf_headers(
                        free_create_csrf
                    ),
                )

            assert free_created.status_code == 201, (
                free_created.text
            )
            free_created_body = free_created.json()
            free_trip_id = int(
                free_created_body["id"]
            )
            assert free_created_body["destination"] == (
                "Воркута, Россия"
            )
            assert free_created_body["budget"][
                "program_total"
            ] == 50.0
            assert (
                "нет подтверждённой стоимости билетов"
                in free_created_body["budget"]["scope_note"]
            )
            assert free_created_body["itinerary"]["tips"] == [
                "Купить билеты заранее."
            ]
            assert db.get_trip(
                free_trip_id,
                USER_FREE_CREATE,
            ) is not None
            assert len(db.get_checklist(free_trip_id)) == 1
            assert db.get_ai_request_usage(
                USER_FREE_CREATE,
                create_usage_date,
            ) == 1
            free_generate_mock.assert_called_once()
            with db.connect() as conn:
                free_ai_row = conn.execute(
                    """
                    SELECT status
                    FROM ai_requests
                    WHERE tg_id = ?
                    ORDER BY claimed_at DESC
                    LIMIT 1
                    """,
                    (USER_FREE_CREATE,),
                ).fetchone()
            assert free_ai_row["status"] == "succeeded"
        _ok("бесплатный маршрут сохраняется с предупреждением в веб-бюджете")

        with TestClient(app) as exact_create_client:
            _, exact_create_csrf = _login(
                exact_create_client,
                USER_EXACT_CREATE,
                web_auth,
            )
            exact_payload = _new_trip_payload(
                "Воркута, Россия"
            )
            exact_start = datetime.fromisoformat(
                exact_payload["start_date"]
            ).date()
            exact_payload.update(
                {
                    "end_date": (
                        exact_start
                        + timedelta(days=3)
                    ).isoformat(),
                    "children_ages": [],
                    "budget": 45_000,
                    "currency": "RUB",
                    "pace": "насыщенный",
                    "start_time": "10:00",
                    "end_time": "19:00",
                    "trip_goal": "Познакомиться с городом",
                    "interest_categories": [
                        "main_sights",
                        "museums",
                        "theatre",
                        "cinema",
                        "water_trips",
                        "parks",
                        "zoos",
                        "amusement_parks",
                        "shopping_malls",
                    ],
                    "location_preference": "anywhere",
                    "stay_area": "",
                    "wishes": "",
                }
            )
            exact_candidate = _vorkuta_candidate(
                exact_payload
            )
            from test_saturated_generator import _city_retry_payload

            exact_retry_candidate = _city_retry_payload(
                Itinerary.model_validate(exact_candidate)
            )

            with (
                patch.object(
                    web_service.vector_memory,
                    "recall_family_memories",
                    return_value=[],
                ),
                patch.object(
                    web_service.vector_memory,
                    "format_memory_context",
                    return_value="",
                ),
                patch.object(
                    planner,
                    "get_weather",
                    return_value=None,
                ),
                patch.object(web_service, "resolve_city_for_planning", return_value=None),
                patch.object(
                    planner,
                    "weather_summary",
                    return_value="",
                ),
                patch.object(
                    planner,
                    "find_city",
                    return_value=None,
                ),
                patch.object(
                    planner,
                    "generate_json",
                    side_effect=[
                        SimpleNamespace(
                            text=json.dumps(payload, ensure_ascii=False),
                            input_tokens=10,
                            output_tokens=20,
                        )
                        for payload in (exact_candidate, exact_retry_candidate)
                    ],
                ) as exact_generate_mock,
                patch.object(
                    planner,
                    "verify_itinerary",
                    return_value=[],
                ) as exact_map_mock,
                patch.object(
                    planner,
                    "check_city_radius",
                    return_value=[],
                ),
                patch.object(
                    planner,
                    "check_verified_place_duplicates",
                    return_value=[],
                ),
                patch.object(
                    web_service,
                    "build_checklist",
                    return_value=[
                        ("Документы", "Проверить паспорта")
                    ],
                ),
                patch.object(
                    web_service.vector_memory,
                    "save_trip_memory",
                    return_value=SimpleNamespace(
                        success=True,
                        requested=0,
                        vector_saved=0,
                        journal_saved=0,
                        error_type="",
                    ),
                ),
            ):
                exact_created = exact_create_client.post(
                    "/api/trips",
                    json=exact_payload,
                    headers=_csrf_headers(
                        exact_create_csrf
                    ),
                )

            assert exact_created.status_code == 201, (
                exact_created.text
            )
            exact_body = exact_created.json()
            exact_days = exact_body["itinerary"]["days"]
            assert exact_body["generation"]["attempts"] == 2
            assert len(exact_days) == 4
            assert all(
                len(day_plan["items"]) == 4
                for day_plan in exact_days
            )
            assert all(
                sum(
                    item["kind"] == kind
                    for item in day_plan["items"]
                ) == expected
                for day_plan in exact_days
                for kind, expected in (
                    ("attraction", 2),
                    ("meal", 1),
                    ("rest", 1),
                )
            )
            assert [
                day_plan["rainy_alternative"]
                for day_plan in exact_days[:3]
            ] == [
                "",
                "Посещение библиотеки",
                "",
            ]
            exact_landmark = next(
                item
                for item in exact_days[3]["items"]
                if item["title"]
                == "Стела «Воркута — город трудовой доблести»"
            )
            assert exact_landmark["duration_min"] == 60
            assert all(
                (
                    planner._clock_minutes(item["time"])
                    + item["duration_min"]
                    <= 19 * 60
                )
                for day_plan in exact_days
                for item in day_plan["items"]
            )
            assert exact_body["itinerary"]["tips"] == [
                "Тёплая одежда обязательна.",
                "Проверяйте расписание заранее.",
            ]
            assert (
                "нет подтверждённой стоимости билетов"
                in exact_body["budget"]["scope_note"]
            )
            assert exact_generate_mock.call_count == 2
            assert exact_map_mock.call_count == 2
            assert len(
                db.list_trips(
                    USER_EXACT_CREATE,
                    100,
                )
            ) == 1
            assert db.get_ai_request_usage(
                USER_EXACT_CREATE,
                create_usage_date,
            ) == 1
            with db.connect() as conn:
                exact_ai_row = conn.execute(
                    """
                    SELECT status
                    FROM ai_requests
                    WHERE tg_id = ?
                    ORDER BY claimed_at DESC
                    LIMIT 1
                    """,
                    (USER_EXACT_CREATE,),
                ).fetchone()
            assert exact_ai_row["status"] == "succeeded"
        _ok(
            "короткий насыщенный маршрут проходит одну "
            "мягкую попытку и сохраняет безопасный fallback"
        )

        with TestClient(app) as reject_client:
            _, reject_csrf = _login(reject_client, USER_REJECT, web_auth)
            reject_payload = _new_trip_payload("Вена, Австрия")
            reject_payload["children_ages"] = []
            reject_profile_before = copy.deepcopy(db.get_user(USER_REJECT))
            with (
                patch.object(
                    web_service.vector_memory,
                    "recall_family_memories",
                    return_value=[],
                ),
                patch.object(
                    web_service.vector_memory,
                    "format_memory_context",
                    return_value="",
                ),
                patch.object(
                    web_service,
                    "generate_itinerary",
                    return_value=SimpleNamespace(
                        problems=["День 1: повторяется место."],
                    ),
                ),
                patch.object(web_service, "resolve_city_for_planning", return_value=None),
                patch.object(
                    web_service,
                    "build_checklist",
                    side_effect=AssertionError(
                        "Нельзя создавать чек-лист для непроверенного маршрута"
                    ),
                ) as rejected_checklist,
                patch.object(
                    web_service.vector_memory,
                    "save_trip_memory",
                    side_effect=AssertionError(
                        "Нельзя сохранять память непроверенного маршрута"
                    ),
                ) as rejected_memory_save,
            ):
                rejected_create = reject_client.post(
                    "/api/trips",
                    json=reject_payload,
                    headers=_csrf_headers(reject_csrf),
                )

            assert rejected_create.status_code == 422, rejected_create.text
            assert "Новая поездка не сохранена" in rejected_create.text
            assert "подтверждённых картой" in rejected_create.text
            assert "День 1: повторяется место" not in rejected_create.text
            rejected_checklist.assert_not_called()
            rejected_memory_save.assert_not_called()
            assert db.get_user(USER_REJECT) == reject_profile_before
            assert db.list_trips(USER_REJECT, 100) == []
            assert db.list_memory_items(USER_REJECT, 100) == []
            assert db.get_ai_request_usage(USER_REJECT, create_usage_date) == 1
        _ok("неполностью проверенный маршрут не попадает в профиль и базу")

        with TestClient(app) as fail_client:
            _, fail_csrf = _login(fail_client, USER_FAIL, web_auth)
            fail_payload = _new_trip_payload("Брюссель, Бельгия")
            fail_profile_before = copy.deepcopy(db.get_user(USER_FAIL))
            with (
                patch.object(
                    web_service.vector_memory,
                    "recall_family_memories",
                    return_value=[],
                ),
                patch.object(
                    web_service.vector_memory,
                    "format_memory_context",
                    return_value="",
                ),
                patch.object(
                    web_service,
                    "generate_itinerary",
                    side_effect=RuntimeError("offline AI"),
                ),
                patch.object(web_service, "resolve_city_for_planning", return_value=None),
                patch.object(
                    web_service,
                    "build_checklist",
                    side_effect=AssertionError(
                        "Чек-лист не должен создаваться после ошибки AI"
                    ),
                ) as fail_checklist,
                patch.object(
                    web_service.vector_memory,
                    "save_trip_memory",
                    side_effect=AssertionError(
                        "Память не должна сохраняться после ошибки AI"
                    ),
                ) as fail_memory_save,
                patch.object(web_app.logger, "exception") as expected_error_log,
            ):
                failed = fail_client.post(
                    "/api/trips",
                    json=fail_payload,
                    headers=_csrf_headers(fail_csrf),
                )

            assert failed.status_code == 502
            assert failed.json() == {
                "detail": (
                    "Не получилось составить маршрут. "
                    "Данные не потеряны — попробуйте немного позже."
                )
            }
            fail_checklist.assert_not_called()
            fail_memory_save.assert_not_called()
            expected_error_log.assert_called_once()
            assert db.get_user(USER_FAIL) == fail_profile_before
            assert db.list_trips(USER_FAIL, 100) == []
            assert db.list_memory_items(USER_FAIL, 100) == []
            assert db.get_ai_request_usage(USER_FAIL, create_usage_date) == 1
            with db.connect() as conn:
                fail_status = conn.execute(
                    """
                    SELECT status
                    FROM ai_requests
                    WHERE tg_id = ?
                    ORDER BY claimed_at DESC
                    LIMIT 1
                    """,
                    (USER_FAIL,),
                ).fetchone()["status"]
            assert fail_status == "failed"
            _ok("ошибка генерации не сохраняет профиль, поездку, чек-лист или память")

        with TestClient(app) as guard_client:
            _, guard_csrf = _login(guard_client, USER_GUARD, web_auth)
            guard_payload = _new_trip_payload("Прага, Чехия")
            guard_profile_before = copy.deepcopy(db.get_user(USER_GUARD))
            active_plan = db.claim_ai_request(
                USER_GUARD,
                create_usage_date,
                "plan",
                "test",
                daily_limit=5,
            )
            assert active_plan.status == "ok"

            with (
                patch.object(
                    web_service.vector_memory,
                    "recall_family_memories",
                    side_effect=AssertionError(
                        "Память не должна вызываться при busy или limit"
                    ),
                ) as guarded_recall,
                patch.object(
                    web_service,
                    "generate_itinerary",
                    side_effect=AssertionError(
                        "AI не должен вызываться при busy или limit"
                    ),
                ) as guarded_generate,
                patch.object(web_service, "resolve_city_for_planning", return_value=None),
            ):
                busy_create = guard_client.post(
                    "/api/trips",
                    json=guard_payload,
                    headers=_csrf_headers(guard_csrf),
                )
                assert busy_create.status_code == 409
                assert busy_create.json() == {
                    "detail": (
                        "Другая операция с маршрутом уже выполняется. "
                        "Подождите немного и повторите запрос."
                    )
                }
                assert db.finish_ai_request(
                    active_plan.claim_id,
                    succeeded=False,
                )

                for _ in range(4):
                    quota_claim = db.claim_ai_request(
                        USER_GUARD,
                        create_usage_date,
                        "plan",
                        "test",
                        daily_limit=5,
                    )
                    assert quota_claim.status == "ok"
                    assert db.finish_ai_request(
                        quota_claim.claim_id,
                        succeeded=True,
                    )

                limited_create = guard_client.post(
                    "/api/trips",
                    json=guard_payload,
                    headers=_csrf_headers(guard_csrf),
                )
                assert limited_create.status_code == 429
                assert limited_create.json() == {
                    "detail": (
                        "На сегодня лимит 5 AI-операций исчерпан. "
                        "Попробуйте завтра."
                    )
                }

            guarded_recall.assert_not_called()
            guarded_generate.assert_not_called()
            assert db.get_user(USER_GUARD) == guard_profile_before
            assert db.list_trips(USER_GUARD, 100) == []
            assert db.list_memory_items(USER_GUARD, 100) == []
            assert db.get_ai_request_usage(USER_GUARD, create_usage_date) == 5
            _ok("создание маршрута соблюдает busy и общий суточный лимит")

        valid_session, _, _ = web_auth.create_session(USER_ONE)
        encoded, signature = valid_session.split(".", 1)
        first = "A" if signature[0] != "A" else "B"
        tampered_session = f"{encoded}.{first}{signature[1:]}"
        expired_session, _, _ = web_auth.create_session(
            USER_ONE,
            now=(
                datetime.now(timezone.utc)
                - timedelta(
                    hours=int(os.getenv("WEB_SESSION_HOURS", "24")) + 1
                )
            ),
        )

        assert web_auth.verify_session(tampered_session) is None
        assert web_auth.verify_session(expired_session) is None

        with TestClient(app) as tampered_client:
            tampered_client.cookies.set(
                web_auth.SESSION_COOKIE,
                tampered_session,
                path="/",
            )
            assert tampered_client.get("/api/me").status_code == 401

        with TestClient(app) as expired_client:
            expired_client.cookies.set(
                web_auth.SESSION_COOKIE,
                expired_session,
                path="/",
            )
            assert expired_client.get("/api/me").status_code == 401
        _ok("подделанная и истёкшая сессии отклоняются")

        with db.connect() as conn:
            assert conn.execute("PRAGMA foreign_key_check").fetchone() is None

    finally:
        db.DB_PATH = original_db_path
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _remove_test_files()

    assert _fingerprint(real_db_path) == real_db_before
    assert not TEST_DB.exists()

    print("\nИтог: веб-кабинет, авторизация и изоляция данных полностью работают.")
    print("Временная база test_web.db удалена.")
    print("Настоящий planner.db не изменился.")


if __name__ == "__main__":
    main()
