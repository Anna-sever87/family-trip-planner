"""FastAPI-приложение веб-кабинета семейных поездок."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Annotated, Literal

import anyio
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator

import db
import access_control
import generation_jobs
import trip_covers
import trip_deletion
import web_service
from generation_feedback import generation_failure_reason
from yandex_ai import YandexAIError
from schemas import TripRequest
from web_auth import (
    CSRF_COOKIE,
    SESSION_COOKIE,
    WebAuthConfigError,
    consume_login_grant,
    cookie_is_secure,
    create_session,
    get_current_user_id,
    require_csrf,
    require_csrf_for_logout,
)


load_dotenv(override=False)

logger = logging.getLogger("web")
BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"


# Map the existing public classifier's FIXED output to log-only codes. Probes
# are constants, never request/model text; classification precedence remains
# owned by generation_feedback, including source > outage > rain > other.
_REJECTION_REASON_CODES = tuple(
    (generation_failure_reason([probe]), code)
    for code, probe in (
        ("SOURCE_SEARCH_LIMIT", "SOURCE_PLAN_UNAVAILABLE: SOURCE_SEARCH_LIMIT"),
        ("SOURCE_ROUTES_UNCONFIRMED", "SOURCE_PLAN_UNAVAILABLE: SOURCE_ROUTES_UNCONFIRMED"),
        ("SOURCE_VISIT_DATA_INCOMPLETE", "SOURCE_PLAN_UNAVAILABLE: SOURCE_VISIT_DATA_INCOMPLETE"),
        ("SOURCE_CANDIDATE_CAPACITY", "SOURCE_PLAN_UNAVAILABLE: SOURCE_CANDIDATE_CAPACITY"),
        ("SOURCE_CONSTRAINTS_INCOMPATIBLE", "SOURCE_PLAN_UNAVAILABLE: SOURCE_CONSTRAINTS_INCOMPATIBLE"),
        ("SOURCE_UNKNOWN", "SOURCE_PLAN_UNAVAILABLE"),
        ("MAP_UNAVAILABLE", "Сервис карты временно недоступен"),
        ("RAIN_PROGRAM", "День 1 (2030-01-01) дождливый, но под открытым небом: Объект. "
         "Не менее половины достопримечательностей должны быть в помещении с indoor = true. "
         "Если достопримечательность одна, она обязана быть в помещении."),
        ("MAP_CONFIRMATION", "place_query"),
        ("TIME_BUDGET", "duration_min"),
        ("CONSTRAINTS", ""),
    )
)


def _log_validation_rejection(error: Exception, *, stage: str) -> None:
    """Log only bounded metadata, never exception/diagnostic/request content.

    Some service failures have no attempts field. Leave it unknown instead of
    inferring calls from the number of problems or exposing unrelated fields.
    """
    diagnostics = getattr(error, "diagnostic_problems", ())
    diagnostics = diagnostics if isinstance(diagnostics, (list, tuple)) else ()
    reason = generation_failure_reason(problem for problem in diagnostics if isinstance(problem, str))
    code = next((code for prefix, code in _REJECTION_REASON_CODES
                 if reason == prefix or (code.startswith("SOURCE_") and reason.startswith(prefix + " "))),
                "CONSTRAINTS")
    attempts = getattr(error, "attempts", None)
    attempts = attempts if type(attempts) is int and 0 <= attempts <= 100 else None
    stage = stage if stage in {"create_validation", "edit_validation"} else "validation"
    logger.warning(
        "trip_validation_rejected reason=%s problem_count=%d stage=%s attempts=%s",
        code, len(diagnostics), stage, attempts if attempts is not None else "unknown",
        extra={"reason_code": code, "problem_count": len(diagnostics),
               "validation_stage": stage, "attempts": attempts},
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Журнал маршрута без подробных сетевых логов, секретов и тел запросов.
    # memory INFO contains only a validated embedding purpose and HTTP status.
    for name in ("planner", "places", "web.service", "yandex_ai", "memory"):
        route_logger = logging.getLogger(name)
        route_logger.setLevel(logging.INFO)
        if not any(getattr(handler, "_trip_diagnostics", False) for handler in route_logger.handlers):
            handler = logging.StreamHandler()
            handler._trip_diagnostics = True
            handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            route_logger.addHandler(handler)
        route_logger.propagate = False
    db.init_db()
    generation_jobs.manager.start()
    trip_covers.manager.start()
    await anyio.to_thread.run_sync(trip_deletion.cleanup_pending_files)
    try:
        yield
    finally:
        await anyio.to_thread.run_sync(trip_covers.manager.close)
        await anyio.to_thread.run_sync(generation_jobs.manager.close)


app = FastAPI(
    title="Семейный планировщик путешествий",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=WEB_DIR / "static"),
    name="static",
)


@app.exception_handler(access_control.AccessDeniedError)
async def access_denied_handler(request: Request, error: access_control.AccessDeniedError):
    quota = isinstance(error, access_control.AccessQuotaError)
    return JSONResponse(status_code=429 if quota else 403,
                        content={"detail": str(error), "code": "access_quota" if quota else "access_denied"})


class ChecklistUpdate(BaseModel):
    done: bool


class CoverVisibilityUpdate(BaseModel):
    model_config = {"extra": "forbid"}
    hidden: bool = Field(strict=True)


class LoginRequest(BaseModel):
    token: str = Field(min_length=32, max_length=200)


class DayEditRequest(BaseModel):
    instruction: str = Field(min_length=3, max_length=600)


class TripCreateRequest(BaseModel):
    destination: str = Field(min_length=2, max_length=100)
    destination_kind: Literal["city", "region"] = "city"
    start_date: str = Field(min_length=10, max_length=10)
    end_date: str = Field(min_length=10, max_length=10)
    adults: int = Field(ge=1, le=10, strict=True)
    children_ages: list[
        Annotated[int, Field(ge=0, le=17, strict=True)]
    ] = Field(default_factory=list, max_length=6)
    budget: float = Field(gt=0, le=1_000_000_000, strict=True)
    travel_cost: float = Field(default=0, ge=0, le=1_000_000_000, strict=True, allow_inf_nan=False)
    lodging_cost: float = Field(default=0, ge=0, le=1_000_000_000, strict=True, allow_inf_nan=False)
    currency: Literal["EUR", "USD", "RUB"] = "EUR"
    pace: Literal["спокойный", "средний", "насыщенный"] = "средний"
    start_time: str = Field(default="09:00", min_length=5, max_length=5)
    end_time: str = Field(default="20:00", min_length=5, max_length=5)
    trip_goal: str = Field(
        default="Познакомиться с городом",
        min_length=2,
        max_length=160,
    )
    interest_categories: list[
        Literal[
            "main_sights",
            "museums",
            "theatre",
            "cinema",
            "water_trips",
            "parks",
            "zoos",
            "amusement_parks",
            "shopping_malls",
        ]
    ] = Field(
        default_factory=lambda: ["main_sights"],
        min_length=1,
        max_length=9,
    )
    location_preference: Literal["anywhere", "nearby"] = "anywhere"
    stay_area: str = Field(default="", max_length=120)
    wishes: str = Field(default="", max_length=500)

    @field_validator("budget", "travel_cost", "lodging_cost", mode="before")
    @classmethod
    def reject_boolean_budget(cls, value):
        if isinstance(value, bool):
            raise ValueError("Суммы бюджета должны быть числами.")
        return value

    @field_validator("interest_categories")
    @classmethod
    def reject_duplicate_interests(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("Не повторяйте выбранные варианты отдыха.")
        return value

    @model_validator(mode="after")
    def validate_schedule_and_location(self):
        def minutes(value: str) -> int:
            try:
                hour_text, minute_text = value.split(":", 1)
                hour = int(hour_text)
                minute = int(minute_text)
            except (TypeError, ValueError) as error:
                raise ValueError("Время должно быть в формате ЧЧ:ММ.") from error
            if (
                len(value) != 5
                or len(hour_text) != 2
                or len(minute_text) != 2
                or not 0 <= hour <= 23
                or not 0 <= minute <= 59
            ):
                raise ValueError("Время должно быть в формате ЧЧ:ММ.")
            return hour * 60 + minute

        start = minutes(self.start_time)
        end = minutes(self.end_time)
        if not 6 * 60 <= start <= 23 * 60 + 30:
            raise ValueError("Начало дня должно быть между 06:00 и 23:30.")
        if not 6 * 60 <= end <= 23 * 60 + 30:
            raise ValueError("Окончание дня должно быть между 06:00 и 23:30.")
        if start >= end:
            raise ValueError("Окончание дня должно быть позже начала.")
        if end - start < 4 * 60:
            raise ValueError("Для маршрута нужно выделить не меньше 4 часов в день.")
        if self.location_preference == "nearby" and len(self.stay_area.strip()) < 2:
            raise ValueError(
                "Укажите район проживания или ближайшую станцию."
            )
        return self


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    # OpenStreetMap requires browser map requests to include a valid Referer.
    # The origin-only value keeps paths and query parameters private.
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https://tile.openstreetmap.org; "
        "connect-src 'self'; font-src 'self'; object-src 'none'; "
        "base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
    )
    if request.url.path == "/" or request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, private"
        response.headers["Pragma"] = "no-cache"
    if os.getenv("WEB_BASE_URL", "").strip().lower().startswith("https://"):
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


@app.exception_handler(web_service.TripNotFoundError)
async def trip_not_found_handler(_: Request, __: web_service.TripNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": "Поездка не найдена."},
    )


@app.exception_handler(web_service.TripDataError)
async def trip_data_handler(_: Request, error: web_service.TripDataError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={"detail": str(error)},
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(WEB_DIR / "index.html")


def _set_session_cookies(response: Response, tg_id: int, *, expected_grant: str | None = None) -> Response:
    try:
        session_token, csrf_token, expires = create_session(tg_id, expected_grant=expected_grant)
    except WebAuthConfigError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    max_age = max(
        1,
        int((expires - datetime.now(timezone.utc)).total_seconds()),
    )
    secure = cookie_is_secure()
    response.set_cookie(
        SESSION_COOKIE,
        session_token,
        max_age=max_age,
        expires=expires,
        httponly=True,
        secure=secure,
        samesite="strict",
        path="/",
    )
    response.set_cookie(
        CSRF_COOKIE,
        csrf_token,
        max_age=max_age,
        expires=expires,
        httponly=False,
        secure=secure,
        samesite="strict",
        path="/",
    )
    return response


@app.post("/api/login")
def login(payload: LoginRequest):
    consumed = consume_login_grant(payload.token)
    if consumed is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "Ссылка уже использована или устарела. "
                "Запросите новую командой /web в Telegram."
            ),
        )
    tg_id, access_grant = consumed
    return _set_session_cookies(JSONResponse({"ok": True}), tg_id, expected_grant=access_grant)


@app.get("/api/me")
def me(tg_id: int = Depends(get_current_user_id)) -> dict:
    profile = db.get_user(tg_id) or {}
    return {
        "name": profile.get("name") or "Путешественник",
        "adults": int(profile.get("adults", 2)),
        "children_ages": profile.get("children_ages", []),
        "currency": profile.get("currency", "EUR"),
        "pace": profile.get("pace", "средний"),
        "wishes": profile.get("wishes", ""),
        "capabilities": {**web_service.planning_capabilities(), "cover_images": True},
        "access": access_control.status(tg_id),
        "generation_jobs": {
            "enabled": bool(profile.get("account_key")),
            "owner": generation_jobs.owner_token(profile["account_key"]) if profile.get("account_key") else "",
        },
    }


@app.post("/api/logout")
def logout(_: int = Depends(require_csrf_for_logout)):
    response = JSONResponse({"ok": True})
    response.delete_cookie(SESSION_COOKIE, path="/")
    response.delete_cookie(CSRF_COOKIE, path="/")
    return response


@app.get("/api/trips")
def trips(
    limit: int = Query(default=50, ge=1, le=100),
    tg_id: int = Depends(get_current_user_id),
) -> dict:
    return {"trips": web_service.list_user_trips(tg_id, limit)}


@app.post("/api/generation-jobs", status_code=status.HTTP_202_ACCEPTED)
def create_generation_job(payload: TripCreateRequest, http_request: Request,
                          tg_id: int = Depends(require_csrf)) -> dict:
    try:
        return generation_jobs.manager.submit(
            tg_id, http_request.state.account_key,
            http_request.headers.get("Idempotency-Key", ""),
            TripRequest.model_validate(payload.model_dump()),
            **({"access_grant": http_request.state.access_grant} if access_control.enabled() else {}),
        )
    except generation_jobs.JobNotFoundError as error:
        raise HTTPException(status_code=404, detail="Заявка не найдена.") from error
    except (generation_jobs.JobBusyError, generation_jobs.JobConflictError) as error:
        # Messages are fixed by the receipt manager, never provider output.
        raise HTTPException(status_code=409, detail=str(error)) from error
    except web_service.TripCreateValidationError as error:
        message = generation_jobs.MESSAGES.get(error.code, generation_jobs.MESSAGES["validation"])
        raise HTTPException(status_code=422, detail=message) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail="Проверьте анкету и ключ заявки. Генерация не запущена.") from error


def _generation_job_status(http_request: Request, tg_id: int, **identifier) -> dict:
    try:
        return generation_jobs.get_job(tg_id, http_request.state.account_key, **identifier)
    except (generation_jobs.JobNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Заявка не найдена.") from error


@app.get("/api/generation-jobs")
def generation_job_by_key(http_request: Request, key: str,
                          tg_id: int = Depends(get_current_user_id)) -> dict:
    return _generation_job_status(http_request, tg_id, key=key)


@app.get("/api/generation-jobs/{job_id}")
def generation_job_status(job_id: str, http_request: Request,
                          tg_id: int = Depends(get_current_user_id)) -> dict:
    return _generation_job_status(http_request, tg_id, job_id=job_id)


@app.post("/api/trips", status_code=status.HTTP_201_CREATED)
async def create_trip(
    payload: TripCreateRequest,
    http_request: Request,
    tg_id: int = Depends(require_csrf),
) -> dict:
    request = TripRequest.model_validate(payload.model_dump())
    try:
        return await anyio.to_thread.run_sync(
            partial(web_service.create_trip, tg_id, request,
                    expected_account_key=http_request.state.account_key,
                    **({"_access_grant": http_request.state.access_grant} if access_control.enabled() else {})),
        )
    except web_service.EditBusyError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Другая операция с маршрутом уже выполняется. "
                "Подождите немного и повторите запрос."
            ),
        ) from error
    except web_service.AIDailyLimitError as error:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"На сегодня лимит {error.limit} AI-операций исчерпан. "
                "Попробуйте завтра."
            ),
        ) from error
    except web_service.TripCreateValidationError as error:
        _log_validation_rejection(error, stage="create_validation")
        detail = str(error)
        if error.code in web_service.CITY_PREFLIGHT_CODES:
            detail = {
                "message": generation_jobs.MESSAGES[error.code], "code": error.code,
                "clarification": {"kind": "city", "options": list(error.destination_options),
                                  "attempt_used": False},
            }
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=detail,
        ) from error
    except web_service.TripSavedResponseError as error:
        logger.exception("Не удалось показать сохранённую поездку %s", error.trip_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": str(error), "saved": True, "trip_id": error.trip_id},
        ) from error
    except YandexAIError as error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(error)) from error
    except access_control.AccessDeniedError:
        raise
    except Exception as error:
        logger.exception(
            "Ошибка веб-создания поездки для пользователя %s",
            tg_id,
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Не получилось составить маршрут. "
                "Данные не потеряны — попробуйте немного позже."
            ),
        ) from error


@app.get("/api/trips/{trip_id}")
def trip_detail(trip_id: int, tg_id: int = Depends(get_current_user_id)) -> dict:
    return web_service.get_trip_detail(tg_id, trip_id)


@app.delete("/api/trips/{trip_id}")
def delete_trip(trip_id: int, http_request: Request, tg_id: int = Depends(require_csrf)) -> dict:
    try:
        return web_service.delete_trip(tg_id, trip_id, expected_account_key=http_request.state.account_key)
    except trip_deletion.TripDeleteBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except trip_deletion.TripDeleteStorageError as error:
        # This exception is possible only before the route deletion commits.
        raise HTTPException(status_code=503, detail="Не удалось безопасно подготовить удаление файлов. Поездка сохранена.") from error


@app.get("/api/trips/{trip_id}/cover")
def trip_cover_status(trip_id: int, http_request: Request,
                      tg_id: int = Depends(get_current_user_id)) -> dict:
    try:
        return trip_covers.manager.get_status(tg_id, http_request.state.account_key, trip_id)
    except trip_covers.CoverNotFoundError as error:
        raise HTTPException(status_code=404, detail="Поездка не найдена.") from error


@app.post("/api/trips/{trip_id}/cover")
def create_trip_cover(trip_id: int, http_request: Request,
                      tg_id: int = Depends(require_csrf)):
    try:
        result = trip_covers.manager.submit(tg_id, http_request.state.account_key, trip_id,
            **({"access_grant": http_request.state.access_grant} if access_control.enabled() else {}))
        return JSONResponse(result, status_code=202 if result["status"] in {"queued", "running"} else 200)
    except trip_covers.CoverNotFoundError as error:
        raise HTTPException(status_code=404, detail="Поездка не найдена.") from error
    except (trip_covers.CoverBusyError, trip_covers.CoverLimitError) as error:
        raise HTTPException(status_code=429, detail=str(error)) from error
    except trip_covers.CoverDestinationError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/trips/{trip_id}/cover/image")
def trip_cover_image(trip_id: int, http_request: Request,
                     tg_id: int = Depends(get_current_user_id)):
    try:
        image = trip_covers.manager.get_image(tg_id, http_request.state.account_key, trip_id)
    except trip_covers.CoverNotFoundError as error:
        raise HTTPException(status_code=404, detail="Иллюстрация недоступна.") from error
    return Response(image.data, media_type=image.mime_type,
                    headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})


@app.patch("/api/trips/{trip_id}/cover")
def update_trip_cover_visibility(trip_id: int, payload: CoverVisibilityUpdate, http_request: Request,
                                 tg_id: int = Depends(require_csrf)) -> dict:
    try:
        return trip_covers.manager.set_hidden(tg_id, http_request.state.account_key, trip_id, payload.hidden)
    except trip_covers.CoverNotFoundError as error:
        raise HTTPException(status_code=404, detail="Поездка не найдена.") from error
    except trip_covers.CoverStateError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/api/trips/{trip_id}/reminders")
def trip_reminders(trip_id: int, tg_id: int = Depends(get_current_user_id)) -> dict:
    detail = web_service.get_trip_detail(tg_id, trip_id)
    return {"text": detail["reminders_text"]}


@app.get("/api/trips/{trip_id}/pdf")
def trip_pdf(trip_id: int, tg_id: int = Depends(get_current_user_id)):
    content = web_service.get_trip_pdf(tg_id, trip_id)
    return Response(
        content=content,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="trip_{trip_id}.pdf"',
            "Cache-Control": "no-store",
        },
    )


@app.patch("/api/checklist-items/{item_id}")
def checklist_update(
    item_id: int,
    payload: ChecklistUpdate,
    tg_id: int = Depends(require_csrf),
) -> dict:
    return {"item": web_service.set_checklist_done(tg_id, item_id, payload.done)}


@app.post("/api/trips/{trip_id}/days/{day_number}/edit")
async def edit_day(
    trip_id: int,
    day_number: int,
    payload: DayEditRequest,
    http_request: Request = None,
    tg_id: int = Depends(require_csrf),
) -> dict:
    if access_control.enabled() and http_request is None:
        raise access_control.AccessDeniedError("Для операции нужен действующий вход в кабинет.")
    try:
        return await anyio.to_thread.run_sync(
            partial(web_service.edit_trip_day, tg_id, trip_id, day_number, payload.instruction,
                    **({"_access_grant": http_request.state.access_grant} if access_control.enabled() else {})),
        )
    except web_service.EditRejectedError as error:
        _log_validation_rejection(error, stage="edit_validation")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "message": "Изменения не сохранены. Старый маршрут сохранён без изменений.",
                "problems": error.problems,
            },
        ) from error
    except web_service.EditConflictError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(error),
        ) from error
    except web_service.TripSavedResponseError as error:
        logger.exception("Не удалось показать сохранённую правку поездки %s", error.trip_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"message": str(error), "saved": True, "trip_id": error.trip_id},
        ) from error
    except web_service.EditBusyError as error:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Другая операция с маршрутом уже выполняется. "
                "Подождите немного и повторите запрос."
            ),
        ) from error
    except web_service.AIDailyLimitError as error:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"На сегодня лимит {error.limit} AI-операций исчерпан. "
                "Попробуйте завтра."
            ),
        ) from error
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
    except (web_service.TripNotFoundError, web_service.TripDataError):
        raise
    except access_control.AccessDeniedError:
        raise
    except Exception as error:
        logger.exception("Ошибка веб-редактирования поездки %s", trip_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Не получилось изменить маршрут. Старый вариант сохранён. "
                "Попробуйте немного позже."
            ),
        ) from error


if __name__ == "__main__":
    uvicorn.run(
        "web_app:app",
        host="127.0.0.1",
        port=8000,
        workers=1,
    )
