"""Безопасный вход в веб-кабинет по одноразовой ссылке из Telegram."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode, urlsplit

from dotenv import load_dotenv
from fastapi import HTTPException, Request, status

import db
import access_control


load_dotenv(override=False)

SESSION_COOKIE = "family_trip_session"
CSRF_COOKIE = "family_trip_csrf"
DEFAULT_LOGIN_MINUTES = 15
DEFAULT_SESSION_HOURS = 24

UNSAFE_SECRET_VALUES = {
    "change-me",
    "change_me",
    "случайная_секретная_строка_не_короче_32_символов",
}


class WebAuthConfigError(RuntimeError):
    """Веб-вход не настроен в .env."""


@dataclass(frozen=True)
class SessionData:
    tg_id: int
    csrf_hash: str
    expires_at: datetime
    account_key: str
    grant_id: str = ""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _positive_env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise WebAuthConfigError(
            f"{name} должен быть целым положительным числом."
        ) from error

    if value < 1:
        raise WebAuthConfigError(
            f"{name} должен быть целым положительным числом."
        )

    return value


def _secret() -> bytes:
    value = os.getenv("WEB_SESSION_SECRET", "").strip()
    if (
        len(value) < 32
        or value.casefold() in UNSAFE_SECRET_VALUES
        or value.casefold().startswith("example-")
    ):
        raise WebAuthConfigError(
            "В .env нужен WEB_SESSION_SECRET длиной не меньше 32 символов."
        )
    return value.encode("utf-8")


def _urlsafe_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _urlsafe_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_login_token(tg_id: int, now: Optional[datetime] = None) -> str:
    """Создаёт одноразовый токен, сохраняя в БД только его хеш."""

    if int(tg_id) <= 0:
        raise ValueError("tg_id должен быть положительным.")

    access = access_control.require_access(int(tg_id), now=now)
    current = (now or _utc_now()).astimezone(timezone.utc)
    db.purge_expired_web_login_tokens(current)
    expires = current + timedelta(
        minutes=_positive_env_int(
            "WEB_LOGIN_TOKEN_MINUTES",
            DEFAULT_LOGIN_MINUTES,
        )
    )
    token = secrets.token_urlsafe(32)
    if access_control.enabled():
        token += "." + access["grant_id"]
    db.create_web_login_token(
        int(tg_id),
        _token_hash(token),
        expires,
    )
    return token


def build_login_url(tg_id: int) -> str:
    """Возвращает готовую одноразовую ссылку на кабинет."""

    base_url = os.getenv("WEB_BASE_URL", "").strip().rstrip("/")
    parsed = urlsplit(base_url)
    loopback_hosts = {"127.0.0.1", "localhost", "::1"}
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or (
            parsed.scheme != "https"
            and parsed.hostname.casefold() not in loopback_hosts
        )
    ):
        raise WebAuthConfigError(
            "WEB_BASE_URL должен быть публичным HTTPS-адресом. "
            "Обычный HTTP разрешён только для localhost."
        )

    token = issue_login_token(tg_id)
    return f"{base_url}/#{urlencode({'login': token})}"


def consume_login_grant(token: str) -> Optional[tuple[int, str]]:
    """Consume once and retain the validated grant through session creation."""

    value = (token or "").strip()
    if len(value) < 32 or len(value) > 200:
        return None
    tg_id = db.consume_web_login_token(_token_hash(value))
    if tg_id is None:
        return None
    access = access_control.require_access(tg_id)
    if access_control.enabled() and not access["owner"]:
        if "." not in value or value.rsplit(".", 1)[1] != access["grant_id"]:
            raise access_control.AccessDeniedError("Приглашение изменилось. Запросите новую ссылку /web.")
    return tg_id, access["grant_id"]


def consume_login_token(token: str) -> Optional[int]:
    """Backward-compatible exchange for callers that only need Telegram ID."""
    result = consume_login_grant(token)
    return result[0] if result is not None else None


def create_session(tg_id: int, now: Optional[datetime] = None, *,
                   expected_grant: Optional[str] = None) -> tuple[str, str, datetime]:
    """Создаёт подписанную HttpOnly-сессию и отдельный CSRF-токен."""

    access = access_control.require_access(int(tg_id), now=now)
    if expected_grant is not None and expected_grant != access["grant_id"]:
        raise access_control.AccessDeniedError("Доступ изменился. Запросите новую ссылку /web.")
    account_key = db.get_account_key(int(tg_id))
    if not account_key:
        raise WebAuthConfigError("Профиль удалён или его данные сейчас удаляются.")
    current = (now or _utc_now()).astimezone(timezone.utc)
    expires = current + timedelta(
        hours=_positive_env_int(
            "WEB_SESSION_HOURS",
            DEFAULT_SESSION_HOURS,
        )
    )
    csrf_token = secrets.token_urlsafe(24)
    payload = {
        "tg_id": int(tg_id),
        "account": account_key,
        "exp": int(expires.timestamp()),
        "csrf": _token_hash(csrf_token),
        "nonce": secrets.token_urlsafe(12),
        "grant": access["grant_id"],
    }
    encoded = _urlsafe_encode(
        json.dumps(
            payload,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    signature = hmac.new(
        _secret(),
        encoded.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{encoded}.{_urlsafe_encode(signature)}", csrf_token, expires


def verify_session(token: str, now: Optional[datetime] = None) -> Optional[SessionData]:
    """Проверяет подпись и срок сессии без доверия данным браузера."""

    try:
        encoded, supplied_signature = token.split(".", 1)
        expected_signature = hmac.new(
            _secret(),
            encoded.encode("ascii"),
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(
            _urlsafe_decode(supplied_signature),
            expected_signature,
        ):
            return None

        payload = json.loads(_urlsafe_decode(encoded).decode("utf-8"))
        tg_id = int(payload["tg_id"])
        expires = datetime.fromtimestamp(int(payload["exp"]), timezone.utc)
        csrf_hash = str(payload["csrf"])
        account_key = str(payload["account"])
    except (
        binascii.Error,
        KeyError,
        TypeError,
        ValueError,
        UnicodeError,
        json.JSONDecodeError,
    ):
        return None

    current = (now or _utc_now()).astimezone(timezone.utc)
    if tg_id <= 0 or expires <= current or len(csrf_hash) != 64:
        return None
    current_account_key = db.get_account_key(tg_id)
    if not current_account_key or not hmac.compare_digest(account_key, current_account_key):
        return None

    return SessionData(tg_id=tg_id, csrf_hash=csrf_hash, expires_at=expires, account_key=account_key,
                       grant_id=str(payload.get("grant", "")))


def _require_session_access(session: SessionData) -> dict:
    access = access_control.require_access(session.tg_id)
    if access_control.enabled() and not access["owner"] and session.grant_id != access["grant_id"]:
        raise access_control.AccessDeniedError("Доступ изменился. Откройте новую ссылку /web в боте.")
    return access


def get_current_user_id(request: Request) -> int:
    """FastAPI dependency: Telegram ID только из проверенной сессии."""

    token = request.cookies.get(SESSION_COOKIE, "")
    try:
        session = verify_session(token)
    except WebAuthConfigError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Войдите через одноразовую ссылку из Telegram-бота.",
        )
    request.state.access_grant = _require_session_access(session)["grant_id"]
    request.state.account_key = session.account_key
    return session.tg_id


def _require_csrf(request: Request, *, check_access: bool = True) -> int:
    """Проверяет сессию и защиту изменяющих запросов."""

    token = request.cookies.get(SESSION_COOKIE, "")
    try:
        session = verify_session(token)
    except WebAuthConfigError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error

    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Сессия истекла. Откройте новую ссылку в Telegram-боте.",
        )

    header_token = request.headers.get("X-CSRF-Token", "")
    cookie_token = request.cookies.get(CSRF_COOKIE, "")
    if (
        not header_token
        or not cookie_token
        or not hmac.compare_digest(header_token, cookie_token)
        or not hmac.compare_digest(_token_hash(header_token), session.csrf_hash)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Защитный токен страницы устарел. Обновите страницу.",
        )

    if check_access:
        request.state.access_grant = _require_session_access(session)["grant_id"]
    request.state.account_key = session.account_key
    return session.tg_id


def require_csrf(request: Request) -> int:
    return _require_csrf(request)


def require_csrf_for_logout(request: Request) -> int:
    """An expired invitation must not prevent clearing valid session cookies."""
    return _require_csrf(request, check_access=False)


def cookie_is_secure() -> bool:
    if os.getenv("WEB_BASE_URL", "").strip().lower().startswith("https://"):
        return True
    configured = os.getenv("WEB_COOKIE_SECURE", "").strip().lower()
    if configured:
        return configured in {"1", "true", "yes", "on"}
    return os.getenv("WEB_BASE_URL", "").strip().lower().startswith("https://")
