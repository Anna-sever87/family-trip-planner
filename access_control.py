"""Temporary invitation access and atomic paid-attempt budgets.

No provider calls, raw invitation tokens, or guest Telegram IDs are persisted.
Access ledgers deliberately survive profile deletion; reservations are not refunds.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import db


class AccessDeniedError(RuntimeError):
    def __init__(self, message: str = "Доступ закрыт. Используйте действующее приглашение."):
        super().__init__(message)


class AccessQuotaError(AccessDeniedError):
    pass


def _mode() -> str:
    return os.getenv("ACCESS_MODE", "open").strip().lower()


def enabled() -> bool:
    """Invalid configuration is guarded too, never an implicit open mode."""
    return _mode() != "open"


def _valid_mode() -> str:
    mode = _mode()
    if mode not in {"open", "invite"} or (mode == "invite" and not _owner_ids()):
        raise AccessDeniedError("Доступ временно закрыт. Обратитесь к владельцу приложения.")
    return mode


def _user_id(tg_id: int) -> int:
    if isinstance(tg_id, bool) or not isinstance(tg_id, int) or not 0 < tg_id < 2**63:
        raise AccessDeniedError()
    return tg_id


def _owner_ids() -> set[int]:
    try:
        parts = re.split(r"[\s,;]+", os.getenv("ACCESS_OWNER_IDS", "").strip())
        owners = {_user_id(int(part)) for part in parts if part}
    except (ValueError, AccessDeniedError):
        return set()
    return owners


def is_owner(tg_id: int) -> bool:
    try:
        _user_id(tg_id)
    except AccessDeniedError:
        return False
    return tg_id in _owner_ids()


def _now(now: datetime | None = None) -> datetime:
    value = now if now is not None else datetime.now(timezone.utc)
    if not isinstance(value, datetime):
        raise ValueError("now должен быть datetime.")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _stamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


def _future(value: str, now: datetime) -> bool:
    try:
        parsed = datetime.fromisoformat(value)
        return parsed.tzinfo is not None and parsed.astimezone(timezone.utc) > now
    except (ValueError, TypeError, OverflowError):
        return False


def _create_invites_table(conn: sqlite3.Connection, name: str) -> None:
    if name not in {"access_invites", "access_invites_route5"}:
        raise ValueError("Invalid invitation table")
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {name} (
            invite_id TEXT PRIMARY KEY,
            token_hash TEXT NOT NULL UNIQUE,
            created_by INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            max_users INTEGER NOT NULL CHECK(max_users BETWEEN 1 AND 3),
            route_limit INTEGER NOT NULL CHECK(route_limit BETWEEN 1 AND 5),
            revoked_at TEXT
        )
    """)


def _migrate_invite_route_limit(conn: sqlite3.Connection) -> None:
    schema = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'access_invites'").fetchone()
    if not re.search(r"CHECK\s*\(\s*route_limit\s+BETWEEN\s+1\s+AND\s+3\s*\)", schema["sql"], re.I):
        return
    if conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'access_invites_route5'").fetchone():
        raise RuntimeError("Обнаружена незавершённая миграция приглашений.")
    _create_invites_table(conn, "access_invites_route5")
    conn.execute("""
        INSERT INTO access_invites_route5
        (invite_id, token_hash, created_by, created_at, expires_at, max_users, route_limit, revoked_at)
        SELECT invite_id, token_hash, created_by, created_at, expires_at, max_users, route_limit, revoked_at
        FROM access_invites
    """)
    # Do not rename the old parent: child foreign keys must keep its final name.
    conn.execute("DROP TABLE access_invites")
    conn.execute("ALTER TABLE access_invites_route5 RENAME TO access_invites")


def init_schema() -> None:
    """Migrate invitation constraints transactionally, preserving grants and usage."""
    with db.connect() as conn:
        # SQLite requires this before BEGIN when replacing a referenced table.
        # This dedicated connection closes after commit; normal connections use FK=ON.
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS access_settings (
                id INTEGER PRIMARY KEY CHECK(id = 1),
                subject_key TEXT NOT NULL
            )
        """)
        conn.execute("INSERT OR IGNORE INTO access_settings(id, subject_key) VALUES (1, ?)",
                     (secrets.token_hex(32),))
        _create_invites_table(conn, "access_invites")
        _migrate_invite_route_limit(conn)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS access_grants (
                grant_id TEXT PRIMARY KEY,
                invite_id TEXT NOT NULL REFERENCES access_invites(invite_id),
                subject_hash TEXT NOT NULL,
                joined_at TEXT NOT NULL,
                route_used INTEGER NOT NULL DEFAULT 0 CHECK(route_used >= 0),
                cover_used INTEGER NOT NULL DEFAULT 0 CHECK(cover_used >= 0),
                memory_used INTEGER NOT NULL DEFAULT 0 CHECK(memory_used >= 0),
                UNIQUE(invite_id, subject_hash)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS access_current_grants (
                subject_hash TEXT PRIMARY KEY,
                grant_id TEXT NOT NULL UNIQUE REFERENCES access_grants(grant_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS access_route_attempts (
                claim_id TEXT PRIMARY KEY,
                subject_hash TEXT NOT NULL,
                grant_id TEXT REFERENCES access_grants(grant_id),
                usage_day TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_access_route_day ON access_route_attempts(usage_day)")
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("Миграция приглашений остановлена: нарушена связь данных.")


def _subject(conn: sqlite3.Connection, tg_id: int) -> str:
    _user_id(tg_id)
    row = conn.execute("SELECT subject_key FROM access_settings WHERE id = 1").fetchone()
    if row is None:
        raise AccessDeniedError()
    return hmac.new(bytes.fromhex(row["subject_key"]), str(tg_id).encode("ascii"), hashlib.sha256).hexdigest()


def _grant(conn: sqlite3.Connection, subject: str, now: datetime) -> sqlite3.Row | None:
    row = conn.execute("""
        SELECT g.*, i.expires_at, i.route_limit, i.revoked_at
        FROM access_current_grants AS c
        JOIN access_grants AS g ON g.grant_id = c.grant_id AND g.subject_hash = c.subject_hash
        JOIN access_invites AS i ON i.invite_id = g.invite_id
        WHERE c.subject_hash = ?
    """, (subject,)).fetchone()
    if row is None or row["revoked_at"] is not None or not _future(row["expires_at"], now):
        return None
    return row


def _public_status(conn: sqlite3.Connection | None, tg_id: int, now: datetime) -> dict[str, Any]:
    owner = is_owner(tg_id)
    mode = _mode()
    result = {"owner": owner, "allowed": False, "expires_at": None, "route_remaining": 0,
              "route_limit": None, "grant_id": None}
    if mode not in {"open", "invite"} or (mode == "invite" and not _owner_ids()):
        return result
    if mode == "open" or owner:
        return {**result, "allowed": True, "route_remaining": None,
                "grant_id": "open" if mode == "open" else "owner"}
    assert conn is not None
    row = _grant(conn, _subject(conn, tg_id), now)
    if row is not None:
        result.update(allowed=True, expires_at=row["expires_at"], grant_id=row["grant_id"],
                      route_limit=int(row["route_limit"]),
                      route_remaining=max(0, int(row["route_limit"]) - int(row["route_used"])))
    return result


def status(tg_id: int, now: datetime | None = None) -> dict[str, Any]:
    _user_id(tg_id)
    current = _now(now)
    if _mode() != "invite" or is_owner(tg_id) or not _owner_ids():
        return _public_status(None, tg_id, current)
    with db.connect() as conn:
        return _public_status(conn, tg_id, current)


def require_access(tg_id: int, now: datetime | None = None) -> dict[str, Any]:
    _user_id(tg_id)
    _valid_mode()
    result = status(tg_id, now)
    if not result["allowed"]:
        raise AccessDeniedError()
    return result


def _require_owner(owner_id: int) -> None:
    _valid_mode()
    if not is_owner(owner_id):
        raise AccessDeniedError("Это действие доступно только владельцу приложения.")


def _bounded(value: int, upper: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
        raise ValueError(f"{field}: допустимо целое число от 1 до {upper}.")
    return value


def create_invite(owner_id: int, days: int = 7, max_users: int = 1, route_limit: int = 5,
                  *, now: datetime | None = None) -> dict[str, Any]:
    _require_owner(owner_id)
    _bounded(days, 30, "days")
    _bounded(max_users, 3, "max_users")
    _bounded(route_limit, 5, "route_limit")
    current = _now(now)
    expires = _stamp(current + timedelta(days=days))
    token = secrets.token_urlsafe(32)
    invite_id = uuid4().hex
    with db.connect() as conn:
        conn.execute("""
            INSERT INTO access_invites(invite_id, token_hash, created_by, created_at,
                                       expires_at, max_users, route_limit)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (invite_id, hashlib.sha256(token.encode("ascii")).hexdigest(), owner_id,
              _stamp(current), expires, max_users, route_limit))
    return {"invite_id": invite_id, "token": token, "expires_at": expires,
            "max_users": max_users, "route_limit": route_limit}


def redeem_invite(tg_id: int, token: str, *, now: datetime | None = None) -> dict[str, Any]:
    _user_id(tg_id)
    _valid_mode()
    if not isinstance(token, str) or re.fullmatch(r"[A-Za-z0-9_-]{32,128}", token) is None:
        raise AccessDeniedError("Приглашение недействительно или срок его действия закончился.")
    current = _now(now)
    digest = hashlib.sha256(token.encode("ascii")).hexdigest()
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        invitation = conn.execute("SELECT * FROM access_invites WHERE token_hash = ?", (digest,)).fetchone()
        if invitation is None or invitation["revoked_at"] is not None or not _future(invitation["expires_at"], current):
            raise AccessDeniedError("Приглашение недействительно или срок его действия закончился.")
        if is_owner(tg_id):
            return _public_status(conn, tg_id, current)
        subject = _subject(conn, tg_id)
        # An owner-issued different link can extend an exhausted grant. Reopening
        # the same link or changing links before exhaustion never resets usage.
        active = _grant(conn, subject, current)
        if active is not None and (active["route_used"] < active["route_limit"]
                                   or active["invite_id"] == invitation["invite_id"]):
            return _public_status(conn, tg_id, current)
        previous = conn.execute("SELECT grant_id FROM access_grants WHERE invite_id = ? AND subject_hash = ?",
                                (invitation["invite_id"], subject)).fetchone()
        if previous is None:
            joined = conn.execute("SELECT COUNT(*) FROM access_grants WHERE invite_id = ?",
                                  (invitation["invite_id"],)).fetchone()[0]
            if joined >= invitation["max_users"]:
                raise AccessDeniedError("Все места по этому приглашению уже заняты.")
            grant_id = uuid4().hex
            conn.execute("INSERT INTO access_grants(grant_id, invite_id, subject_hash, joined_at) VALUES (?, ?, ?, ?)",
                         (grant_id, invitation["invite_id"], subject, _stamp(current)))
        else:
            grant_id = previous["grant_id"]
        conn.execute("INSERT INTO access_current_grants(subject_hash, grant_id) VALUES (?, ?) "
                     "ON CONFLICT(subject_hash) DO UPDATE SET grant_id = excluded.grant_id", (subject, grant_id))
        return _public_status(conn, tg_id, current)


def list_invites(owner_id: int) -> list[dict[str, Any]]:
    _require_owner(owner_id)
    with db.connect() as conn:
        rows = conn.execute("""
            SELECT i.invite_id, i.created_at, i.expires_at, i.max_users, i.route_limit,
                   i.revoked_at, COUNT(g.grant_id) AS joined_users
            FROM access_invites AS i LEFT JOIN access_grants AS g ON g.invite_id = i.invite_id
            GROUP BY i.invite_id ORDER BY i.created_at DESC, i.invite_id
        """).fetchall()
    return [dict(row) for row in rows]


def revoke_invite(owner_id: int, invite_id: str, *, now: datetime | None = None) -> bool:
    _require_owner(owner_id)
    if not isinstance(invite_id, str) or re.fullmatch(r"[a-f0-9]{32}", invite_id) is None:
        return False
    with db.connect() as conn:
        return conn.execute("UPDATE access_invites SET revoked_at = COALESCE(revoked_at, ?) WHERE invite_id = ?",
                            (_stamp(_now(now)), invite_id)).rowcount == 1


def _global_limit() -> int:
    try:
        value = int(os.getenv("ACCESS_GLOBAL_ROUTE_DAILY_LIMIT", "12"))
    except (ValueError, TypeError):
        return 0
    return value if 1 <= value <= 1000 else 0


def _assert_expected_grant(tg_id: int, grant_id: str) -> None:
    # Local import avoids a module cycle. Called inside the same write
    # transaction that checks the live grant and reserves the attempt.
    import access_guard
    expected = access_guard.expected_grant(tg_id)
    if expected is not None and expected != grant_id:
        raise AccessDeniedError("Доступ изменился. Начните новую операцию.")


def guest_route_remaining(conn: sqlite3.Connection, tg_id: int, now: datetime) -> int | None:
    """Guests use their lifetime grant, owners/open mode keep the daily policy."""
    if _valid_mode() == "open" or is_owner(tg_id):
        return None
    grant = _grant(conn, _subject(conn, tg_id), _now(now))
    if grant is None:
        raise AccessDeniedError()
    _assert_expected_grant(tg_id, grant["grant_id"])
    return max(0, int(grant["route_limit"]) - int(grant["route_used"]))


def reserve_route(conn: sqlite3.Connection, tg_id: int, claim_id: str,
                  now: datetime | None = None) -> None:
    """Caller owns BEGIN IMMEDIATE and atomically commits this with ai_requests."""
    _user_id(tg_id)
    if _valid_mode() == "open":
        _assert_expected_grant(tg_id, "open")
        return
    if not conn.in_transaction:
        raise RuntimeError("Резервирование требует активной транзакции.")
    if not isinstance(claim_id, str) or not claim_id or len(claim_id) > 128:
        raise ValueError("Некорректный идентификатор операции.")
    current = _now(now)
    subject = _subject(conn, tg_id)
    grant = None if is_owner(tg_id) else _grant(conn, subject, current)
    if not is_owner(tg_id) and grant is None:
        raise AccessDeniedError()
    _assert_expected_grant(tg_id, grant["grant_id"] if grant is not None else "owner")
    previous = conn.execute("SELECT subject_hash FROM access_route_attempts WHERE claim_id = ?", (claim_id,)).fetchone()
    if previous is not None:
        if previous["subject_hash"] != subject:
            raise AccessDeniedError()
        return
    if grant is not None and grant["route_used"] >= grant["route_limit"]:
        raise AccessQuotaError("Лимит попыток создания и правки маршрутов по приглашению исчерпан.")
    day = current.date().isoformat()
    used = conn.execute("SELECT COUNT(*) FROM access_route_attempts WHERE usage_day = ?", (day,)).fetchone()[0]
    if used >= _global_limit():
        raise AccessQuotaError("Общий дневной лимит приложения исчерпан. Попробуйте завтра.")
    conn.execute("INSERT INTO access_route_attempts(claim_id, subject_hash, grant_id, usage_day, created_at) "
                 "VALUES (?, ?, ?, ?, ?)", (claim_id, subject, grant["grant_id"] if grant is not None else None,
                                          day, _stamp(current)))
    if grant is not None:
        conn.execute("UPDATE access_grants SET route_used = route_used + 1 WHERE grant_id = ?", (grant["grant_id"],))


def reserve_aux(tg_id: int, kind: str, now: datetime | None = None) -> None:
    """Reserve one cover or memory attempt, independently: three each per grant."""
    _user_id(tg_id)
    if kind not in {"cover", "memory"}:
        raise ValueError("Неизвестный тип дополнительной операции.")
    mode = _valid_mode()
    if mode == "open" or is_owner(tg_id):
        _assert_expected_grant(tg_id, "open" if mode == "open" else "owner")
        return
    current = _now(now)
    column = {"cover": "cover_used", "memory": "memory_used"}[kind]
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        grant = _grant(conn, _subject(conn, tg_id), current)
        if grant is None:
            raise AccessDeniedError()
        _assert_expected_grant(tg_id, grant["grant_id"])
        if grant[column] >= 3:
            label = "обложек" if kind == "cover" else "операций памяти"
            raise AccessQuotaError(f"Лимит {label} по приглашению исчерпан.")
        conn.execute(f"UPDATE access_grants SET {column} = {column} + 1 WHERE grant_id = ?", (grant["grant_id"],))
