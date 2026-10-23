"""SQLite: профили семей, поездки, чек-листы, память и напоминания."""

from __future__ import annotations

import json
import hashlib
import hmac
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional, Union
from uuid import uuid4


DB_PATH = "planner.db"

REMINDER_STATUS_SENDING = "sending"
REMINDER_STATUS_SENT = "sent"
REMINDER_STATUS_FAILED = "failed"
DEFAULT_REMINDER_LEASE_MINUTES = 15
DEFAULT_REMINDER_MAX_ATTEMPTS = 3


@dataclass(frozen=True)
class AIRequestClaim:
    """Результат атомарного резервирования AI-запроса."""

    status: str
    claim_id: Optional[str]
    used: int
    limit: int


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Открывает безопасное соединение для одного действия."""

    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")

    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


def _utc_datetime(
    value: Optional[datetime] = None,
) -> datetime:
    """Возвращает дату со временем в UTC и часовым поясом."""

    result = value or datetime.now(
        timezone.utc
    )

    if result.tzinfo is None:
        result = result.replace(
            tzinfo=timezone.utc
        )

    return result.astimezone(
        timezone.utc
    )


def _timestamp(
    value: Optional[datetime] = None,
) -> str:
    return _utc_datetime(
        value
    ).isoformat(
        timespec="seconds"
    )


def _now() -> str:
    return _timestamp()


def _parse_timestamp(
    value: Any,
) -> Optional[datetime]:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(
            str(value)
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    return _utc_datetime(
        parsed
    )


def _iso_date(
    value: Union[
        str,
        date,
        datetime,
    ],
    field_name: str,
) -> str:
    if isinstance(
        value,
        datetime,
    ):
        result = value.date()
    elif isinstance(
        value,
        date,
    ):
        result = value
    else:
        try:
            result = date.fromisoformat(
                str(value)
            )
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                f"{field_name} должен быть датой "
                "в формате ГГГГ-ММ-ДД."
            ) from error

    return result.isoformat()


def _table_exists(
    conn: sqlite3.Connection,
    table_name: str,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()

    return row is not None


def _column_names(
    conn: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    return {
        str(row["name"])
        for row in conn.execute(
            f"PRAGMA table_info({table_name})"
        ).fetchall()
    }


def _create_checklist_table(
    conn: sqlite3.Connection,
) -> None:
    conn.execute(
        """
        CREATE TABLE checklist_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trip_id INTEGER NOT NULL,
            category TEXT,
            text TEXT,
            done INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (trip_id)
                REFERENCES trips(id)
                ON DELETE CASCADE
        )
        """
    )


def _checklist_has_cascade(
    conn: sqlite3.Connection,
) -> bool:
    if not _table_exists(
        conn,
        "checklist_items",
    ):
        return False

    foreign_keys = conn.execute(
        "PRAGMA foreign_key_list(checklist_items)"
    ).fetchall()

    return any(
        row["table"] == "trips"
        and row["from"] == "trip_id"
        and row["to"] == "id"
        and str(row["on_delete"]).upper()
        == "CASCADE"
        for row in foreign_keys
    )


def _migrate_checklist_table(
    conn: sqlite3.Connection,
) -> None:
    """Добавляет связь старой таблице без потери данных."""

    if not _table_exists(
        conn,
        "checklist_items",
    ):
        _create_checklist_table(conn)
        return

    if _checklist_has_cascade(conn):
        return

    orphan_count = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM checklist_items AS c
            LEFT JOIN trips AS t
                ON t.id = c.trip_id
            WHERE t.id IS NULL
            """
        ).fetchone()[0]
    )

    if orphan_count:
        raise RuntimeError(
            "В planner.db найдены пункты чек-листа "
            "без поездки. Автоматическая миграция "
            "остановлена, чтобы не потерять данные."
        )

    legacy_table = "checklist_items_before_fk"

    if _table_exists(
        conn,
        legacy_table,
    ):
        raise RuntimeError(
            "Обнаружена незавершённая миграция "
            "таблицы checklist_items."
        )

    conn.execute(
        """
        ALTER TABLE checklist_items
        RENAME TO checklist_items_before_fk
        """
    )

    _create_checklist_table(conn)

    conn.execute(
        """
        INSERT INTO checklist_items (
            id,
            trip_id,
            category,
            text,
            done
        )
        SELECT
            id,
            trip_id,
            category,
            text,
            COALESCE(done, 0)
        FROM checklist_items_before_fk
        ORDER BY id
        """
    )

    conn.execute(
        """
        DROP TABLE checklist_items_before_fk
        """
    )


def init_db() -> None:
    """Создаёт новые таблицы и безопасно обновляет старую БД."""

    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                tg_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL DEFAULT '',
                adults INTEGER NOT NULL DEFAULT 2,
                children_ages TEXT NOT NULL DEFAULT '[]',
                currency TEXT NOT NULL DEFAULT 'EUR',
                pace TEXT NOT NULL DEFAULT 'средний',
                wishes TEXT NOT NULL DEFAULT '',
                created_at TEXT
            )
            """
        )

        user_columns = _column_names(
            conn,
            "users",
        )

        if "pace" not in user_columns:
            conn.execute(
                """
                ALTER TABLE users
                ADD COLUMN pace TEXT
                NOT NULL DEFAULT 'средний'
                """
            )

        if "wishes" not in user_columns:
            conn.execute(
                """
                ALTER TABLE users
                ADD COLUMN wishes TEXT
                NOT NULL DEFAULT ''
                """
            )

        # UUID меняется при удалении и новом создании профиля.
        # Старые подписанные веб-сессии не получают доступ к новой семье.
        for column, definition in (
            ("account_key", "TEXT NOT NULL DEFAULT ''"),
            ("deletion_token", "TEXT NOT NULL DEFAULT ''"),
            ("deletion_started_at", "TEXT"),
        ):
            if column not in user_columns:
                conn.execute(f"ALTER TABLE users ADD COLUMN {column} {definition}")
        conn.execute(
            "UPDATE users SET account_key = lower(hex(randomblob(16))) "
            "WHERE account_key = '' OR account_key IS NULL"
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS users_account_key_after_insert
            AFTER INSERT ON users WHEN NEW.account_key = ''
            BEGIN
                UPDATE users SET account_key = lower(hex(randomblob(16)))
                WHERE tg_id = NEW.tg_id;
            END
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_account_key ON users(account_key)"
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trips (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tg_id INTEGER NOT NULL,
                destination TEXT,
                start_date TEXT,
                end_date TEXT,
                request_json TEXT,
                itinerary_json TEXT,
                revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT,
                FOREIGN KEY (tg_id)
                    REFERENCES users(tg_id)
            )
            """
        )

        trip_columns = _column_names(
            conn,
            "trips",
        )

        if "revision" not in trip_columns:
            conn.execute(
                """
                ALTER TABLE trips
                ADD COLUMN revision INTEGER
                NOT NULL DEFAULT 1
                """
            )

        # Если от старой версии осталась поездка без профиля,
        # создаём минимальный профиль и сохраняем поездку.
        conn.execute(
            """
            INSERT OR IGNORE INTO users (
                tg_id,
                created_at
            )
            SELECT DISTINCT
                trips.tg_id,
                ?
            FROM trips
            LEFT JOIN users
                ON users.tg_id = trips.tg_id
            WHERE users.tg_id IS NULL
            """,
            (_now(),),
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_trips_user
            ON trips(tg_id)
            """
        )

        _migrate_checklist_table(conn)

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_checklist_trip
            ON checklist_items(trip_id)
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tg_id INTEGER NOT NULL,
                memory_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                text TEXT NOT NULL,
                destination TEXT NOT NULL DEFAULT '',
                source_trip_id INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (tg_id, memory_id),
                FOREIGN KEY (tg_id)
                    REFERENCES users(tg_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (source_trip_id)
                    REFERENCES trips(id)
                    ON DELETE SET NULL
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_memory_user_updated
            ON memory_items(
                tg_id,
                updated_at DESC,
                id DESC
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_memory_source_trip
            ON memory_items(source_trip_id)
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reminder_deliveries (
                trip_id INTEGER NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                claimed_at TEXT NOT NULL,
                sent_at TEXT,
                last_error TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (trip_id, kind),
                CHECK (
                    status IN (
                        'sending',
                        'sent',
                        'failed'
                    )
                ),
                CHECK (attempts >= 0),
                FOREIGN KEY (trip_id)
                    REFERENCES trips(id)
                    ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_reminders_status_claimed
            ON reminder_deliveries(
                status,
                claimed_at
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS web_login_tokens (
                token_hash TEXT PRIMARY KEY,
                tg_id INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                used_at TEXT,
                FOREIGN KEY (tg_id)
                    REFERENCES users(tg_id)
                    ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_web_login_tokens_expiry
            ON web_login_tokens(expires_at)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_web_login_tokens_user
            ON web_login_tokens(tg_id)
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_requests (
                claim_id TEXT PRIMARY KEY,
                tg_id INTEGER NOT NULL,
                usage_date TEXT NOT NULL,
                kind TEXT NOT NULL,
                source TEXT NOT NULL,
                trip_id INTEGER,
                status TEXT NOT NULL,
                claimed_at TEXT NOT NULL,
                finished_at TEXT,
                CHECK (
                    status IN (
                        'running',
                        'succeeded',
                        'failed',
                        'expired'
                    )
                ),
                FOREIGN KEY (tg_id)
                    REFERENCES users(tg_id)
                    ON DELETE CASCADE,
                FOREIGN KEY (trip_id)
                    REFERENCES trips(id)
                    ON DELETE SET NULL
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_ai_requests_user_date
            ON ai_requests(
                tg_id,
                usage_date
            )
            """
        )

        # Даже разные источники (Telegram и веб) не должны
        # одновременно запускать два дорогих запроса одной семьи.
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
            idx_ai_requests_running_user
            ON ai_requests(tg_id)
            WHERE status = 'running'
            """
        )

        # Удаление аккаунта не должно обнулять платную дневную квоту.
        # Здесь нет Telegram ID, профиля, маршрутов или содержимого запросов.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_quota_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                hmac_key TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO ai_quota_settings (id, hmac_key) "
            "VALUES (1, lower(hex(randomblob(32))))"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_quota_retention (
                key_hash TEXT PRIMARY KEY,
                usage_date TEXT NOT NULL,
                used INTEGER NOT NULL CHECK (used >= 0)
            )
            """
        )

        # Additive, server-owned receipts: no prompts or itinerary payloads.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS generation_jobs (
                job_id TEXT PRIMARY KEY,
                tg_id INTEGER NOT NULL,
                account_key TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed','interrupted')),
                trip_id INTEGER,
                failure_code TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                finished_at TEXT,
                UNIQUE(account_key, idempotency_key),
                FOREIGN KEY(tg_id) REFERENCES users(tg_id) ON DELETE CASCADE,
                FOREIGN KEY(account_key) REFERENCES users(account_key) ON DELETE CASCADE,
                FOREIGN KEY(trip_id) REFERENCES trips(id) ON DELETE SET NULL
            )
        """)
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_generation_jobs_active_account "
                     "ON generation_jobs(account_key) WHERE status IN ('queued','running')")

        violation = conn.execute(
            "PRAGMA foreign_key_check"
        ).fetchone()

        if violation is not None:
            raise RuntimeError(
                "В planner.db нарушена связь "
                f"между таблицами: {tuple(violation)}"
            )


    # Invitation ledgers are independent of profile deletion and migrate only
    # after the main schema transaction is closed (no nested write lock).
    import access_control
    access_control.init_schema()


def save_user(
    tg_id: int,
    name: str = "",
    adults: int = 2,
    children_ages: Optional[List[int]] = None,
    currency: str = "EUR",
    pace: Optional[str] = None,
    wishes: Optional[str] = None,
) -> None:
    """Обновляет профиль; None сохраняет прежние темп и пожелания."""

    ages = json.dumps(
        (
            children_ages
            if children_ages is not None
            else []
        ),
        ensure_ascii=False,
    )

    insert_pace = (
        "средний"
        if pace is None
        else pace
    )
    insert_wishes = (
        ""
        if wishes is None
        else wishes
    )

    with connect() as conn:
        _ensure_user_exists(conn, tg_id)
        conn.execute(
            """
            INSERT INTO users (
                tg_id,
                name,
                adults,
                children_ages,
                currency,
                pace,
                wishes,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tg_id) DO UPDATE SET
                name = excluded.name,
                adults = excluded.adults,
                children_ages =
                    excluded.children_ages,
                currency = excluded.currency,
                pace = CASE
                    WHEN ? IS NULL
                        THEN users.pace
                    ELSE excluded.pace
                END,
                wishes = CASE
                    WHEN ? IS NULL
                        THEN users.wishes
                    ELSE excluded.wishes
                END
            """,
            (
                tg_id,
                name,
                adults,
                ages,
                currency,
                insert_pace,
                insert_wishes,
                _now(),
                pace,
                wishes,
            ),
        )


def get_user(
    tg_id: int,
) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM users
            WHERE tg_id = ?
            """,
            (tg_id,),
        ).fetchone()

    if row is None:
        return None

    data = dict(row)

    try:
        ages = json.loads(
            data.get("children_ages")
            or "[]"
        )
    except (
        TypeError,
        ValueError,
    ):
        ages = []

    data["children_ages"] = (
        ages
        if isinstance(ages, list)
        else []
    )

    return data


def _ensure_user_exists(
    conn: sqlite3.Connection,
    tg_id: int,
) -> None:
    """Создаёт минимальный профиль, если его ещё нет."""

    _assert_user_not_deleting(conn, tg_id)

    conn.execute(
        """
        INSERT OR IGNORE INTO users (
            tg_id,
            created_at
        )
        VALUES (?, ?)
        """,
        (
            tg_id,
            _now(),
        ),
    )


def _assert_user_not_deleting(conn: sqlite3.Connection, tg_id: int) -> None:
    row = conn.execute(
        "SELECT deletion_token FROM users WHERE tg_id = ?", (tg_id,)
    ).fetchone()
    if row is not None and row["deletion_token"]:
        raise RuntimeError("Удаление данных пользователя ещё выполняется.")


def get_account_key(tg_id: int) -> Optional[str]:
    """Возвращает идентификатор только действующего, не удаляемого профиля."""

    with connect() as conn:
        row = conn.execute(
            "SELECT account_key FROM users WHERE tg_id = ? AND deletion_token = ''",
            (tg_id,),
        ).fetchone()
    return str(row["account_key"]) if row is not None else None


def _ai_quota_key(conn: sqlite3.Connection, tg_id: int, day: str) -> str:
    secret = conn.execute("SELECT hmac_key FROM ai_quota_settings WHERE id = 1").fetchone()[0]
    return hmac.new(bytes.fromhex(secret), f"{tg_id}:{day}".encode("ascii"), hashlib.sha256).hexdigest()


def _ai_usage_count(conn: sqlite3.Connection, tg_id: int, day: str) -> int:
    active = int(conn.execute(
        "SELECT COUNT(*) FROM ai_requests WHERE tg_id = ? AND usage_date = ?", (tg_id, day)
    ).fetchone()[0])
    retained = conn.execute(
        "SELECT used FROM ai_quota_retention WHERE key_hash = ? AND usage_date = ?",
        (_ai_quota_key(conn, tg_id, day), day),
    ).fetchone()
    return active + (int(retained["used"]) if retained is not None else 0)


def _retain_ai_usage(conn: sqlite3.Connection, tg_id: int, day: str) -> None:
    """Вызывается в одной транзакции с удалением исходных AI-записей."""

    conn.execute("DELETE FROM ai_quota_retention WHERE usage_date < ?", (day,))
    used = _ai_usage_count(conn, tg_id, day)
    if used:
        conn.execute(
            "INSERT INTO ai_quota_retention (key_hash, usage_date, used) VALUES (?, ?, ?) "
            "ON CONFLICT(key_hash) DO UPDATE SET used = excluded.used",
            (_ai_quota_key(conn, tg_id, day), day, used),
        )


def claim_ai_request(
    tg_id: int,
    usage_date: Union[
        str,
        date,
        datetime,
    ],
    kind: str,
    source: str,
    *,
    trip_id: Optional[int] = None,
    daily_limit: int = 3,
    lease_minutes: int = 30,
    now: Optional[datetime] = None,
    expected_account_key: Optional[str] = None,
) -> AIRequestClaim:
    """Атомарно резервирует один платный AI-запрос.

    Все источники используют общий дневной счётчик. Незавершённая
    операция блокирует новый запрос только до окончания аренды.
    Неуспешные и протухшие запросы остаются в дневной квоте.
    """

    day = _iso_date(
        usage_date,
        "usage_date",
    )
    kind = str(kind).strip()
    source = str(source).strip()

    if not kind:
        raise ValueError(
            "kind не должен быть пустым."
        )

    if not source:
        raise ValueError(
            "source не должен быть пустым."
        )

    if (
        not isinstance(
            daily_limit,
            int,
        )
        or isinstance(
            daily_limit,
            bool,
        )
        or daily_limit <= 0
    ):
        raise ValueError(
            "daily_limit должен быть "
            "положительным целым числом."
        )

    if (
        not isinstance(
            lease_minutes,
            int,
        )
        or isinstance(
            lease_minutes,
            bool,
        )
        or lease_minutes <= 0
    ):
        raise ValueError(
            "lease_minutes должен быть "
            "положительным целым числом."
        )

    current = _utc_datetime(
        now
    )
    current_text = _timestamp(
        current
    )
    stale_before_text = _timestamp(
        current
        - timedelta(
            minutes=lease_minutes
        )
    )

    with connect() as conn:
        # BEGIN IMMEDIATE сериализует проверку лимита и INSERT.
        conn.execute(
            "BEGIN IMMEDIATE"
        )
        conn.execute("DELETE FROM ai_quota_retention WHERE usage_date < ?", (day,))

        if expected_account_key is not None:
            owner = conn.execute(
                "SELECT 1 FROM users WHERE tg_id = ? AND account_key = ? AND deletion_token = ''",
                (tg_id, expected_account_key),
            ).fetchone()
            if owner is None:
                # Уже авторизованный запрос мог ожидать рабочий поток.
                # Старый запрос не создаёт удалённый профиль заново.
                used = _ai_usage_count(conn, tg_id, day)
                return AIRequestClaim("busy", None, used, daily_limit)

        deleting = conn.execute(
            "SELECT 1 FROM users WHERE tg_id = ? AND deletion_token != ''",
            (tg_id,),
        ).fetchone()
        if deleting is not None:
            used = _ai_usage_count(conn, tg_id, day)
            return AIRequestClaim("busy", None, used, daily_limit)

        _ensure_user_exists(
            conn,
            tg_id,
        )

        if trip_id is not None:
            owned_trip = conn.execute(
                """
                SELECT 1
                FROM trips
                WHERE id = ? AND tg_id = ?
                """,
                (
                    trip_id,
                    tg_id,
                ),
            ).fetchone()

            if owned_trip is None:
                raise ValueError(
                    "AI-запрос нельзя связать "
                    "с чужой или несуществующей "
                    "поездкой."
                )

        # Освобождаем только действительно протухшую аренду.
        # Строка остаётся в журнале и продолжает учитывать квоту.
        conn.execute(
            """
            UPDATE ai_requests
            SET status = 'expired',
                finished_at = ?
            WHERE tg_id = ?
                AND status = 'running'
                AND claimed_at <= ?
            """,
            (
                current_text,
                tg_id,
                stale_before_text,
            ),
        )

        used = _ai_usage_count(conn, tg_id, day)

        running = conn.execute(
            """
            SELECT 1
            FROM ai_requests
            WHERE tg_id = ?
                AND status = 'running'
            LIMIT 1
            """,
            (tg_id,),
        ).fetchone()

        if running is not None:
            return AIRequestClaim(
                status="busy",
                claim_id=None,
                used=used,
                limit=daily_limit,
            )

        import access_control
        guest_remaining = access_control.guest_route_remaining(conn, tg_id, current)
        if guest_remaining is not None:
            # Invitation attempts may all be used today, including after the
            # owner revokes a spent grant and explicitly issues a new one.
            daily_limit = used + guest_remaining

        if guest_remaining is None and used >= daily_limit:
            return AIRequestClaim(
                status="limit",
                claim_id=None,
                used=used,
                limit=daily_limit,
            )

        claim_id = uuid4().hex

        access_control.reserve_route(conn, tg_id, claim_id, now=current)

        conn.execute(
            """
            INSERT INTO ai_requests (
                claim_id,
                tg_id,
                usage_date,
                kind,
                source,
                trip_id,
                status,
                claimed_at,
                finished_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'running', ?, NULL)
            """,
            (
                claim_id,
                tg_id,
                day,
                kind,
                source,
                trip_id,
                current_text,
            ),
        )

        return AIRequestClaim(
            status="ok",
            claim_id=claim_id,
            used=used + 1,
            limit=daily_limit,
        )


def finish_ai_request(
    claim_id: str,
    *,
    succeeded: bool,
    now: Optional[datetime] = None,
) -> bool:
    """Завершает ровно ту аренду, чей claim_id ещё выполняется."""

    claim_id = str(
        claim_id
    ).strip()

    if not claim_id:
        return False

    status = (
        "succeeded"
        if succeeded
        else "failed"
    )

    with connect() as conn:
        cursor = conn.execute(
            """
            UPDATE ai_requests
            SET status = ?,
                finished_at = ?
            WHERE claim_id = ?
                AND status = 'running'
            """,
            (
                status,
                _timestamp(
                    now
                ),
                claim_id,
            ),
        )

        return cursor.rowcount > 0


def get_ai_request_usage(
    tg_id: int,
    usage_date: Union[
        str,
        date,
        datetime,
    ],
) -> int:
    """Возвращает число запросов, потративших дневную квоту."""

    day = _iso_date(
        usage_date,
        "usage_date",
    )

    with connect() as conn:
        return _ai_usage_count(conn, tg_id, day)


def save_trip(
    tg_id: int,
    request: Dict[str, Any],
    itinerary: Dict[str, Any],
) -> int:
    """Сохраняет поездку и возвращает её номер."""

    with connect() as conn:
        _ensure_user_exists(
            conn,
            tg_id,
        )

        cursor = conn.execute(
            """
            INSERT INTO trips (
                tg_id,
                destination,
                start_date,
                end_date,
                request_json,
                itinerary_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tg_id,
                request.get(
                    "destination",
                    "",
                ),
                str(
                    request.get(
                        "start_date",
                        "",
                    )
                ),
                str(
                    request.get(
                        "end_date",
                        "",
                    )
                ),
                json.dumps(
                    request,
                    ensure_ascii=False,
                    default=str,
                ),
                json.dumps(
                    itinerary,
                    ensure_ascii=False,
                    default=str,
                ),
                _now(),
            ),
        )

        return int(cursor.lastrowid)


def save_trip_bundle(
    tg_id: int,
    request: Dict[str, Any],
    itinerary: Dict[str, Any],
    checklist_items: List[tuple],
    *,
    name: str = "",
    adults: int = 2,
    children_ages: Optional[List[int]] = None,
    currency: str = "EUR",
    pace: str = "средний",
    wishes: str = "",
    generation_job_id: Optional[str] = None,
    expected_account_key: Optional[str] = None,
) -> int:
    """Атомарно сохраняет профиль, поездку и весь чек-лист."""

    items = list(checklist_items)
    if not items:
        raise ValueError("Чек-лист новой поездки не должен быть пустым.")

    ages = json.dumps(
        children_ages if children_ages is not None else [],
        ensure_ascii=False,
    )
    created_at = _now()

    with connect() as conn:
        if generation_job_id is not None:
            conn.execute("BEGIN IMMEDIATE")
            receipt = conn.execute(
                "SELECT 1 FROM generation_jobs AS jobs JOIN users ON users.tg_id = jobs.tg_id "
                "AND users.account_key = jobs.account_key WHERE jobs.job_id = ? AND jobs.tg_id = ? "
                "AND jobs.account_key = ? AND jobs.status = 'running' AND users.deletion_token = ''",
                (generation_job_id, tg_id, expected_account_key),
            ).fetchone()
            if receipt is None:
                raise ValueError("Заявка создания уже завершена или аккаунт изменился.")
        _ensure_user_exists(conn, tg_id)
        conn.execute(
            """
            INSERT INTO users (
                tg_id,
                name,
                adults,
                children_ages,
                currency,
                pace,
                wishes,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tg_id) DO UPDATE SET
                name = excluded.name,
                adults = excluded.adults,
                children_ages = excluded.children_ages,
                currency = excluded.currency,
                pace = excluded.pace,
                wishes = excluded.wishes
            """,
            (
                tg_id,
                name,
                adults,
                ages,
                currency,
                pace,
                wishes,
                created_at,
            ),
        )

        cursor = conn.execute(
            """
            INSERT INTO trips (
                tg_id,
                destination,
                start_date,
                end_date,
                request_json,
                itinerary_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tg_id,
                request.get("destination", ""),
                str(request.get("start_date", "")),
                str(request.get("end_date", "")),
                json.dumps(
                    request,
                    ensure_ascii=False,
                    default=str,
                ),
                json.dumps(
                    itinerary,
                    ensure_ascii=False,
                    default=str,
                ),
                created_at,
            ),
        )
        trip_id = int(cursor.lastrowid)

        conn.executemany(
            """
            INSERT INTO checklist_items (
                trip_id,
                category,
                text,
                done
            )
            VALUES (?, ?, ?, 0)
            """,
            [
                (trip_id, category, text)
                for category, text in items
            ],
        )

        saved_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM checklist_items
            WHERE trip_id = ?
            """,
            (trip_id,),
        ).fetchone()[0]
        if int(saved_count) != len(items):
            raise RuntimeError("Чек-лист новой поездки сохранился не полностью.")

        if generation_job_id is not None:
            # Receipt and trip commit together: a lost response never loses the ID.
            conn.execute("UPDATE generation_jobs SET status='succeeded', trip_id=?, finished_at=?, "
                         "failure_code='' WHERE job_id=? AND account_key=? AND status='running'",
                         (trip_id, _now(), generation_job_id, expected_account_key))
        return trip_id


def update_itinerary(
    trip_id: int,
    itinerary: Dict[str, Any],
) -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE trips
            SET itinerary_json = ?
            WHERE id = ?
            """,
            (
                json.dumps(
                    itinerary,
                    ensure_ascii=False,
                    default=str,
                ),
                trip_id,
            ),
        )


def update_trip_plan(
    trip_id: int,
    tg_id: int,
    request: Dict[str, Any],
    itinerary: Dict[str, Any],
    expected_revision: Optional[int] = None,
) -> bool:
    """Атомарно обновляет маршрут только у его владельца.

    Вместе с маршрутом сохраняет свежую погоду в request_json.
    Направление, даты и остальные постоянные параметры поездки
    изменять через эту функцию нельзя.
    """

    if not isinstance(
        request,
        dict,
    ):
        raise TypeError(
            "request должен быть словарём."
        )

    if not isinstance(
        itinerary,
        dict,
    ):
        raise TypeError(
            "itinerary должен быть словарём."
        )

    request_json = json.dumps(
        request,
        ensure_ascii=False,
        default=str,
    )
    itinerary_json = json.dumps(
        itinerary,
        ensure_ascii=False,
        default=str,
    )

    fixed_fields = (
        "destination",
        "start_date",
        "end_date",
        "start_time",
        "end_time",
        "adults",
        "children_ages",
        "budget",
        "travel_cost",
        "lodging_cost",
        "currency",
        "pace",
        "trip_goal",
        "interest_categories",
        "location_preference",
        "stay_area",
    )

    with connect() as conn:
        conn.execute(
            "BEGIN IMMEDIATE"
        )

        row = conn.execute(
            """
            SELECT request_json, revision
            FROM trips
            WHERE id = ? AND tg_id = ?
            """,
            (
                trip_id,
                tg_id,
            ),
        ).fetchone()

        if row is None:
            return False

        if (
            expected_revision is not None
            and int(row["revision"])
            != expected_revision
        ):
            return False

        try:
            saved_request = json.loads(
                row["request_json"]
                or "{}"
            )
        except (
            TypeError,
            ValueError,
        ) as error:
            raise ValueError(
                "Сохранённая анкета поездки "
                "повреждена."
            ) from error

        if not isinstance(
            saved_request,
            dict,
        ):
            raise ValueError(
                "Сохранённая анкета поездки "
                "имеет неверный формат."
            )

        for field_name in fixed_fields:
            if field_name in {"travel_cost", "lodging_cost"}:
                # В старых анкетах этих полей не было: их значение равно
                # нулю, а редактирование дня не меняет постоянные расходы.
                if saved_request.get(field_name, 0) != request.get(field_name, 0):
                    raise ValueError(
                        "При редактировании нельзя изменять постоянные "
                        f"параметры поездки: {field_name}."
                    )
                continue
            if (
                field_name in saved_request
                and saved_request.get(
                    field_name
                )
                != request.get(
                    field_name
                )
            ):
                raise ValueError(
                    "При редактировании нельзя "
                    "изменять постоянные параметры "
                    f"поездки: {field_name}."
                )

        cursor = conn.execute(
            """
            UPDATE trips
            SET request_json = ?,
                itinerary_json = ?,
                revision = revision + 1
            WHERE id = ? AND tg_id = ?
            """,
            (
                request_json,
                itinerary_json,
                trip_id,
                tg_id,
            ),
        )

        return cursor.rowcount > 0


def _trip_from_row(
    row: sqlite3.Row,
) -> Dict[str, Any]:
    data = dict(row)

    data["request"] = json.loads(
        data.pop("request_json")
        or "{}"
    )

    data["itinerary"] = json.loads(
        data.pop("itinerary_json")
        or "{}"
    )

    return data


def get_trip(
    trip_id: int,
    tg_id: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """При переданном tg_id не возвращает чужую поездку."""

    query = (
        "SELECT * FROM trips WHERE id = ?"
    )
    params: List[Any] = [trip_id]

    if tg_id is not None:
        query += " AND tg_id = ?"
        params.append(tg_id)

    with connect() as conn:
        row = conn.execute(
            query,
            params,
        ).fetchone()

    return (
        _trip_from_row(row)
        if row
        else None
    )


def list_trips(
    tg_id: int,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM trips
            WHERE tg_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (
                tg_id,
                limit,
            ),
        ).fetchall()

    return [
        _trip_from_row(row)
        for row in rows
    ]


def list_reminder_candidate_trips(
    from_date: Union[
        str,
        date,
        datetime,
    ],
    through_date: Union[
        str,
        date,
        datetime,
    ],
) -> List[Dict[str, Any]]:
    """Возвращает поездки из узкого окна дат для диспетчера."""

    start = _iso_date(
        from_date,
        "from_date",
    )
    finish = _iso_date(
        through_date,
        "through_date",
    )

    if finish < start:
        raise ValueError(
            "through_date не может быть "
            "раньше from_date."
        )

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM trips
            WHERE end_date >= ?
                AND start_date <= ?
            ORDER BY start_date, id
            """,
            (
                start,
                finish,
            ),
        ).fetchall()

    return [
        _trip_from_row(row)
        for row in rows
    ]


def delete_trip(
    trip_id: int,
    tg_id: int,
) -> bool:
    """Удаляет только свою поездку и её чек-лист."""

    with connect() as conn:
        conn.execute(
            """
            DELETE FROM checklist_items
            WHERE trip_id IN (
                SELECT id
                FROM trips
                WHERE id = ? AND tg_id = ?
            )
            """,
            (
                trip_id,
                tg_id,
            ),
        )

        cursor = conn.execute(
            """
            DELETE FROM trips
            WHERE id = ? AND tg_id = ?
            """,
            (
                trip_id,
                tg_id,
            ),
        )

        return cursor.rowcount > 0


def save_checklist(
    trip_id: int,
    items: List[tuple],
) -> None:
    """Полностью заменяет чек-лист поездки."""

    with connect() as conn:
        conn.execute(
            """
            DELETE FROM checklist_items
            WHERE trip_id = ?
            """,
            (trip_id,),
        )

        conn.executemany(
            """
            INSERT INTO checklist_items (
                trip_id,
                category,
                text,
                done
            )
            VALUES (?, ?, ?, 0)
            """,
            [
                (
                    trip_id,
                    category,
                    text,
                )
                for category, text in items
            ],
        )


def get_checklist(
    trip_id: int,
) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM checklist_items
            WHERE trip_id = ?
            ORDER BY id
            """,
            (trip_id,),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


def get_checklist_item(
    item_id: int,
    tg_id: int,
) -> Optional[Dict[str, Any]]:
    """Возвращает пункт только владельцу поездки."""

    with connect() as conn:
        row = conn.execute(
            """
            SELECT c.*
            FROM checklist_items AS c
            JOIN trips AS t
                ON t.id = c.trip_id
            WHERE c.id = ? AND t.tg_id = ?
            """,
            (
                item_id,
                tg_id,
            ),
        ).fetchone()

    return (
        dict(row)
        if row
        else None
    )


def set_item_done(
    item_id: int,
    done: bool,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE checklist_items
            SET done = ?
            WHERE id = ?
            """,
            (
                1 if done else 0,
                item_id,
            ),
        )


def set_checklist_item_done(
    item_id: int,
    tg_id: int,
    done: bool,
) -> bool:
    """Атомарно меняет пункт чек-листа только владельцу поездки."""

    with connect() as conn:
        cursor = conn.execute(
            """
            UPDATE checklist_items
            SET done = ?
            WHERE id = ?
                AND EXISTS (
                    SELECT 1
                    FROM trips
                    WHERE trips.id = checklist_items.trip_id
                        AND trips.tg_id = ?
                )
            """,
            (
                1 if done else 0,
                item_id,
                tg_id,
            ),
        )

        return cursor.rowcount > 0


def create_web_login_token(
    tg_id: int,
    token_hash: str,
    expires_at: datetime,
) -> None:
    """Сохраняет хеш одноразовой ссылки для входа в веб-интерфейс."""

    token_hash = str(token_hash).strip()

    if not token_hash:
        raise ValueError(
            "token_hash не должен быть пустым."
        )

    with connect() as conn:
        _ensure_user_exists(
            conn,
            tg_id,
        )

        conn.execute(
            """
            INSERT INTO web_login_tokens (
                token_hash,
                tg_id,
                expires_at,
                created_at,
                used_at
            )
            VALUES (?, ?, ?, ?, NULL)
            """,
            (
                token_hash,
                tg_id,
                _timestamp(expires_at),
                _now(),
            ),
        )


def consume_web_login_token(
    token_hash: str,
    now: Optional[datetime] = None,
) -> Optional[int]:
    """Однократно поглощает действующий токен и возвращает tg_id."""

    token_hash = str(token_hash).strip()

    if not token_hash:
        return None

    current = _utc_datetime(now)
    current_text = _timestamp(current)

    with connect() as conn:
        conn.execute(
            "BEGIN IMMEDIATE"
        )

        row = conn.execute(
            """
            SELECT tokens.tg_id, tokens.expires_at, tokens.used_at
            FROM web_login_tokens AS tokens
            JOIN users ON users.tg_id = tokens.tg_id
            WHERE tokens.token_hash = ? AND users.deletion_token = ''
            """,
            (token_hash,),
        ).fetchone()

        if row is None or row["used_at"] is not None:
            return None

        expires_at = _parse_timestamp(
            row["expires_at"]
        )

        if (
            expires_at is None
            or expires_at <= current
        ):
            return None

        cursor = conn.execute(
            """
            UPDATE web_login_tokens
            SET used_at = ?
            WHERE token_hash = ?
                AND used_at IS NULL
            """,
            (
                current_text,
                token_hash,
            ),
        )

        if cursor.rowcount == 0:
            return None

        return int(row["tg_id"])


def purge_expired_web_login_tokens(
    now: Optional[datetime] = None,
) -> int:
    """Удаляет просроченные токены и возвращает число удалённых строк."""

    with connect() as conn:
        cursor = conn.execute(
            """
            DELETE FROM web_login_tokens
            WHERE expires_at <= ?
            """,
            (
                _timestamp(now),
            ),
        )

        return cursor.rowcount


def save_memory_item(
    tg_id: int,
    memory_id: str,
    kind: str,
    text: str,
    destination: str = "",
    source_trip_id: Optional[int] = None,
) -> int:
    """Создаёт или обновляет запись журнала памяти."""

    memory_id = str(memory_id).strip()
    kind = str(kind).strip()
    text = str(text).strip()
    destination = str(destination).strip()

    if not memory_id:
        raise ValueError(
            "memory_id не должен быть пустым."
        )

    if not kind:
        raise ValueError(
            "kind не должен быть пустым."
        )

    if not text:
        raise ValueError(
            "text не должен быть пустым."
        )

    now = _now()

    with connect() as conn:
        _ensure_user_exists(
            conn,
            tg_id,
        )

        if source_trip_id is not None:
            owned_trip = conn.execute(
                """
                SELECT 1
                FROM trips
                WHERE id = ? AND tg_id = ?
                """,
                (
                    source_trip_id,
                    tg_id,
                ),
            ).fetchone()

            if owned_trip is None:
                raise ValueError(
                    "Нельзя связать память "
                    "с чужой или несуществующей "
                    "поездкой."
                )

        conn.execute(
            """
            INSERT INTO memory_items (
                tg_id,
                memory_id,
                kind,
                text,
                destination,
                source_trip_id,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tg_id, memory_id)
            DO UPDATE SET
                kind = excluded.kind,
                text = excluded.text,
                destination =
                    excluded.destination,
                source_trip_id =
                    excluded.source_trip_id,
                updated_at =
                    excluded.updated_at
            """,
            (
                tg_id,
                memory_id,
                kind,
                text,
                destination,
                source_trip_id,
                now,
                now,
            ),
        )

        row = conn.execute(
            """
            SELECT id
            FROM memory_items
            WHERE tg_id = ? AND memory_id = ?
            """,
            (
                tg_id,
                memory_id,
            ),
        ).fetchone()

        if row is None:
            raise RuntimeError(
                "Не удалось сохранить "
                "запись памяти."
            )

        return int(row["id"])


def get_memory_item(
    item_id: int,
    tg_id: int,
) -> Optional[Dict[str, Any]]:
    """Возвращает запись памяти только владельцу."""

    with connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM memory_items
            WHERE id = ? AND tg_id = ?
            """,
            (
                item_id,
                tg_id,
            ),
        ).fetchone()

    return (
        dict(row)
        if row
        else None
    )


def get_memory_item_by_memory_id(
    tg_id: int,
    memory_id: str,
) -> Optional[Dict[str, Any]]:
    """Находит журнал по идентификатору Pinecone."""

    with connect() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM memory_items
            WHERE tg_id = ? AND memory_id = ?
            """,
            (
                tg_id,
                memory_id,
            ),
        ).fetchone()

    return (
        dict(row)
        if row
        else None
    )


def list_memory_items(
    tg_id: int,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Возвращает последние записи памяти семьи."""

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM memory_items
            WHERE tg_id = ?
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (
                tg_id,
                limit,
            ),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


def delete_memory_item(
    item_id: int,
    tg_id: int,
) -> bool:
    """Удаляет локальную запись только у владельца."""

    with connect() as conn:
        cursor = conn.execute(
            """
            DELETE FROM memory_items
            WHERE id = ? AND tg_id = ?
            """,
            (
                item_id,
                tg_id,
            ),
        )

        return cursor.rowcount > 0


def delete_memory_item_by_memory_id(
    tg_id: int,
    memory_id: str,
) -> bool:
    """Удаляет запись журнала по ID Pinecone."""

    with connect() as conn:
        cursor = conn.execute(
            """
            DELETE FROM memory_items
            WHERE tg_id = ? AND memory_id = ?
            """,
            (
                tg_id,
                memory_id,
            ),
        )

        return cursor.rowcount > 0


def clear_memory_items(
    tg_id: int,
) -> int:
    """Очищает журнал и возвращает число удалённых строк."""

    with connect() as conn:
        cursor = conn.execute(
            """
            DELETE FROM memory_items
            WHERE tg_id = ?
            """,
            (tg_id,),
        )

        return max(
            0,
            cursor.rowcount,
        )


def is_reminder_sent(
    trip_id: int,
    kind: str,
) -> bool:
    """Проверяет, подтверждена ли отправка напоминания."""

    with connect() as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM reminder_deliveries
            WHERE trip_id = ?
                AND kind = ?
                AND status = 'sent'
            """,
            (
                trip_id,
                str(kind).strip(),
            ),
        ).fetchone()

    return row is not None


def claim_reminder(
    trip_id: int,
    kind: str,
    now: Optional[datetime] = None,
    *,
    lease_minutes: int = (
        DEFAULT_REMINDER_LEASE_MINUTES
    ),
    max_attempts: int = (
        DEFAULT_REMINDER_MAX_ATTEMPTS
    ),
) -> bool:
    """Атомарно резервирует попытку отправки напоминания."""

    kind = str(kind).strip()

    if not kind:
        raise ValueError(
            "kind не должен быть пустым."
        )

    if lease_minutes <= 0:
        raise ValueError(
            "lease_minutes должен быть "
            "больше нуля."
        )

    if max_attempts <= 0:
        raise ValueError(
            "max_attempts должен быть "
            "больше нуля."
        )

    current = _utc_datetime(
        now
    )
    current_text = _timestamp(
        current
    )
    stale_before = current - timedelta(
        minutes=lease_minutes
    )

    with connect() as conn:
        # Блокировка не даёт двум циклам одновременно
        # забрать одно и то же напоминание.
        conn.execute(
            "BEGIN IMMEDIATE"
        )

        row = conn.execute(
            """
            SELECT
                status,
                attempts,
                claimed_at
            FROM reminder_deliveries
            WHERE trip_id = ? AND kind = ?
            """,
            (
                trip_id,
                kind,
            ),
        ).fetchone()

        if row is None:
            conn.execute(
                """
                INSERT INTO reminder_deliveries (
                    trip_id,
                    kind,
                    status,
                    attempts,
                    claimed_at,
                    sent_at,
                    last_error
                )
                VALUES (?, ?, 'sending', 1, ?, NULL, '')
                """,
                (
                    trip_id,
                    kind,
                    current_text,
                ),
            )

            return True

        status = str(
            row["status"]
        )
        attempts = int(
            row["attempts"]
        )

        if (
            status == REMINDER_STATUS_SENT
            or attempts >= max_attempts
        ):
            return False

        if status == REMINDER_STATUS_SENDING:
            claimed_at = _parse_timestamp(
                row["claimed_at"]
            )

            if (
                claimed_at is not None
                and claimed_at > stale_before
            ):
                return False

        cursor = conn.execute(
            """
            UPDATE reminder_deliveries
            SET status = 'sending',
                attempts = attempts + 1,
                claimed_at = ?,
                sent_at = NULL,
                last_error = ''
            WHERE trip_id = ? AND kind = ?
            """,
            (
                current_text,
                trip_id,
                kind,
            ),
        )

        return cursor.rowcount > 0


def mark_reminder_sent(
    trip_id: int,
    kind: str,
    sent_at: Optional[datetime] = None,
) -> bool:
    """Подтверждает успешную отправку в Telegram."""

    with connect() as conn:
        cursor = conn.execute(
            """
            UPDATE reminder_deliveries
            SET status = 'sent',
                sent_at = ?,
                last_error = ''
            WHERE trip_id = ?
                AND kind = ?
                AND status = 'sending'
            """,
            (
                _timestamp(
                    sent_at
                ),
                trip_id,
                str(kind).strip(),
            ),
        )

        return cursor.rowcount > 0


def mark_reminder_failed(
    trip_id: int,
    kind: str,
    error: Any,
) -> bool:
    """Сохраняет ошибку, не сбрасывая счётчик попыток."""

    error_text = " ".join(
        str(error).split()
    )[:1000]

    with connect() as conn:
        cursor = conn.execute(
            """
            UPDATE reminder_deliveries
            SET status = 'failed',
                sent_at = NULL,
                last_error = ?
            WHERE trip_id = ?
                AND kind = ?
                AND status = 'sending'
            """,
            (
                error_text,
                trip_id,
                str(kind).strip(),
            ),
        )

        return cursor.rowcount > 0


def get_reminder_statuses(
    trip_id: int,
    tg_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Возвращает статусы с необязательной проверкой владельца."""

    query = (
        "SELECT r.* "
        "FROM reminder_deliveries AS r "
        "JOIN trips AS t "
        "ON t.id = r.trip_id "
        "WHERE r.trip_id = ?"
    )
    params: List[Any] = [
        trip_id
    ]

    if tg_id is not None:
        query += " AND t.tg_id = ?"
        params.append(
            tg_id
        )

    query += " ORDER BY r.kind"

    with connect() as conn:
        rows = conn.execute(
            query,
            params,
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]
