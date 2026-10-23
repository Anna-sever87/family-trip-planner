"""Автономная проверка общего дневного лимита AI-запросов.

Тест использует только test_ai_limit.db и не изменяет planner.db.
Сеть, Telegram, YandexGPT и Pinecone не вызываются.
"""

from __future__ import annotations

import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from typing import Optional

import db


PROJECT_DIR = Path(__file__).resolve().parent
TEST_DB_PATH = PROJECT_DIR / "test_ai_limit.db"

TEST_USER_ID = 92001
OTHER_USER_ID = 92002
STALE_USER_ID = 92003

USAGE_DAY = "2026-10-10"
NEXT_DAY = "2026-10-11"


def _fingerprint(
    path: Path,
) -> Optional[tuple[int, str]]:
    """Возвращает размер и SHA-256 файла."""

    if not path.exists():
        return None

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as source:
        for chunk in iter(
            lambda: source.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return (
        path.stat().st_size,
        digest.hexdigest(),
    )


def _remove_test_files() -> None:
    for suffix in (
        "",
        "-wal",
        "-shm",
        "-journal",
    ):
        Path(
            f"{TEST_DB_PATH}{suffix}"
        ).unlink(
            missing_ok=True
        )


def _ok(
    message: str,
) -> None:
    print(
        f"[OK] {message}"
    )


def _claim_together(
    barrier: Barrier,
    source: str,
    trip_id: int,
    now: datetime,
) -> db.AIRequestClaim:
    barrier.wait(
        timeout=10
    )

    return db.claim_ai_request(
        TEST_USER_ID,
        USAGE_DAY,
        "itinerary_edit",
        source,
        trip_id=trip_id,
        daily_limit=5,
        now=now,
    )


def _test_schema() -> None:
    with db.connect() as conn:
        columns = {
            str(
                row["name"]
            )
            for row in conn.execute(
                """
                PRAGMA table_info(ai_requests)
                """
            ).fetchall()
        }

        assert {
            "claim_id",
            "tg_id",
            "usage_date",
            "kind",
            "source",
            "trip_id",
            "status",
            "claimed_at",
            "finished_at",
        }.issubset(
            columns
        )

        index_row = conn.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE type = 'index'
                AND name = 'idx_ai_requests_running_user'
            """
        ).fetchone()

        assert index_row is not None

        index_sql = " ".join(
            str(
                index_row["sql"]
            ).lower().split()
        )

        assert (
            "unique index"
            in index_sql
        )

        assert (
            "where status = 'running'"
            in index_sql
        )

        foreign_keys = conn.execute(
            """
            PRAGMA foreign_key_list(ai_requests)
            """
        ).fetchall()

        assert any(
            row["table"] == "users"
            and row["from"] == "tg_id"
            and str(
                row["on_delete"]
            ).upper() == "CASCADE"
            for row in foreign_keys
        )

        assert any(
            row["table"] == "trips"
            and row["from"] == "trip_id"
            and str(
                row["on_delete"]
            ).upper() == "SET NULL"
            for row in foreign_keys
        )

    _ok(
        "таблица журнала и уникальная "
        "активная аренда созданы повторяемо"
    )


def _test_concurrent_claim(
    trip_id: int,
    now: datetime,
) -> str:
    barrier = Barrier(
        2
    )

    with ThreadPoolExecutor(
        max_workers=2
    ) as executor:
        futures = [
            executor.submit(
                _claim_together,
                barrier,
                source,
                trip_id,
                now,
            )
            for source in (
                "telegram",
                "web",
            )
        ]

        claims = [
            future.result(
                timeout=15
            )
            for future in futures
        ]

    statuses = sorted(
        claim.status
        for claim in claims
    )

    assert statuses == [
        "busy",
        "ok",
    ]

    accepted = next(
        claim
        for claim in claims
        if claim.status == "ok"
    )

    rejected = next(
        claim
        for claim in claims
        if claim.status == "busy"
    )

    assert accepted.claim_id
    assert accepted.used == 1
    assert accepted.limit == 5
    assert rejected.claim_id is None
    assert rejected.used == 1

    _ok(
        "два одновременных источника "
        "получают ровно одну AI-аренду"
    )

    return str(
        accepted.claim_id
    )


def _test_finish_and_limit(
    first_claim_id: str,
    trip_id: int,
    now: datetime,
) -> None:
    assert db.finish_ai_request(
        first_claim_id,
        succeeded=True,
        now=now
        + timedelta(
            minutes=1
        ),
    ) is True

    assert db.finish_ai_request(
        first_claim_id,
        succeeded=False,
        now=now
        + timedelta(
            minutes=2
        ),
    ) is False

    second = db.claim_ai_request(
        TEST_USER_ID,
        USAGE_DAY,
        "trip_generation",
        "telegram",
        trip_id=trip_id,
        daily_limit=2,
        now=now
        + timedelta(
            minutes=3
        ),
    )

    assert second.status == "ok"
    assert second.claim_id
    assert second.used == 2

    assert db.finish_ai_request(
        str(
            second.claim_id
        ),
        succeeded=False,
        now=now
        + timedelta(
            minutes=4
        ),
    ) is True

    limited = db.claim_ai_request(
        TEST_USER_ID,
        USAGE_DAY,
        "itinerary_edit",
        "web",
        trip_id=trip_id,
        daily_limit=2,
        now=now
        + timedelta(
            minutes=5
        ),
    )

    assert limited == db.AIRequestClaim(
        status="limit",
        claim_id=None,
        used=2,
        limit=2,
    )

    assert db.get_ai_request_usage(
        TEST_USER_ID,
        USAGE_DAY,
    ) == 2

    with db.connect() as conn:
        statuses = [
            str(
                row["status"]
            )
            for row in conn.execute(
                """
                SELECT status
                FROM ai_requests
                WHERE tg_id = ?
                    AND usage_date = ?
                ORDER BY claimed_at, claim_id
                """,
                (
                    TEST_USER_ID,
                    USAGE_DAY,
                ),
            ).fetchall()
        ]

    assert sorted(
        statuses
    ) == [
        "failed",
        "succeeded",
    ]

    _ok(
        "успех и ошибка завершаются точно, "
        "а ошибка тоже сохраняет дневную квоту"
    )


def _test_users_and_days(
    trip_id: int,
    now: datetime,
) -> None:
    other = db.claim_ai_request(
        OTHER_USER_ID,
        USAGE_DAY,
        "trip_generation",
        "telegram",
        daily_limit=1,
        now=now,
    )

    assert other.status == "ok"
    assert other.claim_id
    assert other.used == 1

    assert db.finish_ai_request(
        str(
            other.claim_id
        ),
        succeeded=True,
        now=now,
    ) is True

    assert db.get_ai_request_usage(
        OTHER_USER_ID,
        USAGE_DAY,
    ) == 1

    assert db.get_ai_request_usage(
        TEST_USER_ID,
        USAGE_DAY,
    ) == 2

    next_day = db.claim_ai_request(
        TEST_USER_ID,
        NEXT_DAY,
        "itinerary_edit",
        "web",
        trip_id=trip_id,
        daily_limit=1,
        now=now
        + timedelta(
            days=1
        ),
    )

    assert next_day.status == "ok"
    assert next_day.claim_id
    assert next_day.used == 1

    assert db.finish_ai_request(
        str(
            next_day.claim_id
        ),
        succeeded=True,
        now=now
        + timedelta(
            days=1,
            minutes=1,
        ),
    ) is True

    assert db.get_ai_request_usage(
        TEST_USER_ID,
        NEXT_DAY,
    ) == 1

    _ok(
        "квота изолирована по пользователям "
        "и начинается заново в новый день"
    )


def _test_stale_claim(
    now: datetime,
) -> None:
    old = db.claim_ai_request(
        STALE_USER_ID,
        USAGE_DAY,
        "trip_generation",
        "telegram",
        daily_limit=5,
        lease_minutes=30,
        now=now,
    )

    assert old.status == "ok"
    assert old.claim_id

    fresh = db.claim_ai_request(
        STALE_USER_ID,
        USAGE_DAY,
        "itinerary_edit",
        "web",
        daily_limit=5,
        lease_minutes=30,
        now=now
        + timedelta(
            minutes=31
        ),
    )

    assert fresh.status == "ok"
    assert fresh.claim_id
    assert fresh.claim_id != old.claim_id
    assert fresh.used == 2

    # Поздний ответ старого процесса не имеет права завершить
    # новую аренду: UPDATE привязан к точному claim_id.
    assert db.finish_ai_request(
        str(
            old.claim_id
        ),
        succeeded=True,
        now=now
        + timedelta(
            minutes=32
        ),
    ) is False

    with db.connect() as conn:
        old_status = conn.execute(
            """
            SELECT status
            FROM ai_requests
            WHERE claim_id = ?
            """,
            (
                old.claim_id,
            ),
        ).fetchone()

        fresh_status = conn.execute(
            """
            SELECT status
            FROM ai_requests
            WHERE claim_id = ?
            """,
            (
                fresh.claim_id,
            ),
        ).fetchone()

    assert old_status["status"] == "expired"
    assert fresh_status["status"] == "running"

    assert db.finish_ai_request(
        str(
            fresh.claim_id
        ),
        succeeded=True,
        now=now
        + timedelta(
            minutes=33
        ),
    ) is True

    _ok(
        "протухшая аренда освобождается, "
        "а её старый ID не завершает новую"
    )


def _test_owner_and_foreign_keys(
    trip_id: int,
    now: datetime,
) -> None:
    try:
        db.claim_ai_request(
            OTHER_USER_ID,
            NEXT_DAY,
            "itinerary_edit",
            "web",
            trip_id=trip_id,
            daily_limit=5,
            now=now,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Удалось связать AI-запрос "
            "с чужой поездкой."
        )

    with db.connect() as conn:
        violations = conn.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        try:
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
                    claimed_at
                )
                VALUES (
                    'invalid-foreign-key',
                    999999999,
                    '2026-10-12',
                    'test',
                    'test',
                    NULL,
                    'failed',
                    '2026-10-12T00:00:00+00:00'
                )
                """
            )

        except sqlite3.IntegrityError:
            pass

        else:
            raise AssertionError(
                "Внешний ключ пользователя "
                "не сработал."
            )

    assert violations == []

    _ok(
        "чужая поездка отклоняется, "
        "внешние ключи базы не нарушены"
    )


def main() -> None:
    original_db_path = db.DB_PATH
    real_db_path = Path(
        original_db_path
    ).resolve()
    real_db_before = _fingerprint(
        real_db_path
    )

    try:
        _remove_test_files()

        db.DB_PATH = str(
            TEST_DB_PATH
        )

        db.init_db()
        db.init_db()

        _test_schema()

        trip_id = db.save_trip(
            TEST_USER_ID,
            {
                "destination": "Париж",
                "start_date": "2026-10-12",
                "end_date": "2026-10-12",
            },
            {
                "title": "Тестовый маршрут",
                "days": [],
            },
        )

        base_time = datetime(
            2026,
            10,
            10,
            10,
            0,
            tzinfo=timezone.utc,
        )

        first_claim_id = (
            _test_concurrent_claim(
                trip_id,
                base_time,
            )
        )

        _test_finish_and_limit(
            first_claim_id,
            trip_id,
            base_time,
        )

        _test_users_and_days(
            trip_id,
            base_time,
        )

        _test_stale_claim(
            base_time,
        )

        _test_owner_and_foreign_keys(
            trip_id,
            base_time,
        )

        print(
            "\nИтог: единый атомарный журнал "
            "и дневной лимит AI-запросов "
            "полностью работают."
        )

    finally:
        db.DB_PATH = original_db_path

        _remove_test_files()

        assert _fingerprint(
            real_db_path
        ) == real_db_before, (
            "Настоящий planner.db изменился."
        )

        print(
            "Временная база "
            "test_ai_limit.db удалена."
        )

        print(
            "Настоящий planner.db "
            "не изменился."
        )


if __name__ == "__main__":
    main()
