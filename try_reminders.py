"""Полная автономная проверка напоминаний.

Тест проверяет БД, расписание и тексты без сети и Telegram.
Использует только test_reminders.db и не изменяет planner.db.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import db
import reminders


PROJECT_DIR = Path(__file__).resolve().parent
TEST_DB_PATH = PROJECT_DIR / "test_reminders.db"

TEST_USER_ID = 101
OTHER_USER_ID = 202
STATUS_USER_ID = 303


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


def _restore_env(
    name: str,
    value: Optional[str],
) -> None:
    if value is None:
        os.environ.pop(
            name,
            None,
        )
    else:
        os.environ[name] = value


@contextmanager
def _environment(
    *,
    tz: Optional[str] = None,
    days_before: Optional[str] = None,
) -> Iterator[None]:
    """Временно меняет настройки и возвращает прежние."""

    old_tz = os.environ.get(
        "REMINDER_TZ"
    )

    old_days = os.environ.get(
        "REMINDER_DAYS_BEFORE"
    )

    try:
        _restore_env(
            "REMINDER_TZ",
            tz,
        )

        _restore_env(
            "REMINDER_DAYS_BEFORE",
            days_before,
        )

        yield

    finally:
        _restore_env(
            "REMINDER_TZ",
            old_tz,
        )

        _restore_env(
            "REMINDER_DAYS_BEFORE",
            old_days,
        )


class _LogCapture(
    logging.Handler
):
    def __init__(self) -> None:
        super().__init__()
        self.messages = []

    def emit(
        self,
        record: logging.LogRecord,
    ) -> None:
        self.messages.append(
            record.getMessage()
        )


def _assert_runtime_error(
    callback,
    expected: str,
) -> None:
    try:
        callback()

    except RuntimeError as error:
        assert expected in str(
            error
        )

    else:
        raise AssertionError(
            "Ожидался RuntimeError."
        )


def _remove_test_files() -> None:
    """Удаляет временную базу и файлы SQLite."""

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


def _test_settings() -> None:
    with _environment(
        tz="Asia/Novosibirsk",
        days_before="5",
    ):
        assert str(
            reminders.get_tz()
        ) == "Asia/Novosibirsk"

        assert (
            reminders.get_days_before()
            == 5
        )

    with _environment(
        tz="Unknown/Nowhere",
    ):
        _assert_runtime_error(
            reminders.get_tz,
            "REMINDER_TZ",
        )

    for invalid in (
        "0",
        "31",
        "abc",
        "",
    ):
        with _environment(
            tz="UTC",
            days_before=invalid,
        ):
            _assert_runtime_error(
                reminders.get_days_before,
                "от 1 до 30",
            )

    _ok(
        "часовой пояс и срок первого "
        "напоминания проходят проверку"
    )


def _test_plan_and_dst() -> None:
    trip = {
        "id": 501,
        "start_date": "2026-03-30",
        "end_date": "2026-03-31",
    }

    with _environment(
        tz="Europe/Berlin",
        days_before="2",
    ):
        plan = reminders.reminder_plan(
            trip
        )

        assert [
            kind
            for kind, _ in plan
        ] == [
            "before",
            "eve",
            "day1",
            "day2",
        ]

        assert (
            plan[0][1].date().isoformat()
            == "2026-03-28"
        )

        assert all(
            value.tzinfo is not None
            and value.utcoffset() is not None
            for _, value in plan
        )

        # 29 марта в Берлине начинается
        # летнее время.
        assert (
            plan[0][1].utcoffset()
            == timedelta(
                hours=1
            )
        )

        assert (
            plan[1][1].utcoffset()
            == timedelta(
                hours=2
            )
        )

        local_value = (
            reminders.normalize_datetime(
                datetime(
                    2026,
                    3,
                    29,
                    8,
                    tzinfo=timezone.utc,
                )
            )
        )

        assert (
            local_value.hour
            == 10
        )

        assert (
            local_value.utcoffset()
            == timedelta(
                hours=2
            )
        )

    with _environment(
        tz="UTC",
        days_before="3",
    ):
        for broken in (
            {
                "start_date": "не дата",
                "end_date": "2026-03-31",
            },
            {
                "start_date": "2026-04-02",
                "end_date": "2026-04-01",
            },
        ):
            try:
                reminders.reminder_plan(
                    broken
                )

            except ValueError:
                pass

            else:
                raise AssertionError(
                    "Неверные даты должны "
                    "отклоняться."
                )

    _ok(
        "расписание использует стабильные "
        "виды и корректно переживает DST"
    )


def _test_due_and_malformed() -> None:
    trip = {
        "id": 510,
        "destination": "Париж",
        "start_date": "2026-07-10",
        "end_date": "2026-07-10",
        "created_at": (
            "2026-07-01T09:00:00+07:00"
        ),
    }

    with _environment(
        tz="Asia/Novosibirsk",
        days_before="3",
    ):
        tz = reminders.get_tz()

        due = reminders.due_reminders(
            [
                trip
            ],
            lambda trip_id, kind: False,
            datetime(
                2026,
                7,
                7,
                10,
                tzinfo=tz,
            ),
        )

        assert [
            kind
            for _, kind in due
        ] == [
            "before"
        ]

        almost_expired = (
            reminders.due_reminders(
                [
                    trip
                ],
                lambda trip_id, kind: False,
                datetime(
                    2026,
                    7,
                    7,
                    21,
                    59,
                    tzinfo=tz,
                ),
            )
        )

        assert len(
            almost_expired
        ) == 1

        expired = reminders.due_reminders(
            [
                trip
            ],
            lambda trip_id, kind: False,
            datetime(
                2026,
                7,
                7,
                22,
                tzinfo=tz,
            ),
        )

        assert expired == []

        old_alias_sent = (
            reminders.due_reminders(
                [
                    trip
                ],
                lambda trip_id, kind: (
                    kind == "d3"
                ),
                datetime(
                    2026,
                    7,
                    7,
                    10,
                    tzinfo=tz,
                ),
            )
        )

        assert (
            old_alias_sent
            == []
        )

        created_late = dict(
            trip,
            created_at=(
                "2026-07-07T10:01:00+07:00"
            ),
        )

        suppressed = reminders.due_reminders(
            [
                created_late
            ],
            lambda trip_id, kind: False,
            datetime(
                2026,
                7,
                7,
                10,
                2,
                tzinfo=tz,
            ),
        )

        assert suppressed == []

        # Naive datetime считается временем
        # из REMINDER_TZ.
        naive_due = reminders.due_reminders(
            [
                trip
            ],
            lambda trip_id, kind: False,
            datetime(
                2026,
                7,
                7,
                10,
            ),
        )

        assert len(
            naive_due
        ) == 1

    valid = {
        "id": 511,
        "start_date": "2026-08-10",
        "end_date": "2026-08-10",
        "created_at": (
            "2026-08-01T00:00:00+00:00"
        ),
    }

    broken = {
        "id": 512,
        "start_date": "сломано",
        "end_date": "2026-08-10",
        "created_at": (
            "2026-08-01T00:00:00+00:00"
        ),
    }

    broken_created = {
        "id": 513,
        "start_date": "2026-08-10",
        "end_date": "2026-08-10",
        "created_at": "сломано",
    }

    capture = _LogCapture()

    old_level = (
        reminders.logger.level
    )

    reminders.logger.setLevel(
        logging.WARNING
    )

    reminders.logger.addHandler(
        capture
    )

    try:
        with _environment(
            tz="UTC",
            days_before="3",
        ):
            due = reminders.due_reminders(
                [
                    broken,
                    valid,
                    broken_created,
                ],
                lambda trip_id, kind: False,
                datetime(
                    2026,
                    8,
                    7,
                    10,
                    tzinfo=timezone.utc,
                ),
            )

    finally:
        reminders.logger.removeHandler(
            capture
        )

        reminders.logger.setLevel(
            old_level
        )

    assert len(
        due
    ) == 1

    assert (
        due[0][0]["id"]
        == 511
    )

    assert (
        due[0][1]
        == "before"
    )

    assert len(
        capture.messages
    ) == 2

    _ok(
        "окно отправки и created_at работают, "
        "повреждённые поездки пропускаются"
    )


def _test_descriptions() -> None:
    request = {
        "destination": "Берлин",
        "start_date": "2026-06-10",
        "end_date": "2026-06-13",
    }

    trip_id = db.save_trip(
        STATUS_USER_ID,
        request,
        {
            "title": "Берлин",
            "days": [],
        },
    )

    with db.connect() as conn:
        conn.execute(
            """
            UPDATE trips
            SET created_at = ?
            WHERE id = ?
            """,
            (
                "2026-06-08T12:00:00+00:00",
                trip_id,
            ),
        )

        conn.executemany(
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
            VALUES (?, ?, ?, 1, ?, NULL, '')
            """,
            [
                (
                    trip_id,
                    "eve",
                    "sent",
                    "2026-06-09T18:00:00+00:00",
                ),
                (
                    trip_id,
                    "day1",
                    "sending",
                    "2026-06-10T08:00:00+00:00",
                ),
                (
                    trip_id,
                    "day2",
                    "failed",
                    "2026-06-11T08:00:00+00:00",
                ),
            ],
        )

        rows = conn.execute(
            """
            SELECT *
            FROM reminder_deliveries
            WHERE trip_id = ?
            ORDER BY kind
            """,
            (
                trip_id,
            ),
        ).fetchall()

    trip = db.get_trip(
        trip_id,
        STATUS_USER_ID,
    )

    assert trip is not None

    with _environment(
        tz="UTC",
        days_before="3",
    ):
        text = reminders.describe_plan(
            trip,
            rows,
            now=datetime(
                2026,
                6,
                12,
                22,
                tzinfo=timezone.utc,
            ),
        )

        for expected in (
            "пропущено",
            "отправлено",
            "отправляется",
            "ошибка отправки",
            "срок прошёл",
            "ожидает",
        ):
            assert (
                expected
                in text
            )

        old_set = reminders.describe_plan(
            trip,
            {
                "d3"
            },
            now=datetime(
                2026,
                6,
                6,
                tzinfo=timezone.utc,
            ),
        )

        assert (
            "за 3 дня до поездки: отправлено"
            in old_set
        )

    _ok(
        "описание расписания различает все "
        "статусы и принимает строки SQLite"
    )


def _test_texts() -> None:
    trip = {
        "id": 530,
        "destination": "Париж",
        "start_date": "2026-10-12",
        "request": {
            "weather": {
                "days": [
                    {
                        "tmin": 8.5,
                        "tmax": 19,
                        "rain_mm": 0,
                    }
                ]
            }
        },
        "itinerary": {
            "days": [
                {
                    "date": "2026-10-12",
                    "theme": "Монмартр",
                    "items": [
                        {
                            "time": "09:00",
                            "title": "Музей",
                        },
                        None,
                    ],
                    "rainy_alternative": (
                        "Интерактивный центр"
                    ),
                }
            ]
        },
    }

    checklist = [
        {
            "text": "Паспорта",
            "done": True,
        },
        {
            "text": "Аптечка",
            "done": False,
        },
        None,
    ]

    with _environment(
        tz="Asia/Novosibirsk",
        days_before="5",
    ):
        before = (
            reminders.build_reminder_text(
                "before",
                trip,
                checklist,
            )
        )

        assert (
            "Через 5 дней"
            in before
        )

        assert (
            "Аптечка"
            in before
        )

        assert (
            "/checklist 530"
            in before
        )

        old_before = (
            reminders.build_reminder_text(
                "d3",
                trip,
                checklist,
            )
        )

        assert (
            "Через 5 дней"
            in old_before
        )

        eve = (
            reminders.build_reminder_text(
                "eve",
                trip,
                checklist,
            )
        )

        assert (
            "от 8.5 до 19 °C"
            in eve
        )

        assert (
            "/pdf 530"
            in eve
        )

        day = (
            reminders.build_reminder_text(
                "day1",
                trip,
                checklist,
            )
        )

        assert (
            "09:00 Музей"
            in day
        )

        assert (
            "Интерактивный центр"
            in day
        )

        assert (
            "/trip 530"
            in day
        )

        for kind in (
            "before",
            "eve",
            "day1",
            "неизвестно",
        ):
            result = (
                reminders.build_reminder_text(
                    kind,
                    {
                        "request": {
                            "weather": "сломано"
                        },
                        "itinerary": {
                            "days": [
                                None
                            ]
                        },
                    },
                    [
                        None,
                        {
                            "done": False
                        },
                    ],
                )
            )

            assert isinstance(
                result,
                str,
            )

            assert result

    _ok(
        "тексты учитывают настройку и "
        "устойчивы к неполным данным"
    )


def main() -> None:
    original_db_path = db.DB_PATH

    original_tz = os.environ.get(
        "REMINDER_TZ"
    )

    original_days = os.environ.get(
        "REMINDER_DAYS_BEFORE"
    )

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

        timestamp = datetime.fromisoformat(
            db._now()
        )

        assert (
            timestamp.tzinfo
            is not None
        )

        assert (
            timestamp.utcoffset()
            is not None
        )

        _ok(
            "время базы сохраняется "
            "вместе с часовым поясом"
        )

        with db.connect() as conn:
            columns = {
                str(
                    row["name"]
                )
                for row in conn.execute(
                    """
                    PRAGMA table_info(
                        reminder_deliveries
                    )
                    """
                ).fetchall()
            }

            assert {
                "trip_id",
                "kind",
                "status",
                "attempts",
                "claimed_at",
                "sent_at",
                "last_error",
            }.issubset(
                columns
            )

            foreign_keys = conn.execute(
                """
                PRAGMA foreign_key_list(
                    reminder_deliveries
                )
                """
            ).fetchall()

            assert any(
                row["table"] == "trips"
                and row["from"] == "trip_id"
                and str(
                    row["on_delete"]
                ).upper() == "CASCADE"
                for row in foreign_keys
            )

        _ok(
            "таблица напоминаний создаётся "
            "повторно и использует FK CASCADE"
        )

        request = {
            "destination": "Париж",
            "start_date": "2026-11-10",
            "end_date": "2026-11-12",
        }

        other_request = {
            "destination": "Рим",
            "start_date": "2026-12-20",
            "end_date": "2026-12-22",
        }

        itinerary = {
            "title": "Тестовый маршрут",
            "days": [],
        }

        trip_id = db.save_trip(
            TEST_USER_ID,
            request,
            itinerary,
        )

        other_trip_id = db.save_trip(
            OTHER_USER_ID,
            other_request,
            itinerary,
        )

        candidates = (
            db.list_reminder_candidate_trips(
                "2026-11-09",
                "2026-11-13",
            )
        )

        assert [
            item["id"]
            for item in candidates
        ] == [
            trip_id
        ]

        assert (
            candidates[0]["tg_id"]
            == TEST_USER_ID
        )

        _ok(
            "диспетчер выбирает поездки "
            "только из нужного окна дат"
        )

        base_time = datetime(
            2026,
            11,
            7,
            10,
            0,
            tzinfo=timezone.utc,
        )

        assert db.claim_reminder(
            trip_id,
            "d3",
            base_time,
        ) is True

        assert db.claim_reminder(
            trip_id,
            "d3",
            base_time,
        ) is False

        assert db.claim_reminder(
            trip_id,
            "d3",
            base_time.replace(
                minute=14
            ),
        ) is False

        assert db.claim_reminder(
            trip_id,
            "d3",
            base_time.replace(
                minute=16
            ),
        ) is True

        statuses = (
            db.get_reminder_statuses(
                trip_id,
                TEST_USER_ID,
            )
        )

        assert len(
            statuses
        ) == 1

        assert (
            statuses[0]["status"]
            == "sending"
        )

        assert (
            statuses[0]["attempts"]
            == 2
        )

        _ok(
            "двойная отправка блокируется, "
            "а зависшая попытка освобождается"
        )

        assert db.mark_reminder_failed(
            trip_id,
            "d3",
            "ошибка сети",
        ) is True

        assert db.claim_reminder(
            trip_id,
            "d3",
            base_time.replace(
                minute=17
            ),
        ) is True

        assert db.mark_reminder_failed(
            trip_id,
            "d3",
            "повторная ошибка сети",
        ) is True

        assert db.claim_reminder(
            trip_id,
            "d3",
            base_time.replace(
                minute=18
            ),
        ) is False

        statuses = (
            db.get_reminder_statuses(
                trip_id,
                TEST_USER_ID,
            )
        )

        first_status = next(
            item
            for item in statuses
            if item["kind"] == "d3"
        )

        assert (
            first_status["attempts"]
            == 3
        )

        assert (
            first_status["status"]
            == "failed"
        )

        _ok(
            "ошибки записываются, "
            "число повторных попыток ограничено"
        )

        assert db.claim_reminder(
            trip_id,
            "d1",
            base_time,
        ) is True

        assert db.mark_reminder_sent(
            trip_id,
            "d1",
            base_time,
        ) is True

        assert db.is_reminder_sent(
            trip_id,
            "d1",
        ) is True

        assert db.claim_reminder(
            trip_id,
            "d1",
            base_time.replace(
                hour=12
            ),
        ) is False

        _ok(
            "успешное напоминание "
            "не отправляется повторно"
        )

        assert (
            db.get_reminder_statuses(
                trip_id,
                OTHER_USER_ID,
            )
            == []
        )

        assert len(
            db.get_reminder_statuses(
                trip_id,
                TEST_USER_ID,
            )
        ) == 2

        assert db.get_trip(
            trip_id,
            OTHER_USER_ID,
        ) is None

        _ok(
            "статусы защищены "
            "от другого пользователя"
        )

        assert db.claim_reminder(
            999999,
            "d3",
            base_time,
        ) is False

        with db.connect() as conn:
            assert conn.execute(
                "SELECT COUNT(*) FROM reminder_deliveries WHERE trip_id = ?",
                (999999,),
            ).fetchone()[0] == 0

        _ok(
            "устаревшее напоминание без поездки "
            "не резервируется и не создаёт запись"
        )

        # Проверяем ограничение БД отдельно от защитного отказа claim_reminder.
        try:
            with db.connect() as conn:
                conn.execute(
                    """
                    INSERT INTO reminder_deliveries (
                        trip_id, kind, status, attempts, claimed_at
                    ) VALUES (?, ?, 'sending', 1, ?)
                    """,
                    (999999, "d3", base_time.isoformat()),
                )

        except sqlite3.IntegrityError as exc:
            assert "FOREIGN KEY" in str(exc), str(exc)

        else:
            raise AssertionError(
                "Удалось создать напоминание "
                "для несуществующей поездки."
            )

        _ok(
            "внешний ключ запрещает "
            "напоминание без поездки"
        )

        assert db.claim_reminder(
            other_trip_id,
            "d3",
            base_time,
        ) is True

        assert db.delete_trip(
            other_trip_id,
            OTHER_USER_ID,
        ) is True

        assert (
            db.get_reminder_statuses(
                other_trip_id
            )
            == []
        )

        with db.connect() as conn:
            reminder_count = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM reminder_deliveries
                    WHERE trip_id = ?
                    """,
                    (
                        other_trip_id,
                    ),
                ).fetchone()[0]
            )

            violations = conn.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()

        assert (
            reminder_count
            == 0
        )

        assert violations == []

        _ok(
            "удаление поездки удаляет "
            "её журнал напоминаний"
        )

        _test_settings()
        _test_plan_and_dst()
        _test_due_and_malformed()
        _test_descriptions()
        _test_texts()

        print(
            "\nИтог: база данных, расписание "
            "и тексты напоминаний полностью работают."
        )

    finally:
        db.DB_PATH = original_db_path

        _restore_env(
            "REMINDER_TZ",
            original_tz,
        )

        _restore_env(
            "REMINDER_DAYS_BEFORE",
            original_days,
        )

        _remove_test_files()

        assert _fingerprint(
            real_db_path
        ) == real_db_before, (
            "Настоящий planner.db изменился."
        )

        print(
            "Временная база "
            "test_reminders.db удалена."
        )

        print(
            "Настоящий planner.db "
            "не изменился."
        )


if __name__ == "__main__":
    main()
