"""Автономная проверка Telegram-интеграции напоминаний.

Тест не запускает polling, планировщик, сеть и настоящие Telegram API.
Используется только временная SQLite-база; planner.db не изменяется.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parent

import db
import dotenv
import httpx
import telebot
from apscheduler.schedulers import background as aps_background


PROJECT_DIR = SCRIPT_DIR
TEST_DB_PATH = PROJECT_DIR / "test_bot_reminders.db"

OWNER_ID = 81001
OTHER_ID = 81002
FAIL_ID = 81003
EVENTS = []


def _remove_test_db() -> None:
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


def _fingerprint(
    path: Path,
):
    if not path.exists():
        return None

    content = path.read_bytes()
    return (
        len(content),
        hashlib.sha256(
            content
        ).hexdigest(),
    )


def _ok(
    text: str,
) -> None:
    print(
        f"[OK] {text}"
    )


def _message(
    text: str,
    chat_id: int,
    chat_type: str = "private",
):
    return SimpleNamespace(
        text=text,
        content_type="text",
        chat=SimpleNamespace(
            id=chat_id,
            type=chat_type,
        ),
        from_user=SimpleNamespace(
            id=chat_id,
            first_name="Тест",
        ),
    )


class FakeBot:
    """Минимальная локальная замена TeleBot."""

    def __init__(self) -> None:
        self.messages = []
        self.handlers = []
        self.fail_once = set()
        self.raise_get_me = False
        self.command_calls = 0
        self.commands = []
        self.get_me_calls = 0
        self.polling_calls = 0

    def message_handler(
        self,
        *args,
        **kwargs,
    ):
        def register(function):
            self.handlers.append(
                (
                    "message",
                    function,
                )
            )
            return function

        return register

    def callback_query_handler(
        self,
        *args,
        **kwargs,
    ):
        def register(function):
            self.handlers.append(
                (
                    "callback",
                    function,
                )
            )
            return function

        return register

    def send_message(
        self,
        chat_id: int,
        text: str,
        *args,
        **kwargs,
    ):
        if chat_id in self.fail_once:
            self.fail_once.remove(
                chat_id
            )
            raise RuntimeError(
                "тестовая ошибка Telegram"
            )

        self.messages.append(
            {
                "chat_id": chat_id,
                "text": str(text),
                "kwargs": kwargs,
            }
        )
        return SimpleNamespace(
            message_id=len(
                self.messages
            )
        )

    def send_chat_action(
        self,
        *args,
        **kwargs,
    ) -> None:
        return None

    def answer_callback_query(
        self,
        *args,
        **kwargs,
    ) -> None:
        return None

    def edit_message_text(
        self,
        *args,
        **kwargs,
    ) -> None:
        return None

    def edit_message_reply_markup(
        self,
        *args,
        **kwargs,
    ) -> None:
        return None

    def send_document(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise AssertionError(
            "PDF не должен создаваться."
        )

    def set_my_commands(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.command_calls += 1
        if args:
            self.commands = list(
                args[0]
            )

    def get_me(
        self,
    ):
        self.get_me_calls += 1
        EVENTS.append(
            "get_me"
        )

        if self.raise_get_me:
            raise RuntimeError(
                "неверный Telegram-токен"
            )

        return SimpleNamespace(
            username="offline_bot"
        )

    def infinity_polling(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.polling_calls += 1
        EVENTS.append(
            "polling"
        )

    def clear(self) -> None:
        self.messages.clear()


class FakeScheduler:
    """Проверяет конфигурацию, не создавая фонового потока."""

    instances = []

    def __init__(
        self,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs
        self.jobs = []
        self.started = False
        self.shutdown_calls = []
        self.__class__.instances.append(
            self
        )

    def add_job(
        self,
        function,
        trigger=None,
        **kwargs,
    ) -> None:
        self.jobs.append(
            {
                "function": function,
                "trigger": trigger,
                "kwargs": kwargs,
            }
        )

    def start(self) -> None:
        self.started = True
        EVENTS.append(
            "scheduler_start"
        )

    def shutdown(
        self,
        wait=True,
    ) -> None:
        self.shutdown_calls.append(
            wait
        )
        EVENTS.append(
            "scheduler_shutdown"
        )


def _import_bot(
    fake_bot: FakeBot,
):
    sys.modules.pop(
        "bot",
        None,
    )
    FakeScheduler.instances.clear()

    with (
        patch.object(
            dotenv,
            "load_dotenv",
            return_value=False,
        ),
        patch.object(
            telebot,
            "TeleBot",
            return_value=fake_bot,
        ),
        patch.object(
            aps_background,
            "BackgroundScheduler",
            FakeScheduler,
        ),
        patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": (
                    "offline-test-token"
                ),
                "DAILY_LIMIT": "5",
                "REMINDER_TZ": "UTC",
                "REMINDER_DAYS_BEFORE": "3",
            },
            clear=False,
        ),
    ):
        return importlib.import_module(
            "bot"
        )


def _save_trip(
    tg_id: int,
    destination: str,
    start_date: str,
    end_date: str,
) -> int:
    itinerary_days = []
    current = datetime.fromisoformat(
        start_date
    ).date()
    finish = datetime.fromisoformat(
        end_date
    ).date()

    while current <= finish:
        itinerary_days.append(
            {
                "date": current.isoformat(),
                "theme": "Семейный день",
                "items": [
                    {
                        "time": "10:00",
                        "title": "Музей",
                    }
                ],
                "rainy_alternative": (
                    "Интерактивный центр"
                ),
            }
        )
        current = current.fromordinal(
            current.toordinal() + 1
        )

    trip_id = db.save_trip(
        tg_id,
        {
            "destination": destination,
            "start_date": start_date,
            "end_date": end_date,
            "weather": {
                "source": "forecast",
                "days": [
                    {
                        "date": start_date,
                        "tmin": 8,
                        "tmax": 15,
                        "rain_mm": 0,
                    }
                ],
            },
        },
        {
            "title": destination,
            "days": itinerary_days,
        },
    )

    db.save_checklist(
        trip_id,
        [
            (
                "Документы",
                "Паспорта",
            ),
        ],
    )

    return trip_id


def _set_created_at(
    trip_id: int,
    value: str,
) -> None:
    with db.connect() as conn:
        conn.execute(
            """
            UPDATE trips
            SET created_at = ?
            WHERE id = ?
            """,
            (
                value,
                trip_id,
            ),
        )


def _status(
    trip_id: int,
    kind: str,
):
    return next(
        item
        for item in db.get_reminder_statuses(
            trip_id
        )
        if item["kind"] == kind
    )


def main() -> None:
    original_db_path = db.DB_PATH
    real_db_path = Path(
        original_db_path
    ).resolve()
    real_db_before = _fingerprint(
        real_db_path
    )
    original_tz = os.environ.get(
        "REMINDER_TZ"
    )
    original_days = os.environ.get(
        "REMINDER_DAYS_BEFORE"
    )

    fake_bot = FakeBot()
    bot_module = None

    try:
        _remove_test_db()
        db.DB_PATH = str(
            TEST_DB_PATH
        )

        bot_module = _import_bot(
            fake_bot
        )

        assert bot_module.bot is fake_bot
        assert FakeScheduler.instances == []
        assert not fake_bot.messages
        assert fake_bot.command_calls == 0
        assert fake_bot.get_me_calls == 0
        assert fake_bot.polling_calls == 0

        _ok(
            "импорт не запускает Telegram, polling "
            "или фоновый scheduler"
        )

        scheduler = (
            bot_module.build_reminder_scheduler(
                bot_module.reminders.get_tz()
            )
        )

        assert len(
            FakeScheduler.instances
        ) == 1
        assert scheduler.started is False
        assert len(scheduler.jobs) == 1

        job = scheduler.jobs[0]
        assert job["function"] is (
            bot_module.dispatch_reminders
        )
        assert job["trigger"] == "interval"
        assert job["kwargs"]["minutes"] == 5
        assert job["kwargs"]["max_instances"] == 1
        assert job["kwargs"]["coalesce"] is True
        assert (
            job["kwargs"]["next_run_time"].tzinfo
            is not None
        )

        _ok(
            "scheduler настроен на 5 минут, один "
            "экземпляр и немедленный первый запуск"
        )

        FakeScheduler.instances.clear()
        EVENTS.clear()

        with (
            patch.object(
                dotenv,
                "load_dotenv",
                return_value=False,
            ),
            patch.object(
                telebot,
                "TeleBot",
                return_value=fake_bot,
            ),
            patch.object(
                aps_background,
                "BackgroundScheduler",
                FakeScheduler,
            ),
            patch.dict(
                os.environ,
                {
                    "TELEGRAM_BOT_TOKEN": (
                        "offline-test-token"
                    ),
                    "REMINDER_TZ": "UTC",
                    "REMINDER_DAYS_BEFORE": "3",
                },
                clear=False,
            ),
        ):
            runpy.run_path(
                bot_module.__file__,
                run_name="__main__",
            )

        assert len(
            FakeScheduler.instances
        ) == 1
        main_scheduler = (
            FakeScheduler.instances[0]
        )
        assert main_scheduler.started is True
        assert (
            main_scheduler.shutdown_calls
            == [False]
        )
        assert fake_bot.command_calls == 1
        assert fake_bot.get_me_calls == 1
        assert fake_bot.polling_calls == 1
        assert EVENTS == [
            "get_me",
            "scheduler_start",
            "polling",
            "scheduler_shutdown",
        ]
        assert "/reminders 3" in (
            bot_module.HELP_TEXT
        )
        assert any(
            command.command == "reminders"
            for command in fake_bot.commands
        )

        _ok(
            "обычный запуск стартует scheduler, а "
            "после polling останавливает его; "
            "команда добавлена в помощь и меню"
        )

        invalid_bot = FakeBot()
        invalid_bot.raise_get_me = True
        FakeScheduler.instances.clear()
        EVENTS.clear()

        with (
            patch.object(
                dotenv,
                "load_dotenv",
                return_value=False,
            ),
            patch.object(
                telebot,
                "TeleBot",
                return_value=invalid_bot,
            ),
            patch.object(
                aps_background,
                "BackgroundScheduler",
                FakeScheduler,
            ),
            patch.dict(
                os.environ,
                {
                    "TELEGRAM_BOT_TOKEN": (
                        "invalid-test-token"
                    ),
                    "REMINDER_TZ": "UTC",
                    "REMINDER_DAYS_BEFORE": "3",
                },
                clear=False,
            ),
        ):
            try:
                runpy.run_path(
                    bot_module.__file__,
                    run_name="__main__",
                )
            except RuntimeError as error:
                assert "Telegram-токен" in str(
                    error
                )
            else:
                raise AssertionError(
                    "Неверный токен должен "
                    "останавливать запуск."
                )

        assert EVENTS == [
            "get_me"
        ]
        assert FakeScheduler.instances == []
        assert invalid_bot.command_calls == 0
        assert invalid_bot.polling_calls == 0

        _ok(
            "неверный Telegram-токен не запускает "
            "scheduler и отправку напоминаний"
        )

        owner_trip = _save_trip(
            OWNER_ID,
            "Прага",
            "2030-05-10",
            "2030-05-11",
        )

        fake_bot.clear()
        bot_module.handle_reminders(
            _message(
                f"/reminders {owner_trip}",
                OWNER_ID,
            )
        )

        assert (
            f"№{owner_trip}"
            in fake_bot.messages[-1]["text"]
        )
        assert (
            "Напоминания по поездке"
            in fake_bot.messages[-1]["text"]
        )

        fake_bot.clear()
        bot_module.handle_reminders(
            _message(
                f"/reminders {owner_trip}",
                OTHER_ID,
            )
        )

        assert (
            "Такой поездки у вас нет"
            in fake_bot.messages[-1]["text"]
        )

        fake_bot.clear()
        bot_module.handle_reminders(
            _message(
                f"/reminders {owner_trip}",
                OWNER_ID,
                chat_type="group",
            )
        )

        assert (
            "только в личном чате"
            in fake_bot.messages[-1]["text"]
        )

        _ok(
            "/reminders показывает только свою "
            "поездку и защищён для личного чата"
        )

        fail_trip = _save_trip(
            FAIL_ID,
            "Рим",
            "2026-11-10",
            "2026-11-10",
        )
        success_trip = _save_trip(
            OTHER_ID,
            "Париж",
            "2026-11-10",
            "2026-11-10",
        )

        for trip_id in (
            fail_trip,
            success_trip,
        ):
            _set_created_at(
                trip_id,
                "2026-11-01T00:00:00+00:00",
            )

        fake_bot.clear()
        fake_bot.fail_once.add(
            FAIL_ID
        )
        now = datetime(
            2026,
            11,
            7,
            10,
            tzinfo=timezone.utc,
        )

        with (
            patch.object(
                httpx,
                "get",
                side_effect=AssertionError(
                    "диспетчер не должен "
                    "обращаться к сети"
                ),
            ),
            patch.object(
                bot_module.logger,
                "exception",
            ),
        ):
            bot_module.dispatch_reminders(
                now
            )

        failed = _status(
            fail_trip,
            "before",
        )
        sent = _status(
            success_trip,
            "before",
        )

        assert failed["status"] == "failed"
        assert failed["attempts"] == 1
        assert sent["status"] == "sent"
        assert sent["attempts"] == 1
        assert [
            item["chat_id"]
            for item in fake_bot.messages
        ] == [
            OTHER_ID
        ]

        _ok(
            "ошибка одной отправки записывается и "
            "не останавливает остальные поездки"
        )

        with patch.object(
            httpx,
            "get",
            side_effect=AssertionError(
                "диспетчер не должен "
                "обращаться к сети"
            ),
        ):
            bot_module.dispatch_reminders(
                now
            )

        retried = _status(
            fail_trip,
            "before",
        )

        assert retried["status"] == "sent"
        assert retried["attempts"] == 2
        assert [
            item["chat_id"]
            for item in fake_bot.messages
        ] == [
            OTHER_ID,
            FAIL_ID,
        ]

        bot_module.dispatch_reminders(
            now
        )

        assert len(
            fake_bot.messages
        ) == 2

        _ok(
            "повтор после ошибки работает, а успешно "
            "отправленные сообщения не дублируются"
        )

        fresh_trip = _save_trip(
            OWNER_ID + 200,
            "Вена",
            "2026-11-08",
            "2026-11-08",
        )
        fallback_trip = _save_trip(
            OWNER_ID + 201,
            "Лондон",
            "2026-11-08",
            "2026-11-08",
        )

        for trip_id in (
            fresh_trip,
            fallback_trip,
        ):
            _set_created_at(
                trip_id,
                "2026-11-01T00:00:00+00:00",
            )

        def fake_get_weather(
            destination: str,
            start_date: str,
            end_date: str,
        ):
            assert start_date == "2026-11-08"
            assert end_date == "2026-11-08"

            if destination == "Вена":
                return {
                    "city": "Вена",
                    "source": "forecast",
                    "days": [
                        {
                            "date": "2026-11-08",
                            "tmin": 1,
                            "tmax": 2,
                            "rain_mm": 3,
                        }
                    ],
                }

            if destination == "Лондон":
                return None

            raise AssertionError(
                "Неожиданная поездка в проверке погоды."
            )

        fake_bot.clear()
        eve_now = datetime(
            2026,
            11,
            7,
            18,
            tzinfo=timezone.utc,
        )

        with patch.object(
            bot_module,
            "get_weather",
            side_effect=fake_get_weather,
        ) as weather_mock:
            bot_module.dispatch_reminders(
                eve_now
            )

        assert _status(
            fresh_trip,
            "eve",
        )["status"] == "sent"
        assert _status(
            fallback_trip,
            "eve",
        )["status"] == "sent"
        assert weather_mock.call_count == 2
        weather_mock.assert_any_call(
            "Вена",
            "2026-11-08",
            "2026-11-08",
        )
        weather_mock.assert_any_call(
            "Лондон",
            "2026-11-08",
            "2026-11-08",
        )

        eve_messages = {
            item["chat_id"]: item["text"]
            for item in fake_bot.messages
        }
        fresh_text = eve_messages[
            OWNER_ID + 200
        ]
        fallback_text = eve_messages[
            OWNER_ID + 201
        ]

        assert "от 1 до 2 °C" in fresh_text
        assert (
            "Свежий прогноз сейчас недоступен."
            not in fresh_text
        )
        assert (
            "Свежий прогноз сейчас недоступен."
            in fallback_text
        )
        assert "от 8 до 15 °C" not in fallback_text
        assert "Погода в первый день" not in fallback_text

        _ok(
            "накануне используется свежий прогноз, "
            "а при сбое старая погода не показывается"
        )

        # Проверяем календарный день пользователя:
        # 18:30 UTC — уже следующий день в Новосибирске.
        timezone_trip = _save_trip(
            OWNER_ID + 100,
            "Токио",
            "2030-06-01",
            "2030-06-01",
        )
        _set_created_at(
            timezone_trip,
            "2026-10-07T18:30:00+00:00",
        )

        class FixedDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                value = datetime(
                    2026,
                    10,
                    7,
                    18,
                    30,
                    tzinfo=timezone.utc,
                )
                return (
                    value.astimezone(tz)
                    if tz is not None
                    else value.replace(
                        tzinfo=None
                    )
                )

        with (
            patch.dict(
                os.environ,
                {
                    "REMINDER_TZ": (
                        "Asia/Novosibirsk"
                    )
                },
                clear=False,
            ),
            patch.object(
                bot_module,
                "datetime",
                FixedDateTime,
            ),
        ):
            assert (
                bot_module.used_today(
                    OWNER_ID + 100
                )
                == 1
            )

        _ok(
            "дневной лимит считается по REMINDER_TZ, "
            "а не по часовому поясу сервера"
        )

        print(
            "\nИтог: Telegram-интеграция "
            "напоминаний полностью работает."
        )

    finally:
        db.DB_PATH = original_db_path
        _remove_test_db()

        if original_tz is None:
            os.environ.pop(
                "REMINDER_TZ",
                None,
            )
        else:
            os.environ[
                "REMINDER_TZ"
            ] = original_tz

        if original_days is None:
            os.environ.pop(
                "REMINDER_DAYS_BEFORE",
                None,
            )
        else:
            os.environ[
                "REMINDER_DAYS_BEFORE"
            ] = original_days

        sys.modules.pop(
            "bot",
            None,
        )

    assert _fingerprint(
        real_db_path
    ) == real_db_before
    assert not TEST_DB_PATH.exists()

    print(
        "Временная база "
        "test_bot_reminders.db удалена."
    )
    print(
        "Настоящий planner.db "
        "не изменился."
    )


if __name__ == "__main__":
    main()