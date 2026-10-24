"""Автономный smoke-тест интеграции bot.py с памятью.

Тест не запускает Telegram polling и не обращается
к Telegram API, YandexGPT, Pinecone, погоде или картам.
Настоящий planner.db не изменяется.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import db
import dotenv
import telebot

from schemas import Itinerary, TripRequest


PROJECT_DIR = Path(__file__).resolve().parent
REAL_DB_PATH = PROJECT_DIR / "planner.db"
TEST_DB_PATH = (
    PROJECT_DIR / "test_bot_memory.db"
)

CHAT_ID = 71001
USER_ID = 71001


def remove_test_db() -> None:
    """Удаляет временную БД и файлы SQLite."""

    for path in (
        TEST_DB_PATH,
        Path(f"{TEST_DB_PATH}-wal"),
        Path(f"{TEST_DB_PATH}-shm"),
        Path(f"{TEST_DB_PATH}-journal"),
    ):
        path.unlink(
            missing_ok=True
        )


def fingerprint(
    path: Path,
) -> tuple[int, str] | None:
    """Фиксирует размер и хеш настоящей БД."""

    if not path.exists():
        return None

    content = path.read_bytes()

    return (
        len(content),
        hashlib.sha256(
            content
        ).hexdigest(),
    )


def ok(
    message: str,
) -> None:
    print(
        f"[OK] {message}"
    )


class FakeBot:
    """Локальная замена Telegram-клиента."""

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.actions: list[tuple] = []
        self.answers: list[tuple] = []
        self.edits: list[tuple] = []
        self.handlers: list[tuple] = []

    def message_handler(
        self,
        *args,
        **kwargs,
    ):
        def register(
            function,
        ):
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
        def register(
            function,
        ):
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
    ) -> SimpleNamespace:
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
        chat_id: int,
        action: str,
        *args,
        **kwargs,
    ) -> None:
        self.actions.append(
            (
                chat_id,
                action,
            )
        )

    def answer_callback_query(
        self,
        callback_id: str,
        text: str = "",
        *args,
        **kwargs,
    ) -> None:
        self.answers.append(
            (
                callback_id,
                text,
            )
        )

    def edit_message_text(
        self,
        text: str,
        chat_id: int,
        message_id: int,
        *args,
        **kwargs,
    ) -> None:
        self.edits.append(
            (
                text,
                chat_id,
                message_id,
            )
        )

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
            "PDF не должен создаваться "
            "в этом тесте."
        )

    def set_my_commands(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise AssertionError(
            "setup_commands не должен "
            "запускаться при импорте."
        )

    def get_me(
        self,
    ) -> None:
        raise AssertionError(
            "Telegram API не должен "
            "вызываться."
        )

    def infinity_polling(
        self,
        *args,
        **kwargs,
    ) -> None:
        raise AssertionError(
            "Polling не должен запускаться."
        )

    def clear(
        self,
    ) -> None:
        self.messages.clear()
        self.actions.clear()
        self.answers.clear()
        self.edits.clear()


class FakeThread:
    """Запоминает задачу, но не запускает поток."""

    created: list["FakeThread"] = []

    def __init__(
        self,
        target=None,
        args=(),
        kwargs=None,
        daemon=None,
        name=None,
        **extra,
    ) -> None:
        self.target = target
        self.args = tuple(
            args
        )
        self.kwargs = dict(
            kwargs or {}
        )
        self.daemon = daemon
        self.name = name
        self.started = False

        self.__class__.created.append(
            self
        )

    def start(
        self,
    ) -> None:
        self.started = True


def message(
    text: str,
    chat_id: int = CHAT_ID,
    user_id: int = USER_ID,
    chat_type: str = "private",
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        content_type="text",
        chat=SimpleNamespace(
            id=chat_id,
            type=chat_type,
        ),
        from_user=SimpleNamespace(
            id=user_id,
            first_name="Анна",
        ),
    )


def callback(
    action: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        id="offline-callback",
        data=(
            f"memory_clear:{action}"
        ),
        from_user=SimpleNamespace(
            id=USER_ID,
        ),
        message=SimpleNamespace(
            message_id=501,
            chat=SimpleNamespace(
                id=CHAT_ID,
                type="private",
            ),
        ),
    )


def success_result(
    count: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        success=True,
        requested=count,
        vector_saved=count,
        journal_saved=count,
        error_type="",
    )


def test_itinerary() -> Itinerary:
    return Itinerary.model_validate(
        {
            "title": "Офлайн-маршрут",
            "summary": (
                "Проверка интеграции."
            ),
            "days": [
                {
                    "day": 1,
                    "date": (
                        "2026-11-10"
                    ),
                    "theme": (
                        "Спокойный день"
                    ),
                    "items": [],
                    "rainy_alternative": (
                        "Музей"
                    ),
                }
            ],
            "tips": [
                "Проверить расписание."
            ],
        }
    )


def output_text(
    fake_bot: FakeBot,
) -> str:
    return "\n".join(
        item["text"]
        for item in fake_bot.messages
    )


def import_bot_safely(
    fake_bot: FakeBot,
):
    """Импортирует bot без .env и Telegram."""

    sys.modules.pop(
        "bot",
        None,
    )

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
        patch.dict(
            os.environ,
            {
                "TELEGRAM_BOT_TOKEN": (
                    "offline-test-token"
                ),
                "DAILY_LIMIT": "5",
            },
            clear=False,
        ),
    ):
        return importlib.import_module(
            "bot"
        )


def main() -> None:
    original_db_path = db.DB_PATH
    real_db_before = fingerprint(
        REAL_DB_PATH
    )

    fake_bot = FakeBot()
    bot_module = None

    try:
        remove_test_db()
        db.DB_PATH = str(
            TEST_DB_PATH
        )

        bot_module = (
            import_bot_safely(
                fake_bot
            )
        )

        assert (
            bot_module.bot
            is fake_bot
        )

        assert TEST_DB_PATH.exists()
        assert not fake_bot.messages
        assert not fake_bot.actions

        ok(
            "безопасный импорт использует "
            "FakeBot и временную БД"
        )

        request_data = {
            "destination": (
                "Париж, Франция"
            ),
            "start_date": (
                "2026-11-10"
            ),
            "end_date": (
                "2026-11-10"
            ),
            "adults": 2,
            "children_ages": [5],
            "budget": 300,
            "currency": "EUR",
            "pace": "спокойный",
            "wishes": (
                "дневной отдых обязателен"
            ),
        }

        request = TripRequest(
            **request_data
        )

        FakeThread.created.clear()

        with patch.object(
            bot_module.threading,
            "Thread",
            FakeThread,
        ), patch.object(bot_module, "resolve_city_for_planning", return_value=None):
            bot_module.start_generation(
                message("/plan"),
                dict(request_data),
            )

        assert len(
            FakeThread.created
        ) == 1

        route_thread = (
            FakeThread.created[0]
        )

        assert (
            route_thread.target
            is bot_module.generation_job
        )

        assert route_thread.started

        assert route_thread.args[:2] == (
            CHAT_ID,
            USER_ID,
        )

        assert isinstance(
            route_thread.args[2],
            TripRequest,
        )

        profile = db.get_user(
            CHAT_ID
        )

        assert profile is not None
        assert profile["pace"] == (
            "спокойный"
        )
        assert profile["wishes"] == (
            "дневной отдых обязателен"
        )

        bot_module.generating.discard(
            CHAT_ID
        )

        ok(
            "start_generation сохраняет "
            "профиль и передаёт оба ID"
        )

        calls: dict = {}

        matches = [
            SimpleNamespace(
                memory_id=(
                    "family-pref#offline"
                ),
                text=(
                    "Семья любит музеи"
                ),
                kind="custom",
                score=0.91,
                updated_at="",
                destination="",
            )
        ]

        def fake_recall(
            chat_id,
            user_id,
            query,
            *,
            top_k,
            min_score,
            destination,
        ):
            calls["recall"] = (
                chat_id,
                user_id,
                query,
                top_k,
                min_score,
                destination,
            )

            return matches

        def fake_generate(
            received_request,
            *,
            memory_context,
            ground_candidates,
        ):
            assert ground_candidates is False
            calls["generate"] = (
                received_request,
                memory_context,
            )

            return SimpleNamespace(
                itinerary=(
                    test_itinerary()
                ),
                problems=[],
                weather_text=(
                    "Осадков нет."
                ),
                weather={
                    "source": "forecast",
                    "days": [],
                },
                attempts=1,
                prompt_tokens=10,
                completion_tokens=20,
            )

        fake_bot.clear()

        bot_module.generating.add(
            CHAT_ID
        )

        with (
            patch.object(
                bot_module.vector_memory,
                "recall_family_memories",
                side_effect=fake_recall,
            ),
            patch.object(
                bot_module.vector_memory,
                "format_memory_context",
                return_value=(
                    "КОНТЕКСТ ПАМЯТИ"
                ),
            ) as format_mock,
            patch.object(
                bot_module,
                "generate_itinerary",
                side_effect=fake_generate,
            ),
            patch.object(
                bot_module,
                "build_checklist",
                return_value=[
                    (
                        "Документы",
                        "Проверить паспорта",
                    )
                ],
            ),
            patch.object(
                bot_module,
                "format_itinerary",
                return_value=(
                    "ГОТОВЫЙ МАРШРУТ"
                ),
            ),
            patch.object(
                bot_module.vector_memory,
                "save_trip_memory",
                return_value=(
                    success_result(4)
                ),
            ) as save_trip_mock,
        ):
            bot_module.generation_job(
                CHAT_ID,
                USER_ID,
                request,
            )

        recall_call = calls[
            "recall"
        ]

        assert recall_call[:2] == (
            CHAT_ID,
            USER_ID,
        )

        assert (
            "Париж, Франция"
            in recall_call[2]
        )

        assert recall_call[3:] == (
            8,
            0.35,
            request.destination,
        )

        format_mock.assert_called_once_with(
            matches
        )

        assert calls["generate"] == (
            request,
            "КОНТЕКСТ ПАМЯТИ",
        )

        trips = db.list_trips(
            CHAT_ID
        )

        assert len(trips) == 1

        trip_id = int(
            trips[0]["id"]
        )

        checklist = db.get_checklist(
            trip_id
        )

        assert checklist[0]["text"] == (
            "Проверить паспорта"
        )

        save_trip_mock.assert_called_once_with(
            CHAT_ID,
            USER_ID,
            CHAT_ID,
            request,
            trip_id=trip_id,
        )

        assert (
            CHAT_ID
            not in bot_module.generating
        )

        assert (
            "ГОТОВЫЙ МАРШРУТ"
            in output_text(
                fake_bot
            )
        )

        assert (
            "/memory"
            in output_text(
                fake_bot
            )
        )

        ok(
            "generation_job выполняет "
            "recall → planner → БД → memory"
        )

        db.save_memory_item(
            CHAT_ID,
            "family-pref#manual",
            "custom",
            (
                "Пользователь попросил "
                "запомнить: нужен "
                "дневной отдых"
            ),
        )

        fake_bot.clear()

        bot_module.handle_memory(
            message("/memory")
        )

        assert (
            "Личная заметка"
            in output_text(
                fake_bot
            )
        )

        assert (
            "дневной отдых"
            in output_text(
                fake_bot
            )
        )

        ok(
            "/memory показывает журнал "
            "SQLite без Pinecone"
        )

        fake_bot.clear()

        bot_module.memory_jobs.add(
            CHAT_ID
        )

        with patch.object(
            bot_module.vector_memory,
            "save_custom_memory",
            return_value=(
                success_result()
            ),
        ) as custom_mock:
            bot_module._save_custom_memory_job(
                CHAT_ID,
                USER_ID,
                "Нужен дневной отдых",
            )

        custom_mock.assert_called_once_with(
            CHAT_ID,
            USER_ID,
            CHAT_ID,
            "Нужен дневной отдых",
        )

        assert (
            CHAT_ID
            not in bot_module.memory_jobs
        )

        assert (
            "Запомнила"
            in output_text(
                fake_bot
            )
        )

        fake_bot.clear()

        bot_module.memory_jobs.add(
            CHAT_ID
        )

        with patch.object(
            bot_module.vector_memory,
            "save_feedback_memory",
            return_value=(
                success_result()
            ),
        ) as feedback_mock:
            bot_module._save_feedback_memory_job(
                CHAT_ID,
                USER_ID,
                trip_id,
                "Париж, Франция",
                "Понравились музеи",
            )

        feedback_mock.assert_called_once_with(
            CHAT_ID,
            USER_ID,
            CHAT_ID,
            trip_id=trip_id,
            destination=(
                "Париж, Франция"
            ),
            text=(
                "Понравились музеи"
            ),
        )

        assert (
            CHAT_ID
            not in bot_module.memory_jobs
        )

        assert (
            "Отзыв сохранён"
            in output_text(
                fake_bot
            )
        )

        ok(
            "ручная память и отзыв "
            "используют bridge API"
        )

        fake_bot.clear()

        bot_module.generating.discard(
            CHAT_ID
        )

        bot_module.handle_forget_memory(
            message(
                "/forget_memory"
            )
        )

        markup = fake_bot.messages[-1][
            "kwargs"
        ]["reply_markup"]

        callback_values = {
            button.callback_data
            for row in markup.keyboard
            for button in row
        }

        assert callback_values == {
            "memory_clear:confirm",
            "memory_clear:cancel",
        }

        fake_bot.clear()
        FakeThread.created.clear()
        bot_module.memory_jobs.clear()

        with patch.object(
            bot_module.threading,
            "Thread",
            FakeThread,
        ):
            bot_module.handle_memory_clear(
                callback(
                    "confirm"
                )
            )

        assert len(
            FakeThread.created
        ) == 1

        clear_thread = (
            FakeThread.created[0]
        )

        assert (
            clear_thread.target
            is bot_module._clear_memory_job
        )

        assert clear_thread.args == (
            CHAT_ID,
            USER_ID,
        )

        bot_module._finish_memory_job(
            CHAT_ID
        )

        fake_bot.clear()

        bot_module.memory_jobs.add(
            CHAT_ID
        )

        with patch.object(
            bot_module.vector_memory,
            "clear_saved_memories",
            return_value=True,
        ) as clear_mock:
            bot_module._clear_memory_job(
                CHAT_ID,
                USER_ID,
            )

        clear_mock.assert_called_once_with(
            CHAT_ID,
            USER_ID,
            CHAT_ID,
            confirm=True,
        )

        assert (
            CHAT_ID
            not in bot_module.memory_jobs
        )

        assert (
            "Память полностью очищена"
            in output_text(
                fake_bot
            )
        )

        assert len(
            db.list_trips(
                CHAT_ID
            )
        ) == 1

        ok(
            "очистка требует подтверждения "
            "и не удаляет поездки"
        )

        fake_bot.clear()

        bot_module.handle_non_private(
            message(
                "/memory",
                chat_id=-100777,
                user_id=777,
                chat_type="group",
            )
        )

        assert (
            "только в личном чате"
            in output_text(
                fake_bot
            )
        )

        ok(
            "память недоступна в общем "
            "групповом журнале"
        )

        print(
            "\nИтог: интеграция Telegram-бота "
            "с памятью работает автономно."
        )

    finally:
        if bot_module is not None:
            bot_module.generating.clear()
            bot_module.memory_jobs.clear()

        db.DB_PATH = (
            original_db_path
        )

        remove_test_db()

    assert (
        fingerprint(
            REAL_DB_PATH
        )
        == real_db_before
    ), (
        "Настоящий planner.db изменился."
    )

    assert not TEST_DB_PATH.exists()

    print(
        "Временная база "
        "test_bot_memory.db удалена."
    )

    print(
        "Настоящий planner.db "
        "не изменился."
    )


if __name__ == "__main__":
    main()
