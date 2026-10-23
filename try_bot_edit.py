"""Автономная проверка Telegram-команды /edit.

Тест не запускает polling, YandexGPT, карту, погоду или Telegram API.
Используется временная SQLite-база; planner.db не изменяется.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import db
import dotenv
import telebot
from planner import PlanResult
from schemas import Itinerary


SCRIPT_DIR = Path(__file__).resolve().parent
TEST_DB_PATH = SCRIPT_DIR / "test_bot_edit.db"

OWNER_ID = 91001
OTHER_ID = 91002


def _ok(text: str) -> None:
    print(f"[OK] {text}")


def _fingerprint(path: Path):
    if not path.exists():
        return None

    content = path.read_bytes()
    return (
        len(content),
        hashlib.sha256(content).hexdigest(),
    )


def _remove_test_db() -> None:
    for suffix in (
        "",
        "-wal",
        "-shm",
        "-journal",
    ):
        Path(
            f"{TEST_DB_PATH}{suffix}"
        ).unlink(missing_ok=True)


def _message(
    text: str,
    chat_id: int,
):
    return SimpleNamespace(
        text=text,
        content_type="text",
        chat=SimpleNamespace(
            id=chat_id,
            type="private",
        ),
        from_user=SimpleNamespace(
            id=chat_id,
            first_name="Тест",
        ),
    )


class FakeBot:
    def __init__(self) -> None:
        self.messages = []
        self.handlers = []
        self.commands = []

    def message_handler(
        self,
        *args,
        **kwargs,
    ):
        def register(function):
            self.handlers.append(function)
            return function

        return register

    def callback_query_handler(
        self,
        *args,
        **kwargs,
    ):
        def register(function):
            self.handlers.append(function)
            return function

        return register

    def send_message(
        self,
        chat_id: int,
        text: str,
        *args,
        **kwargs,
    ):
        self.messages.append(
            {
                "chat_id": chat_id,
                "text": str(text),
                "kwargs": kwargs,
            }
        )
        return SimpleNamespace(
            message_id=len(self.messages)
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
        return None

    def set_my_commands(
        self,
        commands,
    ) -> None:
        self.commands = list(commands)

    def clear(self) -> None:
        self.messages.clear()


def _import_bot(fake_bot: FakeBot):
    sys.modules.pop("bot", None)

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
                    "offline-edit-test-token"
                ),
                "REMINDER_TZ": "UTC",
                "REMINDER_DAYS_BEFORE": "3",
            },
            clear=False,
        ),
    ):
        return importlib.import_module("bot")


def _attraction(
    title: str,
    query: str,
) -> dict:
    return {
        "time": "09:00",
        "kind": "attraction",
        "title": title,
        "area": "Центр",
        "place_query": query,
        "description": "Семейное посещение",
        "duration_min": 90,
        "cost": 20,
        "kids_note": "Есть детская зона",
        "indoor": True,
    }


def _day(
    number: int,
    date_value: str,
    attraction: dict,
) -> dict:
    return {
        "day": number,
        "date": date_value,
        "theme": f"День {number}",
        "items": [
            attraction,
            {
                "time": "12:30",
                "kind": "meal",
                "title": "Обед",
                "area": "Центр",
                "place_query": "",
                "description": "Обед с детьми",
                "duration_min": 60,
                "cost": 40,
                "kids_note": "",
                "indoor": True,
            },
            {
                "time": "13:30",
                "kind": "rest",
                "title": "Отдых",
                "area": "Центр",
                "place_query": "",
                "description": "Дневной отдых",
                "duration_min": 60,
                "cost": 0,
                "kids_note": "",
                "indoor": True,
            },
        ],
        "rainy_alternative": "Музей Карнавале",
    }


def _route() -> dict:
    return {
        "title": "Семейный Париж",
        "summary": "Спокойная поездка с ребёнком.",
        "days": [
            _day(
                1,
                "2026-11-10",
                _attraction(
                    "Лувр",
                    "Musée du Louvre, Paris, France",
                ),
            ),
            _day(
                2,
                "2026-11-11",
                _attraction(
                    "Музей Орсе",
                    "Musée d'Orsay, Paris, France",
                ),
            ),
        ],
        "tips": ["Покупайте билеты заранее"],
    }


def _edited_route() -> dict:
    data = copy.deepcopy(_route())
    data["days"][0]["theme"] = (
        "Интерактивный день"
    )
    data["days"][0]["items"][0] = (
        _attraction(
            "Городок науки и индустрии",
            "Cité des sciences, Paris, France",
        )
    )
    return data


def _request() -> dict:
    return {
        "destination": "Париж, Франция",
        "start_date": "2026-11-10",
        "end_date": "2026-11-11",
        "adults": 2,
        "children_ages": [5],
        "budget": 600.0,
        "currency": "EUR",
        "pace": "спокойный",
        "wishes": "интерактивные музеи",
        "weather": None,
        "weather_text": "Старый прогноз",
    }


def _save_trip() -> int:
    trip_id = db.save_trip(
        OWNER_ID,
        _request(),
        _route(),
    )
    db.save_checklist(
        trip_id,
        [("Документы", "Паспорта")],
    )
    reminder_time = datetime(
        2026,
        11,
        7,
        10,
        tzinfo=timezone.utc,
    )
    assert db.claim_reminder(
        trip_id,
        "before",
        reminder_time,
    ) is True
    assert db.mark_reminder_sent(
        trip_id,
        "before",
        reminder_time,
    ) is True
    return trip_id


def main() -> None:
    original_db_path = db.DB_PATH
    real_db_path = Path(original_db_path).resolve()
    real_db_before = _fingerprint(real_db_path)
    fake_bot = FakeBot()

    try:
        _remove_test_db()
        db.DB_PATH = str(TEST_DB_PATH)
        bot_module = _import_bot(fake_bot)

        assert bot_module.bot is fake_bot
        assert not fake_bot.messages
        assert "/edit 3 1 текст" in bot_module.HELP_TEXT
        bot_module.setup_commands()
        assert any(
            command.command == "edit"
            for command in fake_bot.commands
        )

        _ok(
            "безопасный импорт не запускает Telegram "
            "или сеть; /edit добавлена в помощь и меню"
        )

        trip_id = _save_trip()

        fake_bot.clear()
        bot_module.handle_edit(
            _message(
                f"/edit {trip_id} 1 Добавь парк",
                OTHER_ID,
            )
        )
        assert "Такой поездки у вас нет" in (
            fake_bot.messages[-1]["text"]
        )

        fake_bot.clear()
        bot_module.handle_edit(
            _message(
                f"/edit {trip_id} 7 Добавь парк",
                OWNER_ID,
            )
        )
        assert "нет такого дня" in (
            fake_bot.messages[-1]["text"]
        )

        fake_bot.clear()
        bot_module.handle_edit(
            _message(
                f"/edit {trip_id} 1 "
                + "я" * 601,
                OWNER_ID,
            )
        )
        assert "не длиннее 600" in (
            fake_bot.messages[-1]["text"]
        )

        _ok(
            "чужая поездка, неверный день и слишком "
            "длинное пожелание отклоняются до AI"
        )

        assert bot_module._reserve_generation_job(
            OWNER_ID
        ) is True
        fake_bot.clear()
        bot_module.handle_edit(
            _message(
                f"/edit {trip_id} 1 Добавь парк",
                OWNER_ID,
            )
        )
        assert "уже выполняется" in (
            fake_bot.messages[-1]["text"]
        )
        bot_module._finish_generation_job(
            OWNER_ID
        )

        _ok(
            "параллельная генерация или правка блокируется"
        )

        successful_result = PlanResult(
            itinerary=Itinerary.model_validate(
                _edited_route()
            ),
            problems=[],
            weather_text="Свежий прогноз",
            attempts=1,
            prompt_tokens=100,
            completion_tokens=200,
            weather={
                "source": "forecast",
                "days": [],
            },
        )
        captured = {}

        def fake_edit(
            request,
            current_itinerary,
            target_day,
            edit_request,
        ):
            captured["target_day"] = target_day
            captured["edit_request"] = edit_request
            captured["old"] = (
                current_itinerary.model_dump()
            )
            return successful_result

        assert bot_module._reserve_generation_job(
            OWNER_ID
        ) is True
        fake_bot.clear()

        with (
            patch.object(
                bot_module,
                "edit_itinerary",
                side_effect=fake_edit,
            ),
            patch.object(
                bot_module,
                "format_itinerary",
                return_value="ОБНОВЛЁННЫЙ МАРШРУТ",
            ),
        ):
            bot_module.edit_generation_job(
                OWNER_ID,
                trip_id,
                1,
                "Добавь интерактивный музей",
            )

        updated = db.get_trip(
            trip_id,
            OWNER_ID,
        )
        assert updated is not None
        assert updated["id"] == trip_id
        assert updated["itinerary"] == (
            successful_result.itinerary.model_dump(
                mode="json"
            )
        )
        assert updated["request"]["weather_text"] == (
            "Свежий прогноз"
        )
        assert captured["target_day"] == 1
        assert captured["edit_request"] == (
            "Добавь интерактивный музей"
        )
        assert len(db.get_checklist(trip_id)) == 1
        statuses = db.get_reminder_statuses(
            trip_id,
            OWNER_ID,
        )
        assert len(statuses) == 1
        assert statuses[0]["status"] == "sent"
        assert not bot_module._is_generation_active(
            OWNER_ID
        )
        assert any(
            "обновлён" in item["text"]
            for item in fake_bot.messages
        )

        _ok(
            "успешная правка обновляет ту же поездку, "
            "сохраняя чек-лист и напоминания"
        )

        before_failed = db.get_trip(
            trip_id,
            OWNER_ID,
        )
        problem_result = PlanResult(
            itinerary=Itinerary.model_validate(
                _route()
            ),
            problems=[
                "Добавленное место не найдено на карте"
            ],
        )
        assert bot_module._reserve_generation_job(
            OWNER_ID
        ) is True

        with patch.object(
            bot_module,
            "edit_itinerary",
            return_value=problem_result,
        ):
            bot_module.edit_generation_job(
                OWNER_ID,
                trip_id,
                1,
                "Добавь неизвестное место",
            )

        assert db.get_trip(
            trip_id,
            OWNER_ID,
        ) == before_failed

        assert bot_module._reserve_generation_job(
            OWNER_ID
        ) is True

        with (
            patch.object(
                bot_module,
                "edit_itinerary",
                side_effect=RuntimeError(
                    "тестовый сбой AI"
                ),
            ),
            patch.object(
                bot_module.logger,
                "exception",
            ),
        ):
            bot_module.edit_generation_job(
                OWNER_ID,
                trip_id,
                1,
                "Добавь парк",
            )

        assert db.get_trip(
            trip_id,
            OWNER_ID,
        ) == before_failed
        assert not bot_module._is_generation_active(
            OWNER_ID
        )

        _ok(
            "замечания проверки и исключение AI не "
            "перезаписывают старый маршрут"
        )

        print(
            "\nИтог: Telegram-команда /edit "
            "безопасно изменяет выбранный день."
        )

    finally:
        db.DB_PATH = original_db_path
        _remove_test_db()
        sys.modules.pop("bot", None)

    assert _fingerprint(real_db_path) == real_db_before
    assert not TEST_DB_PATH.exists()
    print(
        "Временная база test_bot_edit.db удалена."
    )
    print(
        "Настоящий planner.db не изменился."
    )


if __name__ == "__main__":
    main()
