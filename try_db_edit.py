"""Автономная проверка безопасного обновления поездки."""

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import db


TEST_DB = Path(__file__).with_name(
    "test_db_edit.db"
)


def _fingerprint(path: Path):
    if not path.exists():
        return None

    content = path.read_bytes()
    return (
        len(content),
        hashlib.sha256(content).hexdigest(),
    )


def _ok(text: str) -> None:
    print(f"[OK] {text}")


def main() -> None:
    original_path = db.DB_PATH
    real_db_path = Path(original_path).resolve()
    real_db_before = _fingerprint(real_db_path)

    try:
        TEST_DB.unlink(missing_ok=True)
        db.DB_PATH = str(TEST_DB)
        db.init_db()

        request = {
            "destination": "Париж, Франция",
            "start_date": "2026-11-10",
            "end_date": "2026-11-12",
            "adults": 2,
            "children_ages": [5],
            "budget": 600,
            "currency": "EUR",
            "pace": "спокойный",
            "wishes": "музеи",
            "weather": None,
            "weather_text": "Старый прогноз",
        }
        old_plan = {
            "title": "Старый маршрут",
            "days": [],
        }
        trip_id = db.save_trip(
            101,
            request,
            old_plan,
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

        new_request = dict(request)
        new_request["weather"] = {
            "source": "forecast"
        }
        new_request["weather_text"] = (
            "Свежий прогноз"
        )
        new_plan = {
            "title": "Обновлённый маршрут",
            "days": [{"day": 1}],
        }

        assert db.update_trip_plan(
            trip_id,
            101,
            new_request,
            new_plan,
        ) is True

        updated = db.get_trip(
            trip_id,
            101,
        )
        assert updated is not None
        assert updated["request"]["weather_text"] == (
            "Свежий прогноз"
        )
        assert updated["itinerary"] == new_plan
        assert len(db.get_checklist(trip_id)) == 1
        statuses = db.get_reminder_statuses(
            trip_id,
            101,
        )
        assert len(statuses) == 1
        assert statuses[0]["status"] == "sent"

        _ok(
            "владелец обновляет маршрут и свежую "
            "погоду без удаления чек-листа и напоминаний"
        )

        before_foreign = db.get_trip(
            trip_id,
            101,
        )
        assert db.update_trip_plan(
            trip_id,
            202,
            new_request,
            {"title": "Чужая правка"},
        ) is False
        assert db.get_trip(
            trip_id,
            101,
        ) == before_foreign

        _ok(
            "другой пользователь не может изменить поездку"
        )

        changed_fixed = dict(new_request)
        changed_fixed["budget"] = 1

        try:
            db.update_trip_plan(
                trip_id,
                101,
                changed_fixed,
                new_plan,
            )
        except ValueError as error:
            assert "budget" in str(error)
        else:
            raise AssertionError(
                "Изменение бюджета должно отклоняться."
            )

        assert db.get_trip(
            trip_id,
            101,
        ) == before_foreign

        _ok(
            "постоянные параметры поездки защищены"
        )

        db.update_itinerary(
            trip_id,
            {"legacy": True},
        )
        assert db.get_trip(
            trip_id,
            101,
        )["itinerary"] == {
            "legacy": True
        }

        _ok(
            "старый update_itinerary сохраняет совместимость"
        )

        with db.connect() as conn:
            assert conn.execute(
                "PRAGMA foreign_key_check"
            ).fetchone() is None

        _ok(
            "внешние ключи базы не нарушены"
        )

    finally:
        db.DB_PATH = original_path
        TEST_DB.unlink(missing_ok=True)

    assert _fingerprint(real_db_path) == real_db_before
    assert not TEST_DB.exists()

    print(
        "\nИтог: безопасное сохранение "
        "изменённого маршрута полностью работает."
    )
    print(
        "Временная база test_db_edit.db удалена."
    )
    print(
        "Настоящий planner.db не изменился."
    )


if __name__ == "__main__":
    main()