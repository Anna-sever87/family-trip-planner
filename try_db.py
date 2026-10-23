"""Полная автономная проверка SQLite без сетевых запросов."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import db


PROJECT_DIR = Path(__file__).resolve().parent
TEST_DB_PATH = PROJECT_DIR / "test_planner.db"
TEST_USER_ID = 111
OTHER_USER_ID = 222
ATOMIC_USER_ID = 333


def _remove_test_files() -> None:
    """Удаляет тестовую БД и служебные файлы SQLite."""

    for path in (
        TEST_DB_PATH,
        Path(f"{TEST_DB_PATH}-wal"),
        Path(f"{TEST_DB_PATH}-shm"),
        Path(f"{TEST_DB_PATH}-journal"),
    ):
        path.unlink(missing_ok=True)


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def main() -> None:
    original_db_path = db.DB_PATH

    try:
        _remove_test_files()
        db.DB_PATH = str(TEST_DB_PATH)

        db.init_db()
        db.init_db()
        _ok(
            "init_db создаёт схему и повторно "
            "выполняется без ошибок"
        )

        with db.connect() as conn:
            tables = {
                str(row["name"])
                for row in conn.execute(
                    """
                    SELECT name
                    FROM sqlite_master
                    WHERE type = 'table'
                    """
                ).fetchall()
            }

            assert {
                "users",
                "trips",
                "checklist_items",
                "memory_items",
            }.issubset(tables)

            user_columns = {
                str(row["name"])
                for row in conn.execute(
                    "PRAGMA table_info(users)"
                ).fetchall()
            }

            assert {
                "pace",
                "wishes",
            }.issubset(user_columns)

        _ok(
            "созданы таблицы и новые поля "
            "семейного профиля"
        )

        # Старый вызов с пятью позиционными
        # аргументами обязан продолжить работать.
        db.save_user(
            TEST_USER_ID,
            "Анна",
            2,
            [4, 9],
            "EUR",
        )

        profile = db.get_user(
            TEST_USER_ID
        )

        assert profile is not None
        assert profile["name"] == "Анна"
        assert profile["children_ages"] == [4, 9]
        assert profile["pace"] == "средний"
        assert profile["wishes"] == ""

        _ok(
            "старый API save_user совместим "
            "с новой схемой"
        )

        db.save_user(
            TEST_USER_ID,
            "Анна",
            2,
            [4, 9],
            "EUR",
            pace="спокойный",
            wishes=(
                "музеи, парки и дневной отдых"
            ),
        )

        profile = db.get_user(
            TEST_USER_ID
        )

        assert profile is not None
        assert profile["pace"] == "спокойный"
        assert profile["wishes"] == (
            "музеи, парки и дневной отдых"
        )

        # Обычный старый вызов передаёт
        # pace=None и wishes=None.
        # Эти значения не должны стираться.
        db.save_user(
            TEST_USER_ID,
            "Анна П.",
            3,
            [4, 9],
            "RUB",
        )

        profile = db.get_user(
            TEST_USER_ID
        )

        assert profile is not None
        assert profile["name"] == "Анна П."
        assert profile["adults"] == 3
        assert profile["currency"] == "RUB"
        assert profile["pace"] == "спокойный"
        assert profile["wishes"] == (
            "музеи, парки и дневной отдых"
        )

        _ok(
            "pace и wishes сохраняются "
            "при значениях None"
        )

        db.save_user(
            OTHER_USER_ID,
            "Другая семья",
            1,
            [7],
            "EUR",
        )

        atomic_request = {
            "destination": "Казань",
            "start_date": "2026-12-01",
            "end_date": "2026-12-02",
        }
        atomic_itinerary = {
            "title": "Семейная Казань",
            "days": [],
        }
        atomic_trip_id = db.save_trip_bundle(
            ATOMIC_USER_ID,
            atomic_request,
            atomic_itinerary,
            [
                ("Документы", "Проверить паспорта"),
                ("Техника", "Взять зарядные устройства"),
            ],
            name="Атомарная семья",
            adults=2,
            children_ages=[6],
            currency="RUB",
            pace="спокойный",
            wishes="интерактивные музеи",
        )

        atomic_profile = db.get_user(ATOMIC_USER_ID)
        assert atomic_profile is not None
        assert atomic_profile["name"] == "Атомарная семья"
        assert atomic_profile["children_ages"] == [6]
        assert atomic_profile["wishes"] == "интерактивные музеи"
        assert db.get_trip(atomic_trip_id, ATOMIC_USER_ID) is not None
        assert len(db.get_checklist(atomic_trip_id)) == 2

        with db.connect() as conn:
            conn.execute(
                """
                CREATE TRIGGER fail_atomic_bundle_test
                BEFORE INSERT ON checklist_items
                WHEN NEW.text = '__FORCED_ROLLBACK__'
                BEGIN
                    SELECT RAISE(ABORT, 'forced rollback');
                END
                """
            )

        try:
            db.save_trip_bundle(
                ATOMIC_USER_ID,
                {
                    "destination": "Лион",
                    "start_date": "2027-01-10",
                    "end_date": "2027-01-10",
                },
                {"title": "Не должен сохраниться", "days": []},
                [("Тест", "__FORCED_ROLLBACK__")],
                name="Не должно сохраниться",
                adults=5,
                children_ages=[],
                currency="EUR",
                pace="насыщенный",
                wishes="не должно сохраниться",
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError(
                "Искусственный сбой не отменил пакетное сохранение"
            )
        finally:
            with db.connect() as conn:
                conn.execute(
                    "DROP TRIGGER IF EXISTS fail_atomic_bundle_test"
                )

        assert db.get_user(ATOMIC_USER_ID) == atomic_profile
        assert [
            item["id"]
            for item in db.list_trips(ATOMIC_USER_ID)
        ] == [atomic_trip_id]
        assert len(db.get_checklist(atomic_trip_id)) == 2

        _ok(
            "профиль, поездка и чек-лист сохраняются атомарно, "
            "а при сбое полностью откатываются"
        )

        request = {
            "destination": "Париж",
            "start_date": "2026-11-10",
            "end_date": "2026-11-12",
        }

        itinerary = {
            "title": "Семейный Париж",
            "days": [
                {
                    "date": "2026-11-10",
                    "items": [],
                }
            ],
        }

        trip_id = db.save_trip(
            TEST_USER_ID,
            request,
            itinerary,
        )

        trip = db.get_trip(
            trip_id,
            TEST_USER_ID,
        )

        assert trip is not None
        assert trip["destination"] == "Париж"
        assert trip["request"] == request
        assert trip["itinerary"] == itinerary

        assert (
            db.get_trip(
                trip_id,
                OTHER_USER_ID,
            )
            is None
        )

        assert [
            item["id"]
            for item in db.list_trips(
                TEST_USER_ID
            )
        ] == [trip_id]

        changed_itinerary = {
            "title": "Обновлённый маршрут",
            "days": [],
        }

        db.update_itinerary(
            trip_id,
            changed_itinerary,
        )

        updated_trip = db.get_trip(
            trip_id,
            TEST_USER_ID,
        )

        assert updated_trip is not None
        assert updated_trip["itinerary"] == (
            changed_itinerary
        )

        _ok(
            "поездки сохраняются, обновляются "
            "и изолируются по владельцу"
        )

        db.save_checklist(
            trip_id,
            [
                (
                    "Документы",
                    "Проверить паспорта",
                ),
                (
                    "Дети",
                    "Взять любимую игрушку",
                ),
            ],
        )

        checklist = db.get_checklist(
            trip_id
        )

        assert len(checklist) == 2

        first_item_id = int(
            checklist[0]["id"]
        )

        assert (
            db.get_checklist_item(
                first_item_id,
                TEST_USER_ID,
            )
            is not None
        )

        assert (
            db.get_checklist_item(
                first_item_id,
                OTHER_USER_ID,
            )
            is None
        )

        db.set_item_done(
            first_item_id,
            True,
        )

        checklist = db.get_checklist(
            trip_id
        )

        assert int(
            checklist[0]["done"]
        ) == 1

        _ok(
            "чек-лист сохраняется, отмечается "
            "и защищён от чужого чтения"
        )

        memory_row_id = (
            db.save_memory_item(
                TEST_USER_ID,
                "memory-family-1",
                "preference",
                "Семья любит музеи",
                destination="Париж",
                source_trip_id=trip_id,
            )
        )

        memory_item = db.get_memory_item(
            memory_row_id,
            TEST_USER_ID,
        )

        assert memory_item is not None
        assert memory_item["memory_id"] == (
            "memory-family-1"
        )
        assert memory_item["source_trip_id"] == (
            trip_id
        )

        # Такая же пара tg_id + memory_id
        # должна обновить строку без дубля.
        upserted_row_id = (
            db.save_memory_item(
                TEST_USER_ID,
                "memory-family-1",
                "preference",
                (
                    "Семья любит музеи "
                    "и спокойные прогулки"
                ),
                destination="Париж",
                source_trip_id=trip_id,
            )
        )

        assert (
            upserted_row_id
            == memory_row_id
        )

        memories = db.list_memory_items(
            TEST_USER_ID
        )

        assert len(memories) == 1
        assert memories[0]["text"] == (
            "Семья любит музеи "
            "и спокойные прогулки"
        )

        second_memory_row_id = (
            db.save_memory_item(
                TEST_USER_ID,
                "memory-family-2",
                "restriction",
                "Нужен дневной отдых",
            )
        )

        assert len(
            db.list_memory_items(
                TEST_USER_ID
            )
        ) == 2

        assert (
            db.get_memory_item_by_memory_id(
                TEST_USER_ID,
                "memory-family-2",
            )
            is not None
        )

        _ok(
            "журнал памяти поддерживает "
            "создание, upsert, список и чтение"
        )

        # У другого пользователя такой же
        # memory_id является отдельной записью.
        other_memory_row_id = (
            db.save_memory_item(
                OTHER_USER_ID,
                "memory-family-1",
                "preference",
                (
                    "Другая семья любит "
                    "активный отдых"
                ),
            )
        )

        assert (
            other_memory_row_id
            != memory_row_id
        )

        assert (
            db.get_memory_item(
                memory_row_id,
                OTHER_USER_ID,
            )
            is None
        )

        assert (
            db.delete_memory_item(
                memory_row_id,
                OTHER_USER_ID,
            )
            is False
        )

        try:
            db.save_memory_item(
                OTHER_USER_ID,
                "forbidden-memory",
                "feedback",
                (
                    "Попытка связать "
                    "с чужой поездкой"
                ),
                source_trip_id=trip_id,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Чужую поездку удалось "
                "привязать к памяти"
            )

        assert (
            db.delete_memory_item_by_memory_id(
                OTHER_USER_ID,
                "memory-family-1",
            )
            is True
        )

        assert (
            db.get_memory_item_by_memory_id(
                TEST_USER_ID,
                "memory-family-1",
            )
            is not None
        )

        _ok(
            "записи памяти разных "
            "пользователей изолированы"
        )

        assert (
            db.delete_memory_item(
                second_memory_row_id,
                TEST_USER_ID,
            )
            is True
        )

        assert (
            db.get_memory_item(
                second_memory_row_id,
                TEST_USER_ID,
            )
            is None
        )

        _ok(
            "удаление отдельной записи "
            "памяти работает"
        )

        assert (
            db.delete_trip(
                trip_id,
                OTHER_USER_ID,
            )
            is False
        )

        assert (
            db.get_trip(
                trip_id,
                TEST_USER_ID,
            )
            is not None
        )

        assert (
            db.delete_trip(
                trip_id,
                TEST_USER_ID,
            )
            is True
        )

        assert (
            db.get_trip(
                trip_id,
                TEST_USER_ID,
            )
            is None
        )

        assert (
            db.get_checklist(
                trip_id
            )
            == []
        )

        memory_after_trip_delete = (
            db.get_memory_item(
                memory_row_id,
                TEST_USER_ID,
            )
        )

        assert (
            memory_after_trip_delete
            is not None
        )

        assert (
            memory_after_trip_delete[
                "source_trip_id"
            ]
            is None
        )

        _ok(
            "delete_trip удаляет чек-лист "
            "и сохраняет память без ссылки "
            "на поездку"
        )

        db.save_memory_item(
            TEST_USER_ID,
            "memory-family-3",
            "feedback",
            (
                "Прошлая поездка "
                "понравилась"
            ),
        )

        assert (
            db.clear_memory_items(
                TEST_USER_ID
            )
            == 2
        )

        assert (
            db.list_memory_items(
                TEST_USER_ID
            )
            == []
        )

        assert (
            db.clear_memory_items(
                TEST_USER_ID
            )
            == 0
        )

        _ok(
            "полная очистка локального "
            "журнала памяти работает"
        )

        with db.connect() as conn:
            violations = conn.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()

        assert violations == []

        _ok(
            "PRAGMA foreign_key_check "
            "не нашёл нарушений"
        )

        print(
            "\nИтог: SQLite и журнал памяти "
            "полностью работают."
        )
    finally:
        db.DB_PATH = original_db_path
        _remove_test_files()
        print(
            "Временная база "
            "test_planner.db удалена."
        )


if __name__ == "__main__":
    main()
