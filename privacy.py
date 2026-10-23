"""Подтверждённое удаление активных данных только собственного аккаунта.

Резервные копии администратора, скачанные документы и общий кеш карт
не принадлежат активному аккаунту и этой функцией не изменяются.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import db
import memory as vector_memory
import reminders


# Рабочий каталог совпадает с DB_PATH и кешем карт, в том числе /data
# при контейнерном запуске с исходным кодом в /app.
OUTPUT_DIR = Path("output")
DELETION_LEASE_MINUTES = 10


class AccountDeletionError(RuntimeError):
    """Полное удаление не подтверждено; запрос можно повторить."""


class AccountDeletionBusyError(AccountDeletionError):
    """Сначала нужно дождаться текущей операции."""


@dataclass(frozen=True)
class AccountDeletionResult:
    deleted: bool
    trips_deleted: int = 0
    pdfs_deleted: int = 0


def _pdf_paths(trip_ids: tuple[int, ...]) -> tuple[Path, ...]:
    """Разрешает только точные старые имена PDF в каталоге output."""

    if OUTPUT_DIR.is_symlink():
        raise AccountDeletionError("Каталог экспорта является ссылкой; удаление остановлено.")
    output_root = OUTPUT_DIR.resolve()
    paths = []
    for trip_id in trip_ids:
        candidate = OUTPUT_DIR / f"trip_{trip_id}.pdf"
        if candidate.is_symlink():
            raise AccountDeletionError("PDF поездки является ссылкой; удаление остановлено.")
        resolved = candidate.resolve()
        if resolved.parent != output_root:
            raise AccountDeletionError("PDF поездки находится вне каталога экспорта.")
        if resolved.exists():
            if not resolved.is_file():
                raise AccountDeletionError("На месте PDF поездки находится каталог.")
            paths.append(resolved)
    return tuple(paths)


def delete_all_user_data(
    tg_id: int,
    *,
    actor_tg_id: int,
    confirm: bool = False,
) -> AccountDeletionResult:
    """Очищает Pinecone, затем активный SQLite-профиль и его точные PDF.

    Вызов доступен только после явного подтверждения владельцем.
    Telegram-обработчик дополнительно резервирует свои маршрутные и
    фоновые операции памяти, а веб-запрос использует сессию и CSRF.
    Во время сетевого удаления запись-аренда блокирует новые AI-запросы,
    запись профиля, поездок, памяти и выдачу ссылок входа этой семьи.
    SQLite не блокирует остальные семьи на время запроса в Pinecone.
    При ошибке локальные записи сохраняются для повторной попытки;
    память Pinecone или часть PDF уже могут быть удалены.
    """

    if (
        isinstance(tg_id, bool)
        or not isinstance(tg_id, int)
        or tg_id <= 0
        or isinstance(actor_tg_id, bool)
        or not isinstance(actor_tg_id, int)
        or actor_tg_id != tg_id
    ):
        raise PermissionError("Можно удалить только собственный личный аккаунт.")
    if confirm is not True:
        raise ValueError("Для удаления всех данных требуется явное подтверждение.")

    token = uuid4().hex
    current = datetime.now(timezone.utc)
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        user = conn.execute(
            "SELECT account_key, deletion_token, deletion_started_at FROM users WHERE tg_id = ?",
            (tg_id,),
        ).fetchone()
        had_profile = user is not None
        if user is None:
            # Неуспешная запись журнала могла оставить только Pinecone.
            # Временный профиль нужен лишь для блокировки параллельных
            # запросов; после завершения или ошибки он снова удаляется.
            db._ensure_user_exists(conn, tg_id)
            user = conn.execute(
                "SELECT account_key, deletion_token, deletion_started_at FROM users WHERE tg_id = ?",
                (tg_id,),
            ).fetchone()
        if user["deletion_token"]:
            started = db._parse_timestamp(user["deletion_started_at"])
            if started is None or started > current - timedelta(minutes=DELETION_LEASE_MINUTES):
                raise AccountDeletionBusyError("Удаление данных уже выполняется.")
        running = conn.execute(
            "SELECT 1 FROM ai_requests WHERE tg_id = ? AND status = 'running' LIMIT 1",
            (tg_id,),
        ).fetchone()
        if running is not None:
            raise AccountDeletionBusyError("Дождитесь завершения создания или изменения маршрута.")
        account_key = str(user["account_key"])
        trip_ids = tuple(int(row[0]) for row in conn.execute(
            "SELECT id FROM trips WHERE tg_id = ? ORDER BY id", (tg_id,)
        ).fetchall())
        pdf_paths = _pdf_paths(trip_ids)
        conn.execute(
            "UPDATE users SET deletion_token = ?, deletion_started_at = ? WHERE tg_id = ?",
            (token, current.isoformat(), tg_id),
        )

    try:
        if vector_memory.clear_family_memories(tg_id, tg_id, confirm=True) is not True:
            raise AccountDeletionError(
                "Pinecone не подтвердил очистку памяти. Локальные данные сохранены; повторите удаление."
            )
        # Проверка lease выполняется до удаления каждого точного набора
        # файлов и SQL-строк. Сеть уже завершена, транзакция короткая.
        with db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            owner = conn.execute(
                "SELECT 1 FROM users WHERE tg_id = ? AND account_key = ? AND deletion_token = ?",
                (tg_id, account_key, token),
            ).fetchone()
            if owner is None:
                raise AccountDeletionError("Операция удаления устарела. Повторите запрос.")
            removed_pdfs = 0
            for pdf_path in pdf_paths:
                # Повторная проверка предотвращает следование новой ссылке.
                if pdf_path.is_symlink() or pdf_path.resolve().parent != OUTPUT_DIR.resolve():
                    raise AccountDeletionError("Путь PDF изменился; удаление остановлено.")
                if pdf_path.exists():
                    pdf_path.unlink()
                    removed_pdfs += 1
            # Отношения trips -> users в старой схеме не имели CASCADE.
            # Остальные таблицы удаляются их существующими внешними ключами.
            db._retain_ai_usage(conn, tg_id, current.astimezone(reminders.get_tz()).date().isoformat())
            conn.execute("DELETE FROM trips WHERE tg_id = ?", (tg_id,))
            conn.execute("DELETE FROM users WHERE tg_id = ? AND account_key = ?", (tg_id, account_key))
        return AccountDeletionResult(had_profile, len(trip_ids), removed_pdfs)
    except Exception as error:
        with db.connect() as conn:
            if had_profile:
                conn.execute(
                    "UPDATE users SET deletion_token = '', deletion_started_at = NULL "
                    "WHERE tg_id = ? AND account_key = ? AND deletion_token = ?",
                    (tg_id, account_key, token),
                )
            else:
                conn.execute(
                    "DELETE FROM users WHERE tg_id = ? AND account_key = ? AND deletion_token = ?",
                    (tg_id, account_key, token),
                )
        if isinstance(error, AccountDeletionError):
            raise
        raise AccountDeletionError(
            "Удаление не завершено. Локальный профиль сохранён; повторите запрос."
        ) from error
