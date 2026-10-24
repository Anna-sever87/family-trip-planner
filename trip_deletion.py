"""Delete one owned trip, preserving family memory and paid-attempt ledgers.

The SQL deletion and exact-file cleanup receipts commit atomically. Files are
removed only afterward; a disk failure never pretends the route still exists.
Pending cleanup is owner/account-bound on retry and bounded on web startup.
No model, remote memory, message, recursive delete or client path is involved.
"""
import hashlib
import logging
from pathlib import Path
import re

import db


OUTPUT_DIR = Path("output")
_COVER_NAME = re.compile(r"[0-9a-f]{32}\.(?:jpg|png)\Z")
_BUSY = "Дождитесь завершения текущей операции с маршрутом, обложкой или напоминанием и повторите удаление."


class TripDeleteNotFoundError(LookupError):
    pass


class TripDeleteBusyError(RuntimeError):
    pass


class TripDeleteStorageError(RuntimeError):
    pass


def _root(kind):
    if kind == "cover":
        import trip_covers
        value = trip_covers.manager.configured_cache_dir()
    elif kind == "pdf":
        value = OUTPUT_DIR
    else:
        raise TripDeleteStorageError("Не удалось безопасно проверить файлы поездки. Поездка сохранена.")
    value = Path(value)
    if any(parent.is_symlink() for parent in (value, *value.parents)):
        raise TripDeleteStorageError("Не удалось безопасно проверить каталог файлов поездки.")
    return value.resolve()


def _fingerprint(root):
    return hashlib.sha256(str(root).encode("utf-8")).hexdigest()


def _path(kind, name, trip_id, fingerprint=None):
    valid = (isinstance(name, str) and (
        (kind == "cover" and _COVER_NAME.fullmatch(name))
        or (kind == "pdf" and name == f"trip_{trip_id}.pdf")))
    if not valid:
        raise TripDeleteStorageError("Не удалось безопасно проверить имена файлов поездки.")
    root = _root(kind)
    if fingerprint is not None and fingerprint != _fingerprint(root):
        raise TripDeleteStorageError("Каталог файлов изменился; очистка ожидает проверки.")
    target = root / name
    if target.is_symlink() or target.resolve().parent != root or (target.exists() and not target.is_file()):
        raise TripDeleteStorageError("Не удалось безопасно проверить путь файла поездки.")
    return target, _fingerprint(root)


def _cover_referenced(conn, name, *, excluding_trip=None):
    for table in ("trip_covers", "trip_cover_overrides"):
        if not db._table_exists(conn, table):
            continue
        query = f"SELECT 1 FROM {table} WHERE file_name=?"
        values = (name,)
        if excluding_trip is not None:
            query += " AND trip_id!=?"
            values += (excluding_trip,)
        if conn.execute(query + " LIMIT 1", values).fetchone() is not None:
            return True
    return False


def _files_for_trip(conn, trip_id, account_key):
    files = []
    for table in ("trip_covers", "trip_cover_overrides"):
        if not db._table_exists(conn, table):
            continue
        row = conn.execute(f"SELECT account_key,file_name FROM {table} WHERE trip_id=?", (trip_id,)).fetchone()
        if row is None:
            continue
        if row["account_key"] != account_key:
            raise TripDeleteStorageError("Не удалось подтвердить принадлежность файлов. Поездка сохранена.")
        if row["file_name"]:
            target, fingerprint = _path("cover", row["file_name"], trip_id)
            if target.exists() and not _cover_referenced(conn, row["file_name"], excluding_trip=trip_id):
                files.append(("cover", row["file_name"], fingerprint))
    name = f"trip_{trip_id}.pdf"
    target, fingerprint = _path("pdf", name, trip_id)
    if target.exists():
        files.append(("pdf", name, fingerprint))
    return sorted(set(files))


def _busy(conn, tg_id, account_key, trip_id):
    if conn.execute("SELECT 1 FROM ai_requests WHERE tg_id=? AND status='running' LIMIT 1", (tg_id,)).fetchone():
        return True
    if conn.execute("SELECT 1 FROM generation_jobs WHERE account_key=? AND status IN ('queued','running') LIMIT 1",
                    (account_key,)).fetchone():
        return True
    if conn.execute("SELECT 1 FROM reminder_deliveries WHERE trip_id=? AND status='sending' LIMIT 1",
                    (trip_id,)).fetchone():
        return True
    return (db._table_exists(conn, "trip_covers") and conn.execute(
        "SELECT 1 FROM trip_covers WHERE trip_id=? AND status IN ('queued','running')", (trip_id,)).fetchone() is not None)


def _cleanup_one(identifier):
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM trip_file_cleanup WHERE id=?", (identifier,)).fetchone()
        if row is None:
            return 0
        # Never erase a file while a live route or another cover owns it.
        if conn.execute("SELECT 1 FROM trips WHERE id=?", (row["trip_id"],)).fetchone():
            raise TripDeleteStorageError("Поездка ещё существует; очистка остановлена.")
        if row["kind"] == "cover" and _cover_referenced(conn, row["file_name"]):
            conn.execute("DELETE FROM trip_file_cleanup WHERE id=?", (identifier,))
            return 0
        target, _ = _path(row["kind"], row["file_name"], row["trip_id"], row["root_fingerprint"])
        removed = int(target.exists())
        target.unlink(missing_ok=True)
        conn.execute("DELETE FROM trip_file_cleanup WHERE id=?", (identifier,))
        return removed


def cleanup_pending_files(*, tg_id=None, account_key=None, trip_id=None, limit=100):
    """Bounded exact-file cleanup; errors leave durable receipts for retry."""
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Invalid cleanup bound")
    conditions, values = [], []
    for key, value in (("tg_id", tg_id), ("account_key", account_key), ("trip_id", trip_id)):
        if value is not None:
            conditions.append(key + "=?")
            values.append(value)
    where = " WHERE " + " AND ".join(conditions) if conditions else ""
    removed, pending = 0, True
    try:
        with db.connect() as conn:
            identifiers = [row[0] for row in conn.execute(
                "SELECT id FROM trip_file_cleanup" + where + " ORDER BY id LIMIT ?", (*values, limit))]
        for identifier in identifiers:
            try:
                removed += _cleanup_one(identifier)
            except Exception:
                # No identity, file path, provider text or route data in logs.
                logging.getLogger("web.deletion").warning("trip_file_cleanup_pending")
        with db.connect() as conn:
            pending = conn.execute("SELECT 1 FROM trip_file_cleanup" + where + " LIMIT 1", values).fetchone() is not None
    except Exception:
        logging.getLogger("web.deletion").warning("trip_file_cleanup_unconfirmed")
    return removed, pending


def delete_owned_trip(tg_id, account_key, trip_id):
    if (type(tg_id) is not int or tg_id <= 0 or type(trip_id) is not int or trip_id <= 0
            or not isinstance(account_key, str) or not account_key):
        raise TripDeleteNotFoundError("Поездка не найдена.")
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        owner = conn.execute("SELECT deletion_token FROM users WHERE tg_id=? AND account_key=?",
                             (tg_id, account_key)).fetchone()
        if owner is None:
            raise TripDeleteNotFoundError("Поездка не найдена.")
        if owner["deletion_token"]:
            raise TripDeleteBusyError(_BUSY)
        trip = conn.execute("SELECT 1 FROM trips WHERE id=? AND tg_id=?", (trip_id, tg_id)).fetchone()
        if trip is None:
            pending = conn.execute("SELECT 1 FROM trip_file_cleanup WHERE tg_id=? AND account_key=? AND trip_id=? LIMIT 1",
                                   (tg_id, account_key, trip_id)).fetchone()
            if pending is None:
                raise TripDeleteNotFoundError("Поездка не найдена.")
        else:
            if _busy(conn, tg_id, account_key, trip_id):
                raise TripDeleteBusyError(_BUSY)
            for kind, name, fingerprint in _files_for_trip(conn, trip_id, account_key):
                conn.execute("INSERT OR IGNORE INTO trip_file_cleanup "
                             "(tg_id,account_key,trip_id,kind,file_name,root_fingerprint,created_at) VALUES (?,?,?,?,?,?,?)",
                             (tg_id, account_key, trip_id, kind, name, fingerprint, db._now()))
                recorded = conn.execute("SELECT tg_id,account_key,trip_id FROM trip_file_cleanup "
                                        "WHERE kind=? AND file_name=? AND root_fingerprint=?",
                                        (kind, name, fingerprint)).fetchone()
                if tuple(recorded) != (tg_id, account_key, trip_id):
                    raise TripDeleteStorageError("У файла уже есть другая запись очистки. Поездка сохранена.")
            # A previously finished receipt is now explicitly about deletion,
            # never a fresh successful generation or a reason to retry AI.
            conn.execute("UPDATE generation_jobs SET status='failed',failure_code='deleted',"
                         "destination_options_json='[]' WHERE trip_id=? AND tg_id=? AND account_key=?",
                         (trip_id, tg_id, account_key))
            conn.execute("DELETE FROM trips WHERE id=? AND tg_id=?", (trip_id, tg_id))
    removed, pending = cleanup_pending_files(tg_id=tg_id, account_key=account_key, trip_id=trip_id)
    message = "Поездка удалена. Общая память семьи и использованные попытки сохранены."
    if pending:
        message += " Очистка некоторых файлов пока не подтверждена; она будет повторена при следующем запуске сервера."
    return {"deleted": True, "trip_id": trip_id, "files_deleted": removed,
            "cleanup_pending": pending, "message": message}
