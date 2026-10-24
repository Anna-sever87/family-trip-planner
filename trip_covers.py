"""Owner-bound, bounded and durable decorative cover jobs.

Reads never start generation. Each trip has one immutable receipt, including
failures; neither a refresh nor restart can submit its paid POST a second time.
The attempt ledger contains no account/trip/prompt and survives account deletion.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import re
import threading
import unicodedata
from uuid import uuid4

import db
import access_control
import access_guard
from cover_prompts import COVER_DISCLOSURE, build_cover_prompt
from cover_subjects import select_saved_cover_subject
import yandex_art


MAX_ACTIVE = 2
MESSAGES = {
    "disabled": "Генерация иллюстраций пока не подключена. Маршрут доступен без обложки.",
    "empty": "Для поездки можно создать отдельную ИИ-иллюстрацию направления.",
    "queued": "Иллюстрация ожидает обработки. Маршрут уже сохранён и доступен.",
    "running": "Иллюстрация создаётся. Можно закрыть страницу; маршрут доступен без неё.",
    "ready": "ИИ-иллюстрация готова. Это художественная обложка, не фотография места.",
    "hidden": "Иллюстрация скрыта. Файл сохранён; восстановление не запускает генерацию.",
    "failed": "Иллюстрацию создать не удалось. Маршрут сохранён; платный запрос автоматически не повторяется.",
    "interrupted": "Создание иллюстрации прервано или результат не подтверждён. Новый платный запрос автоматически не отправляется.",
    "busy": "Обложки уже создаются. Новый запрос не запущен и лимит не потрачен.",
    "limit": "Дневной лимит иллюстраций исчерпан. Маршрут доступен без обложки.",
    "destination": "Для обложки нужны однозначные название направления и страна. Маршрут доступен без неё.",
    "no_verified_subject": "Не удалось выбрать подходящее подтверждённое место из сохранённого маршрута для обложки. Это не означает, что в поездке нет достопримечательностей. Используется нейтральное оформление; платная генерация не запускается.",
}
_FILE_NAME = re.compile(r"[0-9a-f]{32}\.(?:jpg|png)\Z")
_REVIEW_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_IMAGE_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _safe_caption(label, location):
    """Only bounded public labels; no request/prompt strings or markup."""
    for value, maximum in ((label, 120), (location, 240)):
        if (not isinstance(value, str) or not value or len(value) > maximum
                or value != value.strip()
                or not any(unicodedata.category(char).startswith("L") for char in value)
                or any(not (unicodedata.category(char)[0] in "LMN"
                            or char in " .,'’()-\"«»–—") for char in value)):
            return "", ""
    return label, location


def _prompt_caption(prompt):
    label, location = getattr(prompt, "subject_label", ""), getattr(prompt, "subject_location", "")
    checked = _safe_caption(label, location)
    if (label or location) and checked == ("", ""):
        raise CoverDestinationError("Некорректная подпись иллюстрации.")
    return checked


class CoverNotFoundError(LookupError):
    pass


class CoverBusyError(RuntimeError):
    pass


class CoverLimitError(RuntimeError):
    pass


class CoverDestinationError(ValueError):
    pass


class CoverStateError(ValueError):
    pass


def enabled():
    if os.getenv("IMAGE_GENERATION_ENABLED", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    if not (os.getenv("YANDEX_ART_API_KEY", "").strip() or os.getenv("YANDEX_API_KEY", "").strip()):
        return False
    try:
        yandex_art.configured_model_uri()
    except yandex_art.YandexArtError:
        return False
    return True


def daily_limit():
    raw = os.getenv("IMAGE_GENERATION_DAILY_LIMIT", "10").strip()
    # Invalid or excessively high configuration fails closed.
    return int(raw) if raw.isdecimal() and len(raw) <= 2 and 1 <= int(raw) <= 50 else 0


def init_schema():
    with db.connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS trip_covers (
            trip_id INTEGER PRIMARY KEY,
            account_key TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('queued','running','ready','failed','interrupted')),
            operation_id TEXT NOT NULL DEFAULT '',
            file_name TEXT NOT NULL DEFAULT '',
            mime_type TEXT NOT NULL DEFAULT '',
            hidden INTEGER NOT NULL DEFAULT 0 CHECK(hidden IN (0,1)),
            failure_code TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            finished_at TEXT NOT NULL DEFAULT '',
            subject_label TEXT NOT NULL DEFAULT '',
            subject_location TEXT NOT NULL DEFAULT '',
            subject_image_sha256 TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(trip_id) REFERENCES trips(id) ON DELETE CASCADE,
            FOREIGN KEY(account_key) REFERENCES users(account_key) ON DELETE CASCADE
        )""")
        if "hidden" not in {row["name"] for row in conn.execute("PRAGMA table_info(trip_covers)")}:
            conn.execute("ALTER TABLE trip_covers ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0 CHECK(hidden IN (0,1))")
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(trip_covers)")}
        # No backfill: today's subject selection is not evidence about an old
        # image. The existing immutable receipt, bytes and quota stay untouched.
        for name in ("subject_label", "subject_location", "subject_image_sha256"):
            if name not in columns:
                conn.execute(f"ALTER TABLE trip_covers ADD COLUMN {name} TEXT NOT NULL DEFAULT ''")
        conn.execute("""CREATE TABLE IF NOT EXISTS trip_cover_overrides (
            trip_id INTEGER PRIMARY KEY,
            account_key TEXT NOT NULL,
            file_name TEXT NOT NULL,
            mime_type TEXT NOT NULL,
            source_receipt TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            image_sha256 TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(trip_id) REFERENCES trips(id) ON DELETE CASCADE,
            FOREIGN KEY(account_key) REFERENCES users(account_key) ON DELETE CASCADE
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS cover_attempts (
            attempt_id TEXT PRIMARY KEY,
            utc_day TEXT NOT NULL
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cover_attempts_day ON cover_attempts(utc_day)")


def _owned_trip(conn, tg_id, account_key, trip_id):
    if not isinstance(account_key, str) or not account_key:
        raise CoverNotFoundError("Поездка не найдена.")
    row = conn.execute(
        "SELECT t.id,t.destination,t.start_date,t.end_date,t.request_json,t.itinerary_json "
        "FROM trips t JOIN users u ON u.tg_id=t.tg_id "
        "WHERE t.id=? AND t.tg_id=? AND u.account_key=? AND u.deletion_token=''",
        (trip_id, tg_id, account_key),
    ).fetchone()
    if row is None:
        raise CoverNotFoundError("Поездка не найдена.")
    return row


def _prompt_for(trip):
    label = str(trip["destination"] or "").strip()
    if "," not in label:
        # Only application-owned, unambiguous defaults already used by the app.
        countries = {"москва": "Россия", "moscow": "Россия", "сочи": "Россия", "sochi": "Россия",
                     "байкал": "Россия", "озеро байкал": "Россия", "baikal": "Россия",
                     "lake baikal": "Россия", "нячанг": "Вьетнам", "nha trang": "Вьетнам"}
        country = countries.get(label.casefold())
        if country is None:
            raise CoverDestinationError(MESSAGES["destination"])
        destination = label
    else:
        destination, country = (part.strip() for part in label.rsplit(",", 1))
    try:
        prompt = build_cover_prompt(destination, country, trip["start_date"], trip["end_date"])
        if prompt.generation_allowed:
            return prompt
        subject = select_saved_cover_subject(
            trip["request_json"], trip["itinerary_json"], destination=destination, country=country,
            start_date=trip["start_date"], end_date=trip["end_date"],
        )
        return (build_cover_prompt(destination, country, trip["start_date"], trip["end_date"], subject=subject)
                if subject is not None else prompt)
    except ValueError:
        raise CoverDestinationError(MESSAGES["destination"]) from None


class TripCovers:
    def __init__(self, *, executor_factory=ThreadPoolExecutor, cache_dir=None):
        self.executor_factory = executor_factory
        self._cache_override = Path(cache_dir).resolve() if cache_dir is not None else None
        self.cache_dir = self._cache_override or Path("output/covers").resolve()
        self.executor = None
        self.clients = 0
        self.closing = False
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(MAX_ACTIVE)

    def configured_cache_dir(self):
        """Resolve the shared cache without starting workers or changing jobs.

        Telegram never starts this manager, but deletion must still use the
        same IMAGE_CACHE_DIR as the web process. Explicit test overrides and
        an already running manager retain their fixed directory.
        """
        if self._cache_override is not None:
            return self._cache_override
        if self.executor is not None:
            return self.cache_dir
        return Path(os.getenv("IMAGE_CACHE_DIR", "").strip() or "output/covers")

    def start(self):
        with self.lock:
            if self.closing:
                raise CoverBusyError(MESSAGES["busy"])
            if self.executor is not None:
                self.clients += 1
                return
            # Resolve after the web app has loaded .env. Relative paths follow
            # the runtime working directory: /data/output/covers in Docker,
            # not the read-only application source directory /app.
            self.cache_dir = self._cache_override or Path(
                os.getenv("IMAGE_CACHE_DIR", "").strip() or "output/covers"
            ).resolve()
            init_schema()
            with db.connect() as conn:
                conn.execute("UPDATE trip_covers SET status='interrupted',finished_at=? "
                             "WHERE status IN ('queued','running')", (db._now(),))
            self.executor = self.executor_factory(max_workers=1, thread_name_prefix="trip-cover")
            self.clients = 1

    def close(self, *, wait=True):
        with self.lock:
            if self.clients > 1:
                self.clients -= 1
                return
            self.clients = 0
            self.closing = True
            executor, self.executor = self.executor, None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=False)
        if wait:
            with self.lock:
                self.closing = False

    def _path(self, name):
        if not isinstance(name, str) or not _FILE_NAME.fullmatch(name):
            raise CoverNotFoundError("Иллюстрация недоступна.")
        target = self.cache_dir / name
        if target.is_symlink() or target.resolve().parent != self.cache_dir:
            raise CoverNotFoundError("Иллюстрация недоступна.")
        return target

    def _status(self, row=None):
        state = row["status"] if row else ("empty" if enabled() else "disabled")
        hidden = bool(row["hidden"]) if row is not None else False
        label, location = "", ""
        if state == "ready":
            try:
                validated, displayed = self._read_display_image_record(row)
                if not hidden:
                    label, location = self._display_caption(displayed, validated)
            except CoverNotFoundError:
                state = "failed"
        return {"status": state, "message": MESSAGES["hidden" if state == "ready" and hidden else state], "disclosure": COVER_DISCLOSURE,
                "available": enabled(), "hidden": hidden,
                "subject_label": label, "subject_location": location,
                "image_url": f'/api/trips/{row["trip_id"]}/cover/image' if state == "ready" and not hidden else None}

    @staticmethod
    def _display_caption(displayed, validated):
        # Overrides are different images. Never apply the original receipt's
        # caption to an override, or recompute an old caption from today's trip.
        if ("source_receipt" in displayed.keys()
                or any(key not in displayed.keys() for key in
                       ("subject_label", "subject_location", "subject_image_sha256"))):
            return "", ""
        digest = displayed["subject_image_sha256"]
        if (not isinstance(digest, str) or not _IMAGE_DIGEST.fullmatch(digest)
                or hashlib.sha256(validated.data).hexdigest() != digest):
            return "", ""
        return _safe_caption(displayed["subject_label"], displayed["subject_location"])

    def _display_candidates(self, row):
        with db.connect() as conn:
            override = conn.execute("SELECT * FROM trip_cover_overrides WHERE trip_id=? AND account_key=?",
                                    (row["trip_id"], row["account_key"])).fetchone()
        # Once a reviewed image replaces a rejected original, never resurface
        # that original on disk corruption or loss. The UI can use its neutral
        # local fallback; both receipts and files remain preserved for audit.
        return [override] if override is not None else [row]

    def _display_image_available(self, row):
        try:
            self._read_display_image(row)
        except CoverNotFoundError:
            return False
        return True

    def _read_display_image(self, row):
        return self._read_display_image_record(row)[0]

    def _read_display_image_record(self, row):
        # Missing/corrupt reviewed files fail closed without altering originals.
        for candidate in self._display_candidates(row):
            try:
                with self._path(candidate["file_name"]).open("rb") as handle:
                    content = handle.read(yandex_art.MAX_IMAGE_BYTES + 1)
                if "image_sha256" in candidate.keys() and hashlib.sha256(content).hexdigest() != candidate["image_sha256"]:
                    continue
                validated = yandex_art.validate_image(content)
                if validated.mime_type == candidate["mime_type"]:
                    return validated, candidate
            except (CoverNotFoundError, OSError, yandex_art.YandexArtError):
                continue
        raise CoverNotFoundError("Иллюстрация недоступна.")

    def get_status(self, tg_id, account_key, trip_id):
        with db.connect() as conn:
            trip = _owned_trip(conn, tg_id, account_key, trip_id)
            row = conn.execute("SELECT * FROM trip_covers WHERE trip_id=? AND account_key=?",
                               (trip_id, account_key)).fetchone()
        if row is None and enabled():
            try:
                if not _prompt_for(trip).generation_allowed:
                    return self._no_subject_status()
            except CoverDestinationError:
                return self._no_subject_status()
        return self._status(row)

    def _no_subject_status(self):
        return {**self._status(), "available": False, "reason": "no_verified_subject",
                "message": MESSAGES["no_verified_subject"]}

    def set_hidden(self, tg_id, account_key, trip_id, hidden):
        """Change presentation only: preserve the receipt, image and paid quota."""
        if type(hidden) is not bool:
            raise ValueError("Требуется логическое значение hidden.")
        with db.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            _owned_trip(conn, tg_id, account_key, trip_id)
            row = conn.execute("SELECT * FROM trip_covers WHERE trip_id=? AND account_key=?",
                               (trip_id, account_key)).fetchone()
            if row is None or row["status"] != "ready":
                raise CoverStateError("Изменить видимость можно только у готовой обложки.")
            conn.execute("UPDATE trip_covers SET hidden=? WHERE trip_id=? AND account_key=?",
                         (int(hidden), trip_id, account_key))
            row = conn.execute("SELECT * FROM trip_covers WHERE trip_id=? AND account_key=?",
                               (trip_id, account_key)).fetchone()
        return self._status(row)

    def get_image(self, tg_id, account_key, trip_id):
        with db.connect() as conn:
            _owned_trip(conn, tg_id, account_key, trip_id)
            row = conn.execute("SELECT * FROM trip_covers "
                               "WHERE trip_id=? AND account_key=? AND status='ready'",
                               (trip_id, account_key)).fetchone()
        if row is None:
            raise CoverNotFoundError("Иллюстрация недоступна.")
        validated = self._read_display_image(row)
        with db.connect() as conn:
            _owned_trip(conn, tg_id, account_key, trip_id)
        return validated

    @staticmethod
    def _same_review(override, source_receipt, prompt_version, image_digest):
        return (override["source_receipt"] == source_receipt
                and override["prompt_version"] == prompt_version
                and override["image_sha256"] == image_digest)

    def install_reviewed_image(self, tg_id, account_key, trip_id, image, source_receipt, prompt_version):
        """Internal, local installation of a separately reviewed image.

        This is deliberately not an HTTP/upload endpoint. It never calls an AI
        provider or touches quota. One override is permitted per trip; replay of
        the same receipt, prompt version and bytes is a no-op, while any change
        is rejected. The initial generation receipt and original file survive.
        """
        if (not isinstance(source_receipt, str) or not _REVIEW_LABEL.fullmatch(source_receipt)
                or not isinstance(prompt_version, str) or not _REVIEW_LABEL.fullmatch(prompt_version)):
            raise ValueError("Требуются безопасные идентификаторы проверки иллюстрации.")
        if not isinstance(image, yandex_art.GeneratedImage):
            raise ValueError("Требуется проверенное изображение.")
        validated = yandex_art.validate_image(image.data)
        digest = hashlib.sha256(validated.data).hexdigest()
        with db.connect() as conn:
            _owned_trip(conn, tg_id, account_key, trip_id)
            original = conn.execute("SELECT * FROM trip_covers WHERE trip_id=? AND account_key=?",
                                    (trip_id, account_key)).fetchone()
            if original is None or original["status"] != "ready":
                raise CoverStateError("Проверенная обложка устанавливается только для готовой исходной заявки.")
            old = conn.execute("SELECT * FROM trip_cover_overrides WHERE trip_id=? AND account_key=?",
                               (trip_id, account_key)).fetchone()
        if old is not None:
            if not self._same_review(old, source_receipt, prompt_version, digest):
                raise CoverStateError("Для поездки уже установлена другая проверенная обложка; замена запрещена.")
            # A replay must also preserve a subsequent user choice to hide it.
            return self.get_status(tg_id, account_key, trip_id)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        name = uuid4().hex + (".jpg" if validated.mime_type == "image/jpeg" else ".png")
        target = self._path(name)
        temporary = self.cache_dir / (uuid4().hex + ".tmp")
        committed = False
        installed = False
        try:
            with temporary.open("xb") as handle:
                handle.write(validated.data)
                handle.flush()
                os.fsync(handle.fileno())
            with db.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                _owned_trip(conn, tg_id, account_key, trip_id)
                original = conn.execute("SELECT status FROM trip_covers WHERE trip_id=? AND account_key=?",
                                        (trip_id, account_key)).fetchone()
                if original is None or original["status"] != "ready":
                    raise CoverStateError("Исходная заявка обложки недоступна.")
                old = conn.execute("SELECT * FROM trip_cover_overrides WHERE trip_id=? AND account_key=?",
                                   (trip_id, account_key)).fetchone()
                if old is not None:
                    if not self._same_review(old, source_receipt, prompt_version, digest):
                        raise CoverStateError("Для поездки уже установлена другая проверенная обложка; замена запрещена.")
                else:
                    if target.exists():
                        raise FileExistsError("Файл новой иллюстрации уже существует.")
                    os.replace(temporary, target)
                    installed = True
                    conn.execute("INSERT INTO trip_cover_overrides "
                                 "(trip_id,account_key,file_name,mime_type,source_receipt,prompt_version,image_sha256,created_at) "
                                 "VALUES (?,?,?,?,?,?,?,?)", (trip_id, account_key, name, validated.mime_type,
                                                            source_receipt, prompt_version, digest, db._now()))
                    conn.execute("UPDATE trip_covers SET hidden=0 WHERE trip_id=? AND account_key=?",
                                 (trip_id, account_key))
                    committed = True
            # A failed COMMIT still needs to remove the unreferenced new file.
        except BaseException:
            committed = False
            raise
        finally:
            temporary.unlink(missing_ok=True)
            if installed and not committed:
                target.unlink(missing_ok=True)
        return self.get_status(tg_id, account_key, trip_id)

    def submit(self, tg_id, account_key, trip_id, *, access_grant=None):
        access_grant = access_guard.snapshot(tg_id, access_grant)
        with self.lock:
            acquired = False
            try:
                with db.connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    trip = _owned_trip(conn, tg_id, account_key, trip_id)
                    old = conn.execute("SELECT * FROM trip_covers WHERE trip_id=? AND account_key=?",
                                       (trip_id, account_key)).fetchone()
                    if old is not None:
                        return self._status(old)
                    if not enabled():
                        return self._status()
                    prompt = _prompt_for(trip)
                    if not prompt.generation_allowed:
                        return self._no_subject_status()
                    label, location = _prompt_caption(prompt)
                    if self.executor is None or self.closing:
                        raise CoverBusyError(MESSAGES["busy"])
                    digest = hashlib.sha256((prompt.cache_key + "\n" + yandex_art.configured_model_uri()).encode()).hexdigest()
                    active = conn.execute("SELECT COUNT(*) FROM trip_covers WHERE status IN ('queued','running')").fetchone()[0]
                    if active >= MAX_ACTIVE:
                        raise CoverBusyError(MESSAGES["busy"])
                    today = datetime.now(timezone.utc).date().isoformat()
                    used = conn.execute("SELECT COUNT(*) FROM cover_attempts WHERE utc_day=?", (today,)).fetchone()[0]
                    if used >= daily_limit():
                        raise CoverLimitError(MESSAGES["limit"])
                    acquired = self.slots.acquire(blocking=False)
                    if not acquired:
                        raise CoverBusyError(MESSAGES["busy"])
                    # Reserve before dispatch; even uncertain POSTs consume one attempt.
                    conn.execute("INSERT INTO cover_attempts(attempt_id,utc_day) VALUES (?,?)", (uuid4().hex, today))
                    conn.execute("INSERT INTO trip_covers "
                                 "(trip_id,account_key,cache_key,status,created_at,subject_label,subject_location) "
                                 "VALUES (?,?,?,'queued',?,?,?)",
                                 (trip_id, account_key, digest, db._now(), label, location))
            except BaseException:
                if acquired:
                    self.slots.release()
                raise
            try:
                self.executor.submit(self._run, tg_id, account_key, trip_id, prompt, access_grant)
            except BaseException:
                self.slots.release()
                self._finish(trip_id, account_key, "interrupted", "dispatch")
        return self.get_status(tg_id, account_key, trip_id)

    @staticmethod
    def _finish(trip_id, account_key, state, code):
        with db.connect() as conn:
            conn.execute("UPDATE trip_covers SET status=?,failure_code=?,finished_at=? "
                         "WHERE trip_id=? AND account_key=? AND status IN ('queued','running')",
                         (state, code, db._now(), trip_id, account_key))

    def _save_image(self, tg_id, account_key, trip_id, image):
        validated = yandex_art.validate_image(image.data)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        name = uuid4().hex + (".jpg" if validated.mime_type == "image/jpeg" else ".png")
        target = self._path(name)
        temporary = self.cache_dir / (uuid4().hex + ".tmp")
        committed = False
        try:
            with temporary.open("xb") as handle:
                handle.write(validated.data)
                handle.flush()
                os.fsync(handle.fileno())
            with db.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                _owned_trip(conn, tg_id, account_key, trip_id)
                os.replace(temporary, target)
                changed = conn.execute("UPDATE trip_covers SET status='ready',file_name=?,mime_type=?,finished_at=?,subject_image_sha256=? "
                                       "WHERE trip_id=? AND account_key=? AND status='running'",
                                       (name, validated.mime_type, db._now(), hashlib.sha256(validated.data).hexdigest(),
                                        trip_id, account_key)).rowcount
                if changed != 1:
                    raise CoverNotFoundError("Иллюстрация недоступна.")
            committed = True
        finally:
            temporary.unlink(missing_ok=True)
            if not committed:
                target.unlink(missing_ok=True)

    def _run(self, tg_id, account_key, trip_id, prompt, access_grant=None):
        try:
            with db.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                _owned_trip(conn, tg_id, account_key, trip_id)
                claimed = conn.execute("SELECT * FROM trip_covers WHERE trip_id=? AND account_key=?",
                                       (trip_id, account_key)).fetchone()
                expected_key = hashlib.sha256((prompt.cache_key + "\n" + yandex_art.configured_model_uri()).encode()).hexdigest()
                if (claimed is None or claimed["cache_key"] != expected_key
                        or (claimed["subject_label"], claimed["subject_location"]) != _prompt_caption(prompt)):
                    raise CoverStateError("Заявка иллюстрации изменилась.")
                changed = conn.execute("UPDATE trip_covers SET status='running' "
                                       "WHERE trip_id=? AND account_key=? AND status='queued'",
                                       (trip_id, account_key)).rowcount
            if changed != 1:
                return
            if self.closing or not enabled():
                self._finish(trip_id, account_key, "interrupted", "disabled")
                return
            with access_guard.operation(tg_id, expected_grant=access_grant):
                access_control.reserve_aux(tg_id, "cover")
                access_guard.check()
                operation = yandex_art.submit(prompt)
            # The current Images API responds synchronously. This compatibility
            # column contains a LOCAL receipt, never a pollable provider ID.
            # Persist it before saving the image; there is no second paid POST.
            with db.connect() as conn:
                _owned_trip(conn, tg_id, account_key, trip_id)
                changed = conn.execute("UPDATE trip_covers SET operation_id=? "
                                       "WHERE trip_id=? AND account_key=? AND status='running'",
                                       (operation.operation_id, trip_id, account_key)).rowcount
            if changed != 1:
                return
            if not operation.done or operation.image is None:
                raise yandex_art.YandexArtError("invalid_response", submission_uncertain=True)
            self._save_image(tg_id, account_key, trip_id, operation.image)
        except access_control.AccessDeniedError:
            self._finish(trip_id, account_key, "interrupted", "access")
        except yandex_art.YandexArtError as error:
            self._finish(trip_id, account_key, "interrupted" if error.submission_uncertain else "failed", error.code)
        except CoverNotFoundError:
            self._finish(trip_id, account_key, "interrupted", "account_change")
        except Exception:
            # Never log exception strings, provider responses, URLs or prompts.
            logging.getLogger("web.covers").error("cover_job_failed")
            try:
                self._finish(trip_id, account_key, "interrupted", "internal")
            except Exception:
                logging.getLogger("web.covers").error("cover_status_write_failed")
        finally:
            self.slots.release()


manager = TripCovers()
