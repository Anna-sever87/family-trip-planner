"""Bounded durable creation receipts; never resume paid work after a restart.

The single web process owns two in-flight slots, with no unbounded work queue.
Only a request digest is stored. The request itself lives in the worker closure.
On geographic ambiguity, at most five sanitized public place labels are kept
for owner-bound clarification, never the questionnaire or raw provider response.
Trip/receipt commit is atomic in db.save_trip_bundle. Old pending receipts are
interrupted on web startup, not automatically retried. Status is owner-bound
and contains no request, prompts, model output, private memory or token counts.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import threading
from uuid import UUID, uuid4

import db
import access_control
import access_guard
import web_service
from generation_quality import KEY as QUALITY_KEY, safe_coverage
from schemas import TripRequest
from yandex_ai import YandexAIError
from weather import safe_city_suggestions, safe_city_suggestion_details


MAX_WORKERS = 2
MESSAGES = {
    "queued": "Заявка принята. Можно закрыть страницу: состояние сохранено.",
    "running": "Маршрут составляется и проверяется. Можно закрыть страницу и вернуться позже.",
    "succeeded": "Результат сохранён. Откройте поездку и проверьте полноту программы. Повторный запуск не нужен.",
    "interrupted": "Сервер был перезапущен до подтверждения результата. Платная операция автоматически не повторялась. Проверьте список поездок перед новым запуском.",
    "account_change": "Обработка остановлена из-за изменения состояния аккаунта. Генерация по этой заявке не запускалась автоматически повторно. Проверьте список поездок перед новым запуском.",
    "validation": "Маршрут не прошёл обязательные проверки. Новая поездка не сохранена; повторная генерация автоматически не запускалась.",
    "validation_error": "Маршрут не прошёл обязательные проверки. Новая поездка не сохранена; повторная генерация автоматически не запускалась.",
    "city_ambiguous": "Найдено несколько одноимённых населённых пунктов. Выберите нужный вариант или уточните город, регион и страну. Создание маршрута не запускалось; попытка не списана.",
    "city_not_found": "Не удалось подтвердить город в указанной стране и регионе. Проверьте название направления. Создание маршрута не запускалось; попытка не списана.",
    "city_unavailable": "Сервис определения города временно недоступен или вернул некорректный ответ. Создание маршрута не запускалось; попытка не списана. Автоматического повтора нет.",
    "city_invalid": "Нужно уточнить направление в формате «город, регион, страна». Создание маршрута не запускалось; попытка не списана.",
    "insufficient_data": "Недостаточно подтверждённых данных для программы по этим условиям. Новая поездка не сохранена. Это не означает, что подходящих мест нет.",
    "search_limit": "Достигнут предел проверки доступных данных. Новая поездка не сохранена; автоматического повторного запуска нет.",
    "invalid_request": "Условия поездки требуют уточнения. Новая поездка не сохранена; повторная генерация автоматически не запускалась.",
    "unsupported_preferences": "Не удалось проверить выполнение пожеланий или близость к жилью по имеющимся данным. Уточните условия; маршрут не создан и операция не повторена.",
    "busy": "Другая операция уже выполняется. Эта заявка не запустила новую генерацию.",
    "limit": "Дневной лимит исчерпан. Эта заявка не запустила новую генерацию.",
    "access": "Доступ завершён, отозван или исчерпан лимит приглашения. Следующие платные обращения остановлены; уже отправленный запрос мог быть оплачен.",
    "provider": "Сервис генерации не завершил операцию. Автоматического повторного платного запуска нет; проверьте список поездок.",
    "provider_format": "ИИ не смог корректно оформить маршрут. Новая поездка не сохранена; повторная генерация автоматически не запускается.",
    "internal": "Не удалось подтвердить завершение операции. Автоматического повторного запуска нет; проверьте список поездок.",
    "deleted": "Ранее созданная поездка удалена. Повторная генерация не запускалась.",
}


class JobBusyError(RuntimeError):
    pass


class JobConflictError(ValueError):
    pass


class JobNotFoundError(LookupError):
    pass


def owner_token(account_key):
    """Opaque browser storage namespace; not an authentication credential."""
    return hashlib.sha256(("generation-storage-v1:" + account_key).encode()).hexdigest()


def _key(value):
    if not isinstance(value, str) or len(value) != 36:
        raise ValueError("Требуется корректный ключ заявки.")
    try:
        if str(UUID(value)) != value:
            raise ValueError
    except ValueError:
        raise ValueError("Требуется корректный ключ заявки.") from None
    return value


def _owner(conn, tg_id, account_key):
    if not isinstance(account_key, str) or not account_key or conn.execute(
        "SELECT 1 FROM users WHERE tg_id=? AND account_key=? AND deletion_token=''",
        (tg_id, account_key),
    ).fetchone() is None:
        raise JobNotFoundError("Заявка не найдена.")


def public_status(row, conn=None):
    state = row["status"]
    code = row["failure_code"] if state == "failed" else state
    if state == "interrupted" and row["failure_code"] == "account_change":
        code = "account_change"
    if state == "succeeded" and row["trip_id"] is None:
        code = "deleted"
    message = MESSAGES.get(code, MESSAGES["internal"])
    if conn is not None and state == "succeeded" and row["trip_id"] is not None:
        trip = conn.execute("SELECT request_json, itinerary_json FROM trips WHERE id=? AND tg_id=?",
                            (row["trip_id"], row["tg_id"])).fetchone()
        if trip is not None:
            try:
                quality = safe_coverage(json.loads(trip["request_json"]).get(QUALITY_KEY))
                days = json.loads(trip["itinerary_json"]).get("days", [])
                free = sum(isinstance(day, dict) and day.get("day_mode") == "free" for day in days)
                if free or quality.get("quality_status") == "partial":
                    message = ("Программа составлена частично и сохранена. Не все дни или выбранные интересы удалось заполнить. "
                               "Откройте поездку: там указаны ограничения. Повторный запуск не требуется.")
            except (ValueError, TypeError, AttributeError):
                pass  # Saved result remains available; do not expose damaged/private data.
    payload = {"job_id": row["job_id"], "status": state,
            "trip_id": row["trip_id"] if state == "succeeded" else None,
            "message": message}
    if state == "failed" and code in web_service.CITY_PREFLIGHT_CODES:
        options, details = (), ()
        if code == "city_ambiguous":
            raw = row["destination_options_json"] if "destination_options_json" in row.keys() else "[]"
            if isinstance(raw, str) and len(raw) <= 4096:
                try:
                    saved = json.loads(raw)
                    if isinstance(saved, dict) and set(saved) == {"version", "options", "details"} and type(saved["version"]) is int and saved["version"] == 1:
                        options = safe_city_suggestions(saved["options"])
                        details = safe_city_suggestion_details(saved["details"], options)
                    else:
                        options = safe_city_suggestions(saved)
                except (ValueError, TypeError):
                    pass
        payload["clarification"] = {"kind": "city", "options": list(options), "attempt_used": False}
        if details:
            payload["clarification"]["details"] = list(details)
    return payload


def get_job(tg_id, account_key, *, job_id=None, key=None):
    identifier = _key(job_id if job_id is not None else key)
    column = "job_id" if job_id is not None else "idempotency_key"
    with db.connect() as conn:
        _owner(conn, tg_id, account_key)
        row = conn.execute(f"SELECT * FROM generation_jobs WHERE {column}=? AND tg_id=? AND account_key=?",
                           (identifier, tg_id, account_key)).fetchone()
        if row is None:
            raise JobNotFoundError("Заявка не найдена.")
        return public_status(row, conn)


class GenerationJobs:
    def __init__(self, *, workers=MAX_WORKERS, executor_factory=ThreadPoolExecutor):
        if type(workers) is not int or not 1 <= workers <= MAX_WORKERS:
            raise ValueError("Invalid worker bound")
        self.workers = workers
        self.executor_factory = executor_factory
        self.executor = None
        self.closing = False
        self.clients = 0
        self.slots = threading.BoundedSemaphore(workers)
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if self.closing:
                raise JobBusyError("Обработка завершается. Автоматический перезапуск невозможен.")
            if self.executor is not None:
                self.clients += 1
                return
            with db.connect() as conn:
                conn.execute("UPDATE generation_jobs SET status=CASE WHEN trip_id IS NULL THEN 'interrupted' "
                             "ELSE 'succeeded' END, finished_at=? WHERE status IN ('queued','running')", (db._now(),))
            self.executor = self.executor_factory(max_workers=self.workers, thread_name_prefix="trip-generation")
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

    def submit(self, tg_id, account_key, key, request: TripRequest, *, access_grant=None):
        access_grant = access_guard.snapshot(tg_id, access_grant)
        key = _key(key)
        digest = hashlib.sha256(json.dumps(request.model_dump(mode="json"), ensure_ascii=False,
                                          sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        with self.lock:
            if self.executor is None:
                raise JobBusyError("Создание временно недоступно. Заявка не запущена.")
            acquired = False
            try:
                with db.connect() as conn:
                    conn.execute("BEGIN IMMEDIATE")
                    _owner(conn, tg_id, account_key)
                    old = conn.execute("SELECT * FROM generation_jobs WHERE account_key=? AND idempotency_key=?",
                                       (account_key, key)).fetchone()
                    if old is not None:
                        if old["request_digest"] != digest:
                            raise JobConflictError("Ключ уже принадлежит другой анкете. Новая операция не запущена.")
                        return public_status(old, conn)
                    # Replays remain valid after the original travel date passes.
                    normalized = web_service._normalized_trip_request(request)
                    if conn.execute("SELECT 1 FROM generation_jobs WHERE account_key=? AND status IN ('queued','running')",
                                    (account_key,)).fetchone() is not None:
                        raise JobBusyError("Заявка уже выполняется. Дождитесь её результата.")
                    acquired = self.slots.acquire(blocking=False)
                    if not acquired:
                        raise JobBusyError("Все места обработки заняты. Заявка не запущена и квота не расходована.")
                    job_id = str(uuid4())
                    conn.execute("INSERT INTO generation_jobs(job_id,tg_id,account_key,idempotency_key,request_digest,status,created_at) "
                                 "VALUES (?,?,?,?,?,'queued',?)", (job_id, tg_id, account_key, key, digest, db._now()))
            except BaseException:
                # Also release on transaction/commit failure, not just INSERT.
                if acquired:
                    self.slots.release()
                raise
            try:
                self.executor.submit(self._run, job_id, tg_id, account_key, normalized, access_grant)
            except BaseException:
                self.slots.release()
                self._finish(job_id, "interrupted", "")
                # The receipt exists: return it rather than claiming no acceptance.
                return get_job(tg_id, account_key, job_id=job_id)
        return get_job(tg_id, account_key, job_id=job_id)

    @staticmethod
    def _finish(job_id, state, code, destination_options=(), destination_details=()):
        options = safe_city_suggestions(destination_options) if state == "failed" and code == "city_ambiguous" else ()
        details = safe_city_suggestion_details(destination_details, options)
        saved = {"version": 1, "options": options, "details": details} if details else options
        with db.connect() as conn:
            # Atomic successful save wins over a later rendering/memory error.
            conn.execute("UPDATE generation_jobs SET status=?, failure_code=?, finished_at=?, destination_options_json=? "
                         "WHERE job_id=? AND status IN ('queued','running')",
                         (state, code, db._now(), json.dumps(saved, ensure_ascii=False), job_id))

    def _run(self, job_id, tg_id, account_key, request, access_grant=None):
        try:
            with db.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                _owner(conn, tg_id, account_key)
                changed = conn.execute("UPDATE generation_jobs SET status='running' "
                                       "WHERE job_id=? AND status='queued'", (job_id,)).rowcount
            if changed != 1:
                return
            try:
                with access_guard.operation(tg_id, expected_grant=access_grant):
                    web_service.create_trip(tg_id, request, expected_account_key=account_key, generation_job_id=job_id)
            except access_control.AccessDeniedError:
                self._finish(job_id, "failed", "access")
            except web_service.TripSavedResponseError:
                # The saved receipt, not the exception's untrusted text/ID, is authoritative.
                self._finish(job_id, "failed", "internal")
            except web_service.TripCreateValidationError as error:
                code = getattr(error, "code", "validation")
                self._finish(job_id, "failed", code if code in {
                    "insufficient_data", "search_limit", "invalid_request", "validation_error", "provider_format", "unsupported_preferences",
                    *web_service.CITY_PREFLIGHT_CODES,
                } else "validation", getattr(error, "destination_options", ()), getattr(error, "destination_details", ()))
            except web_service.AIDailyLimitError:
                self._finish(job_id, "failed", "limit")
            except web_service.EditBusyError:
                self._finish(job_id, "failed", "busy")
            except YandexAIError:
                self._finish(job_id, "failed", "provider")
            except Exception:
                self._finish(job_id, "failed", "internal")
            else:
                # Success requires the atomic persistence hook, not a returned dict.
                self._finish(job_id, "failed", "internal")
        except JobNotFoundError:
            # A deletion lease also hides an EXISTING account. If deletion
            # subsequently fails, do not leave a queued receipt without a worker.
            # A completed deletion cascades receipts, so this update is a no-op.
            try:
                self._finish(job_id, "interrupted", "account_change")
            except Exception:
                logging.getLogger("web.jobs").error("generation_job_status_write_failed")
        except Exception:
            logging.getLogger("web.jobs").error("generation_job_persistence_failed")
            try:
                self._finish(job_id, "failed", "internal")
            except Exception:
                # Database remains unavailable: startup will mark it interrupted.
                logging.getLogger("web.jobs").error("generation_job_status_write_failed")
        finally:
            self.slots.release()


manager = GenerationJobs()
