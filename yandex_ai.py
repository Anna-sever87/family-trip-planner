"""Клиент для генерации структурированных ответов через YandexGPT."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any

import httpx
import access_guard

logger = logging.getLogger("yandex_ai")


API_URL = (
    "https://ai.api.cloud.yandex.net/"
    "foundationModels/v1/completion"
)

class YandexAIError(RuntimeError):
    """Понятная ошибка при работе с Yandex AI Studio."""


@dataclass(frozen=True)
class GenerationResult:
    """Текст ответа модели и статистика токенов."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise YandexAIError(
            f"Не заполнена переменная {name} "
            "в файле .env."
        )

    return value


def _as_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _http_error_message(
    status_code: int,
) -> str:
    hints = {
        400: (
            "Проверьте URI модели, параметры "
            "и JSON-схему запроса."
        ),
        401: (
            "API-ключ недействителен или истёк."
        ),
        403: (
            "Сервисному аккаунту не хватает "
            "прав для YandexGPT."
        ),
        429: (
            "Превышена квота запросов. "
            "Попробуйте немного позже."
        ),
    }

    hint = hints.get(
        status_code,
        "Сервис временно недоступен.",
    )

    return (
        "Yandex AI Studio вернул "
        f"HTTP {status_code}. {hint}"
    )


def _send_request(
    payload: dict[str, Any],
    headers: dict[str, str],
) -> httpx.Response:
    """Одна отправка на логическую генерацию, без скрытого платного повтора.

    После таймаута нельзя установить, обработал ли провайдер первый запрос.
    Повтором управляет пользователь, а не транспортный слой.
    """
    access_guard.check()
    logger.info("Yandex AI: отправка одного HTTP-запроса; автоматический повтор отключён")
    try:
        response = httpx.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=httpx.Timeout(90.0, connect=10.0),
        )
    except httpx.RequestError as error:
        # Не пишем URL/headers/body/str(error): они могут содержать ключи.
        logger.warning("Yandex AI: сетевая ошибка %s; запрос не повторён", type(error).__name__)
        message = (
            "Yandex AI Studio не ответил вовремя. "
            "Автоматический повтор отключён, чтобы не отправить оплачиваемый запрос дважды."
            if isinstance(error, httpx.ReadTimeout) else
            "Не удалось подключиться к Yandex AI Studio. Запрос не повторялся автоматически."
        )
        raise YandexAIError(message) from None

    logger.info("Yandex AI: получен HTTP %s", response.status_code)
    if response.is_error:
        raise YandexAIError(_http_error_message(response.status_code))
    return response


def generate_json(
    messages: list[dict[str, str]],
    temperature: float = 0.3,
    json_schema: dict[str, Any] | None = None,
) -> GenerationResult:
    """Возвращает JSON, при наличии схемы — строго по ней."""

    api_key = _required_env(
        "YANDEX_API_KEY"
    )
    folder_id = _required_env(
        "YANDEX_FOLDER_ID"
    )

    model_uri = os.getenv(
        "YANDEX_MODEL_URI",
        (
            f"gpt://{folder_id}/"
            "yandexgpt-5-lite"
        ),
    ).strip()

    try:
        max_tokens = int(
            os.getenv(
                "YANDEX_MAX_TOKENS",
                "8000",
            )
        )
    except ValueError:
        raise YandexAIError(
            "YANDEX_MAX_TOKENS должен быть "
            "целым числом."
        ) from None

    if max_tokens <= 0:
        raise YandexAIError(
            "YANDEX_MAX_TOKENS должен быть "
            "больше нуля."
        )

    if not 0 <= temperature <= 1:
        raise YandexAIError(
            "Температура YandexGPT должна "
            "быть от 0 до 1."
        )

    if json_schema is not None:
        if not isinstance(json_schema, dict):
            raise YandexAIError(
                "JSON-схема должна быть объектом."
            )

        if json_schema.get("type") != "object":
            raise YandexAIError(
                "Корнем JSON-схемы должен "
                "быть объект."
            )

    prepared_messages: list[
        dict[str, str]
    ] = []

    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role not in {
            "system",
            "user",
            "assistant",
        }:
            raise YandexAIError(
                "Неизвестная роль сообщения: "
                f"{role!r}."
            )

        if (
            not isinstance(content, str)
            or not content.strip()
        ):
            raise YandexAIError(
                "Текст сообщения для YandexGPT "
                "не должен быть пустым."
            )

        prepared_messages.append(
            {
                "role": role,
                "text": content,
            }
        )

    payload: dict[str, Any] = {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": False,
            "temperature": temperature,
            "maxTokens": str(max_tokens),
        },
        "messages": prepared_messages,
    }

    if json_schema is None:
        payload["jsonObject"] = True
    else:
        payload["jsonSchema"] = {
            "schema": json_schema,
        }

    headers = {
        "Authorization": (
            f"Api-Key {api_key}"
        ),
        "Content-Type": "application/json",
        "x-folder-id": folder_id,
    }

    response = _send_request(
        payload,
        headers,
    )

    try:
        data = response.json()
    except ValueError:
        raise YandexAIError(
            "Yandex AI Studio вернул ответ "
            "не в формате JSON."
        ) from None

    if not isinstance(data, dict):
        raise YandexAIError(
            "Yandex AI Studio вернул "
            "неожиданный формат ответа."
        )

    result = data.get(
        "result",
        data,
    )

    if not isinstance(result, dict):
        raise YandexAIError(
            "В ответе YandexGPT "
            "отсутствует результат."
        )

    alternatives = result.get(
        "alternatives",
        [],
    )

    if (
        not isinstance(alternatives, list)
        or not alternatives
    ):
        raise YandexAIError(
            "YandexGPT не вернул "
            "ни одного варианта ответа."
        )

    alternative = alternatives[0]

    if not isinstance(alternative, dict):
        raise YandexAIError(
            "YandexGPT вернул некорректный "
            "вариант ответа."
        )

    status = str(
        alternative.get("status", "")
    )

    if "CONTENT_FILTER" in status:
        raise YandexAIError(
            "Ответ YandexGPT остановлен "
            "фильтром безопасности."
        )

    if "TRUNCATED" in status:
        raise YandexAIError(
            "Ответ YandexGPT обрезан. "
            "Нужно увеличить "
            "YANDEX_MAX_TOKENS."
        )

    if status and status not in {
        "FINAL",
        "ALTERNATIVE_STATUS_FINAL",
    }:
        raise YandexAIError(
            "YandexGPT вернул незавершённый "
            f"ответ: {status}."
        )

    message = alternative.get(
        "message",
        {},
    )

    if not isinstance(message, dict):
        raise YandexAIError(
            "В ответе YandexGPT "
            "отсутствует сообщение."
        )

    text = message.get(
        "text",
        "",
    )

    if (
        not isinstance(text, str)
        or not text.strip()
    ):
        raise YandexAIError(
            "YandexGPT вернул пустой текст."
        )

    usage = result.get(
        "usage",
        {},
    )

    if not isinstance(usage, dict):
        usage = {}

    return GenerationResult(
        text=text,
        input_tokens=_as_int(
            usage.get("inputTextTokens")
        ),
        output_tokens=_as_int(
            usage.get("completionTokens")
        ),
    )
