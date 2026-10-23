"""Bounded Yandex Cloud Images client. Never retries or logs paid requests.

One call to submit means exactly one synchronous POST. The returned receipt is
local, not a Yandex operation ID. The caller owns durable deduplication and billing
permission. The retired asynchronous Image Generation API is never contacted.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
from io import BytesIO
import json
import os
import re
import struct
import warnings
from uuid import uuid4

import httpx
from PIL import Image, UnidentifiedImageError

from cover_prompts import CoverPrompt


GENERATE_URL = "https://ai.api.cloud.yandex.net/v1/images/generations"
IMAGE_SIZE = "1536x1024"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_BASE64_CHARS = ((MAX_IMAGE_BYTES + 2) // 3) * 4
MAX_JSON_BYTES = MAX_BASE64_CHARS + 64 * 1024
MAX_DIMENSION = 4096
MAX_PIXELS = 16_000_000
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
_ERRORS = {
    "configuration": "Не настроен доступ к генерации иллюстраций Yandex Cloud.",
    "invalid_prompt": "Не удалось подготовить безопасный промпт обложки.",
    "invalid_operation": "Некорректный идентификатор операции изображения.",
    "network": "Сервис изображений не ответил. Новый платный запрос автоматически не отправляется.",
    "unauthorized": "Yandex Cloud отклонил ключ генерации изображений.",
    "forbidden": "Не хватает прав Yandex Cloud на генерацию изображений.",
    "rate_limited": "Достигнут лимит сервиса изображений Yandex Cloud.",
    "bad_request": "Сервис изображений отклонил параметры запроса.",
    "not_found": "Операция изображения не найдена или срок хранения истёк.",
    "redirect": "Неожиданное перенаправление сервиса изображений отклонено.",
    "provider_unavailable": "Сервис изображений временно недоступен.",
    "invalid_response": "Сервис изображений вернул некорректный ответ.",
    "response_too_large": "Ответ сервиса изображений превышает допустимый размер.",
    "invalid_image": "Сервис вернул повреждённое или неподдерживаемое изображение.",
    "operation_failed": "Yandex Cloud не смог создать иллюстрацию. Маршрут сохранён без неё.",
}


class YandexArtError(RuntimeError):
    """Only fixed messages and safe codes; never a provider response or URL."""

    def __init__(self, code: str, *, submission_uncertain: bool = False, status_code: int | None = None):
        self.code = code if code in _ERRORS else "invalid_response"
        self.submission_uncertain = bool(submission_uncertain)
        self.status_code = status_code if isinstance(status_code, int) and 100 <= status_code <= 599 else None
        super().__init__(_ERRORS[self.code])


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes = field(repr=False)
    mime_type: str
    width: int
    height: int


@dataclass(frozen=True)
class ArtOperation:
    """Compatibility result: operation_id is a local receipt, never pollable."""

    operation_id: str
    done: bool
    image: GeneratedImage | None = None


def configured_model_uri() -> str:
    """Public model identity suitable for cache isolation; never returns a key."""
    folder = os.getenv("YANDEX_FOLDER_ID", "").strip()
    if not _SAFE_ID.fullmatch(folder):
        raise YandexArtError("configuration")
    uri = os.getenv("YANDEX_ART_MODEL_URI", "").strip() or f"art://{folder}/aliceai-image-art-3.0"
    if not re.fullmatch(r"art://" + re.escape(folder) + r"/[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?", uri):
        raise YandexArtError("configuration")
    return uri


def _headers() -> dict[str, str]:
    key = (os.getenv("YANDEX_ART_API_KEY", "").strip()
           or os.getenv("YANDEX_API_KEY", "").strip())
    if not key or len(key) > 4096 or not key.isascii() or any(not 33 <= ord(char) <= 126 for char in key):
        raise YandexArtError("configuration")
    folder = os.getenv("YANDEX_FOLDER_ID", "").strip()
    if not _SAFE_ID.fullmatch(folder):
        raise YandexArtError("configuration")
    return {"Authorization": "Api-Key " + key, "Content-Type": "application/json",
            "OpenAI-Project": folder, "Accept": "application/json", "Accept-Encoding": "identity"}


def _request(method: str, url: str, payload: dict | None = None) -> dict:
    headers = _headers()
    uncertain = method == "POST"
    try:
        # No redirects, environmental proxy, HTTP retry or fallback endpoint.
        with httpx.Client(follow_redirects=False, trust_env=False,
                          timeout=httpx.Timeout(180.0, connect=10.0, write=10.0, pool=10.0)) as client:
            with client.stream(method, url, headers=headers, json=payload) as response:
                status = response.status_code
                if status != 200:
                    code = ({400: "bad_request", 401: "unauthorized", 403: "forbidden",
                             404: "not_found", 429: "rate_limited"}.get(status)
                            or ("redirect" if 300 <= status < 400 else "provider_unavailable"))
                    raise YandexArtError(code, submission_uncertain=uncertain and status >= 500,
                                         status_code=status)
                if (response.headers.get("content-encoding", "identity").lower() not in {"", "identity"}
                        or response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                        != "application/json"):
                    raise YandexArtError("invalid_response", submission_uncertain=uncertain)
                length = response.headers.get("content-length")
                if length is not None:
                    if not length.isdecimal():
                        raise YandexArtError("invalid_response", submission_uncertain=uncertain)
                    if len(length) > 10 or int(length) > MAX_JSON_BYTES:
                        raise YandexArtError("response_too_large", submission_uncertain=uncertain)
                raw = bytearray()
                for chunk in response.iter_bytes(chunk_size=64 * 1024):
                    if len(raw) + len(chunk) > MAX_JSON_BYTES:
                        raise YandexArtError("response_too_large", submission_uncertain=uncertain)
                    raw.extend(chunk)
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("not an object")
        return data
    except YandexArtError:
        raise
    except httpx.RequestError:
        raise YandexArtError("network", submission_uncertain=uncertain) from None
    except (ValueError, UnicodeError, RecursionError):
        raise YandexArtError("invalid_response", submission_uncertain=uncertain) from None


def validate_image(data: bytes) -> GeneratedImage:
    """Decode an in-memory bounded JPEG/PNG; preserve the original bytes."""
    if not isinstance(data, bytes) or not data or len(data) > MAX_IMAGE_BYTES:
        raise YandexArtError("invalid_image")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        if not data.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82"):
            raise YandexArtError("invalid_image")
    elif data.startswith(b"\xff\xd8\xff"):
        if not data.endswith(b"\xff\xd9"):
            raise YandexArtError("invalid_image")
    else:
        raise YandexArtError("invalid_image")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data), formats=("JPEG", "PNG")) as picture:
                width, height = picture.size
                if (not 1 <= width <= MAX_DIMENSION or not 1 <= height <= MAX_DIMENSION
                        or width * height > MAX_PIXELS or getattr(picture, "n_frames", 1) != 1):
                    raise YandexArtError("invalid_image")
                mime_type = "image/jpeg" if picture.format == "JPEG" else "image/png"
                picture.verify()
            # verify checks container integrity; load checks the compressed pixels.
            with Image.open(BytesIO(data), formats=("JPEG", "PNG")) as picture:
                picture.load()
    except YandexArtError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, EOFError, struct.error,
            Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise YandexArtError("invalid_image") from None
    return GeneratedImage(data, mime_type, width, height)


def _parse_image_result(data: dict) -> ArtOperation:
    """Accept a single inline image only; never fetch provider-supplied URLs."""
    if "error" in data:
        raise YandexArtError("operation_failed")
    images = data.get("data")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise YandexArtError("invalid_response", submission_uncertain=True)
    encoded = images[0].get("b64_json")
    if not isinstance(encoded, str) or not encoded:
        raise YandexArtError("invalid_response", submission_uncertain=True)
    if len(encoded) > MAX_BASE64_CHARS:
        raise YandexArtError("response_too_large", submission_uncertain=True)
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise YandexArtError("invalid_image", submission_uncertain=True) from None
    image = validate_image(image_bytes)
    return ArtOperation("local-" + uuid4().hex, True, image)


def submit(prompt: CoverPrompt) -> ArtOperation:
    """One paid creation attempt. A timeout is uncertain, never auto-retried."""
    if (not isinstance(prompt, CoverPrompt) or not isinstance(prompt.provider_prompt, str)
            or not prompt.provider_prompt.strip() or len(prompt.provider_prompt) > 500):
        raise YandexArtError("invalid_prompt")
    # n, response_format, output_format and stream are explicitly unsupported by
    # the current provider. Only its documented model/prompt/size are sent.
    payload = {"model": configured_model_uri(), "prompt": prompt.provider_prompt, "size": IMAGE_SIZE}
    return _parse_image_result(_request("POST", GENERATE_URL, payload))


def get_operation(operation_id: str) -> ArtOperation:
    """Compatibility guard: current Images results have no provider polling API.

    Old receipts must not trigger a retired endpoint, another paid POST, or a URL
    supplied by a provider. The application marks interrupted work explicitly.
    """
    raise YandexArtError("invalid_operation")
