"""Однократно добавляет безопасные локальные настройки веб-кабинета в .env."""

from __future__ import annotations

import secrets
from pathlib import Path


ENV_PATH = Path(__file__).with_name(".env")


def main() -> None:
    if not ENV_PATH.exists():
        raise SystemExit("Файл .env не найден.")

    original = ENV_PATH.read_text(encoding="utf-8")
    lines = original.splitlines()
    values = {}
    for line in lines:
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()

    additions = []
    defaults = {
        "WEB_BASE_URL": "http://127.0.0.1:8000",
        "WEB_LOGIN_TOKEN_MINUTES": "15",
        "WEB_SESSION_HOURS": "24",
        "WEB_COOKIE_SECURE": "false",
    }
    for key, value in defaults.items():
        if not values.get(key):
            additions.append(f"{key}={value}")

    saved_secret = values.get("WEB_SESSION_SECRET", "")
    if (
        len(saved_secret) < 32
        or saved_secret.casefold().startswith(("change-me", "change_me", "случайная_"))
    ):
        additions.append(
            "WEB_SESSION_SECRET=" + secrets.token_urlsafe(48)
        )

    if additions:
        separator = "" if original.endswith(("\n", "\r")) else "\n"
        block = "\n# Веб-кабинет\n" + "\n".join(additions) + "\n"
        ENV_PATH.write_text(original + separator + block, encoding="utf-8")
        print("[OK] Настройки веб-кабинета добавлены в .env. Секрет не выводился.")
    else:
        print("[OK] Настройки веб-кабинета уже заполнены.")


if __name__ == "__main__":
    main()
