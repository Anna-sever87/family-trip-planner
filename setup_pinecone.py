import os
from pathlib import Path

from dotenv import load_dotenv, set_key
from pinecone import Pinecone, ServerlessSpec


ENV_PATH = Path(__file__).resolve().parent / ".env"

INDEX_NAME = "family-trip-memory-yandex-256"
INDEX_DIMENSION = 256
INDEX_METRIC = "cosine"


def normalized_metric(value):
    value = getattr(value, "value", value)
    return str(value).lower()


def verify_index(client):
    description = client.describe_index(INDEX_NAME)

    dimension = getattr(description, "dimension", None)
    metric = normalized_metric(
        getattr(description, "metric", "")
    )

    if dimension != INDEX_DIMENSION:
        raise RuntimeError(
            f"У индекса размерность {dimension}, "
            f"а требуется {INDEX_DIMENSION}"
        )

    if metric != INDEX_METRIC:
        raise RuntimeError(
            f"У индекса метрика {metric}, "
            f"а требуется {INDEX_METRIC}"
        )

    return description


def main():
    if not ENV_PATH.exists():
        raise RuntimeError(
            f"Файл .env не найден: {ENV_PATH}"
        )

    load_dotenv(ENV_PATH, override=True)

    api_key = os.getenv(
        "PINECONE_API_KEY",
        "",
    ).strip()

    if not api_key:
        raise RuntimeError(
            "В .env не заполнен PINECONE_API_KEY"
        )

    client = Pinecone(api_key=api_key)

    if client.has_index(INDEX_NAME):
        verify_index(client)

        print(
            f"[OK] Индекс {INDEX_NAME} "
            "уже существует и подходит проекту"
        )
    else:
        print(
            f"Создаю индекс {INDEX_NAME}..."
        )

        client.create_index(
            name=INDEX_NAME,
            dimension=INDEX_DIMENSION,
            metric=INDEX_METRIC,
            vector_type="dense",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
            deletion_protection="disabled",
            timeout=180,
        )

        verify_index(client)

        print(
            f"[OK] Индекс {INDEX_NAME} создан"
        )

    result = set_key(
        dotenv_path=ENV_PATH,
        key_to_set="PINECONE_INDEX_NAME",
        value_to_set=INDEX_NAME,
        quote_mode="never",
        encoding="utf-8",
    )

    if result[0] is not True:
        raise RuntimeError(
            "Не удалось обновить PINECONE_INDEX_NAME "
            "в файле .env"
        )

    print(
        "[OK] В .env автоматически изменён "
        "PINECONE_INDEX_NAME"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"[ОШИБКА] "
            f"{type(error).__name__}: {error}"
        )
        raise SystemExit(1)
