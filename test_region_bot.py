"""Офлайн-регрессии региональной анкеты Telegram.

bot.py не импортируется: при импорте он требует токен и
инициализирует внешние зависимости. Структура обработчиков
проверяется через AST, а parser и formatter — обычными вызовами.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import anketa
from formatting import format_itinerary
from schemas import (
    Itinerary,
    ItineraryDay,
    ItineraryItem,
    TripRequest,
)


BASE_DIR = Path(__file__).resolve().parent
BOT_SOURCE = (BASE_DIR / "bot.py").read_text(
    encoding="utf-8"
)
BOT_TREE = ast.parse(BOT_SOURCE)


def _function_source(name: str) -> str:
    node = next(
        item
        for item in BOT_TREE.body
        if isinstance(
            item,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        )
        and item.name == name
    )
    return ast.get_source_segment(
        BOT_SOURCE,
        node,
    ) or ""


def _questions() -> dict:
    node = next(
        item
        for item in BOT_TREE.body
        if isinstance(item, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "QUESTIONS"
            for target in item.targets
        )
    )
    return ast.literal_eval(node.value)


class DestinationKindParserTests(unittest.TestCase):
    def test_buttons_and_short_answers_map_to_schema_values(
        self,
    ) -> None:
        cases = {
            "Город / место": "city",
            "  ГОРОД  ": "city",
            "место": "city",
            "Регион / озеро": "region",
            "регион": "region",
            "ОЗЕРО": "region",
        }

        for answer, expected in cases.items():
            with self.subTest(answer=answer):
                self.assertEqual(
                    anketa.parse_destination_kind(answer),
                    expected,
                )

    def test_unknown_kind_is_not_silently_guessed(
        self,
    ) -> None:
        for answer in ("", "не знаю", "Байкал"):
            with self.subTest(answer=answer):
                with self.assertRaises(ValueError):
                    anketa.parse_destination_kind(answer)

    def test_legacy_request_still_defaults_to_city(
        self,
    ) -> None:
        request = TripRequest(
            destination="Казань, Россия",
            start_date="2099-06-01",
            end_date="2099-06-01",
            budget=50_000,
        )

        self.assertEqual(
            request.destination_kind,
            "city",
        )


class TelegramFlowSourceTests(unittest.TestCase):
    def test_questions_explain_city_and_region_destinations(
        self,
    ) -> None:
        questions = _questions()

        self.assertIn(
            "destination_kind",
            questions,
        )
        self.assertIn(
            "регион",
            questions["destination_kind"].casefold(),
        )
        self.assertIn(
            "озер",
            questions["destination"].casefold(),
        )
        self.assertIn(
            "Байкал, Россия",
            questions["destination"],
        )

    def test_plan_starts_with_kind_then_destination(
        self,
    ) -> None:
        plan_source = _function_source(
            "handle_plan"
        )
        answer_source = _function_source(
            "handle_answer"
        )
        ask_source = _function_source("ask")

        self.assertIn(
            '"step": "destination_kind"',
            plan_source,
        )
        self.assertIn(
            '"destination_kind"',
            ask_source,
        )
        self.assertIn(
            "destination_kind_keyboard()",
            ask_source,
        )
        self.assertIn(
            "anketa.parse_destination_kind",
            answer_source,
        )
        self.assertIn(
            'next_step = "destination"',
            answer_source,
        )

    def test_kind_keyboard_is_removed_and_cancel_still_clears_state(
        self,
    ) -> None:
        answer_source = _function_source(
            "handle_answer"
        )
        cancel_source = _function_source(
            "handle_cancel"
        )

        self.assertIn(
            '"destination_kind",',
            answer_source,
        )
        self.assertIn(
            "types.ReplyKeyboardRemove()",
            answer_source,
        )
        self.assertIn(
            "states.pop",
            cancel_source,
        )
        self.assertIn(
            "types.ReplyKeyboardRemove()",
            cancel_source,
        )


class RegionalFormattingTests(unittest.TestCase):
    def test_formatter_shows_base_and_transfer_note(
        self,
    ) -> None:
        itinerary = Itinerary(
            title="Байкал",
            summary="Поездка по региону.",
            days=[
                ItineraryDay(
                    day=1,
                    date="2099-11-10",
                    theme="Листвянка",
                    base_area="Листвянка",
                    transfer_note=(
                        "Из Иркутска около 1,5 часа."
                    ),
                    items=[
                        ItineraryItem(
                            time="10:00",
                            title="Байкальский музей",
                            description="Знакомство с озером.",
                        )
                    ],
                )
            ],
        )

        text = format_itinerary(
            itinerary,
            "RUB",
            50_000,
        )

        self.assertIn(
            "📍 База дня: Листвянка",
            text,
        )
        self.assertIn(
            "🚗 Перемещения: Из Иркутска",
            text,
        )

    def test_legacy_day_without_regional_fields_still_formats(
        self,
    ) -> None:
        item = SimpleNamespace(
            time="12:00",
            kind="rest",
            title="Отдых",
            area="",
            cost=0,
            description="Спокойный перерыв.",
            kids_note="",
            verified=False,
        )
        day = SimpleNamespace(
            day=1,
            date="2099-06-01",
            theme="Центр города",
            items=[item],
            rainy_alternative="",
        )
        itinerary = SimpleNamespace(
            title="Город",
            summary="Обычный маршрут.",
            days=[day],
            tips=[],
        )

        text = format_itinerary(
            itinerary,
            "RUB",
            10_000,
        )

        self.assertIn(
            "📅 День 1",
            text,
        )
        self.assertNotIn(
            "База дня:",
            text,
        )
        self.assertNotIn(
            "Переезд:",
            text,
        )


if __name__ == "__main__":
    unittest.main()
