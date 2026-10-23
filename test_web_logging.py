"""Read-only web rejection logging: fixed categories, never diagnostic text."""

import asyncio
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import web_app
import web_service
from generation_feedback import generation_failure_message
from test_generation_feedback import RAIN_DAY_PROBLEM


class WebRejectionLoggingTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(web_service.db, "connect",
                                             side_effect=AssertionError("No database")))
        self.stack.enter_context(patch.object(web_service, "generate_itinerary",
                                             side_effect=AssertionError("No AI")))
        self.stack.enter_context(patch.object(web_service.vector_memory, "recall_family_memories",
                                             side_effect=AssertionError("No private memory")))
        self.secrets = ("SECRET_OWNER_93822", "SECRET_MEMORY_BAIKAL", "SECRET_PROMPT",
                        "Bearer-secret-access-token", "https://private.invalid/key=secret")

    def capture(self, error, stage="create_validation"):
        with self.assertLogs(web_app.logger, level="WARNING") as captured:
            web_app._log_validation_rejection(error, stage=stage)
        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        self.assertIsNone(record.exc_info)
        self.assertIsNone(record.stack_info)
        for secret in self.secrets:
            self.assertNotIn(secret, str(record.__dict__))
        return record

    def test_create_handler_logs_category_without_diagnostic_text_or_exception_message(self):
        diagnostics = [RAIN_DAY_PROBLEM.replace("Музей истории Сочи", " ".join(self.secrets)),
                       "place_query: " + self.secrets[1]]
        error = web_service.TripCreateValidationError(
            generation_failure_message(diagnostics), diagnostic_problems=diagnostics,
        )
        error.attempts = 3
        error.memory_context = self.secrets[1]
        error.raw_completion = self.secrets[2]
        with patch.object(web_app.anyio.to_thread, "run_sync", AsyncMock(side_effect=error)) as execute, \
                self.assertLogs(web_app.logger, level="WARNING") as captured, \
                self.assertRaises(web_app.HTTPException) as raised:
            asyncio.run(web_app.create_trip(
                web_app.TripCreateRequest(destination="Тестгород, Россия", start_date="2030-01-01",
                                          end_date="2030-01-01", adults=2, budget=30000.0),
                SimpleNamespace(state=SimpleNamespace(account_key=self.secrets[0])), tg_id=93822,
            ))
        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail, str(error))
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(len(captured.records), 1)
        record = captured.records[0]
        self.assertEqual((record.reason_code, record.problem_count, record.validation_stage, record.attempts),
                         ("RAIN_PROGRAM", 2, "create_validation", 3))
        for forbidden in (*self.secrets, "Тестгород", "place_query", "93822"):
            self.assertNotIn(forbidden, str(record.__dict__))
        self.assertIsNone(record.exc_info)

    def test_edit_handler_logs_safe_metadata_and_keeps_public_response(self):
        error = web_service.EditRejectedError(["duration_min " + " ".join(self.secrets)])
        error.attempts = 2
        with patch.object(web_app.anyio.to_thread, "run_sync", AsyncMock(side_effect=error)) as execute, \
                self.assertLogs(web_app.logger, level="WARNING") as captured, \
                self.assertRaises(web_app.HTTPException) as raised:
            asyncio.run(web_app.edit_day(93567, 2, web_app.DayEditRequest(instruction=self.secrets[2]), tg_id=93822))
        self.assertEqual(raised.exception.status_code, 422)
        self.assertEqual(raised.exception.detail["problems"], error.problems)
        self.assertIn("Старый маршрут сохранён", raised.exception.detail["message"])
        self.assertEqual(execute.call_count, 1)
        record = captured.records[0]
        self.assertEqual((record.reason_code, record.validation_stage, record.attempts),
                         ("TIME_BUDGET", "edit_validation", 2))
        for forbidden in (*self.secrets, "93567", "93822", "duration_min"):
            self.assertNotIn(forbidden, str(record.__dict__))
        self.assertIsNone(record.exc_info)

    def test_existing_source_outage_rain_precedence_is_preserved(self):
        rain = RAIN_DAY_PROBLEM.replace("Музей истории Сочи", self.secrets[2])
        cases = [
            ([rain, "place_query"], "RAIN_PROGRAM"),
            ([rain, "Сервис карты временно недоступен"], "MAP_UNAVAILABLE"),
            ([rain, "Сервис карты временно недоступен", "SOURCE_PLAN_UNAVAILABLE: SOURCE_ROUTES_UNCONFIRMED"],
             "SOURCE_ROUTES_UNCONFIRMED"),
            ([rain, "SOURCE_PLAN_UNAVAILABLE: SOURCE_NOT_KNOWN " + self.secrets[1]], "SOURCE_UNKNOWN"),
            (["SOURCE_PLAN_UNAVAILABLE: SOURCE_CANDIDATE_CAPACITY",
              "SOURCE_PLAN_UNAVAILABLE: SOURCE_SEARCH_LIMIT"], "SOURCE_SEARCH_LIMIT"),
        ]
        for diagnostics, expected in cases:
            for values in (diagnostics, list(reversed(diagnostics))):
                with self.subTest(expected=expected, reversed=values != diagnostics):
                    error = web_service.TripCreateValidationError(self.secrets[0], diagnostic_problems=values)
                    self.assertEqual(self.capture(error).reason_code, expected)

    def test_source_counts_remain_numeric_classifier_data_not_raw_logged_fragments(self):
        for code in ("SOURCE_SEARCH_LIMIT", "SOURCE_ROUTES_UNCONFIRMED", "SOURCE_VISIT_DATA_INCOMPLETE",
                     "SOURCE_CANDIDATE_CAPACITY", "SOURCE_CONSTRAINTS_INCOMPATIBLE"):
            error = web_service.TripCreateValidationError(self.secrets[0], diagnostic_problems=[
                f"SOURCE_PLAN_UNAVAILABLE: {code} requestedDays=7 usableCards=6 " + self.secrets[4],
            ])
            with self.subTest(code=code):
                record = self.capture(error)
                self.assertEqual(record.reason_code, code)
                self.assertNotIn("requestedDays", record.getMessage())
                self.assertNotIn("usableCards", record.getMessage())

    def test_unknown_text_missing_diagnostics_and_invalid_items_cannot_leak(self):
        class SecretObject:
            def __str__(self):
                raise AssertionError("Do not stringify arbitrary diagnostics")

        for diagnostics in ([], [" ".join(self.secrets)], [None, SecretObject()], self.secrets[0], None):
            error = ValueError(" ".join(self.secrets))
            error.diagnostic_problems = diagnostics
            with self.subTest(kind=type(diagnostics).__name__):
                record = self.capture(error)
                self.assertEqual(record.reason_code, "CONSTRAINTS")
                self.assertEqual(record.problem_count, len(diagnostics) if isinstance(diagnostics, (list, tuple)) else 0)

    def test_attempts_are_optional_strict_bounded_integers(self):
        for value, expected in ((None, None), (0, 0), (3, 3), (100, 100), (True, None),
                                (-1, None), (101, None), (3.0, None), (self.secrets[3], None)):
            error = web_service.TripCreateValidationError(self.secrets[0])
            error.attempts = value
            with self.subTest(value=value):
                record = self.capture(error)
                self.assertEqual(record.attempts, expected)
                if expected is None:
                    self.assertIn("attempts=unknown", record.getMessage())

    def test_stage_and_unexpected_classifier_output_are_allowlisted(self):
        error = web_service.TripCreateValidationError(self.secrets[0])
        with patch.object(web_app, "generation_failure_reason", return_value=" ".join(self.secrets)):
            record = self.capture(error, stage=self.secrets[1])
        self.assertEqual(record.reason_code, "CONSTRAINTS")
        self.assertEqual(record.validation_stage, "validation")

    def test_successful_route_is_not_logged_as_rejected(self):
        expected = {"id": 9}
        with patch.object(web_app.anyio.to_thread, "run_sync", AsyncMock(return_value=expected)) as execute, \
                patch.object(web_app.logger, "warning") as warning:
            actual = asyncio.run(web_app.create_trip(
                web_app.TripCreateRequest(destination="Тестгород, Россия", start_date="2030-01-01",
                                          end_date="2030-01-01", adults=2, budget=30000.0),
                SimpleNamespace(state=SimpleNamespace(account_key="synthetic")), tg_id=93822,
            ))
        self.assertEqual(actual, expected)
        self.assertEqual(execute.call_count, 1)
        warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
