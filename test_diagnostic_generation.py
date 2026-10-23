"""The opt-in diagnostic captures candidates without bypassing quota or budget."""

import json
import io
import os
import socket
import tempfile
import unittest
from contextlib import ExitStack, contextmanager, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from schemas import TripRequest
from tools import diagnose_generation as diagnostic


class DiagnosticGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "new-run"
        self.request = TripRequest(
            destination="Тестовый край, Россия", destination_kind="region",
            start_date="2099-06-01", end_date="2099-06-03",
            adults=2, children_ages=[], budget=30000, currency="RUB",
            pace="средний", start_time="09:00", end_time="19:00",
        )

    @contextmanager
    def _context(self, *, calls=1, claim_status="ok", failure=None, account=True):
        def generate(*args, **kwargs):
            for _ in range(calls):
                diagnostic.planner.generate_json(
                    [{"role": "user", "content": "SECRET_PRIVATE_PROMPT"}],
                    json_schema={"type": "object"}, temperature=0.2,
                )
            return SimpleNamespace(
                problems=[], attempts=calls, prompt_tokens=11 * calls,
                completion_tokens=13 * calls, total_tokens=24 * calls,
                weather={"city": "Тест", "source": "forecast", "days": []},
                weather_text="Прогноз погоды (Тест)",
                itinerary=SimpleNamespace(model_dump=lambda **_: {"title": "Synthetic", "days": []}),
            )
        with ExitStack() as stack:
            stack.enter_context(patch.object(socket.socket, "connect", side_effect=AssertionError("Network forbidden")))
            stack.enter_context(patch.dict(os.environ, {
                "YANDEX_FOLDER_ID": "SECRET_FOLDER", "YANDEX_MODEL_URI": "original-model-uri",
            }))
            stack.enter_context(patch.object(diagnostic.db, "get_account_key", return_value="SECRET_ACCOUNT" if account else None))
            claim = stack.enter_context(patch.object(diagnostic.db, "claim_ai_request", return_value=SimpleNamespace(
                status=claim_status, claim_id="SECRET_CLAIM" if claim_status == "ok" else None, used=4, limit=5,
            )))
            finish = stack.enter_context(patch.object(diagnostic.db, "finish_ai_request", return_value=True))
            stack.enter_context(patch.object(diagnostic.web_service, "_daily_ai_limit", return_value=5))
            completion = stack.enter_context(patch.object(diagnostic.planner, "generate_json",
                return_value=SimpleNamespace(text='{"candidate": "raw"}', input_tokens=11, output_tokens=13),
                side_effect=failure,
            ))
            plan = stack.enter_context(patch.object(diagnostic.planner, "generate_itinerary", side_effect=generate))
            yield SimpleNamespace(claim=claim, finish=finish, completion=completion, plan=plan)

    def _run(self, **kwargs):
        return diagnostic.run_diagnostic(self.request, 123, 6, self.output, **kwargs)

    def test_raw_usage_schema_and_synthetic_request_are_saved_without_private_context(self):
        with self._context(calls=3) as mocks:
            report = self._run(model="yandexgpt-5.1")
            self.assertEqual(os.environ["YANDEX_MODEL_URI"], "original-model-uri")
            self.assertIs(diagnostic.planner.generate_json, mocks.completion)
            mocks.plan.assert_called_once_with(self.request, max_attempts=3, memory_context="", ground_candidates=True)
            self.assertEqual(mocks.completion.call_count, 3)
            self.assertEqual(mocks.claim.call_args.kwargs["daily_limit"], 5)
            self.assertEqual(mocks.claim.call_args.kwargs["trip_id"], 6)
            mocks.finish.assert_called_once_with("SECRET_CLAIM", succeeded=True)
        self.assertEqual((report["status"], report["completion_calls_attempted"], report["total_tokens"]), ("succeeded", 3, 72))
        self.assertFalse(report["trip_saved"])
        self.assertEqual(report["weather"], {"city": "Тест", "source": "forecast", "days": []})
        self.assertEqual(report["weather_text"], "Прогноз погоды (Тест)")
        self.assertEqual(json.loads((self.output / "request.json").read_text(encoding="utf-8")), self.request.model_dump(mode="json"))
        for number in range(1, 4):
            self.assertEqual((self.output / f"call-{number:02d}-raw.txt").read_text(encoding="utf-8"), '{"candidate": "raw"}')
            self.assertTrue((self.output / f"call-{number:02d}-schema.json").exists())
        self.assertNotIn("SECRET", "".join(path.read_text(encoding="utf-8") for path in self.output.iterdir()))

    def test_fourth_completion_is_blocked_before_ai_client(self):
        with self._context(calls=4) as mocks:
            report = self._run()
            self.assertEqual(mocks.completion.call_count, 3)
            mocks.finish.assert_called_once_with("SECRET_CLAIM", succeeded=False)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_type"], "DiagnosticBudgetExceeded")
        self.assertTrue(report["blocked_extra_completion"])
        self.assertEqual(report["completion_calls_attempted"], 3)

    def test_candidate_first_city_is_explicit_and_does_not_change_default(self):
        import city_planner
        original = city_planner.ground_city_candidates
        self.request = self.request.model_copy(update={"destination_kind": "city", "destination": "Сочи, Россия"})
        with self._context(calls=2) as mocks:
            report = self._run(city_candidates_first=True)
            mocks.plan.assert_called_once_with(self.request, max_attempts=3, memory_context="",
                                               ground_candidates=False, city_candidates_first=True)
        self.assertIs(city_planner.ground_city_candidates, original)
        self.assertEqual(report["pipeline"], "city_candidates_first")
        self.assertFalse(report["trip_saved"])

    def test_candidate_first_region_fails_before_output_or_quota_claim(self):
        with self._context() as mocks, self.assertRaises(ValueError):
            self._run(city_candidates_first=True)
        mocks.claim.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_city_center_capture_keeps_only_real_public_context_and_restores_resolver(self):
        import city_planner
        center = {"name": "Сочи", "lat": 43.59699, "lon": 39.72477, "country_code": "RU",
                  "admin1": "Краснодарский край", "geoname_id": 123, "private": "SECRET_VALUE"}
        self.request = self.request.model_copy(update={"destination_kind": "city", "destination": "Сочи, Россия"})
        with self._context(calls=2) as mocks, patch.object(city_planner, "find_city", return_value=center) as resolver:
            generate = mocks.plan.side_effect

            def with_center(*args, **kwargs):
                self.assertIs(city_planner.find_city(self.request.destination), center)
                return generate(*args, **kwargs)

            mocks.plan.side_effect = with_center
            report = self._run(city_candidates_first=True)
            self.assertIs(city_planner.find_city, resolver)
            resolver.assert_called_once_with(self.request.destination)
        self.assertEqual(report["resolved_city"], {key: center[key] for key in
                         ("name", "lat", "lon", "country_code", "admin1")})
        self.assertNotIn("SECRET", (self.output / "report.json").read_text(encoding="utf-8"))

    def test_city_resolver_failure_does_not_fabricate_center_and_restores_patch(self):
        import city_planner
        self.request = self.request.model_copy(update={"destination_kind": "city", "destination": "Сочи, Россия"})
        with self._context() as mocks, patch.object(city_planner, "find_city", side_effect=ValueError("bad center")) as resolver:
            mocks.plan.side_effect = lambda *args, **kwargs: city_planner.find_city(self.request.destination)
            report = self._run(city_candidates_first=True)
            self.assertIs(city_planner.find_city, resolver)
        self.assertEqual(report["status"], "failed")
        self.assertNotIn("resolved_city", report)

    def test_busy_or_exhausted_quota_never_calls_planner(self):
        for status in ("busy", "limit"):
            with self.subTest(status=status), self._context(claim_status=status) as mocks:
                report = diagnostic.run_diagnostic(self.request, 123, 6, Path(self.temporary.name) / status)
                mocks.plan.assert_not_called()
                mocks.completion.assert_not_called()
                mocks.finish.assert_not_called()
                self.assertEqual((report["status"], report["reason"]), ("blocked", status))

    def test_completion_error_is_redacted_and_quota_is_finished_once(self):
        with self._context(failure=RuntimeError("SECRET_EXCEPTION_URL")) as mocks:
            report = self._run(model="yandexgpt-5.1")
            self.assertEqual(mocks.completion.call_count, 1)
            mocks.finish.assert_called_once_with("SECRET_CLAIM", succeeded=False)
            self.assertEqual(os.environ["YANDEX_MODEL_URI"], "original-model-uri")
        self.assertEqual(report["calls"][0]["error_type"], "RuntimeError")
        self.assertNotIn("SECRET", (self.output / "report.json").read_text(encoding="utf-8"))

    def test_existing_output_directory_blocks_even_quota_claim(self):
        self.output.mkdir()
        with self._context() as mocks, self.assertRaises(FileExistsError):
            self._run()
        mocks.claim.assert_not_called()
        mocks.plan.assert_not_called()

    def test_final_artifact_write_failure_marks_quota_failed_without_retry_or_http(self):
        original_write = diagnostic._write_json

        def fail_final_artifact(path, *args, **kwargs):
            if Path(path).name == "final-itinerary.json":
                raise OSError("offline simulated final artifact failure")
            return original_write(path, *args, **kwargs)

        with self._context(calls=1) as mocks, \
             patch.object(diagnostic, "_write_json", side_effect=fail_final_artifact), \
             patch.object(diagnostic.yandex_ai.httpx, "post", side_effect=AssertionError("No HTTP allowed")) as post:
            report = self._run()
            mocks.plan.assert_called_once()
            mocks.completion.assert_called_once()
            mocks.finish.assert_called_once_with("SECRET_CLAIM", succeeded=False)
            post.assert_not_called()
            self.assertIs(diagnostic.yandex_ai.httpx.post, post)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["error_type"], "OSError")
        self.assertEqual(report["completion_calls_attempted"], 1)
        self.assertFalse(report["trip_saved"])
        self.assertFalse((self.output / "final-itinerary.json").exists())
        self.assertTrue((self.output / "call-01-raw.txt").exists())
        saved_report = json.loads((self.output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_report["status"], "failed")
        self.assertTrue(saved_report["quota_finished"])

    def test_missing_account_cannot_create_a_new_profile_or_call_ai(self):
        with self._context(account=False) as mocks:
            report = self._run()
            mocks.claim.assert_not_called()
            mocks.plan.assert_not_called()
        self.assertEqual(report["reason"], "account_unavailable")

    def test_invalid_budget_is_rejected_before_any_output_or_claim(self):
        for budget in (0, 1, 4, True, 1.5):
            with self.subTest(budget=budget), self._context() as mocks, self.assertRaises(ValueError):
                self._run(max_attempts=budget)
            mocks.claim.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_cli_rejects_one_attempt_before_request_or_quota_access(self):
        with self._context() as mocks, redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            diagnostic.main([
                "--source-trip", "6", "--output-dir", str(self.output),
                "--request-file", "not-read-request.json",
                "--max-attempts", "1", "--run-once",
            ])
        self.assertEqual(stopped.exception.code, 2)
        mocks.claim.assert_not_called()
        mocks.plan.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_http_400_saves_only_sanitized_message_and_restores_post(self):
        httpx = diagnostic.yandex_ai.httpx
        secret_values = {
            "YANDEX_API_KEY": "test-api-key-123456",
            "SOME_TOKEN": "test-token-123456",
            "some_password": "test-password-123456",
        }
        response = httpx.Response(400, json={
            "error": {"message": "Invalid schema: " + " ".join(secret_values.values())
                      + " SECRET_FOLDER HTTPS://example.invalid/private?q=value gpt://folder/model"},
            "request": {"prompt": "SECRET_PRIVATE_PROMPT"},
            "debug": "SECRET_DEBUG_PAYLOAD",
        }, headers={"Authorization": "SECRET_HEADER"}, request=httpx.Request("POST", "https://example.invalid"))

        def completion(*args, **kwargs):
            diagnostic.yandex_ai.httpx.post("https://example.invalid", headers={"Authorization": "SECRET_HEADER"}).raise_for_status()

        with self._context(failure=completion) as mocks, patch.dict(os.environ, secret_values):
            with patch.object(httpx, "post", return_value=response) as post:
                report = self._run()
                self.assertIs(httpx.post, post)
                post.assert_called_once()
                mocks.finish.assert_called_once_with("SECRET_CLAIM", succeeded=False)
        captured = json.loads((self.output / "call-01-error.json").read_text(encoding="utf-8"))
        self.assertEqual(set(captured), {"http_status", "provider_message"})
        self.assertEqual(captured["http_status"], 400)
        self.assertIn("Invalid schema", captured["provider_message"])
        self.assertIn("[redacted]", captured["provider_message"])
        self.assertEqual(captured["provider_message"].count("[redacted-url]"), 2)
        self.assertEqual(report["status"], "failed")
        all_output = "".join(path.read_text(encoding="utf-8") for path in self.output.iterdir())
        for value in (*secret_values.values(), "SECRET", "example.invalid", "gpt://folder"):
            self.assertNotIn(value, all_output)

    def test_non_string_provider_message_does_not_capture_nested_payload(self):
        httpx = diagnostic.yandex_ai.httpx
        response = httpx.Response(400, json={"message": {"prompt": "SECRET_PRIVATE_PROMPT"}},
                                  request=httpx.Request("POST", "https://example.invalid"))

        def completion(*args, **kwargs):
            diagnostic.yandex_ai.httpx.post("https://example.invalid").raise_for_status()

        with self._context(failure=completion), patch.object(httpx, "post", return_value=response) as post:
            self._run()
            self.assertIs(httpx.post, post)
        captured = json.loads((self.output / "call-01-error.json").read_text(encoding="utf-8"))
        self.assertEqual(captured, {"http_status": 400, "provider_message": ""})

    def test_http_post_is_restored_after_transport_exception(self):
        httpx = diagnostic.yandex_ai.httpx

        def completion(*args, **kwargs):
            diagnostic.yandex_ai.httpx.post("https://example.invalid")

        with self._context(failure=completion) as mocks:
            with patch.object(httpx, "post", side_effect=httpx.ReadTimeout("SECRET_TRANSPORT_ERROR")) as post:
                report = self._run()
                self.assertIs(httpx.post, post)
                post.assert_called_once()
                mocks.finish.assert_called_once_with("SECRET_CLAIM", succeeded=False)
        self.assertEqual(report["error_type"], "ReadTimeout")
        self.assertFalse((self.output / "call-01-error.json").exists())
        self.assertNotIn("SECRET", (self.output / "report.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
