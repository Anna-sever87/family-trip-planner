"""No real network: one logical generation must send at most one HTTP request."""

import os
import unittest
from unittest.mock import patch

import httpx
import yandex_ai


class YandexRequestBoundsTests(unittest.TestCase):
    def test_success_is_sent_once_and_logs_no_payload_or_key(self):
        response = httpx.Response(200, json={"result": {}})
        with patch.object(yandex_ai.httpx, "post", return_value=response) as post, self.assertLogs("yandex_ai", level="INFO") as captured:
            result = yandex_ai._send_request({"private": "secret-payload"}, {"Authorization": "secret-key"})
        self.assertIs(result, response)
        post.assert_called_once()
        self.assertIn("HTTP 200", "\n".join(captured.output))
        self.assertNotIn("secret", "\n".join(captured.output))

    def test_error_status_never_retries_even_when_previously_retryable(self):
        for code in (400, 401, 403, 429, 500, 502, 503, 504):
            with self.subTest(code=code), patch.object(yandex_ai.httpx, "post", return_value=httpx.Response(code, text="secret-response")) as post, self.assertLogs("yandex_ai", level="INFO") as captured:
                with self.assertRaises(yandex_ai.YandexAIError) as error:
                    yandex_ai._send_request({}, {})
                post.assert_called_once()
                self.assertIn(f"HTTP {code}", str(error.exception))
                self.assertNotIn("secret-response", str(error.exception) + "\n".join(captured.output))

    def test_network_errors_never_retry_or_disclose_request(self):
        request = httpx.Request("POST", "https://example.invalid/?secret-url=yes")
        for error_type in (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError, httpx.WriteError, httpx.RemoteProtocolError):
            with self.subTest(error_type=error_type), patch.object(yandex_ai.httpx, "post", side_effect=error_type("secret-error", request=request)) as post, self.assertLogs("yandex_ai", level="WARNING") as captured:
                with self.assertRaises(yandex_ai.YandexAIError) as error:
                    yandex_ai._send_request({"private": "secret-payload"}, {"Authorization": "secret-key"})
                post.assert_called_once()
                combined = str(error.exception) + "\n".join(captured.output)
                self.assertNotIn("secret", combined)
                self.assertIn("повтор", combined)

    def test_generate_json_preserves_schema_usage_and_one_post(self):
        response = httpx.Response(200, json={"result": {
            "alternatives": [{"status": "FINAL", "message": {"text": '{"ok":true}'}}],
            "usage": {"inputTextTokens": "11", "completionTokens": "7"},
        }})
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        with patch.dict(os.environ, {"YANDEX_API_KEY": "test-only", "YANDEX_FOLDER_ID": "test-only"}), patch.object(yandex_ai.httpx, "post", return_value=response) as post:
            result = yandex_ai.generate_json([{"role": "user", "content": "test-only"}], json_schema=schema)
        self.assertEqual((result.text, result.input_tokens, result.output_tokens), ('{"ok":true}', 11, 7))
        post.assert_called_once()
        self.assertEqual(post.call_args.kwargs["json"]["jsonSchema"]["schema"], schema)

    def test_generate_json_does_not_repeat_malformed_success_body(self):
        with patch.dict(os.environ, {"YANDEX_API_KEY": "test-only", "YANDEX_FOLDER_ID": "test-only"}), patch.object(yandex_ai.httpx, "post", return_value=httpx.Response(200, text="not json")) as post:
            with self.assertRaises(yandex_ai.YandexAIError):
                yandex_ai.generate_json([{"role": "user", "content": "test-only"}])
        post.assert_called_once()


if __name__ == "__main__":
    unittest.main()
