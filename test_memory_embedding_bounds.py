"""Embedding transport sends once and never logs private inputs or secrets."""

import socket
import unittest
from unittest.mock import patch

import httpx
import memory


TEST_ENV = {
    "YANDEX_API_KEY": "secret-api-key",
    "YANDEX_FOLDER_ID": "secret-folder-id",
    "YANDEX_EMBEDDING_DOC_URI": "secret-document-model-uri",
    "YANDEX_EMBEDDING_QUERY_URI": "secret-query-model-uri",
}
PRIVATE_TEXT = "secret-private-trip-preferences"


class MemoryEmbeddingBoundsTests(unittest.TestCase):
    def setUp(self):
        self.env_guard = patch.object(memory, "_required_env", side_effect=TEST_ENV.__getitem__)
        self.env_guard.start()
        self.addCleanup(self.env_guard.stop)
        self.network_guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Real network forbidden"))
        self.network_guard.start()
        self.addCleanup(self.network_guard.stop)

    def test_success_sends_once_for_query_and_document_and_preserves_vector(self):
        vector = [0.25] * memory.EMBEDDING_DIMENSION
        for purpose in ("query", "document"):
            with self.subTest(purpose=purpose), patch.object(memory.httpx, "post", return_value=httpx.Response(200, json={"embedding": vector})) as post, self.assertLogs("memory", level="INFO") as logs:
                result = memory._embedding(PRIVATE_TEXT, purpose=purpose)
            post.assert_called_once()
            self.assertEqual(result, vector)
            self.assertEqual(post.call_args.kwargs["json"]["text"], PRIVATE_TEXT)
            expected_uri = TEST_ENV["YANDEX_EMBEDDING_QUERY_URI" if purpose == "query" else "YANDEX_EMBEDDING_DOC_URI"]
            self.assertEqual(post.call_args.kwargs["json"]["modelUri"], expected_uri)
            captured = "\n".join(logs.output)
            self.assertIn(f"purpose={purpose}", captured)
            self.assertIn("HTTP 200", captured)
            self.assertNotIn("secret", captured)

    def _assert_network_error_is_not_repeated(self, error_type):
        request = httpx.Request("POST", "https://example.invalid/?secret-query=yes", headers={"Authorization": "secret-header"})
        error = error_type("secret-error-description", request=request)
        with patch.object(memory.httpx, "post", side_effect=error) as post, self.assertLogs("memory", level="INFO") as logs:
            with self.assertRaises(memory.MemoryBackendError) as raised:
                memory._embedding(PRIVATE_TEXT, purpose="query")
        post.assert_called_once()
        captured = "\n".join(logs.output) + str(raised.exception)
        self.assertIn("purpose=query", captured)
        self.assertIn(error_type.__name__, captured)
        self.assertIn("повтор", captured)
        self.assertNotIn("secret", captured)

    def test_read_timeout_sends_once_without_disclosing_request_or_exception(self):
        self._assert_network_error_is_not_repeated(httpx.ReadTimeout)

    def test_connect_error_sends_once_without_disclosing_request_or_exception(self):
        self._assert_network_error_is_not_repeated(httpx.ConnectError)

    def _assert_http_error_is_not_repeated(self, status):
        response = httpx.Response(status, text="secret-response-body")
        with patch.object(memory.httpx, "post", return_value=response) as post, self.assertLogs("memory", level="INFO") as logs:
            with self.assertRaises(memory.MemoryBackendError) as raised:
                memory._embedding(PRIVATE_TEXT, purpose="document")
        post.assert_called_once()
        captured = "\n".join(logs.output) + str(raised.exception)
        self.assertIn("purpose=document", captured)
        self.assertIn(f"HTTP {status}", captured)
        self.assertNotIn("secret", captured)

    def test_quota_error_sends_once(self):
        self._assert_http_error_is_not_repeated(429)

    def test_server_errors_send_once(self):
        for status in (500, 502, 503, 504):
            with self.subTest(status=status):
                self._assert_http_error_is_not_repeated(status)

    def test_malformed_response_is_not_retried_or_logged(self):
        with patch.object(memory.httpx, "post", return_value=httpx.Response(200, text="secret-malformed-json")) as post, self.assertLogs("memory", level="INFO") as logs:
            with self.assertRaises(memory.MemoryBackendError) as raised:
                memory._embedding(PRIVATE_TEXT, purpose="query")
        post.assert_called_once()
        self.assertNotIn("secret", "\n".join(logs.output) + str(raised.exception))

    def test_vector_validation_remains_active_without_retry(self):
        for vector in ([0.5], [float("nan")] * memory.EMBEDDING_DIMENSION, [True] * memory.EMBEDDING_DIMENSION):
            response = httpx.Response(200)
            with self.subTest(first=vector[0]), patch.object(response, "json", return_value={"result": {"embedding": vector}}), patch.object(memory.httpx, "post", return_value=response) as post:
                with self.assertRaises(memory.MemoryBackendError):
                    memory._embedding(PRIVATE_TEXT, purpose="document")
            post.assert_called_once()

    def test_invalid_purpose_never_reaches_transport_or_logs_private_value(self):
        with patch.object(memory.httpx, "post") as post, patch.object(memory.logger, "info") as info, patch.object(memory.logger, "warning") as warning:
            with self.assertRaises(ValueError) as raised:
                memory._embedding(PRIVATE_TEXT, purpose="secret-invalid-purpose")
        post.assert_not_called()
        info.assert_not_called()
        warning.assert_not_called()
        self.assertNotIn("secret", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
