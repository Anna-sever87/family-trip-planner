"""No network or paid requests: current Images API and defensive transport."""

import base64
from dataclasses import replace
from io import BytesIO
import json
import os
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from cover_prompts import build_cover_prompt
import yandex_art as art


_REAL_HTTPX_CLIENT = httpx.Client


class YandexArtTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {
            "YANDEX_API_KEY": "public-test-key", "YANDEX_ART_API_KEY": "",
            "YANDEX_FOLDER_ID": "public-test-folder", "YANDEX_ART_MODEL_URI": "",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.prompt = build_cover_prompt("Москва", "Россия", "2026-10-28", "2026-11-03")
        self.requests = []
        self.client_options = []

    @staticmethod
    def image_bytes(format="JPEG", size=(16, 9)):
        output = BytesIO()
        Image.new("RGB", size, (16, 43, 70)).save(output, format=format)
        return output.getvalue()

    def transport(self, response=None, *, error=None, handler=None):
        def receive(request):
            self.requests.append(request)
            if error is not None:
                raise error
            if handler is not None:
                return handler(request)
            return response or self.finished_response()

        def client(**kwargs):
            self.client_options.append(kwargs)
            return _REAL_HTTPX_CLIENT(transport=httpx.MockTransport(receive), **kwargs)

        return patch.object(art.httpx, "Client", side_effect=client)

    def finished_response(self, image=None, **updates):
        result = {"created": 1700000000,
                  "data": [{"b64_json": base64.b64encode(image or self.image_bytes()).decode("ascii")}]}
        result.update(updates)
        return httpx.Response(200, json=result)

    def test_submit_makes_exactly_one_current_images_post_with_compact_prompt(self):
        with self.transport():
            operation = art.submit(self.prompt)
        self.assertTrue(operation.done)
        self.assertRegex(operation.operation_id, r"^local-[0-9a-f]{32}$")
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), art.GENERATE_URL)
        body = json.loads(request.content)
        self.assertEqual(body, {"model": "art://public-test-folder/aliceai-image-art-3.0",
                                "prompt": self.prompt.provider_prompt, "size": "1536x1024"})
        self.assertLessEqual(len(body["prompt"]), 500)
        self.assertEqual(request.headers["Authorization"], "Api-Key public-test-key")
        self.assertEqual(request.headers["OpenAI-Project"], "public-test-folder")
        self.assertEqual(request.headers["Accept-Encoding"], "identity")
        self.assertFalse(self.client_options[0]["follow_redirects"])
        self.assertFalse(self.client_options[0]["trust_env"])
        self.assertEqual(self.client_options[0]["timeout"].read, 180)

    def test_optional_dedicated_key_takes_precedence_without_touching_text_key(self):
        with patch.dict(os.environ, {"YANDEX_ART_API_KEY": "public-art-key"}), self.transport():
            art.submit(self.prompt)
        self.assertEqual(self.requests[0].headers["Authorization"], "Api-Key public-art-key")
        self.assertEqual(os.environ["YANDEX_API_KEY"], "public-test-key")

    def test_explicit_model_must_remain_in_configured_folder(self):
        with patch.dict(os.environ, {"YANDEX_ART_MODEL_URI": "art://public-test-folder/aliceai-image-art-3.0"}):
            self.assertEqual(art.configured_model_uri(), "art://public-test-folder/aliceai-image-art-3.0")
        for value in ("https://evil.invalid", "art://other-folder/aliceai-image-art-3.0", "art://public-test-folder/aliceai-image-art-3.0?key=secret"):
            with self.subTest(value=value), patch.dict(os.environ, {"YANDEX_ART_MODEL_URI": value}):
                with self.assertRaises(art.YandexArtError) as caught:
                    art.configured_model_uri()
                self.assertEqual(caught.exception.code, "configuration")

    def test_missing_or_invalid_credentials_fail_before_network(self):
        for settings in ({"YANDEX_API_KEY": ""}, {"YANDEX_FOLDER_ID": ""},
                         {"YANDEX_API_KEY": "unsafe\nheader"}, {"YANDEX_API_KEY": "ключ"},
                         {"YANDEX_FOLDER_ID": "../escape"}):
            with self.subTest(settings=settings), patch.dict(os.environ, settings), self.transport():
                with self.assertRaises(art.YandexArtError):
                    art.submit(self.prompt)
        self.assertFalse(self.requests)

    def test_post_timeout_has_uncertain_status_no_retry_and_safe_message(self):
        with self.transport(error=httpx.ReadTimeout("LEAK_SENTINEL url key prompt")):
            with self.assertRaises(art.YandexArtError) as caught:
                art.submit(self.prompt)
        self.assertEqual(len(self.requests), 1)
        self.assertTrue(caught.exception.submission_uncertain)
        self.assertEqual(caught.exception.code, "network")
        self.assertNotIn("LEAK_SENTINEL", str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)

    def test_retired_polling_never_makes_a_network_call(self):
        with self.transport():
            with self.assertRaises(art.YandexArtError) as caught:
                art.get_operation("operation-1")
        self.assertFalse(caught.exception.submission_uncertain)
        self.assertEqual(caught.exception.code, "invalid_operation")
        self.assertFalse(self.requests)

    def test_local_receipt_is_not_a_provider_operation_id_and_cannot_be_polled(self):
        with self.transport():
            operation = art.submit(self.prompt)
            with self.assertRaises(art.YandexArtError):
                art.get_operation(operation.operation_id)
        self.assertEqual(len(self.requests), 1)

    def test_unsafe_operation_ids_never_reach_network(self):
        for value in ("", "..", "x/y", "x?token=value", "x%2fy", "x#fragment", "x\n", "https://evil.invalid", "a" * 129, None, True):
            with self.subTest(value=value), self.transport(), self.assertRaises(art.YandexArtError) as caught:
                art.get_operation(value)
            self.assertEqual(caught.exception.code, "invalid_operation")
        self.assertFalse(self.requests)

    def test_redirect_is_rejected_without_following_or_disclosing_body(self):
        response = httpx.Response(307, headers={"Location": "https://evil.invalid/LEAK_SENTINEL"}, text="LEAK_SENTINEL")
        with self.transport(response), self.assertRaises(art.YandexArtError) as caught:
            art.submit(self.prompt)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(caught.exception.code, "redirect")
        self.assertNotIn("LEAK_SENTINEL", str(caught.exception))

    def test_http_errors_are_fixed_safe_messages_without_paid_retry(self):
        for status, code in ((400, "bad_request"), (401, "unauthorized"), (403, "forbidden"),
                             (404, "not_found"), (429, "rate_limited"), (500, "provider_unavailable")):
            before = len(self.requests)
            with self.subTest(status=status), self.transport(httpx.Response(status, text="LEAK_SENTINEL")):
                with self.assertRaises(art.YandexArtError) as caught:
                    art.submit(self.prompt)
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(caught.exception.status_code, status)
                self.assertEqual(caught.exception.submission_uncertain, status >= 500)
                self.assertNotIn("LEAK_SENTINEL", str(caught.exception))
            self.assertEqual(len(self.requests), before + 1)

    def test_content_length_is_bounded_before_reading(self):
        response = httpx.Response(200, headers={"Content-Type": "application/json", "Content-Length": str(art.MAX_JSON_BYTES + 1)}, content=b"{}")
        with self.transport(response), self.assertRaises(art.YandexArtError) as caught:
            art.submit(self.prompt)
        self.assertEqual(caught.exception.code, "response_too_large")
        self.assertTrue(caught.exception.submission_uncertain)

    def test_streamed_body_is_bounded_even_without_length(self):
        class Stream(httpx.SyncByteStream):
            def __iter__(self):
                yield b" " * 100
                yield b" " * 100
        response = httpx.Response(200, headers={"Content-Type": "application/json"}, stream=Stream())
        with patch.object(art, "MAX_JSON_BYTES", 150), self.transport(response):
            with self.assertRaises(art.YandexArtError) as caught:
                art.submit(self.prompt)
        self.assertEqual(caught.exception.code, "response_too_large")

    def test_invalid_response_types_encoding_and_json_are_rejected(self):
        responses = [httpx.Response(200, text="not json"),
                     httpx.Response(200, headers={"Content-Type": "application/json"}, content=b"{broken"),
                     httpx.Response(200, json=[]), httpx.Response(200, json=True),
                     httpx.Response(200, headers={"Content-Type": "application/json", "Content-Length": "nope"}, content=b"{}")]
        for response in responses:
            with self.subTest(response=response), self.transport(response), self.assertRaises(art.YandexArtError) as caught:
                art.submit(self.prompt)
            self.assertEqual(caught.exception.code, "invalid_response")

    def test_finished_jpeg_and_png_are_validated_and_kept_byte_for_byte(self):
        for format, mime in (("JPEG", "image/jpeg"), ("PNG", "image/png")):
            original = self.image_bytes(format)
            with self.subTest(format=format), self.transport(self.finished_response(original)):
                operation = art.submit(self.prompt)
            self.assertTrue(operation.done)
            self.assertEqual(operation.image, art.GeneratedImage(original, mime, 16, 9))
            self.assertNotIn(str(original), repr(operation))

    def test_submit_returns_finished_image_without_second_request(self):
        with self.transport(self.finished_response()):
            operation = art.submit(self.prompt)
        self.assertTrue(operation.done)
        self.assertEqual(len(self.requests), 1)

    def test_old_async_response_is_rejected_without_fallback(self):
        with self.transport(httpx.Response(200, json={"id": "operation-1"})):
            with self.assertRaises(art.YandexArtError) as caught:
                art.submit(self.prompt)
            self.assertEqual(caught.exception.code, "invalid_response")
            self.assertTrue(caught.exception.submission_uncertain)
        self.assertEqual(len(self.requests), 1)

    def test_operation_failure_ignores_provider_error_text(self):
        response = httpx.Response(200, json={"error": {"code": 7, "message": "LEAK_SENTINEL"}})
        with self.transport(response), self.assertRaises(art.YandexArtError) as caught:
            art.submit(self.prompt)
        self.assertEqual(caught.exception.code, "operation_failed")
        self.assertNotIn("LEAK_SENTINEL", str(caught.exception))

    def test_exactly_one_inline_base64_image_is_required_without_url_download(self):
        for data in ({}, {"data": []}, {"data": {}}, {"data": [None]}, {"data": [{}, {}]},
                     {"data": [{"url": "https://evil.invalid/LEAK_SENTINEL"}]},
                     {"data": [{"b64_json": None}]}, {"data": [{"b64_json": ""}]}):
            with self.subTest(data=data), self.transport(httpx.Response(200, json=data)):
                with self.assertRaises(art.YandexArtError) as caught:
                    art.submit(self.prompt)
                self.assertEqual(caught.exception.code, "invalid_response")
        self.assertTrue(all(str(request.url) == art.GENERATE_URL for request in self.requests))

    def test_base64_is_strict_and_bounded(self):
        for image in ("!!!", "data:image/png;base64,AAAA", "AAAA\n", "юникод"):
            response = httpx.Response(200, json={"data": [{"b64_json": image}]})
            with self.subTest(image=image), self.transport(response), self.assertRaises(art.YandexArtError) as caught:
                art.submit(self.prompt)
            self.assertEqual(caught.exception.code, "invalid_image")
        response = httpx.Response(200, json={"data": [{"b64_json": "a" * 20}]})
        with patch.object(art, "MAX_BASE64_CHARS", 16), self.transport(response):
            with self.assertRaises(art.YandexArtError) as caught:
                art.submit(self.prompt)
            self.assertEqual(caught.exception.code, "response_too_large")

    def test_svg_html_gif_corruption_and_trailing_payload_are_not_images(self):
        for data in (b"<svg/>", b"<html/>", self.image_bytes("GIF"), self.image_bytes()[:-5],
                     self.image_bytes("PNG")[:-10], self.image_bytes() + b"secret"):
            with self.subTest(prefix=data[:8]), self.assertRaises(art.YandexArtError) as caught:
                art.validate_image(data)
            self.assertEqual(caught.exception.code, "invalid_image")

    def test_decoded_image_dimensions_pixels_and_bytes_are_bounded(self):
        with self.assertRaises(art.YandexArtError):
            art.validate_image(self.image_bytes(size=(4097, 1)))
        with patch.object(art, "MAX_PIXELS", 100), self.assertRaises(art.YandexArtError):
            art.validate_image(self.image_bytes())
        with patch.object(art, "MAX_IMAGE_BYTES", 10), self.assertRaises(art.YandexArtError):
            art.validate_image(self.image_bytes())

    def test_invalid_or_oversized_prompt_is_rejected_before_network(self):
        for prompt in (None, "arbitrary prompt", replace(self.prompt, provider_prompt=""),
                       replace(self.prompt, provider_prompt="a" * 501)):
            with self.subTest(prompt=type(prompt)), self.transport(), self.assertRaises(art.YandexArtError) as caught:
                art.submit(prompt)
            self.assertEqual(caught.exception.code, "invalid_prompt")
        self.assertFalse(self.requests)


if __name__ == "__main__":
    unittest.main()
