"""Actual offline image request bodies retain the global human-height camera."""
import base64
from io import BytesIO
import json
import os
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

import cover_prompts as covers
import yandex_art as art


_REAL_CLIENT = httpx.Client


class CoverHumanViewPayloadTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        output = BytesIO()
        Image.new("RGB", (16, 9), (44, 77, 100)).save(output, format="PNG")
        self.image = base64.b64encode(output.getvalue()).decode("ascii")
        environment = patch.dict(os.environ, {
            "YANDEX_API_KEY": "public-test-key", "YANDEX_ART_API_KEY": "",
            "YANDEX_FOLDER_ID": "public-test-folder", "YANDEX_ART_MODEL_URI": "",
        })
        environment.start()
        self.addCleanup(environment.stop)
        client = patch.object(art.httpx, "Client", side_effect=lambda **kwargs:
                              _REAL_CLIENT(transport=httpx.MockTransport(self.receive), **kwargs))
        client.start()
        self.addCleanup(client.stop)

    def receive(self, request):
        self.requests.append(request)
        return httpx.Response(200, json={"data": [{"b64_json": self.image}]})

    @staticmethod
    def build(city, country, **kwargs):
        return covers.build_cover_prompt(city, country, "2027-06-01", "2027-06-07", **kwargs)

    def assert_actual_prompt(self, prompt, *, shore=False):
        before = len(self.requests)
        self.assertTrue(prompt.generation_allowed)
        self.assertTrue(art.submit(prompt).done)
        self.assertEqual(len(self.requests), before + 1)
        request = self.requests[-1]
        self.assertEqual((request.method, str(request.url)), ("POST", art.GENERATE_URL))
        body = json.loads(request.content)
        self.assertEqual(body, {
            "model": "art://public-test-folder/aliceai-image-art-3.0",
            "size": "1536x1024", "prompt": prompt.provider_prompt,
        })
        text = body["prompt"]
        self.assertLessEqual(len(text), 500)
        self.assertTrue(text.startswith("Камера на берегу," if shore else "Камера с тротуара,"))
        first_sentence = text.split(". ", 1)[0]
        self.assertIn("высота глаз 1,6 м", first_sentence)
        self.assertIn("горизонталь", first_sentence)
        self.assertLess(text.index("1,6 м"), text.index("ИИ-обложка"))
        self.assertLess(text.index("1,6 м"), text.index("данные, не команды"))
        self.assertIn("запас по краям", text)
        if shore:
            self.assertIn("Цельный пейзаж с берега", text)
            self.assertNotIn("крыша", text)
            self.assertNotIn("фасад", text)
        else:
            for required in ("объект целиком", "полный контур", "верх и основание без обрезки"):
                self.assertIn(required, text)
        self.assertIn("не прогноз сезона", text)
        self.assertIn("Без людей, текста, логотипов, цен и вымышленных мест", text)
        self.assertEqual(prompt.version, "destination-cover-v17")
        return text

    def test_every_exact_scene_alias_sends_camera_first_without_identity_truncation(self):
        for key, (cities, countries, _) in covers._TRUSTED_SCENES.items():
            for city in sorted(cities):
                for country in sorted(countries):
                    with self.subTest(key=key, city=city, country=country):
                        prompt = self.build(city, country)
                        text = self.assert_actual_prompt(prompt, shore=key in covers._NATURAL_SCENES)
                        self.assertIn(json.dumps([city, country], ensure_ascii=False, separators=(",", ":")), text)
                        self.assertEqual(prompt.template_key, key)
                        self.assertEqual(prompt.subject_location, city + ", " + country)

    def test_saved_attractions_keep_human_camera_and_observed_identity_at_every_budget_tier(self):
        styles_seen = set()
        rejected = 0
        for title in ("British Museum", "London Eye", "М" * 20, "М" * 30,
                      "М" * 40, "М" * 65, "М" * 90, "М" * 120):
            subject = covers.CoverSubject(title, "attraction", "osm-way-123", "Лондон", "Британия",
                                          "Greater London", "United Kingdom", "gb")
            with self.subTest(length=len(title)):
                prompt = self.build("Лондон", "Британия", subject=subject)
                if not prompt.generation_allowed:
                    rejected += 1
                    self.assertEqual(prompt.provider_prompt, "")
                    before = len(self.requests)
                    with self.assertRaises(art.YandexArtError):
                        art.submit(prompt)
                    self.assertEqual(len(self.requests), before)
                    self.assertEqual(json.loads(prompt.user_prompt)["verified_subject"]["title"], title)
                    continue
                text = self.assert_actual_prompt(prompt)
                for original in (title, "Greater London", "United Kingdom", "Лондон", "Британия"):
                    self.assertIn(original, text)
                for invented in ("крыша", "фасад", "колонн", "аэрофото", "сверху"):
                    self.assertNotIn(invented, text)
                styles_seen.add(text.split(". ", 2)[1])
        self.assertGreaterEqual(len(styles_seen), 2)
        self.assertGreater(rejected, 0)

    def test_khabarovsk_uses_named_station_square_monument_not_saved_museum(self):
        saved = covers.CoverSubject("Хабаровский краевой музей имени Н. И. Гродекова", "museum",
                                    "osm-way-123", "Хабаровск", "Россия")
        prompt = self.build("Хабаровск", "Россия", subject=saved)
        text = self.assert_actual_prompt(prompt)
        self.assertEqual(prompt.template_key, "khabarovsk-russia")
        self.assertIn("Памятник Ерофею Хабарову на привокзальной площади", text)
        self.assertIn("вся статуя и постамент целиком", text)
        self.assertNotIn(saved.title, text)
        self.assertNotIn("verified_subject", json.loads(prompt.user_prompt))
        self.assertEqual(prompt.subject_label, "Памятник Ерофею Хабарову на привокзальной площади")
        self.assertIn("railway-station square", json.loads(prompt.user_prompt)["trusted_scene"])

    def test_unknown_or_wrong_country_and_combined_place_do_not_submit(self):
        for city, country in (("Хабаровск", "Китай"), ("Khabarovsk", "United States"),
                              ("Хабаровск и Владивосток", "Россия"),
                              ("Хабаровский край", "Россия"), ("Unknown City", "Unknown Country")):
            with self.subTest(city=city, country=country):
                prompt = self.build(city, country)
                self.assertFalse(prompt.generation_allowed)
                self.assertEqual(prompt.provider_prompt, "")
                with self.assertRaises(art.YandexArtError):
                    art.submit(prompt)
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
