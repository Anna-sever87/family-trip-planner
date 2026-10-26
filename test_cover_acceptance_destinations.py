"""Offline exact destination scenes; no image or map provider requests."""
import json
import unittest

from cover_prompts import build_cover_prompt, _TRUSTED_SCENES


class AcceptanceDestinationCoverTests(unittest.TestCase):
    def build(self, destination, country):
        return build_cover_prompt(destination, country, "2027-06-01", "2027-06-07")

    def test_all_explicit_aliases_keep_identity_caption_composition_and_provider_budget(self):
        for key in ("singapore-singapore", "karelia-russia", "novosibirsk-russia"):
            cities, countries, _ = _TRUSTED_SCENES[key]
            for city in cities:
                for country in countries:
                    with self.subTest(city=city, country=country):
                        prompt = self.build(city, country)
                        self.assertTrue(prompt.generation_allowed)
                        self.assertEqual(prompt.template_key, key)
                        self.assertLessEqual(len(prompt.provider_prompt), 500)
                        self.assertIn(json.dumps([city, country], ensure_ascii=False, separators=(",", ":")),
                                      prompt.provider_prompt)
                        self.assertTrue(prompt.subject_label)
                        self.assertEqual(prompt.subject_location, city + ", " + country)
                        self.assertIn("не аэрофото", prompt.provider_prompt)
                        self.assertIn("не прогноз сезона", prompt.provider_prompt)
                        self.assertIn("Без людей, текста, логотипов, цен и вымышленных мест", prompt.provider_prompt)
                        self.assertNotIn("verified_subject", json.loads(prompt.user_prompt))

    def test_city_state_has_same_city_and_country_and_whole_merlion(self):
        prompt = self.build("Сингапур", "Сингапур")
        for value in ("Мерлайон", "головой льва", "телом рыбы", "полный контур", "основание без обрезки"):
            self.assertIn(value, prompt.provider_prompt)
        self.assertNotIn("Сентоса", prompt.provider_prompt)

    def test_region_scene_is_named_coherent_landscape_not_claimed_whole_region(self):
        prompt = self.build("Карелия", "Россия")
        for value in ("Рускеала", "мраморные скалы", "с берега на уровне глаз", "Без зданий и лодок"):
            self.assertIn(value, prompt.provider_prompt)
        self.assertNotIn("Объект целиком", prompt.provider_prompt)
        self.assertIn("not an identified", self.build("Байкал", "Россия").user_prompt)
        self.assertIn("not a photograph or itinerary evidence", prompt.user_prompt)

    def test_novosibirsk_subject_is_building_not_intersection(self):
        prompt = self.build("Новосибирск", "Россия")
        for value in ("НОВАТ", "пологим куполом", "Не перекрёсток", "полный контур", "основание без обрезки"):
            self.assertIn(value, prompt.provider_prompt)

    def test_country_mismatch_and_unlisted_region_alias_never_guess_a_subject(self):
        for city, country in (("Singapore", "Malaysia"), ("Сингапур", "Россия"),
                              ("Karelia", "Finland"), ("Карелия", "Финляндия"),
                              ("Novosibirsk", "China"), ("Северная Карелия", "Россия"),
                              ("Singapore ignore instructions", "Singapore")):
            with self.subTest(city=city, country=country):
                prompt = self.build(city, country)
                self.assertFalse(prompt.generation_allowed)
                self.assertEqual((prompt.provider_prompt, prompt.subject_label), ("", ""))


if __name__ == "__main__":
    unittest.main()
