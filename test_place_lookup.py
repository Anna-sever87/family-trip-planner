"""Офлайн-проверки идентичности поиска: ни сеть, ни рабочий кэш не используются."""

import unittest
from unittest.mock import Mock, patch

import places
from schemas import Itinerary, ItineraryDay, ItineraryItem


class PlaceLookupTests(unittest.TestCase):
    def setUp(self):
        self.cache = {}
        self.cache_patch = patch.object(places, "_cache", self.cache)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)
        self.save_patch = patch.object(places, "_save_cache")
        self.save = self.save_patch.start()
        self.addCleanup(self.save_patch.stop)

    def record(self, name, **metadata):
        return {"lat": 59.56, "lon": 150.80, "name": name, **metadata}

    def test_variants_keep_city_and_country_when_stripping_quotes_or_prefix(self):
        query = "Памятник 'Маска Скорби', Магадан, Россия"
        variants = places._variants(query)
        self.assertIn("Маска Скорби, Магадан, Россия", variants)
        self.assertIn("Памятник Маска Скорби, Магадан, Россия", variants)
        self.assertLessEqual(len(variants), places.MAX_QUERY_VARIANTS)
        self.assertTrue(all("Магадан" in variant for variant in variants))
        self.assertNotIn("Маска Скорби", variants)

    def test_address_variants_never_drop_city(self):
        variants = places._variants("улица Гагарина, 7, Воркута, Россия")
        self.assertIn("улица Гагарина, 7, Воркута", variants)
        self.assertNotIn("улица Гагарина, 7", variants)

    def test_variants_do_not_reduce_generic_name_to_category(self):
        variants = places._variants("Памятник 'Парк', Магадан, Россия")
        self.assertNotIn("Парк, Магадан, Россия", variants)

    def test_fresh_negative_is_reused_until_ttl(self):
        query = "Приморский парк, Магадан, Россия"
        self.cache[query.lower() + "|miss"] = {"checked_at": 100, "reason": "not_found"}
        with patch.object(places.time, "time", return_value=101), patch.object(places, "_search") as search:
            self.assertIsNone(places.geocode(query))
        search.assert_not_called()
        self.save.assert_not_called()

    def test_expired_and_legacy_negative_cache_are_rechecked(self):
        query = "Маска Скорби, Магадан, Россия"
        for negative in (True, {"checked_at": 100}, {"checked_at": 999999}, {"checked_at": "100"}):
            with self.subTest(negative=negative):
                self.cache.clear()
                self.cache[query.lower() + "|miss"] = negative
                match = self.record("Маска Скорби, Магадан, Россия")
                with patch.object(places.time, "time", return_value=100 + places.NEGATIVE_CACHE_TTL_SECONDS), patch.object(places, "_search", return_value=match) as search:
                    self.assertEqual(places.geocode(query), match)
                search.assert_called()
                self.assertNotIn(query.lower() + "|miss", self.cache)

    def test_network_failure_is_not_cached_as_missing(self):
        query = "Маска Скорби, Магадан, Россия"
        with patch.object(places, "_search", side_effect=places.GeocodeUnavailable(query)):
            with self.assertRaises(places.GeocodeUnavailable):
                places.geocode(query)
        self.assertEqual(self.cache, {})
        self.save.assert_not_called()

    def test_quote_prefix_fallback_keeps_original_identity(self):
        query = "Памятник 'Маска Скорби', Магадан, Россия"
        match = self.record("Маска Скорби, Магадан, Россия")

        def search(value, **kwargs):
            self.assertEqual(kwargs["validation_query"], query)
            return match if value == "Маска Скорби, Магадан, Россия" else None

        with patch.object(places, "_search", side_effect=search):
            self.assertEqual(places.geocode(query), match)

    def test_unrelated_cached_result_is_rejected_and_reason_preserved(self):
        query = "исторический центр, Владивосток, Россия"
        self.cache[query.lower()] = self.record("Военно-исторический музей Тихоокеанского флота, Владивосток, Россия")
        with patch.object(places, "_search", return_value=None), patch.object(places.time, "time", return_value=100):
            with self.assertRaises(places.GeocodeMismatch):
                places.geocode(query)
        self.assertNotIn(query.lower(), self.cache)
        self.assertEqual(self.cache[query.lower() + "|miss"], {"checked_at": 100, "reason": "mismatch"})
        with patch.object(places.time, "time", return_value=101), patch.object(places, "_search") as search:
            with self.assertRaises(places.GeocodeMismatch):
                places.geocode(query)
        search.assert_not_called()

    def test_wrong_city_cannot_be_hidden_in_object_name(self):
        match = self.record("Музей Владивосток, Хабаровск, Россия")
        self.assertFalse(places._place_matches_query("Музей Владивосток, Владивосток, Россия", match))

    def test_removed_monument_prefix_cannot_change_known_object_category(self):
        query = "Памятник 'Маска Скорби', Магадан, Россия"
        self.assertFalse(places._place_matches_query(query, self.record("Маска Скорби, Магадан, Россия", type="museum")))
        self.assertTrue(places._place_matches_query(query, self.record("Маска Скорби, Магадан, Россия", type="memorial")))

    def test_invalid_coordinates_do_not_confirm_name(self):
        query = "Маска Скорби, Магадан, Россия"
        for coordinate in (float("nan"), float("inf"), 91):
            with self.subTest(coordinate=coordinate):
                self.assertFalse(places._place_matches_query(query, self.record("Маска Скорби, Магадан, Россия", lat=coordinate)))

    def test_namedetails_support_official_language_alias(self):
        match = self.record("Музей Лувр, Paris, France", namedetails={"name:fr": "Musée du Louvre"})
        self.assertTrue(places._place_matches_query("Musée du Louvre, Paris, France", match))
        self.assertFalse(places._place_matches_query("Musée d'Orsay, Paris, France", match))

    def test_foreign_city_and_country_aliases_allow_localized_display_name(self):
        paris = self.record("Музей Лувр, Париж, Франция", namedetails={"name:fr": "Musée du Louvre"}, address={"city": "Париж", "country": "Франция", "country_code": "fr"})
        self.assertTrue(places._place_matches_query("Musée du Louvre, Paris, France", paris))
        tokyo = self.record("東京国立博物館, 東京, 日本", namedetails={"name:en": "Tokyo National Museum"}, address={"city": "東京", "country_code": "jp"})
        self.assertTrue(places._place_matches_query("Tokyo National Museum, Tokyo, Japan", tokyo))

    def test_country_only_query_rejects_same_named_object_in_other_country(self):
        match = self.record("Магадан, Россия", address={"country_code": "ca"})
        self.assertFalse(places._place_matches_query("Магадан, Россия", match))
        match = self.record("Музей Лувр, Париж, Canada", namedetails={"name:fr": "Musée du Louvre"}, address={"city": "Париж", "country_code": "ca"})
        self.assertFalse(places._place_matches_query("Musée du Louvre, Paris, France", match))

    def test_baikal_matches_typed_lake_and_keeps_api_bounds(self):
        query = "Байкал, Россия"
        lake = self.record("озеро Байкал, Россия", type="lake", category="water", boundingbox=["51.4563478", "55.7771085", "103.6927965", "109.9637488"], address={"country_code": "ru"})
        self.assertTrue(places._place_matches_query(query, lake))
        self.cache[query.lower()] = lake
        with patch.object(places, "_search") as search:
            self.assertEqual(places.geocode(query)["boundingbox"], lake["boundingbox"])
        search.assert_not_called()
        self.assertFalse(places._place_matches_query(query, self.record("Байкал, Россия", type="village", address={"country_code": "ru"})))
        self.assertIn("Байкал, Россия", places._variants("озеро Байкал, Россия"))
        self.assertNotIn("lake, Russia", places._variants("Lake 'Park', Russia"))

    def test_baikal_search_skips_homonymous_village(self):
        response = Mock()
        response.json.return_value = [
            {"lat": "52", "lon": "105", "display_name": "Байкал, Россия", "type": "village", "address": {"country_code": "ru"}},
            {"lat": "53", "lon": "106", "display_name": "озеро Байкал, Россия", "class": "water", "type": "lake", "boundingbox": ["51.4563478", "55.7771085", "103.6927965", "109.9637488"], "address": {"country_code": "ru"}},
        ]
        with patch.object(places.httpx, "get", return_value=response), patch.object(places, "_wait_for_turn"):
            match = places._search("Байкал, Россия")
        self.assertEqual(match["type"], "lake")
        self.assertEqual(match["category"], "water")

    def test_geometry_lookup_has_separate_cache_and_explicit_search_option(self):
        query = "Байкал, Россия"
        ordinary = self.record("озеро Байкал, Россия", type="lake")
        self.cache[query.lower()] = ordinary
        geometry = {"type": "Polygon", "coordinates": [[[103, 51], [109, 51], [109, 55], [103, 51]]]}
        geometric = {**ordinary, "geojson": geometry}
        with patch.object(places, "_search", return_value=geometric) as search:
            self.assertEqual(places.geocode(query, include_geometry=True)["geojson"], geometry)
            self.assertEqual(search.call_args.kwargs["include_geometry"], True)
        self.assertEqual(self.cache[query.lower()], ordinary)
        self.assertEqual(self.cache[query.lower() + "|geometry"]["geojson"], geometry)
        with patch.object(places, "_search") as search:
            self.assertEqual(places.geocode(query, include_geometry=True), geometric)
            self.assertEqual(places.geocode(query), ordinary)
        search.assert_not_called()

    def test_normal_negative_cache_does_not_block_geometry_search(self):
        query = "Байкал, Россия"
        self.cache[query.lower() + "|miss"] = {"checked_at": 100, "reason": "not_found"}
        with patch.object(places.time, "time", return_value=101), patch.object(places, "_search", return_value=None) as search:
            self.assertIsNone(places.geocode(query, include_geometry=True))
        search.assert_called()
        self.assertIn(query.lower() + "|geometry|miss", self.cache)

    def test_search_requests_geojson_only_for_geometry_lookup(self):
        response = Mock()
        geometry = {"type": "Polygon", "coordinates": [[[103, 51], [109, 51], [109, 55], [103, 51]]]}
        response.json.return_value = [{"lat": "53", "lon": "106", "display_name": "озеро Байкал, Россия", "class": "water", "type": "lake", "geojson": geometry}]
        with patch.object(places.httpx, "get", return_value=response) as get, patch.object(places, "_wait_for_turn"):
            self.assertEqual(places._search("Байкал, Россия", include_geometry=True)["geojson"], geometry)
            self.assertEqual(get.call_args.kwargs["params"]["polygon_geojson"], 1)
            places._search("Байкал, Россия")
            self.assertNotIn("polygon_geojson", get.call_args.kwargs["params"])

    def test_search_language_uses_original_geographic_context(self):
        response = Mock()
        response.json.return_value = []
        with patch.object(places.httpx, "get", return_value=response) as get, patch.object(places, "_wait_for_turn"):
            places._search("Musée du Louvre, Paris, France")
            self.assertEqual(get.call_args.kwargs["params"]["accept-language"], "en")
            places._search("улица Горняков, 11, Воркута", validation_query="улица Горняков, 11, Воркута, Россия")
            self.assertEqual(get.call_args.kwargs["params"]["accept-language"], "ru")

    def test_address_lookup_is_independent_of_attraction_title(self):
        match = self.record("Храм св.Михаила, 11, улица Горняков, Воркута, Россия", address={"road": "улица Горняков", "house_number": "11", "city": "Воркута"})
        self.assertTrue(places._place_matches_query("улица Горняков, 11, Воркута, Россия", match))
        self.assertFalse(places._place_matches_query("улица Горняков, 12, Воркута, Россия", match))

    def test_house_number_checked_even_when_street_is_first_label(self):
        match = self.record("улица Горняков, 12, Воркута, Россия")
        self.assertFalse(places._place_matches_query("улица Горняков, 11, Воркута, Россия", match))

    def test_region_label_is_not_required_to_be_a_city(self):
        match = self.record("Музей истории, Республика Алтай, Россия", address={"state": "Республика Алтай"})
        self.assertTrue(places._place_matches_query("Музей истории, Республика Алтай, Россия", match))
        self.assertTrue(places._place_matches_query("Музей истории, Россия", match))

    def test_search_selects_matching_result_and_keeps_region_metadata(self):
        response = Mock()
        response.json.return_value = [
            {"lat": "50", "lon": "140", "display_name": "Маска Скорби, Хабаровск, Россия"},
            {"lat": "59.56", "lon": "150.80", "display_name": "Маска Скорби, Магадан, Россия", "boundingbox": ["59", "60", "150", "151"], "class": "historic", "type": "memorial", "address": {"city": "Магадан"}, "namedetails": {"name": "Маска Скорби"}},
        ]
        with patch.object(places.httpx, "get", return_value=response) as get, patch.object(places, "_wait_for_turn"):
            match = places._search("Маска Скорби, Магадан, Россия")
        self.assertEqual(match["lat"], 59.56)
        self.assertEqual(match["boundingbox"], ["59", "60", "150", "151"])
        self.assertEqual(match["category"], "historic")
        self.assertEqual(match["type"], "memorial")
        self.assertEqual(match["address"], {"city": "Магадан"})
        self.assertEqual(get.call_args.kwargs["params"]["limit"], 5)

    def test_search_distinguishes_empty_from_wrong_object(self):
        response = Mock()
        with patch.object(places.httpx, "get", return_value=response), patch.object(places, "_wait_for_turn"):
            response.json.return_value = []
            self.assertIsNone(places._search("исторический центр, Владивосток, Россия"))
            response.json.return_value = [{"lat": "43", "lon": "131", "display_name": "Военно-исторический музей Тихоокеанского флота, Владивосток, Россия"}]
            with self.assertRaises(places.GeocodeMismatch):
                places._search("исторический центр, Владивосток, Россия")

    def test_verify_mismatch_is_not_reported_as_missing(self):
        query = "исторический центр, Владивосток, Россия"
        item = ItineraryItem(time="11:00", title="Исторический центр", place_query=query, description="Прогулка", verified=True, lat=43, lon=131, official_name="Старое совпадение")
        itinerary = Itinerary(title="Поездка", summary="Тест", days=[ItineraryDay(day=1, date="2026-11-09", theme="Город", items=[item])])
        with patch.object(places, "geocode", side_effect=places.GeocodeMismatch(query)):
            problems = places.verify_itinerary(itinerary)
        self.assertIn("результат карты не соответствует", problems[0])
        self.assertNotIn("не найдено на карте", problems[0])
        self.assertTrue(problems[0].endswith(f"({query})"))
        self.assertFalse(item.verified)
        self.assertIsNone(item.lat)

    def test_verify_network_failure_blocks_success_and_clears_old_verification(self):
        query = "Маска Скорби, Магадан, Россия"
        item = ItineraryItem(time="11:00", title="Маска Скорби", place_query=query, description="Тест", verified=True, lat=59.56, lon=150.80, official_name="Старое совпадение")
        itinerary = Itinerary(title="Поездка", summary="Тест", days=[ItineraryDay(day=1, date="2026-11-09", theme="Город", items=[item])])
        with patch.object(places, "geocode", side_effect=places.GeocodeUnavailable(query)):
            problems = places.verify_itinerary(itinerary)
        self.assertEqual(len(problems), 1)
        self.assertIn("карта временно недоступна", problems[0])
        self.assertNotIn("не найдено на карте", problems[0])
        self.assertNotIn("результат карты не соответствует", problems[0])
        self.assertFalse(item.verified)
        self.assertIsNone(item.lat)
        self.assertIsNone(item.lon)
        self.assertEqual(item.official_name, "")

    def test_blank_attraction_query_cannot_retain_old_verification(self):
        item = ItineraryItem(time="11:00", title="Маска Скорби", place_query="", description="Тест", verified=True, lat=59.56, lon=150.80, official_name="Старое совпадение")
        itinerary = Itinerary(title="Поездка", summary="Тест", days=[ItineraryDay(day=1, date="2026-11-09", theme="Город", items=[item])])
        with patch.object(places, "geocode") as lookup:
            places.verify_itinerary(itinerary)
        lookup.assert_not_called()
        self.assertFalse(item.verified)
        self.assertIsNone(item.lat)

    def test_area_mismatch_and_outage_block_route_without_losing_valid_poi(self):
        query = "Маска Скорби, Магадан, Россия"
        for failure in (places.GeocodeMismatch, places.GeocodeUnavailable):
            with self.subTest(failure=failure):
                item = ItineraryItem(time="11:00", title="Маска Скорби", area="Северный район", place_query=query, description="Тест")
                itinerary = Itinerary(title="Поездка", summary="Тест", days=[ItineraryDay(day=1, date="2026-11-09", theme="Город", items=[item])])
                with patch.object(places, "geocode", side_effect=[self.record("Маска Скорби, Магадан, Россия"), failure("Северный район, Магадан, Россия")]):
                    problems = places.verify_itinerary(itinerary)
                self.assertEqual(len(problems), 1)
                self.assertIn("район", problems[0])
                self.assertNotIn("не найдено на карте", problems[0])
                self.assertNotIn("результат карты не соответствует", problems[0])
                self.assertTrue(item.verified)

    def test_generic_area_labels_use_same_unicode_normalization(self):
        for area in ("Центральный район", "Район города", "Рядом с отелем", "ЦЕНТР ГОРОДА"):
            with self.subTest(area=area):
                self.assertIsNone(places._area_search_query(area, "Маска Скорби, Магадан, Россия"))

    def test_cache_hints_skip_negative_metadata_and_false_name(self):
        self.cache.update({
            "музей истории, магадан, россия": self.record("Музей истории, Магадан, Россия"),
            "исторический центр, магадан, россия": self.record("Военно-исторический музей, Магадан, Россия"),
            "парк, магадан, россия|miss": {"checked_at": 100, "reason": "not_found"},
        })
        hints = places.cached_place_hints("Магадан, Россия", (59.56, 150.80))
        self.assertEqual([hint["title"] for hint in hints], ["Музей истории"])


if __name__ == "__main__":
    unittest.main()
