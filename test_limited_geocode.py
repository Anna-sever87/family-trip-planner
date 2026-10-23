"""Ограниченный поиск: только подменённые HTTP-ответы и кэш в памяти."""

from copy import deepcopy
import unittest
from unittest.mock import Mock, call, patch

import httpx

import places


class LimitedGeocodeTests(unittest.TestCase):
    query = "Маска Скорби, Магадан, Россия"
    variants = [query, "Памятник Маска Скорби, Магадан, Россия", "Маска Скорби, Магадан"]
    now = 10_000

    def setUp(self):
        self.cache = {}
        self.save = self._patch("_save_cache")
        self._patch("_cache", self.cache)
        self._patch("_wait_for_turn")
        self._patch("_variants", return_value=self.variants)
        self._patch("time.time", return_value=self.now)
        self._patch("time.sleep")
        # Ошибка в самом тесте не должна приводить к реальному HTTP-запросу.
        self._patch("httpx.get", side_effect=AssertionError("HTTP must be mocked"))

    def _patch(self, name, *args, **kwargs):
        patcher = patch("places." + name, *args, **kwargs)
        mocked = patcher.start()
        self.addCleanup(patcher.stop)
        return mocked

    def record(self, name=None):
        return {"lat": 59.56, "lon": 150.80, "name": name or self.query}

    def response(self, data):
        response = Mock()
        response.json.return_value = data
        return response

    def stale_cache(self):
        self.cache[self.query.lower()] = self.record("Другой памятник, Магадан, Россия")
        self.cache[self.query.lower() + "|miss"] = {
            "checked_at": self.now - places.NEGATIVE_CACHE_TTL_SECONDS - 1,
            "reason": "not_found",
        }
        self.cache["unrelated"] = self.record("Другой город, Россия")
        return deepcopy(self.cache)

    def test_one_variant_miss_is_incomplete_without_negative_cache(self):
        with patch.object(places, "_search", return_value=None) as search:
            with self.assertRaises(places.GeocodeIncomplete) as caught:
                places.geocode(self.query, max_variants=1)
        self.assertIsInstance(caught.exception, places.GeocodeUnavailable)
        self.assertEqual(caught.exception.reason, "not_found")
        search.assert_called_once_with(self.query, validation_query=self.query)
        self.assertEqual(self.cache, {})
        self.save.assert_not_called()

    def test_mismatch_keeps_stale_positive_and_negative_cache_untouched(self):
        before = self.stale_cache()
        with patch.object(places, "_search", side_effect=places.GeocodeMismatch(self.query)) as search:
            with self.assertRaises(places.GeocodeIncomplete) as caught:
                places.geocode(self.query, max_variants=1)
        self.assertEqual(caught.exception.reason, "mismatch")
        search.assert_called_once()
        self.assertEqual(self.cache, before)
        self.save.assert_not_called()

    def test_wrong_identity_returned_by_search_is_incomplete_mismatch(self):
        wrong = self.record("Другой памятник, Магадан, Россия")
        with patch.object(places, "_search", return_value=wrong) as search:
            with self.assertRaises(places.GeocodeIncomplete) as caught:
                places.geocode(self.query, max_variants=1)
        self.assertEqual(caught.exception.reason, "mismatch")
        search.assert_called_once()
        self.assertEqual(self.cache, {})
        self.save.assert_not_called()

    def test_configured_variant_limit_is_respected(self):
        with patch.object(places, "_search", return_value=None) as search:
            with self.assertRaises(places.GeocodeIncomplete):
                places.geocode(self.query, max_variants=2)
        self.assertEqual(search.call_args_list, [
            call(variant, validation_query=self.query) for variant in self.variants[:2]
        ])
        self.save.assert_not_called()

    def test_http_failure_has_at_most_two_attempts_and_no_cache_mutation(self):
        before = self.stale_cache()
        error = httpx.ReadTimeout("offline test")
        with patch.object(places.httpx, "get", side_effect=error) as get:
            with self.assertRaises(places.GeocodeUnavailable) as caught:
                places.geocode(self.query, max_variants=1)
        self.assertNotIsInstance(caught.exception, places.GeocodeIncomplete)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(self.cache, before)
        self.save.assert_not_called()

    def test_second_http_attempt_can_confirm_and_cache_place(self):
        valid = self.response([{
            "lat": "59.56", "lon": "150.80", "display_name": self.query,
        }])
        with patch.object(places.httpx, "get", side_effect=[httpx.ConnectError("offline test"), valid]) as get:
            result = places.geocode(self.query, max_variants=1)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(result["name"], self.query)
        self.assertEqual(result["lat"], 59.56)
        self.assertEqual(self.cache[self.query.lower()], result)
        self.save.assert_called_once_with()

    def test_empty_http_response_is_one_attempt_without_full_negative(self):
        with patch.object(places.httpx, "get", return_value=self.response([])) as get:
            with self.assertRaises(places.GeocodeIncomplete) as caught:
                places.geocode(self.query, max_variants=1)
        self.assertEqual(caught.exception.reason, "not_found")
        get.assert_called_once()
        self.assertEqual(self.cache, {})
        self.save.assert_not_called()

    def test_mismatched_http_response_is_one_attempt_without_full_negative(self):
        wrong = self.response([{
            "lat": "59.56", "lon": "150.80", "display_name": "Другой памятник, Магадан, Россия",
        }])
        with patch.object(places.httpx, "get", return_value=wrong) as get:
            with self.assertRaises(places.GeocodeIncomplete) as caught:
                places.geocode(self.query, max_variants=1)
        self.assertEqual(caught.exception.reason, "mismatch")
        get.assert_called_once()
        self.assertEqual(self.cache, {})
        self.save.assert_not_called()

    def test_positive_limited_lookup_replaces_stale_cache_and_removes_miss(self):
        self.stale_cache()
        match = self.record()
        with patch.object(places, "_search", return_value=match) as search:
            self.assertEqual(places.geocode(self.query, max_variants=1), match)
        search.assert_called_once()
        self.assertEqual(self.cache[self.query.lower()], match)
        self.assertNotIn(self.query.lower() + "|miss", self.cache)
        self.assertIn("unrelated", self.cache)
        self.save.assert_called_once_with()

    def test_default_lookup_still_exhausts_variants_and_writes_negative(self):
        with patch.object(places, "_search", return_value=None) as search:
            self.assertIsNone(places.geocode(self.query))
        self.assertEqual(search.call_args_list, [
            call(variant, validation_query=self.query) for variant in self.variants
        ])
        self.assertEqual(self.cache, {
            self.query.lower() + "|miss": {"checked_at": self.now, "reason": "not_found"},
        })
        self.save.assert_called_once_with()

    def test_default_mismatch_still_removes_wrong_positive_and_writes_negative(self):
        self.stale_cache()
        with patch.object(places, "_search", return_value=None) as search:
            with self.assertRaises(places.GeocodeMismatch):
                places.geocode(self.query)
        self.assertEqual(search.call_count, len(self.variants))
        self.assertNotIn(self.query.lower(), self.cache)
        self.assertEqual(self.cache[self.query.lower() + "|miss"], {
            "checked_at": self.now, "reason": "mismatch",
        })
        self.assertIn("unrelated", self.cache)
        self.save.assert_called_once_with()

    def test_fresh_full_negative_cache_is_respected_without_search(self):
        for reason in ("not_found", "mismatch"):
            with self.subTest(reason=reason):
                self.cache.clear()
                self.cache[self.query.lower() + "|miss"] = {
                    "checked_at": self.now - 1, "reason": reason,
                }
                before = deepcopy(self.cache)
                with patch.object(places, "_search") as search:
                    if reason == "mismatch":
                        with self.assertRaises(places.GeocodeMismatch):
                            places.geocode(self.query, max_variants=1)
                    else:
                        self.assertIsNone(places.geocode(self.query, max_variants=1))
                search.assert_not_called()
                self.assertEqual(self.cache, before)
        self.save.assert_not_called()

    def test_confirmed_cache_is_respected_without_search(self):
        match = self.record()
        self.cache[self.query.lower()] = match
        with patch.object(places, "_search") as search:
            self.assertEqual(places.geocode(self.query, max_variants=1), match)
        search.assert_not_called()
        self.save.assert_not_called()

    def test_invalid_limit_is_rejected_before_cache_or_network(self):
        for limit in (0, -1, True, False, 1.5, "1", places.MAX_QUERY_VARIANTS + 1):
            with self.subTest(limit=limit):
                with patch.object(places, "_load_cache") as load, patch.object(places, "_search") as search:
                    with self.assertRaises(ValueError):
                        places.geocode(self.query, max_variants=limit)
                load.assert_not_called()
                search.assert_not_called()
        self.save.assert_not_called()

    def test_geometry_limited_miss_does_not_change_ordinary_cache(self):
        self.cache[self.query.lower()] = self.record()
        before = deepcopy(self.cache)
        with patch.object(places, "_search", return_value=None) as search:
            with self.assertRaises(places.GeocodeIncomplete):
                places.geocode(self.query, include_geometry=True, max_variants=1)
        search.assert_called_once_with(self.query, validation_query=self.query, include_geometry=True)
        self.assertEqual(self.cache, before)
        self.save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
