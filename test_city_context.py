"""Map-only edit snapshots: exact scope, identity, isolation and bounded recovery."""

from copy import deepcopy
import unittest
from unittest.mock import patch

import city_context
from city_candidates import CityCandidate, _canonical_record
from places import verify_itinerary
from schemas import Itinerary, ItineraryDay, ItineraryItem
from test_city_candidate_planner import request, default_candidates


class CityContextTests(unittest.TestCase):
    def setUp(self):
        self.request = request()
        self.center = dict(name="Тестгород", lat=55.0, lon=83.0, country_code="RU")
        self.pool = default_candidates()
        self.original = Itinerary(title="Test", summary="", city_schedule=True, days=[
            ItineraryDay(day=1, date=self.request.start_date, theme="Test", items=[
                ItineraryItem(time="09:00", title=c.title, area=c.area, place_query=c.place_query,
                              description="Осмотр", duration_min=90, indoor=True)
                for c in self.pool])])
        self.assertEqual(verify_itinerary(self.original,
                         verified_records={c.place_query: c.record for c in self.pool}), [])

    def build(self):
        return city_context.build_city_context(self.request, self.pool, self.center)

    def test_roundtrip_bound_snapshot_is_not_aliased_to_input(self):
        before = deepcopy(self.pool)
        with patch("httpx.get", side_effect=AssertionError("No HTTP")):
            payload = self.build()
            pool, center = city_context.load_city_context(payload, self.request)
            city_context.check_context_places(self.original, pool)
        self.assertEqual(self.pool, before)
        self.assertEqual(center["country_code"], "ru")
        pool[0].record["lat"] = 0
        self.assertEqual(payload["candidates"][0]["record"]["lat"], 55.0)

    def test_foreign_or_modified_request_is_rejected(self):
        for changes in ({"destination": "Другой город, Россия"}, {"budget": 31000},
                        {"children_ages": [5]}, {"start_date": "2099-10-27"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                city_context.load_city_context(self.build(), self.request.model_copy(update=changes))

    def test_bad_snapshot_fields_rejected(self):
        for changes in ({"version": True}, {"version": 2}, {"captured_at": "2099-01-01"},
                        {"pool_scope": "model"}, {"candidates": []}, {"unexpected": True}):
            payload = self.build()
            payload.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                city_context.load_city_context(payload, self.request)

    def test_candidate_identity_or_bound_name_cannot_be_changed(self):
        for field, value in (("candidate_id", "osm-node-999"), ("title", "Другой музей"),
                             ("place_query", "Другой музей, Тестгород, Россия"), ("area", "Иной город")):
            payload = self.build()
            payload["candidates"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                city_context.load_city_context(payload, self.request)

    def test_duplicate_pool_and_invalid_coordinates_rejected(self):
        payload = self.build()
        payload["candidates"][1] = deepcopy(payload["candidates"][0])
        with self.assertRaises(ValueError):
            city_context.load_city_context(payload, self.request)
        for value in (True, float("nan"), 91):
            center = dict(self.center, lat=value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                city_context.build_city_context(self.request, self.pool, center)

    def test_selected_place_exact_location_and_provenance_required(self):
        for changes in ({"lat": 55.000001}, {"verified": False}, {"provider": "2gis"},
                        {"official_name": "New name"}, {"title": "New title"}, {"area": "Другой город"}):
            itinerary = self.original.model_copy(deep=True)
            itinerary.days[0].items[0] = itinerary.days[0].items[0].model_copy(update=changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                city_context.check_context_places(itinerary, self.pool)

    def test_recovery_only_rechecks_existing_visits_not_coordinate_urls(self):
        calls = []
        def lookup(title, country, center, budget):
            calls.append(title)
            candidate = next(c for c in self.pool if c.title == title)
            return _canonical_record(title, candidate.record, center)
        before = self.original.model_dump()
        payload = city_context.recover_city_context(self.request, self.original,
                                                    resolved_city=self.center, lookup=lookup)
        self.assertEqual(calls, [c.title for c in self.pool])
        self.assertEqual(payload["pool_scope"], "existing_visits_rechecked")
        self.assertEqual(self.original.model_dump(), before)
        pool, _ = city_context.load_city_context(payload, self.request)
        city_context.check_context_places(self.original, pool)

    def test_recovery_does_not_treat_empty_or_moved_lookup_as_same_place(self):
        with self.assertRaises(ValueError):
            city_context.recover_city_context(self.request, self.original,
                                              resolved_city=self.center, lookup=lambda *args: None)
        def moved(title, country, center, budget):
            candidate = next(c for c in self.pool if c.title == title)
            record = _canonical_record(title, candidate.record, center)
            record["lat"] += .001
            return record
        with self.assertRaises(ValueError):
            city_context.recover_city_context(self.request, self.original,
                                              resolved_city=self.center, lookup=moved)

    def test_bad_existing_snapshot_cannot_fall_back_to_network(self):
        with patch.object(city_context, "recover_city_context") as recover, self.assertRaises(ValueError):
            city_context.prepare_stored_city_context({city_context.CONTEXT_KEY: {}}, self.request, self.original)
        recover.assert_not_called()

    def unknown_country_fixture(self):
        request = self.request.model_copy(update={"destination": "Helsinki, Finland"})
        center = dict(name="Helsinki", lat=60.1699, lon=24.9384, country_code="FI", country="Finland")
        title = "Helsinki City Museum"
        query = f"{title}, Helsinki, Finland"
        candidate = CityCandidate("osm-node-201", title, query, "Helsinki", dict(
            name=query, lat=center["lat"], lon=center["lon"], type="museum", provider="osm",
            osm_type="node", osm_id=201,
            address={"city": "Helsinki", "country": "Finland", "country_code": "fi"},
        ))
        return request, center, [candidate]

    def test_exact_provider_country_fallback_roundtrips_without_alias_or_lookup(self):
        request, center, candidates = self.unknown_country_fixture()
        before = deepcopy((center, candidates))
        with patch("httpx.get", side_effect=AssertionError("No HTTP")) as http, \
                patch("places.geocode", side_effect=AssertionError("No geocode")) as geocode:
            payload = city_context.build_city_context(request, candidates, center)
            loaded, resolved = city_context.load_city_context(payload, request)
        http.assert_not_called()
        geocode.assert_not_called()
        self.assertEqual((center, candidates), before)
        self.assertEqual(resolved["country"], "Finland")
        self.assertEqual(resolved["country_code"], "fi")
        self.assertEqual(loaded[0].candidate_id, "osm-node-201")
        self.assertEqual(loaded[0].place_query, candidates[0].place_query)

    def test_unknown_country_requires_matching_provider_label_not_guessed_translation(self):
        request, center, candidates = self.unknown_country_fixture()
        for label in (None, "", "Sweden", "Finnland", "Финляндия", "Finlandia", {}, "Finland\n"):
            changed = deepcopy(center)
            if label is None:
                changed.pop("country")
            else:
                changed["country"] = label
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "Не удалось подтвердить"):
                city_context.build_city_context(request, candidates, changed)
        payload = city_context.build_city_context(request, candidates, center)
        payload["resolved_city"]["country"] = "Sweden"
        with self.assertRaises(ValueError):
            city_context.load_city_context(payload, request)

    def test_country_fallback_keeps_existing_exact_case_normalization_and_admin_binding(self):
        request, center, candidates = self.unknown_country_fixture()
        request = request.model_copy(update={"destination": "Helsinki, Uusimaa, FINLAND"})
        center["admin1"] = "Uusimaa"
        payload = city_context.build_city_context(request, candidates, center)
        self.assertEqual(payload["resolved_city"]["country"], "Finland")
        center["admin1"] = "Lapland"
        with self.assertRaises(ValueError):
            city_context.build_city_context(request, candidates, center)

    def test_known_country_code_cannot_be_overridden_by_provider_label(self):
        center = dict(self.center, country_code="US", country="Россия")
        with self.assertRaises(ValueError):
            city_context.build_city_context(self.request, self.pool, center)
        # Old known-country snapshots did not contain a provider label.
        payload = self.build()
        self.assertNotIn("country", payload["resolved_city"])
        loaded, center = city_context.load_city_context(payload, self.request)
        self.assertEqual(len(loaded), len(self.pool))
        self.assertEqual(center["country_code"], "ru")

    def test_huge_integer_coordinates_fail_with_safe_value_error_not_overflow(self):
        for key in ("lat", "lon"):
            for value in (10 ** 1000, -(10 ** 1000)):
                center = dict(self.center, **{key: value})
                with self.subTest(key=key, positive=value > 0):
                    with self.assertRaisesRegex(ValueError, "Не удалось подтвердить"):
                        city_context.build_city_context(self.request, self.pool, center)
                    payload = self.build()
                    payload["resolved_city"][key] = value
                    with self.assertRaisesRegex(ValueError, "Не удалось подтвердить"):
                        city_context.load_city_context(payload, self.request)

    def test_context_failure_does_not_claim_that_no_paid_call_occurred(self):
        self.assertIn("Исходная поездка сохранена без изменений", city_context.CONTEXT_ERROR)
        self.assertNotIn("платное", city_context.CONTEXT_ERROR)
        self.assertNotIn("не запущено", city_context.CONTEXT_ERROR)


if __name__ == "__main__":
    unittest.main()
