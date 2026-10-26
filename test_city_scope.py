"""Offline explicit-city-only evidence: proximity is not municipal membership."""

from copy import deepcopy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import city_candidates
from city_preferences import select_city_preferences
from city_quality import (CITY_ONLY_WARNING, check_city_quality, city_only_record_allowed,
                          city_only_requested)
import places
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def request(**changes):
    values = dict(destination="Kemer, Турция", start_date="2099-06-01", end_date="2099-06-01",
                  adults=2, children_ages=[], budget=60000, currency="RUB", pace="средний",
                  start_time="09:00", end_time="18:00", interest_categories=["parks"],
                  wishes="В пределах выбранного города")
    values.update(changes)
    return TripRequest(**values)


def record(*, city="Kemer", title="Kemer Kuğulu Park", number=1, **changes):
    value = dict(name=f"{title}, {city}, Türkiye", namedetails={"name": title},
                 address={"city": city, "state": "Antalya", "country": "Türkiye", "country_code": "tr"},
                 lat=36.60, lon=30.55, provider="osm", osm_type="way", osm_id=number,
                 category="leisure", type="park", locality=city,
                 place_query=f"{title}, {city}, Türkiye")
    value.update(changes)
    return value


def candidate(value):
    return city_candidates.CityCandidate(
        candidate_id="-".join(map(str, places.map_record_identity(value))),
        title=value["name"].split(",", 1)[0], place_query=value["place_query"],
        area=value["locality"], record=deepcopy(value))


class CityScopeTests(unittest.TestCase):
    def test_narrow_explicit_scope_phrases(self):
        phrases = ("в пределах выбранного города", "Только в городе, без музеев", "Строго в самом городе",
                   "в границах города", "в черте города", "не выезжать за пределы города",
                   "без выездов за город", "Only within the city", "within city limits",
                   "do not leave the city", "don't leave the city", "Без музеев, только в городе",
                   "Не нужны музеи, только в городе")
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertTrue(city_only_requested(request(wishes=phrase)))
                self.assertTrue(city_only_requested(request(wishes="", trip_goal=phrase)))

    def test_nearby_negated_and_complex_wishes_do_not_acquire_scope(self):
        for phrase in ("", "Город и окрестности", "Можно выехать за город", "не только в пределах города",
                       "не только в городе", "not only within the city", "не ограничиваться городом",
                       "Только музеи в городе", "Не хочу только в городе", "В пределах города не обязательно",
                       "Не обязательно в пределах города", "not limited to within city limits",
                       "Не нужно оставаться только в городе", "Не ограничивайте маршрут только в городе",
                       "Not staying only in the city"):
            with self.subTest(phrase=phrase):
                self.assertFalse(city_only_requested(request(wishes=phrase)))
                self.assertTrue(city_only_record_allowed(request(wishes=phrase), None))
        self.assertFalse(city_only_requested(request(destination_kind="region")))

    def test_negated_scope_preserves_candidates_and_full_semantic_wishes(self):
        pool = [candidate(record()), candidate(record(city="Muratpaşa", number=2))]
        identifiers = [item.candidate_id for item in pool]
        for phrase in ("Не нужно оставаться только в городе", "Не ограничивайте маршрут только в городе",
                       "Not staying only in the city"):
            with self.subTest(phrase=phrase):
                completion = Mock(return_value=SimpleNamespace(text=json.dumps(
                    {"status": "selected", "candidate_ids": identifiers})))
                result = select_city_preferences(request(wishes=phrase), pool, completion=completion)
                self.assertEqual(result.candidates, pool)
                self.assertEqual(result.status, "success")
                self.assertNotIn(CITY_ONLY_WARNING, result.warnings)
                completion.assert_called_once()
                data = json.loads(completion.call_args.args[0][1]["content"])
                self.assertEqual(data["request"]["wishes"], phrase)
                self.assertEqual([item["candidate_id"] for item in data["verified_candidates"]], identifiers)

    def test_exact_city_or_town_country_and_identity_are_needed(self):
        self.assertTrue(city_only_record_allowed(request(), record()))
        town = record()
        town["address"]["town"] = town["address"].pop("city")
        self.assertTrue(city_only_record_allowed(request(), town))
        before = deepcopy(town)
        self.assertTrue(city_only_record_allowed(request(), town))
        self.assertEqual(town, before)

    def test_independently_reviewed_geographic_aliases_keep_city_boundary(self):
        paris = record(city="Paris")
        paris["address"].update(country="France", country_code="fr")
        self.assertTrue(city_only_record_allowed(request(destination="Париж, Франция"), paris))
        # The reviewed municipality alias is independent of any POI name:ru.
        kemer = record(namedetails={"name": "Kemer Kuğulu Park", "name:ru": "Кемер"})
        self.assertTrue(city_only_record_allowed(request(destination="Кемер, Турция"), kemer))
        kemer["namedetails"].pop("name:ru")
        self.assertTrue(city_only_record_allowed(request(destination="Кемер, Турция"), kemer))
        kemer["address"]["city"] = "Antalya"
        self.assertFalse(city_only_record_allowed(request(destination="Кемер, Турция"), kemer))

    def test_actual_kemer_failure_shape_nearby_antalyas_are_not_city(self):
        # Same source-address shape as the live failure, synthetic identities.
        for city, title in (("Muratpaşa", "Baby mall"), ("Antalya", "Fener Parkı")):
            observed = record(city=city, title=title, lat=36.85, lon=30.75)
            self.assertLess(places.distance_km(36.60, 30.55, observed["lat"], observed["lon"]), 40)
            self.assertFalse(city_only_record_allowed(request(), observed))
            self.assertFalse(city_only_record_allowed(request(destination="Кемер, Турция"), observed))
            self.assertTrue(city_only_record_allowed(request(wishes="Город и окрестности"), observed))

    def test_district_county_suburb_or_record_labels_do_not_prove_city(self):
        for field in ("county", "municipality", "suburb", "city_district", "state"):
            observed = record()
            observed["address"].pop("city")
            observed["address"][field] = "Kemer"
            self.assertFalse(city_only_record_allowed(request(), observed), field)
        observed = record(city="Muratpaşa")
        observed.update(locality="Kemer", name="Kemer", place_query="Kemer, Kemer, Türkiye")
        observed["address"]["county"] = "Kemer"
        self.assertFalse(city_only_record_allowed(request(), observed))

    def test_conflicting_smaller_settlement_is_not_relabelled(self):
        for field in ("town", "village", "hamlet"):
            observed = record()
            observed["address"][field] = "Beldibi"
            self.assertFalse(city_only_record_allowed(request(), observed), field)

    def test_wrong_country_missing_country_and_identity_fail_closed(self):
        for address_change in ({"country_code": "ru", "country": "Россия"},
                               {"country_code": "ru"}, {"country": "Россия"},
                               {"country_code": ""}, {"country": ""}):
            observed = record()
            observed["address"].update(address_change)
            self.assertFalse(city_only_record_allowed(request(), observed), address_change)
        for changes in ({"osm_id": None}, {"provider": "invented"}, {"address": {}}, {"address": None}):
            self.assertFalse(city_only_record_allowed(request(), record(**changes)))
        self.assertFalse(city_only_record_allowed(request(destination="Kemer"), record()))

    def test_prefilter_keeps_all_original_wishes_for_semantics_and_not_mutated(self):
        local, outside = candidate(record()), candidate(record(city="Muratpaşa", number=2))
        trip = request(wishes="В пределах выбранного города; не посещать музеи")
        before = deepcopy((trip, local, outside))
        completion = Mock(return_value=SimpleNamespace(text=json.dumps(
            {"status": "selected", "candidate_ids": [local.candidate_id]})))
        result = select_city_preferences(trip, [local, outside], memory_context="Можно выехать за город",
                                         completion=completion)
        self.assertEqual(result.status, "success")
        self.assertEqual(result.candidates, [local])
        self.assertIn(CITY_ONLY_WARNING, result.warnings)
        completion.assert_called_once()
        data = json.loads(completion.call_args.args[0][1]["content"])
        self.assertEqual(data["request"]["wishes"], trip.wishes)
        self.assertEqual([item["candidate_id"] for item in data["verified_candidates"]], [local.candidate_id])
        self.assertEqual((trip, local, outside), before)

    def test_prefilter_empty_fails_before_paid_selection_and_rejects_ai_reinsertion(self):
        outside = candidate(record(city="Muratpaşa", number=2))
        completion = Mock(side_effect=AssertionError("No AI for an unconfirmed city pool"))
        result = select_city_preferences(request(), [outside], completion=completion)
        self.assertEqual(result.status, "insufficient_data")
        self.assertEqual(result.problems, [CITY_ONLY_WARNING])
        completion.assert_not_called()
        local = candidate(record())
        completion = Mock(return_value=SimpleNamespace(text=json.dumps(
            {"status": "selected", "candidate_ids": [outside.candidate_id]})))
        result = select_city_preferences(request(), [local, outside], completion=completion)
        self.assertEqual(result.status, "invalid_ai_response")
        self.assertEqual(result.candidates, [])

    def test_nearby_default_retains_full_structured_reserve(self):
        pool = [candidate(record()), candidate(record(city="Muratpaşa", number=2))]
        completion = Mock(side_effect=AssertionError("No structured AI"))
        result = select_city_preferences(request(wishes=""), pool, completion=completion)
        self.assertEqual(result.candidates, pool)
        self.assertEqual(result.status, "success")
        completion.assert_not_called()

    def test_final_validation_blocks_selector_bypass_or_unbound_record(self):
        observed = record(city="Muratpaşa", lat=36.85, lon=30.75)
        item = ItineraryItem(time="09:00", kind="attraction", title="Fener Parkı", duration_min=90,
                             description="Synthetic source-bound visit",
                             place_query=observed["place_query"], area="Muratpaşa", verified=True,
                             lat=observed["lat"], lon=observed["lon"], provider="osm", indoor=False)
        itinerary = Itinerary(title="Test", summary="", days=[
            ItineraryDay(day=1, date="2099-06-01", theme="", items=[item])])
        bound = {item.place_query: observed}
        self.assertTrue(any("принадлежность" in value for value in check_city_quality(itinerary, request(), bound)))
        self.assertEqual(check_city_quality(itinerary, request(wishes="Город и окрестности"), bound), [])
        observed["address"]["city"] = "Kemer"
        self.assertEqual(check_city_quality(itinerary, request(), bound), [])
        observed["lat"] += .001  # No passing source address detached from item identity.
        self.assertTrue(any("принадлежность" in value for value in check_city_quality(itinerary, request(), bound)))

    def test_discovery_filters_before_usable_pool_without_new_http_or_ai_budget(self):
        center = {"name": "Kemer", "country_code": "TR", "country": "Türkiye", "lat": 36.60, "lon": 30.55}
        local = record()
        outside = record(city="Muratpaşa", title="Fener Parkı", number=2, lat=36.85, lon=30.75)
        proposals = [{"title": value["name"].split(",", 1)[0], "record": value,
                      "categories": ["parks"], "identity": places.map_record_identity(value)}
                     for value in (outside, local)]
        completion = Mock(return_value=SimpleNamespace(text='{"titles": []}', input_tokens=0, output_tokens=0))
        with patch.object(city_candidates, "destination_name_hints", return_value=[]), \
             patch.object(city_candidates, "_map_name_hints", return_value=proposals), \
             patch.object(city_candidates, "_category_search", return_value=[]), \
             patch.object(city_candidates.httpx, "get", side_effect=AssertionError("No real HTTP")):
            result = city_candidates.ground_city_candidates(request(), center, completion=completion,
                                                            allow_partial_program=True)
        self.assertEqual(result.status, "success")
        self.assertEqual([item.title for item in result.candidates], ["Kemer Kuğulu Park"])
        self.assertIn(CITY_ONLY_WARNING, result.warnings)
        self.assertEqual(result.diagnostics["detail_rejection_reasons"]["city_only_membership_unconfirmed"], 1)
        self.assertLessEqual(result.map_requests, 36)
        self.assertLessEqual(result.diagnostics["name_limit"], 36)
        self.assertLessEqual(result.diagnostics["usable_limit"], 24)
        self.assertLessEqual(completion.call_count, 1)


if __name__ == "__main__":
    unittest.main()
