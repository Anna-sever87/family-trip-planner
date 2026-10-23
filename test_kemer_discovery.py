"""Observed Kemer discovery regressions; no network or runtime data writes.

Public Nominatim search observations, 2026-10-22, countrycodes=tr,
bounded=1, 40-km viewbox about (36.59778, 30.56056), tr,en:
Phaselis Orenyeri (with the original Turkish diacritic) -> [];
Phaselis -> relation/9978345 below. This is map identity, not VisitFacts.
The ministry also uses the standalone proper name Phaselis:
https://www.muze.gov.tr/muze-detay?DistId=PHS&SectionId=PHS01
The municipality identifies the same Mustafa Ertugrul Aker park as Olbia:
https://www.kemer.bel.tr/tr/haberler/mustafa-ertugrul-aker-parki-sil-bastan-yapildi.html
These reviewed names replace seeds; they do not loosen the global matcher.
"""

from contextlib import nullcontext
from copy import deepcopy
import json
import socket
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import city_candidates
import places
from city_candidates import CityCandidateResult
from schemas import TripRequest


CENTER = {"name": "Кемер", "lat": 36.59778, "lon": 30.56056,
          "country_code": "tr", "admin1": "Анталья"}
PHASELIS = {
    "osm_type": "relation", "osm_id": 9978345,
    "lat": "36.5228115", "lon": "30.5527365",
    "display_name": "Faselis, Main Street, Tekirova, Kemer, Antalya, Akdeniz Bölgesi, Türkiye",
    "namedetails": {"name": "Faselis", "name:de": "Phaselis", "name:en": "Phaselis Ancient City",
                    "name:fr": "Phaselis", "name:la": "Phaselis", "name:tr": "Faselis",
                    "alt_name": "Phaselis Antik Kenti", "old_name": "Faseliz Harabesi"},
    "address": {"historic": "Faselis", "road": "Main Street", "village": "Tekirova",
                "town": "Kemer", "province": "Antalya", "ISO3166-2-lvl4": "TR-07",
                "region": "Akdeniz Bölgesi", "country": "Türkiye", "country_code": "tr"},
    "class": "historic", "type": "archaeological_site", "addresstype": "historic",
}
YORUK = {
    "osm_type": "way", "osm_id": 176919554,
    "lat": "36.6004167", "lon": "30.5764295",
    "display_name": "Folklorik Yörük Parkı, Merkez, Kemer, Antalya, Akdeniz Bölgesi, Türkiye",
    "namedetails": {"name": "Folklorik Yörük Parkı"},
    "address": {"park": "Folklorik Yörük Parkı", "suburb": "Merkez", "town": "Kemer",
                "province": "Antalya", "region": "Akdeniz Bölgesi",
                "country": "Türkiye", "country_code": "tr"},
    "class": "leisure", "type": "park", "addresstype": "park",
}
OLBIA = {
    "osm_type": "way", "osm_id": 176923510,
    "lat": "36.6005668", "lon": "30.5685221",
    "display_name": "Olbia, Merkez, Kemer, Antalya, Akdeniz Bölgesi, Türkiye",
    "namedetails": {"name": "Olbia"},
    "address": {"park": "Olbia", "suburb": "Merkez", "town": "Kemer",
                "province": "Antalya", "ISO3166-2-lvl4": "TR-07",
                "region": "Akdeniz Bölgesi", "country": "Türkiye", "country_code": "tr"},
    "class": "leisure", "type": "park", "addresstype": "park",
}


class KemerDiscoveryTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "getaddrinfo"), (sqlite3, "connect")):
            blocker = patch.object(owner, name, side_effect=AssertionError("No external I/O"))
            blocker.start()
            self.addCleanup(blocker.stop)
        for target, value in (("places._cached_nearby_named", None),
                              ("places._wait_for_turn", nullcontext())):
            blocker = patch(target, return_value=value)
            blocker.start()
            self.addCleanup(blocker.stop)

    @staticmethod
    def response(records):
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: deepcopy(records),
                               content=json.dumps(records, ensure_ascii=False).encode("utf-8"))

    def test_empty_exact_search_is_discovery_miss_not_rejection_by_scope(self):
        result = CityCandidateResult()
        with patch("city_candidates.httpx.get", return_value=self.response([])) as get:
            self.assertIsNone(city_candidates._lookup("Phaselis Örenyeri", "Турция", CENTER, result))
        self.assertEqual((get.call_count, result.map_requests), (1, 1))
        self.assertEqual(get.call_args.kwargs["params"]["q"], "Phaselis Örenyeri, Турция")
        self.assertEqual(result.rejected, [{"title": "Phaselis Örenyeri", "reason": "limited_search_unconfirmed",
                                           "detail": "empty_result", "record_reasons": ""}])

    def test_observed_provider_alias_passes_unchanged_identity_and_locality_guards(self):
        before = deepcopy(PHASELIS)
        record = city_candidates._from_provider(PHASELIS)
        self.assertIsNone(city_candidates._canonical_record("Phaselis Örenyeri", record, CENTER))
        canonical = city_candidates._canonical_record("Phaselis", record, CENTER)
        self.assertIsNotNone(canonical)
        self.assertEqual(places.map_record_identity(canonical), ("osm", "relation", 9978345))
        self.assertEqual(canonical["place_query"], "Faselis, Tekirova, Türkiye")
        self.assertEqual(canonical["locality"], "Tekirova")
        self.assertAlmostEqual(places.distance_km(CENTER["lat"], CENTER["lon"],
                                                canonical["lat"], canonical["lon"]), 8.36535149)
        self.assertNotIn("indoor", canonical)
        self.assertNotIn("cost", canonical)
        self.assertEqual(PHASELIS, before)

    def test_reviewed_name_changes_one_lookup_not_country_scope_or_guards(self):
        for mutation in ("country", "radius", "peak", "identity"):
            raw = deepcopy(PHASELIS)
            if mutation == "country":
                raw["address"].update(country="Greece", country_code="gr")
            elif mutation == "radius":
                raw["lat"] = "38.0"
            elif mutation == "peak":
                raw["type"] = "peak"
            else:
                raw.pop("osm_id")
            with self.subTest(mutation=mutation):
                self.assertIsNone(city_candidates._canonical_record(
                    "Phaselis", city_candidates._from_provider(raw), CENTER))

    def test_olbia_is_one_park_not_the_same_named_confectionery(self):
        canonical = city_candidates._canonical_record("Olbia", city_candidates._from_provider(OLBIA), CENTER)
        self.assertEqual(places.map_record_identity(canonical), ("osm", "way", 176923510))
        self.assertEqual(canonical["place_query"], "Olbia, Kemer, Türkiye")
        # This second result was observed in the same bounded response.
        shop = deepcopy(OLBIA)
        shop.update(osm_type="node", osm_id=10154841690, lat="36.8578829", lon="30.6013241",
                    display_name="Olbia, 252. Sokak, Hurma Mahallesi, Konyaaltı, Antalya, Akdeniz Bölgesi, 07130, Türkiye",
                    type="confectionery", addresstype="shop", **{"class": "shop"})
        shop["address"] = {"shop": "Olbia", "road": "252. Sokak", "suburb": "Hurma Mahallesi",
                           "town": "Konyaaltı", "province": "Antalya", "region": "Akdeniz Bölgesi",
                           "postcode": "07130", "country": "Türkiye", "country_code": "tr"}
        self.assertIsNone(city_candidates._canonical_record("Olbia", city_candidates._from_provider(shop), CENTER))

    def test_reviewed_catalogue_keeps_one_seed_per_site_and_primary_urls(self):
        from destination_name_hints import destination_name_hints
        hints = destination_name_hints("Кемер, Анталья, Турция")
        self.assertEqual([hint["title"] for hint in hints],
                         ["Phaselis", "Folklorik Yörük Parkı", "Olbia", "Ayışığı Parkı"])
        self.assertEqual(hints[0]["source_url"], "https://www.muze.gov.tr/muze-detay?DistId=PHS&SectionId=PHS01")
        self.assertEqual(hints[2]["source_url"],
                         "https://www.kemer.bel.tr/tr/haberler/mustafa-ertugrul-aker-parki-sil-bastan-yapildi.html")
        self.assertTrue(all(set(hint) == {"title", "place_query", "locality", "source_url"} for hint in hints))

    def test_policy4_names_replay_one_and_reviewed_names_three_unique_map_records(self):
        request = TripRequest(destination="Кемер, Анталья, Турция", start_date="2026-10-28",
                              end_date="2026-10-31", adults=2, children_ages=[], budget=2500,
                              travel_cost=1200, lodging_cost=500, currency="EUR", pace="средний",
                              start_time="09:00", end_time="19:00")
        for name, park, expected in (("Phaselis Örenyeri", "Mustafa Ertuğrul Aker Parkı", 1),
                                     ("Phaselis", "Mustafa Ertuğrul Aker Parkı", 2),
                                     ("Phaselis", "Olbia", 3)):
            titles = [name, "Folklorik Yörük Parkı", park, "Ayışığı Parkı"]
            hints = [{"title": title, "source_url": "https://www.muze.gov.tr/"} for title in titles]
            proposals = SimpleNamespace(text=json.dumps({"titles": titles}), input_tokens=1, output_tokens=1)
            # Provider q above was title-only with countrycodes=tr. This mock
            # tests canonical pipeline behavior, not a claim that the runtime
            # country-suffixed query was additionally sent over the network.
            observations = {"Phaselis, Турция": [PHASELIS], "Folklorik Yörük Parkı, Турция": [YORUK],
                            "Olbia, Турция": [OLBIA]}
            def get(_url, *, params, **_kwargs):
                if _url == city_candidates._SHOPPING_DISCOVERY_URL:
                    # This replay isolates the historical name evidence, not
                    # present-day category availability. The real primary and
                    # reserve stages still run and count their HTTP sends.
                    return self.response({"elements": []})
                self.assertEqual(params["countrycodes"], "tr")
                self.assertEqual(params["bounded"], 1)
                return self.response(observations.get(params["q"], []))
            with self.subTest(name=name, park=park), patch("city_candidates.destination_name_hints", return_value=hints), \
                    patch("city_candidates.httpx.get", side_effect=get) as http, \
                    patch("places._save_cache", side_effect=AssertionError("No cache writes")):
                result = city_candidates.ground_city_candidates(request, CENTER,
                    completion=lambda *_args, **_kwargs: proposals, allow_partial_program=True)
            # One Overpass + three default category searches + four seed names.
            self.assertEqual((len(result.candidates), result.map_requests, http.call_count), (expected, 8, 8))
            queries = [call.kwargs["params"].get("q") for call in http.call_args_list]
            self.assertEqual(queries[1:4], ["[attraction]", "[museum]", "[park]"])
            self.assertEqual(sum(query in observations for query in queries), expected)
            self.assertEqual(len({c.candidate_id for c in result.candidates}), expected)
            self.assertEqual(result.problems, [])
            # A larger identity pool is not a claim of feasible days/access.
            self.assertTrue(all("indoor" not in c.record for c in result.candidates))

    def test_added_identity_does_not_bypass_eight_km_daily_pair_limit(self):
        from city_candidates import CityCandidate
        from city_planner import _incompatible_same_day_pairs
        pool = []
        for title, raw in (("Phaselis", PHASELIS), ("Folklorik Yörük Parkı", YORUK), ("Olbia", OLBIA)):
            record = city_candidates._canonical_record(title, city_candidates._from_provider(raw), CENTER)
            pool.append(CityCandidate("-".join(map(str, places.map_record_identity(record))),
                                      record["place_query"].split(",", 1)[0], record["place_query"],
                                      record["locality"], record))
        pairs = [set(pair["candidate_ids"]) for pair in _incompatible_same_day_pairs(pool)]
        self.assertIn({"osm-relation-9978345", "osm-way-176919554"}, pairs)
        self.assertNotIn({"osm-way-176923510", "osm-way-176919554"}, pairs)


if __name__ == "__main__":
    unittest.main()
