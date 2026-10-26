"""Independent offline invariants for the bounded regional discovery bridge."""

from copy import deepcopy
from dataclasses import replace
import unittest
from unittest.mock import patch

import candidate_grounding as grounding
import regional_discovery as discovery
from test_regional_discovery import base, counter, record, req, scope


class RegionalReserveInvariantTests(unittest.TestCase):
    def setUp(self):
        for path in ("httpx.get", "httpx.post", "sqlite3.connect", "places._load_cache",
                     "places._save_cache", "places._remember_nearby_alias", "grounding.generate_json"):
            if path == "grounding.generate_json":
                blocker = patch.object(grounding, "generate_json", side_effect=AssertionError("No external I/O"))
            else:
                blocker = patch(path, side_effect=AssertionError("No external I/O"))
            blocker.start()
            self.addCleanup(blocker.stop)

    def test_full_polygon_hole_is_not_replaced_by_its_bounding_box(self):
        area = scope()
        geometry = deepcopy(area.geometry)
        geometry["coordinates"].append([[100.5,50.5],[101.5,50.5],[101.5,51.5],[100.5,51.5],[100.5,50.5]])
        area = replace(area, geometry=geometry)
        self.assertFalse(discovery.inside_admin(record(), area))
        self.assertTrue(discovery.inside_admin(record(lon=102.), area))

    def test_gap_between_multipolygon_parts_is_not_an_inside_point(self):
        geometry = {"type":"MultiPolygon", "coordinates":[
            [[[100,50],[100.5,50],[100.5,53],[100,53],[100,50]]],
            [[[102,50],[103,50],[103,53],[102,53],[102,50]]],
        ]}
        area = replace(scope(), geometry=geometry)
        self.assertFalse(discovery.inside_admin(record(), area))
        self.assertTrue(discovery.inside_admin(record(lon=102.5), area))

    def test_contradicting_country_label_cannot_reenter_via_regional_exception(self):
        for side in ("base", "poi"):
            left = base()
            right = record("Музей культуры", kind="museum", identity=200)
            if side == "base":
                left["map_address"]["country"] = "France"
            else:
                right["address"]["country"] = "France"
            with self.subTest(side=side):
                self.assertFalse(grounding._administratively_compatible(left["map_address"], right["address"]))
                self.assertFalse(grounding._regional_base_compatible(left, right, scope()))

    def pool(self):
        return {key:base(record(f"Село {index}",kind="village",identity=index+1,lon=101+.01*index),key=key)
                for index,key in enumerate(grounding.BASE_KEYS)}

    def test_only_unused_base_key_can_be_replaced_and_old_pois_stay_byte_equal(self):
        bases = self.pool()
        result = counter()
        result.base_facts = deepcopy(list(bases.values()))
        result.confirmed_hints = [{"title":"Старинная церковь", "base_key":"base_1",
                                   "allowed_base_keys":["base_1", "base_2"]}]
        before = deepcopy(result.confirmed_hints)
        old_bases = deepcopy(bases)
        town = {**record(), "base_query":"Большой город, Россия"}
        with patch.object(discovery, "discover_town", return_value=town), \
                patch.object(discovery, "category_records", return_value=[]):
            grounding._regional_map_reserve(req(), scope(), bases, result, 18)
        self.assertEqual(result.confirmed_hints, before)
        self.assertEqual(bases["base_1"], old_bases["base_1"])
        self.assertEqual(bases["base_2"], old_bases["base_2"])
        self.assertEqual(bases["base_3"]["base_query"], "Большой город, Россия")
        self.assertEqual(len(result.base_facts), 3)
        self.assertEqual({item["base_key"] for item in result.base_facts}, set(grounding.BASE_KEYS))

    def test_no_replacement_when_all_base_keys_already_support_confirmed_pois(self):
        bases = self.pool()
        result = counter()
        result.base_facts = deepcopy(list(bases.values()))
        result.confirmed_hints = [{"title":"Старинная церковь", "base_key":"base_1",
                                   "allowed_base_keys":list(grounding.BASE_KEYS)}]
        before = deepcopy((bases, result.base_facts, result.confirmed_hints))
        with patch.object(discovery, "discover_town") as town, \
                patch.object(discovery, "category_records", return_value=[]):
            grounding._regional_map_reserve(req(), scope(), bases, result, 18)
        town.assert_not_called()
        self.assertEqual((bases, result.base_facts, result.confirmed_hints), before)

    def test_exhausted_http_budget_makes_no_network_or_cache_call(self):
        result = counter(42,42)
        self.assertIsNone(discovery.discover_town(scope(), result))
        self.assertEqual(discovery.category_records(req(), scope(), base(), result), [])
        self.assertEqual((result.reserved_http_attempts_upper_bound, result.regional_discovery_calls), (42,0))


if __name__ == "__main__":
    unittest.main()
