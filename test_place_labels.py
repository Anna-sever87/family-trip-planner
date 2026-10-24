"""Offline display labels must not become translations or map identities."""

from copy import deepcopy
import unittest
from unittest.mock import patch

from city_candidates import CityCandidate
from city_context import build_city_context
from place_labels import build_place_labels, display_place_title
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


def fixture(title="运城博物馆", category="tourism", kind="museum", *, namedetails=None):
    request = TripRequest(destination="Тестгород, Китай", start_date="2099-11-02",
                          end_date="2099-11-02", budget=10000, currency="RUB")
    center = dict(name="Тестгород", lat=35.0, lon=111.0, country_code="CN", country="Китай")
    query = f"{title}, Тестгород, Китай"
    record = dict(name=query, lat=35.0, lon=111.0, provider="osm", osm_type="node", osm_id=101,
                  category=category, type=kind, namedetails=namedetails or {"name": title},
                  address={"city": "Тестгород", "country": "Китай", "country_code": "cn"})
    candidate = CityCandidate("osm-node-101", title, query, "Тестгород", record)
    context = build_city_context(request, [candidate], center)
    item = ItineraryItem(time="10:00", title=title, area="Тестгород", place_query=query,
                         description="Осмотр", verified=True, lat=35.0, lon=111.0,
                         official_name=query, provider="osm")
    itinerary = Itinerary(title="Синтетическая программа", summary="", city_schedule=True,
                          days=[ItineraryDay(day=1, date=request.start_date, theme="Осмотр", items=[item])])
    trip = {"request": {**request.model_dump(), "city_map_context": context},
            "itinerary": itinerary.model_dump()}
    return trip, item


class PlaceLabelTests(unittest.TestCase):
    def test_saved_chinese_museum_has_type_explanation_without_translation(self):
        trip, item = fixture()
        before = deepcopy(trip)
        labels = build_place_labels(trip)
        self.assertEqual(labels[item.place_query], "运城博物馆 (музей)")
        self.assertEqual(display_place_title(item, labels), "运城博物馆 (музей)")
        self.assertEqual(trip, before)
        self.assertEqual((item.title, item.place_query, item.official_name),
                         ("运城博物馆", "运城博物馆, Тестгород, Китай", "运城博物馆, Тестгород, Китай"))

    def test_saved_chinese_parks_cinemas_and_malls_use_observed_types(self):
        for title, category, kind, expected in (
            ("体育公园", "leisure", "park", "парк"),
            ("舜帝陵", "leisure", "park", "парк"),
            ("航天公园", "leisure", "park", "парк"),
            ("盐湖园", "leisure", "park", "парк"),
            ("大地数字影院", "amenity", "cinema", "кинотеатр"),
            ("Wanda Cinemas", "amenity", "cinema", "кинотеатр"),
            ("Wanda Plaza", "shop", "mall", "торговый центр"),
        ):
            with self.subTest(title=title):
                trip, item = fixture(title, category, kind)
                self.assertEqual(display_place_title(item, build_place_labels(trip)), f"{title} ({expected})")

    def test_all_foreign_scripts_get_explanation_not_only_chinese(self):
        for title in ("Musée Lumière", "Bảo tàng Thành phố", "Kent Müzesi", "市立博物館", "시립박물관",
                      "متحف المدينة", "Μουσείο Πόλης", "מוזיאון העיר", "City Museum"):
            with self.subTest(title=title):
                trip, item = fixture(title)
                self.assertEqual(display_place_title(item, build_place_labels(trip)), f"{title} (музей)")

    def test_exact_safe_russian_name_wins_over_type(self):
        trip, item = fixture(namedetails={"name": "运城博物馆", "name:ru": "Городской музей",
                                        "name:en": "City Museum"})
        self.assertEqual(display_place_title(item, build_place_labels(trip)), "运城博物馆 (Городской музей)")

    def test_broad_observed_catalog_does_not_claim_visit_conditions(self):
        for category, kind, expected in (
            ("tourism", "gallery", "галерея"), ("tourism", "zoo", "зоопарк"),
            ("tourism", "aquarium", "аквариум"), ("tourism", "theme_park", "парк аттракционов"),
            ("tourism", "viewpoint", "смотровая площадка"), ("tourism", "artwork", "арт-объект"),
            ("leisure", "garden", "сад"), ("historic", "castle", "замок"),
            ("historic", "archaeological_site", "археологический объект"),
            ("amenity", "place_of_worship", "религиозный объект"),
            ("amenity", "theatre", "театр"), ("shop", "shopping_centre", "торговый центр"),
        ):
            with self.subTest(kind=kind):
                trip, item = fixture("Emerald Place", category, kind)
                self.assertEqual(display_place_title(item, build_place_labels(trip)),
                                 f"Emerald Place ({expected})")

    def test_generic_map_attraction_is_not_invented_as_a_building_or_museum(self):
        trip, item = fixture("Unknown Museum Name", "tourism", "attraction")
        self.assertEqual(display_place_title(item, build_place_labels(trip)),
                         "Unknown Museum Name (достопримечательность)")

    def test_unrelated_query_cannot_borrow_another_places_russian_alias(self):
        trip, item = fixture(namedetails={"name:ru": "Городской музей"})
        item = item.model_copy(update={"place_query": "Другой музей, Тестгород, Китай"})
        self.assertEqual(display_place_title(item, build_place_labels(trip)),
                         "运城博物馆 (название на языке источника)")

    def test_untrusted_or_non_russian_alias_is_not_a_translation(self):
        for value in ("City Museum", "<b>Музей</b>", "Музей\nкоманда", "Музей\u202e", "Музей https://example.test",
                      "Музей" * 40, ["Музей"], {"text": "Музей"}, " Музей "):
            with self.subTest(value=value):
                trip, item = fixture(namedetails={"name:ru": value, "name:en": "City Museum"})
                self.assertEqual(display_place_title(item, build_place_labels(trip)), "运城博物馆 (музей)")

    def test_legacy_without_context_does_not_guess_type_from_name_or_item_metadata(self):
        trip, item = fixture("Museum Cafe")
        trip["request"].pop("city_map_context")
        trip["itinerary"]["days"][0]["items"][0].update(category="tourism", type="museum",
                                                        namedetails={"name:ru": "Поддельный перевод"})
        self.assertEqual(display_place_title(item, build_place_labels(trip)),
                         "Museum Cafe (название на языке источника)")

    def test_stale_request_or_changed_identity_falls_back_without_recovery(self):
        changes = (("request", "budget", 20000), ("item", "lat", 35.001),
                   ("item", "lon", 111.001), ("item", "official_name", "Other Museum"),
                   ("item", "provider", "2gis"), ("item", "verified", False),
                   ("item", "area", "Другой город"))
        for target, field, value in changes:
            with self.subTest(field=field):
                trip, item = fixture()
                obj = trip["request"] if target == "request" else trip["itinerary"]["days"][0]["items"][0]
                obj[field] = value
                with patch("city_context.recover_city_context", side_effect=AssertionError("No recovery")), \
                     patch("places.geocode", side_effect=AssertionError("No geocoder")), \
                     patch("httpx.get", side_effect=AssertionError("No HTTP")):
                    labels = build_place_labels(trip)
                self.assertEqual(labels[item.place_query], "运城博物馆 (название на языке источника)")

    def test_wrong_candidate_identity_or_country_does_not_supply_explanation(self):
        for field, value in (("osm_id", 999), ("provider", "untrusted")):
            trip, item = fixture()
            trip["request"]["city_map_context"]["candidates"][0]["record"][field] = value
            self.assertEqual(display_place_title(item, build_place_labels(trip)),
                             "运城博物馆 (название на языке источника)")
        trip, item = fixture()
        trip["request"]["city_map_context"]["candidates"][0]["record"]["address"]["country_code"] = "ru"
        self.assertEqual(display_place_title(item, build_place_labels(trip)),
                         "运城博物馆 (название на языке источника)")

    def test_context_is_loaded_once_and_projection_has_no_side_effects(self):
        trip, item = fixture()
        from city_context import load_city_context
        before = deepcopy(trip)
        with patch("city_context.load_city_context", wraps=load_city_context) as load, \
             patch("places._load_cache", side_effect=AssertionError("No cache read")), \
             patch("places.geocode", side_effect=AssertionError("No lookup")), \
             patch("httpx.get", side_effect=AssertionError("No HTTP")):
            labels = build_place_labels(trip)
            for _ in range(4):
                display_place_title(item, labels)
        self.assertEqual(load.call_count, 1)
        self.assertEqual(trip, before)

    def test_russian_already_explained_and_non_attractions_are_unchanged(self):
        for title in ("Городской музей", "Louvre (Лувр)", "Музей LEGO", "123"):
            self.assertEqual(display_place_title({"title": title}), title)
        for kind in ("meal", "rest", "transfer", "transport"):
            self.assertEqual(display_place_title({"title": "Lunch", "kind": kind}), "Lunch")

    def test_display_dictionary_for_other_title_cannot_relabel_item(self):
        trip, item = fixture()
        self.assertEqual(display_place_title(item, {item.place_query: "Other place (музей)"}),
                         "运城博物馆 (название на языке источника)")

    def test_bad_or_missing_legacy_structure_never_starts_lookup(self):
        for trip in (None, {}, {"itinerary": {}}, {"itinerary": {"days": None}},
                     {"itinerary": {"days": [{"items": None}]}},
                     {"itinerary": {"days": [{"items": [{"title": []}]}]}}):
            with self.subTest(trip=trip):
                self.assertEqual(build_place_labels(trip), {})
        trip, item = fixture()
        trip["request"]["city_map_context"] = {"broken": True}
        self.assertEqual(display_place_title(item, build_place_labels(trip)),
                         "运城博物馆 (название на языке источника)")

    def test_model_itinerary_supported_without_mutating_it(self):
        trip, item = fixture()
        trip["itinerary"] = Itinerary.model_validate(trip["itinerary"])
        before = trip["itinerary"].model_dump()
        self.assertEqual(display_place_title(item, build_place_labels(trip)), "运城博物馆 (музей)")
        self.assertEqual(trip["itinerary"].model_dump(), before)


if __name__ == "__main__":
    unittest.main()
