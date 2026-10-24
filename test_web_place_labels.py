"""Owner-authorized, read-only web projection of saved foreign place names."""

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

with patch("dotenv.load_dotenv", return_value=False):
    import db
    import places
    import web_app
    import web_auth
    import web_service
from fastapi.testclient import TestClient
from city_context import CONTEXT_KEY
from test_place_label_rendering import foreign_trip


class WebPlaceLabelTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="web-place-labels-"))
        self.stack.enter_context(patch.object(db, "DB_PATH", str(Path(directory) / "synthetic.db")))
        self.stack.enter_context(patch.dict("os.environ", {
            "WEB_SESSION_SECRET": "synthetic-offline-web-labels-secret-not-a-real-key-12345", "ACCESS_MODE": "open"}))
        db.init_db()
        db.save_user(101, "Synthetic", 2, [], "RUB")
        db.save_user(202, "Other", 2, [], "RUB")
        # asyncio uses a local socketpair on Windows; block DNS/real HTTP rather
        # than that internal wakeup pipe. The suite guard additionally blocks
        # all non-socketpair socket operations.
        for owner, name in ((socket, "getaddrinfo"), (places.httpx.HTTPTransport, "handle_request"),
                            (places.httpx, "get"), (places, "geocode"), (places, "_save_cache"),
                            (web_service, "generate_itinerary"), (db, "claim_ai_request"),
                            (web_service.vector_memory, "recall_family_memories")):
            self.stack.enter_context(patch.object(owner, name, side_effect=AssertionError(f"Unexpected I/O: {name}")))
        # No lifespan: no cover executor, cleanup worker or production paths.
        self.client = TestClient(web_app.app)
        self.addCleanup(self.client.close)
        self.login(101)

    def login(self, owner):
        session, _, _ = web_auth.create_session(owner)
        self.client.cookies.set(web_auth.SESSION_COOKIE, session)

    def save(self, trip):
        return db.save_trip(101, trip["request"], trip["itinerary"])

    @staticmethod
    def snapshot():
        with db.connect() as conn:
            return {table: [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
                    for table in ("trips", "ai_requests", "access_route_attempts", "generation_jobs", "memory_items")}

    def test_authenticated_trip_detail_adds_display_only_title_to_timeline_and_map(self):
        trip = foreign_trip()
        original = deepcopy(trip)
        trip_id = self.save(trip)
        before = self.snapshot()
        response = self.client.get(f"/api/trips/{trip_id}")
        self.assertEqual(response.status_code, 200, response.text)
        detail = response.json()
        item = detail["itinerary"]["days"][0]["items"][0]
        point = detail["map_points"][0]
        expected = "合成博物馆 (Синтетический музей)"
        self.assertEqual((item["display_title"], point["display_title"]), (expected, expected))
        self.assertEqual((item["title"], point["title"]), ("合成博物馆", "合成博物馆"))
        saved_item = original["itinerary"]["days"][0]["items"][0]
        for key in ("title", "place_query", "official_name", "lat", "lon", "time", "duration_min", "cost"):
            self.assertEqual(item[key], saved_item[key], key)
        self.assertEqual((point["lat"], point["lon"]), (saved_item["lat"], saved_item["lon"]))
        self.assertEqual(point["official_name"], saved_item["official_name"])
        self.assertEqual(item["provider"], "osm")
        self.assertEqual(item["source_url"], point["source_url"])
        self.assertTrue(item["source_url"].startswith("https://www.openstreetmap.org/"))
        self.assertTrue(item["verified"])
        self.assertEqual(detail["itinerary"]["days"][0]["items"][1]["display_title"], "Lunch")
        self.assertNotIn(CONTEXT_KEY, detail["request"])
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(trip, original)
        self.assertNotIn("display_title", before["trips"][0]["itinerary_json"])

    def test_missing_invalid_or_mismatched_saved_context_never_invents_a_translation(self):
        stale = foreign_trip()["request"][CONTEXT_KEY]
        stale["request_sha256"] = "0" * 64
        for context in (None, {}, {"version": 1}, stale):
            with self.subTest(context=context):
                trip = foreign_trip()
                if context is None:
                    trip["request"].pop(CONTEXT_KEY)
                else:
                    trip["request"][CONTEXT_KEY] = context
                trip_id = self.save(trip)
                before = self.snapshot()
                response = self.client.get(f"/api/trips/{trip_id}")
                self.assertEqual(response.status_code, 200, response.text)
                detail = response.json()
                self.assertEqual(detail["itinerary"]["days"][0]["items"][0]["display_title"],
                                 "合成博物馆 (название на языке источника)")
                self.assertEqual(detail["map_points"][0]["display_title"],
                                 "合成博物馆 (название на языке источника)")
                self.assertNotIn("Синтетический музей", response.text)
                self.assertEqual(self.snapshot(), before)

    def test_category_explanation_is_not_a_translation_and_never_promotes_unverified_map_item(self):
        trip = foreign_trip(russian="")
        trip_id = self.save(trip)
        before = self.snapshot()
        response = self.client.get(f"/api/trips/{trip_id}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["itinerary"]["days"][0]["items"][0]["display_title"], "合成博物馆 (музей)")
        self.assertEqual(self.snapshot(), before)
        trip["itinerary"]["days"][0]["items"][0]["verified"] = False
        trip_id = self.save(trip)
        before = self.snapshot()
        response = self.client.get(f"/api/trips/{trip_id}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(response.json()["itinerary"]["days"][0]["items"][0]["verified"])
        self.assertEqual(response.json()["map_points"], [])
        self.assertEqual(self.snapshot(), before)

    def test_owner_isolation_happens_before_label_projection_and_anonymous_read_is_denied(self):
        trip_id = self.save(foreign_trip())
        before = self.snapshot()
        self.login(202)
        with patch.object(web_service, "build_place_labels", side_effect=AssertionError("No foreign trip projection")):
            response = self.client.get(f"/api/trips/{trip_id}")
            self.assertEqual(response.status_code, 404)
            self.client.cookies.clear()
            self.assertEqual(self.client.get(f"/api/trips/{trip_id}").status_code, 401)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
