"""Country input and same-place saved correction: isolated, no paid providers."""

from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

with patch("dotenv.load_dotenv", return_value=False):
    import db
    import trip_covers
    import web_app
    import web_auth
    import web_service

from fastapi.testclient import TestClient
from city_context import build_city_context, check_context_places, load_city_context
from city_candidates import CityCandidate
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest
import weather


def saved_fixture():
    request = TripRequest(destination="Тестгород", start_date="2099-06-01", end_date="2099-06-01",
                          budget=50000, currency="RUB", trip_goal="Познакомиться с городом",
                          interest_categories=["museums"])
    center = {"name": "Тестгород", "lat": 55.0, "lon": 83.0, "country_code": "RU", "country": "Россия"}
    title, query = "Городской музей", "Городской музей, Тестгород, Россия"
    record = {"name": query, "lat": 55.0, "lon": 83.0, "osm_type": "node", "osm_id": 101,
              "provider": "osm", "category": "tourism", "type": "museum",
              "address": {"city": "Тестгород", "country": "Россия", "country_code": "ru"}}
    candidate = CityCandidate("osm-node-101", title, query, "Тестгород", record)
    context = build_city_context(request, [candidate], center)
    item = ItineraryItem(time="10:00", title=title, place_query=query, area="Тестгород",
                         description="Осмотр", duration_min=90, cost=0, verified=True,
                         lat=55.0, lon=83.0, official_name=query, provider="osm")
    itinerary = Itinerary(title="Синтетический маршрут", summary="Не изменять", city_schedule=True,
                          days=[ItineraryDay(day=1, date=request.start_date, theme="Музей", items=[item])])
    return {"request": {**request.model_dump(), "city_map_context": context,
                        "weather_text": "Сохранённая погода", "generation_coverage": {"quality_status": "partial"}},
            "itinerary": itinerary.model_dump()}


class CountryInputTests(unittest.TestCase):
    def test_separate_and_legacy_complete_input_have_same_canonical_destination(self):
        for destination, country, expected in (
            ("Москва", "Россия", "Москва, Россия"),
            (" Москва ", " Россия ", "Москва, Россия"),
            ("Москва, Россия", None, "Москва, Россия"),
            ("Москва, Россия", "RU", "Москва, Россия"),
            ("Москва, Москва, Россия", "Russia", "Москва, Москва, Россия"),
            ("Сеул", "Южная Корея", "Сеул, Южная Корея"),
            ("Пекин", "Китай", "Пекин, Китай"),
            ("Байкал", "Россия", "Байкал, Россия"),
        ):
            with self.subTest(destination=destination, country=country):
                self.assertEqual(web_service.destination_with_country(destination, country), expected)

    def test_missing_conflicting_or_unsafe_country_is_not_silently_fixed(self):
        for destination, country in (
            ("Москва", None), ("Москва", ""), ("Москва", "  "),
            ("Paris, France", "Германия"), ("Пекин, Китай", "Россия"),
            ("Москва, Москва, Россия", "Китай"), ("Москва, Москва", "Россия"),
            ("Москва", "Россия, Китай"), ("Москва", "<Россия>"),
            ("Москва", "Россия\n"), ("Москва", "Россия\u202e"),
            ("Москва", "123"), ("Москва", "А" * 81), ("Москва,", "Россия"),
            ("Москва, район, регион, Россия", None),
        ):
            with self.subTest(destination=destination, country=country), self.assertRaises(ValueError):
                web_service.destination_with_country(destination, country)

    def test_public_payload_has_separate_country_but_internal_schema_is_unchanged(self):
        payload = {**saved_fixture()["request"], "destination": "Тестгород", "country": "Россия"}
        validated = web_app.TripCreateRequest.model_validate(payload)
        request = TripRequest.model_validate(validated.model_dump())
        self.assertEqual(request.destination, "Тестгород, Россия")
        self.assertNotIn("country", request.model_dump())
        self.assertNotIn("city_map_context", request.model_dump())
        with self.assertRaises(ValueError):
            web_app.TripCreateRequest.model_validate({**payload, "country": None})


class CountryCorrectionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="country-input-")))
        self.stack.enter_context(patch.object(db, "DB_PATH", str(self.directory / "synthetic.db")))
        self.stack.enter_context(patch.dict("os.environ", {
            "ACCESS_MODE": "open", "WEB_COOKIE_SECURE": "0", "WEB_BASE_URL": "http://testserver",
            "WEB_SESSION_SECRET": "synthetic-country-session-1234567890", "DAILY_LIMIT": "3",
        }))
        db.init_db()
        trip_covers.init_schema()
        for owner in (1, 2):
            db.save_user(owner, "Synthetic", 2, [], "RUB")
        self.account = db.get_account_key(1)
        self.fixture = saved_fixture()
        self.trip = db.save_trip(1, self.fixture["request"], self.fixture["itinerary"])
        self.other = db.save_trip(2, self.fixture["request"], self.fixture["itinerary"])
        db.save_checklist(self.trip, [("Synthetic", "Unchanged")])
        self.file = self.directory / "trip.pdf"
        self.file.write_bytes(b"synthetic-existing-pdf")
        self.generate = self.stack.enter_context(patch.object(web_service, "generate_itinerary", side_effect=AssertionError("No paid route")))
        self.recall = self.stack.enter_context(patch.object(web_service.vector_memory, "recall_family_memories", side_effect=AssertionError("No memory")))
        self.resolve = self.stack.enter_context(patch.object(web_service, "resolve_city_for_planning", side_effect=AssertionError("No geocoder")))
        self.claim = self.stack.enter_context(patch.object(db, "claim_ai_request", side_effect=AssertionError("No quota")))
        self.image = self.stack.enter_context(patch.object(web_service, "_enqueue_trip_cover_safely", side_effect=AssertionError("No image")))

    def snapshot(self):
        with db.connect() as conn:
            return {row[0]: [tuple(item) for item in conn.execute(f'SELECT * FROM "{row[0]}" ORDER BY 1')]
                    for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")}

    def correct(self, *, trip=None, revision=1, country="Россия", account=None):
        return web_service.correct_trip_country(1, self.trip if trip is None else trip, country, revision,
                                                expected_account_key=self.account if account is None else account)

    def raw_trip(self):
        with db.connect() as conn:
            return dict(conn.execute("SELECT * FROM trips WHERE id=?", (self.trip,)).fetchone())

    def client(self, owner=1):
        client = TestClient(web_app.app)
        self.addCleanup(client.close)
        if owner is not None:
            token, csrf, _ = web_auth.create_session(owner)
            client.cookies.set(web_auth.SESSION_COOKIE, token)
            client.cookies.set(web_auth.CSRF_COOKIE, csrf)
            client.headers["X-CSRF-Token"] = csrf
        return client

    def test_confirmed_country_changes_only_destination_binding_and_revision(self):
        with db.connect() as conn:
            conn.execute("INSERT INTO trip_covers(trip_id,account_key,cache_key,status,file_name,created_at,subject_label) "
                         "VALUES(?,?,?,'ready',?,?,?)", (self.trip, self.account, "old-key", "synthetic.png", db._now(), "Музей"))
            conn.execute("INSERT INTO cover_attempts(attempt_id,utc_day) VALUES('synthetic-attempt','2099-06-01')")
        before, raw = self.snapshot(), self.raw_trip()
        result = self.correct()
        self.assertEqual(result, {"id": self.trip, "destination": "Тестгород, Россия", "revision": 2, "changed": True})
        after = self.snapshot()
        self.assertEqual({key: value for key, value in before.items() if key != "trips"},
                         {key: value for key, value in after.items() if key != "trips"})
        current = self.raw_trip()
        self.assertEqual({key: value for key, value in raw.items() if key not in {"destination", "request_json", "revision"}},
                         {key: value for key, value in current.items() if key not in {"destination", "request_json", "revision"}})
        old_request, new_request = json.loads(raw["request_json"]), json.loads(current["request_json"])
        new_context = new_request["city_map_context"]
        context = old_request["city_map_context"]
        self.assertEqual({key: value for key, value in context.items() if key != "request_sha256"},
                         {key: value for key, value in new_context.items() if key != "request_sha256"})
        self.assertNotEqual(context["request_sha256"], new_context["request_sha256"])
        self.assertEqual({key: value for key, value in old_request.items() if key not in {"destination", "city_map_context"}},
                         {key: value for key, value in new_request.items() if key not in {"destination", "city_map_context"}})
        pool, _ = load_city_context(new_context, TripRequest.model_validate(new_request))
        check_context_places(Itinerary.model_validate(json.loads(current["itinerary_json"])), pool)
        self.assertEqual(db.get_trip(self.other, 2)["destination"], "Тестгород")
        self.assertEqual(self.file.read_bytes(), b"synthetic-existing-pdf")
        for method in (self.generate, self.recall, self.resolve, self.claim, self.image):
            method.assert_not_called()

    def test_wrong_country_cannot_relocate_saved_city(self):
        before = self.snapshot()
        with self.assertRaises(web_service.CountryCorrectionError):
            self.correct(country="Китай")
        self.assertEqual(self.snapshot(), before)

    def test_legacy_without_context_and_region_require_a_new_confirmed_process(self):
        for kind in ("missing", "region", "corrupt", "moved_item", "source", "wrong_snapshot", "already_complete"):
            with self.subTest(kind=kind):
                trip = deepcopy(self.fixture)
                if kind == "missing":
                    trip["request"].pop("city_map_context")
                elif kind == "region":
                    trip["request"]["destination_kind"] = "region"
                elif kind == "corrupt":
                    trip["request"]["city_map_context"] = {}
                elif kind == "moved_item":
                    trip["itinerary"]["days"][0]["items"][0]["lat"] += .01
                elif kind == "source":
                    trip["itinerary"]["source_schedule"] = True
                elif kind == "wrong_snapshot":
                    trip["request"]["city_map_context"]["resolved_city"]["country_code"] = "CN"
                else:
                    trip["request"]["destination"] = "Тестгород, Россия"
                number = db.save_trip(1, trip["request"], trip["itinerary"])
                before = self.snapshot()
                with self.assertRaises(web_service.CountryCorrectionError):
                    self.correct(trip=number)
                self.assertFalse(web_service._country_correction_available(db.get_trip(number, 1)))
                self.assertEqual(self.snapshot(), before)

    def test_ownership_account_replacement_and_revision_are_fail_closed(self):
        before = self.snapshot()
        for args, error in (({"trip": self.other}, web_service.TripNotFoundError),
                            ({"account": "old-account"}, web_service.TripNotFoundError),
                            ({"revision": 2}, web_service.EditConflictError)):
            with self.subTest(args=args), self.assertRaises(error):
                self.correct(**args)
        self.assertEqual(self.snapshot(), before)
        self.correct()
        updated = self.snapshot()
        with self.assertRaises(web_service.EditConflictError):
            self.correct()
        self.assertEqual(self.snapshot(), updated)

    def test_two_concurrent_corrections_have_one_commit_and_one_revision_conflict(self):
        def attempt():
            try:
                return self.correct()["revision"]
            except web_service.EditConflictError:
                return "conflict"
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: attempt(), range(2)))
        self.assertCountEqual(results, [2, "conflict"])
        self.assertEqual(self.raw_trip()["revision"], 2)
        self.claim.assert_not_called()
        self.generate.assert_not_called()

    def test_account_deletion_in_progress_cannot_modify_saved_trip(self):
        with db.connect() as conn:
            conn.execute("UPDATE users SET deletion_token='synthetic-deletion' WHERE tg_id=1")
        before = self.snapshot()
        with self.assertRaises(web_service.TripNotFoundError):
            self.correct()
        self.assertEqual(self.snapshot(), before)

    def test_active_route_queue_cover_and_reminder_each_block_correction(self):
        inserts = (
            ("ai_requests", "INSERT INTO ai_requests(claim_id,tg_id,usage_date,kind,source,status,claimed_at) VALUES('busy',1,'2099-06-01','plan','web','running','now')", ()),
            ("generation_jobs", "INSERT INTO generation_jobs(job_id,tg_id,account_key,idempotency_key,request_digest,status,created_at) VALUES(?,1,?,?,'digest','queued','now')", (str(uuid4()), self.account, str(uuid4()))),
            ("trip_covers", "INSERT INTO trip_covers(trip_id,account_key,cache_key,status,created_at) VALUES(?,?,'cache','running','now')", (self.trip, self.account)),
            ("reminder_deliveries", "INSERT INTO reminder_deliveries(trip_id,kind,status,claimed_at) VALUES(?,'day1','sending','now')", (self.trip,)),
        )
        for table, sql, params in inserts:
            with self.subTest(table=table):
                with db.connect() as conn:
                    conn.execute(sql, params)
                before = self.snapshot()
                with self.assertRaises(web_service.EditBusyError):
                    self.correct()
                self.assertEqual(self.snapshot(), before)
                with db.connect() as conn:
                    conn.execute(f'DELETE FROM "{table}"')

    def test_existing_missing_country_is_readable_and_exposes_only_capability(self):
        detail = web_service.get_trip_detail(1, self.trip)
        self.assertTrue(detail["country_correction_available"])
        self.assertEqual(detail["destination"], "Тестгород")
        self.assertNotIn("city_map_context", detail["request"])
        self.correct()
        self.assertFalse(web_service.get_trip_detail(1, self.trip)["country_correction_available"])

    def test_missing_country_is_blocked_before_geocoder_quota_memory_or_generation(self):
        request = TripRequest.model_validate(self.fixture["request"])
        before = self.snapshot()
        with self.assertRaisesRegex(web_service.TripCreateValidationError, "Укажите страну"):
            web_service.create_trip(1, request)
        self.assertEqual(self.snapshot(), before)
        for method in (self.generate, self.recall, self.resolve, self.claim):
            method.assert_not_called()

    def test_provider_wrong_country_still_fails_before_quota(self):
        request = TripRequest.model_validate({**self.fixture["request"], "destination": "Тестгород, Россия"})
        self.resolve.side_effect = weather.resolve_city_for_planning
        with patch.object(weather, "_get_json", return_value={"results": [{
            "id": 1234, "name": "Тестгород", "country": "Китай", "country_code": "CN",
            "latitude": 55.0, "longitude": 83.0, "feature_code": "PPL",
        }]}), self.assertRaises(web_service.TripCreateValidationError) as caught:
            web_service.create_trip(1, request)
        self.assertEqual(caught.exception.code, "city_not_found")
        self.claim.assert_not_called()
        self.recall.assert_not_called()
        self.generate.assert_not_called()

    def test_http_creation_requires_country_before_accepting_a_job(self):
        client = self.client()
        payload = {**self.fixture["request"], "destination": "Тестгород"}
        with patch.object(web_app.generation_jobs.manager, "submit", return_value={"status": "queued"}) as submit:
            for path in ("/api/trips", "/api/generation-jobs"):
                self.assertEqual(client.post(path, json=payload).status_code, 422)
            submit.assert_not_called()
            response = client.post("/api/generation-jobs", json={**payload, "country": "Россия"},
                                   headers={"Idempotency-Key": str(uuid4())})
            self.assertEqual(response.status_code, 202, response.text)
            self.assertEqual(submit.call_args.args[3].destination, "Тестгород, Россия")
        self.claim.assert_not_called()

    def test_http_correction_requires_owner_session_csrf_revision_and_safe_country(self):
        url = f"/api/trips/{self.trip}/country"
        payload = {"country": "Россия", "expected_revision": 1}
        self.assertEqual(self.client(None).patch(url, json=payload).status_code, 401)
        self.assertEqual(self.client(2).patch(url, json=payload).status_code, 404)
        client = self.client()
        csrf = client.headers.pop("X-CSRF-Token")
        self.assertEqual(client.patch(url, json=payload).status_code, 403)
        client.headers["X-CSRF-Token"] = csrf
        for changes in ({"country": "Китай"}, {"country": "Россия, Китай"}, {"expected_revision": True}, {"extra": "ignored?"}):
            self.assertEqual(client.patch(url, json={**payload, **changes}).status_code, 422)
        response = client.patch(url, json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["revision"], 2)
        self.assertEqual(client.patch(url, json=payload).status_code, 409)


if __name__ == "__main__":
    unittest.main()
