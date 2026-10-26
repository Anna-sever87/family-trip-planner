"""Read-only regional transfer disclosure; synthetic data, no routing or AI."""

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
from formatting import format_itinerary
from grounded_schedule import display_regional_transfer_note
import pdf_export
from schemas import Itinerary


def regional_trip():
    """Display fixture with the observed note failure, not a travel recommendation."""
    bases = ["Кондопога, Карелия, Россия"] * 3 + ["Петрозаводск, Карелия, Россия"]
    notes = ["Петрозаводск — Кондопога", "Кондопога — Кондопога",
             "Кондопога — Петрозаводск", "Петрозаводск — Петрозаводск"]
    return {
        "request": {"destination": "Карелия, Россия", "destination_kind": "region",
                    "start_date": "2027-06-01", "end_date": "2027-06-04", "adults": 2,
                    "children_ages": [], "budget": 140000, "currency": "RUB",
                    "travel_cost": 25000, "lodging_cost": 45000},
        "itinerary": {"title": "Синтетический региональный пример", "summary": "Пример отображения.",
                      "days": [{"day": number, "date": f"2027-06-0{number}", "theme": "Программа",
                                "base_area": base, "transfer_note": notes[number - 1],
                                "items": [{"time": "13:20", "kind": "attraction", "title": f"Место {number}",
                                           "place_query": f"Место {number}, Карелия, Россия",
                                           "description": "Синтетическое посещение.", "duration_min": 90,
                                           "cost": 0.0}]} for number, base in enumerate(bases, 1)], "tips": []},
    }


class RegionalTransferDisplayTests(unittest.TestCase):
    def test_transition_uses_actual_adjacent_bases_not_generated_transfer_text(self):
        plan = Itinerary.model_validate(regional_trip()["itinerary"])
        original = plan.model_dump()
        notes = [display_regional_transfer_note(plan, i, "region") for i in range(4)]
        self.assertIn("Начало программы на базе: Кондопога", notes[0])
        self.assertNotIn("Петрозаводск", notes[0])  # No fabricated arrival origin.
        for note in notes[1:3]:
            self.assertIn("База сохраняется: Кондопога", note)
            self.assertIn("Смена базы проживания не запланирована", note)
            self.assertIn("Путь к местам и обратно", note)
            self.assertNotIn("Петрозаводск", note)  # Day 3 is not a change of overnight base.
        self.assertIn("Смена базы проживания: из Кондопога, Карелия, Россия в Петрозаводск", notes[3])
        self.assertIn("До первого посещения", notes[3])
        for phrase in ("Дорожная и сезонная доступность", "время в пути", "условия смены жилья", "не подтверждены"):
            self.assertIn(phrase, notes[3])
        for note in notes:
            self.assertNotRegex(note, r"\d|автобус|такси|паром|руб|RUB")
        self.assertEqual(plan.model_dump(), original)

    def test_source_and_city_transport_notes_are_preserved_exactly(self):
        for source, city, kind in [(True, False, "region"), (False, True, "region"),
                                   (False, False, "city"), (False, False, "")]:
            plan = Itinerary.model_validate(regional_trip()["itinerary"])
            plan.source_schedule, plan.city_schedule = source, city
            plan.days[3].transfer_note = "  Подтверждённый трансфер 90 минут; требуется запись.  "
            self.assertEqual(display_regional_transfer_note(plan, 3, kind), plan.days[3].transfer_note)

    def test_missing_base_does_not_invent_a_transition_or_sameness(self):
        plan = Itinerary.model_validate(regional_trip()["itinerary"])
        plan.days[1].base_area = ""
        self.assertIn("База дня не указана", display_regional_transfer_note(plan, 1, "region"))
        note = display_regional_transfer_note(plan, 2, "region")
        self.assertIn("Предыдущая база не указана", note)
        self.assertNotIn("База сохраняется", note)
        self.assertNotIn("Смена базы проживания: из", note)

    def test_whitespace_and_case_differences_do_not_invent_a_transfer(self):
        plan = Itinerary.model_validate(regional_trip()["itinerary"])
        plan.days[1].base_area = "  КОНДОПОГА,  Карелия, Россия  "
        self.assertIn("База сохраняется", display_regional_transfer_note(plan, 1, "region"))

    def test_telegram_uses_same_readonly_disclosure_with_no_timing_or_cost_changes(self):
        trip = regional_trip()
        plan = Itinerary.model_validate(trip["itinerary"])
        before = plan.model_dump()
        output = format_itinerary(plan, "RUB", 140000, travel_cost=25000, lodging_cost=45000,
                                  saved_request=trip["request"])
        for i in range(4):
            self.assertIn("Перемещения: " + display_regional_transfer_note(plan, i, "region"), output)
        for raw in ("Кондопога — Кондопога", "Петрозаводск — Петрозаводск"):
            self.assertNotIn(raw, output)
        self.assertEqual(output.count("13:20"), 4)
        self.assertIn("Итого запланировано: около 70000 RUB", output)
        self.assertEqual(plan.model_dump(), before)

    def test_pdf_build_renders_same_disclosure_without_mutating_source(self):
        trip = regional_trip()
        before = deepcopy(trip)
        plan = Itinerary.model_validate(trip["itinerary"])
        with patch.object(pdf_export, "_p", wraps=pdf_export._p) as paragraph:
            pdf = pdf_export.build_trip_pdf(trip, [])
        self.assertTrue(pdf.startswith(b"%PDF"))
        text = "\n".join(str(call.args[0]) for call in paragraph.call_args_list)
        for i in range(4):
            self.assertIn("Перемещения: " + display_regional_transfer_note(plan, i, "region"), text)
        self.assertNotIn("Петрозаводск — Петрозаводск", text)
        self.assertIn("из Кондопога, Карелия, Россия в Петрозаводск", text)
        self.assertEqual(trip, before)


class RegionalTransferApiTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="regional-transfer-display-"))
        self.stack.enter_context(patch.object(db, "DB_PATH", str(Path(directory) / "synthetic.db")))
        self.stack.enter_context(patch.dict("os.environ", {
            "WEB_SESSION_SECRET": "synthetic-offline-transfer-display-secret-not-real-12345", "ACCESS_MODE": "open"}))
        db.init_db()
        db.save_user(101, "Synthetic", 2, [], "RUB")
        db.save_user(202, "Other", 2, [], "RUB")
        for owner, name in ((socket, "getaddrinfo"), (places.httpx.HTTPTransport, "handle_request"),
                            (places, "geocode"), (places, "_save_cache"),
                            (web_service, "generate_itinerary"), (db, "claim_ai_request")):
            self.stack.enter_context(patch.object(owner, name, side_effect=AssertionError(f"Unexpected I/O: {name}")))
        self.client = TestClient(web_app.app)  # No lifespan or background production jobs.
        self.addCleanup(self.client.close)
        session, _, _ = web_auth.create_session(101)
        self.client.cookies.set(web_auth.SESSION_COOKIE, session)

    @staticmethod
    def snapshot():
        with db.connect() as conn:
            return {table: [dict(row) for row in conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
                    for table in ("trips", "ai_requests", "access_route_attempts", "generation_jobs", "memory_items")}

    def test_api_corrects_existing_trip_only_in_response_and_preserves_owner_boundary(self):
        trip = regional_trip()
        before_input = deepcopy(trip)
        trip_id = db.save_trip(101, trip["request"], trip["itinerary"])
        before = self.snapshot()
        response = self.client.get(f"/api/trips/{trip_id}")
        self.assertEqual(response.status_code, 200, response.text)
        days = response.json()["itinerary"]["days"]
        plan = Itinerary.model_validate(trip["itinerary"])
        for i, day in enumerate(days):
            self.assertEqual(day["transfer_note"], display_regional_transfer_note(plan, i, "region"))
            for key in ("day", "date", "base_area"):
                self.assertEqual(day[key], trip["itinerary"]["days"][i][key])
            for key in ("time", "duration_min", "cost", "place_query", "description"):
                self.assertEqual(day["items"][0][key], trip["itinerary"]["days"][i]["items"][0][key])
        self.assertEqual(response.json()["budget"]["planned_total"], 70000)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(trip, before_input)
        session, _, _ = web_auth.create_session(202)
        self.client.cookies.set(web_auth.SESSION_COOKIE, session)
        self.assertEqual(self.client.get(f"/api/trips/{trip_id}").status_code, 404)

    def test_api_source_backed_note_and_pdf_remain_readonly(self):
        trip = regional_trip()
        trip["itinerary"]["source_schedule"] = True
        trip["itinerary"]["days"][3]["transfer_note"] = "Подтверждённый трансфер 90 минут; требуется запись."
        trip_id = db.save_trip(101, trip["request"], trip["itinerary"])
        before = self.snapshot()
        response = self.client.get(f"/api/trips/{trip_id}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["itinerary"]["days"][3]["transfer_note"],
                         trip["itinerary"]["days"][3]["transfer_note"])
        with patch.object(pdf_export, "_p", wraps=pdf_export._p) as paragraph:
            content = web_service.get_trip_pdf(101, trip_id)
        self.assertTrue(content.startswith(b"%PDF"))
        self.assertIn(trip["itinerary"]["days"][3]["transfer_note"],
                      "\n".join(str(call.args[0]) for call in paragraph.call_args_list))
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
