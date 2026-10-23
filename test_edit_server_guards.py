"""Server-side edit guards; only external AI/map replies are synthetic."""

import json
import socket
import sqlite3
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from rules import check_rules
from schemas import ItineraryItem
from test_regional_edit_safety import _source
import test_regional_edit_safety as edit_fixture
import test_regional_generation_repair as fixture


class EditServerGuardTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "create_connection"),
                            (socket, "getaddrinfo"), (sqlite3, "connect")):
            blocker = patch.object(owner, name, side_effect=AssertionError("real I/O forbidden"))
            blocker.start()
            self.addCleanup(blocker.stop)

    def changed(self, days=1):
        source = _source(days)
        changed = source.days[0].model_copy(deep=True)
        changed.theme = "Обновлённая программа дня"
        return source, changed

    def run_edit(self, source, changed, **kwargs):
        return edit_fixture.RegionalEditSafetyTests()._run(source, changed, 1, **kwargs)

    def test_negative_ticket_cannot_cancel_an_over_budget_meal_in_real_edit(self):
        source, changed = self.changed()
        fixture._attraction(changed).cost = -100000
        next(item for item in changed.items if item.kind == "meal").cost = 100001
        # Ordinary JSON passes through the production parser and price/rule
        # validators; no budget validation is mocked by the edit fixture.
        with self.assertRaisesRegex(RuntimeError, "Не удалось изменить маршрут"):
            self.run_edit(source, changed)

    def test_model_rejects_invalid_prices_without_numeric_coercion(self):
        template = fixture._attraction(_source(1).days[0]).model_dump()
        for value in (-1, float("nan"), float("inf"), -float("inf"), None,
                      True, "free", "100", 1_000_000_001):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                ItineraryItem.model_validate({**template, "cost": value})
        for value in (0, 12, 12.5):
            self.assertEqual(ItineraryItem.model_validate({**template, "cost": value}).cost, value)

    def test_raw_nonfinite_and_nonnumeric_prices_fail_real_edit(self):
        source, changed = self.changed()
        for value in (float("nan"), float("inf"), "free", True):
            payload = changed.model_dump()
            payload["items"][0]["cost"] = value
            # json.dumps preserves NaN/Infinity tokens, avoiding Pydantic's
            # serialization to null so the inbound parser is exercised.
            raw = json.dumps(payload, ensure_ascii=False)
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "Не удалось изменить маршрут"):
                self.run_edit(source, changed, raw=raw)

    def test_rule_gate_rejects_unvalidated_model_copy_prices_without_crashing(self):
        request = fixture._request().model_copy(update={"end_date": _source(1).days[0].date})
        for value in (-1, float("nan"), float("inf"), -float("inf"), None,
                      True, "free", "100", 10 ** 400):
            source = _source(1)
            source.days[0].items[0] = source.days[0].items[0].model_copy(update={"cost": value})
            with self.subTest(value=value):
                problems = check_rules(source, request)
                self.assertTrue(any("стоимость пункта" in problem for problem in problems), problems)

    def test_untouched_legacy_day_with_invalid_price_blocks_real_edit(self):
        source, changed = self.changed(2)
        fixture._attraction(source.days[1]).cost = -100000
        result, lookup, audit = self.run_edit(source, changed)
        self.assertTrue(any("стоимость пункта" in problem for problem in result.problems), result.problems)
        lookup.assert_not_called()
        audit.assert_not_called()

    def test_edit_rejects_known_block_before_geocoding(self):
        source, changed = self.changed()
        blocked = [dict(aliases=[fixture._attraction(changed).title], reason="Объект закрыт")]
        result, lookup, audit = self.run_edit(source, changed, blocked_places=blocked)
        self.assertTrue(any("исключено из программы" in problem for problem in result.problems), result.problems)
        lookup.assert_not_called()
        audit.assert_not_called()

    def test_edit_rechecks_block_using_official_map_name(self):
        source, changed = self.changed()
        item = fixture._attraction(changed)
        official_name = "Официальное название закрытого музея"
        record = dict(name=official_name, lat=51.85, lon=104.91, type="museum", category="tourism")
        blocked = [dict(official_names=[official_name], reason="Объект закрыт")]
        result, lookup, audit = self.run_edit(source, changed, blocked_places=blocked,
                                             place_overrides={item.place_query: record})
        self.assertTrue(any("исключено из программы" in problem for problem in result.problems), result.problems)
        self.assertGreater(lookup.call_count, 0)
        audit.assert_not_called()

    def test_edit_clears_forged_and_legacy_map_fields_on_all_non_attractions(self):
        source, changed = self.changed(2)
        for day in (source.days[1], changed):
            for item in day.items:
                if item.kind != "attraction":
                    item.verified, item.lat, item.lon = True, 0.0, 0.0
                    item.official_name, item.provider, item.source_url = "Подмена", "2gis", "https://example.invalid"
        result, _, audit = self.run_edit(source, changed)
        self.assertEqual(result.problems, [])
        audit.assert_called_once()
        for day in result.itinerary.days:
            for item in day.items:
                if item.kind != "attraction":
                    self.assertEqual((item.verified, item.lat, item.lon, item.official_name,
                                      item.provider, item.source_url), (False, None, None, "", "", ""))
                else:
                    self.assertTrue(item.verified)
                    self.assertNotEqual((item.lat, item.lon), (0.0, 0.0))

    def test_unblocked_edit_with_valid_prices_still_succeeds(self):
        source, changed = self.changed(2)
        fixture._attraction(changed).cost = 123.5
        result, lookup, audit = self.run_edit(source, changed)
        self.assertEqual(result.problems, [])
        self.assertEqual(fixture._attraction(result.itinerary.days[0]).cost, 123.5)
        self.assertGreater(lookup.call_count, 0)
        audit.assert_called_once()

    def test_model_cannot_label_existing_program_as_free_to_skip_visits(self):
        source, changed = self.changed()
        changed.day_mode = "free"
        changed.items = [item for item in changed.items if item.kind != "attraction"]
        result, lookup, audit = self.run_edit(source, changed)
        self.assertEqual(result.itinerary.days[0].day_mode, "program")
        self.assertTrue(any("достопримечательности" in problem for problem in result.problems))
        lookup.assert_not_called()
        audit.assert_not_called()

    def test_model_free_label_is_discarded_on_otherwise_valid_program_edit(self):
        source, changed = self.changed()
        changed.day_mode = "free"
        result, _, audit = self.run_edit(source, changed)
        self.assertEqual(result.problems, [])
        self.assertEqual(result.itinerary.days[0].day_mode, "program")
        self.assertIsNone(result.itinerary.program_report)
        audit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
