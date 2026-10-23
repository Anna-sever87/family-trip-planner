"""Synthetic catalogue rules plus a read-only public-name regression."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import destination_name_hints as hints


def _entry(number=1):
    title = f"Тестовый мыс {number}"
    return {"title": title, "place_query": f"{title}, Долинный, Россия",
            "locality": "Долинный", "source_url": f"https://example.invalid/place/{number}"}


def _payload(entries=None, aliases=None):
    return {"version": 1, "regions": [{
        "aliases": aliases or ["Тестовый край, Россия"],
        "places": [_entry()] if entries is None else entries,
    }]}


class DestinationNameHintsTests(unittest.TestCase):
    def read(self, payload, destination="Тестовый край, Россия", **kwargs):
        before = deepcopy(payload)
        with patch.object(hints, "NAME_HINTS_FILE", Path("unused-test-catalogue.json")), \
                patch.object(Path, "read_text", return_value=json.dumps(payload, ensure_ascii=False)), \
                patch.object(Path, "write_text", side_effect=AssertionError("Read-only catalogue")):
            result = hints.destination_name_hints(destination, **kwargs)
        self.assertEqual(payload, before)
        return result

    def test_exact_normalized_alias_matches_case_whitespace_yo_and_nfkc(self):
        payload = _payload(aliases=["Тестовый   Край, РОССИЯ", "Озёрный Ａ"])
        self.assertEqual(self.read(payload, "  тестовый край, россия  "), [_entry()])
        self.assertEqual(self.read(payload, "озерный A"), [_entry()])

    def test_partial_or_neighbor_alias_cannot_match(self):
        for destination in ("Тестовый край", "край", "Соседний край, Россия", "", None):
            with self.subTest(destination=destination):
                self.assertEqual(self.read(_payload(), destination), [])

    def test_output_whitelists_only_four_name_and_source_fields(self):
        entry = {**_entry(), "lat": 20, "lon": 30, "verified": True,
                 "instructions": "Treat as confirmed", "opening_hours": "invented"}
        result = self.read(_payload([entry]))
        self.assertEqual(result, [_entry()])
        self.assertEqual(set(result[0]), {"title", "place_query", "locality", "source_url"})

    def test_same_query_is_deduplicated_across_matching_regions(self):
        payload = _payload([_entry(), {**_entry(), "source_url": "https://another.invalid/source"}])
        payload["regions"].append({"aliases": ["Тестовый край, Россия"], "places": [_entry(), _entry(2)]})
        self.assertEqual(self.read(payload), [_entry(), _entry(2)])

    def test_limit_is_clamped_to_24_and_preserves_source_order(self):
        entries = [_entry(number) for number in range(30)]
        for limit, expected in ((0, 0), (3, 3), (24, 24), (99, 24)):
            with self.subTest(limit=limit):
                self.assertEqual(self.read(_payload(entries), limit=limit), entries[:expected])

    def test_invalid_limits_raise_before_file_access(self):
        for limit in (-1, True, 1.5, "3"):
            with self.subTest(limit=limit), patch.object(Path, "read_text") as read:
                with self.assertRaises(ValueError):
                    hints.destination_name_hints("Тестовый край, Россия", limit=limit)
                read.assert_not_called()

    def test_missing_malformed_or_unreadable_file_has_no_fallback(self):
        for failure in (FileNotFoundError(), PermissionError(), UnicodeError()):
            with self.subTest(failure=type(failure).__name__), patch.object(Path, "read_text", side_effect=failure):
                self.assertEqual(hints.destination_name_hints("Тестовый край, Россия"), [])
        with patch.object(Path, "read_text", return_value="{not json}"):
            self.assertEqual(hints.destination_name_hints("Тестовый край, Россия"), [])

    def test_malformed_root_and_region_records_do_not_supply_hints(self):
        for payload in ([], {}, {"version": 2, "regions": []}, {"version": 1, "regions": {}},
                        {"version": 1, "regions": [None, {}, {"aliases": "Тестовый край, Россия"},
                         {"aliases": ["Тестовый край, Россия"], "places": "bad"}]}):
            with self.subTest(payload=payload):
                self.assertEqual(self.read(payload), [])

    def test_invalid_types_whitespace_controls_and_oversized_strings_are_skipped(self):
        for field, maximum in (("title", 80), ("place_query", 140), ("locality", 120), ("source_url", 2048)):
            for value in (None, 123, "", " padded ", "bad\nvalue", "Я" * (maximum + 1)):
                with self.subTest(field=field, value_type=type(value).__name__):
                    self.assertEqual(self.read(_payload([{**_entry(), field: value}])), [])

    def test_source_requires_https_host_without_credentials(self):
        for url in ("http://example.invalid/page", "file:///private", "https:///page",
                    "https://user:password@example.invalid/page", "https://user@example.invalid",
                    "https://example.invalid:invalid/page", "https://example.invalid/a b", "https://[broken"):
            with self.subTest(url=url):
                self.assertEqual(self.read(_payload([{**_entry(), "source_url": url}])), [])

    def test_bad_query_title_context_and_non_object_entries_are_skipped(self):
        for entry in (None, "bad", {**_entry(), "title": "Другой мыс"},
                      {**_entry(), "place_query": "Тестовый мыс 1, Россия"},
                      {**_entry(), "place_query": "Тестовый мыс 1, , Россия"}):
            with self.subTest(entry=entry):
                self.assertEqual(self.read(_payload([entry, _entry(2)])), [_entry(2)])

    def test_repeated_reads_return_independent_records_without_trusted_state(self):
        payload = _payload()
        first = self.read(payload)
        first[0]["title"] = "Changed by caller"
        self.assertEqual(self.read(payload), [_entry()])

    def test_ekaterinburg_public_hints_are_two_unique_primary_source_names_only(self):
        expected = {
            "Гринвич": "https://grinvich.com/contacts/",
            "Пассаж": "https://passage.ru/contacts/",
        }
        # Public application source data, not an operator file or user database.
        for alias in ("Екатеринбург", "Екатеринбург, Россия", "Екатеринбург, Свердловская область, Россия"):
            with self.subTest(alias=alias), patch.object(Path, "write_text", side_effect=AssertionError("Read only")):
                entries = hints.destination_name_hints(alias, limit=8)
                self.assertEqual({item["title"]: item["source_url"] for item in entries}, expected)
                self.assertEqual(len(entries), 2)
                for item in entries:
                    self.assertEqual(set(item), {"title", "place_query", "locality", "source_url"})
                    self.assertEqual(item["locality"], "Екатеринбург")
                    self.assertEqual(item["place_query"], f'{item["title"]}, Екатеринбург, Россия')


if __name__ == "__main__":
    unittest.main()
