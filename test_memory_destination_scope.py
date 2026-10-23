"""Synthetic memory scope checks; no private records, real DB, HTTP or AI."""

import socket
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import memory
import web_service
import test_bot_generation_outcome as bot_fixture
import test_web_service_outcomes as web_fixture


REAL_RECALL = memory.recall_family_memories
REAL_FORMAT = memory.format_memory_context


def row(key, *, kind="destination_wishes", destination="Сочи, Россия", score=.9):
    return {"id": f"family-pref#{key}", "score": score, "metadata": {
        "text": f"SYNTHETIC_{key}", "kind": kind, "destination": destination,
    }}


def candidates(destination="Сочи, Россия"):
    return [row("MATCH", destination=destination),
            row("OTHER_WISH", destination="Другой город, Россия"),
            row("OTHER_FEEDBACK", kind="feedback", destination="Другой город, Россия"),
            row("GLOBAL", kind="custom", destination="")]


class MemoryDestinationScopeTests(unittest.TestCase):
    def setUp(self):
        for owner, name in ((socket.socket, "connect"), (socket, "getaddrinfo"), (sqlite3, "connect")):
            blocker = patch.object(owner, name, side_effect=AssertionError("External I/O forbidden"))
            mock = blocker.start()
            self.addCleanup(blocker.stop)
            self.addCleanup(mock.assert_not_called)
        self.index = Mock()
        self.index.query.return_value = {"matches": candidates()}
        for target, value in (("_pinecone_index", self.index), ("_embedding", [0.0])):
            blocker = patch.object(memory, target, return_value=value)
            mock = blocker.start()
            self.addCleanup(blocker.stop)
            if target == "_embedding":
                self.embedding = mock

    def recall(self, **options):
        return REAL_RECALL(100, 101, "Synthetic query", top_k=8, min_score=.35, **options)

    def assert_context_scoped(self, text):
        self.assertIn("SYNTHETIC_MATCH", text)
        self.assertIn("SYNTHETIC_GLOBAL", text)
        self.assertNotIn("SYNTHETIC_OTHER_WISH", text)
        self.assertNotIn("SYNTHETIC_OTHER_FEEDBACK", text)

    def test_unrelated_destination_wishes_and_feedback_never_reach_context(self):
        matches = self.recall(destination="Сочи, Россия")
        self.assertEqual([match.memory_id for match in matches], ["family-pref#MATCH", "family-pref#GLOBAL"])
        self.assert_context_scoped(REAL_FORMAT(matches))
        self.embedding.assert_called_once_with("Synthetic query", purpose="query")
        self.index.query.assert_called_once()
        kwargs = self.index.query.call_args.kwargs
        self.assertEqual(kwargs["namespace"], memory.memory_namespace(100, 101))
        self.assertEqual((kwargs["top_k"], kwargs["timeout"], kwargs["include_values"]), (8, 30, False))

    def test_exact_normalization_allows_case_and_spacing_but_not_alias_or_country_changes(self):
        self.index.query.return_value = {"matches": [row("MATCH", kind="feedback")]}
        for destination in ("  СОЧИ ,  Россия ", "Сочи,\tРоссия", "сочи,россия"):
            with self.subTest(destination=destination):
                self.assertEqual(len(self.recall(destination=destination)), 1)
        for destination in ("Сочи", "Сочи, Турция", "Sochi, Russia", "Большой Сочи, Россия",
                            "Сочи, Краснодарский край, Россия", "Сочи,,Россия", "", None):
            with self.subTest(destination=destination):
                self.assertEqual(self.recall(destination=destination), [])

    def test_global_preferences_remain_available_across_destinations(self):
        self.index.query.return_value = {"matches": [
            row(kind, kind=kind, destination="") for kind in ("family", "pace", "currency", "custom")
        ]}
        for destination in ("Сочи, Россия", "Нячанг, Вьетнам", ""):
            with self.subTest(destination=destination):
                self.assertEqual([match.kind for match in self.recall(destination=destination)],
                                 ["family", "pace", "currency", "custom"])

    def test_old_call_without_destination_is_compatible_but_returns_global_records_only(self):
        matches = self.recall()
        self.assertEqual([match.kind for match in matches], ["custom"])
        self.assertNotIn("SYNTHETIC_MATCH", REAL_FORMAT(matches))

    def test_scoped_or_unknown_kind_without_destination_is_not_promoted_to_global(self):
        for kind in ("destination_wishes", "feedback", "unknown", "", None):
            for destination in ("", " ", None, 123, ",", "Сочи,,Россия"):
                with self.subTest(kind=kind, destination=destination):
                    self.index.query.return_value = {"matches": [row("BAD", kind=kind, destination=destination)]}
                    self.assertEqual(self.recall(destination="Сочи, Россия"), [])

    def test_even_global_kind_with_destination_cannot_cross_to_another_city(self):
        self.index.query.return_value = {"matches": [row("LOCAL_CUSTOM", kind="custom", destination="Другой город, Россия")]}
        self.assertEqual(self.recall(destination="Сочи, Россия"), [])
        self.assertEqual(len(self.recall(destination="Другой город, Россия")), 1)

    def test_score_id_and_empty_text_guards_remain_effective(self):
        low = row("LOW", score=.2)
        invalid_id = row("ID")
        invalid_id["id"] = "not-owned-format"
        empty_text = row("EMPTY")
        empty_text["metadata"]["text"] = " "
        self.index.query.return_value = {"matches": [low, invalid_id, empty_text, row("MATCH")]}
        self.assertEqual([match.memory_id for match in self.recall(destination="Сочи, Россия")], ["family-pref#MATCH"])

    def test_web_creation_passes_current_destination_and_only_scoped_context(self):
        for kind in ("city", "region"):
            fixture = web_fixture.WebServiceOutcomeTests()
            fixture.setUp()
            try:
                fixture.request.destination_kind = kind
                destination = fixture.request.destination
                self.index.query.return_value = {"matches": candidates(destination)}
                with self.subTest(kind=kind), patch.object(web_service.vector_memory,
                        "recall_family_memories", side_effect=REAL_RECALL) as recall, \
                        patch.object(web_service.vector_memory, "format_memory_context", side_effect=REAL_FORMAT):
                    web_service.create_trip(123, fixture.request)
                self.assertEqual(recall.call_args.kwargs,
                                 {"top_k": 8, "min_score": .35, "destination": destination})
                self.assert_context_scoped(fixture.generate.call_args.kwargs["memory_context"])
                fixture.save.assert_called_once()
                fixture.assert_finished(True)
            finally:
                fixture.doCleanups()

    def test_bot_creation_passes_current_destination_and_only_scoped_context(self):
        for kind in ("city", "region"):
            with self.subTest(kind=kind):
                fixture = bot_fixture.BotGenerationOutcomeTests()
                fixture.setUp()
                fixture.request.destination_kind = kind
                destination = fixture.request.destination
                self.index.query.return_value = {"matches": candidates(destination)}
                recall = Mock(side_effect=REAL_RECALL)
                fixture.namespace["vector_memory"] = SimpleNamespace(
                    recall_family_memories=recall, format_memory_context=REAL_FORMAT)
                fixture.run_job()
                self.assertEqual(recall.call_args.kwargs,
                                 {"top_k": 8, "min_score": .35, "destination": destination})
                self.assert_context_scoped(fixture.namespace["generate_itinerary"].call_args.kwargs["memory_context"])
                fixture.namespace["db"].save_trip.assert_called_once()
                fixture.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", True)

    def test_web_edit_retains_existing_no_memory_retrieval_path(self):
        fixture = web_fixture.WebServiceOutcomeTests()
        fixture.setUp()
        try:
            fixture.edit_day()
            web_service.vector_memory.recall_family_memories.assert_not_called()
            web_service.vector_memory.format_memory_context.assert_not_called()
            fixture.edit.assert_called_once_with(fixture.request, fixture.current, 1, "Добавь парк")
            fixture.update.assert_called_once()
            fixture.assert_finished(True)
        finally:
            fixture.doCleanups()

    def test_bot_edit_retains_existing_no_memory_retrieval_path(self):
        fixture = bot_fixture.BotGenerationOutcomeTests()
        fixture.setUp()
        fixture.run_edit()
        fixture.namespace["vector_memory"].recall_family_memories.assert_not_called()
        fixture.namespace["vector_memory"].format_memory_context.assert_not_called()
        self.assertEqual(fixture.namespace["edit_itinerary"].call_args.kwargs, {})
        fixture.namespace["db"].update_trip_plan.assert_called_once()
        fixture.namespace["_finish_ai_claim"].assert_called_once_with("test-claim", True)


if __name__ == "__main__":
    unittest.main()
