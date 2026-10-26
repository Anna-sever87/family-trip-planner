"""Saved regional routes disclose scope gaps and keep one authoritative clock."""
import copy
import unittest
from unittest.mock import patch

from generation_quality import public_quality, safe_coverage
from schemas import Itinerary, ItineraryDay, ItineraryItem, TripRequest


class RegionalPublicDisclosureTests(unittest.TestCase):
    def test_stored_description_clock_is_removed_without_rewriting_input(self):
        raw = dict(time="09:15", kind="attraction", title="Музей", duration_min=90,
                   description="Музей, посещение с 09:15 до 12:00. Открыт с 09:00 до 18:00.")
        original = copy.deepcopy(raw)
        item = ItineraryItem.model_validate(raw)
        self.assertNotIn("посещение с", item.description)
        self.assertIn("Открыт с 09:00 до 18:00", item.description)
        self.assertEqual((item.time, item.duration_min), ("09:15", 90))
        self.assertEqual(raw, original)

    def test_unrelated_hours_and_other_item_types_are_preserved(self):
        for kind, description in [("attraction", "Касса с 09:00 до 17:00; дорога с 08:00 до 09:00."),
                                  ("attraction", "По правилам музея посещение с 10:00 до 17:00."),
                                  ("attraction", "Бесплатное посещение с 10:00 до 12:00."),
                                  ("attraction", "Посещение с 10:00 до 12:00 бесплатно, позднее нужен билет."),
                                  ("meal", "Посещение с 12:00 до 13:00.")]:
            item = ItineraryItem(time="12:00", title="Пункт", kind=kind, description=description)
            self.assertEqual(item.description, description)

    def test_visit_only_description_has_nonempty_explanation(self):
        item = ItineraryItem(time="09:15", title="Музей", description="Посещение с 09:15 до 12:00.")
        self.assertEqual(item.description, "Время посещения указано в расписании.")

    def test_saved_regional_trip_cannot_claim_requested_multiple_bases(self):
        request = TripRequest(destination="Карелия, Россия", destination_kind="region",
            start_date="2027-06-01", end_date="2027-06-02", budget=100000,
            wishes="Несколько баз проживания со сменой жилья и наземными переездами.")
        plan = Itinerary(title="Регион", summary="", days=[ItineraryDay(day=i,
            date=f"2027-06-0{i}", theme="Места", base_area="Петрозаводск, Карелия, Россия",
            items=[ItineraryItem(time="10:00", title=f"Музей {i}", description="Осмотр")]) for i in (1, 2)])
        before = plan.model_dump()
        quality = public_quality({"quality_status": "complete"}, plan, request)
        self.assertEqual(quality["quality_status"], "partial")
        self.assertIn("regional_scope_narrowed", quality["quality_reasons"])
        self.assertEqual(quality["selected_bases"], 1)
        self.assertNotIn("confirmed_bases", quality)
        self.assertEqual(plan.model_dump(), before)

        plan.days[1].base_area = "Сортавала, Карелия, Россия"
        revised = public_quality(quality, plan, request)
        self.assertEqual(revised["selected_bases"], 2)
        self.assertNotIn("regional_scope_narrowed", revised["quality_reasons"])
        other = public_quality({**quality, "quality_reasons": ["regional_scope_narrowed", "search_limit"]}, plan, request)
        self.assertEqual(other["quality_reasons"], ["search_limit"])
        self.assertEqual(other["quality_status"], "partial")
        unexplained = public_quality({"quality_status": "partial"}, plan, request)
        self.assertEqual(unexplained["quality_status"], "partial")

    def test_prompt_witness_respects_requested_base_count(self):
        from test_regional_map_only_schedule import fixture
        from grounded_schedule import build_grounded_messages
        req, scope, grounding, _, _ = fixture()
        third = copy.deepcopy(grounding.base_facts[0])
        third.update(base_key="third", base_query="Третья база, Россия")
        grounding.base_facts.append(third)
        for wishes, expected in [("Хочу три базы", [1, 3]), ("Без смены жилья, одна база", [1])]:
            with self.subTest(wishes=wishes), patch("regional_schedule_witness.build_witness", return_value=None) as build:
                build_grounded_messages(req.model_copy(update={"wishes": wishes}), scope, grounding, 4)
                self.assertEqual([call.kwargs.get("minimum_bases", 1) for call in build.call_args_list], expected)

    def test_review_diagnostics_allowlist_never_exposes_provider_text(self):
        self.assertEqual(safe_coverage({"selection_review_failure_reason": "invalid_json"}),
                         {"selection_review_failure_reason": "invalid_json"})
        for value in ("secret provider message", {}, []):
            self.assertEqual(safe_coverage({"selection_review_failure_reason": value}), {})


if __name__ == "__main__":
    unittest.main()
