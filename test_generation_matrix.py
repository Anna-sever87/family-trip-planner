"""A failed data scenario is an expected outcome, not a successful itinerary."""
import unittest
from tools.check_generation_matrix import run_matrix


class GenerationMatrixTests(unittest.TestCase):
    def test_matrix_distinguishes_success_data_gaps_and_technical_failures(self):
        report = run_matrix()
        self.assertTrue(report["ok"], [row for row in report["results"] if not row["ok"]])
        self.assertEqual(report["cases"], 46)
        self.assertEqual(report["passed"], 46)
        self.assertEqual(report["ai_calls"], 0)
        self.assertEqual(report["network_calls"], 0)
        self.assertFalse(report["working_db_used"])
        self.assertEqual(report["outcomes"], {"success": 40, "insufficient_data": 4,
                                              "invalid_request": 1, "search_limit": 1})


if __name__ == "__main__":
    unittest.main()
