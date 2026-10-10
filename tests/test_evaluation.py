import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from evaluation import (
    DEFAULT_INPUT,
    proxy_relevance_grades,
    ranking_metrics,
    random_category_baseline,
    run_evaluation,
)
from fashion_preprocessing import validate_and_normalize


class RecommendationEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = validate_and_normalize(pd.read_csv(DEFAULT_INPUT))

    def test_ranking_metrics_match_hand_calculated_example(self):
        metrics = ranking_metrics(
            ["irrelevant", "relevant-a", "relevant-b"],
            {"irrelevant": 0, "relevant-a": 3, "relevant-b": 2},
            k=2,
            relevance_threshold=2,
        )
        expected_ndcg = (7 / 1.584962500721156) / (
            7 + 3 / 1.584962500721156
        )
        self.assertAlmostEqual(metrics["precision@2"], 0.5)
        self.assertAlmostEqual(metrics["recall@2"], 0.5)
        self.assertAlmostEqual(metrics["f1@2"], 0.5)
        self.assertAlmostEqual(metrics["ndcg@2"], expected_ndcg)
        self.assertEqual(metrics["hit_rate@2"], 1.0)

    def test_proxy_relevance_is_attribute_based_and_gender_compatible(self):
        profile = self.data.iloc[0].copy()
        profile["gender"] = "Women"
        candidates = self.data.iloc[[0, 1]].copy()
        candidates.loc[candidates.index[0], "gender"] = "Unisex"
        grades = proxy_relevance_grades(profile, candidates)

        self.assertGreaterEqual(grades[str(candidates.iloc[0]["product_id"])], 4)
        self.assertIsInstance(grades[str(candidates.iloc[1]["product_id"])], int)

    def test_random_category_baseline_is_repeatable(self):
        profile = self.data.iloc[0]
        first = random_category_baseline(profile, self.data, 10, 42)
        second = random_category_baseline(profile, self.data, 10, 42)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 10)

    def test_evaluation_writes_labeled_comparison_and_plot(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            paths = run_evaluation(
                output_dir=output,
                max_profiles=24,
            )
            comparison = pd.read_csv(paths["comparison"])
            report = json.loads(paths["report_json"].read_text(encoding="utf-8"))

            self.assertEqual(len(comparison), 4)
            self.assertEqual(set(comparison["method"]), {
                "Random/category baseline",
                "Content-based ranking",
            })
            self.assertEqual(set(comparison["k"]), {5, 10})
            self.assertTrue(
                comparison[
                    [
                        "precision_at_k",
                        "recall_at_k",
                        "f1_at_k",
                        "ndcg_at_k",
                        "hit_rate_at_k",
                    ]
                ].apply(lambda column: column.between(0, 1).all()).all()
            )
            self.assertEqual(report["evaluation_type"], "simulated_catalog_proxy")
            self.assertFalse(report["real_user_interaction_data_available"])
            self.assertFalse(report["relevance_data_fabricated"])
            self.assertEqual(report["train_test_product_id_overlap"], 0)
            self.assertTrue(paths["plot"].is_file())
            self.assertTrue(paths["report_markdown"].is_file())
            self.assertIn(
                "not a measurement of user satisfaction",
                paths["report_markdown"].read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
