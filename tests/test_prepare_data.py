import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fashion_preprocessing import FashionFeatureEngineer, build_preprocessing_pipeline
from prepare_data import DEFAULT_INPUT, prepare_dataset


class FashionPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(DEFAULT_INPUT)

    def test_feature_engineering_uses_train_price_thresholds_and_valid_rules(self):
        train = self.data.iloc[:100].copy()
        engineer = FashionFeatureEngineer().fit(train)
        probe = train.iloc[:4].copy()
        probe["stock_quantity"] = [0, 1, 11, 31]
        result = engineer.transform(probe)

        self.assertIn("discount_amount", result)
        self.assertIn("discount_percentage", result)
        self.assertIn("price_segment", result)
        self.assertIn("rating_category", result)
        self.assertIn("stock_level", result)
        self.assertTrue({"purchase_month", "purchase_day", "purchase_year"}.issubset(result))
        self.assertEqual(
            result.loc[0, "discount_amount"],
            train.iloc[0]["original_price"] - train.iloc[0]["current_price"],
        )
        self.assertEqual(result.loc[0, "purchase_year"], 2025)
        self.assertEqual(
            result["stock_level"].tolist(),
            ["out_of_stock", "low", "medium", "high"],
        )

    def test_pipeline_ignores_unseen_categories_without_changing_width(self):
        train = self.data.iloc[:100].copy()
        pipeline = build_preprocessing_pipeline().fit(train)
        baseline = pipeline.transform(train.iloc[:2])
        unseen = train.iloc[:2].copy()
        unseen.loc[unseen.index[0], "brand"] = "Previously Unseen Brand"
        transformed = pipeline.transform(unseen)

        self.assertEqual(transformed.shape, baseline.shape)
        self.assertEqual(transformed.shape[1], len(pipeline.named_steps["encoding"].get_feature_names_out()))

    def test_full_preparation_splits_stratifies_and_writes_reusable_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            paths = prepare_dataset(DEFAULT_INPUT, output)
            train = pd.read_csv(paths["train_engineered"])
            test = pd.read_csv(paths["test_engineered"])
            x_train = pd.read_csv(paths["X_train"])
            x_test = pd.read_csv(paths["X_test"])
            y_train = pd.read_csv(paths["y_train"])
            y_test = pd.read_csv(paths["y_test"])
            report = pd.read_json(paths["report"], typ="series")

            self.assertEqual(len(train), 1740)
            self.assertEqual(len(test), 436)
            self.assertFalse(set(train["product_id"]) & set(test["product_id"]))
            self.assertEqual(
                y_train["category"].value_counts().add(
                    y_test["category"].value_counts(), fill_value=0
                ).astype(int).sort_index().to_dict(),
                self.data["category"].value_counts().sort_index().to_dict(),
            )
            self.assertEqual(len(y_test), 436)
            self.assertEqual(list(x_train.columns), list(x_test.columns))
            self.assertNotIn("customer_rating", x_train.columns)
            self.assertNotIn("category", x_train.columns)
            self.assertNotIn("product_id", x_train.columns)
            self.assertTrue(paths["pipeline"].is_file())
            self.assertEqual(report["split"]["train_test_product_id_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
