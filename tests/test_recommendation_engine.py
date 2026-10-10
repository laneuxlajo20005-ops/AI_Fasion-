import unittest
from io import BytesIO
from unittest.mock import patch

import pandas as pd

from context_processor import WeatherLookup
from fashion_preprocessing import validate_and_normalize
from market import app
from recommendation_engine import FashionRecommendationEngine


DATA_PATH = r"C:\Users\Administrator\Documents\market\market\fashion_boutique_fixed_location.csv"


class FashionRecommendationEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.products = validate_and_normalize(pd.read_csv(DATA_PATH))
        cls.engine = FashionRecommendationEngine(cls.products)

    def test_recommendations_are_ranked_real_products_and_repeatable(self):
        preferences = {
            "gender": "Women",
            "location": "Rural",
            "season": "Summer",
            "preferred_category": "Dresses",
            "preferred_color": "Pink",
            "preferred_brand": "Zara",
            "max_budget": "150",
            "minimum_rating": "3.5",
        }
        first = self.engine.recommend(preferences)
        second = self.engine.recommend(preferences)

        self.assertGreater(len(first), 0)
        self.assertLessEqual(len(first), 10)
        self.assertEqual(first, second)
        self.assertEqual(len({item["product_id"] for item in first}), len(first))
        self.assertEqual(
            [item["score"] for item in first],
            sorted((item["score"] for item in first), reverse=True),
        )
        self.assertIn("original_price", self.engine.feature_names)
        for item in first:
            self.assertLessEqual(item["current_price"], 150)
            self.assertGreaterEqual(item["customer_rating"], 3.5)
            self.assertGreater(item["stock_quantity"], 0)
            self.assertTrue(item["reasons"])
            self.assertAlmostEqual(
                sum(item["score_contributions"].values()), item["score"], places=1
            )
            self.assertAlmostEqual(
                sum(factor["contribution_points"] for factor in item["score_explanations"]),
                item["score"],
                places=2,
            )
            self.assertEqual(
                {factor["name"] for factor in item["score_explanations"]},
                {
                    "Catalog preference similarity",
                    "Gender compatibility",
                    "Customer rating quality",
                    "Budget affordability",
                    "Discount value",
                },
            )
            self.assertTrue(item["explanation_evidence"])

    def test_empty_preferences_still_return_ranked_in_stock_products(self):
        results = self.engine.recommend({})

        self.assertEqual(len(results), 10)
        self.assertTrue(all(item["stock_quantity"] > 0 for item in results))
        self.assertEqual(
            [item["score"] for item in results],
            sorted((item["score"] for item in results), reverse=True),
        )

    def test_brand_preference_changes_content_similarity_ranking(self):
        products = self.products.iloc[[0]].copy()
        preferred = products.copy()
        preferred["product_id"] = "TEST-FAVOURITE"
        preferred["brand"] = "FavouriteBrand"
        other = products.copy()
        other["product_id"] = "TEST-OTHER"
        other["brand"] = "OtherBrand"
        engine = FashionRecommendationEngine(pd.concat([preferred, other], ignore_index=True))

        results = engine.recommend({"preferred_brand": "FavouriteBrand"})

        self.assertEqual([item["product_id"] for item in results], ["TEST-FAVOURITE", "TEST-OTHER"])
        self.assertIn("Matches your brand preference (FavouriteBrand)", results[0]["reasons"])
        self.assertIn(
            "Matches your brand preference (FavouriteBrand)",
            results[0]["explanation_evidence"],
        )

    def test_weather_personalization_is_included_in_exact_score_explanation(self):
        weather = {
            "temperature_c": 31.0,
            "humidity_percent": 82.0,
            "weather_code": 0,
            "weather_condition": "Clear",
        }
        result = self.engine.recommend(
            {
                "preferred_category": "Tops",
                "preferred_price_min": 10,
                "preferred_price_max": 100,
                "body_shape": "Pear/triangle-style",
            },
            weather_context=weather,
        )[0]

        by_name = {factor["name"]: factor for factor in result["score_explanations"]}
        self.assertIn("Weather context fit", by_name)
        self.assertIn("Body-shape styling proxy fit", by_name)
        self.assertEqual(by_name["Weather context fit"]["score"], result["weather_score"])
        self.assertEqual(
            by_name["Body-shape styling proxy fit"]["score"],
            result["body_shape_fit"],
        )
        self.assertAlmostEqual(
            sum(factor["contribution_points"] for factor in result["score_explanations"]),
            result["score"],
            places=2,
        )
        self.assertTrue(
            any(reason.startswith("Weather-context proxy fit:") for reason in result["reasons"])
        )

    def test_positive_evidence_is_consistent_with_product_attributes(self):
        results = self.engine.recommend({"max_budget": 500})

        for item in results:
            if any(reason.startswith("Strong customer rating") for reason in item["explanation_evidence"]):
                self.assertGreaterEqual(item["customer_rating"], 4)
            if any(reason.startswith("Discount contributes") for reason in item["explanation_evidence"]):
                self.assertGreater(item["discount_percentage"], 0)
            self.assertTrue(all(
                factor["contribution_points"] >= 0
                for factor in item["score_explanations"]
            ))

    def test_two_profiles_produce_distinct_ranked_lists_and_match_reasons(self):
        everyday = self.engine.recommend(
            {
                "gender": "Women",
                "preferred_categories": ["Tops", "Bottoms"],
                "preferred_colors": ["Blue"],
                "preferred_brands": ["Uniqlo"],
                "preferred_styles": ["Everyday"],
                "preferred_price_min": 20,
                "preferred_price_max": 80,
            }
        )
        occasion = self.engine.recommend(
            {
                "gender": "Women",
                "preferred_categories": ["Dresses"],
                "preferred_colors": ["Red"],
                "preferred_brands": ["Zara"],
                "preferred_styles": ["Occasion"],
                "preferred_price_min": 100,
                "preferred_price_max": 220,
            }
        )

        self.assertNotEqual(
            [item["product_id"] for item in everyday],
            [item["product_id"] for item in occasion],
        )
        self.assertTrue(
            any(item["category"] in {"Tops", "Bottoms"} for item in everyday[:5])
        )
        self.assertTrue(any(item["category"] == "Dresses" for item in occasion[:5]))
        self.assertTrue(all(item["profile_fit"] is not None for item in everyday))
        self.assertTrue(all(item["profile_fit"] is not None for item in occasion))
        for item in everyday + occasion:
            self.assertAlmostEqual(
                sum(item["score_contributions"].values()),
                item["score"],
                places=1,
            )

    def test_profile_inputs_are_validated(self):
        with self.assertRaisesRegex(ValueError, "Unsupported preferred styles"):
            self.engine.recommend({"preferred_styles": ["Unrecorded style"]})
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self.engine.recommend(
                {"preferred_price_min": 100, "preferred_price_max": 50}
            )
        with self.assertRaisesRegex(ValueError, "Unsupported body shape"):
            self.engine.recommend({"body_shape": "Objective hourglass"})

    def test_body_shape_profile_changes_recommendation_ranking(self):
        pear = self.engine.recommend({"body_shape": "Pear/triangle-style"})
        inverted = self.engine.recommend({"body_shape": "Inverted-triangle-style"})

        self.assertNotEqual(
            [item["product_id"] for item in pear],
            [item["product_id"] for item in inverted],
        )
        self.assertTrue(all(item["body_shape_fit"] is not None for item in pear))
        self.assertTrue(all(item["body_shape_fit"] is not None for item in inverted))

    def test_budget_rating_and_stock_constraints_are_applied(self):
        results = self.engine.recommend(
            {"max_budget": 100, "minimum_rating": 4.5}, top_n=10
        )
        self.assertTrue(all(item["current_price"] <= 100 for item in results))
        self.assertTrue(all(item["customer_rating"] >= 4.5 for item in results))
        self.assertTrue(all(item["stock_quantity"] > 0 for item in results))
        for item in results:
            self.assertIn(
                f"Within maximum budget ({100:.2f})",
                item["eligibility_evidence"],
            )
            self.assertIn(
                "Meets minimum rating (4.5/5)",
                item["eligibility_evidence"],
            )
        self.assertEqual(self.engine.recommend({"max_budget": 1}), [])

    def test_rating_fallback_keeps_budget_and_labels_relaxed_filter(self):
        product = self.products.loc[
            self.products["stock_quantity"] > 0
        ].iloc[[0]].copy()
        product.loc[:, "customer_rating"] = 2.0
        budget = float(product.iloc[0]["current_price"])
        engine = FashionRecommendationEngine(product)

        results = engine.recommend(
            {"max_budget": budget, "minimum_rating": 4.5},
            top_n=1,
        )

        self.assertEqual(len(results), 1)
        self.assertLessEqual(results[0]["current_price"], budget)
        self.assertTrue(results[0]["fallback_used"])
        self.assertIn(
            "Minimum rating filter relaxed: no in-stock product within "
            "the budget met the threshold",
            results[0]["eligibility_evidence"],
        )

    def test_invalid_preferences_and_top_n_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported preferred color"):
            self.engine.recommend({"preferred_color": "Not A Dataset Color"})
        with self.assertRaisesRegex(ValueError, "Maximum budget"):
            self.engine.recommend({"max_budget": "-1"})
        with self.assertRaisesRegex(ValueError, "Minimum rating"):
            self.engine.recommend({"minimum_rating": "5.5"})
        with self.assertRaisesRegex(ValueError, "top_n"):
            self.engine.recommend({}, top_n=0)

    def test_flask_page_and_post_use_the_ranked_engine(self):
        app.config.update(TESTING=True)
        client = app.test_client()

        home = client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn(b"AI FASHION", home.data)
        wizard_home = client.get("/personal-details")
        self.assertEqual(wizard_home.status_code, 200)
        self.assertIn(b"Personal details", wizard_home.data)
        response = client.post(
            "/predict",
            data={
                "gender": "Women",
                "location": "Rural",
                "season": "Summer",
                "preferred_category": "Dresses",
                "preferred_color": "",
                "preferred_brand": "",
                "preferred_size": "",
                "max_budget": "150",
                "minimum_rating": "3.5",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Ranked results", response.data)
        self.assertIn(b"Relevance", response.data)
        self.assertIn(b"Why this recommendation?", response.data)
        self.assertIn(b"result-sort", response.data)
        body_shape_response = client.post(
            "/predict",
            data={
                "body_shape": "Pear/triangle-style",
                "preferred_categories": ["Tops", "Bottoms"],
            },
        )
        self.assertEqual(body_shape_response.status_code, 200)
        self.assertIn(b"Pear/triangle-style", body_shape_response.data)

        invalid = client.post("/predict", data={"preferred_color": "Invalid"})
        self.assertEqual(invalid.status_code, 400)
        self.assertIn(b"Unsupported preferred color", invalid.data)

    @patch("market.body_shape_estimator.estimate")
    def test_uploaded_body_estimate_is_used_and_image_is_not_saved(self, estimate):
        from body_shape import BodyShapeEstimate

        app.config.update(TESTING=True)
        client = app.test_client()
        estimate.return_value = BodyShapeEstimate(
            category="Pear/triangle-style",
            measurements={
                "shoulder_to_hip_width_ratio": 0.82,
                "waist_silhouette_proxy_over_hip_width": None,
            },
            explanation="Shoulder width appears smaller than hip width in this image.",
            limitation="Styling estimate only; not an objective determination.",
            landmark_image_data_uri="data:image/jpeg;base64,dGVzdA==",
        )
        response = client.post(
            "/predict",
            data={
                "body_image": (BytesIO(b"test-image"), "body.jpg"),
                "profile_id": "",
            },
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Image", response.data)
        self.assertIn(b"Pear/triangle-style", response.data)
        self.assertIn(b"landmark", response.data.lower())
        self.assertIn("Styling estimate only", response.get_data(as_text=True))
        self.assertTrue(estimate.called)

    @patch("market.body_shape_estimator.estimate")
    def test_cv_failure_keeps_recommendations_available_without_estimate(self, estimate):
        from body_shape import BodyShapeError

        app.config.update(TESTING=True)
        client = app.test_client()
        estimate.side_effect = BodyShapeError("No person pose was detected.")
        response = client.post(
            "/predict",
            data={
                "body_image": (BytesIO(b"invalid"), "body.jpg"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No person pose was detected.", response.data)
        self.assertIn(b"Ranked results", response.data)

    @patch("market.weather_service.fetch_current")
    def test_flask_integrates_weather_lookup_and_graceful_fallback(self, fetch_current):
        app.config.update(TESTING=True)
        client = app.test_client()
        form = {
            "weather_location": "London",
            "season": "Winter",
            "preferred_category": "Outerwear",
        }
        fetch_current.return_value = WeatherLookup(
            {
                "city": "London",
                "country": "United Kingdom",
                "temperature_c": 7.0,
                "humidity_percent": 80.0,
                "weather_code": 61,
                "weather_condition": "Slight rain",
            },
            source="Open-Meteo current weather",
        )
        response = client.post("/predict", data=form)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"London, United Kingdom", response.data)
        self.assertIn(b"Weather-context fit", response.data)

        fetch_current.return_value = WeatherLookup(
            None,
            "Live weather is unavailable (URLError). Recommendations use your selected fashion preferences and season without weather adjustments.",
        )
        fallback = client.post("/predict", data=form)
        self.assertEqual(fallback.status_code, 200)
        self.assertIn(b"Live weather is unavailable", fallback.data)
        self.assertIn(b"Ranked results", fallback.data)


if __name__ == "__main__":
    unittest.main()
