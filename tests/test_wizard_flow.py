import base64
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from body_shape import BodyShapeError, BodyShapeEstimate
from context_processor import WeatherLookup
from market import app, wizard_store


class MultiPageWizardTests(unittest.TestCase):
    def setUp(self):
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def tearDown(self):
        with self.client.session_transaction() as browser_session:
            session_id = browser_session.get("wizard_session_id")
        if session_id:
            wizard_store.delete(session_id)

    def _start_preferences(self, max_budget=None):
        response = self.client.post(
            "/personal-details",
            data={
                "name": "Alex",
                "gender": "Women",
                "city": "London",
                "catalog_location": "Urban",
            },
        )
        self.assertEqual(response.status_code, 302)
        response = self.client.post(
            "/preferences",
            data={
                "season": "Winter",
                "preferred_categories": ["Outerwear", "Tops"],
                "preferred_colors": ["Black"],
                "preferred_brands": ["Zara"],
                "preferred_styles": ["Everyday"],
                "preferred_price_min": "",
                "preferred_price_max": "",
                "max_budget": "" if max_budget is None else str(max_budget),
                "minimum_rating": "",
            },
        )
        self.assertEqual(response.status_code, 302)
        return response

    def test_home_and_project_information_pages(self):
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn(b"AI FASHION", home.data)
        self.assertIn(b"Get started", home.data)
        self.assertIn(b"fashion-smart-mirror.jpg", home.data)
        for path, expected in (
            ("/context", b"Weather information is currently unavailable"),
            ("/about", b"Hybrid Multimodal Context-Aware Personalized Fashion Recommendation System"),
            ("/evaluation", b"Simulated catalog-proxy evaluation"),
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)
                self.assertIn(expected, response.data)

    def test_missing_catalog_shows_a_friendly_unavailable_page(self):
        with patch(
            "market.CATALOG_LOAD_ERROR",
            "The required catalog file is missing.",
        ):
            response = self.client.get("/")
        self.assertEqual(response.status_code, 503)
        self.assertIn(b"The required catalog file is missing.", response.data)
        self.assertNotIn(b"Traceback", response.data)

    def test_evaluation_page_discloses_when_artifacts_are_missing(self):
        with tempfile.TemporaryDirectory() as output_directory:
            with patch("market.EVALUATION_PATH", Path(output_directory)):
                response = self.client.get("/evaluation")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Evaluation has not yet been executed", response.data)
        self.assertIn(b"python final_evaluation.py", response.data)

    def test_route_guards_and_required_personal_details(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/preferences").location, "/personal-details")
        self.assertEqual(self.client.get("/photo-analysis").location, "/personal-details")
        self.assertEqual(self.client.get("/recommendations").location, "/personal-details")
        missing = self.client.post(
            "/personal-details",
            data={"name": "", "gender": "", "city": ""},
        )
        self.assertEqual(missing.status_code, 200)
        self.assertIn(b"Select a gender preference", missing.data)

    def test_personal_preferences_back_navigation_and_refresh_preserve_values(self):
        self._start_preferences()
        back = self.client.get("/personal-details")
        self.assertIn(b'value="Alex"', back.data)
        self.assertIn(b'value="London"', back.data)
        page = self.client.get("/preferences")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'value="Outerwear" checked', page.data)
        self.assertIn(b'value="Winter" selected', page.data)
        self.assertEqual(self.client.get("/photo-analysis").status_code, 200)

    def test_invalid_preferences_are_reported_without_advancing(self):
        self.client.post(
            "/personal-details",
            data={"gender": "Women", "city": "London"},
        )
        response = self.client.post(
            "/preferences",
            data={"preferred_price_min": "200", "preferred_price_max": "50"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Preferred minimum price cannot exceed", response.data)
        self.assertEqual(self.client.get("/photo-analysis").location, "/preferences")

    @patch("market.weather_service.fetch_current")
    def test_skip_photo_completes_recommendation_flow_and_start_over_clears_state(
        self, fetch_weather
    ):
        self._start_preferences()
        fetch_weather.return_value = WeatherLookup(
            {
                "city": "London",
                "country": "United Kingdom",
                "temperature_c": 8.0,
                "humidity_percent": 75.0,
                "weather_code": 3,
                "weather_condition": "Overcast",
            },
            source="Open-Meteo current weather",
        )
        skipped = self.client.post("/photo-analysis", data={"action": "skip"})
        self.assertEqual(skipped.status_code, 302)
        analysis = self.client.get("/ai-analysis")
        self.assertEqual(analysis.status_code, 200)
        self.assertIn(b"Optional photo analysis skipped", analysis.data)
        self.assertIn(b"Live weather context incorporated", analysis.data)
        context_page = self.client.get("/context")
        self.assertIn(b"London, United Kingdom", context_page.data)
        self.assertIn(b"8.0", context_page.data)
        self.assertIn(b"75%", context_page.data)
        results = self.client.get("/recommendations")
        self.assertEqual(results.status_code, 200)
        self.assertIn(b"Your best match", results.data)
        self.assertIn(b"Weather-based picks", results.data)
        self.assertIn(b"Preference matches", results.data)
        self.assertIn(b"Budget-friendly picks", results.data)
        self.assertIn(b"result-sort", results.data)
        self.assertIn(b"Why this product was ranked here", results.data)
        fetch_weather.assert_called_once_with("London")

        with self.client.session_transaction() as browser_session:
            session_id = browser_session["wizard_session_id"]
        self.assertIsNotNone(wizard_store.get(session_id))
        reset = self.client.post("/start-over")
        self.assertEqual(reset.status_code, 302)
        self.assertIsNone(wizard_store.get(session_id))
        self.assertEqual(self.client.get("/ai-analysis").location, "/personal-details")
        new_page = self.client.get("/personal-details")
        self.assertNotIn(b'value="Alex"', new_page.data)

    @patch("market.weather_service.fetch_current")
    def test_budget_fallback_is_explicit_and_does_not_claim_in_budget_picks(
        self, fetch_weather
    ):
        self._start_preferences(max_budget=0)
        fetch_weather.return_value = WeatherLookup(None, "Live weather is unavailable.")
        analysis = self.client.get("/ai-analysis")
        self.assertIn(b"No in-stock products were available within your maximum budget", analysis.data)
        results = self.client.get("/recommendations")
        self.assertEqual(results.status_code, 200)
        self.assertIn(b"Budget fallback shown transparently", results.data)
        self.assertIn(b"No returned recommendations are within your maximum budget", results.data)

    @patch("market.weather_service.fetch_current")
    @patch("market.recommendation_engine.recommend", return_value=[])
    def test_empty_catalog_results_render_a_friendly_state(
        self, _recommend, fetch_weather
    ):
        self._start_preferences()
        fetch_weather.return_value = WeatherLookup(None, "Weather unavailable.")
        analysis = self.client.get("/ai-analysis")
        self.assertEqual(analysis.status_code, 200)
        self.assertIn(b"No in-stock catalog items were available to rank", analysis.data)
        results = self.client.get("/recommendations")
        self.assertEqual(results.status_code, 200)
        self.assertIn(b"No matching catalog items", results.data)

    def test_photo_upload_validation_and_pose_error_offer_retry(self):
        self._start_preferences()
        unsupported = self.client.post(
            "/photo-analysis",
            data={
                "action": "analyze",
                "body_image": (BytesIO(b"not an image"), "portrait.gif"),
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(unsupported.status_code, 200)
        self.assertIn(b"Use a PNG, JPG, JPEG, or WEBP image", unsupported.data)
        self.assertIn(b"Try again", unsupported.data)

        with patch(
            "market.body_shape_estimator.estimate",
            side_effect=BodyShapeError(
                "No person pose was detected. Use one front-facing person."
            ),
        ):
            invalid_pose = self.client.post(
                "/photo-analysis",
                data={
                    "action": "analyze",
                    "body_image": (BytesIO(b"valid-extension-invalid-image"), "body.jpg"),
                },
                content_type="multipart/form-data",
            )
        self.assertEqual(invalid_pose.status_code, 200)
        self.assertIn(b"We couldn&#39;t detect a full-body pose", invalid_pose.data)
        self.assertIn(b"Skip photo analysis", invalid_pose.data)

    @patch("market.body_shape_estimator.estimate")
    def test_webcam_capture_is_analyzed_in_memory_and_remove_clears_estimate(self, estimate):
        self._start_preferences()
        estimate.return_value = BodyShapeEstimate(
            category="Pear/triangle-style",
            measurements={"shoulder_to_hip_width_ratio": 0.82},
            explanation="Shoulders appear narrower than hips in this image.",
            limitation="Experimental styling estimate only.",
            landmark_image_data_uri="data:image/jpeg;base64,dGVzdA==",
        )
        camera_image = "data:image/jpeg;base64," + base64.b64encode(b"camera bytes").decode("ascii")
        analyzed = self.client.post(
            "/photo-analysis",
            data={"action": "analyze", "camera_image": camera_image},
        )
        self.assertEqual(analyzed.status_code, 200)
        self.assertIn(b"Pear/triangle-style", analyzed.data)
        self.assertIn(b"detected pose landmarks", analyzed.data)
        self.assertIn(b"0.820", analyzed.data)
        self.assertTrue(estimate.called)
        with self.client.session_transaction() as browser_session:
            state = wizard_store.get(browser_session["wizard_session_id"])
        serialized_state = json.dumps(state)
        self.assertNotIn("camera bytes", serialized_state)
        self.assertNotIn("landmark_image_data_uri", serialized_state)
        self.assertEqual(
            state["preferences"]["body_shape"],
            "Pear/triangle-style",
        )
        self.client.post("/photo-analysis", data={"action": "continue"})
        saved_estimate_page = self.client.get("/photo-analysis")
        self.assertIn(b"Shoulder To Hip Width Ratio", saved_estimate_page.data)
        self.assertIn(b"0.820", saved_estimate_page.data)
        self.assertIn(b"No confidence value is generated", saved_estimate_page.data)

        removed = self.client.post("/photo-analysis", data={"action": "remove"})
        self.assertEqual(removed.status_code, 302)
        with self.client.session_transaction() as browser_session:
            state = wizard_store.get(browser_session["wizard_session_id"])
        self.assertIsNone(state["body_estimate"])
        self.assertIsNone(state["preferences"]["body_shape"])

    def test_legacy_predict_endpoint_remains_available(self):
        response = self.client.post(
            "/predict",
            data={"preferred_categories": ["Tops"], "gender": "Women"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Ranked results", response.data)


if __name__ == "__main__":
    unittest.main()
