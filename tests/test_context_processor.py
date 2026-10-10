import unittest
from unittest.mock import patch
from urllib.error import URLError

import pandas as pd

from context_processor import WeatherService, score_weather_context
from fashion_preprocessing import validate_and_normalize
from recommendation_engine import FashionRecommendationEngine


DATA_PATH = r"C:\Users\Administrator\Documents\market\market\fashion_boutique_fixed_location.csv"


class WeatherContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.products = validate_and_normalize(pd.read_csv(DATA_PATH))

    @staticmethod
    def product(category, season):
        return {"category": category, "season": season}

    def test_hot_weather_favors_summer_and_light_category_proxy(self):
        hot = {
            "temperature_c": 32.0,
            "humidity_percent": 50.0,
            "weather_code": 0,
        }
        warm_top, _ = score_weather_context(self.product("Tops", "Summer"), hot)
        cold_outerwear, _ = score_weather_context(
            self.product("Outerwear", "Winter"), hot
        )
        self.assertGreater(warm_top, cold_outerwear)

    def test_cold_weather_favors_winter_outerwear_proxy(self):
        cold = {
            "temperature_c": 5.0,
            "humidity_percent": 45.0,
            "weather_code": 3,
        }
        winter_outerwear, reasons = score_weather_context(
            self.product("Outerwear", "Winter"), cold
        )
        summer_dress, _ = score_weather_context(
            self.product("Dresses", "Summer"), cold
        )
        self.assertGreater(winter_outerwear, summer_dress)
        self.assertTrue(any("insulation is not in the data" in reason for reason in reasons))

    def test_rain_favors_relevant_catalog_categories_without_waterproof_claim(self):
        rain = {
            "temperature_c": 19.0,
            "humidity_percent": 60.0,
            "weather_code": 61,
        }
        shoes, reasons = score_weather_context(self.product("Shoes", "Fall"), rain)
        dress, _ = score_weather_context(self.product("Dresses", "Fall"), rain)
        self.assertGreater(shoes, dress)
        self.assertTrue(any("water resistance is not in the data" in r for r in reasons))

    def test_high_humidity_favors_lighter_category_proxy(self):
        humid = {
            "temperature_c": 22.0,
            "humidity_percent": 82.0,
            "weather_code": 2,
        }
        top, reasons = score_weather_context(self.product("Tops", "Spring"), humid)
        outerwear, _ = score_weather_context(
            self.product("Outerwear", "Spring"), humid
        )
        self.assertGreater(top, outerwear)
        self.assertTrue(any("breathability is not measured" in r for r in reasons))

    def test_weather_changes_engine_score_and_reports_component(self):
        preferences = {
            "preferred_category": "Outerwear",
            "season": "Winter",
        }
        products = self.products.loc[
            self.products["category"].isin(["Outerwear", "Dresses"])
        ]
        engine = FashionRecommendationEngine(products)
        baseline = engine.recommend(preferences, top_n=20)
        cold = engine.recommend(
            preferences,
            top_n=20,
            weather_context={
                "temperature_c": 5.0,
                "humidity_percent": 45.0,
                "weather_code": 3,
            },
        )
        self.assertNotEqual(
            [item["score"] for item in baseline],
            [item["score"] for item in cold],
        )
        self.assertTrue(all(item["weather_score"] is not None for item in cold))
        self.assertAlmostEqual(
            sum(cold[0]["score_contributions"].values()),
            cold[0]["score"],
            places=1,
        )

    def test_weather_service_uses_cache_and_returns_live_context(self):
        clock_value = [100.0]
        service = WeatherService(clock=lambda: clock_value[0])

        def api_response(url, _params):
            if "geocoding" in url:
                return {
                    "results": [
                        {
                            "name": "London",
                            "country": "United Kingdom",
                            "latitude": 51.5,
                            "longitude": -0.12,
                        }
                    ]
                }
            return {
                "current": {
                    "temperature_2m": 17.4,
                    "relative_humidity_2m": 72,
                    "weather_code": 61,
                }
            }

        with patch("context_processor._request_json", side_effect=api_response) as request:
            first = service.fetch_current("London")
            second = service.fetch_current(" london ")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(first.context["temperature_c"], 17.4)
        self.assertEqual(first.context["weather_condition"], "Slight rain")
        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertEqual(second.context, first.context)

    def test_weather_service_caches_failure_and_returns_explicit_fallback(self):
        service = WeatherService(clock=lambda: 100.0)
        with patch(
            "context_processor._request_json",
            side_effect=URLError("offline"),
        ) as request:
            first = service.fetch_current("Nowhere")
            second = service.fetch_current("Nowhere")
        self.assertEqual(request.call_count, 1)
        self.assertIsNone(first.context)
        self.assertIn("without weather adjustments", first.message)
        self.assertTrue(second.cached)

    def test_invalid_weather_context_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid temperature or humidity"):
            score_weather_context(
                self.product("Tops", "Summer"),
                {
                    "temperature_c": float("nan"),
                    "humidity_percent": 80,
                    "weather_code": 0,
                },
            )


if __name__ == "__main__":
    unittest.main()
