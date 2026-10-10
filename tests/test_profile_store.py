import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from market import app
from profile_store import ProfileStore


class ProfileStoreTests(unittest.TestCase):
    def test_profiles_are_persisted_and_updated_by_browser_uuid(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ProfileStore(Path(temporary_directory) / "profiles.sqlite3")
            profile_id = str(uuid.uuid4())
            first = store.save(
                profile_id,
                "  Everyday  ",
                {"preferred_categories": ["Tops"]},
            )
            self.assertEqual(first["profile_name"], "Everyday")
            self.assertEqual(
                store.get(profile_id)["preferences"]["preferred_categories"],
                ["Tops"],
            )
            store.save(
                profile_id,
                "Occasion",
                {"preferred_categories": ["Dresses"]},
            )
            loaded = store.get(profile_id)
            self.assertEqual(loaded["profile_name"], "Occasion")
            self.assertEqual(
                loaded["preferences"]["preferred_categories"],
                ["Dresses"],
            )

    def test_invalid_profile_id_and_long_name_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ProfileStore(Path(temporary_directory) / "profiles.sqlite3")
            with self.assertRaisesRegex(ValueError, "identifier is invalid"):
                store.get("not-a-uuid")
            with self.assertRaisesRegex(ValueError, "60 characters"):
                store.save(str(uuid.uuid4()), "x" * 61, {})

    def test_profile_api_validates_preferences_and_round_trips(self):
        app.config.update(TESTING=True)
        client = app.test_client()
        profile_id = str(uuid.uuid4())
        payload = {
            "profile_id": profile_id,
            "profile_name": "Blue everyday",
            "preferences": {
                "gender": "Women",
                "location": "Urban",
                "weather_location": "  London  ",
                "season": "Spring",
                "preferred_categories": ["Tops", "Bottoms"],
                "preferred_colors": ["Blue"],
                "preferred_brands": ["Uniqlo"],
                "preferred_styles": ["Everyday"],
                "preferred_price_min": 25,
                "preferred_price_max": 100,
            },
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = ProfileStore(Path(temporary_directory) / "profiles.sqlite3")
            with patch("market.profile_store", store):
                saved = client.post("/api/profile", json=payload)
                loaded = client.get(f"/api/profile/{profile_id}")
                missing = client.get(f"/api/profile/{uuid.uuid4()}")
                invalid_id = client.get("/api/profile/not-a-uuid")
                invalid = client.post(
                    "/api/profile",
                    json={
                        **payload,
                        "preferences": {"preferred_styles": ["Imaginary"]},
                    },
                )

        self.assertEqual(saved.status_code, 200)
        self.assertEqual(loaded.status_code, 200)
        self.assertEqual(
            loaded.json["preferences"]["preferred_categories"],
            ["Tops", "Bottoms"],
        )
        self.assertEqual(loaded.json["preferences"]["weather_location"], "London")
        self.assertEqual(loaded.json["preferences"]["preferred_category"], "Tops")
        self.assertEqual(invalid.status_code, 400)
        self.assertIn("Unsupported preferred styles", invalid.json["error"])
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(invalid_id.status_code, 400)


if __name__ == "__main__":
    unittest.main()
