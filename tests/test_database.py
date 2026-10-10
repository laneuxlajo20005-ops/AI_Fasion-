import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import psycopg

from database import (
    Database,
    DatabaseUnavailable,
    database_url_from_environment,
)
from market import app
from profile_store import ProfileStore
from wizard_store import WizardStore

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")


class RecordingConnection:
    def __init__(self):
        self.query = None
        self.parameters = None

    def execute(self, query, parameters):
        self.query = query
        self.parameters = parameters
        return self


class DatabaseTests(unittest.TestCase):
    def test_postgresql_mode_uses_postgres_placeholders_without_creating_sqlite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            database_path = Path(temporary_directory) / "readonly" / "profiles.sqlite3"
            database = Database(database_path, "postgresql://example.invalid/db")
            connection = RecordingConnection()

            database.execute(
                connection,
                "SELECT state_json FROM sessions WHERE session_id = ?",
                ("session-id",),
            )

            self.assertFalse(database_path.parent.exists())
            self.assertEqual(
                connection.query,
                "SELECT state_json FROM sessions WHERE session_id = %s",
            )
            self.assertEqual(connection.parameters, ("session-id",))

    def test_production_requires_database_url_but_development_can_use_sqlite(self):
        with self.assertRaisesRegex(RuntimeError, "DATABASE_URL is required"):
            database_url_from_environment({"APP_ENV": "production"})
        with self.assertRaisesRegex(RuntimeError, "DATABASE_URL is required"):
            database_url_from_environment({"VERCEL": "1"})
        self.assertIsNone(database_url_from_environment({"APP_ENV": "development"}))
        self.assertEqual(
            database_url_from_environment(
                {
                    "APP_ENV": "production",
                    "DATABASE_URL": "postgresql://user:secret@example/db",
                }
            ),
            "postgresql://user:secret@example/db",
        )

    def test_migrations_are_versioned_and_idempotent_for_local_tests(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "profiles.sqlite3"
            store = ProfileStore(path)
            wizard = WizardStore(path)
            session_id = wizard.create()
            wizard.save(session_id, {"step": "preferences"})
            store = ProfileStore(path)
            wizard = WizardStore(path)

            self.assertEqual(wizard.get(session_id), {"step": "preferences"})
            with store.database.connection() as connection:
                versions = store.database.execute(
                    connection,
                    "SELECT version FROM schema_migrations",
                ).fetchall()
            self.assertEqual([row["version"] for row in versions], [1])

    def test_postgresql_connection_failure_is_sanitized(self):
        database = Database(
            Path("unused.sqlite3"),
            "postgresql://username:password@example.invalid/database",
        )
        with patch(
            "database.psycopg.connect",
            side_effect=psycopg.OperationalError("sensitive connection detail"),
        ):
            with self.assertRaises(DatabaseUnavailable) as caught:
                database.connect()
        self.assertNotIn("password", str(caught.exception))
        self.assertNotIn("sensitive connection detail", str(caught.exception))

    def test_database_failure_returns_service_unavailable(self):
        app.config.update(TESTING=True)
        client = app.test_client()
        session_id = str(uuid.uuid4())
        with client.session_transaction() as browser_session:
            browser_session["wizard_session_id"] = session_id
        with patch(
            "market.wizard_store.get",
            side_effect=DatabaseUnavailable("do not expose implementation detail"),
        ):
            page = client.get("/personal-details")
        self.assertEqual(page.status_code, 503)
        self.assertIn(b"persistent database is temporarily unavailable", page.data)
        self.assertNotIn(b"implementation detail", page.data)

    @unittest.skipUnless(TEST_DATABASE_URL, "Set TEST_DATABASE_URL for PostgreSQL integration tests")
    def test_postgresql_profile_and_wizard_state_persist_across_store_instances(self):
        database_path = Path("unused.sqlite3")
        profile_id = str(uuid.uuid4())
        session_id = str(uuid.uuid4())
        profile = ProfileStore(database_path, database_url=TEST_DATABASE_URL)
        wizard = WizardStore(database_path, database_url=TEST_DATABASE_URL)
        try:
            profile.save(profile_id, "PostgreSQL integration", {"style": "Classic"})
            wizard.save(session_id, {"step": "recommendations", "ready": True})
            reloaded_profile = ProfileStore(
                database_path, database_url=TEST_DATABASE_URL
            ).get(profile_id)
            reloaded_wizard = WizardStore(
                database_path, database_url=TEST_DATABASE_URL
            ).get(session_id)

            self.assertEqual(
                reloaded_profile["preferences"],
                {"style": "Classic"},
            )
            self.assertEqual(
                reloaded_wizard,
                {"ready": True, "step": "recommendations"},
            )
        finally:
            with profile.database.connection() as connection:
                profile.database.execute(
                    connection,
                    "DELETE FROM fashion_profiles WHERE profile_id = ?",
                    (profile_id,),
                )
                profile.database.execute(
                    connection,
                    "DELETE FROM fashion_wizard_sessions WHERE session_id = ?",
                    (session_id,),
                )


if __name__ == "__main__":
    unittest.main()
