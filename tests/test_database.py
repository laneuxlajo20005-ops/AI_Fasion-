import tempfile
import unittest
from pathlib import Path

from database import Database


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


if __name__ == "__main__":
    unittest.main()
