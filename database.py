"""Persistent database connections and versioned schema migrations."""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import urlsplit

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.rows import dict_row

MIGRATIONS = (
    (
        1,
        (
            """
            CREATE TABLE IF NOT EXISTS fashion_profiles (
                profile_id TEXT PRIMARY KEY,
                profile_name TEXT NOT NULL,
                preferences_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS fashion_wizard_sessions (
                session_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_fashion_wizard_sessions_updated_at
            ON fashion_wizard_sessions (updated_at)
            """,
        ),
    ),
)
MIGRATION_LOCK_ID = 6_804_209_177


class DatabaseUnavailable(RuntimeError):
    """The configured persistent database cannot serve requests."""


def database_url_from_environment(
    environ: Mapping[str, str] | None = None,
) -> str | None:
    values = os.environ if environ is None else environ
    database_url = values.get("DATABASE_URL", "").strip() or None
    production = (
        values.get("VERCEL") == "1"
        or values.get("APP_ENV", "").strip().lower() == "production"
    )
    if production and not database_url:
        raise RuntimeError(
            "DATABASE_URL is required in production; local SQLite is development-only."
        )
    return database_url


class Database:
    def __init__(self, database_path: Path, database_url: str | None = None):
        self.database_path = Path(database_path)
        self.database_url = database_url.strip() if database_url else None
        if self.database_url:
            if urlsplit(self.database_url).scheme not in {"postgres", "postgresql"}:
                raise ValueError("DATABASE_URL must use PostgreSQL")
        else:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def is_postgresql(self) -> bool:
        return self.database_url is not None

    def connect(self) -> Any:
        if self.database_url:
            try:
                options = conninfo_to_dict(self.database_url)
                local_host = options.get("host") in {
                    "localhost",
                    "127.0.0.1",
                    "::1",
                }
                sslmode = options.get("sslmode")
                if sslmode not in {"verify-ca", "verify-full"}:
                    if local_host and sslmode in {None, "disable"}:
                        options["sslmode"] = "disable"
                    else:
                        options["sslmode"] = "require"
                options["connect_timeout"] = 5
                conninfo = make_conninfo(**options)
                return psycopg.connect(conninfo, row_factory=dict_row)
            except psycopg.Error:
                raise DatabaseUnavailable(
                    "The configured PostgreSQL database is unavailable. "
                    "Check DATABASE_URL and database network access."
                ) from None

        try:
            connection = sqlite3.connect(self.database_path, timeout=5)
        except sqlite3.Error:
            raise DatabaseUnavailable(
                "The local development database is unavailable."
            ) from None
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def connection(self) -> Iterator[Any]:
        connection = self.connect()
        try:
            yield connection
            connection.commit()
        except psycopg.Error:
            connection.rollback()
            raise DatabaseUnavailable(
                "The configured PostgreSQL database could not complete the "
                "operation. Check database availability and schema permissions."
            ) from None
        except sqlite3.Error:
            connection.rollback()
            raise DatabaseUnavailable(
                "The local development database could not complete the operation."
            ) from None
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def execute(
        self,
        connection: Any,
        query: str,
        parameters: tuple = (),
    ) -> Any:
        if self.is_postgresql:
            query = query.replace("?", "%s")
        try:
            return connection.execute(query, parameters)
        except psycopg.Error:
            raise DatabaseUnavailable(
                "The configured PostgreSQL database could not complete the "
                "operation. Check database availability and schema permissions."
            ) from None
        except sqlite3.Error:
            raise DatabaseUnavailable(
                "The local development database could not complete the operation."
            ) from None

    def migrate(self) -> None:
        with self.connection() as connection:
            if self.is_postgresql:
                self.execute(
                    connection,
                    "SELECT pg_advisory_xact_lock(?)",
                    (MIGRATION_LOCK_ID,),
                )
            else:
                connection.execute("BEGIN IMMEDIATE")

            self.execute(
                connection,
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
                """,
            )
            applied_versions = {
                row["version"]
                for row in self.execute(
                    connection,
                    "SELECT version FROM schema_migrations",
                ).fetchall()
            }
            known_versions = {version for version, _statements in MIGRATIONS}
            if applied_versions - known_versions:
                raise RuntimeError(
                    "The database schema is newer than this application's migrations."
                )

            for version, statements in MIGRATIONS:
                if version in applied_versions:
                    continue
                for statement in statements:
                    self.execute(connection, statement)
                self.execute(
                    connection,
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (
                        version,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )

def migrate_from_environment() -> None:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise SystemExit(
            "Set DATABASE_URL to the PostgreSQL connection string before migrating."
        )
    database = Database(
        Path(__file__).resolve().parent
        / "instance"
        / "fashion_profiles.sqlite3",
        database_url,
    )
    database.migrate()
    print("Database schema is up to date.")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["migrate"]:
        raise SystemExit("Usage: python -m database migrate")
    migrate_from_environment()
