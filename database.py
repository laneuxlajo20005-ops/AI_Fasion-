"""Database connection helpers for local SQLite and hosted PostgreSQL."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, database_path: Path, database_url: str | None = None):
        self.database_path = Path(database_path)
        self.database_url = database_url
        if not database_url:
            self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> Any:
        if self.database_url:
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as error:
                raise RuntimeError(
                    "PostgreSQL support requires the psycopg package"
                ) from error
            return psycopg.connect(
                self.database_url,
                connect_timeout=5,
                row_factory=dict_row,
            )

        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    def execute(self, connection: Any, query: str, parameters: tuple = ()) -> Any:
        if self.database_url:
            query = query.replace("?", "%s")
        return connection.execute(query, parameters)
