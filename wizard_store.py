"""Short-lived server-side storage for the multi-page recommendation wizard."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from database import Database


class WizardStore:
    """Persist workflow state outside Flask's client-readable session cookie."""

    def __init__(
        self,
        database_path: Path,
        ttl_seconds: int = 86_400,
        database_url: str | None = None,
    ):
        self.database = Database(database_path, database_url)
        self.ttl_seconds = ttl_seconds
        self.database.migrate()
        self._delete_expired()

    @staticmethod
    def _normalize_id(session_id: str) -> str:
        try:
            return str(uuid.UUID(str(session_id)))
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError("Wizard session identifier is invalid") from error

    def _delete_expired(self) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - self.ttl_seconds
        with self.database.connection() as connection:
            self.database.execute(
                connection,
                "DELETE FROM fashion_wizard_sessions WHERE updated_at < ?",
                (cutoff,),
            )

    def create(self) -> str:
        session_id = str(uuid.uuid4())
        self.save(session_id, {})
        return session_id

    def save(self, session_id: str, state: dict[str, Any]) -> None:
        normalized_id = self._normalize_id(session_id)
        state_json = json.dumps(state, sort_keys=True, allow_nan=False)
        updated_at = datetime.now(timezone.utc).timestamp()
        with self.database.connection() as connection:
            self.database.execute(
                connection,
                "DELETE FROM fashion_wizard_sessions WHERE updated_at < ?",
                (updated_at - self.ttl_seconds,),
            )
            self.database.execute(
                connection,
                """
                INSERT INTO fashion_wizard_sessions (session_id, state_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    state_json = excluded.state_json,
                    updated_at = excluded.updated_at
                """,
                (normalized_id, state_json, updated_at),
            )

    def get(self, session_id: str) -> dict[str, Any] | None:
        normalized_id = self._normalize_id(session_id)
        with self.database.connection() as connection:
            row = self.database.execute(
                connection,
                "SELECT state_json, updated_at FROM fashion_wizard_sessions WHERE session_id = ?",
                (normalized_id,),
            ).fetchone()
            if row is None:
                return None
            if row["updated_at"] < (
                datetime.now(timezone.utc).timestamp() - self.ttl_seconds
            ):
                self.database.execute(
                    connection,
                    "DELETE FROM fashion_wizard_sessions WHERE session_id = ?",
                    (normalized_id,),
                )
                return None
        return json.loads(row["state_json"])

    def delete(self, session_id: str) -> None:
        normalized_id = self._normalize_id(session_id)
        with self.database.connection() as connection:
            self.database.execute(
                connection,
                "DELETE FROM fashion_wizard_sessions WHERE session_id = ?",
                (normalized_id,),
            )
