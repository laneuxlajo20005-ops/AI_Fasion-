"""Persistent storage for lightweight, anonymous fashion preference profiles."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from database import Database


class ProfileStore:
    """Persist one preference document per UUID, without authentication."""

    def __init__(self, database_path: Path, database_url: str | None = None):
        self.database = Database(database_path, database_url)
        self.database.migrate()

    @staticmethod
    def _normalize_id(profile_id: str) -> str:
        try:
            return str(uuid.UUID(str(profile_id)))
        except (ValueError, TypeError, AttributeError) as error:
            raise ValueError("Profile identifier is invalid") from error

    def save(
        self,
        profile_id: str,
        profile_name: str,
        preferences: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_id = self._normalize_id(profile_id)
        name = " ".join(str(profile_name).split())
        if not name:
            name = "My fashion profile"
        if len(name) > 60:
            raise ValueError("Profile name must be 60 characters or fewer")
        timestamp = datetime.now(timezone.utc).isoformat()
        preferences_json = json.dumps(preferences, sort_keys=True, allow_nan=False)
        with self.database.connection() as connection:
            self.database.execute(
                connection,
                """
                INSERT INTO fashion_profiles
                    (profile_id, profile_name, preferences_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                    profile_name = excluded.profile_name,
                    preferences_json = excluded.preferences_json,
                    updated_at = excluded.updated_at
                """,
                (normalized_id, name, preferences_json, timestamp, timestamp),
            )
        return {
            "profile_id": normalized_id,
            "profile_name": name,
            "preferences": preferences,
            "updated_at": timestamp,
        }

    def get(self, profile_id: str) -> dict[str, Any] | None:
        normalized_id = self._normalize_id(profile_id)
        with self.database.connection() as connection:
            row = self.database.execute(
                connection,
                """
                SELECT profile_id, profile_name, preferences_json, updated_at
                FROM fashion_profiles
                WHERE profile_id = ?
                """,
                (normalized_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "profile_id": row["profile_id"],
            "profile_name": row["profile_name"],
            "preferences": json.loads(row["preferences_json"]),
            "updated_at": row["updated_at"],
        }
