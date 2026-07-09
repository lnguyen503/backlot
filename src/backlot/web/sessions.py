"""Session (project) metadata store.

A session groups runs/assets into a project. Metadata lives in one JSON index at
runs/sessions.json; each run manifest carries its `session_id` (see RunStore), so
the gallery can filter by session. Deleting a session removes only the index entry,
never the underlying runs/assets.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional


class SessionStore:
    def __init__(self, runs_dir: str):
        self.path = Path(runs_dir) / "sessions.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8")).get("sessions", [])
        except (OSError, json.JSONDecodeError):
            return []

    def _write(self, sessions: list[dict]) -> None:
        self.path.write_text(json.dumps({"sessions": sessions}, indent=2), encoding="utf-8")

    def list(self) -> list[dict]:
        return sorted(self._read(), key=lambda s: s.get("created_at", 0), reverse=True)

    def create(self, name: str, created_at: float) -> dict:
        sessions = self._read()
        sess = {"id": uuid.uuid4().hex[:12], "name": (name or "Untitled").strip(),
                "created_at": created_at}
        sessions.append(sess)
        self._write(sessions)
        return sess

    def rename(self, sid: str, name: str) -> Optional[dict]:
        sessions = self._read()
        for s in sessions:
            if s["id"] == sid:
                s["name"] = name.strip() or s["name"]
                self._write(sessions)
                return s
        return None

    def delete(self, sid: str) -> bool:
        sessions = self._read()
        kept = [s for s in sessions if s["id"] != sid]
        if len(kept) == len(sessions):
            return False
        self._write(kept)
        return True
