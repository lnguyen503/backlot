"""Persist storyboards as JSON, one per board, under runs/storyboards/.

Mirrors web.store.RunStore's cheap file-per-record approach: survives restarts,
human-readable, no DB.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .models import Storyboard


class StoryboardStore:
    def __init__(self, runs_dir: str):
        self.root = Path(runs_dir) / "storyboards"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sb_id: str) -> Path:
        return self.root / f"{sb_id}.json"

    def save(self, sb: Storyboard) -> Storyboard:
        sb.updated_at = time.time()
        self._path(sb.id).write_text(
            sb.model_dump_json(indent=2), encoding="utf-8"
        )
        return sb

    def get(self, sb_id: str) -> Optional[Storyboard]:
        p = self._path(sb_id)
        if not p.exists():
            return None
        return Storyboard.model_validate_json(p.read_text(encoding="utf-8"))

    def list(self, limit: int = 200) -> list[dict]:
        """Lightweight index for the picker — not full boards."""
        items: list[dict] = []
        for f in self.root.glob("sb_*.json"):
            try:
                sb = Storyboard.model_validate_json(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            items.append({
                "id": sb.id, "title": sb.title, "logline": sb.logline,
                "panels": len(sb.panels), "updated_at": sb.updated_at,
            })
        items.sort(key=lambda m: m.get("updated_at", 0), reverse=True)
        return items[:limit]

    def delete(self, sb_id: str) -> bool:
        p = self._path(sb_id)
        if p.exists():
            p.unlink()
            return True
        return False
