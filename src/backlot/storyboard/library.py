"""Global, cross-project Characters & Worlds library (OpenArt-style).

Storyboard AssetCards are per-board; this persists REUSABLE cards — a recurring
cast plus reusable environments / styles / objects, each with its locked
reference (and, for characters, a voice) — so every new episode can pull the
SAME character in instead of re-drafting it. That cross-episode consistency is
the recurring-cast moat for a channel.

One JSON file per card under runs/library/, mirroring StoryboardStore.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .models import AssetCard


class LibraryStore:
    def __init__(self, runs_dir: str):
        self.root = Path(runs_dir) / "library"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, card_id: str) -> Path:
        return self.root / f"{card_id}.json"

    def save(self, card: AssetCard) -> AssetCard:
        """Upsert a card (by id) into the library, stamping saved_at."""
        data = card.model_dump()
        data["saved_at"] = time.time()
        self._path(card.id).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return card

    def get(self, card_id: str) -> Optional[AssetCard]:
        p = self._path(card_id)
        if not p.exists():
            return None
        return AssetCard.model_validate_json(p.read_text(encoding="utf-8"))

    def list(self, bucket: Optional[str] = None) -> list[dict]:
        """All library cards (optionally one bucket), newest first."""
        items: list[dict] = []
        for f in self.root.glob("card_*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            if bucket and data.get("bucket") != bucket:
                continue
            items.append(data)
        items.sort(key=lambda d: d.get("saved_at", 0), reverse=True)
        return items

    def delete(self, card_id: str) -> bool:
        p = self._path(card_id)
        if p.exists():
            p.unlink()
            return True
        return False
