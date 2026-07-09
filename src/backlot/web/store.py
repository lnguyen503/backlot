"""Persist completed runs as manifests and list them for the gallery.

One manifest per run at runs/<run_id>/manifest.json (per spec §3). Cheap to read,
survives restarts, and is the source of truth for the gallery.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional


class RunStore:
    def __init__(self, runs_dir: str):
        self.root = Path(runs_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, run_id: str, workflow: str, params: dict, status: dict,
             created_at: float, session_id: Optional[str] = None) -> dict:
        manifest = {
            "run_id": run_id,
            "workflow": workflow,
            "params": params,
            "state": status.get("state"),
            "outputs": status.get("outputs", []),
            "error": status.get("error"),
            "created_at": created_at,
            "session_id": session_id,
        }
        run_dir = self.root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest

    def list(self, limit: int = 200, session_id: Optional[str] = None) -> list[dict[str, Any]]:
        items: list[dict] = []
        for mf in self.root.glob("*/manifest.json"):
            try:
                items.append(json.loads(mf.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        if session_id is not None:
            items = [m for m in items if m.get("session_id") == session_id]
        items.sort(key=lambda m: m.get("created_at", 0), reverse=True)
        return items[:limit]

    def get(self, run_id: str) -> Optional[dict]:
        mf = self.root / run_id / "manifest.json"
        if not mf.exists():
            return None
        return json.loads(mf.read_text(encoding="utf-8"))

    def delete(self, run_id: str) -> bool:
        """Remove a creation (its run dir). Only deletes a dir that actually holds
        a manifest — protects sibling dirs (storyboards/, library/) — and rejects
        any path-traversal. The referenced media in ComfyUI's output dir is left."""
        if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
            return False
        run_dir = self.root / run_id
        if not (run_dir / "manifest.json").exists():
            return False
        shutil.rmtree(run_dir, ignore_errors=True)
        return not run_dir.exists()

    def delete_many(self, run_ids: list[str]) -> int:
        return sum(1 for rid in run_ids if self.delete(rid))
