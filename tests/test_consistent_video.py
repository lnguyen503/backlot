"""Tests for the consistent-character video pipeline (no ComfyUI required).

Requires the `[faces]` extra (insightface + onnxruntime); skipped otherwise."""
from types import SimpleNamespace

import pytest

pytest.importorskip("insightface")
pytest.importorskip("onnxruntime")

from backlot.pipelines.consistent_video import (  # noqa: E402
    DEMO_BEATS, DEMO_MASTER, KEEP, Beat, _abs, _stage,
)


def _fake_cfg(tmp_path):
    return SimpleNamespace(comfyui=SimpleNamespace(
        output_dir=str(tmp_path / "out"), input_dir=str(tmp_path / "in")))


def test_demo_beats_well_formed():
    assert len(DEMO_BEATS) == 8, "the demo narrative is 8 shots"
    for b in DEMO_BEATS:
        assert isinstance(b, Beat)
        assert b.scene.strip() and b.motion.strip()
    assert "front door" in DEMO_MASTER.lower()


def test_keep_clause_locks_identity():
    # The shared KEEP clause is what holds identity/wardrobe across Kontext edits.
    for token in ("face", "hair", "shirt"):        # shirt = the locked wardrobe
        assert token in KEEP.lower()


def test_abs_composes_output_path(tmp_path):
    cfg = _fake_cfg(tmp_path)
    asset = {"filename": "shot.webp", "subfolder": "backlot"}
    p = _abs(cfg, asset)
    assert p.name == "shot.webp" and p.parent.name == "backlot"


def test_stage_copies_into_input_dir(tmp_path):
    cfg = _fake_cfg(tmp_path)
    (tmp_path / "out" / "backlot").mkdir(parents=True)
    (tmp_path / "in").mkdir(parents=True)
    src = tmp_path / "out" / "backlot" / "frame.png"
    src.write_bytes(b"\x89PNG\r\n")
    name = _stage(cfg, {"filename": "frame.png", "subfolder": "backlot"}, "staged.png")
    assert name == "staged.png"
    assert (tmp_path / "in" / "staged.png").read_bytes() == b"\x89PNG\r\n"
