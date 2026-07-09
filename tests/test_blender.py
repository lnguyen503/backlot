"""Blender bridge tests. Skipped automatically when Blender isn't installed."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.blender import runner, scenes

try:
    _BLENDER = runner.find_blender()
except RuntimeError:
    _BLENDER = None

pytestmark = pytest.mark.skipif(_BLENDER is None, reason="Blender not installed")


def test_find_blender_path():
    assert _BLENDER and Path(_BLENDER).exists()


def test_version():
    assert runner.version().startswith(("4.", "5."))


def test_run_script_tags_and_errors():
    r = runner.run_script("print('HELLO', 1+1)", timeout=120)
    assert r.ok and r.tagged("HELLO") == ["2"]
    bad = runner.run_script("raise RuntimeError('boom')", timeout=120)
    assert not bad.ok          # traceback in stderr -> ok is False even if rc==0


def test_smoke_render(tmp_path):
    out = tmp_path / "render.png"
    r = runner.run_script(scenes.smoke_render(), args=[str(out)], timeout=240)
    assert r.ok, r.stderr[-500:]
    assert out.exists() and out.stat().st_size > 1000
    assert r.tagged("ENGINE_USED")          # an engine was selected


def test_depth_pass(tmp_path):
    beauty, depth = tmp_path / "b.png", tmp_path / "d.png"
    r = runner.run_script(scenes.depth_pass(), args=[str(beauty), str(depth)], timeout=240)
    assert r.ok, r.stderr[-500:]
    assert beauty.exists() and depth.exists()
