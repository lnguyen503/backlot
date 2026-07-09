"""Pure-logic tests for the Blender scene catalogue (no Blender install needed)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.blender import scenes


def test_every_preset_has_a_render_body():
    # A scene in the catalogue with no geometry body would fall back silently.
    assert set(scenes.SCENE_PRESETS) == set(scenes.SCENE_BODIES)


def test_scene_list_shape_and_ranges():
    lst = scenes.scene_list()
    assert lst and all(
        {"key", "label", "motion", "prompt", "near", "far"} <= set(s) for s in lst)
    for s in lst:
        assert s["near"] < s["far"] and s["prompt"]        # sane depth range + a prompt


def test_depth_sequence_embeds_the_scene_body():
    # depth_sequence() must splice the requested scene's geometry into the bpy script.
    city = scenes.depth_sequence("city_flythrough")
    assert "primitive_cube_add" in city and "SEQ_DONE" in city
    # unknown scene falls back to the monkey orbit (never empty)
    assert "primitive_monkey_add" in scenes.depth_sequence("nope")
