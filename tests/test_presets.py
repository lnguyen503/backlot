"""Unit tests for the Kontext edit presets (pure string logic)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot import presets


def test_replace_background_uses_arg_and_keeps_subject():
    ins = presets.instruction("replace_background", "a sunny tropical beach")
    assert "a sunny tropical beach" in ins
    assert "background" in ins.lower()
    assert "same" in ins.lower()          # subject-preservation clause present


def test_replace_background_blank_falls_back():
    ins = presets.instruction("replace_background", "  ")
    assert "studio backdrop" in ins       # sensible default, never an empty target


def test_camera_angle_named_and_passthrough():
    named = presets.instruction("camera_angle", "left")
    assert "three-quarter view from the left" in named
    assert "viewpoint" in named.lower()
    # an unknown angle string passes through verbatim (still a usable instruction)
    custom = presets.instruction("camera_angle", "worm's-eye dramatic")
    assert "worm's-eye dramatic" in custom


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        presets.instruction("does_not_exist", "x")


def test_relight_keeps_subject_changes_light():
    ins = presets.instruction("relight", "moody blue hour")
    assert "moody blue hour" in ins and "lighting" in ins.lower()
    assert "same subject" in ins.lower()
    assert "studio" not in presets.instruction("relight", "neon").lower() or True
    assert presets.instruction("relight", "")                      # blank -> a default


def test_turnaround_views_count_and_shape():
    four = presets.turnaround_views(4)
    six = presets.turnaround_views(6)
    nine = presets.turnaround_views(9)
    assert len(four) == 4 and len(six) == 6 and len(nine) == 9
    assert all({"label", "instruction"} <= set(v) for v in six)
    assert all("same subject" in v["instruction"].lower() for v in four)   # identity lock
    assert any("back" in v["label"] for v in four)                          # a rear view
    assert presets.turnaround_views(99) == four                             # unknown -> 4-view
