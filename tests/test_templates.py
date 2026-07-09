"""Unit tests for use-case templates (scaffold + guidance)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.storyboard import templates as tpl


def test_list_templates_shape():
    items = tpl.list_templates()
    assert items and all({"key", "label", "description", "aspect"} <= set(t) for t in items)
    assert {"short_film", "music_video", "micro_drama", "ugc_ad"} <= {t["key"] for t in items}


def test_scaffold_builds_board_with_panels():
    sb = tpl.scaffold("micro_drama", "My Drama")
    assert sb is not None
    assert sb.title == "My Drama" and sb.aspect == "portrait"
    assert sb.style_notes  # a house style is applied
    assert len(sb.panels) == 4
    # scaffold panels carry a scene hint + shot/mood but an EMPTY prompt (use ✨ AI)
    p = sb.panels[0]
    assert p.scene and p.shot and p.mood and p.image_prompt == ""


def test_scaffold_default_title_and_unknown_key():
    sb = tpl.scaffold("ugc_ad")
    assert sb.title == "UGC Ad (9:16)" and sb.aspect == "portrait"
    assert tpl.scaffold("nope") is None


def test_guidance_lookup():
    assert "MUSIC VIDEO" in tpl.guidance("music_video")
    assert tpl.guidance("nope") == ""
