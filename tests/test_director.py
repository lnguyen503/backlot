"""Director planning tests — orchestration logic, no LLM/GPU (stubbed)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.storyboard import director
from backlot.storyboard.models import Panel, Storyboard


class _FakeAgent:
    def __init__(self, *a):
        pass

    def draft(self, idea, guidance=""):
        return Storyboard(title="Drafted", panels=[Panel(scene=f"s{i}") for i in range(7)])


class _FakeStore:
    saved = None

    def __init__(self, runs):
        pass

    def save(self, sb):
        _FakeStore.saved = sb


class _FakeEng:
    class cfg:
        class paths:
            runs = "."


def test_plan_caps_shots_and_saves(monkeypatch):
    monkeypatch.setattr(director, "StoryboardAgent", _FakeAgent)
    monkeypatch.setattr(director, "StoryboardStore", _FakeStore)
    monkeypatch.setattr(director, "LLMClient",
                        type("L", (), {"from_config": staticmethod(lambda cfg: None)}))
    sb = director._plan(_FakeEng(), "a lighthouse keeper", "9:16, moody", 4, lambda _m: None)
    assert len(sb.panels) == 4                 # capped from 7
    assert _FakeStore.saved is sb              # persisted so the Studio can edit it


def test_plan_keeps_when_under_cap(monkeypatch):
    monkeypatch.setattr(director, "StoryboardAgent", _FakeAgent)
    monkeypatch.setattr(director, "StoryboardStore", _FakeStore)
    monkeypatch.setattr(director, "LLMClient",
                        type("L", (), {"from_config": staticmethod(lambda cfg: None)}))
    sb = director._plan(_FakeEng(), "x", "", 10, lambda _m: None)
    assert len(sb.panels) == 7                 # 7 <= 10, unchanged


def test_render_visuals_skips_done_and_checkpoints(monkeypatch):
    import asyncio

    from backlot.storyboard.models import Asset
    a = Asset(type="image", filename="x.png")
    sb = Storyboard(title="T", panels=[Panel(still=a, video=a), Panel()])   # p0 done, p1 empty
    saved = []

    class _Store:
        def __init__(self, runs):
            pass

        def save(self, b):
            saved.append(True)

    async def fake_still(eng, board, p, **k):
        p.still = a

    async def fake_anim(eng, board, p, backend=None):
        p.video = a

    monkeypatch.setattr(director, "StoryboardStore", _Store)
    monkeypatch.setattr(director.render, "render_panel_still", fake_still)
    monkeypatch.setattr(director.render, "animate_panel", fake_anim)
    monkeypatch.setattr(director.render, "free_vram", lambda eng: None)
    stills = [p.still for p in sb.panels]
    asyncio.run(director._render_visuals(_FakeEng(), sb, "svd", lambda _m: None))
    assert all(p.still and p.video for p in sb.panels)      # p1 got filled in
    assert stills[0] is sb.panels[0].still                  # p0 still untouched (skipped)
    assert saved                                            # checkpointed as it went


def test_direct_resume_unknown_board_raises(monkeypatch):
    import asyncio

    class _Store:
        def __init__(self, runs):
            pass

        def get(self, bid):
            return None

    monkeypatch.setattr(director, "StoryboardStore", _Store)
    eng = type("E", (), {"ensure_started": lambda self: None,
                         "cfg": type("C", (), {"paths": type("P", (), {"runs": "."})})})()
    with pytest.raises(RuntimeError, match="unknown board"):
        asyncio.run(director.direct(eng, board_id="nope"))
