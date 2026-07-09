"""Unit tests for Studio AI prompt-assist (LLM stubbed — pure prompt-shaping logic)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot import assist


class _StubLLM:
    """Captures the messages assist_prompt builds and returns a canned prompt."""
    def __init__(self):
        self.msgs = None

    def chat_json(self, msgs, schema, options=None):
        self.msgs = msgs
        return {"prompt": "  a neon-lit rainy street, cinematic  "}


def test_returns_trimmed_prompt():
    llm = _StubLLM()
    assert assist.assist_prompt(llm, "image", "") == "a neon-lit rainy street, cinematic"


def test_blank_draft_asks_to_invent_and_medium_specific():
    llm = _StubLLM()
    assist.assist_prompt(llm, "audio", "")
    sys_txt, user_txt = llm.msgs[0]["content"], llm.msgs[1]["content"]
    assert "ACE-Step" in sys_txt              # audio guide selected
    assert "Invent" in user_txt and "Draft:" not in user_txt


def test_with_draft_asks_to_improve_and_includes_it():
    llm = _StubLLM()
    assist.assist_prompt(llm, "3d", "a bronze ring")
    sys_txt, user_txt = llm.msgs[0]["content"], llm.msgs[1]["content"]
    assert "depth render" in sys_txt          # 3d guide selected
    assert "Improve" in user_txt and "a bronze ring" in user_txt


def test_unknown_kind_falls_back_to_image():
    llm = _StubLLM()
    assist.assist_prompt(llm, "hologram", "")
    assert "text-to-image" in llm.msgs[0]["content"]
