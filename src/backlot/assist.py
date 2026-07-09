"""AI prompt-assist for the Studio — the ✨ button on every prompt box.

Writes a fresh prompt (blank box) or enriches the user's draft, tuned to the medium
(image / video / audio / 3d). Mirrors the storyboard's per-panel assist but for the
one-shot Studio form. Pure over an LLMClient (Ollama) so it's testable + reusable.
"""
from __future__ import annotations

from .engine.llm import LLMClient

_SCHEMA = {"type": "object", "properties": {"prompt": {"type": "string"}},
           "required": ["prompt"]}

# What a good prompt looks like per medium (the model writes to this shape).
_GUIDE = {
    "image": ("a text-to-image prompt: vivid subject, setting, lighting, composition, colour "
              "and art style as comma-separated descriptive phrases; no camera-motion words"),
    "video": ("a text-to-video prompt: a subject plus its MOTION and camera movement over a few "
              "seconds — concrete and cinematic, describing what changes across the shot"),
    "audio": ("an ACE-Step music prompt: comma-separated genre, instruments, tempo/BPM and mood; "
              "prefer an instrumental unless lyrics are clearly wanted"),
    "3d": ("a style prompt that repaints a 3D depth render: the material, surface, lighting and "
           "mood to apply over locked geometry (e.g. 'polished bronze, studio rim light, ultra "
           "detailed'); NO motion or camera words — motion comes from the 3D scene"),
}


def assist_prompt(llm: LLMClient, kind: str, text: str = "", context: str = "") -> str:
    """Write or improve a generation prompt for `kind`. Returns the prompt string."""
    medium = _GUIDE.get(kind, _GUIDE["image"])
    draft = text.strip()
    task = "Improve and enrich this draft into" if draft else "Invent one strong, creative"
    msgs = [
        {"role": "system",
         "content": (f"You are a prompt engineer for a generative-media studio. Produce {medium}. "
                     "Return ONLY the final prompt text in 'prompt' — a single line, no preamble, "
                     "no quotes, no alternatives or explanation.")},
        {"role": "user",
         "content": (f"{task} a {kind} prompt."
                     + (f"\nDraft: {draft}" if draft else "")
                     + (f"\nContext: {context}" if context else ""))},
    ]
    data = llm.chat_json(msgs, _SCHEMA, options={"temperature": 0.8})
    return data.get("prompt", "").strip()
