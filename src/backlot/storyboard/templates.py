"""Use-case templates — one-click starting points (OpenArt-style).

A template gives a non-skilled user a ready-to-fill board instead of a blank
page: the right aspect, a house visual style, and a handful of pre-labelled
panels (scene beat + shot + mood) — leave the prompt empty and hit the ✨ AI
button to write it. Templates also STEER the "Draft with AI" path via `guidance`.
"""
from __future__ import annotations

from typing import Optional

from .models import Panel, Storyboard

# key -> template. `beats` scaffold the panels (scene/shot/mood); `guidance`
# steers the LLM draft. Keep aspect/style opinionated so the result looks right.
TEMPLATES: dict[str, dict] = {
    "short_film": {
        "label": "Short Film", "aspect": "landscape",
        "description": "A cinematic mini-story with a clear arc.",
        "style_notes": "cinematic 35mm, warm teal-orange grade, shallow depth of field",
        "guidance": ("Structure as a SHORT FILM with a clear arc: establishing shot, character "
                     "introduction, rising action, a turn, resolution. About 5 panels."),
        "beats": [("Establishing shot — set the world", "wide establishing", "calm"),
                  ("Introduce the main character", "medium", "curious"),
                  ("Rising action — the problem appears", "close-up", "tense"),
                  ("The turn — a choice or reveal", "close-up", "dramatic"),
                  ("Resolution — the new normal", "wide", "hopeful")],
    },
    "music_video": {
        "label": "Music Video", "aspect": "landscape",
        "description": "Performer + evocative b-roll, bold and rhythmic.",
        "style_notes": "stylized music-video look, saturated color, bold lighting, dynamic",
        "guidance": ("Structure as a MUSIC VIDEO: a performer/artist plus evocative b-roll, "
                     "rhythmic, bold stylized visuals, strong recurring subject. About 5 panels."),
        "beats": [("Artist hook — striking intro of the performer", "medium", "bold"),
                  ("Verse b-roll — mood imagery", "wide", "moody"),
                  ("Chorus — energetic performance wide", "wide", "energetic"),
                  ("Bridge — an intimate contrast beat", "close-up", "intimate"),
                  ("Outro — memorable final image", "medium", "resonant")],
    },
    "micro_drama": {
        "label": "Micro Drama (9:16)", "aspect": "portrait",
        "description": "Vertical melodrama with a twist and a cliffhanger.",
        "style_notes": "high-contrast dramatic lighting, glossy, emotional close-ups",
        "guidance": ("Structure as a vertical MICRO DRAMA: intense melodrama, a shocking twist, "
                     "end on a cliffhanger. Tight emotional close-ups. About 4 panels."),
        "beats": [("Setup — the situation and the stakes", "close-up", "tense"),
                  ("Confrontation — it escalates", "close-up", "heated"),
                  ("The twist — a shocking reveal", "close-up", "shocked"),
                  ("Cliffhanger — leave them hanging", "close-up", "suspenseful")],
    },
    "explainer": {
        "label": "Explainer", "aspect": "landscape",
        "description": "Explain a product, service, or idea clearly.",
        "style_notes": "clean bright modern look, simple backgrounds, clear focus",
        "guidance": ("Structure as an EXPLAINER: hook question, the problem, the solution / how "
                     "it works, a call to action. Clear and friendly. About 4 panels."),
        "beats": [("Hook — a question or relatable problem", "medium", "friendly"),
                  ("The problem — show the pain point", "medium", "concerned"),
                  ("The solution — how it works", "medium", "clear"),
                  ("Call to action — what to do next", "medium", "upbeat")],
    },
    "ugc_ad": {
        "label": "UGC Ad (9:16)", "aspect": "portrait",
        "description": "Casual, authentic vertical product ad.",
        "style_notes": "casual authentic handheld phone look, natural light, real-feeling",
        "guidance": ("Structure as a vertical UGC AD: a relatable hook, an authentic product "
                     "demo/reaction, a clear call to action. Casual and genuine. About 3 panels."),
        "beats": [("Hook — a relatable everyday moment", "close-up selfie", "casual"),
                  ("Demo — the product in real use", "medium", "delighted"),
                  ("Call to action — try it yourself", "close-up selfie", "enthusiastic")],
    },
    "talking_skit": {
        "label": "Talking Skit (9:16)", "aspect": "portrait",
        "description": "Recurring-cast dialogue skit for Shorts.",
        "style_notes": "warm natural, photorealistic, cozy everyday setting",
        "guidance": ("Structure as a TALKING SKIT with a small recurring cast: a setup line, a "
                     "reaction, an escalation, a punchline. Each panel a close-up of one speaker "
                     "with a spoken 'dialogue' line. About 4 panels."),
        "beats": [("Setup — someone starts the bit", "close-up", "playful"),
                  ("Reaction — a friend responds", "close-up", "amused"),
                  ("Escalation — it gets sillier", "close-up", "animated"),
                  ("Punchline — the payoff", "close-up", "deadpan")],
    },
}


def list_templates() -> list[dict]:
    return [{"key": k, "label": t["label"], "description": t["description"],
             "aspect": t["aspect"]} for k, t in TEMPLATES.items()]


def guidance(key: str) -> str:
    t = TEMPLATES.get(key)
    return t["guidance"] if t else ""


def scaffold(key: str, title: str = "") -> Optional[Storyboard]:
    """Build a ready-to-fill board from a template (empty prompts — use ✨ AI)."""
    t = TEMPLATES.get(key)
    if t is None:
        return None
    panels = [Panel(scene=s, shot=shot, mood=mood) for (s, shot, mood) in t["beats"]]
    return Storyboard(title=title.strip() or t["label"], aspect=t["aspect"],
                      style_notes=t["style_notes"], panels=panels)
