"""Storyboard authoring agent (TODO item 1): turns an *idea* into a structured,
editable board — logline + reusable assets in 4 buckets + ordered panels (scene,
image prompt, shot, camera, mood, motion) — then refines it conversationally,
keeping asset cards so identity stays locked. "Stay the director": model proposes,
human makes the frame-level calls."""
from __future__ import annotations

from typing import Optional

from ..engine.llm import LLMClient
from .models import Asset, AssetCard, Panel, Storyboard

# JSON schema the model must fill. Assets are referenced from panels BY NAME;
# we resolve names -> card ids after construction (the model never sees ids).
_BOARD_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "logline": {"type": "string"},
        "style_notes": {"type": "string"},
        "aspect": {"type": "string", "enum": ["portrait", "landscape", "square"]},
        "assets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "bucket": {"type": "string",
                               "enum": ["style", "character", "environment", "object"]},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "prompt": {"type": "string"},
                },
                "required": ["bucket", "name", "description", "prompt"],
            },
        },
        "panels": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene": {"type": "string"},
                    "image_prompt": {"type": "string"},
                    "shot": {"type": "string"},
                    "camera": {"type": "string"},
                    "mood": {"type": "string"},
                    "motion_prompt": {"type": "string"},
                    "assets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["scene", "image_prompt", "shot", "camera",
                             "mood", "motion_prompt", "assets"],
            },
        },
    },
    "required": ["title", "logline", "style_notes", "aspect", "assets", "panels"],
}

_SYSTEM = (
    "You are a film director's storyboard partner. The user gives you an IDEA, not a "
    "prompt. You turn it into a concrete, shootable storyboard.\n\n"
    "Rules:\n"
    "1. Define the board's REUSABLE ASSETS once, in four buckets: visual style (exactly "
    "one style card), characters, environments, objects. Give each a vivid, specific "
    "description and a standalone image 'prompt' that could generate a clean reference of "
    "it alone (for a character: full-body, neutral pose, plain background; for an "
    "environment: the empty location; for an object: the object on plain background; for "
    "style: a short style phrase like 'cinematic 35mm, warm teal-orange, shallow depth').\n"
    "2. Then write ordered PANELS. Each panel is one beat. Its 'image_prompt' must be a "
    "full, self-contained image prompt that bakes in the visual style and describes the "
    "scene concretely. Reference the relevant assets by their exact NAME in 'assets' so the "
    "same character/place/object recurs and stays consistent.\n"
    "3. 'motion_prompt' describes the ACTION/camera move for animating that still later.\n"
    "4. Keep characters', wardrobe, style and setting consistent across panels by reusing "
    "the SAME asset names — never invent a new name for the same thing.\n"
    "5. Be concrete and visual. No meta commentary. Honor any count/style the user asks for; "
    "otherwise aim for 5-6 panels.\n"
    "6. Write the title, asset names and shot labels in natural Title Case (e.g. 'Elias "
    "Vance', 'Rooftop Garden') — never SNAKE_CASE or ALL-CAPS. Always include exactly one "
    "'style' bucket card."
)


def _to_board(data: dict, idea: str) -> Storyboard:
    """Build a Storyboard from the model's JSON, resolving asset names -> ids."""
    cards: list[AssetCard] = []
    by_name: dict[str, str] = {}
    for a in data.get("assets", []):
        card = AssetCard(
            bucket=a.get("bucket", "object"), name=a.get("name", "asset"),
            description=a.get("description", ""), prompt=a.get("prompt", ""),
        )
        cards.append(card)
        by_name[card.name.strip().lower()] = card.id

    panels: list[Panel] = []
    for p in data.get("panels", []):
        ids = [by_name[n.strip().lower()] for n in p.get("assets", [])
               if n.strip().lower() in by_name]
        panels.append(Panel(
            scene=p.get("scene", ""), image_prompt=p.get("image_prompt", ""),
            shot=p.get("shot", ""), camera=p.get("camera", ""),
            mood=p.get("mood", ""), motion_prompt=p.get("motion_prompt", ""),
            asset_ids=ids,
        ))

    return Storyboard(
        title=data.get("title", "Untitled Storyboard"), idea=idea,
        logline=data.get("logline", ""), style_notes=data.get("style_notes", ""),
        aspect=data.get("aspect", "landscape"), assets=cards, panels=panels,
    )


class StoryboardAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def draft(self, idea: str, guidance: str = "") -> Storyboard:
        """Idea -> a full first-draft board. `guidance` steers structure/format
        (e.g. a use-case template like 'music video' or 'micro drama')."""
        extra = f"\n\nFORMAT GUIDANCE: {guidance}" if guidance else ""
        msgs = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"IDEA: {idea}{extra}\n\nDraft the storyboard."},
        ]
        data = self.llm.chat_json(msgs, _BOARD_SCHEMA)
        sb = _to_board(data, idea)
        sb.chat = [
            {"role": "user", "content": idea},
            {"role": "assistant", "content": sb.logline},
        ]
        return sb

    def refine(self, sb: Storyboard, message: str) -> Storyboard:
        """Apply a conversational instruction, returning an updated board that
        PRESERVES the existing assets/panels except where the user redirects."""
        current = sb.model_dump(include={
            "title", "logline", "style_notes", "aspect", "assets", "panels"
        })
        # Strip rendered outputs from the snapshot the model reasons over.
        for c in current["assets"]:
            for k in ("ref", "source", "seed", "id"):
                c.pop(k, None)
        # Re-key panel asset_ids back to names so the model edits by name.
        id_to_name = {c.id: c.name for c in sb.assets}
        for src, dst in zip(sb.panels, current["panels"]):
            dst["assets"] = [id_to_name.get(i, "") for i in src.asset_ids]
            for k in ("id", "still", "video", "source", "seed", "asset_ids"):
                dst.pop(k, None)
        import json as _json
        msgs = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",
             "content": (f"Current storyboard:\n{_json.dumps(current, indent=2)}\n\n"
                         f"Director's note: {message}\n\n"
                         "Return the FULL updated storyboard. Keep everything the note "
                         "doesn't change; reuse the same asset names.")},
        ]
        data = self.llm.chat_json(msgs, _BOARD_SCHEMA)
        updated = _to_board(data, sb.idea)
        updated.id = sb.id
        updated.created_at = sb.created_at
        # Carry over already-rendered refs/stills/videos by matching names/order.
        _carry_renders(sb, updated)
        updated.chat = sb.chat + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": updated.logline},
        ]
        return updated


_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {"type": "string"},
        "instrumental": {"type": "boolean"},
        "lyrics": {"type": "string"},
    },
    "required": ["tags", "instrumental", "lyrics"],
}


def suggest_score(llm: LLMClient, sb: Storyboard) -> dict:
    """Ask the LLM for an ACE-Step soundtrack brief fitting the board's mood.

    Returns {tags, lyrics}. `tags` is a comma-separated style/genre/instrument/mood
    string (what ACE-Step wants); `lyrics` is "" for an instrumental.
    """
    moods = ", ".join(sorted({p.mood for p in sb.panels if p.mood}))
    msgs = [
        {"role": "system",
         "content": ("You are a film composer. Given a short film's logline, visual style and "
                     "panel moods, propose a fitting score for ACE-Step. 'tags' = a concise "
                     "comma-separated list of genre, instruments, tempo and mood (e.g. "
                     "'ambient cinematic, solo piano, soft strings, slow, melancholic, hopeful'). "
                     "Prefer an instrumental (instrumental=true, lyrics='') for a short like this "
                     "unless lyrics clearly fit; keep any lyrics very short.")},
        {"role": "user",
         "content": (f"Title: {sb.title}\nLogline: {sb.logline}\nVisual style: {sb.style_notes}\n"
                     f"Panel moods: {moods}\n\nPropose the score.")},
    ]
    data = llm.chat_json(msgs, _SCORE_SCHEMA, options={"temperature": 0.6})
    return {"tags": data.get("tags", "ambient cinematic, instrumental, slow, emotional"),
            "lyrics": "" if data.get("instrumental", True) else data.get("lyrics", "")}


_AMBIENT_SCHEMA = {
    "type": "object",
    "properties": {"tags": {"type": "string"}},
    "required": ["tags"],
}


def suggest_ambient(llm: LLMClient, sb: Storyboard) -> str:
    """ACE-Step tags for a subtle ambient/SFX bed (room tone/weather/nature; no music)."""
    envs = ", ".join(sorted({c.name for c in sb.assets if c.bucket == "environment"}))
    scenes = " | ".join(p.scene for p in sb.panels if p.scene)
    msgs = [
        {"role": "system",
         "content": ("You are a sound designer. Propose a subtle AMBIENT/SFX bed — environmental "
                     "atmosphere only (room tone, weather, nature, crowd, hum), NO music/melody. "
                     "'tags' = a short comma-separated list like 'garden ambience, birdsong, breeze'.")},
        {"role": "user",
         "content": (f"Title: {sb.title}\nLogline: {sb.logline}\nEnvironments: {envs}\n"
                     f"Beats: {scenes}\n\nPropose the ambient bed.")},
    ]
    data = llm.chat_json(msgs, _AMBIENT_SCHEMA, options={"temperature": 0.5})
    return data.get("tags", "").strip() or "soft ambient room tone, subtle background atmosphere"


_NARRATION_SCHEMA = {
    "type": "object",
    "properties": {"narration": {"type": "string"}},
    "required": ["narration"],
}


def write_narration(llm: LLMClient, sb: Storyboard, seconds: Optional[float] = None) -> str:
    """Write a concise spoken voiceover for the short (a few short sentences).

    `seconds` (the sequence length) lets us keep the script speakable in that window
    (~2.5 words/sec). Returns the narration text.
    """
    budget = int((seconds or 12) * 2.3) if seconds else 28
    beats = " | ".join(p.scene for p in sb.panels if p.scene)
    msgs = [
        {"role": "system",
         "content": ("You are a film narrator/scriptwriter. Write a SHORT, evocative voiceover "
                     "for a wordless montage — first person or omniscient, present tense, no stage "
                     "directions, no quotes, just the spoken words. It must be speakable in the "
                     f"given time: keep it under about {budget} words total. Make it flow as one "
                     "continuous narration across the beats, not a list.")},
        {"role": "user",
         "content": (f"Title: {sb.title}\nLogline: {sb.logline}\nBeats: {beats}\n\n"
                     f"Write the voiceover (≤ ~{budget} words).")},
    ]
    data = llm.chat_json(msgs, _NARRATION_SCHEMA, options={"temperature": 0.7})
    return data.get("narration", "").strip()


_PANEL_ASSIST_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string"},
        "image_prompt": {"type": "string"},
        "shot": {"type": "string"},
        "camera": {"type": "string"},
        "mood": {"type": "string"},
        "motion_prompt": {"type": "string"},
    },
    "required": ["scene", "image_prompt", "shot", "camera", "mood", "motion_prompt"],
}


def assist_panel(llm: LLMClient, sb: Storyboard, panel: Panel) -> dict:
    """AI-write or improve one panel's prompt fields — the 'help me prompt' button.

    Uses the board (logline/style/aspect) and the panel's LINKED anchors so the
    image prompt references the same named characters/locations (consistency).
    If the panel already has an image prompt it is enriched (more vivid, specific,
    cinematic) keeping intent; if empty, it's written from the scene.
    """
    anchors = sb.cards_for(panel)
    roster = "; ".join(f"{c.bucket}:{c.name} ({c.description})" for c in anchors) or "none linked"
    have = bool(panel.image_prompt.strip())
    task = ("IMPROVE the existing panel: keep its intent and any named characters/locations, but "
            "make image_prompt more vivid, specific and cinematic (lighting, lens, composition, "
            "detail)." if have else
            "WRITE this panel from its scene: a strong, vivid, cinematic image_prompt.")
    msgs = [
        {"role": "system",
         "content": ("You are a film director and prompt engineer for an image/video model. For ONE "
                     "storyboard panel, return improved fields. image_prompt = one or two rich "
                     "descriptive sentences (NOT a list); reference the linked characters/locations "
                     "BY NAME so identity stays consistent; fit the board's visual style. Keep "
                     "shot/camera/mood tight; motion_prompt describes the movement for the video. "
                     "Do NOT invent new named characters.")},
        {"role": "user",
         "content": (f"Board: {sb.title}\nLogline: {sb.logline}\nVisual style: {sb.style_notes}\n"
                     f"Aspect: {sb.aspect}\nLinked anchors: {roster}\n\nCurrent panel —\n"
                     f" scene: {panel.scene}\n image_prompt: {panel.image_prompt}\n shot: {panel.shot}\n"
                     f" camera: {panel.camera}\n mood: {panel.mood}\n motion_prompt: {panel.motion_prompt}\n\n"
                     f"{task} Return every field.")},
    ]
    return llm.chat_json(msgs, _PANEL_ASSIST_SCHEMA, options={"temperature": 0.7})


# Kokoro voice menu (built-in, no reference clip). The LLM casts from this.
KOKORO_VOICES = {
    "am_michael": "American male, warm and steady",
    "am_adam": "American male, younger and earnest",
    "bm_george": "British male, mature and refined",
    "bm_lewis": "British male, deep and calm",
    "af_heart": "American female, warm and friendly",
    "af_bella": "American female, bright and lively",
    "bf_emma": "British female, calm and measured",
    "bf_isabella": "British female, crisp and confident",
}


def cast_voices(llm: LLMClient, characters: list) -> dict:
    """Ask the LLM to cast a Kokoro voice per character (gender/personality match).

    `characters` is a list of AssetCard. Returns {card_id: voice_id}, guaranteeing a
    valid voice for every character (falls back to round-robin if the LLM picks an
    unknown id) and avoiding obvious duplicates where possible.
    """
    if not characters:
        return {}
    menu = "\n".join(f"- {v}: {d}" for v, d in KOKORO_VOICES.items())
    roster = "\n".join(f"{i}. {c.name}: {c.description}" for i, c in enumerate(characters))
    schema = {"type": "object", "properties": {"casting": {"type": "array", "items": {
        "type": "object",
        "properties": {"index": {"type": "integer"}, "voice": {"type": "string"}},
        "required": ["index", "voice"]}}}, "required": ["casting"]}
    msgs = [
        {"role": "system",
         "content": ("You are a casting director assigning voices. For each character pick the "
                     "BEST-matching voice id from the menu — match apparent gender first, then "
                     "personality/age. Prefer giving each character a DISTINCT voice.\n\n"
                     f"Voice menu:\n{menu}")},
        {"role": "user", "content": f"Characters:\n{roster}\n\nCast one voice id per character."},
    ]
    pool = list(KOKORO_VOICES)
    out: dict = {}
    try:
        data = llm.chat_json(msgs, schema, options={"temperature": 0.3})
        for item in data.get("casting", []):
            i = item.get("index")
            v = item.get("voice", "").strip()
            if isinstance(i, int) and 0 <= i < len(characters) and v in KOKORO_VOICES:
                out[characters[i].id] = v
    except Exception:
        pass
    # fill any gaps with round-robin pool picks
    for i, c in enumerate(characters):
        out.setdefault(c.id, pool[i % len(pool)])
    return out


def write_dialogue(llm: LLMClient, sb: Storyboard, speakers: list[str]) -> list[str]:
    """Write one spoken line per panel — a coherent back-and-forth conversation.

    `speakers[i]` is the name of the character speaking panel i. Returns a list of
    lines aligned to panels (short, natural, ~1 sentence each).
    """
    schema = {"type": "object",
              "properties": {"lines": {"type": "array", "items": {"type": "string"}}},
              "required": ["lines"]}
    roster = "\n".join(f"Panel {i+1}: {name} speaks" for i, name in enumerate(speakers))
    msgs = [
        {"role": "system",
         "content": ("You are a screenwriter. Write natural, casual spoken dialogue for a "
                     "short scene — ONE line per panel, in order, forming a believable "
                     "back-and-forth conversation. Each line is what that character SAYS out "
                     "loud (no names, no quotes, no stage directions), one short sentence, "
                     "conversational and warm.")},
        {"role": "user",
         "content": (f"Scene: {sb.logline}\nSetting/style: {sb.style_notes}\n\n{roster}\n\n"
                     f"Write exactly {len(speakers)} lines, one per panel, in order.")},
    ]
    data = llm.chat_json(msgs, schema, options={"temperature": 0.8})
    lines = [str(x) for x in data.get("lines", [])]
    # pad/truncate to match panel count
    while len(lines) < len(speakers):
        lines.append("...")
    return lines[:len(speakers)]


def _carry_renders(old: Storyboard, new: Storyboard) -> None:
    """Preserve generated refs/stills across a refine when the thing is unchanged."""
    old_cards = {c.name.strip().lower(): c for c in old.assets}
    for c in new.assets:
        prev = old_cards.get(c.name.strip().lower())
        if prev and prev.ref and prev.prompt == c.prompt:
            c.ref, c.source, c.seed = prev.ref, prev.source, prev.seed
    # Panels: match by identical image_prompt (a safe "unchanged" signal).
    old_panels = {p.image_prompt: p for p in old.panels}
    for p in new.panels:
        prev = old_panels.get(p.image_prompt)
        if prev:
            p.still, p.video, p.source, p.seed = prev.still, prev.video, prev.source, prev.seed
