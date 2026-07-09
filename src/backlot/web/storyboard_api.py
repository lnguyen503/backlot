"""REST surface for the storyboard / scene planner.

Mounted by web.app. Long-running render endpoints block until the job finishes
(wait=True) and return the updated board with the new asset attached — the
frontend shows a per-item spinner meanwhile. Drafting/refine call the local LLM.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import BaseModel

from ..engine.llm import LLMClient
from ..engine.runtime import Engine
from ..storyboard import render
from ..storyboard.agent import (
    StoryboardAgent, assist_panel, suggest_ambient, suggest_score, write_narration)
from ..storyboard import templates as tpl
from ..storyboard.library import LibraryStore
from ..storyboard.models import AssetCard, Storyboard
from ..storyboard.store import StoryboardStore

router = APIRouter(prefix="/api/storyboards")
lib_router = APIRouter(prefix="/api/library")


class _SB:
    engine: Optional[Engine] = None
    store: Optional[StoryboardStore] = None
    agent: Optional[StoryboardAgent] = None
    library: Optional[LibraryStore] = None


sb_state = _SB()


def init_storyboard(engine: Engine) -> None:
    """Called from app lifespan once the engine is up."""
    sb_state.engine = engine
    sb_state.store = StoryboardStore(engine.cfg.paths.runs)
    sb_state.agent = StoryboardAgent(LLMClient.from_config(engine.cfg))
    sb_state.library = LibraryStore(engine.cfg.paths.runs)


def _eng() -> Engine:
    if sb_state.engine is None:
        raise HTTPException(503, "engine not ready")
    sb_state.engine.ensure_started()
    return sb_state.engine


def _load(sb_id: str) -> Storyboard:
    sb = sb_state.store.get(sb_id) if sb_state.store else None
    if sb is None:
        raise HTTPException(404, "unknown storyboard")
    return sb


class IdeaBody(BaseModel):
    idea: str
    template: str = ""


class BlankBody(BaseModel):
    title: str = ""
    template: str = ""


class ChatBody(BaseModel):
    message: str


class RenderBody(BaseModel):
    model: str = "txt2img_flux"


class AnimateBody(BaseModel):
    backend: str = "svd"          # svd | wan14b_fast | talk (InfiniteTalk lip-sync)
    voice: str = ""               # talk: Kokoro voice id (else the character's own / default)


class AssembleBody(BaseModel):
    fps: int = 24


class ScoreBody(BaseModel):
    tags: str = ""          # blank -> the LLM proposes a fitting soundtrack
    lyrics: str = ""


class NarrateBody(BaseModel):
    text: str = ""          # blank -> the LLM writes a voiceover fitting the board
    backend: str = "chatterbox"
    voice_ref: Optional[str] = None   # None -> synthetic Kokoro ref (privacy-safe)


class AmbientBody(BaseModel):
    tags: str = ""          # blank -> the LLM designs an ambient/SFX bed for the scene


class DirectBody(BaseModel):
    premise: str
    guidance: str = ""              # style / aspect / length hints
    i2v: str = "wan14b_fast"        # motion backend per shot
    music: bool = True
    narrate: bool = False
    max_shots: int = 4


@router.get("")
async def list_boards():
    return {"storyboards": sb_state.store.list() if sb_state.store else []}


@router.post("/direct")
async def direct(body: DirectBody):
    """The Director: plan now, then render→animate→score in the background (returns the board)."""
    import asyncio
    from ..storyboard import director
    if not body.premise.strip():
        raise HTTPException(400, "premise is required")
    eng = _eng()
    sb = director._plan(eng, body.premise, body.guidance, body.max_shots, lambda _m: None)

    async def _rest():
        try:
            await director._render_visuals(eng, sb, body.i2v, lambda _m: None)
            await director._finish(eng, sb, body.music, body.narrate, lambda _m: None)
        except Exception:
            pass

    asyncio.create_task(_rest())
    return sb.model_dump()


@router.get("/templates")
async def list_templates():
    """Use-case templates for the New-board picker."""
    return {"templates": tpl.list_templates()}


@router.post("")
async def create_board(body: IdeaBody):
    if not body.idea.strip():
        raise HTTPException(400, "idea is required")
    if sb_state.agent is None:
        raise HTTPException(503, "agent not ready")
    try:
        sb = sb_state.agent.draft(body.idea.strip(), guidance=tpl.guidance(body.template))
    except Exception as ex:
        raise HTTPException(502, f"draft failed: {ex}")
    sb_state.store.save(sb)
    return sb.model_dump()


@router.post("/blank")
async def create_blank(body: BlankBody):
    """Create a board — empty, or scaffolded from a use-case template."""
    if sb_state.store is None:
        raise HTTPException(503, "store not ready")
    sb = tpl.scaffold(body.template, body.title) if body.template else None
    if sb is None:
        sb = Storyboard(title=body.title.strip() or "Untitled Storyboard")
    sb_state.store.save(sb)
    return sb.model_dump()


@router.get("/{sb_id}")
async def get_board(sb_id: str):
    return _load(sb_id).model_dump()


@router.put("/{sb_id}")
async def replace_board(sb_id: str, board: Storyboard):
    """Save client-side edits (panel text, asset cards, order, title, ...)."""
    existing = _load(sb_id)
    board.id = existing.id
    board.created_at = existing.created_at
    sb_state.store.save(board)
    return board.model_dump()


@router.delete("/{sb_id}")
async def delete_board(sb_id: str):
    return {"deleted": sb_state.store.delete(sb_id)}


@router.post("/{sb_id}/chat")
async def chat(sb_id: str, body: ChatBody):
    sb = _load(sb_id)
    try:
        sb = sb_state.agent.refine(sb, body.message.strip())
    except Exception as ex:
        raise HTTPException(502, f"refine failed: {ex}")
    sb_state.store.save(sb)
    return sb.model_dump()


@router.post("/{sb_id}/assets/{card_id}/render")
async def render_card(sb_id: str, card_id: str, body: RenderBody):
    sb = _load(sb_id)
    card = sb.card(card_id)
    if card is None:
        raise HTTPException(404, "unknown card")
    try:
        await render.render_card_ref(_eng(), sb, card, model=body.model)
    except Exception as ex:
        raise HTTPException(500, f"render failed: {ex}")
    sb_state.store.save(sb)
    return sb.model_dump()


@router.post("/{sb_id}/panels/{panel_id}/assist")
async def assist_panel_ep(sb_id: str, panel_id: str):
    """Let the AI write/improve this panel's scene + prompt (the 'help me prompt' button)."""
    sb = _load(sb_id)
    panel = sb.panel(panel_id)
    if panel is None:
        raise HTTPException(404, "unknown panel")
    if sb_state.engine is None:
        raise HTTPException(503, "engine not ready")
    try:
        data = assist_panel(LLMClient.from_config(sb_state.engine.cfg), sb, panel)
    except Exception as ex:
        raise HTTPException(502, f"assist failed: {ex}")
    for k in ("scene", "image_prompt", "shot", "camera", "mood", "motion_prompt"):
        v = (data.get(k) or "").strip()
        if v:
            setattr(panel, k, v)
    sb_state.store.save(sb)
    return sb.model_dump()


@router.post("/{sb_id}/panels/{panel_id}/render")
async def render_panel(sb_id: str, panel_id: str, body: RenderBody):
    sb = _load(sb_id)
    panel = sb.panel(panel_id)
    if panel is None:
        raise HTTPException(404, "unknown panel")
    try:
        await render.render_panel_still(_eng(), sb, panel, model=body.model)
    except Exception as ex:
        raise HTTPException(500, f"render failed: {ex}")
    sb_state.store.save(sb)
    return sb.model_dump()


@router.post("/{sb_id}/panels/{panel_id}/photo")
async def panel_photo(sb_id: str, panel_id: str, file: UploadFile):
    sb = _load(sb_id)
    panel = sb.panel(panel_id)
    if panel is None:
        raise HTTPException(404, "unknown panel")
    ext = (file.filename or "x.png").rsplit(".", 1)[-1].lower()
    data = await file.read()
    render.set_panel_photo(_eng(), panel, data, ext=ext)
    sb_state.store.save(sb)
    return sb.model_dump()


@router.post("/{sb_id}/panels/{panel_id}/animate")
async def animate(sb_id: str, panel_id: str, body: AnimateBody):
    sb = _load(sb_id)
    panel = sb.panel(panel_id)
    if panel is None:
        raise HTTPException(404, "unknown panel")
    eng = _eng()
    try:
        if body.backend == "talk":
            if not panel.dialogue.strip():
                raise HTTPException(400, "panel has no dialogue to speak")
            char = sb.primary_character(panel)
            voice = body.voice or (char.voice if char else "") or "af_heart"
            render.free_vram(eng)     # InfiniteTalk is heavy — free VRAM first (OOM-safe)
            await render.animate_talk(eng, sb, panel, voice, tts="kokoro")
        else:
            await render.animate_panel(eng, sb, panel, backend=body.backend)
    except HTTPException:
        raise
    except Exception as ex:
        raise HTTPException(500, f"animate failed: {ex}")
    sb_state.store.save(sb)
    return sb.model_dump()


@router.post("/{sb_id}/assemble")
async def assemble(sb_id: str, body: AssembleBody):
    sb = _load(sb_id)
    try:
        render.assemble(_eng(), sb, fps=body.fps)
    except Exception as ex:
        raise HTTPException(500, f"assemble failed: {ex}")
    sb_state.store.save(sb)
    return sb.model_dump()


@router.post("/{sb_id}/score")
async def score(sb_id: str, body: ScoreBody):
    """Generate a soundtrack (LLM-proposed unless tags given) and mux it on."""
    sb = _load(sb_id)
    if sb.assembled is None:
        raise HTTPException(400, "assemble the sequence before scoring")
    eng = _eng()
    if body.tags.strip():
        brief = {"tags": body.tags.strip(), "lyrics": body.lyrics}
    else:
        brief = suggest_score(LLMClient.from_config(eng.cfg), sb)
    try:
        await render.score(eng, sb, brief["tags"], brief.get("lyrics", ""))
        render.mux_audio(eng, sb)
    except Exception as ex:
        raise HTTPException(500, f"score failed: {ex}")
    sb_state.store.save(sb)
    return sb.model_dump()


@router.post("/{sb_id}/ambient")
async def ambient(sb_id: str, body: AmbientBody):
    """Add a low ambient/SFX bed (LLM-designed unless tags given), mixed UNDER music+voice."""
    sb = _load(sb_id)
    if sb.assembled is None:
        raise HTTPException(400, "assemble the sequence before adding ambient")
    eng = _eng()
    tags = body.tags.strip() or suggest_ambient(LLMClient.from_config(eng.cfg), sb)
    try:
        await render.ambient(eng, sb, tags)
        render.mux_audio(eng, sb)
    except Exception as ex:
        raise HTTPException(500, f"ambient failed: {ex}")
    sb_state.store.save(sb)
    return sb.model_dump()


@router.post("/{sb_id}/narrate")
async def narrate(sb_id: str, body: NarrateBody):
    """Add an LLM voiceover (TTS) and mix it over any score (music ducked under it)."""
    sb = _load(sb_id)
    if sb.assembled is None:
        raise HTTPException(400, "assemble the sequence before narrating")
    eng = _eng()
    seconds = render._video_seconds(render._out_abs(eng, sb.assembled))
    text = body.text.strip() or write_narration(
        LLMClient.from_config(eng.cfg), sb, seconds=seconds)
    try:
        render.narrate(eng, sb, text, body.voice_ref, backend=body.backend)
        render.mux_audio(eng, sb)
    except Exception as ex:
        raise HTTPException(500, f"narrate failed: {ex}")
    sb_state.store.save(sb)
    return sb.model_dump()


# ----- Global Characters & Worlds library (reusable across boards) -------------

def _lib() -> LibraryStore:
    if sb_state.library is None:
        raise HTTPException(503, "library not ready")
    return sb_state.library


@lib_router.get("")
async def library_list(bucket: Optional[str] = None):
    return {"cards": _lib().list(bucket)}


@lib_router.post("")
async def library_save(card: AssetCard):
    """Upsert a reusable card (character / world / style / object) into the library."""
    _lib().save(card)
    return card.model_dump()


@lib_router.delete("/{card_id}")
async def library_delete(card_id: str):
    return {"deleted": _lib().delete(card_id)}


@router.post("/{sb_id}/assets/{card_id}/save-to-library")
async def save_card_to_library(sb_id: str, card_id: str):
    """Push one of a board's anchor cards into the global library (same id)."""
    sb = _load(sb_id)
    card = sb.card(card_id)
    if card is None:
        raise HTTPException(404, "unknown card")
    _lib().save(card)
    return {"saved": card.id, "name": card.name}


@router.post("/{sb_id}/library/import/{card_id}")
async def import_card_from_library(sb_id: str, card_id: str):
    """Copy a library card into this board's anchors (idempotent by id)."""
    sb = _load(sb_id)
    card = _lib().get(card_id)
    if card is None:
        raise HTTPException(404, "unknown library card")
    if sb.card(card.id) is None:
        sb.assets.append(card.model_copy(deep=True))
        sb_state.store.save(sb)
    return sb.model_dump()
