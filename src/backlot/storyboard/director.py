"""The Director — autonomous brief → finished multi-shot video.

Given a one-line premise + basic guidance, the Director plans a board with the local
LLM, then executes the whole Studio pipeline (character refs for consistency →
per-panel stills → animate → assemble → LLM-scored music, optional narration) into a
finished video. It's the autonomous orchestrator over the Studio toolset: every step
reuses the same engine functions the UI/MCP surfaces call, so what the Director makes,
a human can keep editing in the Studio (the board is saved).
"""
from __future__ import annotations

from typing import Callable, Optional

from ..engine.llm import LLMClient
from ..engine.runtime import Engine
from . import render
from .agent import StoryboardAgent, suggest_score, write_narration
from .models import Storyboard
from .store import StoryboardStore

# Narration voice reference: None -> render.narrate synthesizes a privacy-safe
# Kokoro reference and clones it expressively (same default as the API).
_NARR_REF = None


def _plan(eng: Engine, premise: str, guidance: str, max_shots: int, log: Callable) -> Storyboard:
    """LLM-plan a board from the premise, TARGETING ~max_shots shots, then cap + persist.
    max_shots is a story-length target (the planner aims for it), not just a hard cap."""
    agent = StoryboardAgent(LLMClient.from_config(eng.cfg))
    hint = f"Plan {max_shots} distinct shots to tell the whole story (beginning to end)."
    sb = agent.draft(premise.strip(), guidance=f"{guidance} {hint}".strip())
    if len(sb.panels) > max_shots:
        sb.panels = sb.panels[:max_shots]
    StoryboardStore(eng.cfg.paths.runs).save(sb)
    log(f"planned '{sb.title}': {len(sb.assets)} anchors, {len(sb.panels)} shots")
    return sb


async def _render_visuals(eng: Engine, sb: Storyboard, i2v: str, log: Callable) -> None:
    """Character refs → per-panel stills → motion clips. **Checkpoints** the board after
    every step so an interruption preserves progress, and **skips** anything already
    rendered so a re-run resumes where it left off."""
    store = StoryboardStore(eng.cfg.paths.runs)
    for c in sb.assets:
        if c.bucket == "character" and c.ref is None:
            render.free_vram(eng)
            await render.render_card_ref(eng, sb, c)
            store.save(sb)
            log(f"ref: {c.name}")
    for i, p in enumerate(sb.panels, 1):
        if p.still is None:
            await render.render_panel_still(eng, sb, p)
            store.save(sb)
        log(f"still {i}/{len(sb.panels)}")
    for i, p in enumerate(sb.panels, 1):
        if p.video is None:
            render.free_vram(eng)                   # i2v is heavy — free VRAM per clip (OOM-safe)
            await render.animate_panel(eng, sb, p, backend=i2v)
            store.save(sb)
        log(f"clip {i}/{len(sb.panels)}")


async def _finish(eng: Engine, sb: Storyboard, music: bool, narrate: bool, log: Callable) -> None:
    """Assemble the clips, then add LLM-scored music (+ optional narration), muxed."""
    render.assemble(eng, sb, fps=24)
    log(f"assembled {len(sb.panels)} shots")
    if narrate:
        secs = render._video_seconds(render._out_abs(eng, sb.assembled))
        vo = write_narration(LLMClient.from_config(eng.cfg), sb, seconds=secs)
        render.narrate(eng, sb, vo, _NARR_REF)
        log("narrated")
    if music:
        brief = suggest_score(LLMClient.from_config(eng.cfg), sb)
        await render.score(eng, sb, brief["tags"], brief.get("lyrics", ""))
        log(f"scored: {brief['tags'][:48]}")
    if sb.score or sb.narration:
        render.mux_audio(eng, sb)
    StoryboardStore(eng.cfg.paths.runs).save(sb)


async def direct(eng: Engine, premise: str = "", *, board_id: Optional[str] = None,
                 guidance: str = "", i2v: str = "wan14b_fast", music: bool = True,
                 narrate: bool = False, max_shots: int = 4,
                 log: Optional[Callable] = None) -> Storyboard:
    """Plan and produce a finished multi-shot video from a premise. Returns the board
    (with `.assembled` = the finished video). `log(msg)` receives progress lines.
    Pass `board_id` to RESUME an existing board (skips planning; renders only what's
    missing) — e.g. after an interrupted long render."""
    log = log or (lambda _m: None)
    eng.ensure_started()
    if board_id:
        sb = StoryboardStore(eng.cfg.paths.runs).get(board_id)
        if sb is None:
            raise RuntimeError(f"unknown board to resume: {board_id}")
        log(f"resuming '{sb.title}': {sum(1 for p in sb.panels if p.video)}/{len(sb.panels)} clips done")
    else:
        log(f"directing: {premise!r}")
        sb = _plan(eng, premise, guidance, max_shots, log)
    await _render_visuals(eng, sb, i2v, log)
    await _finish(eng, sb, music, narrate, log)
    log(f"done -> {sb.assembled.filename if sb.assembled else 'no video'}")
    return sb
