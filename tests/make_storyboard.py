"""Idea -> short video story, end-to-end through the storyboard engine.

Drives every stage the web UI drives, headlessly:
    idea --LLM--> board (assets + ordered panels)
         --FLUX--> character/asset reference stills (consistency anchors)
         --Kontext/FLUX--> per-panel frames-first stills (anchored to refs)
         --img2vid--> per-panel motion clips
         --ffmpeg--> one stitched sequence mp4

Usage:
    .venv\\Scripts\\python.exe tests\\make_storyboard.py \
        --idea "a lonely lighthouse keeper befriends a seagull, 3 panels" \
        --i2v svd --out runs/story.mp4
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.engine.config import load_config           # noqa: E402
from backlot.engine.llm import LLMClient                # noqa: E402
from backlot.engine.runtime import Engine               # noqa: E402
from backlot.storyboard import render                   # noqa: E402
from backlot.storyboard.agent import (                  # noqa: E402
    StoryboardAgent, suggest_score, write_narration)
from backlot.storyboard.store import StoryboardStore    # noqa: E402

CFG = str(Path(__file__).resolve().parents[1] / "config/engine.yaml")


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser(description="Idea -> short video story.")
    ap.add_argument("--idea", default="",
                    help="story idea (omit when --board reuses an existing board)")
    ap.add_argument("--board", default="",
                    help="reuse an existing board id: skip draft/refs/stills, just (re)animate")
    ap.add_argument("--i2v", default="svd", choices=["svd", "wan14b_fast"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs/storyboard_story.mp4"))
    ap.add_argument("--no-video", action="store_true",
                    help="stop after stills (frames-first only)")
    ap.add_argument("--music", action="store_true",
                    help="generate an LLM-scored soundtrack and mux it onto the sequence")
    ap.add_argument("--music-tags", default="",
                    help="override the soundtrack style (else the LLM proposes one)")
    ap.add_argument("--narrate", action="store_true",
                    help="generate an LLM voiceover (TTS) and mix it over the music")
    ap.add_argument("--voice-ref", default=None,
                    help="reference voice WAV to clone for narration (required with --narrate)")
    args = ap.parse_args()

    cfg = load_config(CFG)
    agent = StoryboardAgent(LLMClient.from_config(cfg))
    store = StoryboardStore(cfg.paths.runs)

    if args.board:
        sb = store.get(args.board)
        if sb is None:
            _log(f"unknown board id: {args.board}")
            return
        _log(f"reusing board '{sb.title}' ({len(sb.panels)} panels) — re-animating with {args.i2v}")
    else:
        if not args.idea:
            _log("provide --idea or --board")
            return
        _log(f"drafting board from idea: {args.idea!r}")
        t0 = time.time()
        sb = agent.draft(args.idea)
        store.save(sb)
        _log(f"  -> '{sb.title}' | {len(sb.assets)} assets, {len(sb.panels)} panels "
             f"({time.time()-t0:.0f}s)")

    eng = Engine()
    eng.ensure_started()
    await asyncio.sleep(0.3)

    if not args.board:
        # 1. Reference stills for character cards (the consistency anchors).
        for c in sb.assets:
            if c.bucket == "character":
                _log(f"ref: character '{c.name}'")
                try:
                    await render.render_card_ref(eng, sb, c)
                    store.save(sb)
                except Exception as ex:
                    _log(f"  ! ref failed: {ex}")

        # 2. Frames-first still per panel (anchored to a character ref when present).
        for i, p in enumerate(sb.panels, 1):
            anchor = sb.primary_character(p)
            tag = f"anchored->{anchor.name}" if anchor else "txt2img"
            _log(f"still {i}/{len(sb.panels)} ({p.shot}) [{tag}]")
            try:
                await render.render_panel_still(eng, sb, p)
                store.save(sb)
            except Exception as ex:
                _log(f"  ! still failed: {ex}")

        stills = sum(1 for p in sb.panels if p.still)
        _log(f"stills done: {stills}/{len(sb.panels)}")

        if args.no_video:
            await eng.client.aclose()
            _log("stopping after stills (--no-video)")
            return

    # 3. Animate each panel that has a still.
    for i, p in enumerate(sb.panels, 1):
        if not p.still:
            continue
        _log(f"animate {i}/{len(sb.panels)} via {args.i2v}")
        try:
            await render.animate_panel(eng, sb, p, backend=args.i2v)
            store.save(sb)
        except Exception as ex:
            _log(f"  ! animate failed: {ex}")

    clips = sum(1 for p in sb.panels if p.video)
    _log(f"clips done: {clips}/{len(sb.panels)}")

    # 4. Assemble.
    if not clips:
        _log("no clips; nothing to assemble")
        await eng.client.aclose()
        return
    render.assemble(eng, sb)
    store.save(sb)
    seq_s = render._video_seconds(render._out_abs(eng, sb.assembled))
    _log(f"assembled silent sequence ({len(sb.panels)} panels, {seq_s:.1f}s)")

    llm = LLMClient.from_config(cfg)

    # 5a. Voiceover (TTS) — written to fit the sequence, cloned from --voice-ref.
    if args.narrate:
        vo = write_narration(llm, sb, seconds=seq_s)
        _log(f"narration: {vo!r}")
        try:
            render.narrate(eng, sb, vo, args.voice_ref)
            _log("voiceover generated")
        except Exception as ex:
            _log(f"  ! narration failed: {ex}")

    # 5b. Music score (ducked under any voiceover at mux time).
    if args.music:
        brief = ({"tags": args.music_tags, "lyrics": ""} if args.music_tags
                 else suggest_score(llm, sb))
        _log(f"scoring: tags={brief['tags']!r} "
             f"lyrics={'yes' if brief['lyrics'] else 'instrumental'}")
        try:
            await render.score(eng, sb, brief["tags"], brief["lyrics"])
            _log("score generated")
        except Exception as ex:
            _log(f"  ! scoring failed: {ex}")

    # 5c. Mux whatever audio we produced (mix narration + ducked music).
    if sb.score or sb.narration:
        try:
            render.mux_audio(eng, sb)
            store.save(sb)
            _log("muxed audio onto sequence")
        except Exception as ex:
            _log(f"  ! mux failed: {ex}")

    src = render._out_abs(eng, sb.assembled)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, args.out)
    _log(f"FINAL -> {args.out}  (board id {sb.id})")

    await eng.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
