"""Coffee-shop dialogue scene -> 4 animated, talking, identity-locked panels.

A storyboard "talking scene" test of the full stack:
    idea --LLM--> 4 friends (character cards) + cafe + style
         (deterministic shot/reverse-shot: panel i = close-up of friend i speaking)
    --LLM--> a 4-line conversation
    --FLUX--> character reference portraits (consistency anchors)
    --Kontext--> per-panel close-up stills, anchored to each speaker's ref
    --Chatterbox--> each line in that friend's cloned voice
    --InfiniteTalk--> lip-synced talking clip per panel
    --insightface/GFPGAN--> face re-lock onto every frame (identity nailed)
    --ffmpeg--> per-clip A/V mux + concat -> the scene

Run from the MAIN venv: .venv\\Scripts\\python.exe tests\\make_talk_scene.py
Free ComfyUI VRAM first (InfiniteTalk is heavy):
    curl -X POST :8188/free -d '{"unload_models":true,"free_memory":true}'
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.engine.config import load_config            # noqa: E402
from backlot.engine.llm import LLMClient                 # noqa: E402
from backlot.engine.runtime import Engine                # noqa: E402
from backlot.storyboard import render                    # noqa: E402
from backlot.storyboard.agent import (                  # noqa: E402
    StoryboardAgent, cast_voices, write_dialogue)
from backlot.storyboard.models import Panel              # noqa: E402
from backlot.storyboard.store import StoryboardStore     # noqa: E402

CFG = str(Path(__file__).resolve().parents[1] / "config/engine.yaml")
IDEA = ("Two couples — four old friends — catching up at a cozy neighborhood coffee shop on a "
        "rainy afternoon, warm window light, steam off the mugs. Photorealistic, candid, "
        "cinematic. Four distinct named friends (two couples): give each a clear look.")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs/coffee_scene.mp4"))
    ap.add_argument("--panels", type=int, default=4)
    # Re-lock is OFF by default: the insightface swap softens/closes the mouth and fights
    # InfiniteTalk's lip motion (the raw clips are crisper). Opt in only if identity drifts.
    ap.add_argument("--relock", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args))


async def run(args) -> None:
    cfg = load_config(CFG)
    llm = LLMClient.from_config(cfg)
    store = StoryboardStore(cfg.paths.runs)

    log("drafting cast + cafe + style")
    sb = StoryboardAgent(llm).draft(IDEA)
    sb.aspect = "portrait"
    chars = [c for c in sb.assets if c.bucket == "character"]
    env = next((c for c in sb.assets if c.bucket == "environment"), None)
    if not chars:
        log("no character cards produced; aborting"); return
    log(f"'{sb.title}' — cast: {[c.name for c in chars]}")

    # Deterministic shot/reverse-shot panels: each a close-up of one speaker.
    n = args.panels
    speakers = [chars[i % len(chars)] for i in range(n)]
    sb.panels = []
    for i, sp in enumerate(speakers):
        ids = [sp.id] + ([env.id] if env else [])
        sb.panels.append(Panel(
            scene=f"{sp.name} speaks, leaning in across the little table",
            image_prompt=(f"Tight warm close-up portrait of {sp.name} ({sp.description}), "
                          f"sitting at a small coffee-shop table holding a mug, looking at a "
                          f"friend across the table, mid-conversation, candid, rainy window "
                          f"light behind. {sb.style_notes}"),
            shot="close-up", camera="static", mood="warm, candid",
            asset_ids=ids))

    log("writing the conversation")
    lines = write_dialogue(llm, sb, [s.name for s in speakers])
    for p, line in zip(sb.panels, lines):
        p.dialogue = line
    for i, (s, l) in enumerate(zip(speakers, lines), 1):
        log(f"  {i}. {s.name}: {l}")
    store.save(sb)

    eng = Engine(); eng.ensure_started(); await asyncio.sleep(0.3)

    # 1. Character reference portraits (the consistency anchors).
    for c in chars:
        log(f"ref: {c.name}")
        try:
            await render.render_card_ref(eng, sb, c); store.save(sb)
        except Exception as ex:
            log(f"  ! ref failed: {ex}")

    # 2. Per-panel close-up stills, anchored to the speaker's ref via Kontext.
    for i, p in enumerate(sb.panels, 1):
        log(f"still {i}/{n} (close-up of {speakers[i-1].name})")
        try:
            await render.render_panel_still(eng, sb, p); store.save(sb)
        except Exception as ex:
            log(f"  ! still failed: {ex}")

    voice_of = cast_voices(llm, chars)   # LLM casts a Kokoro voice per character
    for c in chars:
        log(f"  voice: {c.name} -> {voice_of[c.id]}")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    out_dir = Path(cfg.comfyui.output_dir) / "backlot-storyboards"

    # 3. Talk (InfiniteTalk), one clip at a time. Free VRAM BEFORE each — the Wan
    #    model isn't released between runs, so the 2nd heavy run otherwise OOMs.
    for i, p in enumerate(sb.panels, 1):
        if not p.still:
            log(f"panel {i}: no still, skip"); continue
        sp = speakers[i - 1]
        render.free_vram(eng); await asyncio.sleep(2)
        log(f"talk {i}/{n}: {sp.name} ({voice_of[sp.id]}, InfiniteTalk, ~3-4 min)")
        try:
            await render.animate_talk(eng, sb, p, voice_of[sp.id], tts="kokoro"); store.save(sb)
            log(f"  -> {p.video.filename}")
        except Exception as ex:
            log(f"  ! talk failed: {ex}")

    # 4. Face re-lock (OPT-IN via --relock) — softens the mouth, so off by default.
    #    AFTER freeing the Wan model, so insightface doesn't co-reside with InfiniteTalk (OOM).
    if args.relock:
        render.free_vram(eng); await asyncio.sleep(2)
        for i, p in enumerate(sb.panels, 1):
            if p.video is None:
                continue
            log(f"face re-lock {i}/{n}")
            try:
                render.face_relock(eng, sb, p); store.save(sb)
            except Exception as ex:
                log(f"  ! relock failed: {ex}")

    # 5. Per-clip A/V mux (clip length follows its line -> exact sync).
    clips: list[Path] = []
    for i, p in enumerate(sb.panels, 1):
        if not p.video or not p.line_audio:
            continue
        clip_av = out_dir / f"{sb.id}_{p.id}_av.mp4"
        subprocess.run([ff, "-y", "-i", str(render._out_abs(eng, p.video)),
                        "-i", str(render._out_abs(eng, p.line_audio)),
                        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(clip_av)],
                       check=True, capture_output=True)
        clips.append(clip_av)
        log(f"panel {i} av -> {clip_av.name}")

    if not clips:
        log("no talking clips produced"); await eng.client.aclose(); return

    # 6. Concat the per-panel A/V clips into the scene.
    listf = out_dir / f"{sb.id}_concat.txt"
    listf.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips), encoding="utf-8")
    final = out_dir / f"{sb.id}_coffee_scene.mp4"
    subprocess.run([ff, "-y", "-f", "concat", "-safe", "0", "-i", str(listf),
                    "-c:v", "libx264", "-c:a", "aac", str(final)],
                   check=True, capture_output=True)
    sb.assembled = render.Asset(type="video", filename=final.name,
                                subfolder="backlot-storyboards",
                                url=eng.client.view_url(final.name, "backlot-storyboards"))
    store.save(sb)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import shutil; shutil.copy(final, args.out)
    log(f"SCENE -> {args.out}  (board {sb.id}, {len(clips)} talking panels)")
    await eng.client.aclose()


if __name__ == "__main__":
    main()
