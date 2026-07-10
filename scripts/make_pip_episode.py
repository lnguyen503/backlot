"""PIP mini-episode — a multi-shot, identity-consistent, talking short with a
music bed AND ambient SFX. Exercises the full storyboard stack end to end:

    PIP reference (FLUX) --> the consistency anchor
    per-panel stills anchored to PIP via Kontext  --> SAME robot every shot
    per-panel dialogue --Kokoro--> InfiniteTalk    --> lip-synced VOICE
    assemble (audio-preserving)                    --> dialogue kept across shots
    ACE-Step x2                                    --> MUSIC bed + ambient SOUNDS
    ffmpeg 3-leg amix                              --> voice on top, bed ducked under

Idempotent: reuses any still/clip already on the board (so a re-run only renders
what's missing). OOM-safe: frees ComfyUI VRAM before each heavy InfiniteTalk clip.

Run from the MAIN venv:  .venv\\Scripts\\python.exe tests\\make_pip_episode.py
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
from backlot.engine.runtime import Engine                # noqa: E402
from backlot.storyboard import render                    # noqa: E402
from backlot.storyboard.models import AssetCard, Panel, Storyboard  # noqa: E402
from backlot.storyboard.store import StoryboardStore     # noqa: E402

CFG = str(Path(__file__).resolve().parents[1] / "config/engine.yaml")
BOARD_ID = "sb_37cb5526"                 # the PIP board built live (fallback: fresh)
VOICE = "am_adam"

PIP_PROMPT = ("head-and-shoulders portrait of a friendly small gardener robot named PIP, "
              "rounded cream-and-green metal head, big glowing blue camera eyes, tiny leaf "
              "antenna, warm expressive face, looking at camera, plain soft background")

# Each shot: a still prompt (Kontext-anchored to PIP) + the line PIP speaks.
SHOTS = [
    dict(scene="PIP greets the viewer",
         image_prompt="PIP the gardener robot smiling warmly and speaking to camera, rooftop "
                      "garden with tomato plants softly blurred behind",
         dialogue="Welcome to my rooftop garden! The tomatoes are finally ripe.",
         mood="cheerful"),
    dict(scene="PIP tends the sprouts",
         image_prompt="PIP the gardener robot looking down fondly at small green sprouts, "
                      "holding a tiny watering can, rooftop garden beds, soft morning light",
         dialogue="I check on my little sprouts every morning. A splash of water, a bit of sunshine.",
         mood="tender"),
    dict(scene="PIP says goodbye",
         image_prompt="PIP the gardener robot, warm head-and-shoulders close-up, smiling and "
                      "nodding at the camera, rooftop garden at golden hour softly blurred "
                      "behind, warm sky",
         dialogue="Come back soon! There is always something growing up here.",
         mood="warm"),
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _ensure_board(store: StoryboardStore) -> tuple[Storyboard, AssetCard]:
    sb = store.get(BOARD_ID)
    if sb is None:
        sb = Storyboard(id=BOARD_ID, title="PIP — Rooftop Garden")
    sb.aspect = "portrait"
    sb.style_notes = sb.style_notes or "warm cinematic lighting, shallow depth of field, photorealistic"
    pip = next((c for c in sb.assets if c.bucket == "character"), None)
    if pip is None:
        pip = AssetCard(bucket="character", name="PIP",
                        description="a friendly little gardener robot",
                        prompt=PIP_PROMPT, voice=VOICE)
        sb.assets.append(pip)
    while len(sb.panels) < len(SHOTS):
        sb.panels.append(Panel())
    for p, shot in zip(sb.panels, SHOTS):
        if p.video is None:                       # don't disturb an already-rendered shot
            p.scene, p.image_prompt = shot["scene"], shot["image_prompt"]
            p.dialogue, p.mood, p.shot = shot["dialogue"], shot["mood"], "close-up"
            p.asset_ids = [pip.id]
    store.save(sb)
    return sb, pip


async def _acestep(eng: Engine, tags: str, seconds: float) -> Path:
    res = await eng.jobs.run_workflow(
        "music_acestep", {"tags": tags, "lyrics": "", "seconds": round(seconds + 1, 1), "steps": 50},
        wait=True, timeout_s=eng.cfg.timeouts.video_job_s)
    if res["state"] != "completed" or not res["outputs"]:
        raise RuntimeError(f"acestep failed: {res.get('error')}")
    o = res["outputs"][0]
    return Path(eng.cfg.comfyui.output_dir) / o.get("subfolder", "") / o["filename"]


async def run(args) -> None:
    cfg = load_config(CFG)
    store = StoryboardStore(cfg.paths.runs)
    sb, pip = _ensure_board(store)
    log(f"'{sb.title}' — {len(sb.panels)} shots, anchor: {pip.name}")

    eng = Engine(); eng.ensure_started(); await asyncio.sleep(0.3)

    if pip.ref is None:
        log("PIP reference (FLUX)"); await render.render_card_ref(eng, sb, pip); store.save(sb)

    # 1. Per-shot stills, each Kontext-anchored to PIP -> the SAME robot every shot.
    for i, p in enumerate(sb.panels, 1):
        if p.still is None:
            log(f"still {i}/{len(sb.panels)} (Kontext-anchored to PIP)")
            await render.render_panel_still(eng, sb, p); store.save(sb)

    # 2. Talk (InfiniteTalk) per shot — free VRAM before each (OOM-safe).
    for i, p in enumerate(sb.panels, 1):
        if p.video is None:
            render.free_vram(eng); await asyncio.sleep(2)
            log(f"talk {i}/{len(sb.panels)}: PIP ({VOICE}, InfiniteTalk, ~3-4 min)")
            await render.animate_talk(eng, sb, p, VOICE, tts="kokoro"); store.save(sb)
            log(f"  -> {p.video.filename}")

    # 3. Assemble — audio-preserving concat; crossfade smooths shot-to-shot cuts.
    log(f"assembling shots (dialogue preserved, {args.crossfade}s crossfade)")
    render.assemble(eng, sb, fps=24, crossfade=args.crossfade); store.save(sb)
    dur = render._video_seconds(render._out_abs(eng, sb.assembled))
    log(f"  sequence: {sb.assembled.filename} ({dur:.1f}s)")

    # 4. MUSIC bed + ambient SOUNDS (two ACE-Step tracks).
    render.free_vram(eng); await asyncio.sleep(2)
    log("music bed (ACE-Step)")
    music = await _acestep(eng, "whimsical cheerful ukulele and marimba, light playful, "
                                "gentle children's cartoon theme, warm", dur)
    log("ambient garden sounds (ACE-Step)")
    sfx = await _acestep(eng, "gentle outdoor garden ambience, soft birdsong, light warm breeze, "
                              "peaceful nature background, quiet", dur)

    # 5. Final 3-leg mix: dialogue on top, music + ambience ducked under.
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    video = render._out_abs(eng, sb.assembled)
    out = Path(args.out)
    flt = ("[0:a]volume=1.0[a0];[1:a]volume=0.26[a1];[2:a]volume=0.14[a2];"
           "[a0][a1][a2]amix=inputs=3:duration=first:normalize=0[aout]")
    subprocess.run([ff, "-y", "-i", str(video), "-i", str(music), "-i", str(sfx),
                    "-filter_complex", flt, "-map", "0:v", "-map", "[aout]",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)],
                   check=True, capture_output=True)
    log(f"DONE -> {out}  (voice + sounds + music; PIP consistent across {len(sb.panels)} shots)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs/pip_episode.mp4"))
    ap.add_argument("--crossfade", type=float, default=0.4, help="shot-to-shot dissolve, seconds")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
