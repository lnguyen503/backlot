"""Produce a short 'vibe coding' music video end-to-end through the engine.

Order minimizes model swaps: all images (SDXL) -> all clips (SVD) -> song (ACE-Step),
then assemble frames + mux audio with the bundled ffmpeg. Each step is best-effort;
the final assembly uses whatever clips succeeded.
"""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageSequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.engine.comfy_client import ComfyClient  # noqa: E402
from backlot.engine.config import load_config  # noqa: E402
from backlot.engine.job_manager import JobManager  # noqa: E402
from backlot.engine.registry import Registry  # noqa: E402
from backlot.engine.ws_listener import WsListener  # noqa: E402

CFG = str(Path(__file__).resolve().parents[1] / "config/engine.yaml")
SONG_SECONDS = 24

SCENES = [
    "a programmer at night bathed in neon screen glow, multiple monitors full of colorful code, cozy dark room, synthwave aesthetic, cinematic, highly detailed",
    "glowing lines of code streaming through a dark digital space, neon blue and magenta, abstract data flow, depth, cinematic",
    "a sleek desk setup with a mechanical keyboard, small plants, warm rgb lighting, steam rising from coffee, night, bokeh, cinematic",
    "a silhouette of a developer facing a giant wall of glowing code, futuristic, neon, atmospheric haze, cinematic",
    "a retro-futuristic city skyline at night made of circuit boards and neon data streams, synthwave, cinematic",
    "sunrise over a desk after an all-nighter, soft warm light, code glowing on screen, empty coffee cups, hopeful mood, cinematic",
]
TAGS = "upbeat synthwave, retro electronic, male vocals, driving beat, energetic, nostalgic, 80s synths"
LYRICS = (
    "[verse]\n"
    "Neon screen in the dead of night\n"
    "Cursor blinking, the flow feels right\n"
    "Tell the model what I dream\n"
    "Watch it build the whole machine\n\n"
    "[chorus]\n"
    "Vibe coding, ride the flow\n"
    "Lines of light begin to glow\n"
    "Just a whisper, watch it grow\n"
    "Vibe coding, here we go\n"
)


def _abs(cfg, asset) -> Path:
    return Path(cfg.comfyui.output_dir) / asset.get("subfolder", "") / asset["filename"]


async def main():
    cfg = load_config(CFG)
    client = ComfyClient(cfg.comfyui.base_url, uuid.uuid4().hex, cfg.timeouts.http_s)
    jobs = JobManager(cfg, Registry.load(cfg), client)
    ws = WsListener(client, cfg.comfyui.ws_url, jobs.on_event, cfg.timeouts.ws_reconnect_max_s)
    jobs.set_ws(ws)
    ws.start()
    await asyncio.sleep(0.5)
    in_dir = Path(cfg.comfyui.input_dir)

    # Phase A: images (SDXL loaded once)
    staged: list[str] = []
    for i, prompt in enumerate(SCENES):
        res = await jobs.run_workflow(
            "txt2img_sdxl",
            {"positive_prompt": prompt, "width": 1024, "height": 576, "steps": 22},
            wait=True, timeout_s=300,
        )
        if res["state"] == "completed" and res["outputs"]:
            name = f"vibe_scene_{i}.png"
            shutil.copy(_abs(cfg, res["outputs"][0]), in_dir / name)
            staged.append(name)
            print(f"[image {i+1}/{len(SCENES)}] ok -> {name}", flush=True)
        else:
            print(f"[image {i+1}] FAILED: {res['state']}", flush=True)

    # Phase B: clips (SVD loaded once)
    clip_webps: list[Path] = []
    for i, name in enumerate(staged):
        res = await jobs.run_workflow(
            "img2vid_svd",
            {"image": name, "frames": 25, "motion_bucket_id": 110, "steps": 18},
            wait=True, timeout_s=900,
        )
        if res["state"] == "completed" and res["outputs"]:
            clip_webps.append(_abs(cfg, res["outputs"][0]))
            print(f"[clip {i+1}/{len(staged)}] ok", flush=True)
        else:
            print(f"[clip {i+1}] FAILED: {res['state']}", flush=True)

    # Phase C: song (ACE-Step loaded once)
    song = None
    res = await jobs.run_workflow(
        "music_acestep",
        {"tags": TAGS, "lyrics": LYRICS, "seconds": SONG_SECONDS, "steps": 50},
        wait=True, timeout_s=900,
    )
    if res["state"] == "completed" and res["outputs"]:
        song = _abs(cfg, res["outputs"][0])
        print("[song] ok", flush=True)
    else:
        print(f"[song] FAILED: {res['state']}", flush=True)

    await ws.stop()
    await client.aclose()

    # Phase D: assemble
    if not clip_webps:
        print("RESULT: FAIL (no clips)", flush=True)
        return
    frames: list[np.ndarray] = []
    for webp in clip_webps:
        for fr in ImageSequence.Iterator(Image.open(webp)):
            frames.append(np.array(fr.convert("RGB")))
    out_dir = Path(cfg.comfyui.output_dir) / "backlot"
    silent = out_dir / "_vibe_silent.mp4"
    fps = max(1.0, len(frames) / SONG_SECONDS)
    writer = imageio.get_writer(str(silent), fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)
    for f in frames:
        writer.append_data(f)
    writer.close()
    final = out_dir / "vibe_coding_music_video.mp4"
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    if song:
        subprocess.run([ff, "-y", "-i", str(silent), "-i", str(song),
                        "-c:v", "copy", "-c:a", "aac", "-shortest", str(final)], check=True)
    else:
        shutil.copy(silent, final)
    print(f"RESULT: PASS  clips={len(clip_webps)} frames={len(frames)} fps={fps:.2f}", flush=True)
    print(f"FINAL: {final}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
