"""Consistent-character video pipeline: ONE identity across many shots.

Graduates the v1->v4 R&D into a reusable, parameterized capability:

    reference image
        --FLUX Kontext--> face-locked full-body master
        --FLUX Kontext (per beat, anchored to master, no chaining)--> start frames
        --Wan 2.2 i2v (per beat)--> 5s shots
        --insightface swap + GFPGAN restore (per frame)--> identity re-locked
        --ffmpeg stitch--> one consistent clip

Identity is anchored to the reference at EVERY stage (never chained), so drift
doesn't compound. See PROGRESS.md "Multi-shot character consistency".
"""
from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageSequence

from ..engine.runtime import Engine
from ..faceswap import _DEFAULT_MODELS_DIR, FaceLocker, default_models

KEEP = ("Keep his face, glasses, grey hair and grey beard exactly the same. He wears a plain "
        "white shirt with no badge, no emblem, no flower, no logo, no pen and no chest pocket. "
        "Correct human anatomy: exactly two arms and two hands, no extra limbs, no extra hand on "
        "his chest or body, and his hands are not touching his face unless the scene says so. His "
        "head is a normal, natural size in correct proportion to his whole body (not enlarged), "
        "with a natural torso and legs. Photorealistic, cinematic, natural light.")


@dataclass
class Beat:
    scene: str   # Kontext instruction -> the start frame's pose/setting
    motion: str  # Wan prompt -> the action inside the shot


def _abs(cfg, asset) -> Path:
    return Path(cfg.comfyui.output_dir) / asset.get("subfolder", "") / asset["filename"]


def _stage(cfg, asset, name: str) -> str:
    shutil.copy(_abs(cfg, asset), Path(cfg.comfyui.input_dir) / name)
    return name


async def _kontext(eng: Engine, image: str, instruction: str, seed: int):
    return await eng.jobs.run_workflow(
        "edit_kontext",
        {"image": image, "instruction": f"{instruction} {KEEP}", "guidance": 2.5, "steps": 20, "seed": seed},
        wait=True, timeout_s=300,
    )


# i2v backend params. Wan 5B = 24fps landscape; Wan 14B = portrait (head stays in
# frame) with better anatomy. Portrait matches our full-body keyframes.
I2V_BACKENDS = {
    "wan5b": ("img2vid_wan", {"length": 121, "steps": 30}),
    "wan14b": ("img2vid_wan14b", {"width": 720, "height": 1280, "length": 81}),
    "wan14b_fast": ("img2vid_wan14b_fast", {"width": 720, "height": 1280, "length": 81}),
}


async def render_shots(eng: Engine, ref_filename: str, master_scene: str, beats: list[Beat],
                       i2v: str = "wan14b"):
    """Produce the per-shot Wan clips (face-locked keyframes -> motion). Returns clip paths."""
    cfg = eng.cfg
    workflow, base_params = I2V_BACKENDS[i2v]
    res = await _kontext(eng, ref_filename, master_scene, seed=42)
    if res["state"] != "completed" or not res["outputs"]:
        raise RuntimeError(f"master keyframe failed: {res.get('error')}")
    master = _stage(cfg, res["outputs"][0], "cv_master.png")

    clips = []
    for i, beat in enumerate(beats, 1):
        ek = await _kontext(eng, master, beat.scene, seed=100 + i)
        if ek["state"] != "completed" or not ek["outputs"]:
            print(f"[shot {i}] keyframe failed: {ek.get('error')}", flush=True)
            continue
        start = _stage(cfg, ek["outputs"][0], f"cv_start_{i}.png")
        vid = await eng.jobs.run_workflow(
            workflow,
            {"image": start, "positive_prompt": beat.motion, "seed": 200 + i, **base_params},
            wait=True, timeout_s=1800,
        )
        if vid["state"] == "completed" and vid["outputs"]:
            clips.append(_abs(cfg, vid["outputs"][0]))
            print(f"[shot {i}/{len(beats)}] ok -> {clips[-1].name}", flush=True)
        else:
            print(f"[shot {i}] wan failed: {vid.get('error')}", flush=True)
    return clips


def lock_and_stitch(clips: list[Path], locker: FaceLocker, out_path: Path, fps: int = 24) -> dict:
    """Swap+restore the locked identity onto every frame of every clip, then stitch."""
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8, macro_block_size=1)
    total, swapped = 0, 0
    for ci, clip in enumerate(clips, 1):
        n = 0
        for frame in ImageSequence.Iterator(Image.open(clip)):
            bgr = np.array(frame.convert("RGB"))[:, :, ::-1].copy()
            bgr, had = locker.process_frame(bgr)
            swapped += int(had)
            writer.append_data(bgr[:, :, ::-1])
            total += 1
            n += 1
        print(f"[lock {ci}/{len(clips)}] {clip.name}: {n} frames", flush=True)
    writer.close()
    return {"frames": total, "face_swapped": swapped, "duration_s": round(total / fps, 1), "out": str(out_path)}


async def make_consistent_video(ref_filename: str, master_scene: str, beats: list[Beat],
                                out_path: str, models_dir: str = _DEFAULT_MODELS_DIR,
                                i2v: str = "wan14b") -> dict:
    """End-to-end: reference + beats -> one identity-locked stitched clip."""
    eng = Engine()
    eng.ensure_started()
    await asyncio.sleep(0.3)
    clips = await render_shots(eng, ref_filename, master_scene, beats, i2v=i2v)
    if not clips:
        raise RuntimeError("no shots rendered")
    inswapper, gfpgan = default_models(models_dir)
    locker = FaceLocker(str(Path(eng.cfg.comfyui.input_dir) / ref_filename), inswapper, gfpgan)
    result = lock_and_stitch(clips, locker, Path(out_path))
    await eng.client.aclose()
    return result


# A man's evening routine — the demo narrative (override via your own Beat list).
DEMO_BEATS = [
    Beat("Show this exact same man standing just inside the front door of his home, full body.",
         "the man steps forward into his home, walking inside, smooth cinematic tracking shot"),
    Beat("Show this exact same man hanging his house keys on a wall hook by the door, full body.",
         "the man hangs his keys on the wall hook, then lowers his hand"),
    Beat("Show this exact same man walking down the hallway toward his kitchen, front view, full body.",
         "the man walks down the hallway toward his kitchen, the camera follows"),
    Beat("Show this exact same man filling a clear glass with water at his kitchen tap, full body.",
         "the man fills the glass with water from the tap"),
    Beat("Show this exact same man drinking a glass of water in his kitchen, full body.",
         "the man raises the glass and drinks the water, then lowers it"),
    Beat("Show this exact same man walking into his cozy living room, full body.",
         "the man walks into his living room, smooth cinematic camera"),
    Beat("Show this exact same man sitting down on his living room couch, full body.",
         "the man sits down on the couch, settling in, relaxed"),
    Beat("Show this exact same man on the couch facing a glowing television, over-the-shoulder.",
         "the man relaxes on the couch watching the glowing television in the evening"),
]
DEMO_MASTER = ("Full-length full-body photograph of this exact same man standing naturally just "
               "inside the front door of a cozy warm modern home with hardwood floors, his arms "
               "relaxed at his sides holding house keys in one hand, framed from head to feet. "
               "Shot from across the room with a 35mm lens.")


def _cli():
    import argparse

    ap = argparse.ArgumentParser(description="Render a consistent-character video from a reference image.")
    ap.add_argument("--ref", required=True, help="Reference image filename (in ComfyUI input/).")
    ap.add_argument("--out", required=True, help="Output mp4 path.")
    ap.add_argument("--models-dir", default=_DEFAULT_MODELS_DIR)
    ap.add_argument("--i2v", default="wan14b", choices=list(I2V_BACKENDS), help="Image-to-video backend.")
    args = ap.parse_args()
    res = asyncio.run(make_consistent_video(args.ref, DEMO_MASTER, DEMO_BEATS, args.out,
                                            args.models_dir, i2v=args.i2v))
    print(f"RESULT: {res}", flush=True)


if __name__ == "__main__":
    _cli()
