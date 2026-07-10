"""SOTA path: temporally-consistent AI-stylized 3D via Wan VACE video-diffusion.

Blender renders a DEPTH SEQUENCE (real geometry, stable normalization); Wan 2.1
VACE generates the whole clip in ONE pass conditioned on that depth control video
-> coherent motion, no per-frame flicker (the fix for the SDXL+ControlNet baseline).

    .venv\\Scripts\\python.exe tests\\make_vace_animation.py --scene city_flythrough --prompt "..."
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

import subprocess

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageSequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.blender import runner as bl, scenes      # noqa: E402
from backlot.engine.runtime import Engine             # noqa: E402

# fitting default prompts per scene (same casting as the ControlNet runner)
SCENES = {
    "monkey_orbit": dict(near=3.0, far=11.0, prompt=(
        "a photorealistic carved white marble bust of a monkey on a pedestal, bright museum gallery "
        "lighting, soft shadows, intricate chiselled detail, sharp focus, high detail")),
    "sculpture_turntable": dict(near=3.2, far=9.5, prompt=(
        "a polished bronze and gold abstract sculpture, a ring around an orb, brightly lit studio, "
        "reflective burnished metal, rim light, museum backdrop, ultra detailed, sharp focus")),
    "city_flythrough": dict(near=2.0, far=18.0, prompt=(
        "a vibrant neon cyberpunk city street at night, glowing pink and cyan signs, bright neon glow, "
        "rain-soaked reflective street, blade runner, cinematic, ultra detailed, high contrast")),
    "product_turntable": dict(near=3.8, far=9.0, prompt=(
        "a luxury perfume bottle on a polished round pedestal, glossy reflective glass with amber liquid, "
        "metallic gold cap, elegant studio product photography, soft seamless gradient backdrop, "
        "dramatic rim lighting and soft shadows, commercial advertisement, photorealistic, ultra detailed, "
        "sharp focus, high resolution")),
}


def _log(m): print(m, flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="city_flythrough", choices=list(SCENES))
    ap.add_argument("--frames", type=int, default=33, help="4n+1 for Wan (33/49/81)")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--interp-fps", type=int, default=0,
                    help="if >0, motion-interpolate the result to this fps (ffmpeg minterpolate)")
    ap.add_argument("--model", default="wan2.1_vace_1.3B_fp16.safetensors",
                    help="VACE UNet (e.g. wan2.1_vace_14B_fp16.safetensors)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs/vace_animation.mp4"))
    args = ap.parse_args()
    spec = SCENES[args.scene]
    prompt = args.prompt or spec["prompt"]

    eng = Engine(); eng.ensure_started(); await asyncio.sleep(0.3)
    depth_dir = Path(args.out).parent / f"vace_depth_{args.scene}"
    if depth_dir.exists():
        shutil.rmtree(depth_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)

    # 1. Blender depth sequence (the control video).
    _log(f"[1/2] Blender depth sequence: {args.scene}, {args.frames} frames ...")
    r = bl.run_script(scenes.depth_sequence(args.scene),
                      args=[str(depth_dir), args.frames, spec["near"], spec["far"]], timeout=600)
    if not r.ok:
        _log("blender failed:\n" + r.stderr[-800:]); return
    n = len(list(depth_dir.glob("depth_*.png")))
    _log(f"      {n} control frames -> {depth_dir}")

    # 2. Wan VACE: one-pass coherent video conditioned on the depth control.
    _log(f"[2/2] Wan VACE generating (length {n}, seed {args.seed}) ...")
    res = await eng.jobs.run_workflow(
        "vace_depth_video",
        {"positive_prompt": prompt, "control_dir": str(depth_dir).replace("\\", "/"),
         "length": n, "strength": args.strength, "steps": args.steps, "seed": args.seed,
         "fps": float(args.fps), "model": args.model},
        wait=True, timeout_s=max(1800, eng.cfg.timeouts.video_job_s))
    if res["state"] != "completed" or not res["outputs"]:
        _log("VACE failed: " + str(res.get("error"))); await eng.client.aclose(); return

    # collect the webp clip -> mp4
    o = res["outputs"][0]
    src = Path(eng.cfg.comfyui.output_dir) / o.get("subfolder", "") / o["filename"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    raw = str(Path(args.out).with_suffix(".raw.mp4")) if args.interp_fps else args.out
    w = imageio.get_writer(raw, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
    for fr in ImageSequence.Iterator(Image.open(src)):
        w.append_data(np.array(fr.convert("RGB")))
    w.close()
    if args.interp_fps:
        _log(f"      motion-interpolating {args.fps}->{args.interp_fps}fps ...")
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ff, "-y", "-i", raw, "-vf",
                        f"minterpolate=fps={args.interp_fps}:mi_mode=mci:mc_mode=aobmc:"
                        "me_mode=bidir:vsbmc=1", "-c:v", "libx264", "-crf", "16",
                        "-pix_fmt", "yuv420p", args.out], check=True, capture_output=True)
        Path(raw).unlink(missing_ok=True)
    _log(f"DONE -> {args.out}  (source: {o['filename']})")
    await eng.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
