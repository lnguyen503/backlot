"""Temporally-consistent AI-stylized 3D animation: Blender geometry -> AI look.

Keyframe a camera move in Blender -> render a DEPTH SEQUENCE (fixed normalization,
stable across frames) -> restyle every frame through SDXL + depth-ControlNet with a
FIXED seed (geometry locked by depth, noise locked by seed) -> stitch to mp4.

    .venv\\Scripts\\python.exe tests\\make_blender_animation.py --frames 16 --prompt "..."
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import shutil
import sys
import uuid
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.blender import runner, scenes          # noqa: E402
from backlot.engine.runtime import Engine            # noqa: E402

# Per-scene depth range + a fitting default style prompt.
SCENES = {
    "monkey_orbit": dict(near=3.0, far=11.0, prompt=(
        "a photorealistic carved white marble bust of a monkey on a pedestal, "
        "museum studio lighting, intricate chiselled detail, sharp focus")),
    "sculpture_turntable": dict(near=3.2, far=9.5, prompt=(
        "a polished bronze and gold abstract sculpture, a ring around an orb, "
        "reflective burnished metal, dramatic studio lighting, dark museum backdrop")),
    "city_flythrough": dict(near=2.0, far=18.0, prompt=(
        "a neon cyberpunk city at night, rain-soaked streets, glowing signs, "
        "blade runner, volumetric haze, cinematic, ultra detailed")),
}


def _log(m): print(m, flush=True)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="monkey_orbit", choices=list(SCENES))
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--prompt", default="")                  # blank -> scene default
    ap.add_argument("--strength", type=float, default=0.9)   # high = tight geometry lock
    ap.add_argument("--seed", type=int, default=7)           # fixed -> temporal coherence
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--near", type=float, default=-1.0)      # <0 -> scene default
    ap.add_argument("--far", type=float, default=-1.0)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs/blender_animation.mp4"))
    args = ap.parse_args()
    spec = SCENES[args.scene]
    if not args.prompt: args.prompt = spec["prompt"]
    if args.near < 0: args.near = spec["near"]
    if args.far < 0: args.far = spec["far"]

    eng = Engine(); eng.ensure_started(); await asyncio.sleep(0.3)
    in_dir = Path(eng.cfg.comfyui.input_dir)
    out_dir = Path(eng.cfg.comfyui.output_dir)
    scratch = Path(args.out).parent / f"blanim_{uuid.uuid4().hex[:6]}"
    depth_dir = scratch / "depth"; styled_dir = scratch / "styled"
    depth_dir.mkdir(parents=True, exist_ok=True); styled_dir.mkdir(parents=True, exist_ok=True)

    # 1. Blender: animated depth sequence (stable normalization).
    _log(f"[1/3] rendering {args.frames}-frame Blender depth sequence ({args.scene}) ...")
    r = runner.run_script(scenes.depth_sequence(args.scene),
                          args=[str(depth_dir), args.frames, args.near, args.far], timeout=600)
    if not r.ok:
        _log("blender failed:\n" + r.stderr[-800:]); return
    depths = sorted(glob.glob(str(depth_dir / "depth_*.png")))
    _log(f"      {len(depths)} depth frames")

    # 2. Restyle each frame: SDXL + depth-ControlNet, FIXED seed.
    _log(f"[2/3] restyling {len(depths)} frames (SDXL+depth-ControlNet, seed {args.seed}) ...")
    (in_dir / "backlot").mkdir(parents=True, exist_ok=True)
    styled: list[Path] = []
    for i, dp in enumerate(depths, 1):
        staged = f"backlot/blanim_{uuid.uuid4().hex[:8]}.png"
        shutil.copy(dp, in_dir / staged)
        res = await eng.jobs.run_workflow(
            "txt2img_controlnet_depth_sdxl",
            {"positive_prompt": args.prompt, "image": staged,
             "width": 1024, "height": 1024, "steps": args.steps, "cfg": 7.0,
             "strength": args.strength, "seed": args.seed},
            wait=True, timeout_s=eng.cfg.timeouts.image_job_s)
        if res["state"] != "completed" or not res["outputs"]:
            _log(f"  ! frame {i} failed: {res.get('error')}"); continue
        o = res["outputs"][0]
        dst = styled_dir / f"f_{i:04d}.png"
        shutil.copy(out_dir / o.get("subfolder", "") / o["filename"], dst)
        styled.append(dst)
        _log(f"      frame {i}/{len(depths)}")

    if not styled:
        _log("no styled frames"); await eng.client.aclose(); return

    # 3. Stitch.
    _log(f"[3/3] stitching {len(styled)} frames @ {args.fps}fps ...")
    w = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
    for p in styled:
        w.append_data(np.array(Image.open(p).convert("RGB")))
    w.close()
    _log(f"DONE -> {args.out}")
    await eng.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
