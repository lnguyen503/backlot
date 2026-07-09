"""Blender geometry -> AI look: render a Blender depth pass, then restyle it with
SDXL + depth-ControlNet so the prompt's style follows the 3D structure.

The proof of the TODO "Blender passes -> ComfyUI ControlNet" combo. Run from .venv:
    .venv\\Scripts\\python.exe tests\\make_blender_restyle.py --prompt "marble bust" --out out.png
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.blender import runner, scenes          # noqa: E402
from backlot.engine.runtime import Engine            # noqa: E402

DEFAULT_PROMPT = ("a photorealistic carved white marble bust of a monkey on a pedestal, "
                  "museum studio lighting, intricate chiselled detail, sharp focus")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--depth", default="", help="existing depth PNG; else render the demo scene")
    ap.add_argument("--strength", type=float, default=0.8)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs/blender_restyle.png"))
    args = ap.parse_args()

    eng = Engine()
    eng.ensure_started()
    await asyncio.sleep(0.3)
    in_dir = Path(eng.cfg.comfyui.input_dir)
    out_dir = Path(eng.cfg.comfyui.output_dir)

    # 1. Blender depth pass (geometry).
    if args.depth:
        depth = Path(args.depth)
    else:
        scratch = Path(args.out).parent
        scratch.mkdir(parents=True, exist_ok=True)
        beauty = scratch / "_bl_beauty.png"
        depth = scratch / "_bl_depth.png"
        print("[1/3] rendering Blender depth pass ...", flush=True)
        r = runner.run_script(scenes.depth_pass(), args=[str(beauty), str(depth)], timeout=300)
        if not r.ok:
            print("blender failed:\n", r.stderr[-800:]); return
        print(f"      depth -> {depth}", flush=True)

    # 2. Stage depth into ComfyUI input/.
    staged = f"backlot/bl_depth_{uuid.uuid4().hex[:8]}.png"
    (in_dir / "backlot").mkdir(parents=True, exist_ok=True)
    shutil.copy(depth, in_dir / staged)

    # 3. SDXL + depth-ControlNet restyle.
    print(f"[2/3] restyling with SDXL+depth-ControlNet (strength {args.strength}) ...", flush=True)
    res = await eng.jobs.run_workflow(
        "txt2img_controlnet_depth_sdxl",
        {"positive_prompt": args.prompt, "image": staged,
         "width": 1024, "height": 1024, "steps": 30, "cfg": 7.0,
         "strength": args.strength, "seed": 7},
        wait=True, timeout_s=eng.cfg.timeouts.image_job_s,
    )
    if res["state"] != "completed" or not res["outputs"]:
        print("restyle failed:", res.get("error")); await eng.client.aclose(); return
    o = res["outputs"][0]
    src = out_dir / o.get("subfolder", "") / o["filename"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, args.out)
    print(f"[3/3] DONE -> {args.out}", flush=True)
    await eng.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
