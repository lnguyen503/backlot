"""Render ONE VACE chunk in an isolated process (clean VRAM), saving kid frames as PNGs.

Used to render a long depth sequence in chunks without the multi-pass OOM that kills a
long-lived process. Orchestrate from a shell loop (free VRAM + clear queue between calls).
"""
import asyncio, sys, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageSequence
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backlot.engine.runtime import Engine


async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-dir", required=True)
    ap.add_argument("--start", type=int, required=True)     # 0-based offset into sorted depth frames
    ap.add_argument("--count", type=int, required=True)     # 4n+1
    ap.add_argument("--w", type=int, default=832)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--model", default="wan2.1_vace_14B_fp16.safetensors")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="adults, grown men, blurry, deformed, extra limbs, watermark, text")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    ddir = Path(args.depth_dir).resolve()
    depth = sorted(ddir.glob("depth_*.png"))[args.start:args.start + args.count]
    tmp = ddir.parent / f"onechunk_{args.start}"
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for j, fp in enumerate(depth, 1):
        shutil.copy(fp, tmp / f"depth_{j:04d}.png")

    eng = Engine(); eng.ensure_started(); await asyncio.sleep(0.3)
    res = await eng.jobs.run_workflow("vace_depth_video", {
        "positive_prompt": args.prompt, "negative_prompt": args.negative,
        "control_dir": str(tmp).replace("\\", "/"), "length": len(depth),
        "width": args.w, "height": args.h, "load_width": args.w, "load_height": args.h,
        "strength": 1.0, "steps": args.steps, "seed": args.seed, "model": args.model,
        "fps": 16.0}, wait=True, timeout_s=2400)
    if res["state"] != "completed" or not res["outputs"]:
        print("CHUNK_FAILED", str(res.get("error"))[:300]); await eng.client.aclose(); sys.exit(1)
    o = res["outputs"][0]
    webp = Path(eng.cfg.comfyui.output_dir) / o.get("subfolder", "") / o["filename"]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for j, fr in enumerate(ImageSequence.Iterator(Image.open(webp))):
        Image.fromarray(np.array(fr.convert("RGB"))).save(out / f"kid_{args.start + j:04d}.png")
    print(f"CHUNK_OK start={args.start} saved={len(depth)}")
    await eng.client.aclose()


asyncio.run(main())
