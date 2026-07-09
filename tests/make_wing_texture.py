"""Generate a real monarch-butterfly wing texture in ComfyUI (SDXL), then key out
the chroma-green background in Python to produce an RGBA cutout for Blender.

Output: runs/wing_assets/wing_full.png (RGBA), wing_R.png (right half, RGBA).
The Blender butterfly maps wing_R onto both wings (left = mirrored) for symmetry.

    .venv\\Scripts\\python.exe tests\\make_wing_texture.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backlot.engine.runtime import Engine  # noqa: E402

OUT = Path(str(Path(__file__).resolve().parents[1] / "runs/wing_assets"))

POS = (
    "a single monarch butterfly, wings fully spread open and flat, dorsal top-down view, "
    "perfectly symmetric, vibrant deep orange wings with bold black vein lines and thick "
    "black wing borders dotted with small white spots, entomology specimen pinned flat, "
    "centered, isolated on a solid uniform chroma key green screen background, flat even "
    "studio lighting, no shadow, razor sharp focus, ultra detailed, macro photography"
)
NEG = (
    "gradient background, cast shadow, vignette, 3d render, cartoon, illustration, blurry, "
    "soft focus, depth of field, multiple butterflies, hands, text, watermark, frame, "
    "leaves, flowers, plants, perspective tilt, side view"
)


def key_bg(img: Image.Image, thresh: float = 42.0, ramp: float = 22.0) -> Image.Image:
    """Key the uniform background by colour-distance from the sampled corners.
    Robust to whatever flat backdrop SDXL produced (olive/grey/green)."""
    a = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w, _ = a.shape
    c = 40
    corners = np.concatenate([
        a[:c, :c].reshape(-1, 3), a[:c, -c:].reshape(-1, 3),
        a[-c:, :c].reshape(-1, 3), a[-c:, -c:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    dist = np.sqrt(((a - bg) ** 2).sum(axis=-1))
    alpha = np.clip((dist - thresh) / ramp, 0.0, 1.0)
    # fill interior holes: keep only the alpha as-is (butterfly is solid & high-contrast)
    out = np.dstack([a, (alpha * 255)]).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def crop_to_alpha(img: Image.Image, pad: int = 8) -> Image.Image:
    a = np.asarray(img)[..., 3]
    ys, xs = np.where(a > 16)
    if len(xs) == 0:
        return img
    x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + pad)
    return img.crop((x0, y0, x1, y1))


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eng = Engine(); eng.ensure_started(); await asyncio.sleep(0.3)
    print("[1/2] SDXL generating monarch (green screen) ...", flush=True)
    res = await eng.jobs.run_workflow(
        "txt2img_sdxl",
        {"positive_prompt": POS, "negative_prompt": NEG,
         "width": 1024, "height": 1024, "steps": 34, "cfg": 6.5, "seed": 23},
        wait=True, timeout_s=600)
    if res["state"] != "completed" or not res["outputs"]:
        print("FAILED:", res.get("error")); await eng.client.aclose(); return
    o = res["outputs"][0]
    src = Path(eng.cfg.comfyui.output_dir) / o.get("subfolder", "") / o["filename"]
    print("      raw ->", src, flush=True)

    print("[2/2] keying green + cropping ...", flush=True)
    raw = Image.open(src)
    raw.save(OUT / "wing_raw.png")
    cut = crop_to_alpha(key_green(raw))
    cut.save(OUT / "wing_full.png")
    # right half (center -> right tip); body seam covered by 3D body in Blender
    w, h = cut.size
    right = cut.crop((w // 2, 0, w, h))
    right.save(OUT / "wing_R.png")
    cov = (np.asarray(cut)[..., 3] > 16).mean()
    print(f"DONE -> {OUT}  (full {cut.size}, right {right.size}, alpha coverage {cov:.0%})")
    await eng.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
