"""Restyle people in a real video (adults -> kids) while KEEPING the original
background, at higher resolution, with sharpened faces.

Pipeline:
  source video --DepthAnything--> depth (preserves poses/perspective)
               --Wan VACE 14B--> kids clip (same poses), at VACE res
  source video --YOLOv8-seg--> per-frame PERSON masks (feathered)
  composite:  out = mask * upscale(VACE kids) + (1-mask) * ORIGINAL frame   (bg preserved, 720p)
  --GFPGAN--> restore/sharpen faces ; --ffmpeg--> + original audio

    .venv\\Scripts\\python.exe tests\\make_video_restyle.py --src IN.mp4 --start 2 --dur 3 --out OUT.mp4
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import warnings
from pathlib import Path

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageFilter, ImageSequence

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.engine.runtime import Engine                      # noqa: E402
from backlot.faceswap import _FFHQ_512, default_models, providers  # noqa: E402

FF = imageio_ffmpeg.get_ffmpeg_exe()
DEFAULT_PROMPT = (
    "three Indian children about 8 years old in a plain beige classroom, a girl with a blue hair "
    "ribbon talking to two seated boys, same poses and composition, wearing plain colorful casual "
    "t-shirts (red, blue, green), kids clothes, photorealistic, soft natural daylight, sharp focus on faces")


def _log(m): print(m, flush=True)


class FaceEnhancer:
    """Standalone GFPGAN face restore (no identity swap) — sharpens existing faces."""
    def __init__(self, gfpgan_path):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis
        prov = providers()
        self.app = FaceAnalysis(name="buffalo_l", providers=prov)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.g = ort.InferenceSession(gfpgan_path, providers=prov)
        self._gi, self._go = self.g.get_inputs()[0].name, self.g.get_outputs()[0].name

    def _restore(self, frame, kps):
        M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, np.float32), _FFHQ_512, method=cv2.LMEDS)
        if M is None:
            return frame
        aligned = cv2.warpAffine(frame, M, (512, 512), borderMode=cv2.BORDER_REFLECT)
        inp = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose((inp - 0.5) / 0.5, (2, 0, 1))[None]
        out = self.g.run([self._go], {self._gi: inp})[0][0]
        out = np.clip(np.transpose(out, (1, 2, 0)) * 0.5 + 0.5, 0, 1) * 255.0
        restored = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)
        Minv = cv2.invertAffineTransform(M)
        h, w = frame.shape[:2]
        back = cv2.warpAffine(restored, Minv, (w, h), borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(np.full((512, 512), 255, np.uint8), Minv, (w, h))
        mask = cv2.GaussianBlur(cv2.erode(mask, np.ones((15, 15), np.uint8)), (0, 0), 7)
        mask = mask.astype(np.float32)[:, :, None] / 255.0
        return (back * mask + frame * (1 - mask)).astype(np.uint8)

    def enhance(self, frame_bgr):
        for f in self.app.get(frame_bgr):
            frame_bgr = self._restore(frame_bgr, f.kps)
        return frame_bgr


def extract(src, out_dir, start, dur, fps, w, h):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([FF, "-y", "-ss", str(start), "-t", str(dur), "-i", src,
                    "-vf", f"fps={fps},scale={w}:{h}", str(out_dir / "f_%04d.png")],
                   check=True, capture_output=True)
    return sorted(out_dir.glob("f_*.png"))


def depth_frames(frames, ddir, w, h):
    import torch
    from transformers import pipeline
    ddir.mkdir(parents=True, exist_ok=True)
    pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=0 if torch.cuda.is_available() else -1)
    for i, fp in enumerate(frames, 1):
        d = pipe(Image.open(fp).convert("RGB"))["depth"].convert("L").resize((w, h))
        d.save(ddir / f"depth_{i:04d}.png")
        if i % 8 == 0 or i == len(frames):
            _log(f"      depth {i}/{len(frames)}")


def person_masks(frames, mdir, w, h, erode=5, feather=3):
    """Tight person masks: ERODE a few px (keep the seam just inside the body, so no
    background halo) + a small feather for a soft composite edge."""
    from ultralytics import YOLO
    mdir.mkdir(parents=True, exist_ok=True)
    m = YOLO("yolov8m-seg.pt")
    out = []
    for i, fp in enumerate(frames, 1):
        im = Image.open(fp).convert("RGB")
        r = m.predict(im, classes=[0], conf=0.35, verbose=False)[0]
        mask = np.zeros((h, w), np.uint8)
        if r.masks is not None:
            for seg in r.masks.data.cpu().numpy():
                s = np.array(Image.fromarray((seg * 255).astype("uint8")).resize((w, h)))
                mask = np.maximum(mask, s)
        mk = Image.fromarray(mask).filter(ImageFilter.MinFilter(erode)).filter(
            ImageFilter.GaussianBlur(feather))
        mk.save(mdir / f"m_{i:04d}.png")
        out.append(mdir / f"m_{i:04d}.png")
        if i % 8 == 0 or i == len(frames):
            _log(f"      mask {i}/{len(frames)}")
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative", default="adults, grown men, blurry, deformed, extra limbs, watermark, text")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--vace-w", type=int, default=832)
    ap.add_argument("--vace-h", type=int, default=480)
    ap.add_argument("--out-w", type=int, default=1280)
    ap.add_argument("--out-h", type=int, default=720)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--chunk", type=int, default=49, help="VACE frames per pass (4n+1)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", default="wan2.1_vace_14B_fp16.safetensors")
    ap.add_argument("--kids-dir", default="", help="use pre-rendered kid PNGs (skip VACE)")
    ap.add_argument("--no-restore", action="store_true")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "runs/kids_restyle.mp4"))
    args = ap.parse_args()

    work = Path(args.out).parent / f"vr_{Path(args.out).stem}"
    fdir, ddir, mdir = work / "orig", work / "depth", work / "mask"
    nf = len(list(fdir.glob("f_*.png"))) if fdir.exists() else 0
    resume = (nf > 0 and len(list(ddir.glob("depth_*.png"))) == nf
              and len(list(mdir.glob("m_*.png"))) == nf)
    if resume:
        frames = sorted(fdir.glob("f_*.png"))
        _log(f"[resume] reusing {len(frames)} extracted frames/depth/masks")
    else:
        import shutil
        for d in (fdir, ddir, mdir):
            if d.exists():
                shutil.rmtree(d)
        _log(f"[1/6] extract frames @ {args.out_w}x{args.out_h}, {args.fps}fps ...")
        frames = extract(args.src, fdir, args.start, args.dur, args.fps, args.out_w, args.out_h)
        target = ((len(frames) - 1) // 4) * 4 + 1          # VACE wants 4n+1
        for extra in frames[target:]:
            extra.unlink()
        frames = frames[:target]
        _log(f"      {target} frames")
        _log("[2/6] DepthAnything (control for VACE) ...")
        depth_frames(frames, ddir, args.vace_w, args.vace_h)
        _log("[3/6] YOLO person masks (for compositing) ...")
        person_masks(frames, mdir, args.out_w, args.out_h)

    if args.kids_dir:
        kids = [np.array(Image.open(p).convert("RGB"))
                for p in sorted(Path(args.kids_dir).glob("kid_*.png"))]
        frames = frames[:len(kids)]
        _log(f"[4/6] using {len(kids)} pre-rendered kid frames from {args.kids_dir}")
        return await _finish(args, frames, mdir, kids)

    _log(f"[4/6] Wan VACE 14B kids ({args.vace_w}x{args.vace_h}) in chunks of {args.chunk} ...")
    import shutil as _sh
    import httpx as _httpx
    eng = Engine(); eng.ensure_started(); await asyncio.sleep(0.3)
    # free ONCE up front, then keep the 14B loaded across chunks (per-chunk reload churn OOMs)
    try:
        _httpx.post(f"{eng.cfg.comfyui.base_url}/free",
                    json={"unload_models": True, "free_memory": True}, timeout=10)
    except Exception:
        pass
    await asyncio.sleep(2)
    depth_all = sorted(ddir.glob("depth_*.png"))
    kids = []
    ci = 0
    for i in range(0, len(depth_all), args.chunk):
        sl = depth_all[i:i + args.chunk]
        cl = ((len(sl) - 1) // 4) * 4 + 1          # VACE wants 4n+1
        if cl < 5:
            break
        cdir = ddir.parent / f"chunk_{ci}"
        if cdir.exists(): _sh.rmtree(cdir)
        cdir.mkdir()
        for j, fp in enumerate(sl[:cl], 1):
            _sh.copy(fp, cdir / f"depth_{j:04d}.png")
        _log(f"      chunk {ci+1}: frames {i+1}-{i+cl}")
        res = await eng.jobs.run_workflow("vace_depth_video", {
            "positive_prompt": args.prompt, "negative_prompt": args.negative,
            "control_dir": str(cdir).replace("\\", "/"), "length": cl,
            "width": args.vace_w, "height": args.vace_h,
            "load_width": args.vace_w, "load_height": args.vace_h,
            "strength": args.strength, "steps": args.steps, "seed": args.seed,
            "model": args.model, "fps": float(args.fps)}, wait=True, timeout_s=2400)
        if res["state"] != "completed" or not res["outputs"]:
            _log("VACE chunk failed: " + str(res.get("error"))[:300]); await eng.client.aclose(); return
        o = res["outputs"][0]
        webp = Path(eng.cfg.comfyui.output_dir) / o.get("subfolder", "") / o["filename"]
        kids += [np.array(fr.convert("RGB")) for fr in ImageSequence.Iterator(Image.open(webp))]
        ci += 1
    await eng.client.aclose()
    frames = frames[:len(kids)]                    # trim originals to rendered count
    _log(f"      VACE frames: {len(kids)}")
    await _finish(args, frames, mdir, kids)


async def _finish(args, frames, mdir, kids):
    _log("[5/6] composite over ORIGINAL background + GFPGAN face restore ...")
    enh = None if args.no_restore else FaceEnhancer(default_models()[1])
    out_frames = []
    for i, fp in enumerate(frames):
        orig = cv2.cvtColor(np.array(Image.open(fp).convert("RGB")), cv2.COLOR_RGB2BGR)
        kid = kids[min(i, len(kids) - 1)]
        kid = cv2.resize(kid[:, :, ::-1], (args.out_w, args.out_h), interpolation=cv2.INTER_LANCZOS4)
        m = np.array(Image.open(mdir / f"m_{i+1:04d}.png").convert("L"), np.float32)[:, :, None] / 255.0
        comp = (kid * m + orig * (1 - m)).astype(np.uint8)
        if enh is not None:
            comp = enh.enhance(comp)
        out_frames.append(comp[:, :, ::-1])   # back to RGB
        if (i + 1) % 8 == 0 or i == len(frames) - 1:
            _log(f"      composite {i+1}/{len(frames)}")

    _log("[6/6] encode + audio ...")
    silent = str(Path(args.out).with_suffix(".silent.mp4"))
    w = imageio.get_writer(silent, fps=args.fps, codec="libx264", quality=9, macro_block_size=1)
    for f in out_frames:
        w.append_data(f)
    w.close()
    if args.no_audio:
        Path(args.out).unlink(missing_ok=True); Path(silent).rename(args.out)
    else:
        subprocess.run([FF, "-y", "-i", silent, "-ss", str(args.start), "-t", str(args.dur),
                        "-i", args.src, "-c:v", "copy", "-c:a", "aac", "-map", "0:v", "-map", "1:a",
                        "-shortest", args.out], check=True, capture_output=True)
        Path(silent).unlink(missing_ok=True)
    _log(f"DONE -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
