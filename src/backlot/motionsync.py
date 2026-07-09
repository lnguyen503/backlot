"""Motion transfer from a reference clip.

Sample frames from a driving video -> a Depth-Anything depth sequence (near=bright)
-> feed Wan VACE (the same depth-control path as the Blender lane) so a new subject
follows the reference's motion/structure. Depth model auto-downloads on first use
(depth-anything/Depth-Anything-V2-Small-hf, ~99MB, non-gated).
"""
from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio

_DEPTH: dict = {}   # cache the transformers pipeline (loads a model)


def _pipe():
    if "p" not in _DEPTH:
        from transformers import pipeline
        _DEPTH["p"] = pipeline("depth-estimation",
                               model="depth-anything/Depth-Anything-V2-Small-hf")
    return _DEPTH["p"]


def sample_frames(video_path: str, n: int) -> list:
    """Evenly sample n frames across a video -> list of PIL RGB images."""
    from PIL import Image
    r = imageio.get_reader(str(video_path))
    try:
        total = r.count_frames()
    except Exception:
        total = 0
    idxs = ([int(round(i * (total - 1) / max(1, n - 1))) for i in range(n)]
            if total > 1 else list(range(n)))
    frames = []
    for i in idxs:
        try:
            frames.append(Image.fromarray(r.get_data(i)).convert("RGB"))
        except (IndexError, ValueError):
            break
    r.close()
    return frames


def build_depth_control(video_path: str, out_dir: str, n: int = 25, size: int = 768) -> int:
    """Write a normalized depth PNG sequence (near=bright) to out_dir. Returns frame count."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipe = _pipe()
    frames = sample_frames(video_path, n)
    for i, fr in enumerate(frames, 1):
        depth = pipe(fr)["depth"].resize((size, size)).convert("RGB")
        depth.save(out / f"depth_{i:04d}.png")
    return len(frames)
