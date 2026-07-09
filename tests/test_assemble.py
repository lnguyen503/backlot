"""Functional test for the storyboard assembler (uses the bundled ffmpeg).

Verifies the fix for the mp4/webp mixed-format crash: a talking panel's mp4
(with audio) + a silent motion clip concatenate into one sequence that KEEPS the
dialogue audio track. No ComfyUI/Ollama needed — only ffmpeg (via imageio_ffmpeg).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.storyboard import assemble as A
from backlot.storyboard.models import Asset, Panel, Storyboard

FF = imageio_ffmpeg.get_ffmpeg_exe()


def _fake_eng(out_dir: Path):
    cfg = SimpleNamespace(comfyui=SimpleNamespace(output_dir=str(out_dir)))
    client = SimpleNamespace(view_url=lambda fn, sub, typ: f"/view?filename={fn}")
    return SimpleNamespace(cfg=cfg, client=client)


def _make_clip(path: Path, seconds: float, with_audio: bool, w=320, h=240):
    cmd = [FF, "-y", "-f", "lavfi", "-i",
           f"testsrc=size={w}x{h}:rate=24:duration={seconds}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-shortest"]
    subprocess.run(cmd + [str(path)], check=True, capture_output=True)


def test_assemble_mixed_clips_preserves_audio(tmp_path):
    talk = tmp_path / "talk.mp4"          # a "talking panel" clip WITH voice
    motion = tmp_path / "motion.mp4"      # a silent motion clip
    _make_clip(talk, 1.0, with_audio=True)
    _make_clip(motion, 1.0, with_audio=False)

    sb = Storyboard(title="T")
    sb.panels = [
        Panel(video=Asset(type="video", filename="talk.mp4")),
        Panel(video=Asset(type="video", filename="motion.mp4")),
    ]
    eng = _fake_eng(tmp_path)

    out = A.assemble(eng, sb, fps=24)
    out_path = tmp_path / A._SUB / out.filename
    assert out_path.exists() and out_path.stat().st_size > 0
    # the concatenated result must still carry an audio stream (dialogue survived)
    assert A._has_audio(FF, out_path)
    # per-clip audio detection is correct
    assert A._has_audio(FF, talk) and not A._has_audio(FF, motion)


def test_assemble_handles_animated_webp(tmp_path):
    # SVD/Wan output animated webp, which ffmpeg 7.x can't decode — assemble must
    # transcode it (imageio) first. A raw ffmpeg path would fail here.
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    webp = tmp_path / "svd.webp"
    frames = [Image.fromarray(np.full((240, 320, 3), (i * 40) % 255, np.uint8)) for i in range(6)]
    frames[0].save(webp, save_all=True, append_images=frames[1:], duration=125, loop=0)
    assert len(list(imageio.get_reader(str(webp)))) > 1        # it really is animated

    sb = Storyboard(title="W")
    sb.panels = [Panel(video=Asset(type="video", filename="svd.webp"))]
    out = A.assemble(_fake_eng(tmp_path), sb, fps=24)
    out_path = tmp_path / A._SUB / out.filename
    assert out_path.exists() and out_path.stat().st_size > 0
    assert A._has_audio(FF, out_path)                          # silence track added


def test_assemble_no_clips_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no animated panels"):
        A.assemble(_fake_eng(tmp_path), Storyboard(title="empty"), fps=24)
