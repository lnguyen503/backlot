"""Stitch a storyboard's panel clips into one sequence mp4.

The panels' clips are a MIX of formats: SVD/Wan output animated **webp** (silent),
while a talking panel's InfiniteTalk output is a real **mp4 with embedded voice**.
PIL can't read an mp4 as an image sequence (the old frame-by-frame stitch crashed
on talking panels), and a naive concat would drop the dialogue audio. So we
normalise every clip to a fixed WxH/fps mp4 that ALWAYS carries a stereo audio
track (its own voice, else silence) and concat them — preserving per-panel speech.
"""
from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image

from ..engine.runtime import Engine
from .models import Asset, Storyboard

_SUB = "backlot-storyboards"


def _abs(eng: Engine, asset: Asset) -> Path:
    return Path(eng.cfg.comfyui.output_dir) / (asset.subfolder or "") / asset.filename


def _clip_dims(path: Path) -> tuple[int, int]:
    """(w, h) of a clip — imageio for mp4/webm, PIL for animated webp."""
    if path.suffix.lower() in (".mp4", ".webm", ".mov"):
        r = imageio.get_reader(str(path))
        w, h = r.get_meta_data().get("size", (512, 512))
        r.close()
        return int(w), int(h)
    with Image.open(path) as im:
        return im.size


def _has_audio(ff: str, path: Path) -> bool:
    """True if the clip carries an audio stream (parsed from ffmpeg's probe)."""
    r = subprocess.run([ff, "-i", str(path)], capture_output=True, text=True)
    return "Audio:" in (r.stderr or "")


def _to_mp4(clip: Path, dst: Path, fps: int) -> Path:
    """Animated webp -> mp4. ffmpeg 7.x can't decode animated webp (SVD/Wan output);
    imageio/Pillow can, so transcode first. mp4/webm clips pass through unchanged."""
    if clip.suffix.lower() != ".webp":
        return clip
    reader = imageio.get_reader(str(clip))
    src_fps = reader.get_meta_data().get("fps") or fps
    writer = imageio.get_writer(str(dst), fps=src_fps, codec="libx264",
                                quality=8, macro_block_size=1)
    try:
        for frame in reader:
            writer.append_data(frame[:, :, :3] if frame.ndim == 3 and frame.shape[2] == 4 else frame)
    finally:
        writer.close()
        reader.close()
    return dst


def _norm_clip(ff: str, clip: Path, dst: Path, w: int, h: int, fps: int) -> None:
    """Re-encode one clip to a fixed WxH/fps mp4 that ALWAYS has a stereo audio
    track (its own audio if present, else generated silence) so the parts line up."""
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:-1:-1:color=black,setsar=1,fps={fps},format=yuv420p")
    cmd = [ff, "-y", "-i", str(clip)]
    if _has_audio(ff, clip):
        cmd += ["-vf", vf, "-c:v", "libx264", "-c:a", "aac", "-ar", "44100", "-ac", "2"]
    else:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-vf", vf,
                "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-c:a", "aac", "-shortest"]
    subprocess.run(cmd + [str(dst)], check=True, capture_output=True)


def _duration(path: Path) -> float:
    """Seconds of a (normalised) clip."""
    r = imageio.get_reader(str(path))
    m = r.get_meta_data()
    d = m.get("duration") or (r.count_frames() / float(m.get("fps", 24) or 24))
    r.close()
    return float(d)


def _concat_hard(ff: str, parts: list[Path], out: Path, tmp: Path) -> None:
    """Butt-join clips (hard cuts) via the concat demuxer — fast, stream-copy."""
    (tmp / "list.txt").write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    subprocess.run([ff, "-y", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                    "-c", "copy", str(out)], check=True, capture_output=True)


def _concat_xfade(ff: str, parts: list[Path], out: Path, xdur: float) -> None:
    """Cross-dissolve between clips (video xfade + audio acrossfade) so the
    pose/framing jump between independent shots reads as a smooth transition."""
    durs = [_duration(p) for p in parts]
    vf, af = "", ""
    pv, pa, acc = "[0:v]", "[0:a]", durs[0]
    for i in range(1, len(parts)):
        off = max(0.0, acc - xdur)
        vlab, alab = f"[v{i}]", f"[a{i}]"
        vf += f"{pv}[{i}:v]xfade=transition=fade:duration={xdur}:offset={off:.3f}{vlab};"
        af += f"{pa}[{i}:a]acrossfade=d={xdur}{alab};"
        pv, pa, acc = vlab, alab, acc + durs[i] - xdur
    cmd = [ff, "-y"]
    for p in parts:
        cmd += ["-i", str(p)]
    cmd += ["-filter_complex", (vf + af).rstrip(";"), "-map", pv, "-map", pa,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)]
    subprocess.run(cmd, check=True, capture_output=True)


def assemble(eng: Engine, sb: Storyboard, fps: int = 24, crossfade: float = 0.0) -> Asset:
    """Stitch the panels' clips (board order) into one sequence mp4, preserving
    per-panel dialogue and normalising mixed webp/mp4 clips. crossfade>0 cross-
    dissolves between shots (seconds) to smooth the cut; 0 = hard cuts."""
    clips = [_abs(eng, p.video) for p in sb.panels if p.video is not None]
    if not clips:
        raise RuntimeError("no animated panels to assemble")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    w, h = _clip_dims(clips[0])
    out_dir = Path(eng.cfg.comfyui.output_dir) / _SUB
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = f"{sb.id}_{uuid.uuid4().hex[:6]}.mp4"
    tmp = out_dir / f"_asm_{uuid.uuid4().hex[:6]}"
    tmp.mkdir()
    try:
        parts = []
        for i, clip in enumerate(clips):
            src = _to_mp4(clip, tmp / f"src{i}.mp4", fps)
            _norm_clip(ff, src, tmp / f"p{i}.mp4", w, h, fps)
            parts.append(tmp / f"p{i}.mp4")
        if crossfade > 0 and len(parts) > 1:
            _concat_xfade(ff, parts, out_dir / fn, crossfade)
        else:
            _concat_hard(ff, parts, out_dir / fn, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    url = eng.client.view_url(fn, _SUB, "output")
    sb.assembled = Asset(type="video", filename=fn, subfolder=_SUB, url=url)
    return sb.assembled
