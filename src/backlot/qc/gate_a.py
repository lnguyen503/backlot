"""Gate A deterministic media checks (ffmpeg) - shared by clip_qc (previs) and
review_episode (assembled master). Thresholds locked on eps 001-004:
masters fail on freezes >= 0.6s (0.3-0.6 = warnings); clip pre-screen keeps 0.3s.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
FFPROBE = shutil.which("ffprobe") or FF.replace("ffmpeg", "ffprobe")
FREEZE_MIN_S = 0.3           # clip pre-screen (cheap to reroll before assembly)
FREEZE_MASTER_S = 0.6        # assembled master hard-fail
LUFS_TARGET, LUFS_TOL = -14.0, 1.0
AV_DRIFT_MAX_S = 0.2


def probe(path: Path) -> dict:
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries",
                        "stream=codec_type,width,height,avg_frame_rate,duration",
                        "-show_entries", "format=duration", "-of", "json", str(path)],
                       capture_output=True, text=True)
    j = json.loads(r.stdout or "{}")
    out = {"duration": None, "fps": None, "resolution": None,
           "v_dur": None, "a_dur": None}
    try:
        out["duration"] = float(j["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        pass
    for s in j.get("streams", []):
        if s.get("codec_type") == "video":
            num, _, den = (s.get("avg_frame_rate") or "0/1").partition("/")
            if float(den or 1):
                out["fps"] = round(float(num) / float(den or 1), 2)
            out["resolution"] = f"{s.get('width')}x{s.get('height')}"
            out["v_dur"] = float(s["duration"]) if s.get("duration") else None
        elif s.get("codec_type") == "audio":
            out["a_dur"] = float(s["duration"]) if s.get("duration") else None
    return out


def freezes(path: Path, min_s: float = FREEZE_MIN_S) -> list[dict]:
    r = subprocess.run([FF, "-i", str(path), "-vf", f"freezedetect=n=-60dB:d={min_s}",
                        "-map", "0:v", "-f", "null", "-"], capture_output=True, text=True)
    starts = [float(m) for m in re.findall(r"freeze_start: ([\d.]+)", r.stderr)]
    ends = [float(m) for m in re.findall(r"freeze_end: ([\d.]+)", r.stderr)]
    out = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else None   # trailing freeze can lack an end
        dur = (e - s) if e else None
        sev = "high" if (dur or 99) >= 1.0 else "medium"
        out.append({"start": round(s, 2), "end": round(e, 2) if e else None,
                    "duration": round(dur, 2) if dur else None, "severity": sev})
    return out


def loudness(path: Path) -> float | None:
    r = subprocess.run([FF, "-i", str(path), "-filter_complex", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    return float(m[-1]) if m else None


def max_volume(path: Path) -> float | None:
    r = subprocess.run([FF, "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"max_volume:\s*(-?[\d.]+) dB", r.stderr)
    return float(m.group(1)) if m else None


def empty_vo_checks(story: Path, seg_prefix: str = "") -> list[dict]:
    """Phantom-VO lock #3 (the 010 beat-11 lesson: chatterbox SPEAKS an error prompt on
    empty input). For every beat whose scripted vo is empty: (a) no vo wav may exist for
    that index; (b) the beat's assembled segment must be silent pre-mux (max < -60 dB).
    Deterministic and span-exact — the segment IS the beat's span."""
    findings: list[dict] = []
    bj = story / "assets" / "beats.json"
    if not bj.exists():
        return findings
    b = json.loads(bj.read_text(encoding="utf-8"))
    for item in b.get("beats", []):
        if (item.get("vo") or "").strip():
            continue
        i = item["i"]
        wav = story / "assets" / f"vo_{i:02d}.wav"
        if wav.exists():
            findings.append({"beat": i, "rule": "phantom-vo-wav", "severity": "error",
                             "message": f"empty-vo beat {i} has {wav.name} on disk - "
                                        "quarantine it (stage_vo does this automatically)"})
        seg = story / "assets" / f"{seg_prefix}final_seg_{i:02d}.mp4"
        if seg.exists():
            mv = max_volume(seg)
            if mv is not None and mv > -60.0:
                findings.append({"beat": i, "rule": "phantom-vo-audio", "severity": "error",
                                 "message": f"empty-vo beat {i} segment is NOT silent "
                                            f"pre-mux (max_volume {mv:.1f} dB > -60)"})
    return findings
