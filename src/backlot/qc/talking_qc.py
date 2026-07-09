"""QC stage 4 - talking-segment frame-check for talking-host segments,
run BEFORE assembly so a bad take never reaches the cut.

  .venv\\Scripts\\python -m backlot.qc.talking_qc --story <dir> [--no-vlm]
  .venv\\Scripts\\python -m backlot.qc.talking_qc --work <seg work dir> [--no-vlm]

Segment layout: <work>/plan.json + seg_{i}_{emotion}_{intensity}.mp4.
Checks per segment:
- face present in every frame (a lost face = broken take)
- head-pose SNAP at NATIVE fps: a true glitch concentrates the whole jump in one
  frame; a fast-but-natural turn spreads it. Calibrated on real approved
  segments (natural per-frame delta peaks at 7.7 deg @25fps; the old 6fps
  sampling false-failed 5/6 of them at SNAP_DEG=11).
- dead take: near-zero head motion across the whole segment
- expression vs the arc tag via VLM on ONE timestamped CONTACT SHEET per segment
  (full coverage; isolated frames miss tail glitches). scene_match
  below bar = error (rerender); individual VLM glitch reports are WARN only -
  precision on subtle expression glitches (the slow-wink class) is too low
  to auto-rerender on, so they surface to the human at creative-go instead.
Writes <story>/qc/talking.json + talking.md (or into the work dir with --work).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

SNAP_DEG_PER_FRAME = 10.0   # at 25fps; approved-content natural max is 7.7
REF_FPS = 25.0              # threshold scales for other native rates
DEAD_DEG = 0.8              # total pose travel below this = frozen/dead take
SHEET_FPS = 2.0             # contact-sheet sampling
SHEET_MAX = 16              # tile cap per sheet
SHEET_COLS = 4


def _frames(path: Path) -> tuple[float, list[np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or REF_FPS
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return fps, frames


def _poses(frames: list[np.ndarray]) -> list[np.ndarray | None]:
    from .faceid import _app
    app = _app()
    poses = []
    for fr in frames:
        faces = app.get(fr)
        if not faces:
            poses.append(None)
            continue
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]))
        poses.append(np.asarray(f.pose, dtype=float))
    return poses


def pose_findings(poses: list[np.ndarray | None], fps: float) -> list[dict]:
    """Pure pose-series checks (unit-testable): face-lost, pose-snap, dead-take."""
    findings = []
    missing = sum(1 for p in poses if p is None)
    if missing:
        findings.append({"rule": "face-lost", "severity": "error",
                         "message": f"face not detected in {missing}/{len(poses)} "
                                    "frames - broken take",
                         "fix_action": "rerender"})
    snap_thr = SNAP_DEG_PER_FRAME * (REF_FPS / fps if fps else 1.0)
    deltas = [(i + 1, float(np.abs(b - a).max()))
              for i, (a, b) in enumerate(zip(poses, poses[1:]))
              if a is not None and b is not None]
    if deltas:
        i, snap = max(deltas, key=lambda d: d[1])
        if snap > snap_thr:
            findings.append({"rule": "pose-snap", "severity": "error",
                             "message": f"head pose jumps {snap:.1f} deg in ONE "
                                        f"frame (~{i / fps:.1f}s, natural max ~7.7 "
                                        f"@25fps) - glitch/snap",
                             "fix_action": "rerender"})
        valid = [p for p in poses if p is not None]
        travel = float(np.ptp(np.stack(valid), axis=0).max()) if len(valid) > 1 else 0.0
        if travel < DEAD_DEG:
            findings.append({"rule": "dead-take", "severity": "warn",
                             "message": f"total head motion {travel:.2f} deg - "
                                        "reads as a freeze",
                             "fix_action": "rerender"})
    return findings


def contact_sheet(frames: list[np.ndarray], fps: float, out: Path) -> tuple[int, int]:
    """Tile ~SHEET_FPS-sampled frames into one grid image; returns (rows, cols)."""
    step = max(1, round(fps / SHEET_FPS))
    picks = frames[::step][:SHEET_MAX]
    cols = min(SHEET_COLS, len(picks))
    rows = -(-len(picks) // cols)
    h, w = picks[0].shape[:2]
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for k, fr in enumerate(picks):
        r, c = divmod(k, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = fr
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return rows, cols


def vlm_findings(frames: list[np.ndarray], fps: float, emotion: str,
                 intensity: int) -> list[dict]:
    from . import vlm
    findings = []
    with tempfile.TemporaryDirectory() as td:
        sheet = Path(td) / "sheet.jpg"
        rows, cols = contact_sheet(frames, fps, sheet)
        v = vlm.judge([sheet], (
            f"This is a CONTACT SHEET of ONE talking-head clip: {rows}x{cols} tiles "
            "in reading order (left-to-right, top-to-bottom), sampled at "
            f"{SHEET_FPS:g} frames per second, so adjacent tiles are "
            f"{1 / SHEET_FPS:.1f}s apart. The clip is a storyteller delivering a "
            f"segment tagged '{emotion}' at intensity {intensity}/3. "
            "Score scene_match 0-10 for how well her facial expression fits that "
            "emotional tag across ALL tiles (10 = fits throughout; low = wrong "
            "expression, e.g. a smile during a somber goodbye). "
            "Report issues: type 'other' for an expression mismatch or unnatural "
            "flicker (say which tiles); 'anatomy' for a deformed face. "
            "Only report what you actually see."))
    if v["scene_match"] < 6:
        findings.append({"rule": "expression-mismatch", "severity": "error",
                         "message": f"expression vs '{emotion}' tag: "
                                    f"{v['scene_match']}/10 - {v['summary']}",
                         "fix_action": "rerender"})
    for iss in v["issues"]:
        # WARN only: VLM precision on subtle expression glitches is too low to
        # auto-rerender on - these route to the human at creative-go.
        findings.append({"rule": f"vlm-{iss['type']}", "severity": "warn",
                         "message": iss["detail"], "fix_action": "human"})
    return findings


def check_segment(path: Path, emotion: str, intensity: int,
                  use_vlm: bool = True) -> dict:
    fps, frames = _frames(path)
    findings = pose_findings(_poses(frames), fps)
    if use_vlm and frames:
        findings += vlm_findings(frames, fps, emotion, intensity)
    errors = [f for f in findings if f["severity"] == "error"]
    return {"segment": path.stem, "emotion": emotion, "intensity": intensity,
            "verdict": "fail" if errors else "pass",
            "fix_action": "rerender" if errors else None, "findings": findings}


def run_work(work: Path, use_vlm: bool = True) -> list[dict]:
    plan = json.loads((work / "plan.json").read_text(encoding="utf-8"))["segments"]
    out = []
    for i, seg in enumerate(plan):
        mp4 = work / f"seg_{i}_{seg['emotion']}_{seg['intensity']}.mp4"
        if not mp4.exists():
            continue
        r = check_segment(mp4, seg["emotion"], int(seg["intensity"]), use_vlm=use_vlm)
        out.append(r)
        print(f"{work.name}/{mp4.stem}: {r['verdict'].upper()}", flush=True)
        for f in r["findings"]:
            print(f"  {f['severity'].upper():5} [{f['rule']}] {f['message']}", flush=True)
    return out


def run(story: Path, use_vlm: bool = True) -> dict:
    segs = []
    for work in sorted((story / "assets").glob("*_work")):
        if (work / "plan.json").exists():
            segs += run_work(work, use_vlm=use_vlm)
    report = {"story": story.name, "stage": "talking-qc", "vlm": use_vlm,
              "checked": len(segs),
              "failed": sum(1 for s in segs if s["verdict"] == "fail"),
              "verdict": "fail" if any(s["verdict"] == "fail" for s in segs) else "pass",
              "segments": segs}
    qc = story / "qc"
    qc.mkdir(exist_ok=True)
    (qc / "talking.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [f"# talking-qc - {story.name}",
             f"**{report['verdict'].upper()}** - {report['failed']}/{report['checked']} "
             f"segments failed (VLM {'on' if use_vlm else 'OFF'})", ""]
    for s in segs:
        mark = "✅" if s["verdict"] == "pass" else "❌"
        lines.append(f"- {mark} {s['segment']} ({s['emotion']} {s['intensity']})")
        for f in s["findings"]:
            lines.append(f"  - {f['severity']}: [{f['rule']}] {f['message']}")
    (qc / "talking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--story")
    ap.add_argument("--work")
    ap.add_argument("--no-vlm", action="store_true")
    args = ap.parse_args()
    if args.work:
        segs = run_work(Path(args.work), use_vlm=not args.no_vlm)
        sys.exit(1 if any(s["verdict"] == "fail" for s in segs) else 0)
    report = run(Path(args.story), use_vlm=not args.no_vlm)
    print(f"\ntalking-qc: {report['verdict'].upper()} "
          f"({report['failed']}/{report['checked']} failed)")
    sys.exit(1 if report["verdict"] == "fail" else 0)


if __name__ == "__main__":
    main()
