"""QC stage 3 - motion-previs QC on local clips BEFORE assembly:
Gate A freeze pre-screen + LOOP DETECTOR + optional VLM physics spot-check.

  .venv\\Scripts\\python -m backlot.qc.clip_qc --story <dir> [--clip-glob clip_*.mp4]
      [--only N] [--no-vlm]

The loop detector catches the jar/plate ping-pong class: i2v loops a discrete
action to fill the clip duration. Signal: a frame returns to a much earlier
visual state (high similarity across a >=min-gap span) after real motion
happened in between. Ambient cycles (rain, flicker) stay below the excursion
threshold; a discrete action out-and-back exceeds it.
Writes <story>/qc/clips.json + clips.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from . import gate_a
from .beats_lint import load_beats

def decodable(path: Path, tmpdir: Path) -> Path:
    """Wan-fast writes animated WEBP under an .mp4 name; ffmpeg/cv2 can't read it.
    Sniff the content and transcode via PIL (which sniffs, not trusts, extensions)."""
    with path.open("rb") as f:
        head = f.read(16)
    if not (head[:4] == b"RIFF" and head[8:12] == b"WEBP"):
        return path
    import imageio
    from PIL import Image, ImageSequence
    dst = tmpdir / f"{path.stem}_dec.mp4"
    with Image.open(path) as im:
        durations = []
        frames = []
        for frame in ImageSequence.Iterator(im):
            durations.append(frame.info.get("duration", 62))
            frames.append(np.asarray(frame.convert("RGB")))
    fps = round(1000.0 / (sum(durations) / len(durations)), 2) if durations else 16
    w = imageio.get_writer(str(dst), fps=fps, codec="libx264", quality=8,
                           macro_block_size=1)
    try:
        for fr in frames:
            w.append_data(fr)
    finally:
        w.close()
    return dst


SAMPLE_FPS = 6.0
MIN_GAP_S = 1.2        # a return within ~a second is ambient shimmer, not a loop
SIM_MAX_DIFF = 0.035   # mean abs pixel diff (0-1) to call two frames "the same state"
EXCURSION_MIN = 0.10   # how far the clip must travel in between to call it an action
VLM_FRAMES = 4


def _thumbs(path: Path, sample_fps: float = SAMPLE_FPS) -> tuple[np.ndarray, float]:
    """(N, H, W) float32 grayscale thumbnails sampled at ~sample_fps."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    step = max(1, round(fps / sample_fps))
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(g, (48, 27)).astype(np.float32) / 255.0)
        i += 1
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 27, 48), np.float32), fps / step


def loop_events(path: Path) -> list[dict]:
    t, eff_fps = _thumbs(path)
    n = len(t)
    if n < int(MIN_GAP_S * eff_fps) + 2:
        return []
    diff = np.abs(t[:, None] - t[None]).mean(axis=(2, 3))   # (n, n) mean abs diff
    min_gap = int(MIN_GAP_S * eff_fps)
    events = []
    for i in range(n):
        for j in range(i + min_gap, n):
            if diff[i, j] > SIM_MAX_DIFF:
                continue
            excursion = diff[i, i:j + 1].max()
            if excursion >= EXCURSION_MIN:
                events.append({"t1": round(i / eff_fps, 2), "t2": round(j / eff_fps, 2),
                               "return_diff": round(float(diff[i, j]), 4),
                               "excursion": round(float(excursion), 4)})
    # keep the strongest event per return-point second (the pair grid is dense)
    best: dict[int, dict] = {}
    for e in events:
        k = int(e["t2"])
        if k not in best or e["excursion"] > best[k]["excursion"]:
            best[k] = e
    return sorted(best.values(), key=lambda e: -e["excursion"])[:5]


def reversal_event(path: Path) -> dict | None:
    """Boomerang detector (seq 48 item 4): a played-backwards segment MIRRORS its
    frames around a turn point - frame[T+k] ~= frame[T-k]. Calibration on the 001R
    frontier set (21 human-approved clips): the giver's hand returning is NATURAL
    mirror motion even in a real handoff, so mirror-ratio alone false-positives on
    half the healthy set (ratios 0.31-0.59). Fires only on BOTH conditions:
    strong mirror (ratio < 0.45) AND the end state returning to the start state
    after real excursion (the exchange un-happened). Surfaced as WARN for human
    eyeball - like departure-return, it needs labeled boomerangs before it can
    drive auto-reroll."""
    t, eff_fps = _thumbs(path)
    if len(t) < 8:                 # short clip (b-roll spans): resample denser
        t, eff_fps = _thumbs(path, sample_fps=8.0)
    n = len(t)
    if n < 8:
        return None
    baseline = float(np.abs(np.diff(t, axis=0)).mean())
    if baseline < 1e-4:            # static clip - freeze checks own this
        return None
    best = None
    for T in range(n // 4, 3 * n // 4):
        K = min(T, n - 1 - T, int(2 * eff_fps))
        if K < 3:
            continue
        m = float(np.mean([np.abs(t[T + k] - t[T - k]).mean()
                           for k in range(1, K + 1)]))
        if best is None or m < best[1]:
            best = (T, m, K)
    T, m, K = best
    ratio = m / (baseline * K / 2 + 1e-6)
    # end-state return: last second looks like the first second again after motion
    e = max(1, int(eff_fps))
    end_return = float(np.abs(t[:e] - t[-e:]).mean())
    excursion = float(np.abs(t - t[0]).mean(axis=(1, 2)).max())
    if ratio < 0.45 and end_return < 0.35 * excursion and excursion > 0.02:
        return {"turn_s": round(T / eff_fps, 2), "mirror_diff": round(m, 4),
                "ratio": round(ratio, 3), "end_return": round(end_return, 4),
                "excursion": round(excursion, 4)}
    return None


# interaction wording: reversal check runs on these (hand-object exchanges are
# where boomerangs read as broken physics - the 001R plate-rewind class)
INTERACTION = ["passes", "passing", "hands", "handing", "hand to", "places",
               "sets down", "pushes", "presses", "pours", "gives", "takes",
               "reaches", "flips"]


# Departure class (e.g. a rocket launch): a motion prompt that DEMANDS one-way exit.
# Statistical loop detection can't tell "rocket settles back" from "lamplight
# breathes" (probed 2026-07-03: designed light cycles out-score real ping-pongs on
# every appearance metric) - but gated on departure wording, end-vs-start patch
# similarity is unambiguous: the subject's patch must NOT look like frame one again.
DEPARTURE = ["lifts off", "lifting off", "liftoff", "launches", "takes off",
             "taking off", "exits the frame", "exits the top", "leaves the frame",
             "drives away", "flies away", "walks away", "rides away", "sails away",
             "rocket climbs", "ascends"]
GRID_R, GRID_C = 3, 4
DEPART_RETURN_MAX = 0.05     # end state ~= start state at patch level
DEPART_EXCURSION_MIN = 0.12  # after the subject really moved through the patch
                             # (calibrated: rocket patch ret 0.033 / exc 0.177)


def departure_event(t: np.ndarray) -> dict | None:
    """Pure check over (N,H,W) thumbs: any patch whose END state returns to its
    START state after a real excursion - the subject came back instead of leaving."""
    n = len(t)
    if n < 4:
        return None
    ph, pw = t.shape[1] // GRID_R, t.shape[2] // GRID_C
    worst = None
    for r in range(GRID_R):
        for c in range(GRID_C):
            p = t[:, r * ph:(r + 1) * ph, c * pw:(c + 1) * pw]
            d0 = np.abs(p - p[0]).mean(axis=(1, 2))     # distance from frame one
            ret, exc = float(d0[-1]), float(d0.max())
            if ret <= DEPART_RETURN_MAX and exc >= DEPART_EXCURSION_MIN:
                if worst is None or exc > worst["excursion"]:
                    worst = {"patch": [r, c], "return_diff": round(ret, 4),
                             "excursion": round(exc, 4)}
    return worst


def _vlm_check(path: Path, motion: str, style: str) -> list[dict]:
    from . import vlm
    p = gate_a.probe(path)
    dur = p["duration"] or 5.0
    findings = []
    with tempfile.TemporaryDirectory() as td:
        cap = cv2.VideoCapture(str(path))
        paths = []
        for k in range(VLM_FRAMES):
            cap.set(cv2.CAP_PROP_POS_MSEC, (dur * 1000.0) * (k + 0.5) / VLM_FRAMES)
            ok, frame = cap.read()
            if not ok:
                continue
            fp = Path(td) / f"f{k}.jpg"
            cv2.imwrite(str(fp), frame)
            paths.append(fp)
        cap.release()
        if not paths:
            return findings
        v = vlm.judge(paths, (
            "These are frames sampled in order from ONE AI-generated video clip. "
            "Judge motion plausibility across them. Report issues:\n"
            "- physics: objects moving that should not (a parked vehicle drifting or "
            "flying, floating objects), impossible motion\n"
            "- anatomy: limbs/hands/faces deforming or morphing between frames\n"
            "- other: subject teleporting, background warping, identity changing\n"
            f"INTENDED MOTION: {motion}\nEPISODE LOOK: {style}\n"
            "Score scene_match 0-10 for how well the frames show the intended motion. "
            "Only report what you actually see across the frames."))
    for iss in v["issues"]:
        findings.append({"rule": f"vlm-{iss['type']}", "severity": iss["severity"],
                         "message": iss["detail"], "fix_action": "reroll"})
    if v["scene_match"] < 4:
        findings.append({"rule": "motion-mismatch", "severity": "warn",
                         "message": f"VLM motion match {v['scene_match']}/10: {v['summary']}",
                         "fix_action": "reroll"})
    return findings


def check_clip(path: Path, motion: str = "", style: str = "",
               use_vlm: bool = True) -> dict:
    with tempfile.TemporaryDirectory() as td:
        return _check_clip(path, decodable(path, Path(td)), motion, style, use_vlm)


def _check_clip(orig: Path, path: Path, motion: str, style: str,
                use_vlm: bool) -> dict:
    findings = []
    for f in gate_a.freezes(path):
        findings.append({"rule": "freeze", "severity": "error",
                         "message": f"freeze {f['start']}s->{f['end']}s "
                                    f"({f['duration']}s, {f['severity']})",
                         "fix_action": "reroll"})
    loops = loop_events(path)
    if loops:
        e = loops[0]
        findings.append({"rule": "action-loop", "severity": "error",
                         "message": f"clip returns to its {e['t1']}s state at "
                                    f"{e['t2']}s after real motion (excursion "
                                    f"{e['excursion']}) - the jar/plate ping-pong "
                                    f"class ({len(loops)} return points)",
                         "fix_action": "reroll",
                         "loops": loops})
    if any(w in motion.lower() for w in INTERACTION):
        rev = reversal_event(path)
        if rev:
            findings.append({"rule": "boomerang-reversal", "severity": "warn",
                             "message": f"interaction clip may play back on itself "
                                        f"- mirror point {rev['turn_s']}s (ratio "
                                        f"{rev['ratio']}) AND end state returns to "
                                        f"start (return {rev['end_return']} after "
                                        f"excursion {rev['excursion']}) - the "
                                        f"plate-rewind class; EYEBALL the exchange "
                                        f"direction",
                             "fix_action": "reroll", "reversal": rev})
    if any(w in motion.lower() for w in DEPARTURE):
        t, _ = _thumbs(path)
        d = departure_event(t)
        if d:
            # WARN, not error: a transit patch (subject passes through and exits)
            # also returns to start, so this can fire on a GOOD exit. On the
            # labeled set (eps 004+005) it flags exactly the bad rocket; needs
            # more labeled departures before it can drive auto-reroll.
            findings.append({"rule": "departure-return", "severity": "warn",
                             "message": f"motion demands a one-way exit but patch "
                                        f"{d['patch']} ends back at its starting "
                                        f"state (return {d['return_diff']} after "
                                        f"excursion {d['excursion']}) - the rocket "
                                        f"up-and-down class",
                             "fix_action": "reroll"})
    if use_vlm:
        findings += _vlm_check(path, motion, style)
    errors = [f for f in findings if f["severity"] == "error"]
    return {"clip": orig.stem, "verdict": "fail" if errors else "pass",
            "fix_action": "reroll" if errors else None, "findings": findings}


def run(story: Path, clip_glob: str = "clip_*.mp4", only: int | None = None,
        use_vlm: bool = True) -> dict:
    b = load_beats(story)
    meta = {x["i"]: x for x in b.get("beats", []) + b.get("inserts", [])}
    clips = []
    for c in sorted((story / "assets").glob(clip_glob)):
        try:
            i = int(c.stem.split("_")[-1])
        except ValueError:
            i = None
        if only is not None and i != only:
            continue
        motion = meta.get(i, {}).get("motion", "")
        r = check_clip(c, motion, b.get("style", ""), use_vlm=use_vlm)
        clips.append(r)
        print(f"{c.stem}: {r['verdict'].upper()}", flush=True)
        for f in r["findings"]:
            print(f"  {f['severity'].upper():5} [{f['rule']}] {f['message']}", flush=True)
    report = {"story": story.name, "stage": "clip-qc", "vlm": use_vlm,
              "checked": len(clips),
              "failed": sum(1 for c in clips if c["verdict"] == "fail"),
              "verdict": "fail" if any(c["verdict"] == "fail" for c in clips) else "pass",
              "clips": clips}
    qc = story / "qc"
    qc.mkdir(exist_ok=True)
    (qc / "clips.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [f"# clip-qc - {story.name}",
             f"**{report['verdict'].upper()}** - {report['failed']}/{report['checked']} "
             f"clips failed (VLM {'on' if use_vlm else 'OFF'})", ""]
    for c in clips:
        mark = "✅" if c["verdict"] == "pass" else "❌"
        lines.append(f"- {mark} {c['clip']}")
        for f in c["findings"]:
            lines.append(f"  - {f['severity']}: [{f['rule']}] {f['message']}")
    (qc / "clips.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", required=True)
    ap.add_argument("--clip-glob", default="clip_*.mp4")
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--no-vlm", action="store_true")
    args = ap.parse_args()
    report = run(Path(args.story), args.clip_glob, only=args.only,
                 use_vlm=not args.no_vlm)
    print(f"\nclip-qc: {report['verdict'].upper()} "
          f"({report['failed']}/{report['checked']} failed)")
    sys.exit(1 if report["verdict"] == "fail" else 0)


if __name__ == "__main__":
    main()
