"""QC stage 2 - STILL-QC: analyze every rendered still against its beat BEFORE
animating (fix here = $0; i2v inherits every still defect and worsens it).

  .venv\\Scripts\\python -m backlot.qc.still_qc --story <dir> [--only N] [--no-vlm]

Per still: OCR vs expected quoted text -> face checks (clones + hero identity)
-> VLM judgment (scene-match, spatial logic, physics/floaters, style,
cross-episode contamination). Writes <story>/qc/stills.json + stills.md.
Every finding carries a fix_action: 'kontext_repair' (text garble -> repair the
STILL), 'regen' (seed bump / re-generate), or 'human'.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .beats_lint import load_beats
from .rules import _quoted

VLM_FAIL_BELOW = 6  # scene_match under this = wrong scene -> regen

# held/viewed-object trigger for the ORIENTATION check (the backwards-Polaroid
# class): a viewing verb + a front-faced object in the same beat text
_VIEW_WORDS = ("hold", "holds", "holding", "held", "looks at", "looking at",
               "gazes at", "gazing at", "stares at", "staring at")
# NOT "reads/reading": the sign-caption idiom ("a sign reading 'X'") false-triggers
# (001R insert 18) - display-to-camera signs are not viewed objects
_FACED_OBJECTS = ("photo", "photograph", "picture", "framed", "letter", "note",
                  "card", "book", "phone", "portrait")


def _needs_orientation(text: str) -> bool:
    t = text.lower()
    return any(v in t for v in _VIEW_WORDS) and any(o in t for o in _FACED_OBJECTS)


# crowd-context trigger (the 001R un-soccer fans class)
_CROWD_WORDS = ("crowd", "fans", "supporters", "queue", "queuing", "line of",
                "townspeople", "spectators", "audience")


def _needs_crowd_context(text: str) -> bool:
    return any(w in text.lower() for w in _CROWD_WORDS)


def _intended(item: dict) -> str:
    return item.get("kontext") or item.get("still", "")


def _waivers(story: Path) -> dict:
    """qc/waivers.json: {"shot_14": ["text-missing-or-garbled"], ...} - rules a HUMAN
    verified by eye on a specific shot (OCR/VLM false positives). Waived findings
    downgrade to warn with provenance instead of failing the shot forever."""
    p = story / "qc" / "waivers.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def check_one(story: Path, b: dict, item: dict, use_vlm: bool = True) -> dict:
    from . import faceid, ocr
    a = story / "assets"
    png = a / f"shot_{item['i']:02d}.png"
    findings = []
    expected = _quoted(item.get("still", "")) + _quoted(item.get("kontext", ""))
    findings += ocr.check_still_text(png, expected)
    hero = a / f"{b['hero']['name']}.png" if b.get("hero") else None
    findings += faceid.check_faces(png, hero_ref=hero,
                                   expect_hero=bool(item.get("kontext")))
    if use_vlm:
        from . import vlm
        v = vlm.judge([png], vlm.still_prompt(_intended(item), b.get("style", "")))
        if v["scene_match"] < VLM_FAIL_BELOW:
            findings.append({"rule": "scene-mismatch", "severity": "error",
                             "message": f"VLM scene_match {v['scene_match']}/10: "
                                        f"{v['summary']}",
                             "fix_action": "regen"})
        for iss in v["issues"]:
            findings.append({"rule": f"vlm-{iss['type']}", "severity": iss["severity"],
                             "message": iss["detail"],
                             "fix_action": "kontext_repair"
                             if iss["type"] == "text" else "regen"})
        # blocking CROWD-CONTEXT check: background people must read as THIS event's
        # attendees (001R un-soccer fans - context drift survives any model tier)
        if _needs_crowd_context(_intended(item)):
            c = vlm.judge([png], vlm.crowd_prompt(_intended(item), b.get("style", "")))
            if c["scene_match"] < VLM_FAIL_BELOW:
                findings.append({"rule": "crowd-context", "severity": "error",
                                 "message": f"crowd reads as generic extras, not "
                                            f"the event's attendees "
                                            f"({c['scene_match']}/10): {c['summary']}",
                                 "fix_action": "regen"})
        # blocking ORIENTATION check (who-sees-what): held/viewed objects must face
        # the holder, not the camera (009 backwards-Polaroid class, recurred 001R s12)
        if _needs_orientation(_intended(item)):
            o = vlm.judge([png], vlm.orientation_prompt(_intended(item)))
            if o["scene_match"] < VLM_FAIL_BELOW:
                findings.append({"rule": "orientation", "severity": "error",
                                 "message": f"held/viewed object faces the camera, "
                                            f"not the holder ({o['scene_match']}/10): "
                                            f"{o['summary']}",
                                 "fix_action": "regen"})
    waived = _waivers(story).get(f"shot_{item['i']:02d}", [])
    for f in findings:
        if f["rule"] in waived and f["severity"] == "error":
            f["severity"] = "warn"
            f["message"] += " [WAIVED: human-verified]"
            f["fix_action"] = None
    errors = [f for f in findings if f["severity"] == "error"]
    # text repairs first (cheaper + regen would just re-roll the same garble class)
    action = None
    if errors:
        action = ("kontext_repair" if any(f["fix_action"] == "kontext_repair"
                                          for f in errors) else errors[0]["fix_action"])
    return {"shot": item["i"], "file": png.name,
            "verdict": "fail" if errors else "pass",
            "fix_action": action, "findings": findings}


def run(story: Path, only: int | None = None, use_vlm: bool = True) -> dict:
    b = load_beats(story)
    a = story / "assets"
    shots = []
    for item in b.get("beats", []) + b.get("inserts", []):
        if only is not None and item["i"] != only:
            continue
        if not (a / f"shot_{item['i']:02d}.png").exists():
            continue
        r = check_one(story, b, item, use_vlm=use_vlm)
        shots.append(r)
        print(f"shot_{item['i']:02d}: {r['verdict'].upper()}"
              + (f" -> {r['fix_action']}" if r["fix_action"] else ""), flush=True)
        for f in r["findings"]:
            print(f"  {f['severity'].upper():5} [{f['rule']}] {f['message']}", flush=True)
    report = {"story": story.name, "stage": "still-qc", "vlm": use_vlm,
              "checked": len(shots),
              "failed": sum(1 for s in shots if s["verdict"] == "fail"),
              "verdict": "fail" if any(s["verdict"] == "fail" for s in shots) else "pass",
              "shots": shots}
    qc = story / "qc"
    qc.mkdir(exist_ok=True)
    (qc / "stills.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [f"# still-qc - {story.name}",
             f"**{report['verdict'].upper()}** - {report['failed']}/{report['checked']} "
             f"stills failed (VLM {'on' if use_vlm else 'OFF'})", ""]
    for s in shots:
        mark = "✅" if s["verdict"] == "pass" else "❌"
        lines.append(f"- {mark} shot_{s['shot']:02d}"
                     + (f" -> **{s['fix_action']}**" if s["fix_action"] else ""))
        for f in s["findings"]:
            lines.append(f"  - {f['severity']}: [{f['rule']}] {f['message']}")
    (qc / "stills.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", required=True)
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--no-vlm", action="store_true")
    args = ap.parse_args()
    report = run(Path(args.story), only=args.only, use_vlm=not args.no_vlm)
    print(f"\nstill-qc: {report['verdict'].upper()} "
          f"({report['failed']}/{report['checked']} failed)")
    sys.exit(1 if report["verdict"] == "fail" else 0)


if __name__ == "__main__":
    main()
