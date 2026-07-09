"""Pre-frontier identity gate (retro upgrade A, relay seq 30) - $0 machine check that
turns the 011 identity-drift class into a blocking gate before money renders.

  .venv\\Scripts\\python -m backlot.qc.identity_gate --story <dir> [--strict]

For every hero-bearing still (kontext-derived beats, minus wrong-character-waived ones):
best insightface cosine vs the hero ref, three tiers:
  FAIL   < 0.35  (the existing wrong-person floor)
  REVIEW < 0.60  (calibration 2026-07-04: shipped-fine 010 stills score 0.37-0.62 at odd
                  pose/lighting, while disciplined 011 stills score 0.69-0.90 - a binary
                  gate here would fail published frames; REVIEW = ranked human triage)
  PASS  >= 0.60
Exit 1 on FAIL; --strict also fails REVIEW (pre-frontier blocking mode).
Writes qc/identity.json + identity.md with the full ranked table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FAIL_SIM = 0.35
REVIEW_SIM = 0.60


def run(story: Path, strict: bool = False) -> dict:
    from . import faceid
    b = json.loads((story / "assets" / "beats.json").read_text(encoding="utf-8"))
    hero_name = b["hero"]["name"]
    ref_path = story / "assets" / f"{hero_name}.png"
    waivers = {}
    wp = story / "qc" / "waivers.json"
    if wp.exists():
        waivers = json.loads(wp.read_text(encoding="utf-8"))

    ref = faceid.embeddings(ref_path)
    if not ref:
        raise SystemExit(f"no face in hero ref {ref_path}")

    rows = []
    for item in b.get("beats", []):
        if not item.get("kontext"):
            continue
        i = item["i"]
        if "wrong-character" in waivers.get(f"shot_{i:02d}", []):
            rows.append({"shot": i, "cos": None, "tier": "waived"})
            continue
        png = story / "assets" / f"shot_{i:02d}.png"
        if not png.exists():
            continue
        faces = faceid.embeddings(png)
        if not faces:
            rows.append({"shot": i, "cos": None, "tier": "fail",
                         "note": "no face detected in hero shot"})
            continue
        best = max(float(np.dot(ref[0]["emb"], f["emb"])) for f in faces)
        tier = "fail" if best < FAIL_SIM else ("review" if best < REVIEW_SIM else "pass")
        rows.append({"shot": i, "cos": round(best, 3), "tier": tier})

    rows.sort(key=lambda r: (r["cos"] is None, r["cos"] if r["cos"] is not None else 0))
    fails = [r for r in rows if r["tier"] == "fail"]
    reviews = [r for r in rows if r["tier"] == "review"]
    verdict = "fail" if fails or (strict and reviews) else \
        ("review" if reviews else "pass")
    report = {"story": story.name, "hero_ref": ref_path.name, "strict": strict,
              "thresholds": {"fail": FAIL_SIM, "review": REVIEW_SIM},
              "verdict": verdict, "rows": rows}
    qc = story / "qc"
    qc.mkdir(exist_ok=True)
    (qc / "identity.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [f"# identity-gate — {story.name} (ref: {ref_path.name})",
             f"**{verdict.upper()}** — {len(fails)} fail, {len(reviews)} review, "
             f"strict={strict}", ""]
    for r in rows:
        c = "waived" if r["tier"] == "waived" else \
            (f"{r['cos']:.3f}" if r["cos"] is not None else "no-face")
        lines.append(f"- shot_{r['shot']:02d}: {c} -> {r['tier'].upper()}"
                     + (f" ({r.get('note')})" if r.get("note") else ""))
    (qc / "identity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", required=True)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    report = run(Path(args.story), strict=args.strict)
    for r in report["rows"]:
        c = f"{r['cos']:.3f}" if r["cos"] is not None else "-"
        print(f"shot_{r['shot']:02d}: {c} {r['tier'].upper()}")
    print(f"identity-gate: {report['verdict'].upper()}")
    if report["verdict"] == "fail":
        sys.exit(1)


if __name__ == "__main__":
    main()
