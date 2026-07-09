"""QC stage 1 — beats-lint: score beats.json against the known AI-video failure
classes BEFORE rendering (pure code, $0). Most human-caught defects are
authoring failures this blocks.

  .venv\\Scripts\\python -m backlot.qc.beats_lint --story <path-to-story-dir>

Writes <story>/qc/lint.json + lint.md. Exit 1 if any ERROR-severity finding —
the producer must not advance to the stills stage until the beats pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .rules import lint_beats


def load_beats(story: Path) -> dict:
    b = json.loads((story / "assets" / "beats.json").read_text(encoding="utf-8"))
    style = b.get("style", "")
    for item in b.get("beats", []) + b.get("inserts", []):
        if "still" in item:
            item["still"] = item["still"].replace("{style}", style).replace(
                "{kontext}", style)
    return b


def run(story: Path) -> dict:
    findings = lint_beats(load_beats(story))
    errors = [f for f in findings if f["severity"] == "error"]
    warns = [f for f in findings if f["severity"] == "warn"]
    report = {"story": story.name, "stage": "beats-lint",
              "errors": len(errors), "warnings": len(warns),
              "verdict": "fail" if errors else "pass", "findings": findings}
    qc = story / "qc"
    qc.mkdir(exist_ok=True)
    (qc / "lint.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [f"# beats-lint — {story.name}",
             f"**{report['verdict'].upper()}** — {len(errors)} errors, "
             f"{len(warns)} warnings", ""]
    for f in findings:
        mark = "❌" if f["severity"] == "error" else "⚠️"
        lines += [f"- {mark} **{f['loc']}** [{f['rule']}] {f['message']}",
                  f"  - fix: {f['fix']}"]
    (qc / "lint.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", required=True)
    args = ap.parse_args()
    report = run(Path(args.story))
    for f in report["findings"]:
        print(f"{f['severity'].upper():5} {f['loc']:20} [{f['rule']}] {f['message']}")
    print(f"\nbeats-lint: {report['verdict'].upper()} "
          f"({report['errors']} errors, {report['warnings']} warnings)")
    sys.exit(1 if report["verdict"] == "fail" else 0)


if __name__ == "__main__":
    main()
