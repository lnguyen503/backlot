"""OCR text check for stills/frames (RapidOCR, local, CPU).

The recurring tell across eps 001-003 (6+ garble escalations): models render
readable-but-nonsense signage. Rule: every quoted string in the prompt must
appear (fuzzy) in the frame; any OTHER confident text is a garble suspect -
prompts are authored to contain no incidental text (beats-lint enforces that).
i2v inherits still text, so a garble here routes to REPAIR THE STILL, never
a plain clip re-roll.

INSTALL GOTCHA (bit us 2026-07-03): pip-installing rapidocr-onnxruntime pulls in
plain `onnxruntime`, which OVERWRITES onnxruntime-gpu's binaries (same package
dir) and silently drops every insightface QC check to CPU (~10x slower). Fix:
  pip uninstall -y onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu==1.22
(rapidocr imports fine against the gpu build; ignore pip's dependency warning.)
"""
from __future__ import annotations

import difflib
import re
from functools import lru_cache
from pathlib import Path

MIN_CONF = 0.60
MIN_LEN = 4          # ignore tiny fragments ("NO", "24")
FUZZ = 0.72          # SequenceMatcher ratio to count as "matches expected"


@lru_cache(maxsize=1)
def _engine():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def read_text(image: Path) -> list[dict]:
    """[{text, conf}] for confident text regions in the image."""
    result, _ = _engine()(str(image))
    out = []
    for item in result or []:
        _box, text, conf = item[0], item[1], float(item[2])
        text = text.strip()
        if conf >= MIN_CONF and len(re.sub(r"[^A-Za-z0-9]", "", text)) >= MIN_LEN:
            out.append({"text": text, "conf": round(conf, 3)})
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", s.upper()).strip()


def _punct_sig(s: str) -> str:
    """Punctuation signature for the exact-string rule (relay seq 36, the LOSER'S
    apostrophe): the fuzzy matcher strips punctuation, so variants sail through.
    Compare apostrophe presence between expected and matched text explicitly."""
    return "'" if ("'" in s or "’" in s) else ""


def check_still_text(image: Path, expected: list[str]) -> list[dict]:
    """Findings for one still: missing expected text + unexpected (garble-suspect) text."""
    found = read_text(image)
    findings = []
    exp_norm = [_norm(e) for e in expected]
    joined = " ".join(_norm(f["text"]) for f in found)
    matched = set()
    for e_raw, e in zip(expected, exp_norm):
        # multi-line signage splits into regions -> also match the joined text
        best = max([difflib.SequenceMatcher(None, e, _norm(f["text"])).ratio()
                    for f in found]
                   + [difflib.SequenceMatcher(None, e, joined).ratio()], default=0.0)
        if best < FUZZ:
            findings.append({"rule": "text-missing-or-garbled", "severity": "error",
                             "message": f"expected '{e_raw}' not readable in frame "
                                        f"(best OCR match {best:.2f})",
                             "fix_action": "kontext_repair"})
        else:
            matched.add(e)
            # exact-string rule: a fuzzy match that differs in apostrophes is a
            # punctuation VARIANT of the sign text, not the sign text (LOSER'S class)
            best_f = max(found, key=lambda f: difflib.SequenceMatcher(
                None, e, _norm(f["text"])).ratio(), default=None)
            if best_f is not None and \
                    _punct_sig(best_f["text"]) != _punct_sig(e_raw) and \
                    difflib.SequenceMatcher(
                        None, e, _norm(best_f["text"])).ratio() >= FUZZ:
                findings.append({
                    "rule": "text-variant", "severity": "error",
                    "message": f"sign reads '{best_f['text']}' - a punctuation "
                               f"variant of expected '{e_raw}' (exact-string rule)",
                    "fix_action": "regen"})
    for f in found:
        n = _norm(f["text"])
        if any(difflib.SequenceMatcher(None, e, n).ratio() >= FUZZ or n in e
               for e in exp_norm):
            continue
        findings.append({"rule": "unexpected-text", "severity": "warn",
                         "message": f"OCR found text not in the prompt: "
                                    f"'{f['text']}' (conf {f['conf']}) - garble suspect",
                         "fix_action": "kontext_repair"})
    return findings
