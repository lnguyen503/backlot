"""Face-identity checks for stills/frames (insightface buffalo_l embeddings).

Two defect classes from the slate:
- CLONES: "a crowd" renders the same face N times -> any two
  faces in ONE frame with cosine similarity >= CLONE_SIM are clones.
- WRONG CHARACTER: a Kontext-derived still whose face no
  longer matches the hero master ref -> best cosine vs ref < HERO_SIM.

Import note: backlot.faceswap must be imported FIRST (it bootstraps the CUDA
DLL path before onnxruntime loads); we reuse its providers().
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..faceswap import _HAS_CUDA, providers  # noqa: F401  (CUDA bootstrap side effect)

CLONE_SIM = 0.60     # same-identity pairs score ~0.7+; distinct strangers < 0.4
HERO_SIM = 0.35      # Kontext-derived appearances of the SAME person score well above
MIN_FACE = 48        # px; ignore tiny background faces (crowd texture is fine)


@lru_cache(maxsize=1)
def _app():
    import cv2  # noqa: F401
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=providers())
    app.prepare(ctx_id=0 if _HAS_CUDA else -1, det_size=(640, 640))
    return app


def embeddings(image: Path) -> list[dict]:
    """[{bbox, size, emb}] for each detected face big enough to matter."""
    import cv2
    faces = _app().get(cv2.imread(str(image)))
    out = []
    for f in faces:
        w, h = f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]
        if min(w, h) >= MIN_FACE:
            out.append({"bbox": [round(float(x), 1) for x in f.bbox],
                        "size": round(float(min(w, h)), 1),
                        "emb": f.normed_embedding})
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def check_faces(image: Path, hero_ref: Path | None = None,
                expect_hero: bool = False) -> list[dict]:
    findings = []
    faces = embeddings(image)
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            sim = _cos(faces[i]["emb"], faces[j]["emb"])
            if sim >= CLONE_SIM:
                findings.append({
                    "rule": "clones", "severity": "error",
                    "message": f"two faces in frame are the same identity "
                               f"(cos {sim:.2f}) - the crowd-clones class",
                    "fix_action": "regen"})
                break
    if expect_hero and hero_ref is not None:
        ref = embeddings(hero_ref)
        if not ref:
            findings.append({"rule": "hero-ref", "severity": "warn",
                             "message": f"no face found in hero ref {hero_ref.name}",
                             "fix_action": "human"})
        elif not faces:
            findings.append({"rule": "wrong-character", "severity": "error",
                             "message": "no face detected in a hero shot",
                             "fix_action": "regen"})
        else:
            best = max(_cos(ref[0]["emb"], f["emb"]) for f in faces)
            if best < HERO_SIM:
                findings.append({
                    "rule": "wrong-character", "severity": "error",
                    "message": f"no face matches the hero ref (best cos {best:.2f} "
                               f"< {HERO_SIM}) - the wrong-character class",
                    "fix_action": "regen"})
    return findings
