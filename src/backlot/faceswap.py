"""Identity face-swap + restore as a reusable engine component (torch-free).

Locks ONE identity onto rendered video frames: insightface inswapper_128 swaps
a reference face, then a GFPGAN ONNX restore (FFHQ-512 align + feathered paste)
sharpens it and seats it into the frame. Runs on GPU (onnxruntime-gpu CUDA) when
available, else CPU.

CUDA note: onnxruntime-gpu's provider DLLs depend on the NVIDIA cu12 pip wheels;
on Windows those bin dirs must be on PATH BEFORE onnxruntime is imported. This
module does that at import time, so it must be imported before onnxruntime.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

# --- CUDA bootstrap (must run before onnxruntime import) -------------------
_HAS_CUDA = False
try:
    import nvidia  # from nvidia-*-cu12 wheels

    _bins = glob.glob(os.path.join(os.path.dirname(nvidia.__file__), "*", "bin"))
    if _bins:
        os.environ["PATH"] = os.pathsep.join(_bins) + os.pathsep + os.environ.get("PATH", "")
        for _d in _bins:
            try:
                os.add_dll_directory(_d)   # Windows-only API
            except (OSError, AttributeError):
                pass
        _HAS_CUDA = True
except ImportError:
    pass

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from insightface import model_zoo  # noqa: E402
from insightface.app import FaceAnalysis  # noqa: E402

# Standard FFHQ-512 5-point template (GFPGAN/facefusion alignment).
_FFHQ_512 = np.array([
    [192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
    [201.26117, 371.41043], [313.08905, 371.15118],
], dtype=np.float32)


def providers():
    """CUDA+CPU when the CUDA stack is present, else CPU only (safe to pass both)."""
    if _HAS_CUDA and "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class FaceLocker:
    """Swap a reference identity onto frames and restore the face."""

    def __init__(self, ref_image: str, inswapper_path: str, gfpgan_path: str | None = None):
        prov = providers()
        self.app = FaceAnalysis(name="buffalo_l", providers=prov)
        self.app.prepare(ctx_id=0 if _HAS_CUDA else -1, det_size=(640, 640))
        self.swapper = model_zoo.get_model(inswapper_path, providers=prov)
        self.gfpgan = ort.InferenceSession(gfpgan_path, providers=prov) if gfpgan_path else None
        if self.gfpgan:
            self._gi = self.gfpgan.get_inputs()[0].name
            self._go = self.gfpgan.get_outputs()[0].name
        faces = self.app.get(cv2.imread(ref_image))
        if not faces:
            raise ValueError(f"No face found in reference image: {ref_image}")
        self.ref_face = faces[0]

    def _restore(self, frame, kps):
        M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, np.float32), _FFHQ_512, method=cv2.LMEDS)
        if M is None:
            return frame
        aligned = cv2.warpAffine(frame, M, (512, 512), borderMode=cv2.BORDER_REFLECT)
        inp = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose((inp - 0.5) / 0.5, (2, 0, 1))[None]
        out = self.gfpgan.run([self._go], {self._gi: inp})[0][0]
        out = np.clip(np.transpose(out, (1, 2, 0)) * 0.5 + 0.5, 0, 1) * 255.0
        restored = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)
        Minv = cv2.invertAffineTransform(M)
        h, w = frame.shape[:2]
        back = cv2.warpAffine(restored, Minv, (w, h), borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(np.full((512, 512), 255, np.uint8), Minv, (w, h))
        mask = cv2.GaussianBlur(cv2.erode(mask, np.ones((15, 15), np.uint8)), (0, 0), 7)
        mask = mask.astype(np.float32)[:, :, None] / 255.0
        return (back * mask + frame * (1 - mask)).astype(np.uint8)

    def process_frame(self, frame_bgr):
        """Swap ref identity onto every face, then restore. Returns (frame, had_face)."""
        faces = self.app.get(frame_bgr)
        for f in faces:
            frame_bgr = self.swapper.get(frame_bgr, f, self.ref_face, paste_back=True)
        if self.gfpgan:
            for f in self.app.get(frame_bgr):
                frame_bgr = self._restore(frame_bgr, f.kps)
        return frame_bgr, bool(faces)


_DEFAULT_MODELS_DIR = str(Path(__file__).resolve().parents[2] / "models")


def default_models(models_dir: str = _DEFAULT_MODELS_DIR):
    """Locate the swap + restore models, returning (inswapper, gfpgan_or_None)."""
    d = Path(models_dir)
    sw = d / "inswapper_128.onnx"
    gf = d / "GFPGANv1.4.onnx"
    return str(sw), (str(gf) if gf.exists() else None)
