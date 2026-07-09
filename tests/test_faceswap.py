"""Tests for the face-swap engine component (no GPU/model files required).

Requires the `[faces]` extra (insightface + onnxruntime); skipped otherwise."""
from pathlib import Path

import pytest

pytest.importorskip("insightface")
pytest.importorskip("onnxruntime")

from backlot import faceswap  # noqa: E402
from backlot.faceswap import default_models, providers  # noqa: E402


def test_providers_nonempty_and_has_cpu():
    p = providers()
    assert isinstance(p, list) and p, "providers() must return a non-empty list"
    assert p[-1] == "CPUExecutionProvider", "CPU must always be the fallback provider"


def test_providers_cuda_only_when_available():
    # If CUDA was bootstrapped, it must be listed first; otherwise CPU-only.
    p = providers()
    if faceswap._HAS_CUDA and "CUDAExecutionProvider" in __import__("onnxruntime").get_available_providers():
        assert p[0] == "CUDAExecutionProvider"
    else:
        assert p == ["CPUExecutionProvider"]


def test_default_models_paths():
    inswapper, gfpgan = default_models(str(Path(__file__).resolve().parents[1] / "models"))
    assert inswapper.replace("\\", "/").endswith("models/inswapper_128.onnx")
    # gfpgan is None when the file is absent, else points at the restorer
    assert gfpgan is None or gfpgan.replace("\\", "/").endswith("models/GFPGANv1.4.onnx")


def test_ffhq_template_shape():
    assert faceswap._FFHQ_512.shape == (5, 2), "FFHQ alignment template must be 5 points (x,y)"
