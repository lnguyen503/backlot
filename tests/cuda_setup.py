"""CUDA bootstrap for onnxruntime-gpu in this venv.

onnxruntime-gpu's CUDA provider depends on the NVIDIA cu12 DLLs shipped as pip
wheels (nvidia-*-cu12). On Windows, dependent-DLL resolution needs those bin
dirs on PATH *before* onnxruntime is imported -- os.add_dll_directory alone is
not enough. Import this module FIRST (before onnxruntime / insightface), then
call providers() to get the CUDA->CPU provider list (auto-falls back to CPU).
"""
from __future__ import annotations

import glob
import os

HAS_CUDA = False
try:
    import nvidia  # provided by the nvidia-*-cu12 wheels

    _bins = glob.glob(os.path.join(os.path.dirname(nvidia.__file__), "*", "bin"))
    if _bins:
        os.environ["PATH"] = os.pathsep.join(_bins) + os.pathsep + os.environ.get("PATH", "")
        for _d in _bins:
            try:
                os.add_dll_directory(_d)
            except OSError:
                pass
        HAS_CUDA = True
except ImportError:
    pass


def providers():
    """CUDA + CPU when the CUDA stack is present, else CPU only.

    Passing both is safe: onnxruntime silently falls back to CPU per-session if
    CUDA fails to initialize.
    """
    import onnxruntime as ort

    if HAS_CUDA and "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
