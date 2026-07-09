import pytest
from pathlib import Path

from backlot.engine.config import load_config
from backlot.engine.registry import Registry, RegistryError

CFG = str(Path(__file__).resolve().parents[1] / "config/engine.yaml")


def _reg():
    return Registry.load(load_config(CFG))


def test_loads_txt2img():
    cap = _reg().get("txt2img_sdxl")
    assert cap.kind == "image"
    assert "9" in cap.client_outputs
    assert cap.api_graph["4"]["class_type"] == "CheckpointLoaderSimple"


def test_unknown_raises():
    with pytest.raises(RegistryError):
        _reg().get("nope")


def test_params_public_view():
    cap = _reg().get("txt2img_sdxl")
    names = {p.name for p in cap.params()}
    assert "positive_prompt" in names
    assert cap.public_info()["name"] == "txt2img_sdxl"


def test_loads_talkhost_sonic():
    cap = _reg().get("talkhost_sonic")
    assert cap.kind == "video"
    assert "8" in cap.client_outputs
    assert cap.api_graph["6"]["class_type"] == "SONICSampler"
    names = {p.name for p in cap.params()}
    assert {"image", "audio", "min_resolution", "dynamic_scale"} <= names


def test_loads_upscale_esrgan():
    cap = _reg().get("upscale_esrgan")
    assert cap.kind == "edit"
    assert "4" in cap.client_outputs
    assert cap.api_graph["2"]["class_type"] == "UpscaleModelLoader"
    assert cap.api_graph["3"]["class_type"] == "ImageUpscaleWithModel"
    names = {p.name for p in cap.params()}
    assert {"image", "scale_by", "model_name"} <= names


def test_loads_inpaint_sdxl():
    cap = _reg().get("inpaint_sdxl")
    assert cap.kind == "edit"
    assert "10" in cap.client_outputs
    assert cap.api_graph["7"]["class_type"] == "SetLatentNoiseMask"
    assert cap.api_graph["2"]["class_type"] == "LoadImageMask"
    names = {p.name for p in cap.params()}
    assert {"image", "mask", "prompt", "denoise"} <= names
