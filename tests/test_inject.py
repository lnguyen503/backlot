import pytest
from pathlib import Path

from backlot.engine.config import load_config
from backlot.engine.inject import InjectError, assemble_api
from backlot.engine.registry import Registry

CFG = str(Path(__file__).resolve().parents[1] / "config/engine.yaml")


@pytest.fixture
def cap():
    return Registry.load(load_config(CFG)).get("txt2img_sdxl")


def test_required_param_enforced(cap):
    with pytest.raises(InjectError):
        assemble_api(cap, {})


def test_injects_prompt_and_defaults(cap):
    graph, resolved = assemble_api(cap, {"positive_prompt": "a red cube"})
    assert graph["6"]["inputs"]["text"] == "a red cube"
    assert graph["3"]["inputs"]["steps"] == 30
    assert isinstance(graph["3"]["inputs"]["seed"], int)
    assert graph["3"]["inputs"]["seed"] >= 0
    assert resolved["positive_prompt"] == "a red cube"


def test_seed_randomized_when_negative(cap):
    g1, _ = assemble_api(cap, {"positive_prompt": "x", "seed": -1})
    g2, _ = assemble_api(cap, {"positive_prompt": "x", "seed": -1})
    assert g1["3"]["inputs"]["seed"] != g2["3"]["inputs"]["seed"]


def test_seed_passthrough(cap):
    g, _ = assemble_api(cap, {"positive_prompt": "x", "seed": 12345})
    assert g["3"]["inputs"]["seed"] == 12345


def test_bounds_enforced(cap):
    with pytest.raises(InjectError):
        assemble_api(cap, {"positive_prompt": "x", "steps": 9999})


def test_template_not_mutated(cap):
    assemble_api(cap, {"positive_prompt": "mutate?", "seed": 1})
    assert cap.api_graph["6"]["inputs"]["text"] == "a photograph"


def test_talkhost_injects_image_and_audio():
    th = Registry.load(load_config(CFG)).get("talkhost_sonic")
    graph, resolved = assemble_api(th, {"image": "face.png", "audio": "line.wav"})
    assert graph["3"]["inputs"]["image"] == "face.png"
    assert graph["4"]["inputs"]["audio"] == "line.wav"
    assert graph["5"]["inputs"]["min_resolution"] == 768   # default applied
    assert isinstance(graph["6"]["inputs"]["seed"], int) and graph["6"]["inputs"]["seed"] >= 0
    assert resolved["audio"] == "line.wav"


def test_talkhost_requires_image_and_audio():
    th = Registry.load(load_config(CFG)).get("talkhost_sonic")
    with pytest.raises(InjectError):
        assemble_api(th, {"image": "face.png"})  # missing audio


def test_talkhost_random_seed_respects_int32_max():
    th = Registry.load(load_config(CFG)).get("talkhost_sonic")
    for _ in range(50):  # randomized; SONICSampler caps seed at int32
        graph, _ = assemble_api(th, {"image": "f.png", "audio": "a.wav", "seed": -1})
        assert 0 <= graph["6"]["inputs"]["seed"] <= 2147483647
