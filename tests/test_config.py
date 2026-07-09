from pathlib import Path

from backlot.engine.config import load_config

CFG = Path(__file__).resolve().parent.parent / "config" / "engine.yaml"


def test_loads_paths_and_comfy():
    cfg = load_config(CFG)
    assert cfg.comfyui.port == 8188
    assert cfg.paths.workflows.endswith("workflows")
    # relative paths in the shipped config resolve against the repo root
    assert Path(cfg.paths.workflows).is_absolute()
    assert Path(cfg.paths.workflows).is_dir()


def test_job_timeout_by_kind():
    cfg = load_config(CFG)
    assert cfg.job_timeout_s("video") == cfg.timeouts.video_job_s
    assert cfg.job_timeout_s("image") == cfg.timeouts.image_job_s


def test_concurrency_default_off():
    assert load_config(CFG).concurrency.gpu_lease is False
