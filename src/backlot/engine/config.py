"""Load and validate engine.yaml into a typed EngineConfig."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ComfyConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8188
    base_url: str = "http://127.0.0.1:8188"
    ws_url: str = "ws://127.0.0.1:8188/ws"
    install_dir: Optional[str] = None
    user_workflows_dir: Optional[str] = None
    input_dir: Optional[str] = None
    output_dir: Optional[str] = None
    models_base: Optional[str] = None


class ModelsConfig(BaseModel):
    provider: str = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    authoring_model: str = "qwen2.5:3b"
    keep_alive: str = "30m"
    options: dict = Field(default_factory=dict)


class Timeouts(BaseModel):
    http_s: float = 30
    ollama_s: float = 120
    image_job_s: float = 300
    video_job_s: float = 1800
    stall_s: float = 120
    ws_reconnect_max_s: float = 30


class Concurrency(BaseModel):
    gpu_lease: bool = False


class Paths(BaseModel):
    workflows: str
    templates: str
    pipelines: str
    runs: str


class ServerConfig(BaseModel):
    web_host: str = "127.0.0.1"
    web_port: int = 8765


class LoggingConfig(BaseModel):
    level: str = "INFO"
    dir: str = "logs"


class EngineConfig(BaseModel):
    comfyui: ComfyConfig = Field(default_factory=ComfyConfig)
    paths: Paths
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    timeouts: Timeouts = Field(default_factory=Timeouts)
    concurrency: Concurrency = Field(default_factory=Concurrency)
    server: ServerConfig = Field(default_factory=ServerConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def job_timeout_s(self, kind: str) -> float:
        # video and audio are long-running; images are quick.
        return self.timeouts.video_job_s if kind in ("video", "audio") else self.timeouts.image_job_s


def load_config(path: str | Path) -> EngineConfig:
    """Read engine.yaml and validate into EngineConfig (raises on malformed config).

    Relative entries in `paths:` and `logging.dir` resolve against the repo root
    (the parent of the config file's directory), so the shipped config works
    from any checkout location.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    cfg = EngineConfig.model_validate(data)
    root = path.resolve().parent.parent
    for name in ("workflows", "templates", "pipelines", "runs"):
        p = Path(getattr(cfg.paths, name))
        if not p.is_absolute():
            setattr(cfg.paths, name, str(root / p))
    if not Path(cfg.logging.dir).is_absolute():
        cfg.logging.dir = str(root / cfg.logging.dir)
    return cfg
