"""Shared Engine runtime: wires config + registry + client + ws + job manager.

Used by BOTH the MCP server and the web backend so business logic lives in one
place (no duplication between the two adapters).
"""
from __future__ import annotations

import os
import uuid

from .comfy_client import ComfyClient
from .config import load_config
from .job_manager import JobManager
from .registry import Registry
from .ws_listener import WsListener

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_CONFIG = os.environ.get("BACKLOT_CONFIG", os.path.join(_REPO_ROOT, "config", "engine.yaml"))


class Engine:
    def __init__(self, config_path: str = DEFAULT_CONFIG):
        self.cfg = load_config(config_path)
        self.registry = Registry.load(self.cfg)
        self.client = ComfyClient(
            self.cfg.comfyui.base_url, uuid.uuid4().hex, self.cfg.timeouts.http_s
        )
        self.jobs = JobManager(self.cfg, self.registry, self.client)
        self.ws = WsListener(
            self.client, self.cfg.comfyui.ws_url, self.jobs.on_event,
            self.cfg.timeouts.ws_reconnect_max_s,
        )
        self.jobs.set_ws(self.ws)
        self._started = False

    def ensure_started(self) -> None:
        if not self._started:
            self.ws.start()
            self._started = True
