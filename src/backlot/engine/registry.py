"""Workflow registry: discover, load & validate workflow config entries (§4).

A capability = one `workflows/<name>.yaml` + its API-format template graph.
Adding a capability requires no code change (workflow-as-config, §4.1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml

from .config import EngineConfig
from .models import Capability, InjectSpec


class RegistryError(Exception):
    pass


class Registry:
    def __init__(self, capabilities: dict[str, Capability]):
        self._caps = capabilities

    @classmethod
    def load(cls, cfg: EngineConfig) -> "Registry":
        wf_dir = Path(cfg.paths.workflows)
        base = wf_dir.parent  # project root, for resolving relative template paths
        if not wf_dir.is_dir():
            raise RegistryError(f"workflows dir not found: {wf_dir}")
        caps: dict[str, Capability] = {}
        for yml in sorted(wf_dir.glob("*.yaml")):
            cap = cls._load_one(yml, base)
            if cap.name in caps:
                raise RegistryError(f"duplicate capability name: {cap.name}")
            caps[cap.name] = cap
        return cls(caps)

    @staticmethod
    def _load_one(yml: Path, base: Path) -> Capability:
        raw = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        if "name" not in raw or "template" not in raw:
            raise RegistryError(f"{yml.name}: missing 'name' or 'template'")
        api_path = (base / raw["template"]).resolve()
        try:
            api_graph = json.loads(api_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise RegistryError(f"{yml.name}: cannot load template {api_path}: {e}")
        inject = _load_inject(raw.get("inject", {}))
        _validate_inject_paths(yml.name, inject, api_graph)
        ui_rel = raw.get("ui_template")
        ui_path = str(base / ui_rel) if ui_rel and (base / ui_rel).exists() else None
        return Capability(
            name=raw["name"],
            title=raw.get("title", raw["name"]),
            kind=raw.get("kind", "image"),
            description=raw.get("description", ""),
            template_path=str(api_path),
            ui_template_path=ui_path,
            client_outputs=[str(x) for x in raw.get("client_outputs", [])],
            inputs=raw.get("inputs", {}) or {},
            inject=inject,
            api_graph=api_graph,
            learning_png=bool(raw.get("learning_png", False)),
        )

    def list(self, kind: Optional[str] = None) -> list[Capability]:
        return [c for c in self._caps.values() if kind is None or c.kind == kind]

    def get(self, name: str) -> Capability:
        if name not in self._caps:
            raise RegistryError(f"unknown workflow: {name}")
        return self._caps[name]

    def names(self) -> list[str]:
        return list(self._caps.keys())


def _load_inject(raw: dict) -> list[InjectSpec]:
    specs: list[InjectSpec] = []
    for name, spec in (raw or {}).items():
        data = dict(spec)
        data["name"] = name
        specs.append(InjectSpec.model_validate(data))
    return specs


def _validate_inject_paths(src: str, inject: list[InjectSpec], api_graph: dict) -> None:
    """Fail fast if any inject path points at a node missing from the template."""
    for spec in inject:
        node = str(spec.api[0])
        if node not in api_graph:
            raise RegistryError(f"{src}: inject '{spec.name}' -> node {node} not in template")
