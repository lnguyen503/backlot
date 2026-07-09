"""Pydantic data models shared across the engine.

Kept dependency-free (no engine imports) so every module can use these types.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALLED = "stalled"


class ParamType(str, Enum):
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"
    IMAGE = "image"
    AUDIO = "audio"
    SEED = "seed"


class Asset(BaseModel):
    """A user-facing output produced by a capability's client_outputs node."""

    type: Literal["image", "video", "audio"]
    filename: str
    subfolder: str = ""
    url: str
    node_id: str

    def dedupe_key(self) -> tuple[str, str, str]:
        """Identity for idempotent collection across reconnect/resync (§8.3)."""
        return (self.node_id, self.filename, self.subfolder)


class Progress(BaseModel):
    value: int = 0
    max: int = 0

    def percent(self) -> float:
        return round(100.0 * self.value / self.max, 1) if self.max else 0.0


class ViewWorkflow(BaseModel):
    ready: bool = False
    hint: str = ""


class JobStatus(BaseModel):
    run_id: str
    prompt_id: Optional[str] = None
    state: JobState = JobState.QUEUED
    progress: Progress = Field(default_factory=Progress)
    current_node: Optional[str] = None
    queue_position: Optional[int] = None
    outputs: list[Asset] = Field(default_factory=list)
    error: Optional[str] = None
    view_workflow: ViewWorkflow = Field(default_factory=ViewWorkflow)

    def public_dict(self) -> dict[str, Any]:
        """Flatten for MCP/JSON returns (includes computed progress percent)."""
        return {
            "run_id": self.run_id,
            "prompt_id": self.prompt_id,
            "state": self.state.value,
            "progress": {
                "value": self.progress.value,
                "max": self.progress.max,
                "percent": self.progress.percent(),
            },
            "current_node": self.current_node,
            "queue_position": self.queue_position,
            "outputs": [a.model_dump() for a in self.outputs],
            "error": self.error,
            "view_workflow": self.view_workflow.model_dump(),
        }


class InjectSpec(BaseModel):
    """One injectable parameter, as declared in a workflow config entry (§4.2)."""

    name: str
    api: list[Any]                       # path into api graph, e.g. ["3", "inputs", "seed"]
    ui: Optional[dict[str, Any]] = None  # {node, widget} locator (Phase 4)
    type: ParamType = ParamType.STRING
    required: bool = False
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    enum_source: Optional[str] = None
    description: str = ""


class WorkflowParam(BaseModel):
    """Public, caller-facing view of an injectable param (list/describe tools)."""

    name: str
    type: str
    required: bool
    default: Any = None
    description: str = ""


class Capability(BaseModel):
    """A registered workflow: its config plus the loaded API-format graph."""

    name: str
    title: str
    kind: str
    description: str = ""
    template_path: str
    ui_template_path: Optional[str] = None
    client_outputs: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    inject: list[InjectSpec] = Field(default_factory=list)
    api_graph: dict[str, Any] = Field(default_factory=dict)
    learning_png: bool = False

    def params(self) -> list[WorkflowParam]:
        return [
            WorkflowParam(
                name=s.name,
                type=s.type.value,
                required=s.required,
                default=s.default,
                description=s.description,
            )
            for s in self.inject
        ]

    def public_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "kind": self.kind,
            "description": self.description,
            "params": [p.model_dump() for p in self.params()],
        }
