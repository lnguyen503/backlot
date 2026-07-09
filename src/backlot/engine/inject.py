"""Inject caller params into a capability's API-format graph (§4.2).

All values are validated against their InjectSpec (type, bounds) before they
touch a graph — no raw value is spliced in unchecked (injection-defense, §5.1).
"""
from __future__ import annotations

import copy
import random
from typing import Any

from .models import Capability, InjectSpec, ParamType

_SEED_MAX = 2**32 - 1


class InjectError(Exception):
    pass


def assemble_api(cap: Capability, params: dict[str, Any]) -> tuple[dict, dict]:
    """Return (assembled_api_graph, resolved_params).

    Deep-copies the template, resolves+validates each param, randomizes seeds,
    and writes values into the copy. resolved_params records the exact values
    used (including any auto-assigned seed) for the run manifest.
    """
    _check_required(cap, params)
    graph = copy.deepcopy(cap.api_graph)
    resolved: dict[str, Any] = {}
    for spec in cap.inject:
        value = _resolve_value(spec, params)
        if value is None:
            continue
        _write(graph, spec, value)
        resolved[spec.name] = value
    return graph, resolved


def _resolve_value(spec: InjectSpec, params: dict[str, Any]) -> Any:
    provided = spec.name in params and params[spec.name] is not None
    value = params[spec.name] if provided else spec.default
    if spec.type == ParamType.SEED:
        hi = int(spec.max) if spec.max is not None else _SEED_MAX  # some nodes cap at int32
        if value is None or int(value) < 0:
            return random.randint(0, hi)
        return int(value)
    if value is None:
        return None
    return _coerce(spec, value)


def _coerce(spec: InjectSpec, value: Any) -> Any:
    try:
        if spec.type == ParamType.INT:
            value = int(value)
        elif spec.type == ParamType.FLOAT:
            value = float(value)
        elif spec.type == ParamType.BOOL:
            value = bool(value)
        elif spec.type in (ParamType.STRING, ParamType.ENUM, ParamType.IMAGE, ParamType.AUDIO):
            value = str(value)
    except (TypeError, ValueError) as e:
        raise InjectError(f"param '{spec.name}': cannot coerce to {spec.type.value}: {e}")
    _check_bounds(spec, value)
    return value


def _check_bounds(spec: InjectSpec, value: Any) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if spec.min is not None and value < spec.min:
            raise InjectError(f"param '{spec.name}'={value} below min {spec.min}")
        if spec.max is not None and value > spec.max:
            raise InjectError(f"param '{spec.name}'={value} above max {spec.max}")


def _write(graph: dict, spec: InjectSpec, value: Any) -> None:
    node = str(spec.api[0])
    if node not in graph:
        raise InjectError(f"param '{spec.name}': node {node} missing from graph")
    target = graph[node]
    for key in spec.api[1:-1]:
        target = target[key]
    target[spec.api[-1]] = value


def _check_required(cap: Capability, params: dict[str, Any]) -> None:
    for spec in cap.inject:
        missing = spec.name not in params or params[spec.name] is None
        if spec.required and missing and spec.default is None:
            raise InjectError(f"missing required param '{spec.name}'")
