"""Job lifecycle: assemble -> queue -> track via ws -> collect outputs (§8).

Asset collection is idempotent: outputs are stored in a dict keyed by
(node_id, filename, subfolder), so the interim `executed` path, the final
`/history` sweep, and any reconnect `reconcile` replay all converge without
double-counting (§8.3).
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Optional

from .comfy_client import ComfyClient
from .config import EngineConfig
from .inject import assemble_api
from .models import Asset, Capability, JobState, JobStatus
from .registry import Registry
from .ws_listener import WsListener

_KIND_KEYS = (("image", "images"), ("video", "gifs"), ("audio", "audio"))


class _Run:
    def __init__(self, run_id: str, cap: Capability):
        self.cap = cap
        self.status = JobStatus(run_id=run_id)
        self.assets: dict[tuple, Asset] = {}
        self.done = asyncio.Event()


class JobManager:
    def __init__(self, cfg: EngineConfig, registry: Registry, client: ComfyClient,
                 ws: Optional[WsListener] = None):
        self._cfg = cfg
        self._reg = registry
        self._client = client
        self._ws = ws
        self._runs: dict[str, _Run] = {}
        self._by_prompt: dict[str, _Run] = {}

    def set_ws(self, ws: WsListener) -> None:
        self._ws = ws

    async def run_workflow(self, name: str, params: dict[str, Any], wait: bool = False,
                           timeout_s: Optional[float] = None) -> dict:
        cap = self._reg.get(name)
        graph, _resolved = assemble_api(cap, params or {})
        run = _Run(uuid.uuid4().hex, cap)
        self._runs[run.status.run_id] = run
        pid = await self._client.queue_prompt(graph)
        run.status.prompt_id = pid
        run.status.state = JobState.RUNNING
        self._by_prompt[pid] = run
        if self._ws:
            self._ws.track(pid)
        if wait:
            await self._await_done(run, timeout_s or self._cfg.job_timeout_s(cap.kind))
        return run.status.public_dict()

    async def _await_done(self, run: _Run, timeout_s: float) -> None:
        try:
            await asyncio.wait_for(run.done.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            run.status.state = JobState.FAILED
            run.status.error = "timeout"
            await self._client.interrupt()
            self._finish(run)

    async def on_event(self, event: dict) -> None:
        pid = event.get("prompt_id")
        run = self._by_prompt.get(pid) if pid else None
        if run is not None:
            await self._dispatch(run, event)

    async def _dispatch(self, run: _Run, event: dict) -> None:
        etype = event["event"]
        if etype == "progress":
            run.status.progress.value = event["value"]
            run.status.progress.max = event["max"]
        elif etype == "executing":
            run.status.current_node = event.get("node")
            if event.get("node") is None:  # this prompt finished
                await self._finalize(run)
        elif etype == "executed":
            self._collect(run, event.get("output"), run.status.prompt_id)
        elif etype == "error":
            run.status.state = JobState.FAILED
            run.status.error = event.get("message")
            self._finish(run)
        elif etype == "reconcile":
            self._collect_history(run, event.get("history"))
            await self._finalize(run)

    async def _finalize(self, run: _Run) -> None:
        if run.done.is_set():
            return
        try:
            hist = await self._client.get_history(run.status.prompt_id)
            self._collect_history(run, hist.get(run.status.prompt_id))
        except Exception:
            pass
        if run.status.state != JobState.FAILED:
            run.status.state = JobState.COMPLETED
        self._finish(run)

    def _finish(self, run: _Run) -> None:
        run.status.outputs = list(run.assets.values())
        run.status.view_workflow.ready = bool(run.assets)
        run.status.view_workflow.hint = (
            "Drag the output image onto the ComfyUI canvas to load the exact graph."
        )
        if self._ws and run.status.prompt_id:
            self._ws.untrack(run.status.prompt_id)
        run.done.set()

    def _collect_history(self, run: _Run, hist_entry: Optional[dict]) -> None:
        for node_id, out in ((hist_entry or {}).get("outputs") or {}).items():
            self._collect(run, out, node_id)

    def _collect(self, run: _Run, output: Optional[dict], node_id: Optional[str]) -> None:
        if not output or node_id is None:
            return
        # SaveAnimatedWEBP returns its clip under "images" with an "animated" flag.
        animated = bool(any(output.get("animated") or []))
        for kind, key in _KIND_KEYS:
            eff_kind = "video" if (key == "images" and animated) else kind
            for item in output.get(key, []) or []:
                asset = self._asset(eff_kind, item, node_id)
                if asset and asset.node_id in run.cap.client_outputs:
                    run.assets[asset.dedupe_key()] = asset

    def _asset(self, kind: str, item: dict, node_id) -> Optional[Asset]:
        filename = item.get("filename")
        if not filename:
            return None
        subfolder = item.get("subfolder", "")
        return Asset(
            type=kind, filename=filename, subfolder=subfolder, node_id=str(node_id),
            url=self._client.view_url(filename, subfolder, item.get("type", "output")),
        )

    def _lookup(self, run_id: Optional[str], prompt_id: Optional[str]) -> _Run:
        run = self._runs.get(run_id) if run_id else self._by_prompt.get(prompt_id or "")
        if run is None:
            raise KeyError("unknown run")
        return run

    def get_status(self, run_id: Optional[str] = None,
                   prompt_id: Optional[str] = None) -> dict:
        return self._lookup(run_id, prompt_id).status.public_dict()

    async def wait_for(self, run_id: str) -> dict:
        """Await a run's completion (used by the web layer to persist results)."""
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError("unknown run")
        await run.done.wait()
        return run.status.public_dict()

    async def cancel(self, run_id: Optional[str] = None,
                     prompt_id: Optional[str] = None) -> dict:
        run = self._lookup(run_id, prompt_id)
        await self._client.interrupt()
        run.status.state = JobState.CANCELLED
        self._finish(run)
        return {"cancelled": True}
