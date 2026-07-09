"""ComfyUI websocket listener with reconnect + reconcile (§8.2, §8.4).

Design for testability: `parse_message` is a PURE function (unit-tested against a
recorded transcript, exit-gate test 2a). The socket loop in `WsListener.run`
handles connect/reconnect and is exercised by the live integration test (2b).

On every (re)connect the listener reconciles each in-flight prompt_id via
`/history` and emits a synthetic `reconcile` event, so a completion that arrived
during a disconnect window is never lost — and because the JobManager collects
assets into a dedupe-keyed set, replay never double-counts (§8.3).
"""
from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable, Optional

import websockets

from .comfy_client import ComfyClient

EventCb = Callable[[dict], Awaitable[None]]


def parse_message(raw) -> Optional[dict]:
    """Normalize a ComfyUI ws message into an engine event, or None to ignore."""
    if isinstance(raw, (bytes, bytearray)):
        return {"event": "preview"}
    try:
        msg = json.loads(raw)
    except (ValueError, TypeError):
        return None
    mtype = msg.get("type")
    data = msg.get("data") or {}
    pid = data.get("prompt_id")
    if mtype == "status":
        info = ((data.get("status") or {}).get("exec_info") or {})
        return {"event": "status", "queue_remaining": info.get("queue_remaining")}
    if mtype == "execution_start":
        return {"event": "start", "prompt_id": pid}
    if mtype == "executing":
        return {"event": "executing", "prompt_id": pid, "node": data.get("node")}
    if mtype == "progress":
        return {"event": "progress", "prompt_id": pid,
                "value": data.get("value", 0), "max": data.get("max", 0)}
    if mtype == "executed":
        return {"event": "executed", "prompt_id": pid,
                "node": data.get("node"), "output": data.get("output")}
    if mtype == "execution_error":
        return {"event": "error", "prompt_id": pid, "message": str(data)}
    if mtype == "execution_cached":
        return {"event": "cached", "prompt_id": pid}
    return None


class WsListener:
    def __init__(self, client: ComfyClient, ws_url: str, on_event: EventCb,
                 reconnect_max_s: float = 30):
        self._client = client
        self._ws_url = ws_url
        self._on_event = on_event
        self._reconnect_max_s = reconnect_max_s
        self._active: set[str] = set()
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._conn = None  # live connection, for clean stop / drop-to-exercise-reconnect

    @property
    def url(self) -> str:
        return f"{self._ws_url}?clientId={self._client.client_id}"

    def track(self, prompt_id: str) -> None:
        self._active.add(prompt_id)

    def untrack(self, prompt_id: str) -> None:
        self._active.discard(prompt_id)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run())

    async def stop(self) -> None:
        self._stop.set()
        if self._conn is not None:
            await self._conn.close()
        if self._task:
            await asyncio.gather(self._task, return_exceptions=True)

    async def drop_once(self) -> None:
        """Forcibly close the live socket to exercise reconnect+reconcile (§8.4)."""
        if self._conn is not None:
            await self._conn.close()

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(self.url, max_size=None) as ws:
                    self._conn = ws
                    backoff = 1.0
                    await self.reconcile_active()  # catch up anything missed while down
                    await self._read_loop(ws)
            except (OSError, websockets.WebSocketException):
                if self._stop.is_set():
                    break
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._reconnect_max_s)

    async def _read_loop(self, ws) -> None:
        async for raw in ws:
            if self._stop.is_set():
                return
            event = parse_message(raw)
            if event:
                await self._on_event(event)

    async def reconcile_active(self) -> None:
        """For each in-flight prompt, re-sync from /history and emit reconcile."""
        for pid in list(self._active):
            try:
                hist = await self._client.get_history(pid)
            except Exception:
                continue
            if pid in hist:
                await self._on_event(
                    {"event": "reconcile", "prompt_id": pid, "history": hist[pid]}
                )
