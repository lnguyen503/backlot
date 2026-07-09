"""Async HTTP client for the ComfyUI server (§8.1, §8.3).

Distinguishes the two formats only at the call boundary: `queue_prompt` takes the
API-format graph to execute and (optionally) the UI-format graph to embed for the
learning feature via top-level `extra_data.extra_pnginfo.workflow` (§7.2).
"""
from __future__ import annotations

from typing import Any, Optional
from urllib.parse import urlencode

import httpx


class ComfyError(Exception):
    pass


class ComfyClient:
    def __init__(self, base_url: str, client_id: str, http_timeout_s: float = 30,
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        self._base = base_url.rstrip("/")
        self._client_id = client_id
        self._http = httpx.AsyncClient(timeout=http_timeout_s, transport=transport)

    @property
    def client_id(self) -> str:
        return self._client_id

    async def aclose(self) -> None:
        await self._http.aclose()

    async def queue_prompt(self, api_graph: dict, ui_graph: Optional[dict] = None) -> str:
        """POST /prompt. Returns prompt_id. Raises on node_errors (fail fast, §8.1)."""
        body: dict[str, Any] = {"prompt": api_graph, "client_id": self._client_id}
        if ui_graph is not None:
            # extra_data is TOP-LEVEL, not inside prompt — this is what embeds the
            # UI graph into the output PNG metadata (§7.2 primary learning mechanism).
            body["extra_data"] = {"extra_pnginfo": {"workflow": ui_graph}}
        resp = await self._http.post(f"{self._base}/prompt", json=body)
        if resp.status_code != 200:
            raise ComfyError(f"/prompt {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if data.get("node_errors"):
            raise ComfyError(f"node_errors: {data['node_errors']}")
        pid = data.get("prompt_id")
        if not pid:
            raise ComfyError(f"no prompt_id in /prompt response: {data}")
        return pid

    async def get_history(self, prompt_id: str) -> dict:
        resp = await self._http.get(f"{self._base}/history/{prompt_id}")
        resp.raise_for_status()
        return resp.json()

    async def get_object_info(self, node: Optional[str] = None) -> dict:
        url = f"{self._base}/object_info" + (f"/{node}" if node else "")
        resp = await self._http.get(url)
        resp.raise_for_status()
        return resp.json()

    async def interrupt(self) -> None:
        await self._http.post(f"{self._base}/interrupt")

    def view_url(self, filename: str, subfolder: str = "", type_: str = "output") -> str:
        query = urlencode({"filename": filename, "subfolder": subfolder, "type": type_})
        return f"{self._base}/view?{query}"
