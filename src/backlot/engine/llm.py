"""Local LLM client (Ollama) — the authoring brain the engine config reserved.

Phase-2 "authoring" finally wired: a thin synchronous client over the Ollama
native API (`/api/chat`) running at `models.base_url` (default 127.0.0.1:11434).
Used by the storyboard agent to turn an *idea* into a structured board.

Why Ollama native over the OpenAI-compat surface: `/api/chat` accepts a JSON
*schema* in the `format` field, so structured output is enforced server-side and
we get a guaranteed-parseable object back instead of best-effort prompt nudging.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import httpx

from .config import EngineConfig


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, base_url: str, model: str, keep_alive: str = "30m",
                 options: Optional[dict] = None, timeout_s: float = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.options = options or {}
        self.timeout_s = timeout_s

    @classmethod
    def from_config(cls, cfg: EngineConfig) -> "LLMClient":
        m = cfg.models
        return cls(m.base_url, m.authoring_model, getattr(m, "keep_alive", "30m"),
                   dict(getattr(m, "options", {}) or {}), cfg.timeouts.ollama_s)

    def chat(self, messages: list[dict], *, schema: Optional[dict] = None,
             model: Optional[str] = None, options: Optional[dict] = None) -> str:
        """Return the assistant message content for a chat exchange.

        `schema` (a JSON Schema dict) forces structured output via Ollama's
        `format` field. `messages` is the usual [{role, content}, ...] list.
        """
        body: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {**self.options, **(options or {})},
        }
        if schema is not None:
            body["format"] = schema
        try:
            r = httpx.post(f"{self.base_url}/api/chat", json=body, timeout=self.timeout_s)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"ollama chat failed: {e}") from e
        data = r.json()
        content = (data.get("message") or {}).get("content")
        if not content:
            raise LLMError(f"ollama returned no content: {data}")
        return content

    def chat_json(self, messages: list[dict], schema: dict, *,
                  model: Optional[str] = None, options: Optional[dict] = None) -> dict:
        """chat() with a schema, parsed into a dict. Raises LLMError on bad JSON."""
        raw = self.chat(messages, schema=schema, model=model, options=options)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMError(f"ollama returned non-JSON despite schema: {raw[:500]}") from e

    def available(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except httpx.HTTPError:
            return False
