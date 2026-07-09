import json

import httpx
import pytest

from backlot.engine.comfy_client import ComfyClient, ComfyError


def _client(handler):
    return ComfyClient("http://t", "cid", transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_queue_prompt_returns_pid():
    def h(req):
        return httpx.Response(200, json={"prompt_id": "P1", "node_errors": {}})

    c = _client(h)
    assert await c.queue_prompt({"1": {}}) == "P1"
    await c.aclose()


@pytest.mark.asyncio
async def test_node_errors_raise():
    def h(req):
        return httpx.Response(200, json={"prompt_id": None, "node_errors": {"3": "bad"}})

    c = _client(h)
    with pytest.raises(ComfyError):
        await c.queue_prompt({"1": {}})
    await c.aclose()


@pytest.mark.asyncio
async def test_extra_data_sent_for_ui_graph():
    """The primary learning mechanism (§7.2): UI graph must ride in top-level extra_data."""
    seen = {}

    def h(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json={"prompt_id": "P1"})

    c = _client(h)
    await c.queue_prompt({"1": {}}, ui_graph={"nodes": []})
    assert seen["extra_data"]["extra_pnginfo"]["workflow"] == {"nodes": []}
    assert "extra_data" not in seen["prompt"] if isinstance(seen["prompt"], dict) else True
    await c.aclose()


def test_view_url():
    c = _client(lambda r: httpx.Response(200))
    url = c.view_url("a.png", "sub", "output")
    assert "filename=a.png" in url and "subfolder=sub" in url and "type=output" in url
