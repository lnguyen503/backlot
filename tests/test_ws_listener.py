import pytest

from backlot.engine.ws_listener import WsListener, parse_message


def test_parse_executing_null():
    e = parse_message('{"type":"executing","data":{"node":null,"prompt_id":"P"}}')
    assert e == {"event": "executing", "prompt_id": "P", "node": None}


def test_parse_progress():
    e = parse_message('{"type":"progress","data":{"value":3,"max":20,"prompt_id":"P"}}')
    assert e["event"] == "progress" and e["value"] == 3 and e["max"] == 20


def test_parse_executed_carries_output():
    e = parse_message(
        '{"type":"executed","data":{"node":"9","prompt_id":"P","output":{"images":[]}}}'
    )
    assert e["event"] == "executed" and e["node"] == "9" and e["output"] == {"images": []}


def test_parse_binary_is_preview():
    assert parse_message(b"\x00\x01") == {"event": "preview"}


def test_parse_garbage_and_unknown_are_none():
    assert parse_message("not json") is None
    assert parse_message('{"type":"totally_unknown"}') is None


class _FakeClient:
    client_id = "c"

    def __init__(self, history):
        self._history = history

    async def get_history(self, prompt_id):
        return self._history


@pytest.mark.asyncio
async def test_reconcile_emits_for_active_prompt():
    events = []

    async def on_event(e):
        events.append(e)

    hist = {"P": {"outputs": {"9": {"images": [{"filename": "a.png"}]}}}}
    wl = WsListener(_FakeClient(hist), "ws://x/ws", on_event)
    wl.track("P")
    await wl.reconcile_active()
    assert len(events) == 1
    assert events[0]["event"] == "reconcile" and events[0]["prompt_id"] == "P"


@pytest.mark.asyncio
async def test_reconcile_skips_unknown_prompt():
    events = []

    async def on_event(e):
        events.append(e)

    wl = WsListener(_FakeClient({}), "ws://x/ws", on_event)
    wl.track("P")
    await wl.reconcile_active()
    assert events == []
