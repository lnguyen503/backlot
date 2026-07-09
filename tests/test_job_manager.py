"""JobManager tests, including the Phase-1 exit-gate ws-drop reconcile test (2a).

Drives the manager with synthetic events (and a recorded fixture) so the
reconnect/reconcile + no-double-count invariant is verified offline.
"""
import json
from pathlib import Path

import pytest

from backlot.engine.config import load_config
from backlot.engine.job_manager import JobManager
from backlot.engine.registry import Registry

CFG = str(Path(__file__).resolve().parents[1] / "config/engine.yaml")
FIXTURE = Path(__file__).parent / "fixtures" / "ws_drop.json"
HISTORY = {"PID1": {"outputs": {"9": {"images": [
    {"filename": "out_001.png", "subfolder": "", "type": "output"}]}}}}
OUT9 = HISTORY["PID1"]["outputs"]["9"]


class _FakeClient:
    client_id = "test"

    def __init__(self, history, pid="PID1"):
        self._history = history
        self._pid = pid
        self.interrupted = False

    async def queue_prompt(self, graph, ui_graph=None):
        return self._pid

    async def get_history(self, prompt_id):
        return self._history

    async def interrupt(self):
        self.interrupted = True

    def view_url(self, filename, subfolder="", type_="output"):
        return f"http://x/{filename}"


def _jm(history=HISTORY, pid="PID1"):
    cfg = load_config(CFG)
    return JobManager(cfg, Registry.load(cfg), _FakeClient(history, pid))


async def _start(jm, pid="PID1"):
    res = await jm.run_workflow("txt2img_sdxl", {"positive_prompt": "a cat"})
    assert res["prompt_id"] == pid
    return res["run_id"], pid


@pytest.mark.asyncio
async def test_happy_path_completes_with_one_asset():
    jm = _jm()
    rid, pid = await _start(jm)
    await jm.on_event({"event": "progress", "prompt_id": pid, "value": 10, "max": 20})
    await jm.on_event({"event": "executed", "prompt_id": pid, "node": "9", "output": OUT9})
    await jm.on_event({"event": "executing", "prompt_id": pid, "node": None})
    st = jm.get_status(run_id=rid)
    assert st["state"] == "completed"
    assert len(st["outputs"]) == 1
    assert st["outputs"][0]["filename"] == "out_001.png"


@pytest.mark.asyncio
async def test_ws_drop_reconcile_no_double_count():
    jm = _jm()
    rid, pid = await _start(jm)
    # interim output arrives...
    await jm.on_event({"event": "executed", "prompt_id": pid, "node": "9", "output": OUT9})
    # --- ws drops; the executing:null completion is LOST. On reconnect we reconcile: ---
    await jm.on_event({"event": "reconcile", "prompt_id": pid, "history": HISTORY[pid]})
    # the delayed executing:null also replays after resync
    await jm.on_event({"event": "executing", "prompt_id": pid, "node": None})
    st = jm.get_status(run_id=rid)
    assert st["state"] == "completed"
    # executed + reconcile + history all reference the same file -> exactly one asset
    assert len(st["outputs"]) == 1


@pytest.mark.asyncio
async def test_ws_drop_fixture_replay():
    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    pid = fx["prompt_id"]
    jm = _jm(history=fx["history"], pid=pid)
    res = await jm.run_workflow("txt2img_sdxl", {"positive_prompt": "x"})
    rid = res["run_id"]
    from backlot.engine.ws_listener import parse_message
    for raw in fx["transcript_before_drop"]:
        ev = parse_message(raw)
        if ev:
            await jm.on_event(ev)
    # drop, then reconnect reconcile recovers the lost completion
    await jm.on_event({"event": "reconcile", "prompt_id": pid, "history": fx["history"][pid]})
    for raw in fx["transcript_after_reconnect"]:
        ev = parse_message(raw)
        if ev:
            await jm.on_event(ev)
    st = jm.get_status(run_id=rid)
    assert st["state"] == "completed"
    assert len(st["outputs"]) == 1
    assert st["outputs"][0]["filename"] == "fixture_001.png"


@pytest.mark.asyncio
async def test_timeout_marks_failed():
    jm = _jm()
    res = await jm.run_workflow(
        "txt2img_sdxl", {"positive_prompt": "x"}, wait=True, timeout_s=0.05
    )
    assert res["state"] == "failed"
    assert res["error"] == "timeout"


@pytest.mark.asyncio
async def test_cancel_marks_cancelled():
    jm = _jm()
    rid, _ = await _start(jm)
    out = await jm.cancel(run_id=rid)
    assert out["cancelled"] is True
    assert jm.get_status(run_id=rid)["state"] == "cancelled"
