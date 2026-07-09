"""Phase-1 LIVE exit gate (requires a running ComfyUI + a checkpoint).

Part 1: a real txt2img run end-to-end -> an image asset whose file exists.
Part 2 (2b): kill the websocket mid-job and verify the engine reconnects with the
same client_id, reconciles via /history, and still completes with its output.

Run:  .venv/Scripts/python.exe tests/live_phase1.py
"""
from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backlot.engine.comfy_client import ComfyClient  # noqa: E402
from backlot.engine.config import load_config  # noqa: E402
from backlot.engine.job_manager import JobManager  # noqa: E402
from backlot.engine.registry import Registry  # noqa: E402
from backlot.engine.ws_listener import WsListener  # noqa: E402

CFG = str(Path(__file__).resolve().parents[1] / "config/engine.yaml")


def _build():
    cfg = load_config(CFG)
    client = ComfyClient(cfg.comfyui.base_url, uuid.uuid4().hex, cfg.timeouts.http_s)
    jobs = JobManager(cfg, Registry.load(cfg), client)
    ws = WsListener(client, cfg.comfyui.ws_url, jobs.on_event, cfg.timeouts.ws_reconnect_max_s)
    jobs.set_ws(ws)
    return cfg, client, jobs, ws


def _output_exists(cfg, asset) -> bool:
    out = Path(cfg.comfyui.output_dir)
    sub = asset.get("subfolder", "")
    return (out / sub / asset["filename"]).exists()


async def run_basic():
    cfg, client, jobs, ws = _build()
    ws.start()
    await asyncio.sleep(0.5)
    print(f"[basic] client_id={client.client_id}  queuing txt2img...")
    res = await jobs.run_workflow(
        "txt2img_sdxl",
        {"positive_prompt": "a friendly robot reading a book, soft studio light, highly detailed",
         "steps": 20}, wait=True, timeout_s=240,
    )
    files = [a["filename"] for a in res["outputs"]]
    on_disk = all(_output_exists(cfg, a) for a in res["outputs"]) and bool(res["outputs"])
    print(f"[basic] state={res['state']} outputs={files} on_disk={on_disk}")
    await ws.stop()
    await client.aclose()
    return res["state"] == "completed" and on_disk


async def run_wsdrop():
    cfg, client, jobs, ws = _build()
    ws.start()
    await asyncio.sleep(0.5)
    print(f"[wsdrop] client_id={client.client_id}  queuing txt2img...")
    start = await jobs.run_workflow(
        "txt2img_sdxl",
        {"positive_prompt": "a neon city street at night in the rain, cinematic",
         "steps": 25}, wait=False,
    )
    rid = start["run_id"]
    await asyncio.sleep(2.0)
    await ws.drop_once()
    print("[wsdrop] >>> forcibly dropped the websocket mid-job; awaiting recovery <<<")
    state = "running"
    for _ in range(240):
        state = jobs.get_status(run_id=rid)["state"]
        if state in ("completed", "failed", "cancelled"):
            break
        await asyncio.sleep(1)
    st = jobs.get_status(run_id=rid)
    files = [a["filename"] for a in st["outputs"]]
    on_disk = all(_output_exists(cfg, a) for a in st["outputs"]) and bool(st["outputs"])
    print(f"[wsdrop] recovered -> state={st['state']} outputs={files} on_disk={on_disk}")
    await ws.stop()
    await client.aclose()
    return st["state"] == "completed" and on_disk


async def main():
    basic_ok = await run_basic()
    drop_ok = await run_wsdrop()
    print(f"\nEXIT GATE: basic={'PASS' if basic_ok else 'FAIL'}  "
          f"ws-drop={'PASS' if drop_ok else 'FAIL'}")
    print("RESULT:", "PASS" if (basic_ok and drop_ok) else "FAIL")
    sys.exit(0 if (basic_ok and drop_ok) else 1)


if __name__ == "__main__":
    asyncio.run(main())
