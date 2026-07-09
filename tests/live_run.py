"""Reusable live runner: run any registered workflow against ComfyUI.

Usage: live_run.py <workflow_name> '<json_params>' [timeout_s]
"""
from __future__ import annotations

import asyncio
import json
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


async def main():
    name = sys.argv[1]
    params = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 600
    cfg = load_config(CFG)
    client = ComfyClient(cfg.comfyui.base_url, uuid.uuid4().hex, cfg.timeouts.http_s)
    jobs = JobManager(cfg, Registry.load(cfg), client)
    ws = WsListener(client, cfg.comfyui.ws_url, jobs.on_event, cfg.timeouts.ws_reconnect_max_s)
    jobs.set_ws(ws)
    ws.start()
    await asyncio.sleep(0.5)
    print(f"running {name} ...", flush=True)
    try:
        res = await jobs.run_workflow(name, params, wait=True, timeout_s=timeout)
        print("STATE:", res["state"], "OUTPUTS:",
              [a["filename"] for a in res["outputs"]], flush=True)
        if res["state"] != "completed":
            print("ERROR:", res.get("error"), flush=True)
    except Exception as e:
        print("EXCEPTION:", repr(e)[:600], flush=True)
    await ws.stop()
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
