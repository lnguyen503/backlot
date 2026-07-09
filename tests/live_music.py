"""Run music_acestep and report the output track."""
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
    cfg = load_config(CFG)
    reg = Registry.load(cfg)
    print("workflows:", reg.names())
    client = ComfyClient(cfg.comfyui.base_url, uuid.uuid4().hex, cfg.timeouts.http_s)
    jobs = JobManager(cfg, reg, client)
    ws = WsListener(client, cfg.comfyui.ws_url, jobs.on_event, cfg.timeouts.ws_reconnect_max_s)
    jobs.set_ws(ws)
    ws.start()
    await asyncio.sleep(0.5)
    print("queuing music_acestep (first run loads the 7 GB model)...")
    res = await jobs.run_workflow(
        "music_acestep",
        {"tags": "lofi hip hop, chill, mellow piano, soft drums, warm, relaxing, instrumental",
         "seconds": 20, "steps": 50},
        wait=True, timeout_s=900,
    )
    print("music:", res["state"], "outputs:", res["outputs"])
    if not res["outputs"] and res.get("prompt_id"):
        hist = await client.get_history(res["prompt_id"])
        outs = hist.get(res["prompt_id"], {}).get("outputs", {})
        print("DEBUG history outputs:", json.dumps(outs)[:1000])
    await ws.stop()
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
