"""Run img2vid_svd on the staged source still and report the output clip."""
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
    client = ComfyClient(cfg.comfyui.base_url, uuid.uuid4().hex, cfg.timeouts.http_s)
    jobs = JobManager(cfg, Registry.load(cfg), client)
    ws = WsListener(client, cfg.comfyui.ws_url, jobs.on_event, cfg.timeouts.ws_reconnect_max_s)
    jobs.set_ws(ws)
    ws.start()
    await asyncio.sleep(0.5)
    print("queuing img2vid_svd (first run loads the 9 GB SVD model; be patient)...")
    res = await jobs.run_workflow(
        "img2vid_svd",
        {"image": "backlot_src.png", "frames": 25, "motion_bucket_id": 127, "steps": 20},
        wait=True, timeout_s=900,
    )
    print("video:", res["state"], "outputs:", res["outputs"])
    if not res["outputs"] and res.get("prompt_id"):
        hist = await client.get_history(res["prompt_id"])
        outs = hist.get(res["prompt_id"], {}).get("outputs", {})
        print("DEBUG history outputs:", json.dumps(outs)[:1000])
    await ws.stop()
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
