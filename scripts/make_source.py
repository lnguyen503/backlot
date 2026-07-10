"""Generate a 1024x576 source still and stage it into ComfyUI input/ for img2vid."""
from __future__ import annotations

import asyncio
import shutil
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
PROMPT = ("a serene mountain lake at sunrise, mist drifting over the water, pine "
          "trees, dramatic clouds, cinematic, highly detailed")


async def main():
    cfg = load_config(CFG)
    reg = Registry.load(cfg)
    print("registry workflows:", reg.names())  # confirms img2vid_svd loaded too
    client = ComfyClient(cfg.comfyui.base_url, uuid.uuid4().hex, cfg.timeouts.http_s)
    jobs = JobManager(cfg, reg, client)
    ws = WsListener(client, cfg.comfyui.ws_url, jobs.on_event, cfg.timeouts.ws_reconnect_max_s)
    jobs.set_ws(ws)
    ws.start()
    await asyncio.sleep(0.5)
    res = await jobs.run_workflow(
        "txt2img_sdxl",
        {"positive_prompt": PROMPT, "width": 1024, "height": 576, "steps": 25},
        wait=True, timeout_s=240,
    )
    print("source gen:", res["state"], [a["filename"] for a in res["outputs"]])
    asset = res["outputs"][0]
    src = Path(cfg.comfyui.output_dir) / asset.get("subfolder", "") / asset["filename"]
    dst = Path(cfg.comfyui.input_dir) / "backlot_src.png"
    shutil.copy(src, dst)
    print("staged source ->", str(dst), "exists:", dst.exists())
    await ws.stop()
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
