"""Generate a diverse set of images for a quality benchmark across categories."""
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
PROMPTS = [
    ("ui_mockup", "a clean modern SaaS analytics dashboard UI, sidebar navigation, line and bar charts, KPI cards, light theme, professional product design, ui screenshot", 1216, 832),
    ("app_icon", "a rounded square iOS app icon for a weather app, stylized sun behind a cloud, soft blue gradient, glossy, centered, app store style", 1024, 1024),
    ("logo", "a minimalist logo for a coffee brand named BREW, simple bold line art, monochrome, vector, clean negative space", 1024, 1024),
    ("illustration", "a friendly 3d illustration of a cute robot assistant waving hello, soft pastel colors, octane render, clay style, plain white background", 1024, 1024),
    ("photoreal", "a professional studio product photo of a sleek modern smartphone standing upright on a marble surface, soft studio lighting, shallow depth of field, photorealistic", 1216, 832),
    ("text_poster", "a bold motivational poster with large clean typography that reads SHIP IT, vibrant purple to blue gradient background, modern minimal design", 832, 1216),
    ("infographic", "a simple clean infographic showing a three step process with icons and labels reading Plan, Build, Launch, flat vector design, numbered steps, light background", 1216, 832),
]


async def main():
    cfg = load_config(CFG)
    client = ComfyClient(cfg.comfyui.base_url, uuid.uuid4().hex, cfg.timeouts.http_s)
    jobs = JobManager(cfg, Registry.load(cfg), client)
    ws = WsListener(client, cfg.comfyui.ws_url, jobs.on_event, cfg.timeouts.ws_reconnect_max_s)
    jobs.set_ws(ws)
    ws.start()
    await asyncio.sleep(0.5)
    results = {}
    for key, prompt, w, h in PROMPTS:
        res = await jobs.run_workflow(
            "txt2img_flux", {"positive_prompt": prompt, "width": w, "height": h, "steps": 24},
            wait=True, timeout_s=300,
        )
        fn = res["outputs"][0]["filename"] if res["outputs"] else None
        results[key] = fn
        print(f"{key}: {res['state']} -> {fn}", flush=True)
    print("MAP: " + json.dumps(results), flush=True)
    await ws.stop()
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
