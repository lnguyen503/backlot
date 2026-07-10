"""Generate the same benchmark prompts via Nano Banana Pro (Gemini 3 Pro Image)
for a side-by-side vs local FLUX. Reads GEMINI_API_KEY from .env or the environment;
never prints it."""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import httpx

ENV = Path(__file__).resolve().parents[1] / ".env"   # GEMINI_API_KEY=... (or set the env var)
MODEL = "gemini-3-pro-image-preview"
OUT = Path(__file__).resolve().parents[1] / "runs/compare"
OUT.mkdir(parents=True, exist_ok=True)

# Same prompts as the local FLUX set (tests/bench_images.py).
PROMPTS = {
    "ui_mockup": "a clean modern SaaS analytics dashboard UI, sidebar navigation, line and bar charts, KPI cards, light theme, professional product design, ui screenshot",
    "illustration": "a friendly 3d illustration of a cute robot assistant waving hello, soft pastel colors, octane render, clay style, plain white background",
    "text_poster": "a bold motivational poster with large clean typography that reads SHIP IT, vibrant purple to blue gradient background, modern minimal design",
    "infographic": "a simple clean infographic showing a three step process with icons and labels reading Plan, Build, Launch, flat vector design, numbered steps, light background",
}


def get_key() -> str | None:
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GEMINI_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("GEMINI_API_KEY")


def main() -> None:
    key = get_key()
    if not key:
        print("NO_KEY")
        sys.exit(2)
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={key}"
    for name, prompt in PROMPTS.items():
        body = {"contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]}}
        try:
            r = httpx.post(url, json=body, timeout=180)
        except Exception as e:
            print(f"{name}: ERR {type(e).__name__}: {str(e)[:120]}")
            continue
        if r.status_code != 200:
            print(f"{name}: HTTP {r.status_code} {r.text[:160]}")
            continue
        data = r.json()
        saved = False
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                inl = part.get("inlineData") or part.get("inline_data")
                if inl and inl.get("data"):
                    (OUT / f"banana_{name}.png").write_bytes(base64.b64decode(inl["data"]))
                    print(f"{name}: OK -> banana_{name}.png")
                    saved = True
                    break
            if saved:
                break
        if not saved:
            print(f"{name}: NO_IMAGE {json.dumps(data)[:160]}")


if __name__ == "__main__":
    main()
