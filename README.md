# Backlot

**A local AI production studio that drives [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [Blender](https://www.blender.org/) — controllable by humans (web UI) and by AI agents (MCP), on your own GPU, at $0/generation.**

Backlot wraps a running ComfyUI server with a typed engine, a no-build web studio, and an MCP
server, so the same capabilities are drivable three ways:

1. **Web studio** — `http://127.0.0.1:8765`: prompt → image/video/music, gallery, editing tools,
   storyboards, an autonomous Director.
2. **HTTP API** — every feature is an endpoint with a `wait` flag (`POST /api/generate?wait=true`
   returns the finished asset in one call), so other apps and scripts can drive it.
3. **MCP server** — AI agents (e.g. Claude Code) get `list_workflows` / `run_workflow` /
   `direct_video` / `render_3d_animation` tools and can produce finished videos autonomously.

> Backlot is an independent project, not affiliated with or endorsed by Comfy Org (ComfyUI) or the
> Blender Foundation. Names are used descriptively.

## What it can do

| Lane | Capabilities |
|---|---|
| **Image** | FLUX.1-dev · SDXL · FLUX Kontext instruction editing · SDXL inpaint (mask brush) · Real-ESRGAN upscale · Kontext presets (replace background, camera angle, relight, multi-view turnaround) |
| **Video** | Wan 2.2 (5B + 14B + Lightning 4-step) · LTX-Video · SVD image-to-video · motion-sync (reference clip → depth → restyled subject) |
| **Talking heads** | InfiniteTalk (Wan 2.1 14B audio-driven portrait — photoreal lip-sync) · multi-person two-shot (per-speaker audio routed to faces) · Sonic (legacy) |
| **Audio** | ACE-Step text-to-music · Kokoro TTS voice-over (8 voices) |
| **3D / Blender** | Headless Blender bridge (bpy runner, GPU Cycles) · depth-sequence export → **Wan VACE** coherent restyled animation ("one camera move, any world") · SDXL depth-ControlNet per-frame restyle |
| **Storyboard** | Idea → LLM-drafted board (asset cards + panels) → consistent character stills (Kontext anchoring) → animate → assemble (crossfade, dialogue-preserving) → LLM-scored music + ambient + narration |
| **Director** | One premise + guidance → the LLM plans the board, renders refs/stills/clips, assembles and scores a finished video, autonomously (checkpointed + resumable) |
| **QC** | beats-lint (pre-render failure classes) · Gate-A ffmpeg checks (freezes, loudness, A/V drift) · face-identity gate · talking-segment pose/expression checks · still/clip VLM checks |
| **Identity** | Multi-shot character consistency: Kontext face-locked keyframes → i2v → insightface + GFPGAN re-lock |

Everything is **workflow-as-config**: a capability = one YAML (params, types, bounds) + one
ComfyUI API-format JSON template in `workflows/` + `templates/`. Drop in a pair, and it appears in
the UI, the HTTP API, and MCP — no code changes.

## Requirements

- **Python 3.11+**
- **A running ComfyUI server** (default `127.0.0.1:8188`) with the models for the workflows you
  want (see [Models](#models))
- **ffmpeg** on PATH (assembly, transcode; `imageio-ffmpeg` wheels work too)
- Optional: **Ollama** (`127.0.0.1:11434`) for storyboard/Director authoring + ✨ prompt assist
- Optional: **Blender 4.2+/5.x** for the 3D lane (auto-located, or set `BLENDER_EXE`)
- A CUDA GPU. Wan 14B / InfiniteTalk want ≥24 GB VRAM (built on an RTX 5090); the SDXL/SVD/LTX
  lanes run on much less.

## Install

```bash
git clone https://github.com/lnguyen503/backlot
cd backlot
python -m venv .venv
.venv/Scripts/pip install -e .[dev]        # Windows (Linux/mac: .venv/bin/pip)
```

Edit `config/engine.yaml`:
- point `comfyui.base_url` / `ws_url` at your ComfyUI server (defaults match a local install)
- optionally set the `comfyui.*_dir` paths (only needed by disk-staging features like
  image-to-video upload and model checks)

Run the studio:

```bash
.venv/Scripts/python -m backlot.web.app     # → http://127.0.0.1:8765
```

or double-click `start.bat` on Windows.

Run the test suite:

```bash
.venv/Scripts/python -m pytest
```

## MCP setup (drive it from an AI agent)

Register the MCP server with your agent host. For Claude Code:

```bash
claude mcp add backlot -- <repo>/.venv/Scripts/python.exe -m backlot.mcp_server
```

Tools exposed: `list_workflows`, `describe_workflow`, `run_workflow` (blocking or async),
`get_job_status`, `cancel_job`, `direct_video` (premise → finished film),
`list_blender_scenes`, `render_3d_animation` (Blender depth → VACE coherent clip).

The MCP server reads the workflow registry **live per call** — new workflow YAMLs appear without a
restart.

## Models

Model weights are **not** vendored — they live in your ComfyUI models directory. Each workflow
YAML documents what it needs. The main ones:

| Workflow | Weights |
|---|---|
| `txt2img_flux` | FLUX.1-dev fp8 |
| `txt2img_sdxl`, `inpaint_sdxl` | SDXL base 1.0 |
| `edit_kontext` | FLUX.1 Kontext dev |
| `txt2vid_wan`, `img2vid_wan*` | Wan 2.2 (TI2V 5B / I2V A14B + Lightning LoRA) |
| `txt2vid_ltx` | LTX-Video 2B |
| `img2vid_svd` | SVD (svd_xt) |
| `music_acestep` | ACE-Step 3.5B |
| `talkhost_infinitetalk*` | Wan 2.1 I2V 14B GGUF + InfiniteTalk (single/multi) + chinese-wav2vec2-base + lightx2v LoRA (via ComfyUI-WanVideoWrapper + KJNodes) |
| `vace_depth_video` | Wan 2.1 VACE (1.3B or 14B) |
| `upscale_esrgan` | RealESRGAN_x4plus.pth |
| `txt2img_controlnet_depth_sdxl` | SDXL + control-lora-depth-rank256 |

The faceswap identity pipeline additionally fetches `inswapper_128.onnx` + `GFPGANv1.4.onnx` into
`models/` (gitignored) and uses insightface `buffalo_l`.

**⚠ License note:** model weights carry their own licenses, independent of this repo's MIT code
license. Notably **FLUX.1-dev is non-commercial**; Wan, LTX, SVD, ACE-Step and others each have
their own terms. Review the license of every model you use before commercial work.

## Driving it from an AI director

Backlot is designed to be operated by a strategy/creative agent in a separate session — an
**[ai-director](https://github.com/lnguyen503/ai-director)** — over a simple file relay with
human-gated spending. The canonical protocol lives in the ai-director repo; the engine-side
summary (message format, sequencing, the append-log, and the human authorization channel) is in
[`docs/relay-protocol.md`](docs/relay-protocol.md). Hard-won production lessons (storyboard
consistency, stitching, talking heads, VRAM ops) are in
[`docs/production-notes.md`](docs/production-notes.md).

## Layout

```
src/backlot/
  engine/       config · registry · param injection · ComfyUI client · ws listener · job manager
  web/          FastAPI app + no-build web studio (static/index.html) + gallery store + sessions
  storyboard/   models · store · LLM agent · render · assemble · templates · library · director
  blender/      headless bpy runner + scene library (depth sequences, GPU Cycles)
  qc/           beats-lint · gate-A · face-id · talking-QC · still/clip VLM checks
  pipelines/    multi-shot character-consistency pipeline
  faceswap.py   insightface + GFPGAN identity re-lock (GPU)
  mcp_server.py MCP adapter
workflows/      one YAML per capability (params, bounds, docs)
templates/      matching ComfyUI API-format graphs
tests/          unit suite + live verifiers (tests/live_run.py <workflow> '<json>')
```

## License

MIT — see [LICENSE](LICENSE). Model weights and third-party components keep their own licenses.
