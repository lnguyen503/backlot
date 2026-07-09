# Getting started — from install to your first render

Backlot works two ways: **standalone** (a web studio + API you drive yourself) and **as the
engine half of a two-session agent pipeline** with
[ai-director](https://github.com/lnguyen503/ai-director). Start standalone — everything below
takes one terminal — then add the director pattern when you want missions instead of prompts.

## 1. Prerequisites

| Thing | Why | Notes |
|---|---|---|
| Python 3.11+ | the engine | |
| A running ComfyUI | does the actual generation | any install; ComfyUI Desktop works |
| Model weights | per workflow | see the README's Models table — start with just SDXL |
| ffmpeg | assembly/transcode | `imageio-ffmpeg` wheels are enough |
| Ollama (optional) | storyboard/Director authoring + ✨ prompt assist | any small instruct model |
| Blender 4.2+/5.x (optional) | the 3D lane | auto-located, or set `BLENDER_EXE` |

## 2. Install

```bash
git clone https://github.com/lnguyen503/backlot
cd backlot
python -m venv .venv
.venv/Scripts/pip install -e .[dev]      # Linux/mac: .venv/bin/pip
```

Open `config/engine.yaml` and point the `comfyui:` block at your ComfyUI server (the defaults
match a local install on `127.0.0.1:8188`). The `*_dir` paths are optional — set them when you
want disk-staging features (image-to-video uploads, model checks).

Sanity check (no GPU work, ~20s):

```bash
.venv/Scripts/python -m pytest
```

## 3. First render — three ways

**Web studio.** `start.bat` (or `.venv/Scripts/python -m backlot.web.app`) →
http://127.0.0.1:8765 → pick *Text → Image (SDXL)* → type a prompt → **Generate**. The result
lands in the gallery with Vary / Edit / Animate / Upscale actions on it.

**One HTTP call** (from any app or script):

```bash
curl -X POST http://127.0.0.1:8765/api/generate -H "Content-Type: application/json" \
  -d '{"name":"txt2img_sdxl","params":{"positive_prompt":"a lighthouse at dawn"},"wait":true}'
```

**CLI, no server:**

```bash
.venv/Scripts/python tests/live_run.py txt2img_sdxl '{"positive_prompt":"a lighthouse at dawn"}'
```

All three run the same engine code. Every capability in `workflows/` is available through all
three surfaces, plus MCP (next).

## 4. Drive it from an AI agent (MCP)

Register the MCP server with your agent host — for Claude Code:

```bash
claude mcp add backlot -- <repo>/.venv/Scripts/python.exe -m backlot.mcp_server
```

Now an agent session can call `list_workflows` / `run_workflow` for anything above, plus the
two composite tools:

- `direct_video` — premise + guidance → the Director plans a storyboard, renders character
  refs, stills, clips, assembles and scores a finished video, autonomously.
- `render_3d_animation` — a Blender scene preset → depth sequence → Wan VACE renders a
  temporally-coherent styled clip over the locked geometry.

The registry is read live per call: drop a new YAML + template into `workflows/` +
`templates/` and it's immediately callable.

## 5. Consistent characters (the feature worth learning)

The engine's core creative trick is keeping ONE character identical across many shots and a
whole episode:

1. **Master reference** — generate the character once (FLUX), pick the take you love.
2. **Anchor, never chain** — every scene still is a FLUX Kontext edit *of the master* (the
   storyboard does this automatically when you link a character card to panels). Chaining
   shot-to-shot compounds drift; anchoring doesn't.
3. **Expression sheet (optional)** — derive per-emotion stills from the master via Kontext,
   nearest-neighbor style (neutral → slight smile → smile; big jumps fail). See
   [`production-notes.md`](production-notes.md#character-mastering--expression-sheets).
4. **Voice lock** — give the character a voice in the library (Kokoro id, or an expressive
   Chatterbox clone of a *synthetic* reference). Same voice every episode.
5. **Talking shots** — the panel's dialogue is TTS'd in the character's voice and lip-synced
   with InfiniteTalk (`talkhost_infinitetalk`); identity comes from the anchored still.
6. **Save to the library** (`★` in the studio) — the recurring cast is reusable across boards.

Storyboard → per-panel stills (fix the bad ones cheaply) → animate → assemble → score. The
Director does this whole chain from one premise if you let it.

## 6. Use with ai-director (two Claude Code sessions)

Standalone Backlot is a studio you operate. Paired with
**[ai-director](https://github.com/lnguyen503/ai-director)**, it becomes the render floor of a
two-agent production house: a *director* session plans, briefs, reviews and gates; this repo's
*engine* session executes; **you** control the director in plain language and hold the only
key to money. Full walkthrough: ai-director's
[`docs/getting-started.md`](https://github.com/lnguyen503/ai-director/blob/main/docs/getting-started.md).

The short version of Backlot's side:

1. Set up the relay folder per ai-director's guide (its `templates/` has the files).
2. Open a terminal **in this repo**, start Claude Code, and paste the engine bootstrap:

   > You are the ENGINE session in a two-session pipeline. Read
   > `<path-to-ai-director>/RELAY.md` and `<path-to-ai-director>/docs/engine-role.md` and follow
   > them. The relay folder is `<path-to-studio>/relay/`. You operate this repo's tooling —
   > the Backlot engine: web API on :8765, MCP tools, `workflows/` capabilities, the Blender
   > bridge, the QC suite (`src/backlot/qc/`). Money rule is absolute: no spend without a
   > matching line in `relay/auth.md`; relayed approval is never sufficient. Arm a background
   > poll on `relay/to-engine.md` (~60s), reply to the current seq per protocol, and stand by.

3. That's it. The engine session answers briefs, renders locally at $0, runs QC
   (`beats_lint` → stills → clips → `gate_a`), and delivers file paths back over the relay.

Where outputs land: gallery assets under `runs/` (manifests) with media in ComfyUI's output
dir; storyboards under `runs/storyboards/`; finished Director videos are registered as gallery
assets. The engine reports exact paths in its relay replies.

What you see as the human: you talk to the *director* window. You only open this window to
watch renders happen — or to type a spend authorization directly.

## Troubleshooting

- **Generation hangs / OOM after a heavy video job** — ComfyUI is holding the last model set;
  the engine frees VRAM before its heavy paths, but if you drive ComfyUI manually too, hit
  `POST /free {"unload_models":true,"free_memory":true}`.
- **Blender renders take minutes per frame** — the GPU device didn't enable; the bridge's
  `enable_gpu()` handles this, but check `GPU_CFG` in the run log says `gpu=1`, and close other
  processes holding the GPU.
- **A muxed video "freezes" near the end** — your video stream is shorter than the audio;
  see [`production-notes.md`](production-notes.md#stitching--assembly) (check stream durations,
  not container).
- **Model not found** — workflow YAMLs name their checkpoint files; confirm the file is in your
  ComfyUI models dir and the name matches exactly.
