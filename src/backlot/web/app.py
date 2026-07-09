"""FastAPI web backend for the Backlot (image-first MVP).

Thin async adapter over the SAME Engine core the MCP server uses. Endpoints:
  GET  /api/workflows           list capabilities
  GET  /api/workflows/{name}    param schema for one capability
  POST /api/generate            start a job -> {run_id, prompt_id}
  GET  /api/jobs/{run_id}       job status snapshot
  GET  /api/stream/{run_id}     SSE live progress
  GET  /api/assets              gallery (persisted run manifests)
  GET  /                        the studio UI
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import presets
from ..engine.config import load_config
from ..engine.runtime import DEFAULT_CONFIG, Engine
from .edit_api import init_edit, router as edit_router
from .sessions import SessionStore
from .storyboard_api import (
    init_storyboard, sb_state, lib_router as library_router, router as storyboard_router)
from .store import RunStore

_STATIC = Path(__file__).parent / "static"
_TERMINAL = {"completed", "failed", "cancelled"}


class _State:
    engine: Optional[Engine] = None
    store: Optional[RunStore] = None
    sessions: Optional[SessionStore] = None


state = _State()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.engine = Engine()
    state.engine.ensure_started()
    runs = state.engine.cfg.paths.runs
    state.store = RunStore(runs)
    state.sessions = SessionStore(runs)
    init_storyboard(state.engine)
    init_edit(_eng, _stage_source, _persist, state.store.save, sb_state.store)
    yield


app = FastAPI(title="Backlot", lifespan=lifespan)
app.include_router(storyboard_router)
app.include_router(library_router)
app.include_router(edit_router)


def _eng() -> Engine:
    if state.engine is None:
        raise HTTPException(503, "engine not ready")
    return state.engine


class GenBody(BaseModel):
    name: str
    params: dict = {}
    session_id: Optional[str] = None
    wait: bool = False          # external callers: block and return the finished result


class SessionBody(BaseModel):
    name: str = "Untitled"


class EditBody(BaseModel):
    filename: str
    subfolder: str = ""
    instruction: str
    session_id: Optional[str] = None


class PresetBody(BaseModel):
    filename: str
    subfolder: str = ""
    preset: str                       # replace_background | camera_angle | ...
    arg: str = ""                     # e.g. the new background, or a camera angle
    session_id: Optional[str] = None


class UpscaleBody(BaseModel):
    filename: str
    subfolder: str = ""
    scale_by: float = 1.0             # 1.0 = 4x (native), 0.5 = 2x
    session_id: Optional[str] = None


class BlenderBody(BaseModel):
    scene: str = "city_flythrough"    # a key in blender.scenes.SCENE_PRESETS
    prompt: str = ""                  # blank -> the scene's default style prompt
    frames: int = 33                  # 4n+1 for Wan (33/49/81)
    steps: int = 25
    seed: int = 7
    model: str = "wan2.1_vace_1.3B_fp16.safetensors"   # or ..._14B_... for best quality
    wait: bool = False                # True -> block and return the finished asset (for callers)
    session_id: Optional[str] = None


class AnimateBody(BaseModel):
    filename: str
    subfolder: str = ""
    session_id: Optional[str] = None
    motion: int = 110


def _stage_source(eng: Engine, filename: str, subfolder: str, prefix: str) -> str:
    """Copy an output image into ComfyUI input/ and return its input-relative path."""
    src = Path(eng.cfg.comfyui.output_dir) / subfolder / filename
    if not src.exists():
        raise HTTPException(404, "source image not found")
    in_dir = Path(eng.cfg.comfyui.input_dir)
    (in_dir / "backlot").mkdir(parents=True, exist_ok=True)
    staged = f"backlot/{prefix}_{uuid.uuid4().hex[:8]}.png"
    shutil.copy(src, in_dir / staged)
    return staged


@app.get("/api/workflows")
async def workflows():
    return {"workflows": [c.public_info() for c in _eng().registry.list()]}


@app.get("/api/workflows/{name}")
async def workflow(name: str):
    try:
        cap = _eng().registry.get(name)
    except Exception:
        raise HTTPException(404, "unknown workflow")
    return {
        "name": cap.name, "title": cap.title, "kind": cap.kind,
        "inject": [s.model_dump() for s in cap.inject],
        "client_outputs": cap.client_outputs,
    }


@app.post("/api/generate")
async def generate(body: GenBody):
    eng = _eng()
    eng.ensure_started()
    try:
        res = await eng.jobs.run_workflow(body.name, body.params or {}, wait=False)
    except Exception as ex:
        raise HTTPException(400, str(ex))
    run_id = res["run_id"]
    if body.wait:                       # external caller: block, persist, return full result
        status = await eng.jobs.wait_for(run_id)
        state.store.save(run_id, body.name, body.params or {}, status, time.time(),
                         body.session_id)
        return status
    asyncio.create_task(_persist(run_id, body.name, body.params or {}, body.session_id))
    return res


async def _persist(run_id: str, name: str, params: dict,
                   session_id: Optional[str] = None) -> None:
    try:
        status = await _eng().jobs.wait_for(run_id)
        state.store.save(run_id, name, params, status, time.time(), session_id)
    except Exception:
        pass


async def _run_edit(eng: Engine, filename: str, subfolder: str, instruction: str,
                    session_id: Optional[str]) -> dict:
    """Stage a source image and run a FLUX-Kontext edit by instruction."""
    eng.ensure_started()
    staged = _stage_source(eng, filename, subfolder, "edit")
    params = {"image": staged, "instruction": instruction}
    try:
        res = await eng.jobs.run_workflow("edit_kontext", params, wait=False)
    except Exception as ex:
        raise HTTPException(400, str(ex))
    asyncio.create_task(_persist(res["run_id"], "edit_kontext", params, session_id))
    return res


@app.post("/api/edit")
async def edit(body: EditBody):
    """Edit an existing output image by instruction (FLUX Kontext)."""
    return await _run_edit(_eng(), body.filename, body.subfolder, body.instruction,
                           body.session_id)


@app.post("/api/edit/preset")
async def edit_preset(body: PresetBody):
    """One-click Kontext preset — Replace Background, Camera Angle, … (OpenArt-style)."""
    try:
        instruction = presets.instruction(body.preset, body.arg)
    except KeyError as ex:
        raise HTTPException(400, str(ex))
    return await _run_edit(_eng(), body.filename, body.subfolder, instruction,
                           body.session_id)


class AssistBody(BaseModel):
    kind: str = "image"               # image | video | audio | 3d
    prompt: str = ""                  # current draft (blank -> invent a fresh one)


@app.post("/api/assist")
async def assist_prompt(body: AssistBody):
    """✨ Write or improve a generation prompt for the given medium (local LLM)."""
    from .. import assist as assist_mod
    from ..engine.llm import LLMClient
    eng = _eng()
    try:
        text = assist_mod.assist_prompt(LLMClient.from_config(eng.cfg), body.kind, body.prompt)
    except Exception as ex:
        raise HTTPException(502, f"assist failed: {ex}")
    return {"prompt": text}


@app.get("/api/blender/scenes")
async def blender_scenes():
    """The 3D scene catalogue for the UI / programmatic callers."""
    from ..blender import scenes as bscenes
    return {"scenes": bscenes.scene_list()}


@app.post("/api/blender/animate")
async def blender_animate(body: BlenderBody):
    """Render a Blender depth animation, then Wan-VACE restyle it into a coherent clip.

    The same two-stage flow the CLI test scripts use — now callable from the UI
    (wait=False → stream progress) and by other projects / Claude Code (wait=True →
    block and return the finished video asset). Registered as a gallery run.
    """
    from ..blender import runner as bl, scenes as bscenes
    eng = _eng(); eng.ensure_started()
    preset = bscenes.SCENE_PRESETS.get(body.scene)
    if preset is None:
        raise HTTPException(400, f"unknown scene: {body.scene}")
    prompt = body.prompt.strip() or preset["prompt"]
    depth_dir = Path(eng.cfg.paths.runs) / "blender_ui" / f"{body.scene}_{uuid.uuid4().hex[:8]}"
    depth_dir.mkdir(parents=True, exist_ok=True)
    r = await asyncio.to_thread(         # Blender depth (blocking subprocess) off the loop
        bl.run_script, bscenes.depth_sequence(body.scene),
        args=[str(depth_dir), body.frames, preset["near"], preset["far"]], timeout=600)
    if not r.ok:
        raise HTTPException(500, "blender depth render failed: " + (r.stderr[-300:] or "no output"))
    n = len(list(depth_dir.glob("depth_*.png")))
    if not n:
        raise HTTPException(500, "blender produced no depth frames")
    params = {"positive_prompt": prompt, "control_dir": str(depth_dir).replace("\\", "/"),
              "length": n, "strength": 1.0, "steps": body.steps, "seed": body.seed,
              "fps": 16.0, "model": body.model}
    try:
        res = await eng.jobs.run_workflow("vace_depth_video", params,
                                          wait=body.wait, timeout_s=max(1800, eng.cfg.timeouts.video_job_s))
    except Exception as ex:
        raise HTTPException(400, str(ex))
    if body.wait:
        state.store.save(res["run_id"], "vace_depth_video", params, res, time.time(), body.session_id)
        return res
    asyncio.create_task(_persist(res["run_id"], "vace_depth_video", params, body.session_id))
    return res


async def _run_on_image(workflow: str, filename: str, subfolder: str, prefix: str,
                        extra: dict, session_id: Optional[str]) -> dict:
    """Stage an existing output image into ComfyUI input/ and run `workflow` on it."""
    eng = _eng(); eng.ensure_started()
    params = {"image": _stage_source(eng, filename, subfolder, prefix), **extra}
    try:
        res = await eng.jobs.run_workflow(workflow, params, wait=False)
    except Exception as ex:
        raise HTTPException(400, str(ex))
    asyncio.create_task(_persist(res["run_id"], workflow, params, session_id))
    return res


@app.post("/api/animate")
async def animate(body: AnimateBody):
    """Animate an existing output image into a video clip (SVD image->video)."""
    return await _run_on_image("img2vid_svd", body.filename, body.subfolder, "anim",
                               {"motion_bucket_id": body.motion, "frames": 25, "steps": 18},
                               body.session_id)


@app.post("/api/upscale")
async def upscale(body: UpscaleBody):
    """AI super-resolution upscale of an existing image (Real-ESRGAN 4x, scale_by trims)."""
    return await _run_on_image("upscale_esrgan", body.filename, body.subfolder, "upscale",
                               {"scale_by": body.scale_by}, body.session_id)


@app.get("/api/jobs/{run_id}")
async def job(run_id: str):
    try:
        return _eng().jobs.get_status(run_id=run_id)
    except KeyError:
        raise HTTPException(404, "unknown run")


@app.get("/api/stream/{run_id}")
async def stream(run_id: str):
    async def gen():
        while True:
            try:
                st = _eng().jobs.get_status(run_id=run_id)
            except KeyError:
                yield "event: error\ndata: {}\n\n"
                return
            yield f"data: {json.dumps(st)}\n\n"
            if st["state"] in _TERMINAL:
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/assets")
async def assets(session_id: Optional[str] = None):
    if not state.store:
        return {"assets": []}
    return {"assets": state.store.list(session_id=session_id)}


@app.delete("/api/assets/{run_id}")
async def delete_asset(run_id: str):
    """Delete one creation (removes it from the gallery)."""
    if not state.store:
        raise HTTPException(503, "store not ready")
    return {"deleted": state.store.delete(run_id)}


class DeleteAssetsBody(BaseModel):
    ids: list[str]


@app.post("/api/assets/delete")
async def delete_assets(body: DeleteAssetsBody):
    """Bulk-delete creations by run id."""
    if not state.store:
        raise HTTPException(503, "store not ready")
    return {"deleted": state.store.delete_many(body.ids)}


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": state.sessions.list() if state.sessions else []}


@app.post("/api/sessions")
async def create_session(body: SessionBody):
    return state.sessions.create(body.name, time.time())


@app.patch("/api/sessions/{sid}")
async def rename_session(sid: str, body: SessionBody):
    sess = state.sessions.rename(sid, body.name)
    if sess is None:
        raise HTTPException(404, "unknown session")
    return sess


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    return {"deleted": state.sessions.delete(sid)}


@app.get("/")
async def index():
    return FileResponse(_STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


def main() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    # Env overrides let start-lan.bat bind 0.0.0.0 without editing engine.yaml.
    host = os.environ.get("COMFYUI_AI_WEB_HOST", cfg.server.web_host)
    port = int(os.environ.get("COMFYUI_AI_WEB_PORT", cfg.server.web_port))
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
