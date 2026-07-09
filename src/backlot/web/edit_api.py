"""Image-editing endpoints that act on an existing gallery image (mask inpaint / VFX).

Shares app.py's engine + persistence via `init_edit` (called from the app lifespan)
so app.py stays lean. New per-image editing features land here rather than bloating
the main app module.
"""
from __future__ import annotations

import asyncio
import base64
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .. import presets

router = APIRouter()


class _Deps:
    eng = None          # () -> Engine
    stage = None        # (eng, filename, subfolder, prefix) -> staged input path
    persist = None      # (run_id, name, params, session_id) coroutine
    save = None         # (run_id, name, params, status, ts, session_id) — RunStore.save
    sb_store = None     # StoryboardStore (for "send to storyboard")


D = _Deps()


def init_edit(eng_getter, stage_source, persist, save, sb_store) -> None:
    """Wire in app.py's shared helpers (called once from the app lifespan)."""
    D.eng, D.stage, D.persist, D.save, D.sb_store = eng_getter, stage_source, persist, save, sb_store


class InpaintBody(BaseModel):
    filename: str
    subfolder: str = ""
    mask: str                         # PNG data URL; alpha channel = region to regenerate
    prompt: str = ""
    denoise: float = 1.0
    session_id: Optional[str] = None


def _stage_mask(eng, data_url: str) -> str:
    """Decode a base64 PNG mask, feather its edges (soft inpaint blend), and stage it."""
    import io
    from PIL import Image, ImageFilter
    b64 = data_url.split(",", 1)[-1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(10))   # feather white-on-black -> soft edges
    in_dir = Path(eng.cfg.comfyui.input_dir) / "backlot"
    in_dir.mkdir(parents=True, exist_ok=True)
    rel = f"backlot/mask_{uuid.uuid4().hex[:8]}.png"
    img.save(Path(eng.cfg.comfyui.input_dir) / rel)
    return rel


@router.post("/api/lipsync")
async def lipsync(audio: UploadFile, filename: str = Form(...), subfolder: str = Form(""),
                  prompt: str = Form(""), session_id: str = Form("")):
    """Lip-sync a gallery portrait to an audio clip -> talking video (InfiniteTalk)."""
    from ..storyboard import render
    eng = D.eng()
    eng.ensure_started()
    img_rel = D.stage(eng, filename, subfolder, "lipimg")
    in_dir = Path(eng.cfg.comfyui.input_dir) / "backlot"
    in_dir.mkdir(parents=True, exist_ok=True)
    ext = (audio.filename or "a.wav").rsplit(".", 1)[-1].lower()
    aud_rel = f"backlot/lipaud_{uuid.uuid4().hex[:8]}.{ext}"
    (Path(eng.cfg.comfyui.input_dir) / aud_rel).write_bytes(await audio.read())
    render.free_vram(eng)                       # InfiniteTalk is heavy — free VRAM first
    look = prompt.strip() or "a person talking, natural expressive face, photorealistic, cinematic"
    params = {"image": img_rel, "audio": aud_rel, "width": 480, "height": 832,
              "prompt": look, "steps": 6}
    try:
        res = await eng.jobs.run_workflow("talkhost_infinitetalk", params, wait=False,
                                          timeout_s=max(1800, eng.cfg.timeouts.video_job_s))
    except Exception as ex:
        raise HTTPException(400, str(ex))
    asyncio.create_task(D.persist(res["run_id"], "talkhost_infinitetalk", params, session_id or None))
    return res


class ToStoryboardBody(BaseModel):
    filename: str
    subfolder: str = ""
    title: str = ""


@router.post("/api/to-storyboard")
async def to_storyboard(body: ToStoryboardBody):
    """Seed a new storyboard with a gallery image as its first panel's still."""
    from ..storyboard.models import Asset, Panel, Storyboard
    eng = D.eng()
    url = eng.client.view_url(body.filename, body.subfolder, "output")
    still = Asset(type="image", filename=body.filename, subfolder=body.subfolder, url=url)
    sb = Storyboard(title=body.title.strip() or "From image",
                    panels=[Panel(scene="Opening shot", source="photo", still=still)])
    D.sb_store.save(sb)
    return {"board_id": sb.id, "title": sb.title}


class ToMp4Body(BaseModel):
    filename: str
    subfolder: str = ""
    session_id: Optional[str] = None


@router.post("/api/tomp4")
async def tomp4(body: ToMp4Body):
    """Transcode a gallery animated-webp clip to a shareable mp4 (added to the gallery)."""
    from ..storyboard.assemble import _to_mp4
    eng = D.eng()
    src = Path(eng.cfg.comfyui.output_dir) / (body.subfolder or "") / body.filename
    if not src.exists():
        raise HTTPException(404, "source not found")
    if src.suffix.lower() != ".webp":
        return {"url": eng.client.view_url(body.filename, body.subfolder, "output"), "converted": False}
    out_dir = Path(eng.cfg.comfyui.output_dir) / "backlot-storyboards"
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = f"{Path(body.filename).stem}_{uuid.uuid4().hex[:6]}.mp4"
    await asyncio.to_thread(_to_mp4, src, out_dir / fn, 24)
    url = eng.client.view_url(fn, "backlot-storyboards", "output")
    status = {"state": "completed", "error": None,
              "outputs": [{"type": "video", "filename": fn, "subfolder": "backlot-storyboards", "url": url}]}
    D.save("mp4_" + uuid.uuid4().hex[:12], "tomp4", {"src": body.filename}, status,
           time.time(), body.session_id or None)
    return {"url": url, "converted": True}


class VoiceoverBody(BaseModel):
    text: str
    voice: str = "af_heart"           # a Kokoro voice id
    session_id: Optional[str] = None


@router.post("/api/voiceover")
async def voiceover(body: VoiceoverBody):
    """Text -> spoken voiceover (Kokoro TTS, in-process) -> an audio asset in the gallery."""
    from ..storyboard import render
    if not body.text.strip():
        raise HTTPException(400, "text is required")
    eng = D.eng()
    fn = f"vo_{uuid.uuid4().hex[:8]}.wav"
    asset = await asyncio.to_thread(render.tts_kokoro, eng, body.text.strip(), body.voice, fn)
    run_id = "vo_" + uuid.uuid4().hex[:12]
    status = {"state": "completed", "outputs": [asset.model_dump()], "error": None}
    D.save(run_id, "voiceover", {"text": body.text.strip(), "voice": body.voice},
           status, time.time(), body.session_id or None)
    return {"run_id": run_id, "state": "completed", "outputs": [asset.model_dump()]}


@router.post("/api/motionsync")
async def motionsync(video: UploadFile, prompt: str = Form(""), frames: int = Form(25),
                     model: str = Form("wan2.1_vace_1.3B_fp16.safetensors"),
                     session_id: str = Form("")):
    """Motion transfer: a reference clip -> Depth-Anything depth -> Wan VACE regenerates
    a new subject following that motion. Returns a run_id (streams like any video job)."""
    from .. import motionsync as ms
    eng = D.eng()
    eng.ensure_started()
    base = Path(eng.cfg.paths.runs) / "motionsync"
    base.mkdir(parents=True, exist_ok=True)
    tag = uuid.uuid4().hex[:8]
    vid = base / f"ref_{tag}.mp4"
    vid.write_bytes(await video.read())
    ctrl = base / f"ctrl_{tag}"
    n = await asyncio.to_thread(ms.build_depth_control, str(vid), str(ctrl), frames)
    if not n:
        raise HTTPException(400, "could not read frames from the reference video")
    params = {"positive_prompt": prompt.strip() or "a person, cinematic, detailed",
              "control_dir": str(ctrl).replace("\\", "/"), "length": n, "strength": 1.0,
              "steps": 20, "seed": 7, "fps": 16.0, "model": model}
    try:
        res = await eng.jobs.run_workflow("vace_depth_video", params, wait=False,
                                          timeout_s=max(1800, eng.cfg.timeouts.video_job_s))
    except Exception as ex:
        raise HTTPException(400, str(ex))
    asyncio.create_task(D.persist(res["run_id"], "vace_depth_video", params, session_id or None))
    return res


class TurnaroundBody(BaseModel):
    filename: str
    subfolder: str = ""
    views: int = 4                    # 4 or 6 rotated views
    session_id: Optional[str] = None


@router.post("/api/turnaround")
async def turnaround(body: TurnaroundBody):
    """Generate N rotated views of the subject from one image (Kontext rotation set)."""
    eng = D.eng()
    eng.ensure_started()
    staged = D.stage(eng, body.filename, body.subfolder, "turn")
    run_ids = []
    for v in presets.turnaround_views(body.views):
        params = {"image": staged, "instruction": v["instruction"]}
        try:
            res = await eng.jobs.run_workflow("edit_kontext", params, wait=False)
        except Exception as ex:
            raise HTTPException(400, str(ex))
        asyncio.create_task(D.persist(res["run_id"], "edit_kontext", params, body.session_id))
        run_ids.append(res["run_id"])
    return {"run_ids": run_ids}


@router.post("/api/inpaint")
async def inpaint(body: InpaintBody):
    """Regenerate only the masked region of an image from a prompt (SDXL masked latent)."""
    eng = D.eng()
    eng.ensure_started()
    params = {"image": D.stage(eng, body.filename, body.subfolder, "inpaint"),
              "mask": _stage_mask(eng, body.mask), "prompt": body.prompt,
              "denoise": body.denoise}
    try:
        res = await eng.jobs.run_workflow("inpaint_sdxl", params, wait=False)
    except Exception as ex:
        raise HTTPException(400, str(ex))
    asyncio.create_task(D.persist(res["run_id"], "inpaint_sdxl", params, body.session_id))
    return res
