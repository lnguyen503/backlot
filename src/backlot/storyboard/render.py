"""Render a storyboard: card refs, frames-first stills, motion, assembly.

Consistency-first (TODO items 3-6). The key move, borrowed from the proven
`pipelines.consistent_video` R&D: when a panel features a character that already
has a locked reference card, we don't text-to-image it blind — we anchor it to
that reference with FLUX Kontext, so the same face/identity carries across panels
without drift. Panels with no character (or no ref yet) fall back to txt2img.

All functions are async and drive the SAME Engine the web/MCP layers use.
"""
from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np

from ..engine.runtime import Engine
from .models import Asset, AssetCard, Panel, Storyboard

# FLUX-friendly dimensions per aspect (multiples of 64).
_DIMS = {"portrait": (896, 1152), "landscape": (1152, 896), "square": (1024, 1024)}

# Appended to a Kontext edit so the anchored identity survives the transform.
_LOCK = ("Keep this exact same person — identical face, hair, and build. "
         "Correct human anatomy. Photorealistic, cinematic, natural light.")


def _out_abs(eng: Engine, asset: Asset) -> Path:
    return Path(eng.cfg.comfyui.output_dir) / (asset.subfolder or "") / asset.filename


def _stage(eng: Engine, asset: Asset, prefix: str) -> str:
    """Copy an output image into ComfyUI input/ and return its input-relative path."""
    src = _out_abs(eng, asset)
    in_dir = Path(eng.cfg.comfyui.input_dir) / "backlot"
    in_dir.mkdir(parents=True, exist_ok=True)
    rel = f"backlot/{prefix}_{uuid.uuid4().hex[:8]}.png"
    shutil.copy(src, Path(eng.cfg.comfyui.input_dir) / rel)
    return rel


def free_vram(eng: Engine) -> None:
    """Unload ComfyUI models + free VRAM. Call BETWEEN heavy InfiniteTalk/Wan
    runs: the model set is NOT released between calls and the 2nd heavy run in a
    session otherwise OOMs (silently killing the headless process)."""
    import httpx
    try:
        httpx.post(f"{eng.cfg.comfyui.base_url}/free",
                   json={"unload_models": True, "free_memory": True}, timeout=10)
    except httpx.HTTPError:
        pass


def _stage_file(eng: Engine, abs_path: Path, prefix: str, ext: str) -> str:
    """Copy any file into ComfyUI input/ and return its input-relative path."""
    in_dir = Path(eng.cfg.comfyui.input_dir) / "backlot"
    in_dir.mkdir(parents=True, exist_ok=True)
    rel = f"backlot/{prefix}_{uuid.uuid4().hex[:8]}.{ext.lstrip('.')}"
    shutil.copy(abs_path, Path(eng.cfg.comfyui.input_dir) / rel)
    return rel


def _asset_from_output(out: dict) -> Asset:
    return Asset(type=out.get("type", "image"), filename=out["filename"],
                 subfolder=out.get("subfolder", ""), url=out.get("url", ""))


def _style_suffix(sb: Storyboard) -> str:
    return f" {sb.style_notes}" if sb.style_notes else ""


async def render_card_ref(eng: Engine, sb: Storyboard, card: AssetCard,
                          model: str = "txt2img_flux") -> Asset:
    """Generate a clean reference still for a card (the consistency anchor)."""
    w, h = _DIMS.get(sb.aspect, _DIMS["landscape"])
    prompt = (card.prompt or card.description) + _style_suffix(sb)
    seed = card.seed if card.seed >= 0 else 42
    res = await eng.jobs.run_workflow(
        model, {"positive_prompt": prompt, "width": w, "height": h, "seed": seed},
        wait=True, timeout_s=eng.cfg.timeouts.image_job_s,
    )
    if res["state"] != "completed" or not res["outputs"]:
        raise RuntimeError(f"card ref failed: {res.get('error')}")
    card.ref = _asset_from_output(res["outputs"][0])
    card.source = "generated"
    card.seed = seed
    return card.ref


async def render_panel_still(eng: Engine, sb: Storyboard, panel: Panel,
                             model: str = "txt2img_flux") -> Asset:
    """Frames-first still for one panel. Anchors to a character ref via Kontext
    when one exists (consistency), else plain txt2img."""
    seed = panel.seed if panel.seed >= 0 else 100
    char = sb.primary_character(panel)
    if char and char.ref is not None:
        staged = _stage(eng, char.ref, "sb_anchor")
        res = await eng.jobs.run_workflow(
            "edit_kontext",
            {"image": staged,
             "instruction": f"{panel.image_prompt}{_style_suffix(sb)} {_LOCK}",
             "guidance": 2.5, "steps": 20, "seed": seed},
            wait=True, timeout_s=eng.cfg.timeouts.image_job_s,
        )
    else:
        w, h = _DIMS.get(sb.aspect, _DIMS["landscape"])
        res = await eng.jobs.run_workflow(
            model,
            {"positive_prompt": panel.image_prompt + _style_suffix(sb),
             "width": w, "height": h, "seed": seed},
            wait=True, timeout_s=eng.cfg.timeouts.image_job_s,
        )
    if res["state"] != "completed" or not res["outputs"]:
        raise RuntimeError(f"panel still failed: {res.get('error')}")
    panel.still = _asset_from_output(res["outputs"][0])
    panel.source = "generated"
    panel.seed = seed
    return panel.still


def set_panel_photo(eng: Engine, panel: Panel, data: bytes, ext: str = "png") -> Asset:
    """Use a user-provided image as a panel's still (TODO: per-panel photo source)."""
    out_dir = Path(eng.cfg.comfyui.output_dir) / "backlot-storyboards"
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = f"photo_{uuid.uuid4().hex[:8]}.{ext.lstrip('.')}"
    (out_dir / fn).write_bytes(data)
    url = eng.client.view_url(fn, "backlot-storyboards", "output")
    panel.still = Asset(type="image", filename=fn,
                        subfolder="backlot-storyboards", url=url)
    panel.source = "photo"
    return panel.still


# Wan-friendly dims per aspect (~720p, multiples of 16). Wan honors the motion prompt.
_WAN_DIMS = {"portrait": (720, 1280), "landscape": (1280, 720), "square": (960, 960)}

# img2vid backends. svd = robust, aspect-preserving, prompt-free, ~1s/clip (fast default).
# wan = honors the motion prompt, longer beats (81 frames ≈ 5s), respects board aspect.
_I2V = {
    "svd": ("img2vid_svd",
            lambda p, sb: {"motion_bucket_id": 110, "frames": 25, "steps": 18}),
    "wan14b_fast": ("img2vid_wan14b_fast",
                    lambda p, sb: {"positive_prompt": p.motion_prompt or "subtle natural motion",
                                   **dict(zip(("width", "height"),
                                             _WAN_DIMS.get(sb.aspect, _WAN_DIMS["landscape"]))),
                                   "length": 81}),
}


async def animate_panel(eng: Engine, sb: Storyboard, panel: Panel,
                        backend: str = "svd") -> Asset:
    """Turn a locked panel still into a motion clip (bridge to the video lane)."""
    if panel.still is None:
        raise RuntimeError("panel has no still to animate")
    staged = _stage(eng, panel.still, "sb_motion")
    workflow, params = _I2V.get(backend, _I2V["svd"])
    res = await eng.jobs.run_workflow(
        workflow, {"image": staged, **params(panel, sb)},
        wait=True, timeout_s=eng.cfg.timeouts.video_job_s,
    )
    if res["state"] != "completed" or not res["outputs"]:
        raise RuntimeError(f"animate failed: {res.get('error')}")
    panel.video = _asset_from_output(res["outputs"][0])
    return panel.video


# assemble() lives in .assemble (ffmpeg-based, mp4-aware, audio-preserving);
# re-exported here so callers keep using render.assemble(eng, sb, fps=...).
from .assemble import _has_audio, assemble  # noqa: E402,F401


def _video_seconds(path: Path, default: float = 10.0) -> float:
    try:
        r = imageio.get_reader(str(path))
        meta = r.get_meta_data()
        dur = meta.get("duration")
        if not dur:
            dur = r.count_frames() / float(meta.get("fps", 24) or 24)
        r.close()
        return max(1.0, float(dur))
    except Exception:
        return default


async def _acestep_bed(eng: Engine, sb: Storyboard, tags: str, lyrics: str,
                       seconds: Optional[float]) -> Asset:
    """Run ACE-Step for one audio bed (music/ambience/SFX), sized to the sequence."""
    if seconds is None:
        seconds = _video_seconds(_out_abs(eng, sb.assembled)) if sb.assembled else 20.0
    res = await eng.jobs.run_workflow(
        "music_acestep",
        {"tags": tags, "lyrics": lyrics, "seconds": round(seconds + 1.0, 1), "steps": 50},
        wait=True, timeout_s=eng.cfg.timeouts.video_job_s,
    )
    if res["state"] != "completed" or not res["outputs"]:
        raise RuntimeError(f"acestep failed: {res.get('error')}")
    return _asset_from_output(res["outputs"][0])


async def score(eng: Engine, sb: Storyboard, tags: str, lyrics: str = "",
                seconds: Optional[float] = None) -> Asset:
    """Generate a music track (ACE-Step) sized to the assembled sequence."""
    sb.score = await _acestep_bed(eng, sb, tags, lyrics, seconds)
    return sb.score


async def ambient(eng: Engine, sb: Storyboard, tags: str,
                  seconds: Optional[float] = None) -> Asset:
    """A low ambient/SFX bed (ACE-Step) — the 3rd ducked leg under music + voice."""
    sb.ambient = await _acestep_bed(eng, sb, tags, "", seconds)
    return sb.ambient


def _tts(eng: Engine, text: str, ref_wav: str, fn: str,
         backend: str = "chatterbox") -> Asset:
    """Clone `ref_wav` and speak `text` -> WAV in the storyboards output dir.

    Subprocesses `tests/voice_clone.py` in `.venv-cbx` (chatterbox) / `.venv-tts` (f5),
    mirroring the talkshow lane. The voice venv ships CPU-only torch -> --device cpu.
    """
    root = Path(__file__).resolve().parents[3]
    venv = {"chatterbox": root / ".venv-cbx/Scripts/python.exe",
            "f5": root / ".venv-tts/Scripts/python.exe"}[backend]
    out_dir = Path(eng.cfg.comfyui.output_dir) / "backlot-storyboards"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / fn
    cmd = [str(venv), str(root / "tests/voice_clone.py"), "--backend", backend,
           "--ref", str(ref_wav), "--text", text, "--out", str(out_path),
           "--device", "cpu"]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    url = eng.client.view_url(fn, "backlot-storyboards", "output")
    return Asset(type="audio", filename=fn, subfolder="backlot-storyboards", url=url)


_KOKORO: dict = {}   # lang_code -> KPipeline (each loads a model; cache them)


def _kokoro(lang_code: str):
    if lang_code not in _KOKORO:
        import warnings
        warnings.filterwarnings("ignore")
        from kokoro import KPipeline
        _KOKORO[lang_code] = KPipeline(lang_code=lang_code)
    return _KOKORO[lang_code]


def tts_kokoro(eng: Engine, text: str, voice: str, fn: str) -> Asset:
    """Built-in multi-speaker TTS (Kokoro, in the main venv) — distinct male/female
    voices with NO reference clip. Voice ids: am_*/af_* (American), bm_*/bf_*
    (British). Runs in-process; lang_code follows the voice's region prefix."""
    import numpy as _np
    import soundfile as _sf
    lang = "b" if voice[:1] == "b" else "a"
    chunks = [a.detach().cpu().numpy() if hasattr(a, "detach") else _np.asarray(a)
              for _, _, a in _kokoro(lang)(text, voice=voice)]
    wav = _np.concatenate(chunks)
    out_dir = Path(eng.cfg.comfyui.output_dir) / "backlot-storyboards"
    out_dir.mkdir(parents=True, exist_ok=True)
    _sf.write(str(out_dir / fn), wav, 24000)
    url = eng.client.view_url(fn, "backlot-storyboards", "output")
    return Asset(type="audio", filename=fn, subfolder="backlot-storyboards", url=url)


def narrate(eng: Engine, sb: Storyboard, text: str, ref_wav: Optional[str] = None,
            backend: str = "chatterbox") -> Asset:
    """Generate a single voiceover track for the whole short (cloned from ref_wav).
    With no ref_wav, a SYNTHETIC Kokoro reference is generated and cloned
    expressively (privacy-safe — never a real person's voice)."""
    if not ref_wav:
        ref = tts_kokoro(eng, "A clear neutral line for voice cloning.", "af_heart",
                         f"{sb.id}_narr_ref.wav")
        ref_wav = str(_out_abs(eng, ref))
    sb.narration = _tts(eng, text, ref_wav, f"{sb.id}_vo_{uuid.uuid4().hex[:6]}.wav", backend)
    return sb.narration


async def animate_talk(eng: Engine, sb: Storyboard, panel: Panel, voice: str,
                       tts: str = "kokoro", width: int = 480, height: int = 832,
                       prompt: Optional[str] = None) -> Asset:
    """Make a panel's character SPEAK its `dialogue`, lip-synced (InfiniteTalk).
    tts="kokoro" -> `voice`=a Kokoro id; **"expressive"** -> `voice`=a Kokoro id, cloned
    expressively via Chatterbox off a SYNTHETIC reference (privacy-safe — never a real
    person; the locked recipe); "chatterbox"/"f5" -> `voice`=a reference WAV to clone."""
    if panel.still is None:
        raise RuntimeError("panel has no still to animate")
    if not panel.dialogue.strip():
        raise RuntimeError("panel has no dialogue to speak")
    line_fn = f"{sb.id}_{panel.id}_line.wav"
    if tts == "kokoro":
        panel.line_audio = tts_kokoro(eng, panel.dialogue, voice, line_fn)
    elif tts == "expressive":                     # synthetic Kokoro ref -> Chatterbox
        ref = tts_kokoro(eng, "A clear neutral line for voice cloning.", voice, f"ref_{line_fn}")
        panel.line_audio = _tts(eng, panel.dialogue, str(_out_abs(eng, ref)), line_fn, "chatterbox")
    else:
        panel.line_audio = _tts(eng, panel.dialogue, voice, line_fn, tts)
    img_rel = _stage(eng, panel.still, "sb_talk")
    aud_rel = _stage_file(eng, _out_abs(eng, panel.line_audio), "sb_line", "wav")
    look = prompt or (f"two friends chatting at a cozy coffee shop, {sb.style_notes}, "
                      "natural expressive face, photorealistic, cinematic")
    res = await eng.jobs.run_workflow(
        "talkhost_infinitetalk",
        {"image": img_rel, "audio": aud_rel, "width": width, "height": height,
         "prompt": look, "steps": 6},
        wait=True, timeout_s=max(1800, eng.cfg.timeouts.video_job_s),
    )
    if res["state"] != "completed" or not res["outputs"]:
        raise RuntimeError(f"talk failed: {res.get('error')}")
    panel.video = _asset_from_output(res["outputs"][0])
    return panel.video


def face_relock(eng: Engine, sb: Storyboard, panel: Panel,
                models_dir: Optional[str] = None) -> Optional[Asset]:
    """Re-lock the speaker's identity onto every frame of panel.video (insightface
    swap + GFPGAN restore). Preserves expression/lip-motion, locks the face to the
    character's reference. No-op (returns existing video) if no character ref."""
    from ..faceswap import FaceLocker, default_models
    if panel.video is None:
        raise RuntimeError("panel has no video to re-lock")
    char = sb.primary_character(panel)
    if char is None or char.ref is None:
        return panel.video
    ref_path = _out_abs(eng, char.ref)
    inswapper, gfpgan = default_models(models_dir) if models_dir else default_models()
    locker = FaceLocker(str(ref_path), inswapper, gfpgan)

    src = _out_abs(eng, panel.video)
    reader = imageio.get_reader(str(src))
    fps = reader.get_meta_data().get("fps", 25)
    out_dir = Path(eng.cfg.comfyui.output_dir) / "backlot-storyboards"
    fn = f"{sb.id}_{panel.id}_relock.mp4"
    writer = imageio.get_writer(str(out_dir / fn), fps=fps, codec="libx264",
                                quality=8, macro_block_size=1)
    try:
        for frame in reader:
            bgr = np.asarray(frame)[:, :, ::-1].copy()
            bgr, _ = locker.process_frame(bgr)
            writer.append_data(bgr[:, :, ::-1])
    finally:
        writer.close()
        reader.close()
    url = eng.client.view_url(fn, "backlot-storyboards", "output")
    panel.video = Asset(type="video", filename=fn,
                        subfolder="backlot-storyboards", url=url)
    return panel.video


def _audio_legs(sb: Storyboard, has_voice: bool, music_vol: float,
                ambient_vol: float) -> list[tuple[Asset, float]]:
    """External beds to mix under the video, in ffmpeg-input order: narration is
    voice (full); music ducks under any voice; ambient/SFX always sits low."""
    legs: list[tuple[Asset, float]] = []
    if sb.narration is not None:
        legs.append((sb.narration, 1.0))
    if sb.score is not None:
        legs.append((sb.score, music_vol if has_voice else 1.0))
    if sb.ambient is not None:
        legs.append((sb.ambient, ambient_vol))
    return legs


def mux_audio(eng: Engine, sb: Storyboard, music_vol: float = 0.30,
              ambient_vol: float = 0.15) -> Asset:
    """Mux score + narration + ambient onto sb.assembled, UNDER any embedded dialogue.

    Talking panels already carry per-panel speech; we keep that at full volume and
    duck the beds beneath it via amix (ambient stays low under all). Replaces sb.assembled.
    """
    if sb.assembled is None:
        raise RuntimeError("assemble the sequence before muxing audio")
    video = _out_abs(eng, sb.assembled)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    # legs = (ffmpeg stream spec, volume). Input 0 is the video (its own dialogue).
    ext: list[Path] = []
    legs: list[tuple[str, float]] = []
    embedded = _has_audio(ff, video)
    has_voice = embedded or sb.narration is not None
    if embedded:
        legs.append(("0:a", 1.0))
    for asset, vol in _audio_legs(sb, has_voice, music_vol, ambient_vol):
        ext.append(_out_abs(eng, asset)); legs.append((f"{len(ext)}:a", vol))
    if not ext:
        raise RuntimeError("no audio (score/narration/ambient) to mux")

    out_dir = Path(eng.cfg.comfyui.output_dir) / "backlot-storyboards"
    fn = f"{sb.id}_mixed_{uuid.uuid4().hex[:6]}.mp4"
    out_path = out_dir / fn
    cmd = [ff, "-y", "-i", str(video)]
    for path in ext:
        cmd += ["-i", str(path)]
    lg = "".join(f"[{spec}]volume={v}[a{i}];" for i, (spec, v) in enumerate(legs))
    mix = "".join(f"[a{i}]" for i in range(len(legs)))
    flt = f"{lg}{mix}amix=inputs={len(legs)}:duration=longest:normalize=0[aout]"
    cmd += ["-filter_complex", flt, "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)]
    subprocess.run(cmd, check=True, capture_output=True)
    url = eng.client.view_url(fn, "backlot-storyboards", "output")
    sb.assembled = Asset(type="video", filename=fn,
                         subfolder="backlot-storyboards", url=url)
    return sb.assembled
