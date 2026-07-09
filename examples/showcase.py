"""[SHOWCASE] Blender (precise camera + depth) -> Wan VACE restyle. The "same camera move,
3 different worlds" control demo (camera-control demo). FIX: force Cycles on OptiX/CUDA + samples=1 for the
depth pass (headless GPU — the MCP tool hung on CPU under --factory-startup).
  python showcase.py depth <scene> <frames>          # step 1a: GPU depth only, report time
  python showcase.py clip  <scene> <frames> <world>  # step 1: depth->VACE one world
  python showcase.py set   <scene> <frames>          # step 2: 3 worlds + grid
"""
from __future__ import annotations
import json, sys, time, uuid
from pathlib import Path
import httpx, imageio_ffmpeg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backlot.blender import runner as bl, scenes

COMFY = "http://127.0.0.1:8188"
FF = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = Path(__file__).resolve().parents[1] / "runs/showcase"
VACE_T = Path(__file__).resolve().parents[1] / "templates/vace_depth_video.api.json"

# force GPU Cycles for the depth pass (injected right after set_engine)
GPU_SNIPPET = r'''
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
_prefs = bpy.context.preferences.addons['cycles'].preferences
_ok = 0
for _dt in ('OPTIX','CUDA'):
    try:
        _prefs.compute_device_type = _dt
        _prefs.get_devices()
        _ok = sum(1 for d in _prefs.devices if d.type == _dt)
        if _ok: break
    except Exception as e:
        print('GPU_TRY_FAIL', _dt, e)
for d in _prefs.devices:
    d.use = (d.type in ('OPTIX','CUDA'))
sc.cycles.device = 'GPU'
sc.cycles.samples = 1
try: sc.cycles.use_denoising = False
except Exception: pass
print('GPU_CFG', _prefs.compute_device_type, 'gpu_devices=', _ok,
      'enabled=', [d.name for d in _prefs.devices if d.use])
eng = sc.render.engine
'''


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def render_depth(scene: str, frames: int) -> tuple[Path, float]:
    pre = scenes.SCENE_PRESETS[scene]
    ddir = ROOT / scene / "depth"
    ddir.mkdir(parents=True, exist_ok=True)
    script = scenes.depth_sequence(scene).replace(
        "eng = set_engine()\n", "eng = set_engine()\n" + GPU_SNIPPET, 1
    ).replace("sc.render.resolution_x = 768; sc.render.resolution_y = 768",
              "sc.render.resolution_x = 832; sc.render.resolution_y = 480")  # 16:9 for a cinematic clip
    t0 = time.time()
    # factory_startup=False so the machine's GPU prefs load too (belt + suspenders w/ in-script enable)
    r = bl.run_script(script, args=[str(ddir), frames, pre["near"], pre["far"]],
                      factory_startup=False, timeout=900)
    dt = time.time() - t0
    for ln in r.stdout.splitlines():
        if ln.startswith(("GPU_CFG", "GPU_TRY_FAIL", "SEQ_DONE", "ENGINE_USED")):
            log("  " + ln.strip())
    pngs = sorted(ddir.glob("depth_*.png"))
    if not r.ok or not pngs:
        log(f"  !! depth render failed rc={r.returncode} pngs={len(pngs)}")
        log("  stderr tail: " + " ".join(r.stderr.splitlines()[-4:]))
        raise RuntimeError("depth render failed")
    log(f"  DEPTH {scene}: {len(pngs)} frames in {dt:.0f}s ({dt/len(pngs):.1f}s/frame)")
    return ddir, dt


def vace(control_dir: Path, prompt: str, out: Path, length: int, w=832, h=480, steps=25) -> float:
    g = json.loads(VACE_T.read_text(encoding="utf-8"))
    g["4"]["inputs"]["text"] = prompt
    g["7"]["inputs"]["folder"] = str(control_dir)
    g["7"]["inputs"]["width"] = w; g["7"]["inputs"]["height"] = h
    g["8"]["inputs"]["length"] = length; g["8"]["inputs"]["width"] = w; g["8"]["inputs"]["height"] = h
    g["9"]["inputs"]["steps"] = steps
    t0 = time.time()
    pid = httpx.post(f"{COMFY}/prompt", json={"prompt": g}, timeout=120).json()["prompt_id"]
    while time.time() - t0 < 1800:
        h5 = httpx.get(f"{COMFY}/history/{pid}", timeout=30).json().get(pid)
        if h5 and h5.get("status", {}).get("status_str") == "success":
            for _, o in h5["outputs"].items():
                for it in o.get("gifs", []) + o.get("video", []) + o.get("images", []):
                    if str(it["filename"]).endswith((".mp4", ".webp")):
                        raw = httpx.get(f"{COMFY}/view?filename={it['filename']}&subfolder={it.get('subfolder','')}&type=output", timeout=600).content
                        tmp = out.with_suffix(".raw" + Path(it["filename"]).suffix); tmp.write_bytes(raw)
                        # Wan/VACE emits animated webp; ffmpeg 7.x can't decode it -> imageio transcode
                        from backlot.storyboard.assemble import _to_mp4
                        if tmp.suffix.lower() == ".webp":
                            _to_mp4(tmp, out, 16)
                        else:
                            import subprocess
                            subprocess.run([FF, "-y", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt",
                                            "yuv420p", str(out)], check=True, capture_output=True)
                        tmp.unlink(missing_ok=True)
            break
        if h5 and h5.get("status", {}).get("status_str") == "error":
            raise RuntimeError("VACE comfy error")
        time.sleep(10)
    dt = time.time() - t0
    log(f"  VACE -> {out.name} in {dt:.0f}s")
    return dt


# radically different worlds on the SAME camera move (the showcase morph)
WORLDS = {
    "city_flythrough": {
        "neon": ("a vibrant neon cyberpunk city street at night, glowing pink and cyan signs, bright neon "
                 "glow, rain-soaked reflective street, blade runner, cinematic, ultra detailed, high contrast"),
        "temple": ("an ancient stone temple corridor, warm flickering torchlight, mossy carved sandstone "
                   "pillars, golden dust motes floating in the air, atmospheric volumetric light, "
                   "photorealistic, cinematic, ultra detailed"),
        "jungle": ("lush overgrown jungle ruins, thick green vines and moss covering stone pillars, bright "
                   "sunbeams through the canopy, ferns and tropical plants, misty morning haze, "
                   "photorealistic, cinematic, ultra detailed"),
        "underwater": ("a sunken underwater temple canyon, deep blue water, god-rays of sunlight from above, "
                       "colorful coral and seaweed on the pillars, drifting bubbles, caustic light, "
                       "photorealistic, cinematic, ultra detailed"),
        "lava": ("a volcanic cavern corridor, rivers of glowing molten lava between dark basalt pillars, "
                 "floating orange embers, intense fiery orange glow, dramatic, photorealistic, cinematic, "
                 "ultra detailed, high contrast"),
    },
    "monkey_orbit": {
        "marble": scenes.SCENE_PRESETS["monkey_orbit"]["prompt"],
        "gold": ("a polished solid gold statue bust on a pedestal, gleaming reflective gold, dramatic museum "
                 "spotlight, dark background, luxurious, photorealistic, ultra detailed, sharp focus"),
    },
}
MORPH_ORDER = ["neon", "temple", "jungle", "underwater", "lava"]


def upscale_720(src: Path, dst: Path):
    import subprocess
    subprocess.run([FF, "-y", "-i", str(src), "-vf", "scale=1280:720:flags=lanczos", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", str(dst)], check=True, capture_output=True)


def build_set(scene: str, frames: int):
    ddir, d_dt = render_depth(scene, frames)
    outdir = ROOT / scene; times = {}
    worlds = WORLDS.get(scene, {"default": scenes.SCENE_PRESETS[scene]["prompt"]})
    clips = []
    for wk, prompt in worlds.items():
        out = outdir / f"{scene}_{wk}.mp4"
        times[wk] = vace(ddir, prompt, out, frames)
        up = outdir / f"{scene}_{wk}_720.mp4"
        upscale_720(out, up); clips.append(out)
    # side-by-side grid (native 3-up)
    import subprocess
    grid = outdir / f"{scene}_grid.mp4"
    ins = []
    for c in clips: ins += ["-i", str(c)]
    n = len(clips)
    fc = "".join(f"[{i}:v]" for i in range(n)) + f"hstack=inputs={n}[v]"
    subprocess.run([FF, "-y", *ins, "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18", str(grid)], check=True, capture_output=True)
    log(f"SET done: depth {d_dt:.0f}s + VACE {times} -> {outdir}  grid={grid.name}")
    return outdir, d_dt, times


def clip(scene: str, frames: int, world_key: str = None):
    pre = scenes.SCENE_PRESETS[scene]
    prompt = pre["prompt"]
    ddir, d_dt = render_depth(scene, frames)
    out = ROOT / scene / f"{scene}_{world_key or 'default'}.mp4"
    v_dt = vace(ddir, prompt, out, frames)
    log(f"CLIP done -> {out}  (depth {d_dt:.0f}s + VACE {v_dt:.0f}s = {d_dt+v_dt:.0f}s)")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "depth"
    scene = sys.argv[2] if len(sys.argv) > 2 else "city_flythrough"
    frames = int(sys.argv[3]) if len(sys.argv) > 3 else 33
    if cmd == "depth":
        ddir, dt = render_depth(scene, frames)
        log(f"depth-only done -> {ddir} ({dt:.0f}s)")
    elif cmd == "clip":
        clip(scene, frames, sys.argv[4] if len(sys.argv) > 4 else "default")
    elif cmd == "set":
        build_set(scene, frames)
    elif cmd == "worlds":
        # render depth once + VACE every world in WORLDS[scene] (resumable, no grid)
        ddir, d_dt = render_depth(scene, frames)
        for wk, prompt in WORLDS[scene].items():
            out = ROOT / scene / f"{scene}_{wk}.mp4"
            if out.exists(): log(f"  {out.name} exists"); continue
            vace(ddir, prompt, out, frames)
        log(f"WORLDS {scene} done")
    elif cmd == "reelrender":
        # all heavy VACE for the v2 reel: city (5 worlds @ long) + monkey_orbit variety (2 worlds)
        for sc, fr in (("city_flythrough", frames), ("monkey_orbit", 49)):
            ddir, _ = render_depth(sc, fr)
            for wk, prompt in WORLDS[sc].items():
                out = ROOT / sc / f"{sc}_{wk}.mp4"
                if out.exists(): log(f"  {out.name} exists"); continue
                vace(ddir, prompt, out, fr)
            log(f"WORLDS {sc} done")
        log("REELRENDER done")
