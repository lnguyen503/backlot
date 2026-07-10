"""Mini-movie runner: turn a shot list into a finished multi-shot film, fully local.

Per shot:  Flux still  ->  Wan 2.2 14B image->video  ->  30fps interpolated + graded.
Then:      one continuous ACE-Step score  +  per-shot synth SFX (mm_sfx)  ->  assemble
           (title card + hard cuts) into runs/minimovie_<name>.mp4.

Idempotent: a shot whose runs/minimovie/<id>.mp4 already exists is skipped, so a
re-run only regenerates what's missing (and re-runs assembly cheaply).

    .venv\\Scripts\\python.exe tests\\make_minimovie.py
"""
from __future__ import annotations
import asyncio
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import httpx
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from backlot.engine.runtime import Engine  # noqa: E402
import mm_sfx  # noqa: E402

NAME = "dinosaur"
TITLE = "PRIMAL"
OUT = Path(str(Path(__file__).resolve().parents[1] / "runs/minimovie"))
FINAL = Path(str(Path(__file__).resolve().parents[1] / f"runs/minimovie_{NAME}.mp4"))
W, H, FPS = 1280, 720, 30
LOOK = ("battle-scarred dark green tyrannosaurus rex, glowing amber eye, "
        "rainy misty jungle at dusk, cold teal night color grade, volumetric fog, "
        "35mm anamorphic, photorealistic, cinematic")
GRADE = "eq=contrast=1.12:brightness=-0.03:saturation=0.92,vignette=PI/5"

SHOTS = [
    dict(id="01_tremor", kind="tremor", seed=101,
         still=("Cinematic horror establishing shot, a dark rain-soaked jungle trail at dusk, "
                "a muddy puddle in the foreground, dense dripping foliage, ominous empty path, "
                "no animals, volumetric fog, cold teal grade, 35mm, photorealistic"),
         motion=("heavy rain falls and the puddle ripples pulse outward in rhythmic rings as if "
                 "from distant heavy footsteps, mist drifting, leaves trembling, slow ominous "
                 "push-in, tense locked camera")),
    dict(id="02_reveal", kind="reveal", seed=102,
         still=("Cinematic wide shot, an enormous " + LOOK + ", standing among the misty trees at "
                "the edge of a rainy jungle clearing, head beginning to turn toward camera, dramatic"),
         motion=("the tyrannosaurus rex turns its massive head toward the camera and takes one "
                 "heavy step forward out of the trees, jaws parting to begin a low roar, rain "
                 "pouring, mist drifting, slow menacing motion")),
    dict(id="03_attack", kind="attack", seed=103,
         still=("Cinematic horror extreme low-angle close shot of an enraged " + LOOK + ", jaws "
                "erupting wide open in a furious roar, blood-stained teeth, dripping saliva, eye "
                "locked on the viewer, rain and mud flying, lightning rim-light, dread, imminent attack"),
         motion=("the enraged tyrannosaurus rex charges forward and lunges its head toward the "
                 "camera, jaws snapping in a violent roar, saliva flying, rain pouring, explosive "
                 "aggressive movement, the roaring head fills the frame, violent camera shake")),
]


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("CMD FAILED:", " ".join(str(c) for c in cmd))
        print(p.stderr[-1500:])
        raise RuntimeError("ffmpeg/ffprobe failed")
    return p.stdout


def probe_dur(path):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                      "-of", "default=noprint_wrappers=1:nokey=1", str(path)]).strip())


def free_vram(base):
    try:
        httpx.post(base + "/free", json={"unload_models": True, "free_memory": True}, timeout=30)
    except Exception as e:
        print("free skip", e)


async def gen_still(eng, shot):
    dst = Path(eng.cfg.comfyui.input_dir) / f"mm_{shot['id']}.png"
    if dst.exists():
        return dst.name
    print(f"[{shot['id']}] Flux still ...", flush=True)
    res = await eng.jobs.run_workflow("txt2img_flux", {
        "positive_prompt": shot["still"], "guidance": 3.7, "steps": 30,
        "width": W, "height": H, "seed": shot["seed"]}, wait=True, timeout_s=360)
    assert res["state"] == "completed" and res["outputs"], res.get("error")
    o = res["outputs"][0]
    src = Path(eng.cfg.comfyui.output_dir) / o.get("subfolder", "") / o["filename"]
    dst.write_bytes(src.read_bytes())
    return dst.name


async def gen_video(eng, shot, start_name):
    webp = OUT / f"{shot['id']}.webp"
    if webp.exists():
        return webp
    print(f"[{shot['id']}] Wan i2v ...", flush=True)
    free_vram(eng.cfg.comfyui.base_url)
    res = await eng.jobs.run_workflow("img2vid_wan14b", {
        "image": start_name, "positive_prompt": shot["motion"],
        "negative_prompt": ("low quality, worst quality, blurry, distorted, deformed, extra limbs, "
                            "extra heads, warping, morphing, static, calm, smiling, cartoon"),
        "width": W, "height": H, "length": 81, "seed": shot["seed"]}, wait=True, timeout_s=1500)
    assert res["state"] == "completed" and res["outputs"], res.get("error")
    o = res["outputs"][0]
    src = Path(eng.cfg.comfyui.output_dir) / o.get("subfolder", "") / o["filename"]
    webp.write_bytes(src.read_bytes())
    return webp


def encode_shot(shot, webp):
    mp4 = OUT / f"{shot['id']}.mp4"
    if mp4.exists():
        return mp4
    fdir = OUT / f"frames_{shot['id']}"; fdir.mkdir(exist_ok=True)
    im = Image.open(webp); n = getattr(im, "n_frames", 1)
    for i in range(n):
        im.seek(i); im.convert("RGB").save(fdir / f"f_{i:03d}.png")
    run(["ffmpeg", "-y", "-framerate", "16", "-i", str(fdir / "f_%03d.png"),
         "-vf", f"minterpolate=fps={FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,{GRADE}",
         "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(FPS), str(mp4)])
    return mp4


def make_title(dur=1.8):
    png = OUT / "title.png"; mp4 = OUT / "title.mp4"; wav = OUT / "title.wav"
    img = Image.new("RGB", (W, H), (4, 6, 8)); d = ImageDraw.Draw(img)
    try:
        f1 = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 96)
        f2 = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 30)
    except Exception:
        f1 = f2 = ImageFont.load_default()
    tb = d.textbbox((0, 0), TITLE, font=f1)
    d.text(((W - (tb[2] - tb[0])) / 2, H / 2 - 70), TITLE, font=f1, fill=(200, 210, 214))
    sub = "a local mini-movie"
    sb = d.textbbox((0, 0), sub, font=f2)
    d.text(((W - (sb[2] - sb[0])) / 2, H / 2 + 40), sub, font=f2, fill=(120, 130, 134))
    img.save(png)
    run(["ffmpeg", "-y", "-loop", "1", "-t", str(dur), "-i", str(png),
         "-vf", f"fps={FPS},format=yuv420p,fade=t=in:st=0:d=0.4,fade=t=out:st={dur-0.5:.2f}:d=0.5",
         "-c:v", "libx264", "-crf", "17", "-r", str(FPS), str(mp4)])
    # low tension drone under the title
    N = int(dur * mm_sfx.SR); t = np.arange(N) / mm_sfx.SR
    swell = mm_sfx.ss(t / dur)
    drone = (np.sin(2 * np.pi * 40 * t) + 0.7 * np.sin(2 * np.pi * 55 * t)) * swell * 0.5
    drone += mm_sfx.lp(np.random.default_rng(9).standard_normal(N), 120) * swell * 0.3
    st = np.stack([drone, drone], 1); st /= (np.abs(st).max() + 1e-9); st *= 0.6
    st[-int(0.1 * mm_sfx.SR):] *= np.linspace(1, 0, int(0.1 * mm_sfx.SR))[:, None]
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(mm_sfx.SR)
        w.writeframes((st * 32767).astype(np.int16).tobytes())
    return mp4, wav, probe_dur(mp4)


def concat_wavs(paths, out):
    data = []
    for p in paths:
        with wave.open(str(p), "rb") as w:
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, 2)
        data.append(a)
    alld = np.concatenate(data, 0)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(mm_sfx.SR)
        w.writeframes(alld.tobytes())
    return out


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eng = Engine(); eng.ensure_started(); await asyncio.sleep(0.3)

    shot_mp4s, shot_durs = [], []
    for shot in SHOTS:
        start = await gen_still(eng, shot)
        webp = await gen_video(eng, shot, start)
        mp4 = encode_shot(shot, webp)
        dur = probe_dur(mp4)
        shot_mp4s.append(mp4); shot_durs.append(dur)
        print(f"[{shot['id']}] video {dur:.2f}s -> {mp4.name}", flush=True)

    # per-shot SFX
    sfx_wavs = []
    for shot, dur in zip(SHOTS, shot_durs):
        w = OUT / f"{shot['id']}.wav"
        mm_sfx.synth_shot(shot["kind"], dur, str(w))
        sfx_wavs.append(w)
        print(f"[{shot['id']}] sfx {shot['kind']} {dur:.2f}s", flush=True)

    # title
    title_mp4, title_wav, title_dur = make_title()
    total = title_dur + sum(shot_durs)
    print(f"total {total:.2f}s", flush=True)

    # continuous score
    print("score (ACE-Step) ...", flush=True)
    res = await eng.jobs.run_workflow("music_acestep", {
        "tags": "dark ominous cinematic orchestral tension, deep low strings, ambient dread, "
                "building suspense, distant timpani, no vocals, film score",
        "lyrics": "", "seconds": float(int(total) + 2), "steps": 50, "seed": 31},
        wait=True, timeout_s=600)
    assert res["state"] == "completed" and res["outputs"], res.get("error")
    o = res["outputs"][0]
    score = OUT / "score.mp3"
    score.write_bytes((Path(eng.cfg.comfyui.output_dir) / o.get("subfolder", "") / o["filename"]).read_bytes())

    # ---- assemble ----
    sfx_timeline = concat_wavs([title_wav] + sfx_wavs, OUT / "sfx_timeline.wav")
    final_audio = OUT / "final_audio.wav"
    run(["ffmpeg", "-y", "-i", str(sfx_timeline), "-i", str(score), "-filter_complex",
         f"[1:a]atrim=0:{total:.3f},asetpts=PTS-STARTPTS,volume=0.24,afade=t=in:st=0:d=0.6,"
         f"afade=t=out:st={total-0.6:.2f}:d=0.6[sc];[0:a]volume=1.0[sf];"
         f"[sf][sc]amix=inputs=2:duration=first:normalize=0[a]",
         "-map", "[a]", "-ar", "48000", str(final_audio)])

    listf = OUT / "concat.txt"
    listf.write_text("".join(f"file '{p.as_posix()}'\n" for p in [title_mp4] + shot_mp4s))
    full_video = OUT / "full_video.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listf), "-c", "copy", str(full_video)])

    run(["ffmpeg", "-y", "-i", str(full_video), "-i", str(final_audio),
         "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "224k",
         "-shortest", str(FINAL)])
    print("DONE ->", FINAL, flush=True)
    await eng.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
