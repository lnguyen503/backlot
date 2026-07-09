"""[SHOWCASE v2] Build the maker-crowd reel from the rendered assets (run showcase.py reelrender first).
Segments: (1) pipeline reveal (grey Blender depth -> styled), (2) seamless world-morph on one continuous
camera move, (3) camera-range variety (orbit). + minimal titles + ACE-Step music. Wide 16:9 + vertical 9:16.
  python reel.py            # assemble (assets must exist)
  python reel.py music      # (re)generate the music bed only
"""
import subprocess, sys, time
from pathlib import Path
import cv2, numpy as np, imageio.v2 as imageio, imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
import httpx

FF = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = Path(__file__).resolve().parents[1] / "runs/showcase"
CITY = ROOT / "city_flythrough"
MONK = ROOT / "monkey_orbit"
OUT = ROOT / "reel"
OUT.mkdir(parents=True, exist_ok=True)
COMFY = "http://127.0.0.1:8188"
W, H = 832, 480
FONT_B = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 40)
FONT_S = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 28)
MORPH = ["neon", "temple", "jungle", "underwater", "lava"]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _read(path):
    c = cv2.VideoCapture(str(path)); fr = []
    while True:
        ok, f = c.read()
        if not ok: break
        fr.append(f)
    c.release(); return fr


def _writer(path, fps):
    return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1)


def _title(text, sub=None, size=(W, H)):
    """Transparent title overlay PNG (centered)."""
    img = Image.new("RGBA", size, (0, 0, 0, 0)); d = ImageDraw.Draw(img)
    b = d.textbbox((0, 0), text, font=FONT_B); tw = b[2] - b[0]
    x, y = (size[0] - tw) / 2, size[1] / 2 - 40
    d.text((x, y), text, font=FONT_B, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
    if sub:
        b2 = d.textbbox((0, 0), sub, font=FONT_S); sw = b2[2] - b2[0]
        d.text(((size[0] - sw) / 2, y + 52), sub, font=FONT_S, fill=(210, 210, 210, 255),
               stroke_width=2, stroke_fill=(0, 0, 0, 255))
    return img


def seg_reveal(dst: Path):
    """Grey Blender depth flies, then cross-dissolves into the neon world. Labeled."""
    depth = sorted((CITY / "depth").glob("depth_*.png"))
    neon = _read(CITY / "city_flythrough_neon.mp4")
    n = min(len(depth), len(neon))
    lab_d = _title("BLENDER DEPTH PASS", "raw 3D geometry the AI never sees")
    lab_s = _title("ComfyUI · Wan VACE restyle", "same geometry, real world")
    w = _writer(dst, 16)
    XF = 10  # crossfade frames
    span = min(n, 46)  # ~2.9s
    for t in range(span):
        dframe = cv2.imread(str(depth[t])); dframe = cv2.resize(dframe, (W, H))
        if dframe.ndim == 2 or dframe.shape[2] == 1:
            dframe = cv2.cvtColor(dframe, cv2.COLOR_GRAY2BGR)
        nframe = cv2.resize(neon[t], (W, H))
        if t < span - XF:
            fr, lab = dframe, lab_d
        else:
            a = (t - (span - XF)) / XF
            fr = cv2.addWeighted(dframe, 1 - a, nframe, a, 0)
            lab = lab_s if a > 0.5 else lab_d
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        comp = Image.alpha_composite(Image.fromarray(rgb).convert("RGBA"), lab).convert("RGB")
        w.append_data(np.array(comp))
    w.close(); log(f"  reveal -> {dst.name} ({span}f)")


def seg_morph(dst: Path, fps=12):
    """Continuous camera move; world morphs neon->temple->jungle->underwater->lava (no cuts)."""
    clips = [_read(CITY / f"city_flythrough_{k}.mp4") for k in MORPH]
    L = min(len(c) for c in clips); N = len(clips)
    labels = {k: _title("", None) for k in MORPH}
    # lower-third world label
    def lbl(name):
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(img)
        d.text((28, H - 56), name.upper(), font=FONT_B, fill=(255, 255, 255, 235),
               stroke_width=2, stroke_fill=(0, 0, 0, 255)); return img
    lbls = [lbl(k) for k in MORPH]
    w = _writer(dst, fps)
    for t in range(L):
        pos = t / (L - 1) * (N - 1); k = int(pos); a = pos - k
        fa = clips[k][t]; fb = clips[min(k + 1, N - 1)][t]
        fr = cv2.addWeighted(fa, 1 - a, fb, a, 0)
        rgb = cv2.cvtColor(cv2.resize(fr, (W, H)), cv2.COLOR_BGR2RGB)
        lab = lbls[k] if a < 0.5 else lbls[min(k + 1, N - 1)]
        comp = Image.alpha_composite(Image.fromarray(rgb).convert("RGBA"), lab).convert("RGB")
        w.append_data(np.array(comp))
    w.close(); log(f"  morph -> {dst.name} ({L}f @ {fps}fps = {L/fps:.1f}s)")


def seg_variety(dst: Path, fps=14):
    """Different camera PATH (orbit) — marble -> gold morph, proving it's not just a corridor."""
    a = _read(MONK / "monkey_orbit_marble.mp4"); b = _read(MONK / "monkey_orbit_gold.mp4")
    L = min(len(a), len(b))
    lab = _title("DIFFERENT CAMERA PATH", "Blender orbit · same control")
    w = _writer(dst, fps)
    for t in range(L):
        al = min(1.0, max(0.0, (t - L * 0.35) / (L * 0.3)))  # hold marble, morph to gold mid-clip
        fr = cv2.addWeighted(a[t], 1 - al, b[t], al, 0)
        rgb = cv2.cvtColor(cv2.resize(fr, (W, H)), cv2.COLOR_BGR2RGB)
        show = lab if t < L * 0.5 else _title("", None)
        comp = Image.alpha_composite(Image.fromarray(rgb).convert("RGBA"), show).convert("RGB")
        w.append_data(np.array(comp))
    w.close(); log(f"  variety -> {dst.name} ({L}f)")


def gen_music(dst: Path, seconds=26):
    T = Path(__file__).resolve().parents[1] / "templates/music_acestep.api.json"
    import json
    g = json.loads(T.read_text(encoding="utf-8"))
    g["18"]["inputs"]["tags"] = "cinematic electronic, driving synth arpeggio, awe, epic build, instrumental, no vocals"
    g["18"]["inputs"]["lyrics"] = ""
    g["17"]["inputs"]["seconds"] = float(seconds)
    t0 = time.time()
    pid = httpx.post(f"{COMFY}/prompt", json={"prompt": g}, timeout=120).json().get("prompt_id")
    if not pid:
        log("  music: prompt rejected; skipping bed"); return None
    while time.time() - t0 < 600:
        h = httpx.get(f"{COMFY}/history/{pid}", timeout=30).json().get(pid)
        if h and h.get("status", {}).get("status_str") == "success":
            for _, o in h["outputs"].items():
                for it in o.get("audio", []) + o.get("gifs", []):
                    if str(it["filename"]).endswith((".flac", ".wav", ".mp3")):
                        dst.write_bytes(httpx.get(f"{COMFY}/view?filename={it['filename']}&subfolder={it.get('subfolder','')}&type=output", timeout=300).content)
                        log(f"  music -> {dst.name}"); return dst
            return None
        if h and h.get("status", {}).get("status_str") == "error":
            log("  music: comfy error; skipping"); return None
        time.sleep(5)
    return None


def concat(parts, dst, xf=0.4):
    """Concat segments with short crossfades (xfade chain)."""
    # normalize each to W:H, 16fps for a clean xfade
    norm = []
    for i, p in enumerate(parts):
        n = OUT / f"_n{i}.mp4"
        subprocess.run([FF, "-y", "-i", str(p), "-vf", f"scale={W}:{H},fps=24,format=yuv420p",
                        "-c:v", "libx264", str(n)], check=True, capture_output=True)
        norm.append(n)
    cur = norm[0]
    for i in range(1, len(norm)):
        # duration of cur
        c = cv2.VideoCapture(str(cur)); dur = c.get(7) / (c.get(5) or 24); c.release()
        o = OUT / f"_x{i}.mp4"
        subprocess.run([FF, "-y", "-i", str(cur), "-i", str(norm[i]), "-filter_complex",
                        f"[0][1]xfade=transition=fade:duration={xf}:offset={dur-xf:.2f},format=yuv420p[v]",
                        "-map", "[v]", "-c:v", "libx264", str(o)], check=True, capture_output=True)
        cur = o
    subprocess.run([FF, "-y", "-i", str(cur), "-c", "copy", str(dst)], check=True, capture_output=True)


def finish(silent: Path, music: Path, wide: Path, vert: Path):
    c = cv2.VideoCapture(str(silent)); dur = c.get(7) / (c.get(5) or 24); c.release()
    # WIDE 16:9 1080p + music
    if music and music.exists():
        subprocess.run([FF, "-y", "-i", str(silent), "-i", str(music), "-filter_complex",
                        f"[0:v]scale=1920:1080:flags=lanczos[v];[1:a]afade=t=out:st={dur-1.5:.1f}:d=1.5,volume=0.8[a]",
                        "-map", "[v]", "-map", "[a]", "-t", f"{dur:.2f}", "-c:v", "libx264", "-pix_fmt",
                        "yuv420p", "-crf", "19", "-c:a", "aac", str(wide)], check=True, capture_output=True)
    else:
        subprocess.run([FF, "-y", "-i", str(silent), "-vf", "scale=1920:1080:flags=lanczos", "-c:v",
                        "libx264", "-pix_fmt", "yuv420p", "-crf", "19", str(wide)], check=True, capture_output=True)
    # VERTICAL 9:16 1080x1920: center content + blurred fill top/bottom
    vf = ("[0:v]scale=1080:-2:flags=lanczos[fg];"
          "[0:v]scale=1080:1920:flags=lanczos,boxblur=30:10[bg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v]")
    ain = (["-i", str(music)] if music and music.exists() else [])
    amap = (["-map", "1:a", "-c:a", "aac", "-shortest"] if music and music.exists() else [])
    subprocess.run([FF, "-y", "-i", str(silent), *ain, "-filter_complex", vf, "-map", "[v]", *amap,
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", str(vert)],
                   check=True, capture_output=True)
    log(f"  FINISH wide={wide.name} vert={vert.name} ({dur:.1f}s)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "music":
        gen_music(OUT / "bed.flac"); return
    r, m, v = OUT / "s1_reveal.mp4", OUT / "s2_morph.mp4", OUT / "s3_variety.mp4"
    seg_reveal(r); seg_morph(m); seg_variety(v)
    silent = OUT / "_reel_silent.mp4"
    concat([r, m, v], silent)
    music = OUT / "bed.flac"
    if not music.exists(): gen_music(music)
    finish(silent, music if music.exists() else None, OUT / "showcase_reel_wide.mp4", OUT / "showcase_reel_vertical.mp4")
    log("REEL done")


if __name__ == "__main__":
    main()
