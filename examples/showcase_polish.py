"""[SHOWCASE] polish: labeled wide grid + vertical/mobile stacked cut.
Labels via PIL (avoids ffmpeg drawtext Windows font-path escaping)."""
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
D = Path(__file__).resolve().parents[1] / "runs/showcase/city_flythrough"
CW, CH = 832, 480
LABELS = ["NEON", "TEMPLE", "JUNGLE"]
CLIPS = [D / f"city_flythrough_{w.lower()}.mp4" for w in LABELS]
FONT = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 46)


def label_png(size, positions, out: Path):
    """positions: list of (cx, cy) centers for each label."""
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for text, (cx, cy) in zip(LABELS, positions):
        b = d.textbbox((0, 0), text, font=FONT); tw, th = b[2] - b[0], b[3] - b[1]
        x, y = cx - tw / 2, cy - th / 2
        # translucent pill + white text w/ black outline
        d.rectangle([x - 18, y - 10, x + tw + 18, y + th + 14], fill=(0, 0, 0, 140))
        d.text((x, y), text, font=FONT, fill=(255, 255, 255, 255),
               stroke_width=2, stroke_fill=(0, 0, 0, 255))
    img.save(out)


def main():
    # 1) LABELED WIDE GRID (2496x480): labels at bottom-center of each third
    gl = D / "_grid_labels.png"
    label_png((CW * 3, CH), [(CW * (i + 0.5), CH - 46) for i in range(3)], gl)
    grid = D / "city_flythrough_grid.mp4"
    subprocess.run([FF, "-y", "-i", str(grid), "-i", str(gl), "-filter_complex", "[0][1]overlay=0:0",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    str(D / "city_flythrough_grid_labeled.mp4")], check=True, capture_output=True)
    print("labeled grid ok", flush=True)

    # 2) VERTICAL / MOBILE stacked 3-up (832x1440): label at top of each panel
    vl = D / "_vert_labels.png"
    label_png((CW, CH * 3), [(CW * 0.5, CH * i + 40) for i in range(3)], vl)
    fc = "[0:v][1:v][2:v]vstack=inputs=3[s];[s][3]overlay=0:0[v]"
    subprocess.run([FF, "-y", "-i", str(CLIPS[0]), "-i", str(CLIPS[1]), "-i", str(CLIPS[2]),
                    "-i", str(vl), "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "18",
                    str(D / "city_flythrough_vertical.mp4")], check=True, capture_output=True)
    print("vertical stacked ok", flush=True)


if __name__ == "__main__":
    main()
