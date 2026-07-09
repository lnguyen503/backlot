"""One-off: render a single img2vid_wan14b shot to validate the 14B workflow."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backlot.engine.runtime import Engine  # noqa: E402


async def main():
    eng = Engine()
    eng.ensure_started()
    await asyncio.sleep(0.3)
    res = await eng.jobs.run_workflow(
        "img2vid_wan14b",
        {
            "image": sys.argv[1] if len(sys.argv) > 1 else "cv_start_1.png",
            "positive_prompt": "the man stands just inside his front doorway and takes a calm step "
                               "forward, breathing naturally, subtle movement, static locked camera, "
                               "his full body and head stay in frame",
            "width": int(sys.argv[2]) if len(sys.argv) > 2 else 480,
            "height": int(sys.argv[3]) if len(sys.argv) > 3 else 832,
            "length": int(sys.argv[4]) if len(sys.argv) > 4 else 41,
            "seed": 555,
        },
        wait=True, timeout_s=900,
    )
    print("STATE:", res["state"], flush=True)
    print("OUTPUTS:", res.get("outputs"), flush=True)
    if res.get("error"):
        print("ERROR:", res["error"], flush=True)
    await eng.client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
