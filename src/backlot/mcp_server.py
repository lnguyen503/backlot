"""MCP server exposing the ComfyUI engine to Claude Code (§5).

Phase 1 tools: list_workflows, describe_workflow, run_workflow, get_job_status,
cancel_job. The web backend (Phase 3) will reuse the same Engine core.
"""
from __future__ import annotations

from typing import Optional

from mcp.server.fastmcp import FastMCP

from .engine.runtime import Engine

mcp = FastMCP("backlot")
_engine: Optional[Engine] = None


def _eng() -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine()
    return _engine


@mcp.tool()
async def list_workflows(kind: Optional[str] = None) -> dict:
    """List registered capabilities, optionally filtered by kind."""
    return {"workflows": [c.public_info() for c in _eng().registry.list(kind)]}


@mcp.tool()
async def describe_workflow(name: str) -> dict:
    """Full injectable-param schema for one capability."""
    cap = _eng().registry.get(name)
    return {
        "name": cap.name, "title": cap.title, "kind": cap.kind,
        "inputs": cap.inputs, "client_outputs": cap.client_outputs,
        "inject": [s.model_dump() for s in cap.inject],
    }


@mcp.tool()
async def run_workflow(name: str, params: Optional[dict] = None, wait: bool = False,
                       timeout_s: Optional[float] = None) -> dict:
    """Run one capability with explicit params. Returns run_id/prompt_id/state."""
    eng = _eng()
    eng.ensure_started()
    return await eng.jobs.run_workflow(name, params or {}, wait=wait, timeout_s=timeout_s)


@mcp.tool()
async def list_blender_scenes() -> dict:
    """List the 3D scenes available for render_3d_animation (key/label/motion/prompt)."""
    from .blender import scenes as bscenes
    return {"scenes": bscenes.scene_list()}


@mcp.tool()
async def render_3d_animation(scene: str = "city_flythrough", prompt: str = "",
                              frames: int = 33, steps: int = 25, seed: int = 7,
                              model: str = "wan2.1_vace_1.3B_fp16.safetensors",
                              wait: bool = True) -> dict:
    """AI-stylized 3D animation: Blender renders a depth sequence for `scene`, then
    Wan VACE restyles it into one coherent clip (blank prompt = the scene default).
    Scenes come from list_blender_scenes. wait=True blocks and returns the video."""
    import asyncio
    from pathlib import Path
    from .blender import runner as bl, scenes as bscenes
    eng = _eng()
    eng.ensure_started()
    preset = bscenes.SCENE_PRESETS.get(scene)
    if preset is None:
        return {"error": f"unknown scene: {scene}", "scenes": list(bscenes.SCENE_PRESETS)}
    depth_dir = Path(eng.cfg.paths.runs) / "blender_mcp" / f"{scene}_{seed}"
    depth_dir.mkdir(parents=True, exist_ok=True)
    r = await asyncio.to_thread(
        bl.run_script, bscenes.depth_sequence(scene),
        args=[str(depth_dir), frames, preset["near"], preset["far"]], timeout=600)
    if not r.ok:
        return {"error": "blender failed", "stderr": r.stderr[-400:]}
    n = len(list(depth_dir.glob("depth_*.png")))
    return await eng.jobs.run_workflow(
        "vace_depth_video",
        {"positive_prompt": prompt.strip() or preset["prompt"],
         "control_dir": str(depth_dir).replace("\\", "/"), "length": n, "strength": 1.0,
         "steps": steps, "seed": seed, "fps": 16.0, "model": model},
        wait=wait, timeout_s=max(1800, eng.cfg.timeouts.video_job_s))


@mcp.tool()
async def direct_video(premise: str, guidance: str = "", i2v: str = "wan14b_fast",
                       music: bool = True, narrate: bool = False, max_shots: int = 4) -> dict:
    """The Director: turn a one-line premise (+ optional guidance like style/aspect/
    length) into a finished multi-shot video, autonomously — plan → consistent stills →
    animate → assemble → LLM-scored music (+ optional narration). Reuses the Studio
    tools; the board is saved so a human can keep editing it. Heavy (minutes). Returns
    the board id/title and the finished video filename+url."""
    from .storyboard import director
    eng = _eng()
    eng.ensure_started()
    logs: list[str] = []
    sb = await director.direct(eng, premise, guidance=guidance, i2v=i2v, music=music,
                               narrate=narrate, max_shots=max_shots, log=logs.append)
    v = sb.assembled
    return {"board_id": sb.id, "title": sb.title, "logline": sb.logline,
            "shots": len(sb.panels), "video": v.filename if v else None,
            "url": v.url if v else None, "log": logs}


@mcp.tool()
async def get_job_status(run_id: Optional[str] = None,
                         prompt_id: Optional[str] = None) -> dict:
    """Progress + outputs for a run_id or prompt_id."""
    return _eng().jobs.get_status(run_id=run_id, prompt_id=prompt_id)


@mcp.tool()
async def cancel_job(run_id: Optional[str] = None,
                     prompt_id: Optional[str] = None) -> dict:
    """Interrupt a queued/running job."""
    return await _eng().jobs.cancel(run_id=run_id, prompt_id=prompt_id)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
