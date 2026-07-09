"""MCP tool-layer tests. Exercise list/describe + Engine wiring without ComfyUI."""
import pytest

from backlot import mcp_server as M


def test_engine_constructs_and_registers():
    eng = M.Engine()
    assert "txt2img_sdxl" in eng.registry.names()
    # ws is wired into the job manager (no circular-init regressions)
    assert eng.jobs._ws is eng.ws


@pytest.mark.asyncio
async def test_list_workflows_tool():
    res = await M.list_workflows()
    names = {w["name"] for w in res["workflows"]}
    assert "txt2img_sdxl" in names


@pytest.mark.asyncio
async def test_describe_workflow_tool():
    desc = await M.describe_workflow(name="txt2img_sdxl")
    assert desc["kind"] == "image"
    assert any(s["name"] == "positive_prompt" for s in desc["inject"])
