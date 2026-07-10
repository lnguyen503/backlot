"""Smoke-test the MCP server as a real MCP stdio server: handshake + tools."""
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "backlot.mcp_server"],
)


async def main():
    async with stdio_client(PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print("TOOLS:", names)
            res = await session.call_tool("list_workflows", {})
            text = res.content[0].text if res.content else ""
            print("list_workflows ->", text[:300])
            ok = {"list_workflows", "describe_workflow", "run_workflow",
                  "get_job_status", "cancel_job"}.issubset(set(names))
            print("RESULT:", "PASS" if ok and "txt2img_sdxl" in text else "FAIL")


if __name__ == "__main__":
    asyncio.run(main())
