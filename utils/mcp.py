"""MCP injection + tool executor + MCP-based agent loop.

Used by the verifier pipeline to run an LLM agent against a platform's MCP server.
(The eval framework uses plain HTTP via utils.server instead.)
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os

from utils.agent_io import format_tools_text, parse_tool_calls
from utils.llm import LLMClient
from utils.server import start_server, stop_server, wait_for_server


# silence noisy async libs
for _noisy in ["mcp_agent", "mcp", "httpx", "httpcore", "anyio"]:
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)
    logging.getLogger(_noisy).propagate = False


# ── MCP injection ──────────────────────────────────────────────────────────────

def inject_mcp(code: str, platform_name: str) -> str:
    """Inject fastapi-mcp wrapper into generated FastAPI server code."""
    mcp_block = (
        f"    from fastapi_mcp import FastApiMCP as _FastApiMCP\n"
        f"    _mcp_instance = _FastApiMCP(app, name={repr(platform_name)}, headers=['X-Task-ID'])\n"
        f"    _mcp_instance.mount_http()\n"
    )
    lines = code.split("\n")
    new_lines: list[str] = []
    for line in lines:
        if "uvicorn.run(app" in line:
            new_lines.append(mcp_block)
        new_lines.append(line)
    return "\n".join(new_lines)


# ── MCP tool executor ──────────────────────────────────────────────────────────

_MCP_SERVER_KEY = "platform_server"


class MCPToolExecutor:
    """Thin wrapper around mcp_agent for one platform's MCP server."""

    def __init__(self, mcp_url: str, task_id: str, timeout: float = 60.0):
        self.mcp_url = mcp_url
        self.task_id = task_id
        self.timeout = timeout
        self._tools: list[dict] = []
        self._setup()

    def _setup(self) -> None:
        from mcp_agent.app import MCPApp
        from mcp_agent.agents.agent import Agent
        from mcp_agent.config import (
            Settings, MCPSettings, MCPServerSettings, LoggerSettings,
        )
        settings = Settings(
            execution_engine="asyncio",
            logger=LoggerSettings(
                type="none", transports=["none"],
                progress_display=False, level="error",
            ),
            mcp=MCPSettings(
                servers={
                    _MCP_SERVER_KEY: MCPServerSettings(
                        transport="streamable_http",
                        url=self.mcp_url,
                        headers={"X-Task-ID": self.task_id},
                    )
                }
            ),
        )
        self._app = MCPApp(name="awm_agent", settings=settings)
        self._agent = Agent(name="executor", server_names=[_MCP_SERVER_KEY])

    @staticmethod
    def _strip_task_id_header(schema: dict) -> dict:
        """Remove x_task_id / X-Task-ID from tool input schema — it's sent as a transport header."""
        if not schema:
            return schema
        props = dict(schema.get("properties") or {})
        required = list(schema.get("required") or [])
        for key in list(props):
            if key.lower().replace("-", "_") == "x_task_id":
                props.pop(key)
        required = [r for r in required if r.lower().replace("-", "_") != "x_task_id"]
        return {**schema, "properties": props, "required": required}

    async def list_tools(self) -> list[dict]:
        with contextlib.redirect_stderr(io.StringIO()):
            async with self._app.run():
                async with self._agent:
                    result = await asyncio.wait_for(
                        self._agent.list_tools(), timeout=self.timeout
                    )
                    self._tools = [
                        {
                            "name": t.name,
                            "description": t.description or "",
                            "inputSchema": self._strip_task_id_header(t.inputSchema or {}),
                        }
                        for t in result.tools
                    ]
                    return self._tools

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        # strip x_task_id if LLM still passes it — it's sent as transport header
        arguments = {
            k: v for k, v in arguments.items()
            if k.lower().replace("-", "_") != "x_task_id"
        }
        with contextlib.redirect_stderr(io.StringIO()):
            async with self._app.run():
                async with self._agent:
                    result = await asyncio.wait_for(
                        self._agent.call_tool(tool_name, arguments),
                        timeout=self.timeout,
                    )
                    parts = [
                        c.text if hasattr(c, "text") else str(c)
                        for c in result.content
                    ]
                    text = "\n".join(parts)
                    return f"Error: {text}" if result.isError else text


# ── MCP agent loop ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are operating a platform API via MCP tools to complete a specific sub-task.

You have two meta-tools:
  list_tools   — list all available MCP tools (call once at the start)
  call_tool    — call a specific MCP tool
                 arguments: {"tool_name": "<name>", "arguments": "<json-string>"}

Output each tool call as:
<tool_call>
{"name": "list_tools", "arguments": null}
</tool_call>

<tool_call>
{"name": "call_tool", "arguments": {"tool_name": "search_products", "arguments": "{\"query\": \"laptop\"}"}}
</tool_call>

Use the provided task operations as your step-by-step guide — follow their order and use the exact param values given. When all steps are done, write a final summary (no tool call tags) to signal completion."""


async def run_agent_messages_loop(
    mcp: MCPToolExecutor,
    llm_client: LLMClient,
    model: str,
    user_prompt: str,
    max_iterations: int,
) -> list[dict]:
    tools = await mcp.list_tools()
    tools_text = format_tools_text(tools)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    trajectory: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(
            None, llm_client.complete, model, messages, 1024 * 8
        )
        tool_calls = parse_tool_calls(content)

        messages.append({"role": "assistant", "content": content})

        if not tool_calls:
            trajectory.append({"iteration": iteration, "content": content, "is_final": True})
            break

        responses: list[str] = []
        for tc in tool_calls:
            name = tc["name"]
            arguments = tc["arguments"]

            if name == "list_tools":
                tool_response = tools_text

            elif name == "call_tool":
                if isinstance(arguments, dict):
                    tool_name = arguments.get("tool_name", "")
                    inner = arguments.get("arguments", {})
                else:
                    tool_name, inner = "", {}
                if isinstance(inner, str):
                    try:
                        inner = json.loads(inner) if inner.strip() else {}
                    except Exception:
                        inner = {}
                if tool_name.startswith("mcp_tool_"):
                    tool_name = tool_name[len("mcp_tool_"):]
                try:
                    tool_response = await mcp.call_tool(tool_name, inner)
                except asyncio.TimeoutError:
                    tool_response = "Error: tool call timed out"
                except Exception as e:
                    tool_response = f"Error: {e}"
            else:
                tool_response = f"Error: unknown tool '{name}'"

            responses.append(f"[{name}] {tool_response}")
            trajectory.append({
                "iteration": iteration,
                "content": content,
                "tool_call": {"name": name, "arguments": arguments},
                "tool_response": tool_response,
            })

        messages.append({"role": "user", "content": "Tool responses:\n" + "\n---\n".join(responses)})
    else:
        trajectory.append({"iteration": max_iterations, "content": "", "is_final": True, "timed_out": True})

    return trajectory


async def run_sub_agent_loop(
    server_path: str,
    db_path: str,
    platform: str,
    task_temp: str,
    task_id: str,
    scene_idx: int,
    sub_agent_idx: int,
    llm_client: LLMClient,
    model: str,
    user_prompt: str,
    max_iterations: int,
) -> tuple[list[dict], str]:
    """Start a fresh MCP server, run agent loop, stop server. Returns (trajectory, server_logs).
    Each call is isolated — logs only cover this sub-agent's server lifetime."""
    with open(server_path, "r", encoding="utf-8") as f:
        orig = f.read()
    mcp_code = inject_mcp(orig, platform)
    safe = platform.lower().replace(" ", "_").replace("/", "_")
    mcp_path = os.path.join(task_temp, f"{safe}_s{scene_idx}_k{sub_agent_idx}_mcp.py")
    with open(mcp_path, "w", encoding="utf-8") as f:
        f.write(mcp_code)

    proc, port = start_server(mcp_path, db_path)
    trajectory: list[dict] = []
    try:
        if not wait_for_server(port, timeout=25):
            return [], "Server failed to start within 25s"
        mcp_url = f"http://127.0.0.1:{port}/mcp"
        mcp = MCPToolExecutor(mcp_url, task_id)
        trajectory = await run_agent_messages_loop(
            mcp, llm_client, model, user_prompt, max_iterations,
        )
    finally:
        stop_server(proc)
        try:
            server_logs = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
        except Exception:
            server_logs = ""

    return trajectory, server_logs
