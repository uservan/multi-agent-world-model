"""Tool executors for eval agents.

Tool calls are parsed by utils.agent_io.parse_tool_calls into
{"name": <tool>, "arguments": {...}}. The http executor is shared by every actor
(single agent, orchestrator, sub-agents); spawn_subagent / get_queue_status /
get_task_info / wait_task are orchestrator-only and live in multi_agent.py.
"""
from __future__ import annotations

import json

from eval.platform import PlatformRuntime


def make_http_executor(runtime: PlatformRuntime):
    """Returns an async executor that handles the 'http' tool against the live servers."""
    async def execute(call: dict) -> str:
        if call.get("name") != "http":
            return f"Error: unknown tool '{call.get('name')}'. Available: http."
        args = call.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return "Error: http arguments must be a JSON object"
        method = str(args.get("method", "GET")).upper()
        base_url = args.get("base_url", "")
        path = args.get("path", "/")
        params = args.get("params") or {}
        if not base_url:
            return "Error: http call missing base_url"
        # GET/DELETE → query params; otherwise → JSON body
        if method in ("GET", "DELETE"):
            resp = runtime.call(base_url, method, path, params=params)
        else:
            resp = runtime.call(base_url, method, path, body=params)
        return f"[{method} {path}] {resp}"

    return execute
