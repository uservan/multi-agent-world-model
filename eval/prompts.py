"""System prompts for single-agent and multi-agent eval.

Agents are given only the goal + a shuffled list of available platforms (name,
description, base_url). No scene order, no task_operations — the agent plans
everything itself and discovers each API via GET /openapi.json.
"""
from __future__ import annotations


def _platform_section(platform_map: dict[str, dict]) -> str:
    """platform_map: {platform: {url, description}} (already shuffled by caller)."""
    lines = []
    for p, info in platform_map.items():
        desc = (info.get("description") or "").strip()
        lines.append(f"- {p} ({info['url']})\n    {desc}")
    return "Available platforms (in no particular order):\n" + "\n".join(lines)


_HTTP_TOOL = """\
Call any platform's REST API with the http tool:
<tool_call>
{"name": "http", "arguments": {"method": "GET|POST|PUT|PATCH|DELETE", "base_url": "<platform_url>", "path": "/<endpoint>", "params": {...}}}
</tool_call>

- For GET/DELETE, params are query-string parameters; for POST/PUT/PATCH, params are the JSON body.
- The X-Task-ID header is injected automatically — do not include it.
- First call for each platform you use: GET /openapi.json to discover its exact endpoints and parameters. Never guess paths."""


def build_single_agent_prompt(goal: str, platform_map: dict[str, dict]) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the single-agent mode."""
    system = (
        "You are an AI agent that completes a user's task by operating one or more platform REST APIs.\n\n"
        + _platform_section(platform_map) + "\n\n"
        + _HTTP_TOOL + "\n\n"
        + "Plan the work yourself: decide which platforms to use and in what order. "
        + "When the entire task is complete, output a <done> block with a brief summary:\n"
        + "<done>\nwhat you accomplished and key values created\n</done>"
    )
    user = f"Task: {goal}"
    return system, user


def build_orchestrator_prompt(
    goal: str, platform_map: dict[str, dict], max_concurrent: int, max_queue: int
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the multi-agent orchestrator."""
    spawn = f"""\
You may delegate subtasks to sub-agents (up to {max_concurrent} running at once, queue limit {max_queue}):

Spawn a sub-agent (non-blocking, returns a task_id):
<tool_call>
{{"name": "spawn_subagent", "arguments": {{"description": "<full instructions, INCLUDING the exact platform URL(s) the sub-agent must use>", "return_requirements": "<what the sub-agent should report back>"}}}}
</tool_call>

Collect results:
<tool_call>
{{"name": "get_task_results", "arguments": {{"task_ids": ["<task_id>", ...], "blocking": true, "timeout": 30}}}}
</tool_call>
- blocking=false: finished tasks return their result, unfinished return "pending".
- blocking=true: waits until all listed tasks finish or timeout."""

    system = (
        "You are an orchestrator AI agent that completes a user's task, optionally by delegating to sub-agents.\n\n"
        + _platform_section(platform_map) + "\n\n"
        + _HTTP_TOOL + "\n\n"
        + spawn + "\n\n"
        + "Sub-agents cannot see this platform list — you must put the platform URLs they need into their description. "
        + "Plan the decomposition yourself. When the entire task is complete, output <done>:\n"
        + "<done>\nsummary\n</done>"
    )
    user = f"Task: {goal}"
    return system, user


def build_subagent_prompt(description: str, return_requirements: str) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for a spawned sub-agent."""
    system = (
        "You are a sub-agent completing one assigned subtask via platform REST APIs.\n\n"
        + _HTTP_TOOL + "\n\n"
        + "When done, wrap your final answer in <result> tags and output <done>:\n"
        + "<result>\nyour answer (include any IDs/values requested)\n</result>\n<done>"
    )
    user = f"Subtask: {description}\n\nReport back: {return_requirements}"
    return system, user
