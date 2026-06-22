"""Agent I/O helpers: parse <tool_call> blocks from LLM output, format tool lists.

Shared by the verifier pipeline and the eval framework.
"""
from __future__ import annotations

import json
import re


def parse_tool_calls(content: str) -> list[dict]:
    """Extract tool calls from <tool_call>...</tool_call> blocks in LLM output."""
    calls = []
    for i, m in enumerate(
        re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    ):
        try:
            data = json.loads(m.strip())
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                continue
            name = data.get("name", "")
            args = data.get("arguments", {})
            if name.startswith("mcp_tool_"):
                args = {"tool_name": name, "arguments": args or {}}
                name = "call_tool"
            calls.append({"id": f"call_{i}", "name": name, "arguments": args})
        except Exception:
            continue
    return calls


def format_tools_text(tools: list[dict]) -> str:
    actual = [t for t in tools if t.get("name") != "list_tools"]
    lines = [f"Available tools ({len(actual)}):"]
    for t in actual:
        lines.append(f"  - {t['name']}: {(t.get('description') or '')[:120]}")
    return "\n".join(lines)
