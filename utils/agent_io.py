"""Agent I/O helpers: parse <tool_call> blocks from LLM output, format tool lists.

Shared by the verifier pipeline and the eval framework.
"""
from __future__ import annotations

import json
import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function\s*=\s*([^>\s]+)\s*>(.*?)</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter\s*=\s*([^>\s]+)\s*>\s*(.*?)\s*</parameter>", re.DOTALL)
# Kimi-K2.6 native tool-call tokens — they leak into content as text when the request
# doesn't use the OpenAI tools API (this eval uses a text protocol), e.g.
#   <|tool_call_begin|>functions.http:0<|tool_call_argument_begin|>{"method":"GET",...}<|tool_call_end|>
_KIMI_TOOL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*(.*?)\s*<\|tool_call_argument_begin\|>\s*(.*?)\s*<\|tool_call_end\|>",
    re.DOTALL,
)


def _kimi_tool_name(raw_id: str) -> str:
    """'functions.http:0' -> 'http'."""
    nm = raw_id.strip().split(":")[0].strip()
    return nm[len("functions."):] if nm.startswith("functions.") else nm


def _coerce(value: str):
    """Parameter values are raw text; JSON-decode when possible (dict/list/number/bool)."""
    try:
        return json.loads(value)
    except Exception:
        return value


def _parse_block(raw: str) -> tuple[str, dict]:
    """Parse one <tool_call> body into (name, arguments).

    Accepts two forms:
      1. JSON:  {"name": "http", "arguments": {...}}
      2. Qwen native XML:  <function=http><parameter=k>v</parameter>...</function>
    Raises ValueError if neither matches.
    """
    # 1) JSON form
    try:
        data = json.loads(raw)
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict):
            return data.get("name", ""), (data.get("arguments") or {})
    except Exception:
        pass
    # 2) Native XML form (what the Qwen3.5 models reliably emit)
    fm = _FUNCTION_RE.search(raw)
    if fm:
        name = fm.group(1).strip()
        args = {k.strip(): _coerce(v) for k, v in _PARAMETER_RE.findall(fm.group(2))}
        return name, args
    raise ValueError("neither valid JSON nor a <function=...> block")


def parse_tool_calls_detailed(content: str) -> tuple[list[dict], list[str]]:
    """Extract tool calls from <tool_call>...</tool_call> blocks in LLM output.

    Returns (calls, errors). `errors` describes any <tool_call> block that was
    present but could not be parsed, so callers can feed the reason back to the
    model instead of silently dropping the turn.
    """
    calls: list[dict] = []
    errors: list[str] = []
    # Ignore tool-call drafts inside the model's reasoning so they aren't executed.
    content = _THINK_RE.sub("", content)

    def _add(name: str, args: dict) -> None:
        if name.startswith("mcp_tool_"):
            args = {"tool_name": name, "arguments": args or {}}
            name = "call_tool"
        calls.append({"id": f"call_{len(calls)}", "name": name, "arguments": args})

    # 1) Native XML: <function=NAME><parameter=k>v</parameter>...</function>. Matched DIRECTLY (not
    #    requiring a wrapping <tool_call></tool_call>) — GLM-5.2 often omits the closing </tool_call>.
    for fm in _FUNCTION_RE.finditer(content):
        name = fm.group(1).strip()
        args = {k.strip(): _coerce(v) for k, v in _PARAMETER_RE.findall(fm.group(2))}
        _add(name, args)

    # 2) JSON form inside <tool_call>{...}</tool_call> (skip <function> blocks — handled above)
    for i, m in enumerate(_TOOL_CALL_RE.findall(content)):
        raw = m.strip()
        if "<function" in raw:
            continue
        try:
            name, args = _parse_block(raw)
        except Exception as e:
            snippet = raw if len(raw) <= 200 else raw[:200] + "…"
            errors.append(f"tool_call #{i + 1}: could not parse — {e}. Received: {snippet}")
            continue
        _add(name, args)

    # Kimi-K2.6 native token format (a model emits one protocol or the other, so this is
    # additive — only one branch matches for any given turn).
    for j, (raw_id, raw_args) in enumerate(_KIMI_TOOL_RE.findall(content)):
        try:
            args = json.loads(raw_args.strip())
        except Exception as e:
            snippet = raw_args.strip()[:200]
            errors.append(f"kimi tool_call #{j + 1}: bad JSON args — {e}. Received: {snippet}")
            continue
        _add(_kimi_tool_name(raw_id), args if isinstance(args, dict) else {})
    return calls, errors


def parse_tool_calls(content: str) -> list[dict]:
    """Extract tool calls from <tool_call>...</tool_call> blocks (errors dropped)."""
    return parse_tool_calls_detailed(content)[0]


def format_tools_text(tools: list[dict]) -> str:
    actual = [t for t in tools if t.get("name") != "list_tools"]
    lines = [f"Available tools ({len(actual)}):"]
    for t in actual:
        lines.append(f"  - {t['name']}: {(t.get('description') or '')[:120]}")
    return "\n".join(lines)
