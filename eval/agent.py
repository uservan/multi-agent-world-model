"""Generic HTTP agent loop with event recording and token accounting.

Shared by single- and multi-agent modes. An agent gets a system prompt, talks to
an LLM (text <tool_call> protocol), and executes tools via a pluggable executor.
Every turn is appended to a shared EventLog so parallel actors interleave by real
time; each event is tagged with its actor (role) for filtering.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass, field

from utils.agent_io import parse_tool_calls_detailed
from utils.llm import LLMClient


# Nudge when a turn produced neither a tool call nor <done> (e.g. a thinking model
# spent the whole turn reasoning and emitted empty content). The episode must NOT end
# here — only the model may declare completion via <done>.
_CONTINUE_OR_DONE = (
    "You did not call any tool, and you did not output <done>. The task is not over until "
    "YOU declare it complete. If every part of the task is fully done and verified, reply with:\n"
    "<done>\n<brief summary>\n</done>\n"
    "Otherwise, continue working now — issue the next <tool_call> (http) to make progress."
)


def _format_parse_errors(errors: list[str]) -> str:
    """Feedback message handed back to the model when its tool calls don't parse."""
    return (
        "Some of your <tool_call> blocks could not be parsed and were ignored:\n"
        + "\n".join(errors)
        + "\n\nFix the JSON and re-issue them. Each tool call must be ONE valid JSON "
        "object inside <tool_call> tags, with balanced braces, e.g.:\n"
        '<tool_call>\n{"name": "http", "arguments": {"method": "GET", "base_url": "<url>", '
        '"path": "/<endpoint>", "params": {}}}\n</tool_call>'
    )


# ── LLM holder ─────────────────────────────────────────────────────────────────

@dataclass
class ModelClient:
    """Bundles an LLM client with its model + sampling settings for one actor."""
    client: LLMClient
    model: str
    temperature: float = 1.0
    max_tokens: int = 8192

    async def complete(self, messages: list[dict]) -> tuple[str, dict]:
        return await asyncio.to_thread(
            self.client.complete_with_usage, self.model, messages, self.max_tokens, self.temperature
        )


# ── Event log ──────────────────────────────────────────────────────────────────

class EventLog:
    """Thread/async-safe append-only event list with relative timestamps."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self._start = time.time()
        self._lock = threading.Lock()

    def add(
        self,
        role: str,
        content: str,
        tool_calls: list | None = None,
        tool_responses: list | None = None,
        tokens: dict | None = None,
    ) -> None:
        event: dict = {"role": role, "content": content, "ts": round(time.time() - self._start, 3)}
        if tool_calls:
            event["tool_calls"] = tool_calls
        if tool_responses:
            event["tool_responses"] = tool_responses
        if tokens:
            event["tokens"] = tokens
        with self._lock:
            self.events.append(event)


# ── Agent loop ─────────────────────────────────────────────────────────────────

async def run_agent_loop(
    actor: str,
    system_prompt: str,
    user_prompt: str,
    model: ModelClient,
    tool_executor,                 # async (tool_call: dict) -> str
    event_log: EventLog,
    max_turns: int,
    final_prompt: str | None = None,
) -> tuple[str, dict]:
    """Run one agent until <done>, no tool calls, or max_turns.

    Returns (final_text, total_tokens={"in","out"}). Records system/user prompts
    and every assistant turn (with its tool calls + responses) into event_log.

    If `final_prompt` is given, the loop ALWAYS ends with one extra call using
    `final_prompt` as a user message (no tools), and its output becomes `final_text`.
    Used for sub-agents: regardless of how the run went, the last turn is a concise
    summary of the requested info for the orchestrator — so the orchestrator receives
    that summary, never the full trajectory (which would blow up its context).
    """
    event_log.add(actor, system_prompt)          # role-tagged system prompt
    event_log.add(f"{actor}:user", user_prompt)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    total = {"in": 0, "out": 0}
    final_text = ""

    for _ in range(max_turns):
        text, usage = await model.complete(messages)
        final_text = text
        total["in"] += usage.get("in", 0)
        total["out"] += usage.get("out", 0)
        # `text` is clean content (reasoning is carried out-of-band in `usage` and
        # recorded via event_log, never fed back to the model).
        messages.append({"role": "assistant", "content": text})

        tool_calls, parse_errors = parse_tool_calls_detailed(text)
        done = "<done>" in text

        if not tool_calls:
            event_log.add(actor, text, tokens=usage)
            if done:
                break  # only the model ends the episode — via an explicit <done>
            # No tool call AND no <done>: do NOT end here (a thinking model may have spent
            # the turn reasoning with empty content). Tell it what to do and keep going.
            if parse_errors:
                messages.append({"role": "user", "content": _format_parse_errors(parse_errors)})
            else:
                messages.append({"role": "user", "content": _CONTINUE_OR_DONE})
            continue

        responses: list[str] = []
        for tc in tool_calls:
            resp = await tool_executor(tc)
            responses.append(resp)

        event_log.add(actor, text, tool_calls=[t.get("arguments", t) for t in tool_calls],
                      tool_responses=responses, tokens=usage)

        # A <done> in the SAME turn as tool calls is NOT honored: the model must first see
        # these tool results, then declare completion in a clean turn (no tool calls). Guards
        # against premature termination — e.g. spawning async sub-agents and signaling <done>
        # before their results come back, or a <done> written mid-plan as example text. Only
        # the no-tool-call branch above ends the episode, so <done>'s position doesn't matter.

        result_text = "Tool results:\n" + "\n---\n".join(responses)
        if parse_errors:  # some calls ran, others were malformed — note the ignored ones
            result_text += "\n\n" + _format_parse_errors(parse_errors)
        messages.append({"role": "user", "content": result_text})

    # Always end with a dedicated summary turn (no tools) when requested, so the
    # caller (orchestrator) gets a concise result, never the full trajectory.
    if final_prompt:
        messages.append({"role": "user", "content": final_prompt})
        text, usage = await model.complete(messages)
        final_text = text
        total["in"] += usage.get("in", 0)
        total["out"] += usage.get("out", 0)
        event_log.add(actor, text, tokens=usage)

    return final_text, total
