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

from utils.agent_io import parse_tool_calls
from utils.llm import LLMClient


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
) -> tuple[str, dict]:
    """Run one agent until <done>, no tool calls, or max_turns.

    Returns (final_text, total_tokens={"in","out"}). Records system/user prompts
    and every assistant turn (with its tool calls + responses) into event_log.
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
        messages.append({"role": "assistant", "content": text})

        tool_calls = parse_tool_calls(text)
        done = "<done>" in text

        if not tool_calls:
            event_log.add(actor, text, tokens=usage)
            break

        responses: list[str] = []
        for tc in tool_calls:
            resp = await tool_executor(tc)
            responses.append(resp)

        event_log.add(actor, text, tool_calls=[t.get("arguments", t) for t in tool_calls],
                      tool_responses=responses, tokens=usage)

        if done:
            break

        messages.append({"role": "user", "content": "Tool results:\n" + "\n---\n".join(responses)})

    return final_text, total
