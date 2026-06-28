"""Multi-agent eval: orchestrator (model A) + spawned sub-agents (model B).

The orchestrator gets http + spawn_subagent + get_task_results. Sub-agents get
http only and see only the description the orchestrator gives them. All actors
share the same live servers / X-Task-ID, and append to one EventLog (interleaved
by real time). Sub-agent token usage is summed into a shared counter.
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import uuid

from loguru import logger

from eval.agent import EventLog, ModelClient, run_agent_loop
from eval.platform import PlatformRuntime
from eval.prompts import build_orchestrator_prompt, build_subagent_prompt, SUBAGENT_SUMMARY_PROMPT
from eval.scorer import score_task
from eval.tools import make_http_executor


def _extract_result(text: str) -> str:
    m = re.search(r"<result>\s*(.*?)\s*</result>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


async def run_task(
    task: dict,
    resources: dict[str, dict],
    verifiers: dict[str, str],
    orch: ModelClient,
    sub: ModelClient,
    seed: int,
    max_turns: int,
    max_concurrent: int,
    max_queue: int,
    sub_max_turns: int = 30,
    orch_style: str = "neutral",
) -> dict:
    """Run one multi-agent attempt. Returns a complete trajectory dict."""
    task_id = task.get("task_id") or task["label"]
    goal = task.get("prompt", "")

    platforms = list(resources.keys())
    random.Random(seed).shuffle(platforms)

    runtime = PlatformRuntime(task_id)
    event_log = EventLog()
    http_exec = make_http_executor(runtime)

    # ── Worker-pool state ──────────────────────────────────────────────────────
    semaphore = asyncio.Semaphore(max_concurrent)
    pending: dict[str, asyncio.Task] = {}
    results: dict[str, str] = {}
    sub_tokens = {"in": 0, "out": 0}
    sub_counter = {"n": 0}
    queue_size = {"n": 0}

    async def _run_subagent(tid: str, description: str, return_requirements: str) -> None:
        try:
            async with semaphore:
                actor = f"subagent_{sub_counter['n']}"
                sub_counter["n"] += 1
                system, user = build_subagent_prompt(description, return_requirements)
                final_text, tokens = await run_agent_loop(
                    actor, system, user, sub, http_exec, event_log, sub_max_turns,
                    final_prompt=SUBAGENT_SUMMARY_PROMPT,
                )
                sub_tokens["in"] += tokens["in"]
                sub_tokens["out"] += tokens["out"]
                results[tid] = _extract_result(final_text)
        except asyncio.CancelledError:
            raise                                       # task-end cleanup — let it propagate
        except Exception as e:
            results[tid] = f"ERROR: sub-agent failed: {e}"  # never leave it stuck on "pending"
        finally:
            queue_size["n"] -= 1        # free the queue slot
            pending.pop(tid, None)      # drop the finished task handle from the registry

    async def orchestrator_executor(call: dict) -> str:
        name = call.get("name")
        args = call.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}

        if name == "http":
            return await http_exec(call)

        if name == "spawn_subagent":
            if queue_size["n"] >= max_queue:
                return json.dumps({"error": "queue full", "max_queue": max_queue})
            tid = str(uuid.uuid4())[:8]
            t = asyncio.create_task(
                _run_subagent(tid, args.get("description", ""), args.get("return_requirements", ""))
            )
            pending[tid] = t
            queue_size["n"] += 1
            return json.dumps({"task_id": tid, "status": "started"})

        if name == "get_task_results":
            task_ids = args.get("task_ids", [])
            blocking = bool(args.get("blocking", False))
            timeout = float(args.get("timeout", 30))
            if blocking:
                waits = [pending[t] for t in task_ids if t in pending and not pending[t].done()]
                if waits:
                    # Wait up to `timeout`. Sub-agents not finished by then are NOT
                    # cancelled — they keep running (bounded by sub_max_turns) and stay
                    # "pending" for a later poll. The only place we cancel is task end.
                    await asyncio.wait(waits, timeout=timeout)
            out = {t: results.get(t, "pending") for t in task_ids}
            return json.dumps(out, ensure_ascii=False)

        return json.dumps({"error": f"unknown tool: {name}"})

    try:
        # All-or-nothing: every platform must come up, else skip this run entirely
        # (a partial platform set would make scoring meaningless).
        for p in platforms:
            if not runtime.start(p, resources[p]):
                raise RuntimeError(f"platform failed to start: {p}")

        system, user = build_orchestrator_prompt(
            goal, runtime.platform_map(), max_concurrent, max_queue, orch_style)
        agent_error = None
        orch_tokens = {"in": 0, "out": 0}
        try:
            _, orch_tokens = await run_agent_loop(
                "orchestrator", system, user, orch, orchestrator_executor, event_log, max_turns,
            )
        except Exception as e:
            # Orchestrator crashed (e.g. context-length overflow). Don't drop the run —
            # score whatever was accomplished against the DB and record acc + error.
            agent_error = str(e)
            logger.warning(f"[{task_id}] orchestrator loop failed, scoring partial state: {e}")

        # Cancel any sub-agents still running
        for t in pending.values():
            if not t.done():
                t.cancel()
        if pending:
            await asyncio.gather(*pending.values(), return_exceptions=True)

        verifier_results, acc = score_task(runtime, verifiers, task_id)
    finally:
        runtime.cleanup()

    return {
        "task_id": task_id,
        "run_idx": None,
        "seed": seed,
        "mode": "multi",
        "status": "complete",
        "events": event_log.events,
        "verifier_results": verifier_results,
        "error": agent_error,
        "acc": acc,
        "tokens": {"orch": orch_tokens, "sub": sub_tokens},
    }
