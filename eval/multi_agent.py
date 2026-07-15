"""Multi-agent eval: orchestrator (model A) + spawned sub-agents (model B).

The orchestrator gets http + spawn_subagent + get_queue_status + get_task_info +
wait_task. Sub-agents get http only and see only the description the orchestrator
gives them. All actors
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

from eval.agent import EventLog, ModelClient, error_info, run_agent_loop, status_for
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
    # tid -> {"state": waiting|running|finished|error, "actor": event-log actor tag}
    meta: dict[str, dict] = {}
    sub_tokens = {"in": 0, "out": 0}
    sub_counter = {"n": 0}
    queue_size = {"n": 0}

    async def _run_subagent(tid: str, description: str, return_requirements: str) -> None:
        try:
            async with semaphore:
                actor = f"subagent_{sub_counter['n']}"
                sub_counter["n"] += 1
                meta[tid].update(state="running", actor=actor)
                system, user = build_subagent_prompt(description, return_requirements)
                final_text, tokens = await run_agent_loop(
                    actor, system, user, sub, http_exec, event_log, sub_max_turns,
                    final_prompt=SUBAGENT_SUMMARY_PROMPT,
                )
                sub_tokens["in"] += tokens["in"]
                sub_tokens["out"] += tokens["out"]
                results[tid] = _extract_result(final_text)
                meta[tid]["state"] = "finished"
        except asyncio.CancelledError:
            raise                                       # task-end cleanup — let it propagate
        except Exception as e:
            results[tid] = f"ERROR: sub-agent failed: {e}"  # never leave it stuck on "pending"
            meta[tid]["state"] = "error"
        finally:
            queue_size["n"] -= 1        # free the queue slot
            pending.pop(tid, None)      # drop the finished task handle from the registry

    def _subagent_logs(tid: str) -> list[dict]:
        """All EventLog messages of this sub-agent, in order, unabridged."""
        actor = meta.get(tid, {}).get("actor")
        if not actor:
            return []
        logs = []
        for ev in event_log.events:
            if ev.get("role") not in (actor, f"{actor}:user"):
                continue
            entry = {"content": ev.get("content", "")}
            if ev.get("tool_calls"):
                entry["tool_calls"] = ev["tool_calls"]
            if ev.get("tool_responses"):
                entry["tool_responses"] = ev["tool_responses"]
            logs.append(entry)
        return logs

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
            meta[tid] = {"state": "waiting", "actor": ""}
            t = asyncio.create_task(
                _run_subagent(tid, args.get("description", ""), args.get("return_requirements", ""))
            )
            pending[tid] = t
            queue_size["n"] += 1
            return json.dumps({"task_id": tid, "status": "started"})

        if name == "get_queue_status":
            out = {"running": [], "waiting": [], "finished": [], "error": []}
            for tid, m in meta.items():
                out[m["state"]].append(tid)
            return json.dumps(out, ensure_ascii=False)

        if name == "get_task_info":
            tid = args.get("task_id", "")
            if tid not in meta:
                return json.dumps({"error": f"unknown task_id: {tid}"})
            want_summary = bool(args.get("summary", True))
            want_logs = bool(args.get("logs", False))
            out: dict = {"task_id": tid, "state": meta[tid]["state"]}
            if want_summary:
                out["summary"] = results.get(tid, "pending")
            if want_logs:
                out["logs"] = _subagent_logs(tid)
            return json.dumps(out, ensure_ascii=False)

        if name == "wait_task":
            tid = args.get("task_id", "")
            timeout = float(args.get("timeout", 60))
            if tid not in meta:
                return json.dumps({"error": f"unknown task_id: {tid}"})
            t = pending.get(tid)
            if t is not None and not t.done():
                # Wait up to `timeout`. A sub-agent not finished by then is NOT
                # cancelled — it keeps running (bounded by sub_max_turns) and can be
                # waited on again. The only place we cancel is task end.
                await asyncio.wait([t], timeout=timeout)
            state = meta[tid]["state"]
            out = {"task_id": tid, "state": state}
            if state in ("finished", "error"):
                out["summary"] = results.get(tid, "")
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
            # Orchestrator crashed. Don't drop the run — score whatever was accomplished
            # against the DB and record acc + a classified error. Transient failures
            # (timeout/503/…) yield status="error" (re-run, uncounted); deterministic
            # ones (e.g. context-length overflow) stay "complete" and are scored once.
            agent_error = error_info(e)
            logger.warning(f"[{task_id}] orchestrator loop failed ({agent_error['class']}), "
                           f"scoring partial state: {agent_error['type']}: {agent_error['message']}")

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
        "status": status_for(agent_error),
        "events": event_log.events,
        "verifier_results": verifier_results,
        "error": agent_error,
        "acc": acc,
        "tokens": {"orch": orch_tokens, "sub": sub_tokens},
    }
