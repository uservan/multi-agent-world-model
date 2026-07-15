"""Single-agent eval: one LLM, all platforms, http tool only."""
from __future__ import annotations

import random

from loguru import logger

from eval.agent import EventLog, ModelClient, error_info, run_agent_loop, status_for
from eval.platform import PlatformRuntime
from eval.prompts import build_single_agent_prompt
from eval.scorer import score_task
from eval.tools import make_http_executor


async def run_task(
    task: dict,
    resources: dict[str, dict],
    verifiers: dict[str, str],
    orch: ModelClient,
    seed: int,
    max_turns: int,
) -> dict:
    """Run one single-agent attempt. Returns a complete trajectory dict."""
    task_id = task.get("task_id") or task["label"]
    goal = task.get("prompt", "")

    # Shuffle platform order for this run (hide any scene/ordering signal)
    platforms = list(resources.keys())
    random.Random(seed).shuffle(platforms)

    runtime = PlatformRuntime(task_id)
    event_log = EventLog()
    try:
        # All-or-nothing: every platform must come up, else skip this run entirely
        # (a partial platform set would make scoring meaningless).
        for p in platforms:
            if not runtime.start(p, resources[p]):
                raise RuntimeError(f"platform failed to start: {p}")

        system, user = build_single_agent_prompt(goal, runtime.platform_map())
        executor = make_http_executor(runtime)
        agent_error = None
        tokens = {"in": 0, "out": 0}
        try:
            _, tokens = await run_agent_loop(
                "agent", system, user, orch, executor, event_log, max_turns,
            )
        except Exception as e:
            # Agent loop crashed. Don't drop the run — score whatever it managed to do
            # against the DB and record acc + a classified error. Transient failures
            # (timeout/503/…) yield status="error" (re-run, uncounted); deterministic
            # ones (e.g. context-length overflow) stay "complete" and are scored once.
            agent_error = error_info(e)
            logger.warning(f"[{task_id}] agent loop failed ({agent_error['class']}), "
                           f"scoring partial state: {agent_error['type']}: {agent_error['message']}")

        verifier_results, acc = score_task(runtime, verifiers, task_id)
    finally:
        runtime.cleanup()

    return {
        "task_id": task_id,
        "run_idx": None,                 # filled by caller
        "seed": seed,
        "mode": "single",
        "status": status_for(agent_error),
        "events": event_log.events,
        "verifier_results": verifier_results,
        "acc": acc,
        "error": agent_error,
        "tokens": {"orch": tokens, "sub": {"in": 0, "out": 0}},
    }
