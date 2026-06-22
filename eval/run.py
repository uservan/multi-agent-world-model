"""Eval runner: given a resolved EvalConfig, run tasks (cached per task/run), aggregate.

Config resolution lives in eval/config.py (init_run / load_run); eval_main.py picks
which one to call. This module just executes.
"""
from __future__ import annotations

import asyncio
import json

from loguru import logger

from eval import data as eval_data
from eval import multi_agent, single_agent
from eval.agent import ModelClient
from eval.config import EvalConfig
from eval.scorer import aggregate
from utils.llm import LLMClient


# ── Run ─────────────────────────────────────────────────────────────────────────

def _model_clients(cfg: EvalConfig) -> tuple[ModelClient, ModelClient | None]:
    orch = ModelClient(
        client=LLMClient(api_key=cfg.orch_api_key, base_url=cfg.orch_base_url),
        model=cfg.orch_model, temperature=cfg.temperature, max_tokens=cfg.max_completion_tokens,
    )
    sub = None
    if cfg.mode == "multi":
        sub = ModelClient(
            client=LLMClient(api_key=cfg.sub_api_key, base_url=cfg.sub_base_url),
            model=cfg.sub_model, temperature=cfg.temperature, max_tokens=cfg.max_completion_tokens,
        )
    return orch, sub


async def _run_one(cfg, task, resources, verifiers, orch, sub, run_idx) -> dict:
    seed = cfg.base_seed + run_idx
    if cfg.mode == "single":
        traj = await single_agent.run_task(task, resources, verifiers, orch, seed, cfg.max_turns)
    else:
        traj = await multi_agent.run_task(
            task, resources, verifiers, orch, sub, seed,
            cfg.max_turns, cfg.max_concurrent, cfg.max_queue,
        )
    traj["run_idx"] = run_idx
    return traj


def run(cfg: EvalConfig) -> None:
    cfg.traj_dir.mkdir(parents=True, exist_ok=True)

    tasks = eval_data.load_tasks(cfg.task_final)
    descriptions = eval_data.load_platform_descriptions(cfg.platforms_input)
    server_paths = eval_data.load_server_paths(cfg.envs_input)
    orch, sub = _model_clients(cfg)

    total = len(tasks) * cfg.n
    done = skipped = 0

    for task in tasks:
        task_id = task["task_id"]
        platforms = eval_data.task_platforms(task)
        resources = eval_data.resolve_resources(platforms, descriptions, server_paths, cfg.databases_dir)
        verifiers = task.get("metadata", {}).get("verifiers", {})
        if not resources:
            logger.warning(f"[{task_id}] no usable platforms, skipping task")
            continue

        for run_idx in range(cfg.n):
            traj_file = cfg.traj_dir / f"{task_id}-{run_idx}.json"
            if traj_file.exists():
                try:
                    if json.loads(traj_file.read_text(encoding="utf-8")).get("status") == "complete":
                        skipped += 1
                        continue
                except Exception:
                    pass
            try:
                traj = asyncio.run(_run_one(cfg, task, resources, verifiers, orch, sub, run_idx))
                traj_file.write_text(json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
                done += 1
                logger.info(f"[{task_id}-{run_idx}] acc={traj['acc']:.2f} ({done+skipped}/{total})")
            except Exception as e:
                logger.error(f"[{task_id}-{run_idx}] failed: {e}")

    logger.success(f"Eval runs done: {done} new, {skipped} cached. Aggregating…")
    _aggregate(cfg)


def _price(in_cost: float | None, out_cost: float | None) -> tuple[float, float] | None:
    if in_cost is None and out_cost is None:
        return None
    return (in_cost or 0.0, out_cost or 0.0)


def _aggregate(cfg: EvalConfig) -> None:
    aggregate(
        cfg.run_dir,
        _price(cfg.orch_input_cost, cfg.orch_output_cost),
        _price(cfg.sub_input_cost, cfg.sub_output_cost),
    )
