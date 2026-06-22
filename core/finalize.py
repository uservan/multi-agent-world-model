"""Finalize: consolidate verified tasks into a single self-contained eval/training dataset.

Reads the scattered pipeline artifacts and emits one line per fully-verified task to
task_final.jsonl, shaped to match slime-n's Sample (label / prompt / metadata) so the
same file can drive both local eval and RL training without further transformation.

A task is included only if:
  - its data was seeded (DB exists), AND
  - every platform in its scene_platforms has a verified_platforms record.

Each output record:
  {
    "label":  task_id,
    "prompt": goal + goal_supplement,        # merged via the fix logic
    "metadata": {
      "structure", "scene_platforms", "scene_transitions", "budget",
      "expected_outcome",   # supplement-applied
      "task_operations",    # supplement-applied (reference solution)
      "verifiers": {platform: verify_fn_str},   # "" for read-only platforms
      "goal", "goal_supplement",                # raw originals, for traceability
    }
  }

Servers and databases are NOT embedded — eval/slime resolve them by platform name
from outputs/generated/servers/{name}.py and outputs/generated/databases/{name}.db.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger

from core.config import PipelineConfig
from core.verifier_step import (
    load_tasks,
    load_verifiers_gen,
    load_verified_platforms,
    load_seeded_task_ids,
    _all_platforms_verified,
)
from utils.task_utils import load_task_supplements, merge_task_supplement


def _platforms_of(task: dict) -> list[str]:
    """Flattened, de-duplicated platform list across all scenes."""
    return list(dict.fromkeys(
        p for scene in task.get("metadata", {}).get("scene_platforms", []) for p in scene
    ))


def _build_record(task: dict, supplement: dict | None, task_verifiers: dict) -> dict:
    """Merge supplement into the task and shape it as a slime-n Sample record."""
    merged = merge_task_supplement(task, supplement)
    meta = merged.get("metadata", {})
    task_id = merged["task_id"]

    # Per-platform verify_fn string ("" for read-only platforms that need no verification).
    verifiers: dict[str, str] = {}
    for platform in _platforms_of(merged):
        entry = task_verifiers.get(platform, {})
        verifiers[platform] = entry.get("verify_fn", "") or ""

    return {
        "label": task_id,
        "prompt": merged.get("goal", ""),          # goal + goal_supplement (fix logic)
        "metadata": {
            "structure":         meta.get("structure", []),
            "scene_platforms":   meta.get("scene_platforms", []),
            "scene_transitions": meta.get("scene_transitions", []),
            "budget":            meta.get("budget"),
            "expected_outcome":  meta.get("expected_outcome", {}),
            "task_operations":   meta.get("task_operations", {}),
            "verifiers":         verifiers,
            # raw originals for traceability
            "goal":              task.get("goal", ""),
            "goal_supplement":   (supplement or {}).get("goal_supplement", ""),
        },
    }


def run(args: PipelineConfig) -> None:
    """Consolidate fully-verified tasks into task_final.jsonl."""
    tasks = load_tasks(args.tasks_output)
    verifiers = load_verifiers_gen(args.verifier_gen_output)      # {tid: {platform: entry}}
    verified = load_verified_platforms(args.verified_platforms_output)  # {tid: {platform: ctx}}
    supplements = load_task_supplements(args.task_supplements_output)   # {tid: supplement}
    seeded = load_seeded_task_ids(args.data_records)             # set[tid]

    records: list[dict] = []
    skipped_unseeded = skipped_unverified = 0

    for task in tasks:
        task_id = task["task_id"]
        if task_id not in seeded:
            skipped_unseeded += 1
            continue
        if not _all_platforms_verified(task, verified):
            skipped_unverified += 1
            continue
        records.append(_build_record(task, supplements.get(task_id), verifiers.get(task_id, {})))

    out_path = args.task_final_output
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_path, out_path)

    logger.success(
        f"Finalize: wrote {len(records)} tasks to {out_path} "
        f"(skipped {skipped_unverified} not fully verified, {skipped_unseeded} not seeded)"
    )

    _log_agent_breakdown(records)


def _log_agent_breakdown(records: list[dict]) -> None:
    """Log a breakdown by total sub-agent count: how many distinct structures and tasks each."""
    from collections import defaultdict

    by_count: dict[int, dict] = defaultdict(lambda: {"tasks": 0, "structures": set()})
    for rec in records:
        structure = rec["metadata"].get("structure", [])
        total_agents = sum(sum(scene) for scene in structure)
        by_count[total_agents]["tasks"] += 1
        by_count[total_agents]["structures"].add(json.dumps(structure))

    logger.info("Breakdown by total sub-agent count:")
    logger.info(f"  {'#agents':>8} | {'#structures':>11} | {'#tasks':>6}")
    for count in sorted(by_count):
        info = by_count[count]
        logger.info(f"  {count:>8} | {len(info['structures']):>11} | {info['tasks']:>6}")
    logger.info(f"  {'total':>8} | {'':>11} | {len(records):>6}")
