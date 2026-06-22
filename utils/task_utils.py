from __future__ import annotations

import json
import os


def load_task_supplements(path: str) -> dict[str, dict]:
    """Read task_supplements.jsonl → {task_id: supplement}. Last entry wins."""
    result: dict[str, dict] = {}
    if not os.path.exists(path):
        return result
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                tid = item.get("task_id")
                if tid:
                    result[tid] = item
            except json.JSONDecodeError:
                pass
    return result


def merge_task_supplement(task: dict, supplement: dict | None) -> dict:
    """Return a new task dict with supplement fields merged over the originals.

    - goal: appended with goal_supplement text if present
    - metadata.expected_outcome: per-platform overrides applied
    - metadata.task_operations: per-platform overrides applied
    """
    sup = supplement or {}
    meta = task.get("metadata", {})

    base_goal = task.get("goal", "")
    goal_sup = sup.get("goal_supplement", "")
    merged_goal = f"{base_goal}\n\nAdditional context:\n{goal_sup}" if goal_sup else base_goal

    sup_platforms = sup.get("platforms", {})
    merged_outcomes = {**meta.get("expected_outcome", {})}
    merged_ops = {**meta.get("task_operations", {})}
    for platform, pdata in sup_platforms.items():
        if pdata.get("expected_outcome"):
            merged_outcomes[platform] = pdata["expected_outcome"]
        if pdata.get("task_operations") is not None:
            merged_ops[platform] = pdata["task_operations"]

    merged_meta = {**meta, "expected_outcome": merged_outcomes, "task_operations": merged_ops}
    return {**task, "goal": merged_goal, "metadata": merged_meta}
