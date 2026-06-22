from __future__ import annotations
import os
import json

from loguru import logger

_ACTIVE_FIELDS = ["env", "data", "goal_supplement", "verifier", "task_op"]


def _platform_has_content(sug_dict: dict) -> bool:
    return any(sug_dict.get(k) for k in _ACTIVE_FIELDS)


def _load_items(path: str) -> list[dict]:
    """Load suggestion items from a single-task .json or multi-task .jsonl file."""
    if not os.path.exists(path):
        return []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            try:
                item = json.load(f)
                return [item] if isinstance(item, dict) else []
            except json.JSONDecodeError:
                return []
    items: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return items


def load_env_suggestions(path: str) -> dict[str, list[str]]:
    """Read suggestion file → {platform: [env_suggestion, ...]}."""
    result: dict[str, list[str]] = {}
    for item in _load_items(path):
        for platform, sug_dict in item.get("suggestions", {}).items():
            env_sug = sug_dict.get("env", [])
            if env_sug:
                result.setdefault(platform, []).extend(env_sug)
    logger.info(f"Loaded env suggestions for {len(result)} platforms from {path}")
    return result


def load_data_suggestions(path: str) -> dict[str, dict[str, dict]]:
    """Read suggestion file → {task_id: {platform: {data, goal_supplement, task_op}}}."""
    result: dict[str, dict[str, dict]] = {}
    for item in _load_items(path):
        task_id = item.get("task_id")
        if not task_id:
            continue
        for platform, sug_dict in item.get("suggestions", {}).items():
            data_sug = sug_dict.get("data", [])
            if data_sug:
                result.setdefault(task_id, {})[platform] = {
                    "data": data_sug,
                    "goal_supplement": sug_dict.get("goal_supplement", []),
                    "task_op": sug_dict.get("task_op", []),
                }
    logger.info(f"Loaded data suggestions for {len(result)} tasks from {path}")
    return result


def load_goal_supplement_suggestions(path: str) -> dict[str, dict[str, list[str]]]:
    """Read suggestion file → {task_id: {platform: [goal_supplement, ...]}}."""
    result: dict[str, dict[str, list[str]]] = {}
    for item in _load_items(path):
        task_id = item.get("task_id")
        if not task_id:
            continue
        for platform, sug_dict in item.get("suggestions", {}).items():
            sugs = sug_dict.get("goal_supplement", [])
            if sugs:
                result.setdefault(task_id, {})[platform] = sugs
    logger.info(f"Loaded goal_supplement suggestions for {len(result)} tasks from {path}")
    return result


def load_verifier_suggestions(path: str) -> dict[str, dict[str, list[str]]]:
    """Read suggestion file → {task_id: {platform: [verifier_suggestion, ...]}}."""
    result: dict[str, dict[str, list[str]]] = {}
    for item in _load_items(path):
        task_id = item.get("task_id")
        if not task_id:
            continue
        for platform, sug_dict in item.get("suggestions", {}).items():
            verifier_sug = sug_dict.get("verifier", [])
            if verifier_sug:
                result.setdefault(task_id, {})[platform] = verifier_sug
    logger.info(f"Loaded verifier suggestions for {len(result)} tasks from {path}")
    return result


def load_data_suggestions_with_task_op(path: str) -> dict[str, dict[str, dict]]:
    """Alias kept for callers that also need task_op alongside data."""
    return load_data_suggestions(path)
