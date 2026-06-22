"""Load the consolidated eval dataset and resolve per-platform resources.

Sources:
  - task_final.jsonl : tasks (label/prompt/metadata with embedded verifiers)
  - platforms.jsonl  : platform descriptions
  - envs.jsonl       : platform server paths
  - databases/{safe}.db : seed databases
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger


def _safe(platform: str) -> str:
    return platform.lower().replace(" ", "_").replace("/", "_")


def _load_jsonl(path: str) -> list[dict]:
    items: list[dict] = []
    if not os.path.exists(path):
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return items


def load_tasks(path: str) -> list[dict]:
    tasks = _load_jsonl(path)
    logger.info(f"Loaded {len(tasks)} eval tasks from {path}")
    return tasks


def load_platform_descriptions(path: str) -> dict[str, str]:
    """platforms.jsonl → {platform_name: description}."""
    result: dict[str, str] = {}
    for item in _load_jsonl(path):
        name = item.get("name")
        if name:
            result[name] = item.get("description", "") or ""
    return result


def load_server_paths(path: str) -> dict[str, str]:
    """envs.jsonl → {platform_name: server_path}."""
    result: dict[str, str] = {}
    for item in _load_jsonl(path):
        name = item.get("name")
        if name and item.get("server_path"):
            result[name] = item["server_path"]
    return result


def task_platforms(task: dict) -> list[str]:
    """Flattened, de-duplicated platform list across all scenes."""
    return list(dict.fromkeys(
        p for scene in task.get("metadata", {}).get("scene_platforms", []) for p in scene
    ))


def resolve_resources(
    platforms: list[str],
    descriptions: dict[str, str],
    server_paths: dict[str, str],
    databases_dir: str,
) -> dict[str, dict]:
    """For each platform, resolve {description, server_path, seed_db}. Skip if missing."""
    resources: dict[str, dict] = {}
    for p in platforms:
        server_path = server_paths.get(p, "")
        seed_db = os.path.join(databases_dir, f"{_safe(p)}.db")
        if not server_path or not os.path.exists(server_path):
            logger.warning(f"[{p}] server not found, skipping platform")
            continue
        if not os.path.exists(seed_db):
            logger.warning(f"[{p}] seed DB not found, skipping platform")
            continue
        resources[p] = {
            "description": descriptions.get(p, ""),
            "server_path": server_path,
            "seed_db": seed_db,
        }
    return resources
