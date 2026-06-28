"""
Consolidate task-level fix suggestions (goal_supplement, outcome, task_op) into
a single per-task supplement entry saved to task_supplements.jsonl.

Flow per task_id:
  1. Collect goal_supplement / outcome / task_op suggestions across all platforms
  2. One LLM call with original task + suggestions + schema DDL + endpoint list
  3. LLM produces: goal_supplement string, per-platform outcome/task_ops overrides
  4. Merge with any existing supplement entry and save atomically
"""
from __future__ import annotations

import json
import os
import re
import threading
from contextlib import nullcontext
from pathlib import Path

from loguru import logger

from core.config import PipelineConfig
from utils.llm import LLMClient
from utils.task_utils import merge_task_supplement


# ── Prompts ────────────────────────────────────────────────────────────────────

TASK_SUPPLEMENT_SYSTEM = """You are updating a task definition based on failure analysis suggestions.

You will receive:
- Original task: goal, per-platform expected_outcomes, per-platform task_operations (indexed by sub-agent)
- Existing supplement (if any): previously applied goal_supplement and platform overrides
- Per-platform suggestions: goal_supplement hints, outcome corrections, task_op corrections
- Rule 5 violations (if any): cross-sub-agent creation dependencies that must be fixed
- Schema DDL: exact table/column definitions available on each platform
- API endpoints: routes available on each platform

Produce an updated supplement with these fields:
- goal_supplement: concise additional context the agent needs that is NOT in the original goal or existing supplement. Use generic phrasing; do NOT include runtime-generated IDs.
- platforms: for each platform that has suggestions, provide updated expected_outcome and/or task_operations

CONSTRAINTS — ordered by priority:
1. Prefer fixing data-level errors first: wrong field values, wrong IDs, wrong amounts, missing expected return values in existing task_operations — these are the most common root cause and safest to fix
2. task_operations: do NOT change the overall logic or structure; only correct concrete values within existing operations. You may add a step only if it is strictly required for the task to succeed and is clearly supported by the available endpoints and schema — adding steps should be rare. Exception: Rule 5 violations listed for a platform MUST be fixed using constraint 6 — this overrides the no-structural-change rule.
3. expected_outcome: correct factual errors (wrong field names, impossible checks) — only reference tables and columns that exist in the provided schema DDL
4. goal_supplement: add only what the agent genuinely cannot discover at runtime. Do not instruct the agent to use endpoints or tables outside the provided list
5. Do NOT invent data values — every value must come from the existing task_operations, suggestions, or schema DDL
6. Rule 5 fix (Strategy A — independent creation): if Rule 5 violations are listed for a platform, each violating sub-agent MUST independently create its own resource instance with a unique ID — rewrite its steps so it calls the appropriate create/add action itself and all its subsequent steps use its own newly created ID. Never share or copy IDs across sub-agents. Only the violating sub-agent's steps change; other sub-agents remain unchanged.

Return ONLY valid JSON. Use null for any field with no changes needed:
{
  "goal_supplement": "..." | null,
  "platforms": {
    "<platform_name>": {
      "expected_outcome": "..." | null,
      "task_operations": [[{...sub_agent_0_ops...}, ...], [{...sub_agent_1_ops...}, ...]] | null
    }
  }
}"""

TASK_SUPPLEMENT_USER = """Task ID: {task_id}
Original goal: {goal}

Current goal_supplement (replace or extend this based on new suggestions; output null if no longer needed):
{existing_supplement}

Per-platform context and suggestions (outcome and task_operations already reflect any prior supplement — replace them if corrections are needed):
{platform_blocks}

Produce the updated supplement. Only include platforms that actually need changes."""


# ── IO helpers ─────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return items


def load_tasks(path: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in _load_jsonl(path):
        tid = item.get("task_id")
        if tid:
            result[tid] = item
    logger.info(f"Loaded {len(result)} tasks from {path}")
    return result


def load_schemas(path: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in _load_jsonl(path):
        name = item.get("name")
        if name:
            result[name] = item
    return result


def load_envs(path: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in _load_jsonl(path):
        name = item.get("name")
        if name:
            result[name] = item
    return result


def load_task_supplements(path: str) -> dict[str, dict]:
    """Read task_supplements.jsonl → {task_id: supplement}. Last entry wins."""
    result: dict[str, dict] = {}
    for item in _load_jsonl(path):
        tid = item.get("task_id")
        if tid:
            result[tid] = item
    logger.info(f"Loaded task supplements for {len(result)} tasks from {path}")
    return result


def save_task_supplements(path: str, updates: dict[str, dict]) -> None:
    """Merge updates into existing task_supplements.jsonl and write atomically."""
    existing = load_task_supplements(path)
    for task_id, new_entry in updates.items():
        if task_id in existing:
            old = existing[task_id]
            # goal_supplement: replace if new one provided
            if new_entry.get("goal_supplement"):
                old["goal_supplement"] = new_entry["goal_supplement"]
            # platforms: merge new platforms over old
            old_platforms = old.setdefault("platforms", {})
            for platform, pdata in new_entry.get("platforms", {}).items():
                old_platforms[platform] = pdata
        else:
            existing[task_id] = new_entry

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for entry in existing.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
    logger.info(f"Saved task supplements for {len(updates)} tasks to {path}")


def _load_items(path: str) -> list[dict]:
    if path.endswith(".json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                item = json.load(f)
                return [item] if isinstance(item, dict) else []
        except Exception:
            return []
    return _load_jsonl(path)


def load_task_fix_suggestions(
    batch_file: str,
) -> dict[str, dict[str, dict[str, list]]]:
    """Read batch file → {task_id: {platform: {goal_supplement, outcome, task_op}}}."""
    result: dict[str, dict[str, dict[str, list]]] = {}
    for item in _load_items(batch_file):
        task_id = item.get("task_id")
        if not task_id:
            continue
        for platform, sug_dict in item.get("suggestions", {}).items():
            gs = sug_dict.get("goal_supplement", [])
            oc = sug_dict.get("outcome", [])
            to = sug_dict.get("task_op", [])
            if gs or oc or to:
                result.setdefault(task_id, {})[platform] = {
                    "goal_supplement": gs,
                    "outcome": oc,
                    "task_op": to,
                }
    logger.info(f"Loaded task fix suggestions for {len(result)} tasks from {batch_file}")
    return result


def _list_endpoints(server_path: str) -> str:
    """Extract route signatures from a FastAPI server file."""
    try:
        with open(server_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception:
        return "(could not read server)"
    lines = []
    for line in code.splitlines():
        m = re.search(r'@app\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']', line, re.IGNORECASE)
        if m:
            lines.append(f"  {m.group(1).upper()} {m.group(2)}")
    return "\n".join(lines) or "(no endpoints found)"


def _format_schema_ddl(schema_item: dict) -> str:
    parts = []
    for table in schema_item.get("schemas", []):
        ddl = table.get("ddl", "")
        if ddl:
            parts.append(ddl)
    return "\n\n".join(parts) or "(no schema)"


def _robust_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise


# ── Rule 5 detection ──────────────────────────────────────────────────────────

_RULE5_DETECT_PROMPT = """Check for two types of Rule 5 violations within a single platform's task_operations.

Type A — param dependency: sub-agent k (k ≥ 2) uses as a param a value that FIRST appears in a
previous sub-agent's returns from a WRITE action (add_*, create_*, place_*, book_*, submit_*,
register_*, open_*, or any action that creates a new resource at runtime).

Type B — duplicate write return: two or more sub-agents have WRITE actions that return the same
resource identifier (same ID, URL, token, reference code, etc.), meaning they are NOT truly
independently creating resources. Generic status fields (e.g. status: "success") are exempt.
READ actions (get_*, fetch_*, list_*, search_*, retrieve_*) returning the same ID as another sub-agent's
action is expected and normal — do NOT flag these as Type B violations.
Each sub-agent's WRITE actions must produce unique IDs/values.

Platform: {platform}
task_operations:
{ops_json}

For each violation found, include a "violation_type" field ("param" or "duplicate_return") plus:
- Type A: consuming_subagent (1-based), param_key, value, producing_subagent (1-based), producing_action
- Type B: subagent_a (1-based), subagent_b (1-based), shared_value, field_key

Return JSON:
{{"violations": [{{"violation_type": "param", "consuming_subagent": <int>, "param_key": "...", "value": "...", "producing_subagent": <int>, "producing_action": "..."}} | {{"violation_type": "duplicate_return", "subagent_a": <int>, "subagent_b": <int>, "shared_value": "...", "field_key": "..."}}]}}
Return {{"violations": []}} if none found."""


def _detect_rule5_violations(
    client: LLMClient,
    model: str,
    platform: str,
    platform_ops: list,
    max_completion_tokens: int,
) -> list[dict]:
    if len(platform_ops) <= 1:
        return []
    ops_json = json.dumps({platform: platform_ops}, ensure_ascii=False, indent=2)
    prompt = _RULE5_DETECT_PROMPT.format(platform=platform, ops_json=ops_json)
    try:
        raw = client.complete(model, [{"role": "user", "content": prompt}], min(max_completion_tokens, 4096))
        result = _robust_json(raw)
        return result.get("violations", [])
    except Exception as e:
        logger.warning(f"[{platform}] Rule 5 detection failed: {e}")
        return []


# ── Per-task fix ───────────────────────────────────────────────────────────────

def _process_task(
    client: LLMClient,
    model: str,
    task_id: str,
    task: dict,
    platform_suggestions: dict[str, dict[str, list]],
    schemas: dict[str, dict],
    envs: dict[str, dict],
    existing_supplement: dict,
    max_completion_tokens: int,
) -> dict | None:
    merged_task = merge_task_supplement(task, existing_supplement)
    goal = task.get("goal", "")
    merged_meta = merged_task.get("metadata", {})
    merged_outcomes: dict = merged_meta.get("expected_outcome", {})
    merged_ops: dict = merged_meta.get("task_operations", {})

    existing_goal_sup = existing_supplement.get("goal_supplement", "(none)")

    platform_blocks: list[str] = []
    for platform, sugs in platform_suggestions.items():
        schema_ddl = _format_schema_ddl(schemas.get(platform, {}))
        env_item = envs.get(platform, {})
        endpoints = _list_endpoints(env_item.get("server_path", ""))

        current_outcome = merged_outcomes.get(platform, "(none)")
        current_task_ops = merged_ops.get(platform, [])

        gs_text = "\n".join(f"  - {s}" for s in sugs.get("goal_supplement", [])) or "  (none)"
        oc_text = "\n".join(f"  - {s}" for s in sugs.get("outcome", [])) or "  (none)"
        to_text = "\n".join(f"  - {s}" for s in sugs.get("task_op", [])) or "  (none)"

        violations = _detect_rule5_violations(client, model, platform, current_task_ops, max_completion_tokens)
        rule5_block = ""
        if violations:
            vlines = []
            for v in violations:
                if v.get("violation_type") == "duplicate_return":
                    vlines.append(
                        f"  - [duplicate_return] sub-agent {v.get('subagent_a', '?')} and sub-agent {v.get('subagent_b', '?')} "
                        f"both return '{v.get('field_key', '?')}' = {json.dumps(v.get('shared_value', ''))}"
                    )
                else:
                    vlines.append(
                        f"  - [param] sub-agent {v.get('consuming_subagent', '?')} param '{v.get('param_key', '?')}' = "
                        f"{json.dumps(v.get('value', ''))} "
                        f"(first produced by sub-agent {v.get('producing_subagent', '?')}'s {v.get('producing_action', '?')})"
                    )
            rule5_block = f"\nRule 5 violations (must fix using Strategy A — independent creation):\n" + "\n".join(vlines) + "\n"
            logger.info(f"[{task_id}] Rule 5 violations on {platform}: {len(violations)} found")

        platform_blocks.append(
            f"Platform: {platform}\n"
            f"Current expected_outcome:\n  {current_outcome}\n"
            f"Current task_operations (by sub-agent):\n{json.dumps(current_task_ops, ensure_ascii=False, indent=2)}\n"
            f"Goal supplement suggestions:\n{gs_text}\n"
            f"Outcome correction suggestions:\n{oc_text}\n"
            f"Task op correction suggestions:\n{to_text}\n"
            f"{rule5_block}"
            f"Schema DDL:\n{schema_ddl}\n"
            f"API endpoints:\n{endpoints}"
        )

    if not platform_blocks:
        return None

    user_content = TASK_SUPPLEMENT_USER.format(
        task_id=task_id,
        goal=goal,
        existing_supplement=existing_goal_sup,
        platform_blocks="\n\n---\n\n".join(platform_blocks),
    )
    try:
        raw = client.complete(model, [
            {"role": "system", "content": TASK_SUPPLEMENT_SYSTEM},
            {"role": "user", "content": user_content},
        ], max_completion_tokens)
        result = _robust_json(raw)
    except Exception as e:
        logger.warning(f"[{task_id}] task_fix LLM call failed: {e}")
        return None

    goal_sup = result.get("goal_supplement") or None
    platforms_out: dict = {}
    for platform, pdata in (result.get("platforms") or {}).items():
        if not isinstance(pdata, dict):
            continue
        entry: dict = {}
        if pdata.get("expected_outcome"):
            entry["expected_outcome"] = pdata["expected_outcome"]
        if pdata.get("task_operations") is not None:
            entry["task_operations"] = pdata["task_operations"]
        if entry:
            platforms_out[platform] = entry

    if not goal_sup and not platforms_out:
        return None

    supplement: dict = {"task_id": task_id}
    if goal_sup:
        supplement["goal_supplement"] = goal_sup
    if platforms_out:
        supplement["platforms"] = platforms_out
    return supplement


# ── Run ────────────────────────────────────────────────────────────────────────

def run(args: PipelineConfig, batch_file: str, *, lock: threading.Lock | None = None) -> None:
    suggestions = load_task_fix_suggestions(batch_file)
    if not suggestions:
        logger.info("No task fix suggestions found, skipping.")
        return

    tasks = load_tasks(args.tasks_output)
    schemas = load_schemas(args.schemas_output)
    envs = load_envs(args.envs_output)
    client = LLMClient.from_config(args)

    # Read existing supplements outside the lock (LLM calls happen here)
    existing_supplements = load_task_supplements(args.task_supplements_output)
    work_items = [
        (task_id, tasks[task_id], platform_map, existing_supplements.get(task_id, {}))
        for task_id, platform_map in suggestions.items()
        if task_id in tasks
    ]
    logger.info(f"Task fix: {len(work_items)} tasks to process")

    new_supplements: dict[str, dict] = {}
    for task_id, task, platform_map, existing_sup in work_items:
        try:
            supplement = _process_task(
                client, args.model,
                task_id, task, platform_map, schemas, envs,
                existing_sup, args.max_completion_tokens,
            )
            if supplement:
                new_supplements[task_id] = supplement
                logger.info(f"[{task_id}] task supplement generated")
            else:
                logger.info(f"[{task_id}] no changes needed")
        except Exception as e:
            logger.error(f"[{task_id}] unexpected error: {e}")

    if new_supplements:
        with lock if lock else nullcontext():
            save_task_supplements(args.task_supplements_output, new_supplements)
    logger.success(f"Task fix done. {len(new_supplements)} supplements saved.")
