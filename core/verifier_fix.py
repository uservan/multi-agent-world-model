"""
Apply verifier suggestions to fix generated verifier functions.

Flow per (task_id, platform):
  1. Load verifier suggestions from suggestions.jsonl
  2. Load current verify_fn + no_op_result from verifiers_gen.jsonl
  3. Stage 1 LLM: suggestions + goal + expected_outcome + schema_ddl + db_dump + old verify_fn → change plan
  4. Stage 2 LLM (with retries): change plan + old verify_fn → new verify_fn
  5. Sanity check: seed_db as both initial and final (no-op test)
  6. On pass: mark fixed; at end rewrite verifiers_gen.jsonl and clear suggestions
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import nullcontext
from pathlib import Path

from loguru import logger

from core.config import PipelineConfig
from core.verifier_gen import (
    execute_verification_code,
    load_tasks,
    load_schemas,
    _format_schema_ddl,
    _dump_task_data,
    _meta,
)
from utils.llm import LLMClient
from utils.suggestions import load_verifier_suggestions
from utils.task_utils import load_task_supplements, merge_task_supplement


# ── Prompts ────────────────────────────────────────────────────────────────────

ANALYZE_VERIFIER_FIX_SYSTEM = """You are analyzing a broken verifier function and producing a precise change plan.

Your job is to understand what the verifier SHOULD check (based on the goal and expected outcome), what data actually exists (from the schema and DB dump), and what the current verifier is doing wrong (based on the suggestions). Then output:

When the fix is about an OVER-STRICT verifier, make it check the ESSENTIAL OUTCOME with EXISTENCE checks rather than incidental state: confirm a row matching the required final state EXISTS with correct field values (at least one match), instead of requiring an exact total row count, requiring an id to resolve to exactly one row, or failing on harmless extra/duplicate/incomplete rows that do not contradict the outcome. Keep an exact count ONLY when the task explicitly requires a specific number.

1. no_op_result: does a correct verifier return True or False when run with NO changes (initial_db == final_db)?
   - false if the task requires writes (creating/updating/deleting records) — a no-op agent should fail verification
   - true if the task is purely read-only — a no-op agent trivially passes
2. change_plan: a concrete step-by-step plan of exactly what to change. Be specific: name the exact variables, columns, and logic that need to change.

Return ONLY valid JSON:
{
  "no_op_result": false,
  "change_plan": "Step 1: ... Step 2: ..."
}"""

ANALYZE_VERIFIER_FIX_USER = """Task ID: {task_id}
Platform: {platform}
Goal: {goal}
Expected outcome: {expected_outcome}

Schema (DDL):
{schema_ddl}

Current DB rows for this task:
{db_dump}

Current verifier function:
```python
{verify_fn}
```

Identified issues:
{suggestions}

Based on the above, determine no_op_result and write a step-by-step change plan describing exactly what to fix in the verifier."""


FIX_VERIFIER_SYSTEM = """You are fixing a Python verifier function by applying a change plan.

The verifier has signature: def verify_task_completion(initial_db_path: str, final_db_path: str, task_id: str) -> bool

Rules you must follow:
- Keep the same function signature and function name
- Filter ALL queries by task_id using WHERE task_id = ?
- Compare initial_db vs final_db to detect what changed
- Import all libraries inside the function body
- Use try/except around all DB operations — never raise exceptions, return False on error
- Apply ONLY the changes described in the change plan — do not restructure or rewrite unrelated logic

Return ONLY valid JSON, no markdown:
{
  "verify_fn": "complete fixed function code as a string"
}"""

FIX_VERIFIER_USER = """Task ID: {task_id}
Platform: {platform}

Change plan:
{change_plan}

Current verifier function:
```python
{verify_fn}
```

Apply the change plan and return the fixed verify_fn."""


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


def load_verifiers(path: str) -> dict[tuple[str, str], dict]:
    """Read verifiers_gen.jsonl → {(task_id, platform): entry}."""
    result: dict[tuple[str, str], dict] = {}
    for entry in _load_jsonl(path):
        tid = entry.get("task_id", "")
        plat = entry.get("platform", "")
        if tid and plat:
            result[(tid, plat)] = entry
    logger.info(f"Loaded {len(result)} verifier entries from {path}")
    return result


def rewrite_verifiers(path: str, updated: dict[tuple[str, str], dict]) -> None:
    """Rewrite verifiers_gen.jsonl replacing verify_fn (and optionally no_op_result) for fixed pairs."""
    entries = _load_jsonl(path)
    for entry in entries:
        key = (entry.get("task_id", ""), entry.get("platform", ""))
        if key in updated:
            entry["verify_fn"] = updated[key]["verify_fn"]
            entry["status"] = "ok"
            if "no_op_result" in updated[key]:
                entry["no_op_result"] = updated[key]["no_op_result"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
    logger.info(f"Rewrote {len(updated)} fixed verifiers in {path}")


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


# ── Per-pair fix ───────────────────────────────────────────────────────────────

def _process_fix(
    client: LLMClient,
    model: str,
    task_id: str,
    platform: str,
    suggestions: list[str],
    verifier_entry: dict,
    db_path: str,
    goal: str,
    expected_outcome: str,
    schema_item: dict,
    max_retries: int,
    max_completion_tokens: int,
) -> str | None:
    """Returns new verify_fn string on success, None on failure."""
    old_verify_fn = verifier_entry.get("verify_fn", "")
    function_name = verifier_entry.get("function_name", "verify_task_completion")
    no_op_result = bool(verifier_entry.get("no_op_result", False))

    if not old_verify_fn:
        logger.warning(f"[{task_id}::{platform}] no existing verify_fn, skipping")
        return None

    suggestions_block = "\n".join(f"- {s}" for s in suggestions)

    # Stage 1: analyze what to change → produce a change plan
    schema_ddl = _format_schema_ddl(schema_item)
    db_dump = _dump_task_data(db_path, task_id, schema_item)

    try:
        analyze_user = ANALYZE_VERIFIER_FIX_USER.format(
            task_id=task_id,
            platform=platform,
            goal=goal,
            expected_outcome=expected_outcome,
            schema_ddl=schema_ddl,
            db_dump=db_dump,
            verify_fn=old_verify_fn,
            suggestions=suggestions_block,
        )
        raw_analysis = client.complete(model, [
            {"role": "system", "content": ANALYZE_VERIFIER_FIX_SYSTEM},
            {"role": "user", "content": analyze_user},
        ], max_completion_tokens)
        analysis_result = _robust_json(raw_analysis)
        change_plan = analysis_result.get("change_plan", "").strip()
        # Override no_op_result if Stage 1 gives a more accurate classification
        if "no_op_result" in analysis_result:
            no_op_result = bool(analysis_result["no_op_result"])
            logger.info(f"[{task_id}::{platform}] Stage 1 no_op_result={no_op_result}, change plan: {len(change_plan)} chars")
    except Exception as e:
        logger.warning(f"[{task_id}::{platform}] Stage 1 analysis failed: {e}, falling back to suggestions as plan")
        change_plan = ""

    if not change_plan:
        change_plan = f"Apply the following fixes:\n{suggestions_block}"

    # Stage 2: apply change plan → new verify_fn (with retries)
    prev_error: str | None = None
    current_verify_fn = old_verify_fn  # updated to last generated version after each attempt

    for attempt in range(1, max_retries + 1):
        user_content = FIX_VERIFIER_USER.format(
            task_id=task_id,
            platform=platform,
            change_plan=change_plan,
            verify_fn=current_verify_fn,
        )
        if prev_error:
            user_content += f"\n\nPrevious attempt failed sanity check — fix your code:\n{prev_error[:1000]}"

        try:
            raw = client.complete(model, [
                {"role": "system", "content": FIX_VERIFIER_SYSTEM},
                {"role": "user", "content": user_content},
            ], max_completion_tokens)
            result = _robust_json(raw)
            new_fn = result.get("verify_fn", "").strip()
        except Exception as e:
            logger.warning(f"[{task_id}::{platform}] LLM call failed (attempt {attempt}): {e}")
            prev_error = str(e)
            continue

        if not new_fn:
            logger.warning(f"[{task_id}::{platform}] empty verify_fn returned (attempt {attempt})")
            prev_error = "Empty verify_fn returned"
            continue

        check = execute_verification_code(new_fn, function_name, db_path, no_op_result, task_id)
        if check["execution_status"] == "success":
            logger.success(f"[{task_id}::{platform}] verifier fix passed sanity check (attempt {attempt})")
            return new_fn, no_op_result

        prev_error = check.get("error_message", "sanity check failed")
        logger.warning(f"[{task_id}::{platform}] sanity check failed (attempt {attempt}): {prev_error}")
        current_verify_fn = new_fn  # use latest generated version as base for next retry

    logger.error(f"[{task_id}::{platform}] verifier fix failed after {max_retries} attempts")
    return None


# ── Run ────────────────────────────────────────────────────────────────────────

def run(args: PipelineConfig, batch_file: str, *, lock: threading.Lock | None = None) -> None:
    verifier_suggestions = load_verifier_suggestions(batch_file)
    if not verifier_suggestions:
        logger.info("No verifier suggestions found, skipping verifier fix.")
        return

    verifiers = load_verifiers(args.verifier_gen_output)

    tasks_list = load_tasks(args.tasks_output)
    tasks_by_id: dict[str, dict] = {t["task_id"]: t for t in tasks_list}
    task_supplements = load_task_supplements(args.task_supplements_output)

    schemas = load_schemas(args.schemas_output)

    client = LLMClient(api_key=args.api_key, base_url=args.base_url, aws_region=args.aws_region)

    work_items: list[tuple] = []
    for task_id, platform_map in verifier_suggestions.items():
        raw_task = tasks_by_id.get(task_id)
        if not raw_task:
            logger.warning(f"[{task_id}] task not found in tasks.jsonl, skipping")
            continue
        task = merge_task_supplement(raw_task, task_supplements.get(task_id))
        goal = task.get("goal", "")

        for platform, suggestions in platform_map.items():
            key = (task_id, platform)
            if key not in verifiers:
                logger.warning(f"[{task_id}::{platform}] no verifier entry found, skipping")
                continue
            schema_item = schemas.get(platform, {})
            safe = platform.lower().replace(" ", "_").replace("/", "_")
            db_path = os.path.join(args.databases_dir, f"{safe}.db")
            if not os.path.exists(db_path):
                logger.warning(f"[{task_id}::{platform}] seed DB not found, skipping")
                continue
            platform_expected_outcome = _meta(task).get("expected_outcome", {}).get(platform, "")
            work_items.append((task_id, platform, suggestions, verifiers[key], db_path, goal, platform_expected_outcome, schema_item))

    logger.info(f"Verifier fix: {len(work_items)} (task, platform) pairs to fix")

    fixed_pairs: set[tuple[str, str]] = set()
    updated_fns: dict[tuple[str, str], dict] = {}

    for task_id, platform, suggestions, verifier_entry, db_path, goal, expected_outcome, schema_item in work_items:
        try:
            result = _process_fix(
                client, args.model,
                task_id, platform, suggestions, verifier_entry, db_path,
                goal, expected_outcome, schema_item,
                args.max_retries, args.max_completion_tokens,
            )
            if result:
                new_fn, corrected_no_op = result
                fixed_pairs.add((task_id, platform))
                updated_fns[(task_id, platform)] = {"verify_fn": new_fn, "no_op_result": corrected_no_op}
        except Exception as e:
            logger.error(f"[{task_id}::{platform}] unexpected error: {e}")

    if updated_fns:
        with lock if lock else nullcontext():
            rewrite_verifiers(args.verifier_gen_output, updated_fns)

    success = len(fixed_pairs)
    failed = len(work_items) - success
    logger.success(f"Verifier fix done. {success} succeeded, {failed} failed out of {len(work_items)} pairs.")
