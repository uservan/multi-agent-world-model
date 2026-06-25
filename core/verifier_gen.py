"""
Generate and sanity-check verification functions for each task×platform.

Flow per task×platform:
  1. Generate verifier: platform_plan + task_ops + schema + seed data
  2. Sanity check: seed_db as both initial and final (no-write path sanity)
  3. On failure: retry with error context up to max_retries
  4. Save result to verifiers.jsonl
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from utils.llm import LLMClient
from tqdm import tqdm

from core.config import PipelineConfig


# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Python developer specializing in database verification for RL training. Generate a deterministic Python function that compares initial and final SQLite database states to verify if an AI agent completed a task correctly.

Requirements:
1. Signature: def verify_task_completion(initial_db_path: str, final_db_path: str, task_id: str) -> bool
2. ONLY verify WRITE operations — rows added (INSERT) or rows changed (UPDATE) in final_db that differ from initial_db. Do NOT check that any GET/search/read endpoint returned specific values.
3. Filter ALL queries by task_id using WHERE task_id = ?
4. Compare initial_db (before agent) vs final_db (after agent) to detect what changed
5. Return True ONLY when 100% certain the correct write(s) exist in final_db with correct field values
6. Return False in ALL other cases (no write, wrong write, error, partial)
7. If the platform plan shows only read operations are needed (no writes), return True immediately
8. Import all libraries inside the function body
9. Use try/except around all DB operations — never raise exceptions, return False on error
10. Be strict: if the write references the wrong record (wrong price, wrong rating, wrong category, etc.), return False
11. NEVER use system-generated return values from write operations (e.g. auto-generated order_id, record_id, confirmation codes) to locate rows — these are unknown at verification time. Instead verify using known business field values visible in the seed data (e.g. product_id, user_id, quantity, price, status)
12. Check the ESSENTIAL OUTCOME with EXISTENCE checks, not incidental state. Confirm that a row matching the required final state EXISTS with the correct field values (i.e. at least one matching row: SELECT ... WHERE <required business criteria>). Do NOT require an exact total row count, and do NOT require that an id resolves to exactly one row.
13. Be ROBUST to harmless extras: do NOT return False merely because the agent left extra, duplicate, or incomplete rows that do not contradict the required outcome (e.g. an uncommitted duplicate, an extra cart). Judge whether the required correct end state was achieved — not whether the table is pristine.
14. Enforce an exact count ONLY when the task explicitly requires a specific number (e.g. "create exactly 3 X", "shortlist 3 distinct candidates"). Otherwise verify that each required item exists, ignoring harmless extras.

Output format (valid JSON, no markdown fences):
{
  "reasoning": "what specific write checks verify task completion",
  "no_op_result": false,
  "python_code": "complete function code as a string",
  "function_name": "verify_task_completion"
}

no_op_result: what your function returns when the agent does nothing (initial_db == final_db).
- false: the task requires writes, so doing nothing means incomplete
- true: the task requires no writes (read-only platform, or conditional_branch source where agent correctly found nothing)"""


USER_PROMPT_TEMPLATE = """Generate a verification function for the following task on platform: {platform_name}

Task ID: {task_id}
Goal: {goal}
Expected Outcome on {platform_name}: {expected_outcome}

Platform Plan (what each sub-agent should accomplish):
{platform_plan}

Task Operations on {platform_name}:
{task_operations}

Database Schema (DDL):
{schema_ddl}

Seed Data for this task_id (initial DB state before agent runs):
{db_dump}

{conditional_note}

Verification strategy (WRITE-OPS ONLY):
- Identify which tables will have new/modified rows after the agent runs (based on platform plan and expected outcome)
- Compare initial_db vs final_db, ALL queries filtered by task_id = '{task_id}'
- Detect new rows added or rows modified in write tables in final_db vs initial_db
- Cross-reference new writes against seed data to confirm correctness (correct values per task conditions)
- Return True only when the correct write(s) exist with correct field values
- Return False in all other cases
- Do NOT verify read operations (searches, lookups, GET endpoints)

Output format (valid JSON, no markdown fences):
{{
  "reasoning": "what write checks verify this specific task",
  "no_op_result": false,
  "python_code": "complete Python function code",
  "function_name": "verify_task_completion"
}}"""


CONDITIONAL_NOTE = """IMPORTANT: This platform is the SOURCE of a conditional_branch scene transition.
The agent searched this platform but found NO matching results (only distractor data was seeded).
The agent should NOT have made any write operations on this platform.
Verify: final_db has NO new records in write tables for task_id = '{task_id}' compared to initial_db.
If no incorrect writes were made → return True."""


# ── IO helpers ─────────────────────────────────────────────────────────────────

def load_tasks(path: str) -> list[dict]:
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(tasks)} tasks from {path}")
    return tasks


def load_schemas(path: str) -> dict[str, dict]:
    schemas: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("name"):
                    schemas[item["name"]] = item
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(schemas)} schemas from {path}")
    return schemas


def load_existing(output_path: str) -> dict[str, list[dict]]:
    """Load verifiers_gen.jsonl → {task_id: [platform verifier entries]}."""
    result: dict[str, list[dict]] = {}
    if not os.path.exists(output_path):
        return result
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                tid = entry.get("task_id", "")
                if tid:
                    result.setdefault(tid, []).append(entry)
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded existing verifiers for {len(result)} tasks from {output_path}")
    return result


def append_entry(output_path: str, entry: dict) -> None:
    """Append one verifier entry as a JSONL line."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_task_plans(path: str) -> dict[str, dict]:
    plans: dict[str, dict] = {}
    if not os.path.exists(path):
        return plans
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                tid = item.get("task_id", "")
                if tid:
                    plans[tid] = item
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(plans)} task plans from {path}")
    return plans


def load_seeded_task_ids(data_records_path: str) -> set[str]:
    """Return task_ids that have been successfully seeded (status=ok in data_records.jsonl)."""
    seeded: set[str] = set()
    if not os.path.exists(data_records_path):
        logger.warning(f"data_records not found: {data_records_path}")
        return seeded
    with open(data_records_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("status") == "ok" and item.get("task_id"):
                    seeded.add(item["task_id"])
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(seeded)} seeded task_ids from {data_records_path}")
    return seeded


# ── Helpers ────────────────────────────────────────────────────────────────────

def _meta(task: dict) -> dict:
    return task.get("metadata", {})


def _get_no_correct_data_platforms(task: dict) -> set[str]:
    no_correct: set[str] = set()
    meta = _meta(task)
    scene_platforms = meta.get("scene_platforms", [])
    for transition in meta.get("scene_transitions", []):
        if transition.get("pattern") == "conditional_branch":
            from_scene = transition.get("from", -1)
            if 0 <= from_scene < len(scene_platforms):
                for platform in scene_platforms[from_scene]:
                    no_correct.add(platform)
    return no_correct



def _get_platform_plan(task_plans: dict, task_id: str, platform: str) -> dict:
    task_entry = task_plans.get(task_id, {})
    for scene_plan in task_entry.get("scene_plans", []):
        plan = scene_plan.get("plan", {})
        if platform in plan:
            return plan[platform]
    return {}


def _format_schema_ddl(schema_item: dict) -> str:
    return "\n".join(t.get("ddl", "") for t in schema_item.get("schemas", []))


def _format_task_operations(task: dict, platform: str) -> str:
    ops = _meta(task).get("task_operations", {}).get(platform, [])
    if not ops:
        return "  (none)"
    lines = []
    for i, sub_agent in enumerate(ops):
        lines.append(f"  Sub-agent {i + 1}:")
        for step in sub_agent:
            if isinstance(step, dict):
                lines.append(f"    - {step.get('action', '')}({json.dumps(step.get('params', {}))})")
            else:
                lines.append(f"    - {step}")
    return "\n".join(lines)


def _dump_task_data(db_path: str, task_id: str, schema_item: dict) -> str:
    if not os.path.exists(db_path):
        return "(database not found)"

    conn = sqlite3.connect(db_path)
    lines = []
    try:
        for table in schema_item.get("schemas", []):
            table_name = table.get("table", "")
            if not table_name:
                continue
            try:
                cursor = conn.execute(
                    f"SELECT * FROM {table_name} WHERE task_id = ? LIMIT 30",
                    (task_id,),
                )
                rows = cursor.fetchall()
                col_names = [d[0] for d in cursor.description]
                lines.append(f"Table: {table_name} — columns: {', '.join(col_names)}")
                if rows:
                    for row in rows:
                        lines.append(f"  {dict(zip(col_names, row))}")
                else:
                    lines.append("  (no rows for this task_id)")
                lines.append("")
            except Exception as e:
                lines.append(f"Table: {table_name} (query error: {e})\n")
    finally:
        conn.close()

    return "\n".join(lines).strip() or "(no data found)"


def _robust_json_loads(text: str) -> dict:
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


# ── LLM call ──────────────────────────────────────────────────────────────────

_CLASSIFY_SYSTEM = """You are analyzing API task operations. Determine whether the given platform operations include any write actions (create, update, delete, cancel, place order, add, remove, submit, etc.) or are purely read-only (search, get, list, filter, fetch, view, etc.).
Output valid JSON only: {"has_writes": true} or {"has_writes": false}"""


def _llm_classify_ops(
    client: LLMClient,
    model: str,
    task: dict,
    platform: str,
    max_completion_tokens: int = 256,
) -> bool:
    """Ask LLM if platform task_operations contain any write actions. Returns True if has writes."""
    ops_str = _format_task_operations(task, platform)
    messages = [
        {"role": "system", "content": _CLASSIFY_SYSTEM},
        {"role": "user", "content": f"Platform: {platform}\n\nOperations:\n{ops_str}"},
    ]
    try:
        raw = client.complete(model, messages, max_completion_tokens)
        result = _robust_json_loads(raw)
        return bool(result.get("has_writes", True))
    except Exception:
        return True  # assume has writes on failure to avoid skipping real verifiers


def call_llm(
    client: LLMClient,
    model: str,
    task: dict,
    platform: str,
    schema_item: dict,
    db_path: str,
    platform_plan: dict,
    no_correct_data: bool,
    error_history: list[str],
    max_completion_tokens: int = 8192,
) -> dict:
    task_id = task["task_id"]
    schema_ddl = _format_schema_ddl(schema_item)
    task_operations_str = _format_task_operations(task, platform)
    expected_outcome = _meta(task).get("expected_outcome", {}).get(platform, "")
    db_dump = _dump_task_data(db_path, task_id, schema_item)
    conditional_note = CONDITIONAL_NOTE.format(task_id=task_id) if no_correct_data else ""

    user_content = USER_PROMPT_TEMPLATE.format(
        platform_name=platform,
        task_id=task_id,
        goal=task.get("goal", ""),
        expected_outcome=expected_outcome,
        platform_plan=json.dumps(platform_plan, ensure_ascii=False, indent=2) if platform_plan else "(no plan available)",
        task_operations=task_operations_str,
        schema_ddl=schema_ddl,
        db_dump=db_dump,
        conditional_note=conditional_note,
    )

    if error_history:
        error_block = "\n\n".join(f"Error #{i+1}:\n{e}" for i, e in enumerate(error_history))
        user_content += f"\n\nPrevious generation errors to fix:\n{error_block}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return _robust_json_loads(client.complete(model, messages, max_completion_tokens))


# ── Sanity check ───────────────────────────────────────────────────────────────

def execute_verification_code(
    python_code: str,
    function_name: str,
    db_path: str,
    no_op_result: bool,
    task_id: str,
) -> dict:
    """Run the generated function with seed DB as both initial and final (agent did nothing).

    no_op_result: what the LLM declared this verifier should return for a no-op agent.
    - False → task requires writes; doing nothing must return False
    - True  → task is read-only; doing nothing must return True
    """
    if not os.path.exists(db_path):
        return {"execution_status": "error", "error_message": f"Database not found: {db_path}"}

    original_mode = os.stat(db_path).st_mode
    try:
        os.chmod(db_path, 0o444)

        namespace = {
            "sqlite3": sqlite3,
            "json": json,
            "os": os,
            "__builtins__": __builtins__,
        }
        exec(python_code, namespace)

        verify_func = namespace.get(function_name)
        if not verify_func:
            return {"execution_status": "error", "error_message": f"Function '{function_name}' not found"}

        result = verify_func(db_path, db_path, task_id)

        if not isinstance(result, bool):
            return {"execution_status": "error", "error_message": f"Invalid return type: expected bool, got {type(result).__name__}"}

        if result != no_op_result:
            return {
                "execution_status": "error",
                "error_message": (
                    f"Sanity check failed: no-op returned {result}, "
                    f"expected {no_op_result} (declared by LLM as no_op_result)"
                ),
            }

        return {"execution_status": "success"}

    except Exception as e:
        return {"execution_status": "error", "error_message": str(e)}
    finally:
        try:
            os.chmod(db_path, original_mode)
            # SQLite in WAL mode creates -shm/-wal sidecar files that inherit the
            # read-only mode; restore them too so later writers aren't blocked.
            for sidecar in (f"{db_path}-shm", f"{db_path}-wal"):
                if os.path.exists(sidecar):
                    os.chmod(sidecar, original_mode)
        except Exception:
            pass


# ── Per task×platform processing ───────────────────────────────────────────────

def process_task_platform(
    client: LLMClient,
    model: str,
    task: dict,
    platform: str,
    schema_item: dict,
    db_path: str,
    platform_plan: dict,
    no_correct_data: bool,
    max_retries: int,
    max_completion_tokens: int,
) -> dict:
    task_id = task["task_id"]

    # Ask LLM if this platform has any write operations; skip verifier if read-only
    if not _llm_classify_ops(client, model, task, platform, max_completion_tokens):
        logger.info(f"[{task_id}::{platform}] read-only, skipping verifier")
        return {
            "task_id": task_id, "platform": platform,
            "verify_fn": "", "function_name": "",
            "no_op_result": True, "read_only": True, "status": "ok",
        }

    error_history: list[str] = []
    python_code = ""
    function_name = "verify_task_completion"

    for attempt in range(1, max_retries + 1):
        # Step 1: Generate verifier
        try:
            result = call_llm(
                client, model, task, platform, schema_item, db_path,
                platform_plan, no_correct_data, error_history, max_completion_tokens,
            )
        except Exception as e:
            logger.warning(f"[{task_id}::{platform}] LLM call failed (attempt {attempt}): {e}")
            error_history.append(f"LLM error: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            continue

        python_code = result.get("python_code", "")
        function_name = result.get("function_name", "verify_task_completion")
        no_op_result = bool(result.get("no_op_result", False))

        if not python_code:
            error_history.append("Empty python_code from LLM")
            continue

        # Step 2: Sanity check — seed_db as both initial and final
        exec_result = execute_verification_code(python_code, function_name, db_path, no_op_result, task_id)
        if exec_result["execution_status"] == "success":
            logger.success(f"[{task_id}::{platform}] Done (attempt {attempt})")
            return {
                "task_id": task_id, "platform": platform,
                "verify_fn": python_code, "function_name": function_name,
                "no_op_result": no_op_result, "status": "ok",
            }

        err = exec_result.get("error_message", "sanity check failed")
        logger.warning(f"[{task_id}::{platform}] Sanity check failed (attempt {attempt}): {err}")
        error_history.append(f"Sanity check: {err}")
        if attempt < max_retries:
            time.sleep(2 ** attempt)

    logger.error(f"[{task_id}::{platform}] All {max_retries} attempts failed")
    return {
        "task_id": task_id, "platform": platform,
        "verify_fn": python_code, "function_name": function_name,
        "no_op_result": False, "status": "failed",
    }


# ── Run ────────────────────────────────────────────────────────────────────────

def run(args: PipelineConfig) -> None:
    tasks = load_tasks(args.tasks_output)
    schemas = load_schemas(args.schemas_output)
    task_plans = load_task_plans(args.task_plans_output)

    # Only process tasks that have been successfully seeded with data
    seeded = load_seeded_task_ids(args.data_records)

    # Load existing results; build done set at task_id::platform granularity (status=ok only)
    existing: dict[str, list[dict]] = load_existing(args.verifier_gen_output)
    done: set[str] = {
        f"{task_id}::{entry['platform']}"
        for task_id, entries in existing.items()
        for entry in entries
        if entry.get("status") == "ok"
    }

    jobs: list[tuple[dict, str, bool]] = []
    for task in tasks:
        task_id = task["task_id"]
        if task_id not in seeded:
            continue
        no_correct_platforms = _get_no_correct_data_platforms(task)
        all_platforms = [p for scene in _meta(task).get("scene_platforms", []) for p in scene]
        for platform in dict.fromkeys(all_platforms):
            if f"{task_id}::{platform}" in done:
                continue
            if platform not in schemas:
                continue
            safe = platform.lower().replace(" ", "_").replace("/", "_")
            db_path = os.path.join(args.databases_dir, f"{safe}.db")
            if not os.path.exists(db_path):
                logger.warning(f"[{task_id}] DB not found for {platform}, skipping")
                continue
            jobs.append((task, platform, platform in no_correct_platforms))

    logger.info(f"Total jobs: {len(jobs)} (skipped {len(done)} tasks already done)")

    if not jobs:
        logger.success("All task×platform verifiers already generated.")
        return

    client = LLMClient(api_key=args.api_key, base_url=args.base_url, aws_region=args.aws_region)

    success = failed = 0

    def _process(job: tuple[dict, str, bool]) -> dict:
        task, platform, no_correct = job
        task_id = task["task_id"]
        safe = platform.lower().replace(" ", "_").replace("/", "_")
        db_path = os.path.join(args.databases_dir, f"{safe}.db")
        platform_plan = _get_platform_plan(task_plans, task_id, platform)
        return process_task_platform(
            client, args.model,
            task, platform, schemas[platform], db_path,
            platform_plan, no_correct,
            args.max_retries, args.max_completion_tokens,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(_process, job): job for job in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating verifiers"):
            entry = future.result()
            append_entry(args.verifier_gen_output, entry)
            if entry["status"] == "ok":
                success += 1
            else:
                failed += 1
    logger.success(f"Done. {success} OK, {failed} failed out of {len(jobs)} jobs.")
    logger.success(f"Verifiers saved to: {args.verifier_gen_output}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Generate verification functions per task×platform")
    parser.add_argument("--tasks_output", type=str, default=defaults.tasks_output)
    parser.add_argument("--schemas_output", type=str, default=defaults.schemas_output)
    parser.add_argument("--databases_dir", type=str, default=defaults.databases_dir)
    parser.add_argument("--verifier_gen_output", type=str, default=defaults.verifier_gen_output)
    parser.add_argument("--data_records", type=str, default=defaults.data_records)
    parser.add_argument("--task_plans_output", type=str, default=defaults.task_plans_output)
    parser.add_argument("--model", type=str, default=defaults.model)
    parser.add_argument("--api_key", type=str, default=defaults.api_key)
    parser.add_argument("--base_url", type=str, default=defaults.base_url)
    parser.add_argument("--concurrency", type=int, default=defaults.concurrency)
    parser.add_argument("--max_retries", type=int, default=defaults.max_retries)
    parser.add_argument("--max_completion_tokens", type=int, default=defaults.max_completion_tokens)
    parsed = parser.parse_args()

    cfg = PipelineConfig()
    for k, v in vars(parsed).items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    run(cfg)


if __name__ == "__main__":
    main()
