from __future__ import annotations
import os
import json
import shutil
import sqlite3
from contextlib import nullcontext
from pathlib import Path

from loguru import logger

from utils.fix_locks import PlatformLocks
from utils.llm import LLMClient
from utils.suggestions import load_data_suggestions
from utils.task_utils import load_task_supplements, merge_task_supplement
from core.config import PipelineConfig


# ── Prompts ────────────────────────────────────────────────────────────────────

FIX_DATA_SYSTEM = """You are a database engineer patching seed data for multi-agent task testing.

Given:
- Task goal and expected outcome: what the task is trying to accomplish and the final state required
- Task operations: the exact API calls the agents are supposed to make (with expected IDs/values)
- Data suggestions: specific missing or incorrect records found during real agent runs
- Current database state: what already exists in the DB for this task
- Schema DDL: the exact table/column definitions

Generate SQL statements (INSERT, UPDATE, or DELETE) that make the database match the state required for the task to succeed.

Rules:
- Every INSERT must include an explicit integer `id` value — look at existing rows to find max id per table, use max+1; never omit `id` (no AUTOINCREMENT)
- Every INSERT must include task_id in its VALUES
- Every UPDATE and DELETE must filter by task_id in the WHERE clause
- Follow schema DDL exactly — never reference columns that do not exist in the DDL
- For INSERT: include ALL NOT NULL columns
- For UPDATE: only modify the specific fields that need to change
- Do not insert rows that already exist (check current DB state first)
- Use the exact IDs, amounts, and string values from the task operations and suggestions — never invent values
- If the current database state already satisfies all the suggestions (nothing is missing or wrong), set "nothing_to_fix": true and leave "statements" empty — do NOT generate unnecessary SQL
- Return ONLY valid JSON, no markdown

Return:
{
  "nothing_to_fix": false,
  "statements": [
    "INSERT INTO bills (task_id, id, bill_id, ...) VALUES ('...', 5, '...', ...);",
    "UPDATE vendors SET open_balance = 0.0 WHERE task_id = '...' AND vendor_id = '...';"
  ]
}"""

FIX_DATA_USER = """Task ID: {task_id}
Platform: {platform}

Task goal:
{goal}

Expected outcome on this platform:
{expected_outcome}

Task operations (what the agents will do — use these IDs and values when seeding):
{task_ops}

Data issues found during real agent runs (primary source for what to fix):
{suggestions}

Additional goal context from simulation analysis:
{goal_supplement}

Task operation suggestions from simulation analysis:
{task_op_sugs}

Current database state for task_id={task_id}:
{db_dump}

Database schema DDL:
{schema_ddl}

Generate SQL statements (INSERT/UPDATE/DELETE) so the database contains all records the agents need."""


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


def load_schemas(path: str) -> dict[str, dict]:
    schemas = {item["name"]: item for item in _load_jsonl(path) if item.get("name")}
    logger.info(f"Loaded {len(schemas)} schemas")
    return schemas


def load_tasks(path: str) -> dict[str, dict]:
    """Read tasks.jsonl → {task_id: full_task_dict}."""
    result: dict[str, dict] = {}
    for item in _load_jsonl(path):
        task_id = item.get("task_id")
        if not task_id:
            continue
        result[task_id] = item
    logger.info(f"Loaded {len(result)} tasks from {path}")
    return result


def _format_schema_ddl(schema_item: dict) -> str:
    return "\n".join(t.get("ddl", "") for t in schema_item.get("schemas", []))


def _dump_db(db_path: str, task_id: str, schema_item: dict) -> str:
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
                cur = conn.execute(
                    f"SELECT * FROM {table_name} WHERE task_id = ? LIMIT 30",
                    (task_id,),
                )
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                lines.append(f"Table: {table_name} ({', '.join(cols)})")
                for row in rows:
                    lines.append(f"  {dict(zip(cols, row))}")
                if not rows:
                    lines.append("  (no rows for this task_id)")
                lines.append("")
            except Exception as e:
                lines.append(f"Table: {table_name} (error: {e})\n")
    finally:
        conn.close()
    return "\n".join(lines).strip() or "(no data)"


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


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _try_stmts_in_memory(schema_ddl: str, stmts: list[str]) -> tuple[bool, str]:
    """Validate SQL statements against an in-memory copy of the schema."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        for ddl_stmt in (s.strip() for s in schema_ddl.split(";") if s.strip()):
            try:
                conn.execute(ddl_stmt)
            except Exception:
                pass
        for stmt in stmts:
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        conn.commit()
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def _execute_fix_stmts(db_path: str, stmts: list[str]) -> tuple[bool, str]:
    """Copy DB to .tmp, apply fix statements, then atomically replace the original."""
    tmp_path = db_path + ".tmp"
    shutil.copy2(db_path, tmp_path)
    conn = sqlite3.connect(tmp_path, timeout=30)
    failed_stmt = ""
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN")
        for stmt in stmts:
            stmt = stmt.strip()
            if not stmt:
                continue
            failed_stmt = stmt
            conn.execute(stmt)
        conn.execute("COMMIT")
        conn.close()
        os.replace(tmp_path, db_path)
        return True, ""
    except Exception as e:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return False, f"{e}\nstmt: {failed_stmt[:200]}"


# ── LLM call ──────────────────────────────────────────────────────────────────

def _generate_fix_sql(
    client: LLMClient,
    model: str,
    task_id: str,
    platform: str,
    suggestions: list[str],
    goal_supplement: list[str],
    task_op_sugs: list[str],
    goal: str,
    task_ops: list,
    expected_outcome: str,
    db_dump: str,
    schema_ddl: str,
    max_completion_tokens: int,
    prev_error: str | None = None,
) -> list[str] | None:
    suggestions_block = "\n".join(f"- {s}" for s in suggestions) or "(none)"
    goal_sup_block = "\n".join(f"- {s}" for s in goal_supplement) or "(none)"
    task_op_sug_block = "\n".join(f"- {s}" for s in task_op_sugs) or "(none)"
    user_content = FIX_DATA_USER.format(
        task_id=task_id,
        platform=platform,
        goal=goal or "(not available)",
        expected_outcome=expected_outcome or "(not available)",
        task_ops=json.dumps(task_ops, ensure_ascii=False)[:5000] if task_ops else "(none)",
        suggestions=suggestions_block,
        goal_supplement=goal_sup_block,
        task_op_sugs=task_op_sug_block,
        db_dump=db_dump,
        schema_ddl=schema_ddl,
    )
    if prev_error:
        user_content += f"\n\nPrevious attempt failed with error — fix your SQL:\n{prev_error[:400]}"

    raw = client.complete(model, [
        {"role": "system", "content": FIX_DATA_SYSTEM},
        {"role": "user", "content": user_content},
    ], max_completion_tokens)
    result = _robust_json(raw)
    if result.get("nothing_to_fix"):
        return None, raw  # signal: already correct, no statements needed
    stmts = result.get("statements", [])
    return [s.strip() for s in stmts if isinstance(s, str) and s.strip()], raw


# ── Per-platform fix ───────────────────────────────────────────────────────────

def _process_fix(
    client: LLMClient,
    model: str,
    task_id: str,
    platform: str,
    sug_ctx: dict,
    task_ctx: dict,
    schema_item: dict,
    db_path: str,
    max_retries: int,
    max_completion_tokens: int,
    platform_locks: PlatformLocks | None = None,
) -> bool:
    if not os.path.exists(db_path):
        logger.warning(f"[{task_id}::{platform}] DB not found at {db_path}, skipping fix")
        return False

    schema_ddl = _format_schema_ddl(schema_item)
    db_dump = _dump_db(db_path, task_id, schema_item)
    prev_error: str | None = None

    task_ops = task_ctx.get("task_operations", {}).get(platform, [])
    expected_outcome = task_ctx.get("expected_outcome", {}).get(platform, "")

    for attempt in range(1, max_retries + 1):
        try:
            stmts, raw = _generate_fix_sql(
                client, model, task_id, platform,
                sug_ctx.get("data", []),
                sug_ctx.get("goal_supplement", []),
                sug_ctx.get("task_op", []),
                task_ctx.get("goal", ""),
                task_ops,
                expected_outcome,
                db_dump, schema_ddl, max_completion_tokens, prev_error,
            )
        except Exception as e:
            logger.warning(f"[{task_id}::{platform}] fix SQL generation failed (attempt {attempt}): {e}")
            prev_error = str(e)
            continue

        if stmts is None:
            logger.info(f"[{task_id}::{platform}] LLM says data already correct, nothing to fix")
            return True
        if not stmts:
            logger.warning(f"[{task_id}::{platform}] LLM returned no statements (attempt {attempt})")
            prev_error = "No SQL statements returned"
            continue

        mem_ok, mem_err = _try_stmts_in_memory(schema_ddl, stmts)
        if not mem_ok:
            logger.warning(f"[{task_id}::{platform}] memory validation failed (attempt {attempt}): {mem_err}")
            prev_error = mem_err
            continue

        ctx = platform_locks.get(platform) if platform_locks else nullcontext()
        with ctx:
            ok, err = _execute_fix_stmts(db_path, stmts)
        if ok:
            logger.success(f"[{task_id}::{platform}] data fix applied ({len(stmts)} statements)")
            return True

        logger.warning(f"[{task_id}::{platform}] execution failed (attempt {attempt}): {err}")
        prev_error = err

    logger.error(f"[{task_id}::{platform}] data fix failed after {max_retries} attempts")
    return False


# ── Run ────────────────────────────────────────────────────────────────────────

def run(args: PipelineConfig, batch_file: str, *, platform_locks: PlatformLocks | None = None) -> None:
    schemas = load_schemas(args.schemas_output)
    tasks = load_tasks(args.tasks_output)
    task_supplements = load_task_supplements(args.task_supplements_output)
    data_suggestions = load_data_suggestions(batch_file)

    if not data_suggestions:
        logger.info("No data suggestions found, skipping data fix.")
        return

    client = LLMClient(api_key=args.api_key, base_url=args.base_url, aws_region=args.aws_region)

    work_items: list[tuple[str, str, dict, dict, str]] = []
    for task_id, platform_map in data_suggestions.items():
        full_task = tasks.get(task_id)
        if full_task:
            merged = merge_task_supplement(full_task, task_supplements.get(task_id))
            merged_meta = merged.get("metadata", {})
            task_ctx = {
                "goal": merged.get("goal", ""),
                "task_operations": merged_meta.get("task_operations", {}),
                "expected_outcome": merged_meta.get("expected_outcome", {}),
            }
        else:
            task_ctx = {}
        for platform, sug_ctx in platform_map.items():
            if platform not in schemas:
                logger.warning(f"[{task_id}::{platform}] no schema found, skipping")
                continue
            safe = platform.lower().replace(" ", "_").replace("/", "_")
            db_path = os.path.join(args.databases_dir, f"{safe}.db")
            work_items.append((task_id, platform, sug_ctx, task_ctx, db_path))

    logger.info(f"Data fix: {len(work_items)} (task, platform) pairs to patch")

    success = failed = 0
    for task_id, platform, sug_ctx, task_ctx, db_path in work_items:
        try:
            ok = _process_fix(
                client, args.model,
                task_id, platform, sug_ctx, task_ctx,
                schemas[platform], db_path,
                args.max_retries,
                args.max_completion_tokens,
                platform_locks,
            )
            if ok:
                success += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"[{task_id}::{platform}] unexpected error: {e}")
            failed += 1

    logger.success(f"Data fix done. {success} succeeded, {failed} failed out of {len(work_items)} pairs.")
