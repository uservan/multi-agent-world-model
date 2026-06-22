from __future__ import annotations
import os
import json
import hashlib
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from utils.llm import LLMClient
from core.config import PipelineConfig


# ── Prompts ────────────────────────────────────────────────────────────────────

GENERATE_DATA_SYSTEM = """You are a database engineer generating seed data for multi-agent task testing.
Given a platform's task steps across all sub-agents, generate realistic rows that agents will find via READ operations,
plus any prerequisite rows needed by WRITE operations.
Rules:
- Read the schema DDL carefully before generating any rows — every column name, type, and constraint matters
- FOREIGN KEY: referenced rows must exist; insert parent rows before child rows.
- NOT NULL: never omit required columns; use realistic non-null values.
- Data types: integers as numbers, booleans as 1/0, strings as quoted, NULL only for truly optional fields.
- Include task_id in every row.
- If multiple sub-agents share an entity (e.g. same user), generate it once only.
Return ONLY valid JSON, no markdown."""

VALIDATE_DATA_SYSTEM = """You are a database QA engineer. Validate that generated data rows are sufficient to support a multi-agent task.

Check every sub-agent operation:
1. READ (search/list/get/find/check): at least one row must satisfy ALL filter conditions specified in that step
2. WRITE prerequisite (create/update/delete/book/submit): all rows the operation depends on must already exist (e.g. the item to update, the parent entity, any FK reference)
3. Expected outcome: the data state must be consistent with the platform's expected outcome

Return ONLY valid JSON — no markdown, no explanation outside the JSON:
{"valid": true}
or
{"valid": false, "reason": "<concise explanation of what is missing or wrong>"}"""

DISTRACTOR_SYSTEM = """You are a database engineer generating distractor rows for multi-agent task testing.
Only generate distractors for tables that agents actively search or filter (e.g. products, listings, orders, inventory).
Skip tables that hold supporting data with no filter conditions (e.g. users, accounts, sessions, config, settings, credentials).
Distractor rows look realistic but violate at least one filter condition from the task's READ operations,
so the agent will not accidentally select them. Every row must satisfy schema constraints.
Return ONLY valid JSON, no markdown."""


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


def _append_jsonl(path: str, item: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_tasks(tasks_path: str) -> list[dict]:
    tasks = [t for t in _load_jsonl(tasks_path) if t.get("task_id")]
    logger.info(f"Loaded {len(tasks)} tasks")
    return tasks


def load_schemas(path: str) -> dict[str, dict]:
    schemas = {item["name"]: item for item in _load_jsonl(path) if item.get("name")}
    logger.info(f"Loaded {len(schemas)} schemas")
    return schemas


def load_specs(path: str) -> dict[str, dict]:
    specs = {item["name"]: item for item in _load_jsonl(path) if item.get("name")}
    logger.info(f"Loaded {len(specs)} specs")
    return specs


def load_envs(path: str) -> dict[str, dict]:
    envs = {item["name"]: item for item in _load_jsonl(path) if item.get("name")}
    logger.info(f"Loaded {len(envs)} envs")
    return envs


def load_records(path: str) -> dict[str, dict]:
    """Load data_records keyed by task_id, last entry wins."""
    records: dict[str, dict] = {}
    for item in _load_jsonl(path):
        tid = item.get("task_id")
        if tid:
            records[tid] = item
    logger.info(f"Loaded {len(records)} data records")
    return records


# ── Task field helpers ─────────────────────────────────────────────────────────

def _meta(task: dict) -> dict:
    return task.get("metadata", {})




# ── Fingerprint ────────────────────────────────────────────────────────────────

def _task_fingerprint(task: dict) -> str:
    ops = json.dumps(_meta(task).get("task_operations", {}), sort_keys=True)
    return hashlib.md5((ops + task.get("goal", "")).encode()).hexdigest()


# ── Format helpers ─────────────────────────────────────────────────────────────

def _format_schema_ddl(schema_item: dict) -> str:
    return "\n".join(t.get("ddl", "") for t in schema_item.get("schemas", []))


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




# ── Data generation ────────────────────────────────────────────────────────────

def _generate_platform_data(
    client: LLMClient, model: str,
    task_id: str, goal: str, platform: str,
    platform_ops: list,
    platform_plan: dict, outcome: str, schema_item: dict,
    max_retries: int, max_completion_tokens: int = 8192,
    prev_error: str | None = None,
) -> list[dict] | None:
    """Generate all core DB rows for a platform across all sub-agents.
    Returns [{table_name, rows}] on success (may be empty if the task is entirely WRITE
    operations needing no seed data), or None if the LLM call/parse failed."""
    schema_ddl = _format_schema_ddl(schema_item)
    sub_agents = platform_plan.get("sub_agents", [])
    error_note = (
        f"\n\nPrevious attempt failed — fix your data:\n{prev_error[:500]}"
        if prev_error else ""
    )
    sa_sections = []
    for i, ops in enumerate(platform_ops):
        sub_task = sub_agents[i] if i < len(sub_agents) else f"Sub-agent {i + 1}"
        sa_sections.append(
            f"Sub-agent {i + 1} — {sub_task}:\n{json.dumps(ops, ensure_ascii=False, indent=2)}"
        )
    prompt = f"""Generate all database rows needed for {platform} to support {len(platform_ops)} sub-agent(s).

Goal: {goal}
Expected outcome: {outcome}
task_id (include in every row): {task_id}

{"".join(f"{s}{chr(10)}{chr(10)}" for s in sa_sections).rstrip()}

Database schema:
{schema_ddl}

Instructions:
- Study the schema DDL carefully: note each table's PRIMARY KEY, FOREIGN KEY references, NOT NULL columns, and column types before generating any rows.
- PRIMARY KEY values must be unique across all generated rows for that table.
- FOREIGN KEY: always generate the parent row first (e.g. users before orders).
- READ steps (search/get/find/list/check): generate rows the agent will find. ALL filter conditions must match exactly.
- WRITE steps (create/update/delete/book/order/submit): generate prerequisite rows the operation depends on. Do NOT generate what the write operation itself creates — that happens at runtime.
- If multiple sub-agents share rows (e.g. same user), generate them once only.
- Do not add unrelated data or errors.{error_note}

Return JSON:
{{
  "tables": [
    {{
      "table_name": "...",
      "rows": [{{"col": value, ...}}, ...]
    }}
  ]
}}"""

    for attempt in range(1, max_retries + 1):
        try:
            raw = client.complete(model, [{"role": "system", "content": GENERATE_DATA_SYSTEM}, {"role": "user", "content": prompt}], max_completion_tokens)
            result = _robust_json(raw)
            return result.get("tables", [])
        except Exception as e:
            logger.warning(f"generate_platform_data {platform} attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return None


def _validate_platform_data(
    client: LLMClient,
    model: str,
    task_id: str,
    goal: str,
    platform: str,
    platform_ops: list,
    outcome: str,
    core_tables: list[dict],
    max_completion_tokens: int = 4096,
) -> tuple[bool, str]:
    """LLM checks that core rows fully support all sub-agent operations. Returns (valid, reason)."""
    prompt = f"""Platform: {platform}
Goal: {goal}
Expected outcome: {outcome}
task_id: {task_id}

Sub-agent operations:
{json.dumps(platform_ops, ensure_ascii=False, indent=2)}

Generated core data rows:
{json.dumps(core_tables, ensure_ascii=False, indent=2)}

Validate that every sub-agent READ and WRITE operation is fully supported by the data above."""

    try:
        raw = client.complete(model, [
            {"role": "system", "content": VALIDATE_DATA_SYSTEM},
            {"role": "user", "content": prompt},
        ], max_completion_tokens).strip()
        result = _robust_json(raw)
        valid = bool(result.get("valid", False))
        reason = result.get("reason", "")
        return valid, reason
    except Exception as e:
        logger.warning(f"[{task_id}::{platform}] validation LLM call failed ({e}), proceeding")
        return True, ""


def _generate_platform_distractors(
    client: LLMClient, model: str,
    task_id: str, platform_ops: list,
    core_tables: list[dict], schema_item: dict,
    distractor_high: int,
    max_retries: int, max_completion_tokens: int = 8192,
    prev_error: str | None = None,
) -> list[dict]:
    """Generate distractor rows for all sub-agents merged with core tables. Returns [{table_name, rows}]."""
    schema_ddl = _format_schema_ddl(schema_item)
    core_by_table = {t["table_name"]: t.get("rows", []) for t in core_tables}
    error_note = (
        f"\n\nPrevious attempt failed — fix your data:\n{prev_error[:500]}"
        if prev_error else ""
    )
    prompt = f"""Generate distractor rows for these database tables.

task_id (include in every row): {task_id}

All sub-agent steps (READ operations across all sub-agents define the filter conditions distractors must violate):
{json.dumps(platform_ops, ensure_ascii=False, indent=2)}

Database schema:
{schema_ddl}

Existing correct rows (do not duplicate):
{json.dumps(core_by_table, ensure_ascii=False, indent=2)}

Rules:
- Only generate distractors for tables that appear in READ steps with filter conditions (search/list/find/get with params). Skip all other tables (users, accounts, sessions, config, settings, credentials, etc.).
- Each distractor must violate AT LEAST ONE READ filter condition from any sub-agent
- Generate ~{distractor_high} distractors per search/filter table
- Values must still be realistic and satisfy schema constraints
- Include task_id in every row{error_note}

Return JSON:
{{
  "distractors": [
    {{
      "table_name": "...",
      "rows": [{{"col": value, ...}}, ...]
    }}
  ]
}}"""

    for attempt in range(1, max_retries + 1):
        try:
            raw = client.complete(model, [{"role": "system", "content": DISTRACTOR_SYSTEM}, {"role": "user", "content": prompt}], max_completion_tokens)
            distractor_tables = _robust_json(raw).get("distractors", [])
            merged: dict[str, list] = {t["table_name"]: list(t.get("rows", [])) for t in core_tables}
            for t in distractor_tables:
                name = t["table_name"]
                rows = t.get("rows", [])
                if name in merged:
                    merged[name].extend(rows)
                else:
                    merged[name] = list(rows)
            return [{"table_name": n, "rows": r} for n, r in merged.items()]
        except Exception as e:
            logger.warning(f"generate_platform_distractors attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
    return core_tables


def _delete_task_data(task_id: str, platforms: list[str], databases_dir: str) -> None:
    """Delete all rows for task_id from every platform DB."""
    for platform in platforms:
        db_path = os.path.join(databases_dir, f"{platform.lower().replace(' ', '_').replace('/', '_')}.db")
        if not os.path.exists(db_path):
            continue
        try:
            conn = sqlite3.connect(db_path, timeout=10)
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            for (table_name,) in tables:
                try:
                    conn.execute(f"DELETE FROM [{table_name}] WHERE task_id = ?", (task_id,))
                except Exception:
                    pass
            conn.commit()
            conn.close()
            logger.debug(f"[{task_id}::{platform}] cleared existing rows")
        except Exception as e:
            logger.warning(f"delete_task_data {platform}: {e}")


# ── DB helpers ─────────────────────────────────────────────────────────────────

def _rows_to_inserts(tables_raw: list[dict], schema_item: dict) -> list[dict]:
    """Convert row dicts to INSERT statement strings."""
    result = []
    for t in tables_raw:
        table_name = t.get("table_name", "")
        rows = t.get("rows", [])
        if not table_name or not rows:
            continue

        # Schema DDL uses bare `id INTEGER` (no AUTOINCREMENT), so omitted or NULL id stays NULL.
        # SQLAlchemy maps it as primary_key=True and returns None for NULL-id rows.
        # Assign sequential integers here so every row has an explicit non-NULL id.
        next_id = 1
        for row in rows:
            if not isinstance(row, dict):
                continue
            if row.get("id") is None:
                row["id"] = next_id
                next_id += 1
            else:
                try:
                    next_id = max(next_id, int(row["id"]) + 1)
                except (TypeError, ValueError):
                    row["id"] = next_id
                    next_id += 1

        stmts = []
        for row in rows:
            if not isinstance(row, dict) or not row:
                continue
            cols = list(row.keys())
            vals = []
            for v in row.values():
                if v is None:
                    vals.append("NULL")
                elif isinstance(v, bool):
                    vals.append("1" if v else "0")
                elif isinstance(v, (int, float)):
                    vals.append(str(v))
                else:
                    escaped = str(v).replace("'", "''")
                    vals.append(f"'{escaped}'")
            stmts.append(
                f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({', '.join(vals)});"
            )
        if stmts:
            result.append({"table_name": table_name, "insert_statements": stmts})
    return result


def _try_inserts_in_memory(schema_ddl: str, tables: list[dict]) -> tuple[bool, str]:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        for stmt in (s.strip() for s in schema_ddl.split(";") if s.strip()):
            try:
                conn.execute(stmt)
            except Exception:
                pass
        for table in tables:
            for stmt in table.get("insert_statements", []):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            return False, f"FK violations: {violations[:3]}"
        conn.commit()
        return True, ""
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def _execute_inserts(db_path: str, task_id: str, tables: list[dict]) -> tuple[bool, str]:
    import random
    all_stmts = []
    for table in tables:
        stmts = [s.strip() for s in table.get("insert_statements", []) if s.strip()]
        random.shuffle(stmts)
        all_stmts.extend(stmts)

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    current_table = ""
    current_stmt = ""
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN")
        for table in tables:
            name = table.get("table_name", "")
            if name:
                conn.execute(f"DELETE FROM {name} WHERE task_id = ?", (task_id,))
        for table in tables:
            current_table = table.get("table_name", "")
            for stmt in table.get("insert_statements", []):
                stmt = stmt.strip()
                if not stmt:
                    continue
                current_stmt = stmt
                conn.execute(stmt)
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            conn.execute("ROLLBACK")
            return False, f"FK violations: {violations[:3]}"
        conn.execute("COMMIT")
        return True, ""
    except Exception as e:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        return False, f"[table={current_table}] {e}\nstmt: {current_stmt[:200]}"
    finally:
        conn.close()


def _validate_row_counts(db_path: str, task_id: str, tables: list[dict]) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        for table in tables:
            name = table.get("table_name", "")
            if not name:
                continue
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {name} WHERE task_id = ?", (task_id,)).fetchone()[0]
                if count == 0:
                    return False
            except Exception:
                pass
        return True
    finally:
        conn.close()


def _count_task_rows(db_path: str, task_id: str, tables: list[dict]) -> dict[str, int]:
    """Return {table_name: row_count} for this task_id."""
    counts: dict[str, int] = {}
    conn = sqlite3.connect(db_path)
    try:
        for table in tables:
            name = table.get("table_name", "")
            if not name:
                continue
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {name} WHERE task_id = ?", (task_id,)).fetchone()[0]
                counts[name] = count
            except Exception:
                pass
    finally:
        conn.close()
    return counts


# ── helpers ────────────────────────────────────────────────────────────────────

def _get_platform_plan(task_plans: dict, task_id: str, platform: str) -> dict:
    for scene_plan in task_plans.get(task_id, {}).get("scene_plans", []):
        if platform in scene_plan.get("plan", {}):
            return scene_plan["plan"][platform]
    return {}


# ── Per-task processing ────────────────────────────────────────────────────────

def _process_task(
    client: LLMClient,
    model: str,
    task: dict,
    schemas: dict,
    databases_dir: str,
    records: dict,
    task_plans: dict,
    distractor_high: int,
    max_retries: int,
    max_completion_tokens: int,
    data_records_path: str,
    write_lock: threading.Lock,
) -> list[dict]:
    task_id = task["task_id"]
    fp = _task_fingerprint(task)

    existing = records.get(task_id)
    if existing and existing.get("fingerprint") == fp:
        return [{"task_id": task_id, "status": "ok", "_skipped": True}]

    goal = task.get("goal", "")
    all_platforms = list(dict.fromkeys(
        p for scene in _meta(task).get("scene_platforms", []) for p in scene
    ))

    _delete_task_data(task_id, all_platforms, databases_dir)

    platform_counts: dict[str, dict[str, int]] = {}
    failed = False

    for platform in all_platforms:
        if platform not in schemas or not _meta(task).get("task_operations", {}).get(platform):
            continue

        schema_item = schemas[platform]
        db_path = os.path.join(databases_dir, f"{platform.lower().replace(' ', '_').replace('/', '_')}.db")
        schema_ddl = _format_schema_ddl(schema_item)
        platform_ops = _meta(task).get("task_operations", {}).get(platform, [])
        outcome = _meta(task).get("expected_outcome", {}).get(platform, "")
        platform_plan = _get_platform_plan(task_plans, task_id, platform)

        # Step 1: generate core data for all sub-agents at once
        prev_error: str | None = None
        core_tables: list[dict] = []
        tables: list[dict] = []
        mem_ok = False
        no_seed_needed = False

        for attempt in range(1, max_retries + 1):
            result = _generate_platform_data(
                client, model, task_id, goal, platform,
                platform_ops, platform_plan, outcome, schema_item,
                max_retries=1, max_completion_tokens=max_completion_tokens,
                prev_error=prev_error,
            )
            if result is None:
                prev_error = "LLM call/parse failed (no valid response)"
                continue

            # Empty tables is legitimate: a task that is entirely WRITE operations
            # (everything created at runtime) needs no pre-existing seed data.
            if not result:
                no_seed_needed = True
                mem_ok = True
                core_tables = []
                tables = []
                logger.info(f"[{task_id}::{platform}] no seed data needed (all-WRITE task)")
                break

            core_tables = result
            tables = _rows_to_inserts(core_tables, schema_item)
            if not tables:
                gen_names = [t.get("table_name") for t in core_tables]
                schema_names = [t.get("name") or t.get("table_name") for t in schema_item.get("schemas", [])]
                prev_error = f"_rows_to_inserts produced nothing — generated table names {gen_names} do not match schema tables {schema_names}"
                break

            mem_ok, mem_err = _try_inserts_in_memory(schema_ddl, tables)
            if not mem_ok:
                prev_error = mem_err
                continue

            valid, reason = _validate_platform_data(
                client, model, task_id, goal, platform,
                platform_ops, outcome, core_tables, max_completion_tokens,
            )
            if not valid:
                prev_error = f"Data validation failed: {reason}"
                logger.debug(f"[{task_id}::{platform}] validation attempt {attempt}: {reason}")
                continue

            break

        if not mem_ok:
            logger.error(f"[{task_id}::{platform}] core data failed, aborting task: {prev_error}")
            failed = True
            break

        # All-WRITE task: no seed data, so no distractors to insert — platform is done.
        if no_seed_needed:
            platform_counts[platform] = {}
            logger.success(f"[{task_id}::{platform}] done ({len(platform_ops)} sub-agents) | no seed data")
            continue

        # Step 2: generate distractors and insert
        dist_error: str | None = None
        insert_ok = False
        inserted_tables: list[dict] = []

        for _ in range(1, max_retries + 1):
            final_tables = _generate_platform_distractors(
                client, model, task_id, platform_ops, core_tables, schema_item,
                distractor_high,
                max_retries=1, max_completion_tokens=max_completion_tokens,
                prev_error=dist_error,
            )
            all_tables = _rows_to_inserts(final_tables, schema_item)
            ok, err = _execute_inserts(db_path, task_id, all_tables)
            if ok and _validate_row_counts(db_path, task_id, all_tables):
                insert_ok = True
                inserted_tables = all_tables
                break
            if ok:
                err = "insert reported ok but task_id rows missing in DB"
            dist_error = err

        if not insert_ok:
            ok, err = _execute_inserts(db_path, task_id, tables)
            if ok and _validate_row_counts(db_path, task_id, tables):
                insert_ok = True
                inserted_tables = tables

        if not insert_ok:
            logger.error(f"[{task_id}::{platform}] insert failed, aborting task: {err[:200]}")
            failed = True
            break

        counts = _count_task_rows(db_path, task_id, inserted_tables)
        platform_counts[platform] = counts
        counts_str = ", ".join(f"{t}={n}" for t, n in counts.items())
        logger.success(f"[{task_id}::{platform}] done ({len(platform_ops)} sub-agents) | {counts_str}")

    if failed:
        _delete_task_data(task_id, all_platforms, databases_dir)
        return [{"task_id": task_id, "status": "failed"}]

    record = {"task_id": task_id, "fingerprint": fp, "status": "ok", "platforms": platform_counts}
    with write_lock:
        _append_jsonl(data_records_path, record)

    return [record]


# ── Run ────────────────────────────────────────────────────────────────────────

def load_task_plans(path: str) -> dict[str, dict]:
    plans = {t["task_id"]: t for t in _load_jsonl(path) if t.get("task_id")}
    logger.info(f"Loaded {len(plans)} task plans")
    return plans


def run(args: PipelineConfig) -> None:
    tasks = load_tasks(args.tasks_output)
    schemas = load_schemas(args.schemas_output)
    records = load_records(args.data_records)
    task_plans = load_task_plans(args.task_plans_output)

    Path(args.databases_dir).mkdir(parents=True, exist_ok=True)

    client = LLMClient(api_key=args.api_key, base_url=args.base_url, aws_region=args.aws_region)
    write_lock = threading.Lock()

    pending = [t for t in tasks if not (records.get(t["task_id"]) and records[t["task_id"]].get("fingerprint") == _task_fingerprint(t))]
    skipped = len(tasks) - len(pending)
    if skipped:
        logger.info(f"Skipping {skipped} already-completed tasks, {len(pending)} to process.")

    success = failed = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                _process_task,
                client, args.model, task, schemas,
                args.databases_dir, records, task_plans,
                args.distractor_high,
                args.max_retries, args.max_completion_tokens,
                args.data_records, write_lock,
            ): task
            for task in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Seeding data"):
            task_results = future.result()
            for record in task_results:
                if record["status"] == "ok":
                    success += 1
                else:
                    failed += 1

    logger.success(f"Done. {success} generated, {skipped} skipped, {failed} failed.")
    logger.success(f"Records saved to: {args.data_records}")
    logger.success(f"Databases saved to: {args.databases_dir}/")
