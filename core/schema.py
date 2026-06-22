from __future__ import annotations
import os
import json
import hashlib
import argparse
import sqlite3
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from utils.llm import LLMClient
from tqdm import tqdm

from core.config import PipelineConfig


SYSTEM_PROMPT = """You are a database schema designer. Given a platform's description, features, and the actual API operations tasks perform on it, generate SQLite CREATE TABLE statements.

Design principle:
- Use the platform's features as the primary guide for what tables and columns to create — the schema should reflect the full data model of the platform, not just the minimum needed for the listed operations.
- The action params and returns are constraints: every param key AND every return key listed must appear as a column somewhere in the schema. But go beyond them — add columns a real platform of this type would need.

Rules:
- Every table MUST include `task_id TEXT NOT NULL` as the first column
- Every table MUST include `id INTEGER` as the second column (a generic row identifier, no PRIMARY KEY) — this column is required in EVERY table without exception
- In addition to the generic `id`, also include the table's semantic identifier column (e.g. `user_id`, `payment_method_id`, `order_id`) when the operations reference it
- Do NOT add PRIMARY KEY or FOREIGN KEY constraints — omit them entirely
- Use realistic column names and types
- NEVER use SQLite reserved keywords as column or table names — forbidden names include: values, references, index, order, group, select, where, table, column, key, check, default, from, to, into, set, by, on, as, is, in, not, and, or, null, primary, unique, foreign, references, constraint, transaction, commit, rollback, create, drop, alter, insert, update, delete, view, trigger
- If a param/return key is a reserved keyword, append an underscore (e.g. `values` → `values_`, `references` → `ref_id`)
- Return ONLY valid JSON, no markdown fences

The output should be a JSON array of schema objects, each with:
- "table": table name
- "ddl": the full CREATE TABLE statement"""


USER_PROMPT_TEMPLATE = """Platform: {name}
Category: {category} / {subcategory}

Description:
{description}

Features:
{features}

API actions tasks actually call on this platform (format: action: params=[...] returns=[...]):
{actions}

Generate SQLite schemas that:
1. Reflect the platform's full data model based on its description and features
2. Ensure every param AND every return key listed under each action appears as a column in the appropriate table

Output format (valid JSON array):
[
  {{
    "table": "table_name",
    "ddl": "CREATE TABLE table_name (task_id TEXT NOT NULL, id INTEGER, ...);"
  }},
  ...
]"""


FEW_SHOT_EXAMPLES = [
    {
        "name": "Amazon",
        "schemas": [
            {
                "table": "products",
                "ddl": "CREATE TABLE products (task_id TEXT NOT NULL, id INTEGER, name TEXT, brand TEXT, category TEXT, price REAL, price_max REAL, condition TEXT, inventory INTEGER, rating REAL, rating_min REAL, seller_id INTEGER, promo_code TEXT);"
            },
            {
                "table": "sellers",
                "ddl": "CREATE TABLE sellers (task_id TEXT NOT NULL, id INTEGER, name TEXT, rating REAL, location TEXT, total_sales INTEGER);"
            },
            {
                "table": "orders",
                "ddl": "CREATE TABLE orders (task_id TEXT NOT NULL, id INTEGER, product_id INTEGER, user_id INTEGER, quantity INTEGER, status TEXT, created_at TEXT, tracking_number TEXT);"
            },
            {
                "table": "reviews",
                "ddl": "CREATE TABLE reviews (task_id TEXT NOT NULL, id INTEGER, product_id INTEGER, user_id INTEGER, score INTEGER, content TEXT, created_at TEXT);"
            },
            {
                "table": "wishlists",
                "ddl": "CREATE TABLE wishlists (task_id TEXT NOT NULL, id INTEGER, user_id INTEGER, product_id INTEGER, added_at TEXT);"
            },
            {
                "table": "addresses",
                "ddl": "CREATE TABLE addresses (task_id TEXT NOT NULL, id INTEGER, user_id INTEGER, street TEXT, city TEXT, state TEXT, zip TEXT, is_default INTEGER);"
            },
            {
                "table": "payment_methods",
                "ddl": "CREATE TABLE payment_methods (task_id TEXT NOT NULL, id INTEGER, user_id INTEGER, type TEXT, last_four TEXT, expiry TEXT, is_default INTEGER);"
            }
        ]
    }
]


def load_platforms(path: str) -> list[dict]:
    platforms = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    platforms.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    logger.info(f"Loaded {len(platforms)} platforms from {path}")
    return platforms


def extract_platform_operations(tasks_path: str) -> dict[str, dict]:
    # Scan tasks.jsonl and collect, per platform, each action and the set of param/return keys it uses.
    # Stored as {platform: {"action_params": {action: {param_key,...}}, "action_returns": {action: {return_key,...}}}}.
    platform_ops: dict[str, dict] = defaultdict(lambda: {"action_params": defaultdict(set), "action_returns": defaultdict(set)})

    if not os.path.exists(tasks_path):
        logger.warning(f"Tasks file not found: {tasks_path}, skipping operation extraction")
        return platform_ops

    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                task_ops = task.get("metadata", {}).get("task_operations", {})
                for platform, sub_agents in task_ops.items():
                    if not isinstance(sub_agents, list):
                        continue
                    for steps in sub_agents:
                        if not isinstance(steps, list):
                            continue
                        for step in steps:
                            if not isinstance(step, dict):
                                continue
                            action = step.get("action", "")
                            if not action:
                                continue
                            for key in step.get("params", {}).keys():
                                platform_ops[platform]["action_params"][action].add(key)
                            for key in step.get("returns", {}).keys():
                                platform_ops[platform]["action_returns"][action].add(key)
            except json.JSONDecodeError:
                pass

    logger.info(f"Extracted operations for {len(platform_ops)} platforms from {tasks_path}")
    return platform_ops


def compute_ops_fingerprint(ops: dict) -> str:
    # MD5 of sorted action_params + action_returns; detects when new tasks introduce new operations.
    action_params = {k: sorted(v) for k, v in ops.get("action_params", {}).items()}
    action_returns = {k: sorted(v) for k, v in ops.get("action_returns", {}).items()}
    content = json.dumps({"action_params": action_params, "action_returns": action_returns}, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def load_existing_schemas(output_path: str) -> dict[str, dict]:
    # Load already-generated schemas keyed by platform name (last entry wins on duplicates).
    result: dict[str, dict] = {}
    if not os.path.exists(output_path):
        return result
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    if item.get("name"):
                        result[item["name"]] = item
                except json.JSONDecodeError:
                    pass
    logger.info(f"Loaded {len(result)} existing schemas from {output_path}")
    return result


def append_result(output_path: str, item: dict):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def robust_json_loads(text: str) -> list:
    # Strip markdown fences the LLM sometimes wraps around JSON output.
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise


def call_llm(client: LLMClient, model: str, platform: dict, ops: dict, max_completion_tokens: int = 8192, prev_error: str | None = None) -> str:
    # Build prompt from platform info + actual task operations, then call LLM to get CREATE TABLE statements.
    features_text = "\n".join(f"- {f}" for f in platform.get("features", []))
    action_params = ops.get("action_params", {})
    action_returns = ops.get("action_returns", {})
    if action_params:
        actions_text = "\n".join(
            f"- {action}: params=[{', '.join(sorted(params)) if params else ''}] returns=[{', '.join(sorted(action_returns.get(action, set())))}]"
            for action, params in sorted(action_params.items())
        )
    else:
        actions_text = "- (none)"

    user_content = USER_PROMPT_TEMPLATE.format(
        name=platform["name"],
        category=platform.get("category", ""),
        subcategory=platform.get("subcategory", ""),
        description=platform.get("description", ""),
        features=features_text,
        actions=actions_text,
    )

    example = FEW_SHOT_EXAMPLES[0]
    example_text = f"Example for {example['name']}:\n{json.dumps(example['schemas'], ensure_ascii=False, indent=2)}"

    if prev_error:
        user_content += f"\n\nPrevious attempt failed with SQLite error — fix the DDL:\n{prev_error}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{example_text}\n\n---\n\n{user_content}"},
    ]
    return client.complete(model, messages, max_completion_tokens)


def _create_db(db_path: str, schemas: list) -> tuple[bool, str]:
    """Create actual platform db and tables from DDL. Deletes db on failure. Returns (ok, error_message)."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys = OFF")
            for item in schemas:
                ddl = item.get("ddl", "").strip()
                if not ddl:
                    continue
                safe = ddl.replace("CREATE TABLE ", "CREATE TABLE IF NOT EXISTS ")
                for stmt in (s.strip() for s in safe.split(";") if s.strip()):
                    conn.execute(stmt)
            conn.commit()
            return True, ""
        except sqlite3.Error as e:
            conn.close()
            try:
                os.remove(db_path)
            except OSError:
                pass
            return False, str(e)
        finally:
            try:
                conn.close()
            except Exception:
                pass
    except Exception as e:
        try:
            os.remove(db_path)
        except OSError:
            pass
        return False, str(e)


def process_platform(client: LLMClient, model: str, platform: dict, ops: dict, databases_dir: str, max_retries: int, max_completion_tokens: int = 8192) -> dict | None:
    # Generate schema for one platform with retry. Creates actual db before writing to jsonl.
    name = platform["name"]
    safe_name = name.lower().replace(" ", "_").replace("/", "_")
    db_path = os.path.join(databases_dir, f"{safe_name}.db")
    prev_error: str | None = None

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_llm(client, model, platform, ops, max_completion_tokens, prev_error)
            schemas = robust_json_loads(raw)

            ok, err = _create_db(db_path, schemas)
            if not ok:
                logger.warning(f"[{name}] DB creation failed (attempt {attempt}): {err}")
                prev_error = err
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue

            result = {
                "name": name,
                "category": platform.get("category", ""),
                "subcategory": platform.get("subcategory", ""),
                "schemas": schemas,
                "ops_fingerprint": compute_ops_fingerprint(ops),
            }
            logger.success(f"[{name}] Done — {len(schemas)} tables, db created (attempt {attempt})")
            return result

        except Exception as e:
            logger.warning(f"[{name}] Attempt {attempt}/{max_retries} failed: {e}")
            prev_error = str(e)
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    logger.error(f"[{name}] All {max_retries} attempts failed, skipping.")
    return None


def run(args: PipelineConfig):
    platforms = load_platforms(args.scenario_output)
    platform_ops = extract_platform_operations(args.tasks_output)
    existing = load_existing_schemas(args.schemas_output)

    # Determine which platforms need (re)generation:
    # - not yet in existing → generate for the first time
    # - fingerprint changed → new tasks introduced new actions/params, regenerate
    # - fingerprint matches → skip
    Path(args.databases_dir).mkdir(parents=True, exist_ok=True)

    stale: set[str] = set()
    pending: list[dict] = []
    for p in platforms:
        name = p["name"]
        if name not in platform_ops:
            continue
        ops = platform_ops[name]
        current_fp = compute_ops_fingerprint(ops)
        if name not in existing:
            pending.append(p)
        elif existing[name].get("ops_fingerprint") != current_fp:
            stale.add(name)
            pending.append(p)
        else:
            # fingerprint matches — ensure db exists (one-time recovery for schemas generated before db creation was added)
            db_path = os.path.join(args.databases_dir, f"{name.lower().replace(' ', '_').replace('/', '_')}.db")
            if not os.path.exists(db_path):
                ok, err = _create_db(db_path, existing[name].get("schemas", []))
                if not ok:
                    logger.warning(f"[{name}] Could not create db from existing schema ({err[:100]}), regenerating")
                    stale.add(name)
                    pending.append(p)

    if stale:
        # Remove stale entries from schemas file and delete their db files.
        logger.info(f"Re-generating {len(stale)} platforms: {stale}")
        kept_lines: list[str] = []
        with open(args.schemas_output, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    item = json.loads(stripped)
                    if item.get("name") not in stale:
                        kept_lines.append(stripped)
                except json.JSONDecodeError:
                    kept_lines.append(stripped)
        with open(args.schemas_output, "w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")
        for name in stale:
            safe_name = name.lower().replace(" ", "_").replace("/", "_")
            db_path = os.path.join(args.databases_dir, f"{safe_name}.db")
            if os.path.exists(db_path):
                os.remove(db_path)
                logger.debug(f"Deleted stale db: {db_path}")

    logger.info(f"Already up-to-date: {len(existing) - len(stale)}, Pending: {len(pending)}")

    if not pending:
        logger.success("All platforms already processed.")
        return

    client = LLMClient(api_key=args.api_key, base_url=args.base_url, aws_region=args.aws_region)

    Path(args.databases_dir).mkdir(parents=True, exist_ok=True)

    success = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                process_platform, client, args.model, p,
                platform_ops.get(p["name"], {"actions": set(), "params": set()}),
                args.databases_dir, args.max_retries, args.max_completion_tokens,
            ): p
            for p in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating schemas"):
            result = future.result()
            if result is not None:
                append_result(args.schemas_output, result)
                success += 1

    logger.success(f"Done. {success}/{len(pending)} platforms processed.")
    logger.success(f"Output saved to: {args.schemas_output}")


def main():
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Generate SQLite schemas for each platform")
    parser.add_argument("--scenario_output", type=str, default=defaults.scenario_output)
    parser.add_argument("--tasks_output", type=str, default=defaults.tasks_output)
    parser.add_argument("--schemas_output", type=str, default=defaults.schemas_output)
    parser.add_argument("--model", type=str, default=defaults.model)
    parser.add_argument("--api_key", type=str, default=defaults.api_key)
    parser.add_argument("--base_url", type=str, default=defaults.base_url)
    parser.add_argument("--concurrency", type=int, default=defaults.concurrency)
    parser.add_argument("--max_retries", type=int, default=defaults.max_retries)
    parsed = parser.parse_args()

    cfg = PipelineConfig()
    for k, v in vars(parsed).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    run(cfg)


if __name__ == "__main__":
    main()
