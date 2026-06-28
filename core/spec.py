import os
import json
import argparse
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from utils.llm import LLMClient
from tqdm import tqdm

from core.config import PipelineConfig


SYSTEM_PROMPT = """You are an expert API designer. Design atomic REST API endpoints for a platform.

Core rule: every action listed in the input must map to its own dedicated endpoint — one action, one endpoint. No single endpoint may cover more than one listed action. You may add extra endpoints beyond the listed actions (e.g. based on platform features), but you must never merge two listed actions into one endpoint.

Atomicity rules (STRICTLY ENFORCE for every endpoint):
- search / list / browse → returns ONLY [id + at most 3 basic identifying fields]. The result count is a server-side fixed constant (do NOT add max_results or any limit parameter to the endpoint's params — the client cannot control how many results are returned). Agents MUST make a separate detail call to get full information. NEVER include sort_by, order_by, or any result-ordering parameter — result order is always determined by the database insertion order.
- get_detail / get_info → returns all fields for a single record fetched by id
- check / validate / verify → returns {"eligible": bool, "result_value": ...} only — no extra fields
- create / update / delete / book / order / cancel → returns {"id": ..., "status": "..."} only — never the full object
- get_status / confirm → returns current status fields only

These rules apply to ALL platforms. Adapt action names and paths to the platform's domain (e.g. GitHub uses repos/PRs/commits, Airbnb uses listings/bookings).

Return ONLY valid JSON, no markdown fences."""


USER_PROMPT_TEMPLATE = """Platform: {name}

Database Schema (DDL):
{schema_ddl}

Actions performed in tasks, with params and observed return fields (each MUST have its own dedicated endpoint):
{actions_params}

Design REST API endpoints for this platform:
1. Each action above must have exactly one dedicated endpoint — never merge two listed actions into one.
2. You may add extra endpoints based on the platform's features and schema (e.g. a platform with a reviews table should also have a get_review_detail endpoint even if not in the task list).
3. Every endpoint must be atomic — one operation, minimum return fields per the atomicity rules.
4. The listed return fields are the minimum observed from task usage — include them all, and add more fields as appropriate for the endpoint type and schema.

For each endpoint, strictly follow the return field rules:
- search/list/browse → "returns" must contain [id + max 3 basic fields], nothing more; DO NOT add max_results, limit, or any count parameter to "params" — result count is server-side fixed; DO NOT include sort_by, order_by, or any ordering parameter in "params"
- get_detail → "returns" may contain all schema fields for one record
- check/validate → "returns": ["eligible", "result_value"] only
- write (create/update/delete/book/order/cancel) → "returns": ["id", "status"] only
- get_status/confirm → "returns": status-related fields only

Output a JSON array of endpoint objects:
[
  {{
    "action": "action_name_matching_the_task_operations",
    "method": "GET|POST|PUT|DELETE|PATCH",
    "path": "/resource/path",
    "description": "one line: what this endpoint does and when to use it",
    "params": {{
      "param_name": {{
        "type": "str|int|float|bool",
        "required": true
      }}
    }},
    "returns": ["field1", "field2"],
    "tables_read": ["table1"],
    "tables_write": []
  }},
  ...
]"""


FEW_SHOT_EXAMPLES = [
    {
        "name": "Amazon",
        "endpoints": [
            {
                "action": "search_products",
                "method": "GET",
                "path": "/products/search",
                "description": "Search products by keyword and filters; returns basic listing only",
                "params": {
                    "query": {"type": "str", "required": True},
                    "brand": {"type": "str", "required": False},
                    "category": {"type": "str", "required": False},
                    "price_max": {"type": "float", "required": False},
                    "rating_min": {"type": "float", "required": False},
                },
                "returns": ["id", "name", "price"],
                "tables_read": ["products"],
                "tables_write": [],
            },
            {
                "action": "get_product_detail",
                "method": "GET",
                "path": "/products/{product_id}",
                "description": "Get full details of a single product by id",
                "params": {"product_id": {"type": "int", "required": True}},
                "returns": ["id", "name", "brand", "category", "price", "rating", "inventory", "seller_id"],
                "tables_read": ["products"],
                "tables_write": [],
            },
            {
                "action": "check_promo_eligibility",
                "method": "GET",
                "path": "/promos/check",
                "description": "Check if a promo code applies to a product; returns eligibility and discounted price",
                "params": {
                    "product_id": {"type": "int", "required": True},
                    "promo_code": {"type": "str", "required": True},
                },
                "returns": ["eligible", "discounted_price"],
                "tables_read": ["products"],
                "tables_write": [],
            },
            {
                "action": "add_to_cart",
                "method": "POST",
                "path": "/cart/items",
                "description": "Add a product to the cart",
                "params": {
                    "product_id": {"type": "int", "required": True},
                    "quantity": {"type": "int", "required": True},
                },
                "returns": ["id", "status"],
                "tables_read": [],
                "tables_write": ["orders"],
            },
            {
                "action": "get_order_status",
                "method": "GET",
                "path": "/orders/{order_id}/status",
                "description": "Get the current status of an order",
                "params": {"order_id": {"type": "int", "required": True}},
                "returns": ["status", "tracking_number"],
                "tables_read": ["orders"],
                "tables_write": [],
            },
        ],
    },
    {
        "name": "GitHub",
        "endpoints": [
            {
                "action": "list_repos",
                "method": "GET",
                "path": "/repos",
                "description": "List repositories with optional filters; returns basic info only",
                "params": {
                    "language": {"type": "str", "required": False},
                    "visibility": {"type": "str", "required": False},
                },
                "returns": ["id", "name", "visibility"],
                "tables_read": ["repositories"],
                "tables_write": [],
            },
            {
                "action": "get_repo_detail",
                "method": "GET",
                "path": "/repos/{repo_id}",
                "description": "Get full details of a single repository",
                "params": {"repo_id": {"type": "int", "required": True}},
                "returns": ["id", "name", "description", "language", "stars", "visibility", "owner_id"],
                "tables_read": ["repositories"],
                "tables_write": [],
            },
            {
                "action": "create_pull_request",
                "method": "POST",
                "path": "/repos/{repo_id}/pulls",
                "description": "Create a pull request in a repository",
                "params": {
                    "repo_id": {"type": "int", "required": True},
                    "title": {"type": "str", "required": True},
                    "head_branch": {"type": "str", "required": True},
                    "base_branch": {"type": "str", "required": True},
                },
                "returns": ["id", "status"],
                "tables_read": [],
                "tables_write": ["pull_requests"],
            },
            {
                "action": "get_pr_status",
                "method": "GET",
                "path": "/repos/{repo_id}/pulls/{pr_id}/status",
                "description": "Get the current status of a pull request",
                "params": {
                    "repo_id": {"type": "int", "required": True},
                    "pr_id": {"type": "int", "required": True},
                },
                "returns": ["status", "merged", "review_count"],
                "tables_read": ["pull_requests"],
                "tables_write": [],
            },
        ],
    },
]


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
    logger.info(f"Loaded {len(schemas)} platform schemas from {path}")
    return schemas


def extract_action_params(tasks_path: str) -> dict[str, dict[str, dict]]:
    """
    Returns dict[platform_name, dict[action_name, {"params": set, "returns": set}]]
    """
    platform_actions: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"params": set(), "returns": set()})
    )

    if not os.path.exists(tasks_path):
        logger.warning(f"Tasks file not found: {tasks_path}")
        return platform_actions

    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                task = json.loads(line)
                for platform, sub_agents in task.get("metadata", {}).get("task_operations", {}).items():
                    if not isinstance(sub_agents, list):
                        continue
                    for steps in sub_agents:
                        if not isinstance(steps, list):
                            continue
                        for step in steps:
                            if not isinstance(step, dict):
                                continue
                            action = step.get("action", "")
                            if action:
                                for key in step.get("params", {}).keys():
                                    platform_actions[platform][action]["params"].add(key)
                                for key in step.get("returns", {}).keys():
                                    platform_actions[platform][action]["returns"].add(key)
            except json.JSONDecodeError:
                pass

    logger.info(f"Extracted actions for {len(platform_actions)} platforms from {tasks_path}")
    return platform_actions


def load_existing_specs(output_path: str) -> dict[str, dict]:
    # Load already-generated specs keyed by platform name (last entry wins).
    result: dict[str, dict] = {}
    if not os.path.exists(output_path):
        return result
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("name"):
                    result[item["name"]] = item
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(result)} existing specs from {output_path}")
    return result


def append_result(output_path: str, item: dict) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def robust_json_loads(text: str) -> list:
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


def _format_schema_ddl(schema_item: dict) -> str:
    lines = []
    for table in schema_item.get("schemas", []):
        lines.append(table.get("ddl", ""))
    return "\n".join(lines)


def _format_actions_params(action_ops: dict[str, dict]) -> str:
    lines = []
    for action, ops in sorted(action_ops.items()):
        params_str = ", ".join(sorted(ops.get("params", set()))) if ops.get("params") else ""
        returns_str = ", ".join(sorted(ops.get("returns", set()))) if ops.get("returns") else ""
        lines.append(f"  {action}: params=[{params_str}] returns=[{returns_str}]")
    return "\n".join(lines)


def call_llm(
    client: LLMClient,
    model: str,
    platform_name: str,
    schema_item: dict,
    action_params: dict[str, dict],
    max_completion_tokens: int = 8192,
) -> list:
    schema_ddl = _format_schema_ddl(schema_item)
    actions_params_str = _format_actions_params(action_params)

    user_content = USER_PROMPT_TEMPLATE.format(
        name=platform_name,
        schema_ddl=schema_ddl,
        actions_params=actions_params_str,
    )

    # Build few-shot block
    few_shot_lines = []
    for ex in FEW_SHOT_EXAMPLES:
        few_shot_lines.append(
            f"Example for {ex['name']}:\n"
            f"{json.dumps(ex['endpoints'], ensure_ascii=False, indent=2)}"
        )
    few_shot_text = "\n\n".join(few_shot_lines)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{few_shot_text}\n\n---\n\n{user_content}"},
    ]
    return robust_json_loads(client.complete(model, messages, max_completion_tokens))


def process_platform(
    client: LLMClient,
    model: str,
    platform_name: str,
    schema_item: dict,
    action_params: dict[str, dict],
    max_retries: int,
    max_completion_tokens: int = 8192,
) -> dict | None:
    for attempt in range(1, max_retries + 1):
        try:
            endpoints = call_llm(client, model, platform_name, schema_item, action_params, max_completion_tokens)
            result = {
                "name": platform_name,
                "category": schema_item.get("category", ""),
                "subcategory": schema_item.get("subcategory", ""),
                "endpoints": endpoints,
                "ops_fingerprint": schema_item.get("ops_fingerprint", ""),
            }
            logger.success(f"[{platform_name}] Done — {len(endpoints)} endpoints (attempt {attempt})")
            return result
        except Exception as e:
            logger.warning(f"[{platform_name}] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    logger.error(f"[{platform_name}] All {max_retries} attempts failed, skipping.")
    return None


def run(args: PipelineConfig) -> None:
    schemas = load_schemas(args.schemas_output)
    action_params = extract_action_params(args.tasks_output)
    existing = load_existing_specs(args.specs_output)

    # Regenerate if: new platform (not in existing) OR schema's ops_fingerprint changed
    stale: set[str] = set()
    pending: list[str] = []
    for name, schema_item in schemas.items():
        current_fp = schema_item.get("ops_fingerprint", "")
        if name not in existing:
            pending.append(name)
        elif existing[name].get("ops_fingerprint") != current_fp:
            stale.add(name)
            pending.append(name)

    if stale:
        logger.info(f"Re-generating {len(stale)} specs with changed operations: {stale}")
        kept_lines: list[str] = []
        with open(args.specs_output, "r", encoding="utf-8") as f:
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
        with open(args.specs_output, "w", encoding="utf-8") as f:
            for line in kept_lines:
                f.write(line + "\n")

    logger.info(f"Already up-to-date: {len(existing) - len(stale)}, Pending: {len(pending)}")

    if not pending:
        logger.success("All platforms already processed.")
        return

    client = LLMClient.from_config(args)

    success = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                process_platform,
                client, args.model,
                name,
                schemas[name],
                dict(action_params.get(name, {})),
                args.max_retries,
                args.max_completion_tokens,
            ): name
            for name in pending
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating specs"):
            result = future.result()
            if result is not None:
                append_result(args.specs_output, result)
                success += 1

    logger.success(f"Done. {success}/{len(pending)} platforms processed.")
    logger.success(f"Output saved to: {args.specs_output}")


def main() -> None:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Generate API specs for each platform")
    parser.add_argument("--tasks_output", type=str, default=defaults.tasks_output)
    parser.add_argument("--schemas_output", type=str, default=defaults.schemas_output)
    parser.add_argument("--specs_output", type=str, default=defaults.specs_output)
    parser.add_argument("--model", type=str, default=defaults.model)
    parser.add_argument("--api_key", type=str, default=defaults.api_key)
    parser.add_argument("--base_url", type=str, default=defaults.base_url)
    parser.add_argument("--concurrency", type=int, default=defaults.concurrency)
    parser.add_argument("--max_retries", type=int, default=defaults.max_retries)
    parser.add_argument("--max_results", type=int, default=defaults.max_results)
    parsed = parser.parse_args()

    cfg = PipelineConfig()
    for k, v in vars(parsed).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    run(cfg)


if __name__ == "__main__":
    main()
