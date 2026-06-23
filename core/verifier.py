"""
⚠️ DEPRECATED — NOT USED. Do not import or run this module.

This is the OLD monolithic verifier. It was split into three modules, which are
the live pipeline (see main.py):
    core/verifier_gen.py   — generate verifier functions
    core/verifier_step.py  — simulate agents + run verifiers + collect suggestions  (ACTIVE)
    core/verifier_fix.py   — apply fixes from suggestions

Nothing imports core.verifier anymore (grep: 0 references).

HISTORY / KNOWN PITFALL: the agent-simulation prompt here gave the solver only the
action names (see `_build_sim_agent_prompt`, action-only) so it had to discover
IDs/SKUs itself. That stripping was NOT carried over when the logic moved to the
active verifier_step.py, which instead handed the solver the full task_operations
including `params` (leaks IDs/SKUs) and `returns` (leaks the expected answer). That
leak let unsolvable tasks pass verification (e.g. a platform whose search never
exposes the SKU its detail endpoint requires). The action-only behavior has since
been restored in verifier_step.py (`_ops_actions_only`). Kept only for reference.

──────────────────────────────────────────────────────────────────────────────
Generate verification functions for each task×platform and validate with agent simulation.

Flow per task×platform:
  1. Generate verifier: plan + outcome + task_ops + schema + seed data (write-ops only)
  2. Sanity check: seed_db as both initial and final (no-write path sanity)
  3. Start real platform server; LLM makes actual API calls guided by task_ops + prompt_data
  4. Run verifier on initial_db vs sim_db — must return "complete"
  5. On failure after max_retries: ask LLM what data was missing → save prompt_data2 to task_repairs.jsonl
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from loguru import logger
from utils.llm import LLMClient
from tqdm import tqdm

from core.config import PipelineConfig
from utils.server import (
    get_free_port as _get_free_port,
    wait_for_server as _wait_for_server,
    start_server as _start_server,
    stop_server as _stop_server,
)
from utils.mcp import (
    MCPToolExecutor as _MCPToolExecutor,
    run_sub_agent_loop as _run_sub_agent_loop,
    inject_mcp as _inject_mcp,
)


# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert Python developer specializing in database verification for RL training. Generate a deterministic Python function that compares initial and final SQLite database states to verify if an AI agent completed a task correctly.

Requirements:
1. Signature: def verify_task_completion(initial_db_path: str, final_db_path: str, task_id: str) -> dict
2. ONLY verify WRITE operations — rows added (INSERT) or rows changed (UPDATE) in final_db that differ from initial_db. Do NOT check that any GET/search/read endpoint returned specific values.
3. Filter ALL queries by task_id using WHERE task_id = ?
4. Compare initial_db (before agent) vs final_db (after agent) to detect what changed
5. Return {"result": "complete"} ONLY when 100% certain the correct write(s) exist in final_db with correct field values
6. Return {"result": "others"} in ALL other cases (no write, wrong write, error, partial)
7. If the platform plan shows only read operations are needed (no writes), return {"result": "complete"} immediately
8. Import all libraries inside the function body
9. Use try/except around all DB operations — never raise exceptions
10. Be strict: if the write references the wrong record (wrong price, wrong rating, wrong category, etc.), return "others"

Output format (valid JSON, no markdown fences):
{
  "reasoning": "what specific write checks verify task completion",
  "no_op_result": "others",
  "python_code": "complete function code as a string",
  "function_name": "verify_task_completion"
}

no_op_result: what your function returns when the agent does nothing (initial_db == final_db).
- "others": the task requires writes, so doing nothing means incomplete
- "complete": the task requires no writes (read-only platform, or conditional_branch source where agent correctly found nothing)"""


USER_PROMPT_TEMPLATE = """Generate a verification function for the following task on platform: {platform_name}

Task ID: {task_id}
Overall Goal: {goal}
Enriched Goal (with agent upfront info): {new_goal}
Expected Outcome on {platform_name}: {expected_outcome}

Platform Plan (what each sub-agent should accomplish):
{platform_plan}

Agent Upfront Information (what the agent knows before querying — do NOT verify that these were obtained via reads):
{prompt_data}

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
- Return {{"result": "complete"}} only when the correct write(s) exist with correct field values
- Do NOT verify read operations (searches, lookups, GET endpoints)

Output format (valid JSON, no markdown fences):
{{
  "reasoning": "what write checks verify this specific task",
  "no_op_result": "others or complete — what your function returns when agent does nothing",
  "python_code": "complete Python function code",
  "function_name": "verify_task_completion"
}}"""


CONDITIONAL_NOTE = """IMPORTANT: This platform is the SOURCE of a conditional_branch scene transition.
The agent searched this platform but found NO matching results (only distractor data was seeded).
The agent should NOT have made any write operations on this platform.
Verify: final_db has NO new records in write tables for task_id = '{task_id}' compared to initial_db.
If no incorrect writes were made → return {{"result": "complete"}}."""


SIM_AGENT_SYSTEM_PROMPT = """You are an AI agent operating a platform API via MCP tools to complete a specific sub-task.

You have two meta-tools:
  list_tools   — list all available MCP tools (call once at the start)
  call_tool    — call a specific MCP tool
                 arguments: {"tool_name": "<name>", "arguments": "<json-string>"}

Output each tool call as:
<tool_call>
{"name": "list_tools", "arguments": null}
</tool_call>

<tool_call>
{"name": "call_tool", "arguments": {"tool_name": "search_products", "arguments": "{\"query\": \"laptop\"}"}}
</tool_call>

Use the provided task operations as your step-by-step guide — follow their order and use the exact param values given. When all steps are done, write a final summary (no tool call tags) to signal completion."""


MISSING_DATA_SYSTEM = """You are analyzing why an AI agent failed to complete a platform task.
Given the task operations, what the agent was told upfront, and the agent's execution trajectory, identify what additional data should have been provided to the agent upfront in the prompt to allow it to complete the task.

Output JSON only (no markdown fences):
{"extra_prompt_data": {"descriptive_key": "value_or_description", ...}}

Rules:
- Only include data that was genuinely missing AND needed to complete the task operations
- Do not include data that the agent could obtain by querying the platform API
- If nothing was missing, output: {"extra_prompt_data": {}}"""


MISSING_DATA_USER_TEMPLATE = """Task on {platform}:
Goal: {goal}

Platform Plan (what the agent should accomplish):
{platform_plan}

Task Operations (step-by-step guide the agent should follow):
{task_ops}

Prompt Data already provided upfront to the agent:
{prompt_data}

Agent Execution Trajectory (what actually happened, last steps):
{trajectory_summary}

Verification Errors (why verification failed):
{errors}

What additional data should have been in the prompt to allow the agent to complete all task operations?"""


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


def load_done(output_path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("status") == "ok":
                    done.add(f"{item['task_id']}::{item['platform']}")
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(done)} completed verifiers from {output_path}")
    return done


def append_result(output_path: str, entry: dict) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_task_repairs(path: str) -> dict[tuple[str, str], dict]:
    """Load task_repairs.jsonl keyed by (task_id, platform). Multiple entries merged."""
    repairs: dict[tuple[str, str], dict] = {}
    if not os.path.exists(path):
        return repairs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                tid = item.get("task_id", "")
                plat = item.get("platform", "")
                if not tid or not plat:
                    continue
                key = (tid, plat)
                if key in repairs:
                    # merge: later entries update earlier ones
                    repairs[key].update({k: v for k, v in item.items() if v is not None})
                else:
                    repairs[key] = dict(item)
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(repairs)} task repairs from {path}")
    return repairs


def load_task_plans(path: str) -> dict[str, dict]:
    """Load task_plans.jsonl keyed by task_id."""
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


def load_envs(path: str) -> dict[str, dict]:
    """Load envs.jsonl keyed by platform name."""
    envs: dict[str, dict] = {}
    if not os.path.exists(path):
        return envs
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                name = item.get("name", "")
                if name:
                    envs[name] = item
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(envs)} envs from {path}")
    return envs


def _update_repair_prompt_data2(repairs_path: str, task_id: str, platform: str, prompt_data2: dict) -> None:
    """Update task_repairs.jsonl: add/merge prompt_data2 for (task_id, platform)."""
    if not prompt_data2:
        return

    entries: list[dict] = []
    found = False
    if os.path.exists(repairs_path):
        with open(repairs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if item.get("task_id") == task_id and item.get("platform") == platform:
                        existing = item.get("prompt_data2", {})
                        existing.update(prompt_data2)
                        item["prompt_data2"] = existing
                        found = True
                    entries.append(item)
                except json.JSONDecodeError:
                    pass

    if not found:
        entries.append({"task_id": task_id, "platform": platform, "prompt_data2": prompt_data2})

    tmp = repairs_path + ".tmp"
    Path(tmp).parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp, repairs_path)
    logger.debug(f"Updated prompt_data2 for {task_id}::{platform}: {list(prompt_data2.keys())}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_no_correct_data_platforms(task: dict) -> set[str]:
    no_correct: set[str] = set()
    scene_platforms = task.get("scene_platforms", [])
    for transition in task.get("scene_transitions", []):
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
    ops = task.get("task_operations", {}).get(platform, [])
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


# ── LLM call (verifier generation) ────────────────────────────────────────────

def call_llm(
    client: LLMClient,
    model: str,
    task: dict,
    platform: str,
    schema_item: dict,
    db_path: str,
    repair: dict,
    platform_plan: dict,
    no_correct_data: bool,
    error_history: list[str],
    max_completion_tokens: int = 8192,
) -> dict:
    task_id = task["task_id"]
    schema_ddl = _format_schema_ddl(schema_item)
    task_operations_str = _format_task_operations(task, platform)
    expected_outcome = task.get("expected_outcome", {}).get(platform, "")
    db_dump = _dump_task_data(db_path, task_id, schema_item)
    new_goal = repair.get("new_goal") or task.get("goal", "")
    prompt_data = repair.get("prompt_data", {})

    conditional_note = CONDITIONAL_NOTE.format(task_id=task_id) if no_correct_data else ""

    user_content = USER_PROMPT_TEMPLATE.format(
        platform_name=platform,
        task_id=task_id,
        goal=task.get("goal", ""),
        new_goal=new_goal,
        expected_outcome=expected_outcome,
        platform_plan=json.dumps(platform_plan, ensure_ascii=False, indent=2) if platform_plan else "(no plan available)",
        prompt_data=json.dumps(prompt_data, ensure_ascii=False, indent=2) if prompt_data else "(none)",
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
    no_op_result: str,
    task_id: str = "TEST_TASK_ID",
) -> dict:
    """Run the generated function with seed DB as both initial and final (agent did nothing).

    no_op_result: what the LLM declared this verifier should return for a no-op agent.
    - "others"  → task requires writes; doing nothing must not pass
    - "complete" → task is read-only or no-write; doing nothing is correct
    """
    if not os.path.exists(db_path):
        return {"execution_status": "error", "error_message": f"Database not found: {db_path}"}

    expected_result = no_op_result

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

        if not isinstance(result, dict) or "result" not in result:
            return {"execution_status": "error", "error_message": f"Invalid return format: {type(result).__name__}"}

        if result["result"] not in ("complete", "others"):
            return {"execution_status": "error", "error_message": f"Invalid result value: {result['result']}"}

        if result["result"] != expected_result:
            return {
                "execution_status": "error",
                "error_message": (
                    f"Sanity check failed: agent no-op returned '{result['result']}', "
                    f"expected '{expected_result}' (declared by LLM as no_op_result)"
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


# ── Agent simulation ───────────────────────────────────────────────────────────

def _build_sim_agent_prompt(
    new_goal: str,
    platform: str,
    platform_desc: str,
    sub_ops: list,
    sub_agent_idx: int,
    total_sub_agents: int,
    prompt_data: dict,
    prompt_data2: dict,
    prior_results: list,
) -> str:
    # Give the agent only the action sequence — NOT params/returns. Params often embed
    # values that are runtime-discovered (created IDs, search results) or stated in the
    # goal, so handing them over would leak the expected answer and mask server bugs.
    # The agent must derive each param itself from the goal and prior API responses.
    action_steps = [op.get("action", "?") for op in sub_ops]
    ops_str = json.dumps(action_steps, ensure_ascii=False, indent=2)

    upfront_parts = []
    if prompt_data:
        upfront_parts.append(
            "Upfront information:\n" + json.dumps(prompt_data, ensure_ascii=False, indent=2)
        )
    if prompt_data2:
        upfront_parts.append(
            "Additional information:\n" + json.dumps(prompt_data2, ensure_ascii=False, indent=2)
        )
    upfront_str = ("\n\n" + "\n\n".join(upfront_parts)) if upfront_parts else ""

    prior_str = ""
    if prior_results:
        recent = prior_results[-10:]
        prior_str = (
            "\n\nContext from previous sub-agents "
            "(values you may need as params):\n"
            + json.dumps(recent, ensure_ascii=False, indent=2)
        )

    return (
        f"Overall goal: {new_goal}\n\n"
        f"Platform: {platform}\n"
        f"Platform description: {platform_desc}\n\n"
        f"You are sub-agent {sub_agent_idx + 1}/{total_sub_agents} on {platform}.\n\n"
        f"Action sequence — perform these actions in order. You must determine each call's "
        f"params yourself from the overall goal, the upfront information, and the actual "
        f"responses of your earlier API calls (e.g. search results, created IDs):\n"
        f"{ops_str}"
        f"{upfront_str}"
        f"{prior_str}"
    )


async def _simulate_platform_async(
    server_path: str,
    sim_db: str,
    task_id: str,
    task_ops: list,
    new_goal: str,
    prompt_data: dict,
    prompt_data2: dict,
    platform: str,
    platform_desc: str,
    model: str,
    max_iterations: int,
    tmpdir: str,
    api_key: Optional[str],
    base_url: Optional[str],
) -> list[dict]:
    """Start a real server with sim_db, run all sub-agents via MCP, return trajectory."""
    from openai import AsyncOpenAI

    # Inject MCP and write to temp server file
    with open(server_path, "r", encoding="utf-8") as f:
        orig = f.read()
    mcp_code = _inject_mcp(orig, platform)
    safe = platform.lower().replace(" ", "_").replace("/", "_")
    mcp_server_path = os.path.join(tmpdir, f"{safe}_mcp.py")
    with open(mcp_server_path, "w", encoding="utf-8") as f:
        f.write(mcp_code)

    proc, port = _start_server(mcp_server_path, sim_db)
    if not _wait_for_server(port, timeout=25):
        _stop_server(proc)
        raise RuntimeError(f"Server for {platform} failed to start on port {port}")

    openai_client = AsyncOpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=base_url or None,
    )
    mcp_url = f"http://127.0.0.1:{port}/mcp"

    full_trajectory: list[dict] = []
    prior_results: list[dict] = []

    try:
        for ki, sub_ops in enumerate(task_ops):
            if not isinstance(sub_ops, list):
                continue

            prompt = _build_sim_agent_prompt(
                new_goal, platform, platform_desc, sub_ops,
                ki, len(task_ops),
                prompt_data, prompt_data2, prior_results,
            )
            mcp = _MCPToolExecutor(mcp_url)
            sub_traj = await _run_sub_agent_loop(mcp, openai_client, model, prompt, max_iterations)

            for step in sub_traj:
                resp = step.get("tool_response", "")
                if resp and not str(resp).startswith("Error"):
                    try:
                        parsed = json.loads(resp)
                        prior_results.append({"sub_agent": ki, "result": parsed})
                    except Exception:
                        pass

            full_trajectory.append({
                "platform": platform, "sub_agent": ki, "steps": sub_traj,
            })
    finally:
        _stop_server(proc)

    return full_trajectory


def _simulate_platform(
    server_path: str,
    sim_db: str,
    task_id: str,
    task_ops: list,
    new_goal: str,
    prompt_data: dict,
    prompt_data2: dict,
    platform: str,
    platform_desc: str,
    model: str,
    max_iterations: int,
    tmpdir: str,
    api_key: Optional[str],
    base_url: Optional[str],
) -> list[dict]:
    """Sync wrapper for _simulate_platform_async."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _simulate_platform_async(
                server_path, sim_db, task_id, task_ops,
                new_goal, prompt_data, prompt_data2,
                platform, platform_desc, model, max_iterations,
                tmpdir, api_key, base_url,
            )
        )
    finally:
        loop.close()


def _run_verifier_on_sim(
    python_code: str,
    function_name: str,
    initial_db: str,
    final_db: str,
    task_id: str,
) -> dict:
    """Run verifier on actual before/after DB state from simulation."""
    namespace: dict = {
        "sqlite3": sqlite3, "json": json, "os": os,
        "__builtins__": __builtins__,
    }
    try:
        exec(python_code, namespace)
        fn = namespace.get(function_name)
        if not fn:
            return {"result": "others", "error": f"Function '{function_name}' not found"}
        result = fn(initial_db, final_db, task_id)
        if not isinstance(result, dict) or "result" not in result:
            return {"result": "others", "error": f"Invalid return format: {result}"}
        if result["result"] not in ("complete", "others"):
            return {"result": "others", "error": f"Invalid result value: {result['result']}"}
        return result
    except Exception as e:
        return {"result": "others", "error": str(e)}


# ── Missing data analysis ──────────────────────────────────────────────────────

def _ask_missing_data(
    client: LLMClient,
    model: str,
    goal: str,
    platform_plan: dict,
    task_ops: list,
    prompt_data: dict,
    trajectory: list[dict],
    errors: str,
    max_completion_tokens: int,
) -> dict:
    """Ask LLM what data was missing from prompt to allow task completion."""
    # Summarize trajectory to last few tool calls per sub-agent
    traj_steps = []
    for entry in trajectory:
        for step in entry.get("steps", [])[-5:]:
            if step.get("tool_call"):
                traj_steps.append({
                    "sub_agent": entry.get("sub_agent"),
                    "call": step["tool_call"],
                    "response": str(step.get("tool_response", ""))[:300],
                })

    user_content = MISSING_DATA_USER_TEMPLATE.format(
        platform=platform_plan.get("role", "unknown"),
        goal=goal,
        platform_plan=json.dumps(platform_plan, ensure_ascii=False, indent=2) if platform_plan else "(none)",
        task_ops=json.dumps(task_ops, ensure_ascii=False, indent=2),
        prompt_data=json.dumps(prompt_data, ensure_ascii=False, indent=2) if prompt_data else "(none)",
        trajectory_summary=json.dumps(traj_steps, ensure_ascii=False, indent=2),
        errors=errors,
    )

    messages = [
        {"role": "system", "content": MISSING_DATA_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        raw = client.complete(model, messages, max_completion_tokens)
        result = _robust_json_loads(raw)
        return result.get("extra_prompt_data", {})
    except Exception as e:
        logger.warning(f"_ask_missing_data failed: {e}")
        return {}


# ── Per task×platform processing ───────────────────────────────────────────────

def process_task_platform(
    client: LLMClient,
    model: str,
    gen_model: str,
    task: dict,
    platform: str,
    schema_item: dict,
    db_path: str,
    repair: dict,
    platform_plan: dict,
    server_path: str,
    platform_desc: str,
    no_correct_data: bool,
    max_retries: int,
    max_completion_tokens: int,
    max_iterations: int,
    repairs_path: str,
    api_key: Optional[str],
    base_url: Optional[str],
) -> dict:
    task_id = task["task_id"]
    error_history: list[str] = []
    python_code = ""
    function_name = "verify_task_completion"
    last_trajectory: list[dict] = []

    for attempt in range(1, max_retries + 1):
        # Step 1: Generate verifier
        try:
            result = call_llm(
                client, model, task, platform, schema_item, db_path,
                repair, platform_plan, no_correct_data, error_history, max_completion_tokens,
            )
        except Exception as e:
            logger.warning(f"[{task_id}::{platform}] LLM call failed (attempt {attempt}): {e}")
            error_history.append(f"LLM error: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            continue

        python_code = result.get("python_code", "")
        function_name = result.get("function_name", "verify_task_completion")
        no_op_result = result.get("no_op_result", "others")
        if no_op_result not in ("complete", "others"):
            no_op_result = "others"

        if not python_code:
            error_history.append("Empty python_code from LLM")
            continue

        # Step 2: Sanity check — seed_db as both initial and final
        # no_op_result declared by LLM: "others" if writes required, "complete" if read-only/no-write
        exec_result = execute_verification_code(python_code, function_name, db_path, no_op_result, task_id)
        if exec_result["execution_status"] != "success":
            err = exec_result.get("error_message", "sanity check failed")
            logger.warning(f"[{task_id}::{platform}] Sanity check failed (attempt {attempt}): {err}")
            error_history.append(f"Sanity check: {err}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            continue

        # Step 3: Agent simulation — skip if no server available
        if not server_path or not os.path.exists(server_path):
            logger.warning(f"[{task_id}::{platform}] No server file, skipping simulation")
            logger.success(f"[{task_id}::{platform}] Done (sanity only, attempt {attempt})")
            return {
                "task_id": task_id, "platform": platform,
                "verify_fn": python_code, "function_name": function_name, "status": "ok",
            }

        task_ops = task.get("task_operations", {}).get(platform, [])
        if not task_ops:
            logger.success(f"[{task_id}::{platform}] Done (no task ops, attempt {attempt})")
            return {
                "task_id": task_id, "platform": platform,
                "verify_fn": python_code, "function_name": function_name, "status": "ok",
            }

        new_goal = repair.get("new_goal") or task.get("goal", "")
        prompt_data = repair.get("prompt_data", {})
        prompt_data2 = repair.get("prompt_data2", {})

        with tempfile.TemporaryDirectory() as tmpdir:
            sim_db = os.path.join(tmpdir, "sim.db")
            shutil.copy2(db_path, sim_db)

            try:
                trajectory = _simulate_platform(
                    server_path, sim_db, task_id,
                    task_ops, new_goal, prompt_data, prompt_data2,
                    platform, platform_desc, gen_model, max_iterations,
                    tmpdir, api_key, base_url,
                )
                last_trajectory = trajectory
            except Exception as e:
                logger.warning(f"[{task_id}::{platform}] Simulation error (attempt {attempt}): {e}")
                error_history.append(f"Simulation: {e}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                continue

            # Step 4: Run verifier on real before/after state
            sim_result = _run_verifier_on_sim(python_code, function_name, db_path, sim_db, task_id)

            if sim_result.get("result") == "complete":
                logger.success(f"[{task_id}::{platform}] Verified (attempt {attempt})")
                return {
                    "task_id": task_id, "platform": platform,
                    "verify_fn": python_code, "function_name": function_name, "status": "ok",
                }

            err = f"Simulation verifier returned '{sim_result.get('result')}'"
            if "error" in sim_result:
                err += f": {sim_result['error']}"
            logger.warning(f"[{task_id}::{platform}] {err} (attempt {attempt})")
            error_history.append(err)

        if attempt < max_retries:
            time.sleep(2 ** attempt)

    # Step 5: All attempts failed — ask LLM for missing data → save as prompt_data2
    logger.error(f"[{task_id}::{platform}] All {max_retries} attempts failed")
    if last_trajectory and repairs_path:
        try:
            missing = _ask_missing_data(
                client, model,
                task.get("goal", ""), platform_plan,
                task.get("task_operations", {}).get(platform, []),
                repair.get("prompt_data", {}),
                last_trajectory, "\n".join(error_history),
                max_completion_tokens,
            )
            if missing:
                _update_repair_prompt_data2(repairs_path, task_id, platform, missing)
                logger.info(f"[{task_id}::{platform}] Saved prompt_data2 keys: {list(missing.keys())}")
        except Exception as e:
            logger.warning(f"[{task_id}::{platform}] Failed to get missing data: {e}")

    return {
        "task_id": task_id, "platform": platform,
        "verify_fn": python_code, "function_name": function_name, "status": "failed",
    }


# ── Merge ──────────────────────────────────────────────────────────────────────

def _merge_verifiers_into_tasks(verifiers_path: str, tasks_path: str) -> None:
    """Embed verifier code into tasks.jsonl metadata.verifiers after generation."""
    if not os.path.exists(verifiers_path) or not os.path.exists(tasks_path):
        return

    verifier_map: dict[str, dict[str, dict]] = {}
    with open(verifiers_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("status") == "ok":
                    task_id = item["task_id"]
                    platform = item["platform"]
                    verifier_map.setdefault(task_id, {})[platform] = {
                        "verify_fn": item["verify_fn"],
                        "function_name": item.get("function_name", "verify_task_completion"),
                    }
            except (json.JSONDecodeError, KeyError):
                pass

    tasks = load_tasks(tasks_path)
    updated = 0
    for task in tasks:
        task_id = task.get("task_id", "")
        if task_id in verifier_map:
            task.setdefault("metadata", {})["verifiers"] = verifier_map[task_id]
            updated += 1

    tmp_path = tasks_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
    os.replace(tmp_path, tasks_path)
    logger.success(f"Merged verifiers into {updated}/{len(tasks)} tasks in {tasks_path}")


# ── Run ────────────────────────────────────────────────────────────────────────

def run(args: PipelineConfig) -> None:
    tasks = load_tasks(args.tasks_output)
    schemas = load_schemas(args.schemas_output)
    done = load_done(args.verifiers_output)
    repairs = load_task_repairs(args.task_repairs_output)
    task_plans = load_task_plans(args.task_plans_output)
    envs = load_envs(args.envs_output)

    jobs: list[tuple[dict, str, bool]] = []
    for task in tasks:
        no_correct_platforms = _get_no_correct_data_platforms(task)
        all_platforms = [p for scene in task.get("scene_platforms", []) for p in scene]
        for platform in dict.fromkeys(all_platforms):
            key = f"{task['task_id']}::{platform}"
            if key in done or platform not in schemas:
                continue
            safe = platform.lower().replace(" ", "_").replace("/", "_")
            db_path = os.path.join(args.databases_dir, f"{safe}.db")
            if not os.path.exists(db_path):
                logger.warning(f"DB not found for {platform}, skipping")
                continue
            jobs.append((task, platform, platform in no_correct_platforms))

    logger.info(f"Total jobs: {len(jobs)} (skipped {len(done)} already done)")

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
        repair = repairs.get((task_id, platform), {})
        platform_plan = _get_platform_plan(task_plans, task_id, platform)
        env_item = envs.get(platform, {})
        server_path = env_item.get("server_path", "")
        platform_desc = env_item.get("description", "")
        return process_task_platform(
            client, args.model, args.gen_model,
            task, platform, schemas[platform], db_path,
            repair, platform_plan, server_path, platform_desc,
            no_correct, args.max_retries, args.max_completion_tokens,
            args.agent_run_max_iterations, args.task_repairs_output,
            args.api_key, args.base_url,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {executor.submit(_process, job): job for job in jobs}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating verifiers"):
            entry = future.result()
            append_result(args.verifiers_output, entry)
            if entry["status"] == "ok":
                success += 1
            else:
                failed += 1

    logger.success(f"Done. {success} OK, {failed} failed out of {len(jobs)} jobs.")
    logger.success(f"Verifiers saved to: {args.verifiers_output}")
    _merge_verifiers_into_tasks(args.verifiers_output, args.tasks_output)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Generate verification functions per task×platform")
    parser.add_argument("--tasks_output", type=str, default=defaults.tasks_output)
    parser.add_argument("--schemas_output", type=str, default=defaults.schemas_output)
    parser.add_argument("--databases_dir", type=str, default=defaults.databases_dir)
    parser.add_argument("--verifiers_output", type=str, default=defaults.verifiers_output)
    parser.add_argument("--task_repairs_output", type=str, default=defaults.task_repairs_output)
    parser.add_argument("--task_plans_output", type=str, default=defaults.task_plans_output)
    parser.add_argument("--envs_output", type=str, default=defaults.envs_output)
    parser.add_argument("--model", type=str, default=defaults.model)
    parser.add_argument("--gen_model", type=str, default=defaults.gen_model)
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
