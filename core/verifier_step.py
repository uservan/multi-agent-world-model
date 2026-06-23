"""
Simulate agents against real servers, run verifiers, collect diagnostic suggestions.

Flow per task (parallel):
  For each platform in scene order:
    1. Copy seed DB → temp sim_db
    2. For each sub-agent: start server → run HTTP agent loop → stop server (per-sub-agent isolation)
    3. Analyze each sub-agent individually (one LLM call per sub-agent):
       - Server logs → env_suggestions (server/endpoint bugs)
       - Trajectory + DB state → data_suggestions (missing/wrong seed data)
       - Trajectory + task_ops → goal_supplement (values agent couldn't discover)
    4. If all sub-agents completed with no suggestions → run verifier:
       - True:  extract next-scene context
       - False: analyze verifier failure → verifier/data/task_op suggestions
    5. Collect all suggestions per platform → save to suggestions.jsonl
  All platforms pass → write final record to verifier_step.jsonl
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import shutil
import sqlite3
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from utils.llm import LLMClient
from tqdm import tqdm

from core.config import PipelineConfig

from utils.server import (
    wait_for_server as _wait_for_server,
    start_server as _start_server,
    stop_server as _stop_server,
    http_call as _http_call,
)
from utils.agent_io import parse_tool_calls as _parse_tool_calls
from core.verifier_gen import _format_schema_ddl
from utils.task_utils import load_task_supplements, merge_task_supplement


# ── Prompts ────────────────────────────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are an AI agent operating a platform REST API to complete a specific sub-task.

You have one tool:
  api_call — make an HTTP request to the platform API
             arguments: {"method": "GET|POST|PUT|PATCH|DELETE", "path": "/endpoint/path", "params": {}, "body": {}}

Output each tool call as:
<tool_call>
{"name": "api_call", "arguments": {"method": "GET", "path": "/vendors/1", "params": {}, "body": null}}
</tool_call>

<tool_call>
{"name": "api_call", "arguments": {"method": "POST", "path": "/orders", "params": {}, "body": {"item_id": 1, "quantity": 2}}}
</tool_call>

Rules:
- The X-Task-ID header is automatically injected into every api_call — you do NOT need to include it; ignore any mention of it in the OpenAPI spec
- FIRST CALL: always start with GET /openapi.json to discover all available endpoints, their exact paths, and required parameters — never guess paths
- The task operations list only the high-level action names as a reference outline — follow their order, but you MUST discover every concrete value (IDs, SKUs, parameters) yourself via the API (search/list/detail endpoints); exact param values are NOT provided
- params: query string parameters (for GET/DELETE); body: JSON request body (for POST/PUT/PATCH)
- After a create (POST) call, use the returned id to GET the created resource and confirm its full identifier before using it in subsequent path calls
- When all steps are done, signal completion with a <done> block (no tool_call tags inside):
<done>
Brief summary of what was accomplished and any key values created (IDs, statuses, etc.)
</done>"""


NEXT_SCENE_SYSTEM = """You are extracting key data from an agent's platform execution to pass as context to subsequent scenes.
Given what the agent accomplished, identify specific values (IDs, names, statuses, quantities) that the next scene's agent will need.
Only include values that are genuinely needed downstream — do not include everything.
Output valid JSON only: {"context": {"descriptive_key": "value", ...}}
If nothing meaningful was produced, output: {"context": {}}"""

NEXT_SCENE_USER = """Platform: {platform}
Goal on this platform: {goal}
Expected outcome: {expected_outcome}

Agent trajectory (last steps):
{trajectory_summary}

What key data values should be passed to the next scene?"""


FIND_FAILING_ENDPOINTS_SYSTEM = """You are reading uvicorn server logs to identify failing API endpoints and their related upstream calls.

From the HTTP call list, extract endpoints that returned 4xx or 5xx status codes that indicate a server-side bug (not a data issue):
- 500: server crash / unhandled exception — always a server bug
- 405: Method Not Allowed — the route exists with a different method, or the route is missing
- 422: Unprocessable Entity on a POST/PUT/PATCH — may indicate wrong request body schema
- 404: only flag if the path looks like a fixed route (not a dynamic ID lookup like /items/123); do NOT flag if the path differs from a known route only by hyphen/underscore style (e.g. /job_postings vs /job-postings) — that is an agent path error, not a server bug
- 200/201: ignore as failing

For each failing endpoint, infer the FastAPI route pattern (convert path param segments like /bills/bill_dc2201/pay → /bills/{bill_id}/pay).

ONLY if there is an unambiguous causal link: identify a related 200 endpoint whose response value was directly used as a path parameter or body field in the failing call, AND that value appears to be wrong or incomplete (e.g. a create endpoint returned only an integer id, but the failing path used a string-formatted id derived from it). Do NOT include 200 endpoints out of general relevance — only include when you can trace the exact value from the 200 response into the failing request.

CRITICAL — Output format: single valid JSON object only
{
  "failing_endpoints": [
    {"method": "POST", "path": "/bills/{bill_id}/pay", "status": 500}
  ],
  "related_endpoints": [
    {"method": "POST", "path": "/bills", "status": 200, "reason": "create call whose returned id was used in the failing path"}
  ]
}"""

FIND_FAILING_ENDPOINTS_USER = """Server logs (filtered):
{logs}

Which endpoints failed with server-side errors, and which upstream 200 calls may have contributed?"""


ANALYZE_ENV_SYSTEM = """You are reviewing specific FastAPI endpoint implementations for bugs.

Given:
- Filtered server logs: only HTTP calls and tracebacks from a sub-agent's run
- Failing endpoint code: only the route functions that returned errors

Your job is to identify exactly what is wrong in each failing endpoint and how to fix it.

DO NOT flag as env issues:
- 200 with empty arrays — server works, data is missing
- 404 for a specific record ID — server works, that record doesn't exist yet

Each env_suggestion MUST follow this format:
  "[Why] <what the log shows: status code, exception type and message, exact line if available> [Fix] <exactly what to change in the function: which line, what to replace it with>"

CRITICAL — Output format: single valid JSON object only
{
  "has_env_issue": false,
  "env_suggestions": ["[Why] POST /bills/{bill_id}/pay returns 500 — AttributeError: 'NoneType' has no attribute 'amount' at line 142, bill query returned None but code accesses bill.amount directly. [Fix] After bill = session.query(Bill).filter(...).first(), add: if bill is None: session.close(); return {\"id\": 0, \"status\": \"not_found\"}"]
}"""

ANALYZE_ENV_USER = """Sub-agent index: {sub_agent_idx}

Filtered server logs:
{server_logs}

Failing endpoint source code:
```python
{endpoint_code}
```

What is wrong in each failing endpoint, and how should it be fixed?"""


ANALYZE_DATA_SYSTEM = """You are verifying whether the failure analysis is supported by the actual database state, and generating data fix suggestions if needed.

Your job:
1. Read the failure_analysis — it describes what data the agent needed and why it failed
2. Check the actual database tables — does the required data exist?
3. If the data IS present and correct → the failure_analysis may be wrong; return empty lists
4. If the data IS missing or incorrect → generate specific INSERT/UPDATE suggestions to fix it

Sources for specific values in [Fix]:
- Prefer exact IDs/values from the trajectory (what the agent received from API responses)
- If trajectory has no values (e.g. API returned empty), use values from the expected_outcome
- Never use placeholders — every column must have a concrete value

Server/env bugs have already been identified separately — do NOT re-report them.
A 200 with an empty array means the server works but data was not seeded.

CRITICAL — Each data_suggestion must follow this exact two-part format:
  "[Why] <one sentence: what the agent tried, what response it got, why this data is needed> [Fix] <exact SQL operation: INSERT into <table>: col=val, col=val ... / UPDATE <table> SET col=val WHERE col=val>"
- Include the exact ID value (string or integer per schema), amounts, foreign keys, status fields
- Never write a [Fix] without specifying every required column value

CRITICAL — Output format: single valid JSON object only
{
  "data_suggestions": ["[Why] Agent called PATCH /transactions/pay_dc55012 and received 404 — this transaction row does not exist in the DB. [Fix] INSERT into transactions: id='pay_dc55012', amount=3400.0, vendor_id='vend_dc4471', invoice_no='INV-2201', status='unpaid', task_id=<task_id>"],
  "goal_supplement": ["specific value or fact to add to goal that the agent couldn't discover via API"]
}"""

ANALYZE_DATA_USER = """Task ID: {task_id}
Expected outcome (what the verifier will check): {expected_outcome}

Failure analysis (what the agent tried and why it failed):
{failure_analysis}

Initial DB state (seed data — before agent ran):
{initial_db_data}

Final DB state (after agent ran — reflects what agent wrote):
{final_db_data}

Schema DDL for these tables:
{schema_ddl}

Compare initial vs final: if required data was already absent in the initial state, it is a seed data problem. If the agent wrote something but still failed, the issue may be in the values or logic."""


ANALYZE_TABLES_SYSTEM = """You are analyzing a sub-agent's trajectory to determine whether any database data problems prevented the expected outcome from being achieved.

Return both fields empty if:
- The trajectory shows the expected outcome was achieved
- All failures (404, errors) were caused by the agent's own mistakes — the agent guessed, hardcoded, or incorrectly constructed an ID/path that was never returned by any prior API response
- All API calls returned 2xx with no missing resources

Only populate tables_to_check and failure_analysis if: the expected outcome was NOT achieved AND there are failures where the agent correctly used IDs/values obtained from prior API responses (list/search/create) but still got 404 or empty results — meaning the database genuinely lacks the required data.

CRITICAL — Output format: single valid JSON object only, no prose outside it
{
  "tables_to_check": ["vendors", "transactions", "bills"],
  "failure_analysis": "The agent searched for vendor DentalCorp (found, id=vend_dc4471), then called PATCH /transactions/pay_dc55012 which returned 404 — no transaction row with that id exists in the DB. Similarly pay_cl55013 and pay_fs55014 returned 404."
}"""

ANALYZE_TABLES_USER = """Task ID: {task_id}
Goal: {goal}
Expected outcome (what the verifier will check): {expected_outcome}

Sub-agent index: {sub_agent_idx}
Sub-agent operations:
{sub_ops}

Known env/server issues (already fixed separately — exclude failures caused by these):
{env_issues}

Server logs:
{server_logs}

Agent trajectory (API calls and responses):
{trajectory}

Which tables need data fixes, and what exactly failed?"""


FIND_SOFT_ENDPOINTS_SYSTEM = """You are identifying which API endpoints may have server-side bugs based on a data failure analysis.

The failure describes calls that returned 200 but with empty or incorrect data (not 4xx/5xx errors). This may indicate a server-side query bug (wrong WHERE clause, wrong JOIN, wrong filter) rather than missing seed data.

Given the failure_analysis and the tables implicated, identify which API endpoint(s) are responsible for returning empty or wrong data. Infer the FastAPI route pattern for each (e.g. /providers/4471/availability → /providers/{provider_id}/availability).

CRITICAL — Output format: single valid JSON object only
{
  "endpoints": [
    {"method": "GET", "path": "/providers/{provider_id}/availability/check"}
  ]
}"""

FIND_SOFT_ENDPOINTS_USER = """Tables implicated in the failure: {tables_to_check}

Failure analysis:
{failure_analysis}

Tentative data issues found (may be false positives if a server bug is actually filtering/returning the data incorrectly):
{data_suggestions}

Available API endpoints:
{endpoint_list}

Which API endpoints returned empty or wrong data and should be inspected for server-side bugs?"""


ANALYZE_SOFT_ENV_SYSTEM = """You are checking whether a soft API failure (200 with empty or wrong data) is caused by a server-side bug in the endpoint code.

You are also given tentative data suggestions — these were flagged as possible missing/wrong seed data, but some or all may be false positives if a server-side bug is actually filtering or returning data incorrectly.

Look for:
- Wrong or missing WHERE clause (filtering by wrong column or wrong value)
- Type mismatch in a query filter (e.g. comparing string id to int column)
- Wrong JOIN that drops rows
- Logic error that always produces an empty result set
- Missing or incorrect task_id scoping

For each tentative data suggestion, decide: is it a false positive caused by the server bug, or is it a genuine data issue that would still exist even after fixing the server?

Output:
- has_env_issue: true if a clear server-side bug was found, false otherwise
- env_suggestions: list of server bugs to fix (empty if none)
- remaining_data_suggestions: the subset of the original data suggestions that are genuine data issues (not explained by the server bug). If has_env_issue is false, copy all original data suggestions here unchanged.

Each env_suggestion MUST follow this format:
  "[Why] <endpoint + what was returned vs expected> [Fix] <exactly what to change: which line, what to replace>"

CRITICAL — Output format: single valid JSON object only
{
  "has_env_issue": false,
  "env_suggestions": [],
  "remaining_data_suggestions": []
}"""

ANALYZE_SOFT_ENV_USER = """Failure analysis (what the agent called and what empty/wrong response was returned):
{failure_analysis}

Tentative data issues found (may be false positives if server is the real cause):
{data_suggestions}

Endpoint source code to inspect:
```python
{endpoint_code}
```

Is there a server-side bug causing this endpoint to return empty or wrong data even when the data exists in the database?"""


VERIFIER_STAGE1_SYSTEM = """You are doing an initial triage of why a task verifier returned False.

Given the verifier function, its failure, the agent trajectory, task_operations spec, schema, and available API endpoints:
1. root_cause:
   - "verifier": the verifier code itself has a bug (wrong column name, wrong comparison value, wrong logic) — the agent did correct things but verifier checks incorrectly
   - "investigate": not clearly a verifier code bug; need to inspect DB state and server code to determine if it's a data/env/agent/outcome problem
2. failure_summary: what went wrong — reference specific columns, status values, line numbers
3. If root_cause == "investigate":
   - tables_to_check: table names from the schema most relevant to the failure
   - endpoints_to_check: list of {"method": "GET"/"POST"/..., "path": "/..."} to inspect in the server code

CRITICAL — Output format: single valid JSON object only
{
  "root_cause": "verifier" | "investigate",
  "failure_summary": "detailed explanation...",
  "tables_to_check": [],
  "endpoints_to_check": []
}"""

VERIFIER_STAGE1_USER = """Task ID: {task_id}
Goal: {goal}
Expected outcome: {expected_outcome}

Task operations (expected API calls and return values):
{task_ops_summary}

Schema tables:
{schema_ddl}

Available API endpoints:
{endpoint_list}

Verifier function:
{verify_fn}

Verifier result: False
Error/reason: {verify_err}

Agent trajectory (actual API calls and server responses):
{trajectory_summary}

Is this a verifier code bug, or do we need to investigate DB state and server code further?"""


VERIFIER_STAGE2_SYSTEM = """You are determining the root cause of a task failure by inspecting database state and server code.

Given the failure analysis, task_operations spec, expected outcome, initial DB state (before agent ran), final DB state (after agent ran), and relevant server endpoint code:

Determine root_cause:
- "data": required seed data is missing or wrong in the initial DB — the agent couldn't complete because data it needed wasn't there
- "env": the server endpoint implementation is wrong — it returns incorrect values or has missing logic compared to the task_operations spec
- "task": the task definition itself is the problem — goal lacks context the agent needs, expected outcome is wrong or impossible, or task_ops have incorrect values/steps; not fixable by data or env changes alone
- "agent": NONE of the above — the seed data, the endpoints, AND the task definition are ALL correct, and the task IS solvable as-is. The verifier failed purely because the agent (solver) did not perform the required steps successfully: it skipped a needed write, gave up early, or used a value it invented instead of one it should have discovered via the API. This is NOT a defect to fix — the agent should simply try again.

CRITICAL rule for "agent": only choose it when you can POSITIVELY confirm all three are fine:
  1. the initial DB already contained every row the task needed (so NOT "data"), AND
  2. the relevant endpoint code is correct (so NOT "env"), AND
  3. the goal/expected_outcome/task_ops are all correct and achievable (so NOT "task").
If you are unsure whether data, env, or task is at fault, do NOT pick "agent" — pick the actual defect instead. "agent" means "everything is correct, the solver just didn't do it."

Then produce suggestions only for data and env paths, each in [Why]+[Fix] format:
- "data": what seed data is missing or wrong and how to fix it
  "[Why] <what the verifier checks that failed, what data was expected> [Fix] <exact SQL INSERT/UPDATE to fix the seed DB>"
- "env": what the server endpoint does wrong and how to fix it
  "[Why] <what the endpoint returns vs what task_operations expects> [Fix] <exactly what to change in the endpoint code>"

Leave lists empty for paths that are not the root cause ("task" and "agent" produce no suggestions here).

CRITICAL — Output format: single valid JSON object only
{
  "root_cause": "data" | "env" | "task" | "agent",
  "data_suggestions": [],
  "env_suggestions": [],
  "failure_summary": "refined explanation..."
}"""

VERIFIER_STAGE2_USER = """Goal: {goal}

Failure analysis (from Stage 1):
{failure_summary}

Expected outcome:
{expected_outcome}

Task operations (expected API returns — ground truth for what server SHOULD produce):
{task_ops_summary}

Initial DB state (before agent ran — seed data):
{initial_db_data}

Final DB state (after agent ran):
{final_db_data}

Relevant server endpoint code:
{endpoint_code}

Is the problem missing seed data, a server implementation bug, or a task definition issue?"""


TASK_FIX_SYSTEM = """You are analyzing why a task failed at verification and producing fix suggestions for the task definition.

The verifier function is correct, and the failure is not caused by missing seed data or a server bug — the issue is in the task definition itself.

Identify what needs to change across these three dimensions, and produce [Why]+[Fix] suggestions for each:
- goal_supplement: additional context the agent needs but the goal doesn't mention (steps it skipped, values it couldn't discover)
- outcome: corrections to the expected_outcome if it checks the wrong thing or is impossible to achieve
- task_op: corrections to existing task operations — wrong expected values, wrong sequence, or incorrect field names (do NOT invent new operations)

Each suggestion MUST follow this format:
  "[Why] <one sentence: what specifically went wrong> [Fix] <exactly what to change>"

Leave any list empty if that dimension does not need changes.

CRITICAL — Output format: single valid JSON object only
{
  "goal_supplement": ["[Why]...[Fix]..."],
  "outcome": ["[Why]...[Fix]..."],
  "task_op": ["[Why]...[Fix]..."]
}"""

TASK_FIX_USER = """Task ID: {task_id}
Goal: {goal}
Expected outcome: {expected_outcome}

Task operations (what agents should do):
{task_ops_summary}

Failure analysis:
{failure_summary}

Agent trajectory (what actually happened):
{trajectory_summary}

What in the task definition needs to change?"""


ANALYZE_VERIFIER_FIX_SYSTEM = """You are fixing a task verifier that returned False due to a bug in its own code.

Given the failure analysis and verifier source, identify what is wrong with the verifier logic and how to fix it.

Each suggestion MUST follow the [Why]...[Fix]... format:
- [Why] one sentence: what the verifier checks, why that check is wrong, referencing exact line numbers and column/field names
- [Fix] the exact code change: which line to modify and what to replace it with

CRITICAL — Output format: single valid JSON object only
{
  "verifier_suggestions": ["[Why]...[Fix]..."]
}"""

ANALYZE_VERIFIER_FIX_USER = """Failure analysis:
{failure_summary}

Expected outcome: {expected_outcome}

Verifier function:
{verify_fn}

Agent trajectory:
{trajectory_summary}

What is wrong with the verifier code and how should it be fixed?"""




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


def load_verifiers_gen(path: str) -> dict[str, dict[str, dict]]:
    """Load verifiers_gen.jsonl → {task_id: {platform: entry}}."""
    result: dict[str, dict[str, dict]] = {}
    if not os.path.exists(path):
        return result
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                tid = entry.get("task_id", "")
                platform = entry.get("platform", "")
                if tid and platform and entry.get("status") == "ok":
                    result.setdefault(tid, {})[platform] = entry
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded verifiers for {len(result)} tasks from {path}")
    return result


def load_envs(path: str) -> dict[str, dict]:
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
                if item.get("name"):
                    envs[item["name"]] = item
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(envs)} envs from {path}")
    return envs


def load_done(path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("status") == "ok" and item.get("task_id"):
                    done.add(item["task_id"])
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(done)} completed tasks from {path}")
    return done


def load_seeded_task_ids(path: str) -> set[str]:
    seeded: set[str] = set()
    if not os.path.exists(path):
        return seeded
    with open(path, "r", encoding="utf-8") as f:
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
    logger.info(f"Loaded {len(seeded)} seeded task_ids from {path}")
    return seeded


def load_suggested_task_ids(path: str) -> set[str]:
    suggested: set[str] = set()
    if not os.path.exists(path):
        return suggested
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("task_id"):
                    suggested.add(item["task_id"])
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(suggested)} task_ids with existing suggestions from {path}")
    return suggested


def load_verified_platforms(path: str) -> dict[str, dict[str, dict]]:
    """Read verified_platforms.jsonl → {task_id: {platform: context}}. Last entry wins."""
    result: dict[str, dict[str, dict]] = {}
    if not os.path.exists(path):
        return result
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                task_id = item.get("task_id")
                platform = item.get("platform")
                if task_id and platform:
                    result.setdefault(task_id, {})[platform] = item.get("context", {})
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded verified platforms for {len(result)} tasks from {path}")
    return result


def append_verified_platform(path: str, task_id: str, platform: str, context: dict) -> None:
    """Append one verified-platform record. Never rewrites — pure append."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    entry = {"task_id": task_id, "platform": platform, "context": context}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_result(path: str, entry: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Helpers ────────────────────────────────────────────────────────────────────


def _meta(task: dict) -> dict:
    return task.get("metadata", {})


def _robust_json_loads(text: str) -> dict:
    text = text.strip()
    # Strip outer markdown fence if the whole response is wrapped
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

    def _try_loads(s: str) -> dict | None:
        s = re.sub(r'\bTrue\b', 'true', s)
        s = re.sub(r'\bFalse\b', 'false', s)
        s = re.sub(r'\bNone\b', 'null', s)
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    # Fast path: the whole text is JSON
    result = _try_loads(text)
    if result is not None:
        return result

    # Extract from ```json ... ``` block in mixed text
    m = re.search(r'```(?:json)?\s*\n(\{.*?\})\s*\n```', text, re.DOTALL)
    if m:
        result = _try_loads(m.group(1))
        if result is not None:
            return result

    # Scan every '{' position with raw_decode — handles URLs like /path/{id} in surrounding text
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == '{':
            try:
                obj, _ = decoder.raw_decode(text, i)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

    raise json.JSONDecodeError("No valid JSON object found", text, 0)


def _list_endpoints(server_code: str) -> str:
    """Extract all route signatures from FastAPI server source as a compact list."""
    lines = []
    for line in server_code.splitlines():
        m = re.search(r'@app\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']', line, re.IGNORECASE)
        if m:
            lines.append(f"  {m.group(1).upper()} {m.group(2)}")
    return "\n".join(lines) or "(no endpoints found)"


def _extract_endpoint_code(server_code: str, method: str, path: str) -> str:
    """Extract the route function for method+path from FastAPI server source."""
    lines = server_code.splitlines()
    decorator_re = re.compile(
        rf'@app\.{re.escape(method.lower())}\s*\(\s*["\']' + re.escape(path) + r'["\']',
        re.IGNORECASE,
    )
    start = None
    for i, line in enumerate(lines):
        if decorator_re.search(line):
            start = i
            break
    if start is None:
        return ""
    result = []
    for line in lines[start:]:
        if result and re.match(r'\s*@app\.', line):
            break
        result.append(line)
        if len(result) > 80:
            break
    return "\n".join(result)


def _filter_server_logs(logs: str) -> str:
    """Extract HTTP calls summary + user-code-only tracebacks from raw uvicorn logs."""
    if not logs:
        return "(no logs)"

    http_lines: list[str] = []
    error_blocks: list[str] = []

    lines = logs.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # HTTP request lines: capture method + path + status, skip /docs and startup noise
        http_match = re.search(r'"(GET|POST|PUT|PATCH|DELETE) ([^ ]+)[^"]*" (\d+)', line)
        if http_match and "/docs" not in http_match.group(2):
            http_lines.append(f"  {http_match.group(1)} {http_match.group(2)} → {http_match.group(3)}")
            i += 1
            continue

        # ERROR line: start collecting a traceback block
        if line.startswith("ERROR:") or "Exception in ASGI" in line:
            block: list[str] = [line]
            i += 1
            in_tb = False
            user_frames: list[str] = []
            exc_line = ""
            while i < len(lines):
                l = lines[i]
                if l.strip() == "Traceback (most recent call last):":
                    in_tb = True
                    i += 1
                    continue
                if in_tb:
                    # File frame line
                    if l.strip().startswith('File "'):
                        if "site-packages" not in l:
                            # user code frame: keep this + next line (the code)
                            user_frames.append(l.rstrip())
                            if i + 1 < len(lines):
                                user_frames.append(lines[i + 1].rstrip())
                                i += 2
                                continue
                    # Exception line (not a File line, not indented frame)
                    elif not l.startswith(" ") and "Error" in l or (l and not l.startswith(" ") and in_tb and user_frames):
                        exc_line = l.rstrip()
                        i += 1
                        break
                # New top-level log line starts — end of this error block
                elif l.startswith("INFO:") or l.startswith("ERROR:") or l.startswith("WARNING:"):
                    break
                i += 1
            if user_frames or exc_line:
                block.append("Traceback (user code):")
                block.extend(user_frames)
                if exc_line:
                    block.append(exc_line)
            error_blocks.append("\n".join(block))
            continue

        i += 1

    parts: list[str] = []
    if http_lines:
        parts.append("HTTP calls:\n" + "\n".join(http_lines))
    if error_blocks:
        parts.append("Errors:\n" + "\n\n".join(error_blocks))
    return "\n\n".join(parts) or "(no relevant log entries)"


def _dump_db_by_table(db_path: str, task_id: str, schema_item: dict, limit: int = 20) -> dict[str, str]:
    """Dump DB rows per table → {table_name: formatted_string}."""
    if not os.path.exists(db_path):
        return {}
    conn = sqlite3.connect(db_path)
    result: dict[str, str] = {}
    try:
        for table in schema_item.get("schemas", []):
            table_name = table.get("table", "")
            if not table_name:
                continue
            try:
                cur = conn.execute(
                    f"SELECT * FROM {table_name} WHERE task_id = ? LIMIT {limit}",
                    (task_id,),
                )
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                lines = [f"Table: {table_name} ({', '.join(cols)})"]
                for row in rows:
                    lines.append(f"  {dict(zip(cols, row))}")
                if not rows:
                    lines.append("  (no rows for this task_id)")
                result[table_name] = "\n".join(lines)
            except Exception as e:
                result[table_name] = f"Table: {table_name} (error: {e})"
    finally:
        conn.close()
    return result


def _run_verifier(
    python_code: str,
    function_name: str,
    initial_db: str,
    final_db: str,
    task_id: str,
) -> tuple[bool, str]:
    """Execute verifier code; return (passed, error_message)."""
    namespace: dict = {
        "sqlite3": sqlite3, "json": json, "os": os,
        "__builtins__": __builtins__,
    }
    try:
        exec(python_code, namespace)
        fn = namespace.get(function_name)
        if not fn:
            return False, f"Function '{function_name}' not found"
        result = fn(initial_db, final_db, task_id)
        if not isinstance(result, bool):
            return False, f"Expected bool, got {type(result).__name__}: {result}"
        return result, ""
    except Exception as e:
        return False, str(e)


def _apply_data_fixes(db_path: str, sql_stmts: list[str]) -> tuple[bool, str]:
    """Execute SQL fix statements inside a transaction."""
    if not sql_stmts:
        return True, ""
    try:
        conn = sqlite3.connect(db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("BEGIN")
        for stmt in sql_stmts:
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        conn.execute("COMMIT")
        conn.close()
        return True, ""
    except Exception as e:
        try:
            conn.execute("ROLLBACK")
            conn.close()
        except Exception:
            pass
        return False, str(e)


def _summarise_trajectory(trajectory: list[dict], last_n: int = 8) -> str:
    steps = []
    for entry in trajectory:
        for step in entry.get("steps", [])[-last_n:]:
            if step.get("tool_call"):
                steps.append({
                    "sub_agent": entry.get("sub_agent"),
                    "call": step["tool_call"],
                    "response": str(step.get("tool_response", ""))[:300],
                })
    return json.dumps(steps, ensure_ascii=False, indent=2)


# ── LLM helpers ────────────────────────────────────────────────────────────────

def _ask_next_scene_data(
    client: LLMClient,
    model: str,
    platform: str,
    goal: str,
    expected_outcome: str,
    trajectory: list[dict],
    max_completion_tokens: int,
) -> dict:
    user_content = NEXT_SCENE_USER.format(
        platform=platform,
        goal=goal,
        expected_outcome=expected_outcome,
        trajectory_summary=_summarise_trajectory(trajectory),
    )
    messages = [
        {"role": "system", "content": NEXT_SCENE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        raw = client.complete(model, messages, max_completion_tokens)
        result = _robust_json_loads(raw)
        return result.get("context", {})
    except Exception as e:
        logger.warning(f"_ask_next_scene_data failed: {e}")
        return {}



def _analyze_sub_agent(
    sub_agent_idx: int,
    steps: list[dict],
    server_logs: str,
    server_code: str,
    sub_ops: list,
    goal: str,
    task_id: str,
    expected_outcome: str,
    schema_item: dict,
    seed_db: str,
    sim_db: str,
    client: LLMClient,
    model: str,
    max_completion_tokens: int,
) -> tuple[bool, dict]:
    """Analyze one sub-agent's run. Returns (completed, {env, data, goal_supplement})."""
    traj_parts = []
    for step in steps:
        if step.get("tool_call"):
            call = step["tool_call"]
            resp = str(step.get("tool_response", ""))[:500]
            traj_parts.append(f"  call: {json.dumps(call)}\n  response: {resp}")
        elif step.get("is_final"):
            raw_content = str(step.get("content", ""))
            done_match = re.search(r"<done>(.*?)</done>", raw_content, re.DOTALL)
            final_text = done_match.group(1).strip() if done_match else raw_content[:300]
            label = "[DONE]" if step.get("explicit_done") else "[FINAL]"
            traj_parts.append(f"  {label} {final_text}")

    logs_filtered = _filter_server_logs(server_logs or "")
    trajectory_str = "\n".join(traj_parts) or "(no steps)"
    sub_ops_str = json.dumps(sub_ops, ensure_ascii=False, indent=2)

    # Step 1a: find which endpoints failed from logs (skip if no hard errors in logs)
    env_suggestions: list[str] = []
    failing_endpoints: list[dict] = []
    related_endpoints: list[dict] = []
    _has_hard_errors = bool(re.search(r'→\s+[45]\d\d', logs_filtered))
    if _has_hard_errors:
        try:
            find_user = FIND_FAILING_ENDPOINTS_USER.format(logs=logs_filtered)
            raw = client.complete(model, [
                {"role": "system", "content": FIND_FAILING_ENDPOINTS_SYSTEM},
                {"role": "user", "content": find_user},
            ], max_completion_tokens)
            find_result = _robust_json_loads(raw)
            failing_endpoints = find_result.get("failing_endpoints", [])
            related_endpoints = find_result.get("related_endpoints", [])
        except Exception as e:
            logger.warning(f"_analyze_sub_agent find-endpoints step failed (sub_agent={sub_agent_idx}): {e}")

    # Step 1b: extract code for failing endpoints (+ related 200 endpoints), then analyse for env bugs
    if failing_endpoints and server_code:
        try:
            code_blocks: list[str] = []
            for ep in failing_endpoints:
                code = _extract_endpoint_code(server_code, ep.get("method", ""), ep.get("path", ""))
                if code:
                    code_blocks.append(f"# {ep['method']} {ep['path']} → {ep.get('status')}\n{code}")
            for ep in related_endpoints:
                code = _extract_endpoint_code(server_code, ep.get("method", ""), ep.get("path", ""))
                if code:
                    code_blocks.append(f"# [related 200] {ep['method']} {ep['path']} — {ep.get('reason', '')}\n{code}")
            endpoint_code = "\n\n".join(code_blocks) or "(could not extract endpoint code)"
            env_user = ANALYZE_ENV_USER.format(
                sub_agent_idx=sub_agent_idx,
                server_logs=logs_filtered,
                endpoint_code=endpoint_code,
            )
            raw = client.complete(model, [
                {"role": "system", "content": ANALYZE_ENV_SYSTEM},
                {"role": "user", "content": env_user},
            ], max_completion_tokens)
            env_suggestions = _robust_json_loads(raw).get("env_suggestions", [])
        except Exception as e:
            logger.warning(f"_analyze_sub_agent env step failed (sub_agent={sub_agent_idx}): {e}")

    # If env bugs found, stop here — trajectory is unreliable until server is fixed
    if env_suggestions:
        logger.info(f"[sub_agent={sub_agent_idx}] env issues found, skipping data analysis")
        return False, {"env": env_suggestions, "data": [], "goal_supplement": []}

    # Step 2a: identify which tables to check + get a failure analysis
    tables_to_check: list[str] = []
    failure_analysis = ""
    try:
        tables_user = ANALYZE_TABLES_USER.format(
            task_id=task_id,
            goal=goal,
            expected_outcome=expected_outcome,
            sub_agent_idx=sub_agent_idx,
            sub_ops=sub_ops_str,
            env_issues="none",
            server_logs=logs_filtered,
            trajectory=trajectory_str,
        )
        raw = client.complete(model, [
            {"role": "system", "content": ANALYZE_TABLES_SYSTEM},
            {"role": "user", "content": tables_user},
        ], max_completion_tokens)
        tables_result = _robust_json_loads(raw)
        tables_to_check = tables_result.get("tables_to_check", [])
        failure_analysis = tables_result.get("failure_analysis", "")
    except Exception as e:
        logger.warning(f"_analyze_sub_agent tables step failed (sub_agent={sub_agent_idx}): {e}")

    if not tables_to_check and not failure_analysis:
        logger.info(f"[sub_agent={sub_agent_idx}] no failure identified, agent likely completed successfully")
        return True, {"env": [], "data": [], "goal_supplement": []}

    # Step 2b: dump only relevant tables, filter schema DDL, then generate data suggestions
    all_tables_by_name = {
        t.get("table", ""): t
        for t in schema_item.get("schemas", [])
        if t.get("table")
    }
    target_tables = [t for t in tables_to_check if t in all_tables_by_name] or list(all_tables_by_name)
    seed_by_table = _dump_db_by_table(seed_db, task_id, schema_item)
    sim_by_table = _dump_db_by_table(sim_db, task_id, schema_item)
    initial_db = "\n\n".join(seed_by_table.get(t, f"Table: {t} (no data)") for t in target_tables)
    final_db = "\n\n".join(sim_by_table.get(t, f"Table: {t} (no data)") for t in target_tables)
    relevant_ddl = "\n\n".join(
        all_tables_by_name[t].get("ddl", "") for t in target_tables if t in all_tables_by_name
    )

    try:
        data_user = ANALYZE_DATA_USER.format(
            task_id=task_id,
            expected_outcome=expected_outcome,
            failure_analysis=failure_analysis,
            initial_db_data=initial_db[:8000],
            final_db_data=final_db[:8000],
            schema_ddl=relevant_ddl[:8000],
        )
        raw = client.complete(model, [
            {"role": "system", "content": ANALYZE_DATA_SYSTEM},
            {"role": "user", "content": data_user},
        ], max_completion_tokens)
        data_result = _robust_json_loads(raw)
        data_suggestions = data_result.get("data_suggestions", [])
        goal_supplement = data_result.get("goal_supplement", [])
    except Exception as e:
        logger.warning(f"_analyze_sub_agent data step failed (sub_agent={sub_agent_idx}): {e}")
        return False, {"env": env_suggestions, "data": [], "goal_supplement": []}

    # Step 3: check whether the failure is also caused by a soft server-side bug
    # (endpoint returned 200 but empty/wrong data — can co-exist with data issues)
    if failure_analysis and server_code:
        try:
            # Step 3a: identify which endpoints to inspect
            data_sug_block = "\n".join(f"- {s}" for s in data_suggestions) or "(none)"
            soft_find_user = FIND_SOFT_ENDPOINTS_USER.format(
                tables_to_check=", ".join(tables_to_check) or "(unknown)",
                failure_analysis=failure_analysis,
                data_suggestions=data_sug_block,
                endpoint_list=_list_endpoints(server_code),
            )
            raw = client.complete(model, [
                {"role": "system", "content": FIND_SOFT_ENDPOINTS_SYSTEM},
                {"role": "user", "content": soft_find_user},
            ], max_completion_tokens)
            soft_endpoints = _robust_json_loads(raw).get("endpoints", [])

            # Step 3b: extract code and check for server-side bugs
            if soft_endpoints:
                code_blocks: list[str] = []
                for ep in soft_endpoints:
                    code = _extract_endpoint_code(server_code, ep.get("method", ""), ep.get("path", ""))
                    if code:
                        code_blocks.append(f"# {ep['method']} {ep['path']}\n{code}")
                if code_blocks:
                    soft_env_user = ANALYZE_SOFT_ENV_USER.format(
                        failure_analysis=failure_analysis,
                        data_suggestions=data_sug_block,
                        endpoint_code="\n\n".join(code_blocks),
                    )
                    raw = client.complete(model, [
                        {"role": "system", "content": ANALYZE_SOFT_ENV_SYSTEM},
                        {"role": "user", "content": soft_env_user},
                    ], max_completion_tokens)
                    soft_result = _robust_json_loads(raw)
                    extra_env = soft_result.get("env_suggestions", [])
                    has_env_issue = soft_result.get("has_env_issue", False)
                    remaining = soft_result.get("remaining_data_suggestions", data_suggestions)
                    if has_env_issue and extra_env:
                        env_suggestions = env_suggestions + extra_env
                        data_suggestions = remaining
                        logger.info(f"[sub_agent={sub_agent_idx}] soft-failure: env bug confirmed ({len(extra_env)}), data suggestions {len(data_suggestions)} kept / {len(remaining)} remaining")
                    elif extra_env:
                        env_suggestions = env_suggestions + extra_env
                        logger.info(f"[sub_agent={sub_agent_idx}] soft-failure: {len(extra_env)} env suggestion(s), data suggestions kept")
        except Exception as e:
            logger.warning(f"_analyze_sub_agent soft-failure env check failed (sub_agent={sub_agent_idx}): {e}")

    # completed is derived from final suggestions after cross-check — any remaining data or env issues mean sub-agent didn't finish cleanly
    completed = not data_suggestions and not env_suggestions

    return completed, {
        "env": env_suggestions,
        "data": data_suggestions,
        "goal_supplement": goal_supplement,
    }





def _format_task_ops_summary(task_ops: list) -> str:
    """Compact summary of task_operations expected returns for each sub-agent."""
    lines = []
    for ki, sub_ops in enumerate(task_ops):
        if not isinstance(sub_ops, list):
            continue
        lines.append(f"Sub-agent {ki}:")
        for op in sub_ops:
            action = op.get("action", "?")
            returns = op.get("returns", {})
            lines.append(f"  {action} → {json.dumps(returns, ensure_ascii=False)}")
    return "\n".join(lines) or "(no task_operations)"


def _dump_tables_data(db_path: str, task_id: str, table_names: list[str], limit: int = 20) -> str:
    """Dump rows for specific tables filtered by task_id."""
    if not os.path.exists(db_path):
        return "(database not found)"
    lines = []
    try:
        conn = sqlite3.connect(db_path)
        for table in table_names:
            try:
                cur = conn.execute(f"SELECT * FROM {table} WHERE task_id = ? LIMIT {limit}", (task_id,))
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                lines.append(f"Table {table}: {', '.join(cols)}")
                for row in rows:
                    lines.append(f"  {dict(zip(cols, row))}")
                if not rows:
                    lines.append("  (no rows)")
            except Exception as e:
                lines.append(f"Table {table}: (error: {e})")
        conn.close()
    except Exception as e:
        return f"(DB error: {e})"
    return "\n".join(lines) or "(no data)"


def _analyze_verifier_failure(
    verify_fn: str,
    verify_err: str,
    full_trajectory: list[dict],
    goal: str,
    expected_outcome: str,
    task_id: str,
    platform: str,
    client: LLMClient,
    model: str,
    max_completion_tokens: int,
    task_ops: list | None = None,
    schema_item: dict | None = None,
    seed_db: str = "",
    sim_db: str = "",
    server_code: str = "",
) -> dict:
    """Two-stage verifier failure analysis.
    Stage 1: triage using trajectory + schema + endpoints → verifier bug or investigate
    Stage 2 (investigate path): DB state + server code → data / env / goal_supplement / outcome
    """
    empty = {"verifier": [], "env": [], "data": [], "task_op": [], "goal_supplement": [], "outcome": []}

    traj_parts = []
    for entry in full_trajectory:
        ki = entry.get("sub_agent", "?")
        for step in entry.get("steps", []):
            if step.get("tool_call"):
                call = step["tool_call"]
                resp = str(step.get("tool_response", ""))[:300]
                traj_parts.append(f"[sub_agent={ki}] call={json.dumps(call)} resp={resp}")
    trajectory_summary = "\n".join(traj_parts[-30:]) or "(no steps)"
    task_ops_summary = _format_task_ops_summary(task_ops or [])
    schema_ddl = _format_schema_ddl(schema_item or {})
    endpoint_list = _list_endpoints(server_code)

    # ── Stage 1: triage ──────────────────────────────────────────────────────
    root_cause = "investigate"
    failure_summary = "(no summary)"
    tables_to_check: list[str] = []
    endpoints_to_check: list[dict] = []
    try:
        stage1_user = VERIFIER_STAGE1_USER.format(
            task_id=task_id,
            goal=goal,
            expected_outcome=expected_outcome,
            task_ops_summary=task_ops_summary,
            schema_ddl=schema_ddl,
            endpoint_list=endpoint_list,
            verify_fn=verify_fn,
            verify_err=verify_err,
            trajectory_summary=trajectory_summary,
        )
        raw = client.complete(model, [
            {"role": "system", "content": VERIFIER_STAGE1_SYSTEM},
            {"role": "user", "content": stage1_user},
        ], max_completion_tokens)
        s1 = _robust_json_loads(raw)
        root_cause = s1.get("root_cause", "investigate")
        failure_summary = s1.get("failure_summary", "(no summary)")
        tables_to_check = s1.get("tables_to_check", [])
        endpoints_to_check = s1.get("endpoints_to_check", [])
        logger.info(f"[{task_id}::{platform}] verifier Stage 1 root_cause={root_cause}")
    except Exception as e:
        logger.warning(f"_analyze_verifier_failure stage1 failed: {e}")

    # ── Verifier bug → Stage 2: generate fix suggestions ─────────────────────
    if root_cause == "verifier":
        try:
            fix_user = ANALYZE_VERIFIER_FIX_USER.format(
                failure_summary=failure_summary,
                expected_outcome=expected_outcome,
                verify_fn=verify_fn,
                trajectory_summary=trajectory_summary,
            )
            raw = client.complete(model, [
                {"role": "system", "content": ANALYZE_VERIFIER_FIX_SYSTEM},
                {"role": "user", "content": fix_user},
            ], max_completion_tokens)
            result = _robust_json_loads(raw)
            return {**empty, "verifier": result.get("verifier_suggestions", [])}
        except Exception as e:
            logger.warning(f"_analyze_verifier_failure verifier-fix stage failed: {e}")
            return empty

    # ── Investigate → Stage 2: DB state + server code → data/env/goal/outcome ─
    try:
        initial_db_data = _dump_tables_data(seed_db, task_id, tables_to_check) if tables_to_check else "(no tables selected)"
        final_db_data = _dump_tables_data(sim_db, task_id, tables_to_check) if tables_to_check else "(no tables selected)"

        code_blocks = []
        for ep in endpoints_to_check:
            method = ep.get("method", "")
            path = ep.get("path", "")
            code = _extract_endpoint_code(server_code, method, path)
            if code:
                code_blocks.append(f"{method.upper()} {path}:\n{code}")
        endpoint_code = "\n\n".join(code_blocks) or "(no endpoint code found)"

        stage2_user = VERIFIER_STAGE2_USER.format(
            goal=goal,
            failure_summary=failure_summary,
            expected_outcome=expected_outcome,
            task_ops_summary=task_ops_summary,
            initial_db_data=initial_db_data,
            final_db_data=final_db_data,
            endpoint_code=endpoint_code,
        )
        raw = client.complete(model, [
            {"role": "system", "content": VERIFIER_STAGE2_SYSTEM},
            {"role": "user", "content": stage2_user},
        ], max_completion_tokens)
        s2 = _robust_json_loads(raw)
        root_cause2 = s2.get("root_cause", "outcome")
        refined_summary = s2.get("failure_summary", failure_summary)
        logger.info(f"[{task_id}::{platform}] verifier Stage 2 root_cause={root_cause2}")

        if root_cause2 == "data":
            return {**empty, "data": s2.get("data_suggestions", [])}
        if root_cause2 == "env":
            return {**empty, "env": s2.get("env_suggestions", [])}
        if root_cause2 == "agent":
            # data + env + task are all confirmed correct; the solver just didn't
            # do it. Not a defect — skip this round, let the agent try again later.
            return {**empty, "_skip": True}

        # ── task path → Stage 3: goal / outcome / task_op suggestions ────────
        logger.info(f"[{task_id}::{platform}] verifier Stage 3: task definition fix")
        try:
            task_fix_user = TASK_FIX_USER.format(
                task_id=task_id,
                goal=goal,
                expected_outcome=expected_outcome,
                task_ops_summary=task_ops_summary,
                failure_summary=refined_summary,
                trajectory_summary=trajectory_summary,
            )
            raw3 = client.complete(model, [
                {"role": "system", "content": TASK_FIX_SYSTEM},
                {"role": "user", "content": task_fix_user},
            ], max_completion_tokens)
            s3 = _robust_json_loads(raw3)
            return {
                **empty,
                "goal_supplement": s3.get("goal_supplement", []),
                "outcome": s3.get("outcome", []),
                "task_op": s3.get("task_op", []),
            }
        except Exception as e3:
            logger.warning(f"_analyze_verifier_failure task-fix stage failed: {e3}")
            return empty
    except Exception as e:
        logger.warning(f"_analyze_verifier_failure investigate stage failed: {e}")
        return empty


# ── HTTP agent helpers ─────────────────────────────────────────────────────────

class _LLMUnavailable(Exception):
    """The LLM call itself failed during simulation (transport/connection error).

    This is an infrastructure problem, NOT a task defect — we must abort this
    round's verification and leave the task pending, rather than analyze an empty
    trajectory and generate bogus fix suggestions.
    """


async def _run_http_agent_loop(
    base_url: str,
    task_id: str,
    llm_client: LLMClient,
    model: str,
    user_prompt: str,
    max_iterations: int,
) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    trajectory: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        try:
            content = await asyncio.get_running_loop().run_in_executor(
                None, llm_client.complete, model, messages, 1024 * 8
            )
        except Exception as e:
            raise _LLMUnavailable(str(e)) from e
        tool_calls = _parse_tool_calls(content)
        messages.append({"role": "assistant", "content": content})

        has_done = "<done>" in content
        if has_done or not tool_calls:
            trajectory.append({"iteration": iteration, "content": content, "is_final": True, "explicit_done": has_done})
            break

        responses: list[str] = []
        for tc in tool_calls:
            if tc["name"] == "api_call":
                args = tc.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                method = str(args.get("method", "GET")).upper()
                path = str(args.get("path", "/"))
                params = args.get("params") or {}
                body = args.get("body")
                tool_response = await asyncio.get_running_loop().run_in_executor(
                    None, _http_call, base_url, task_id, method, path, params, body
                )
                responses.append(f"[{method} {path}] {tool_response}")
            else:
                tool_response = f"Error: unknown tool '{tc['name']}'. Use 'api_call'."
                responses.append(tool_response)

            trajectory.append({
                "iteration": iteration,
                "content": content,
                "tool_call": tc,
                "tool_response": tool_response,
            })

        messages.append({"role": "user", "content": "API responses:\n" + "\n---\n".join(responses)})
    else:
        trajectory.append({"iteration": max_iterations, "content": content, "is_final": True, "timed_out": True})

    return trajectory


# ── Agent simulation ───────────────────────────────────────────────────────────

def _ops_actions_only(sub_ops: list) -> list:
    """Reduce task operations to just their `action` names.

    The solver is given the step outline as a reference, but NOT the concrete
    `params` (which leak IDs/SKUs) or `returns` (which leak the expected answer).
    It must discover every concrete value itself via the API — so verification
    actually tests whether the task is solvable from discoverable information.
    """
    actions = []
    for step in sub_ops:
        if isinstance(step, dict) and step.get("action"):
            actions.append({"action": step["action"]})
        elif isinstance(step, str) and step:
            actions.append({"action": step})
    return actions


def _build_agent_prompt(
    goal: str,
    platform: str,
    platform_desc: str,
    sub_ops: list,
    sub_agent_idx: int,
    total_sub_agents: int,
    context_data: dict,
    prior_results: list,
) -> str:
    ops_str = json.dumps(_ops_actions_only(sub_ops), ensure_ascii=False, indent=2)

    parts = []
    if context_data:
        parts.append("Context from prior platforms:\n" + json.dumps(context_data, ensure_ascii=False, indent=2))
    if prior_results:
        parts.append("Results from previous sub-agents on this platform:\n" + json.dumps(prior_results[-10:], ensure_ascii=False, indent=2))
    context_str = ("\n\n" + "\n\n".join(parts)) if parts else ""

    return (
        f"Overall goal: {goal}\n\n"
        f"Platform: {platform}\n"
        f"Platform description: {platform_desc}\n\n"
        f"You are sub-agent {sub_agent_idx + 1}/{total_sub_agents} on {platform}.\n"
        f"Your job is to complete ONLY the steps listed below — do not attempt other sub-agents' work.\n\n"
        f"Steps for this sub-agent (execute each in order using the API tools):\n{ops_str}"
        f"{context_str}"
    )


async def _simulate_platform_async(
    server_path: str,
    sim_db: str,
    task_id: str,
    task_ops: list,
    goal: str,
    context_data: dict,
    platform: str,
    platform_desc: str,
    llm_client: LLMClient,
    model: str,
    iterations_multiplier: int,
) -> list[dict]:
    full_trajectory: list[dict] = []
    prior_results: list[dict] = []

    for ki, sub_ops in enumerate(task_ops):
        if not isinstance(sub_ops, list):
            continue

        max_iterations = max(len(sub_ops), 1) * iterations_multiplier
        proc, port = _start_server(server_path, sim_db)
        sub_traj: list[dict] = []
        server_logs = ""
        try:
            if not _wait_for_server(port, timeout=25):
                raise RuntimeError(f"Server for {platform} sub_agent={ki} failed to start")
            prompt = _build_agent_prompt(
                goal, platform, platform_desc, sub_ops,
                ki, len(task_ops), context_data, prior_results,
            )
            sub_traj = await _run_http_agent_loop(
                f"http://127.0.0.1:{port}", task_id, llm_client, model, prompt, max_iterations,
            )
        except _LLMUnavailable:
            raise   # propagate (finally stops the server) — abort this task's verification
        except Exception as e:
            logger.warning(f"[{task_id}] {platform} sub_agent={ki} error: {e}")
        finally:
            _stop_server(proc)
            try:
                server_logs = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
            except Exception:
                server_logs = ""

        for step in sub_traj:
            resp = step.get("tool_response", "")
            if resp and not str(resp).startswith("Error"):
                try:
                    parsed = json.loads(resp)
                    prior_results.append({"sub_agent": ki, "result": parsed})
                except Exception:
                    pass

        full_trajectory.append({
            "platform": platform, "sub_agent": ki,
            "steps": sub_traj, "server_logs": server_logs,
        })

    return full_trajectory


def _simulate_platform(
    server_path: str,
    sim_db: str,
    task_id: str,
    task_ops: list,
    goal: str,
    context_data: dict,
    platform: str,
    platform_desc: str,
    llm_client: LLMClient,
    model: str,
    iterations_multiplier: int,
) -> list[dict]:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _simulate_platform_async(
                server_path, sim_db, task_id, task_ops,
                goal, context_data, platform, platform_desc,
                llm_client, model, iterations_multiplier,
            )
        )
    finally:
        loop.close()


# ── Per-task processing ────────────────────────────────────────────────────────


def process_task(
    client: LLMClient,
    model: str,
    gen_model: str,
    task: dict,
    schemas: dict[str, dict],
    verifiers_gen: dict[str, dict[str, dict]],
    databases_dir: str,
    envs: dict[str, dict],
    max_retries: int,
    max_completion_tokens: int,
    iterations_multiplier: int,
    suggestions_path: str = "",
    verified_platforms: dict[str, dict] | None = None,
    task_supplement: dict | None = None,
) -> dict:
    task_id = task["task_id"]
    task = merge_task_supplement(task, task_supplement)
    goal = task.get("goal", "")
    meta = _meta(task)

    scene_platforms_list = meta.get("scene_platforms", [])

    platform_results: dict[str, dict] = {}
    all_suggestions: dict[str, dict] = {}
    context_data: dict = {}  # accumulated context from all completed scenes

    for scene in scene_platforms_list:
        scene_ctx: dict = {}  # context collected from platforms in this scene only

        for platform in scene:
            # Already verified in a previous run — recover its ctx into scene_ctx and skip
            if verified_platforms and platform in verified_platforms:
                saved = verified_platforms[platform]
                if isinstance(saved, dict):
                    scene_ctx.update(saved)
                logger.info(f"[{task_id}::{platform}] already verified, skipping")
                continue
            if platform not in schemas:
                logger.warning(f"[{task_id}::{platform}] no schema, skipping")
                continue

            task_verifier = verifiers_gen.get(task_id, {}).get(platform)
            if not task_verifier:
                logger.warning(f"[{task_id}::{platform}] no verifier, skipping")
                continue

            safe = platform.lower().replace(" ", "_").replace("/", "_")
            seed_db = os.path.join(databases_dir, f"{safe}.db")
            if not os.path.exists(seed_db):
                logger.warning(f"[{task_id}::{platform}] seed DB not found, skipping")
                continue

            env_item = envs.get(platform, {})
            server_path = env_item.get("server_path", "")
            platform_desc = env_item.get("description", "")
            try:
                with open(server_path, "r", encoding="utf-8") as _sf:
                    server_code = _sf.read()
            except Exception:
                server_code = ""

            if not server_path or not os.path.exists(server_path):
                logger.warning(f"[{task_id}::{platform}] server not found, skipping")
                continue

            task_ops = meta.get("task_operations", {}).get(platform, [])
            expected_outcome = meta.get("expected_outcome", {}).get(platform, "")
            is_read_only = task_verifier.get("read_only", False)
            verify_fn = task_verifier.get("verify_fn", "")
            function_name = task_verifier.get("function_name", "verify_task_completion")

            success = False
            platform_suggestions: dict = {
                "env": [], "data": [], "goal_supplement": [], "verifier": [], "task_op": [], "outcome": [],
            }

            with tempfile.TemporaryDirectory() as tmpdir:
                sim_db = os.path.join(tmpdir, "sim.db")
                shutil.copy2(seed_db, sim_db)
                os.chmod(sim_db, 0o644)

                # Run agent simulation
                try:
                    trajectory = _simulate_platform(
                        server_path, sim_db, task_id, task_ops,
                        goal, context_data, platform, platform_desc,
                        client, gen_model, iterations_multiplier,
                    )
                except _LLMUnavailable as e:
                    # Infra error (LLM unreachable), not a task defect. Abort this round
                    # with NO suggestions so nothing gets "fixed"; the task stays pending
                    # (not added to verified_platforms) and is re-verified next round.
                    logger.warning(f"[{task_id}::{platform}] LLM unavailable — skipping verification this round: {e}")
                    return {"task_id": task_id, "status": "skipped", "suggestions": {}}
                except Exception as e:
                    logger.warning(f"[{task_id}::{platform}] sim error: {e}")
                    all_suggestions[platform] = platform_suggestions
                    logger.error(f"[{task_id}::{platform}] simulation failed, aborting task")
                    return {"task_id": task_id, "status": "failed", "suggestions": all_suggestions}

                # Analyze each sub-agent's run individually
                all_completed = True
                for entry in trajectory:
                    ki = entry.get("sub_agent", 0)
                    sub_ops = task_ops[ki] if isinstance(task_ops, list) and ki < len(task_ops) else []
                    completed, sug = _analyze_sub_agent(
                        ki, entry.get("steps", []), entry.get("server_logs", ""),
                        server_code, sub_ops, goal, task_id,
                        expected_outcome,
                        schemas[platform], seed_db, sim_db,
                        client, model, max_completion_tokens,
                    )
                    if not completed:
                        all_completed = False
                    for k in ["env", "data", "goal_supplement", "task_op"]:
                        platform_suggestions[k].extend(sug.get(k, []))

                # Only run verifier if all sub-agents completed with no blocking issues
                has_sim_issues = any(platform_suggestions[k] for k in ["env", "data", "goal_supplement"])

                if all_completed and not has_sim_issues:
                    if is_read_only:
                        verified, verify_err = True, ""
                    else:
                        verified, verify_err = _run_verifier(verify_fn, function_name, seed_db, sim_db, task_id)

                    if verified:
                        ctx = _ask_next_scene_data(
                            client, model, platform, goal, expected_outcome,
                            trajectory, max_completion_tokens,
                        )
                        scene_ctx.update(ctx)
                        success = True
                        logger.success(f"[{task_id}::{platform}] verified OK")
                        if suggestions_path:
                            append_verified_platform(suggestions_path, task_id, platform, ctx)
                    else:
                        logger.warning(f"[{task_id}::{platform}] verifier False: {verify_err}")
                        vsug = _analyze_verifier_failure(
                            verify_fn, verify_err, trajectory, goal, expected_outcome,
                            task_id, platform,
                            client, model, max_completion_tokens,
                            task_ops=task_ops,
                            schema_item=schemas[platform],
                            seed_db=seed_db,
                            sim_db=sim_db,
                            server_code=server_code,
                        )
                        if vsug.get("_skip"):
                            # Data + env + task all confirmed fine — the solver just
                            # didn't complete it. Not a defect: skip this round with NO
                            # suggestions; the task stays pending and is retried later.
                            logger.info(f"[{task_id}::{platform}] agent fault (task is solvable) — skipping this round")
                            return {"task_id": task_id, "status": "skipped", "suggestions": {}}
                        for k in ["verifier", "env", "data", "task_op", "goal_supplement", "outcome"]:
                            platform_suggestions[k].extend(vsug.get(k, []))
                else:
                    logger.info(f"[{task_id}::{platform}] incomplete or has issues — collecting suggestions, skipping verifier")

            # Save suggestions for this platform if any
            if any(v for v in platform_suggestions.values()):
                all_suggestions[platform] = platform_suggestions

            if not success:
                logger.error(f"[{task_id}::{platform}] not verified, aborting task")
                return {"task_id": task_id, "status": "failed", "suggestions": all_suggestions}

            platform_results[platform] = {
                "verify_fn": verify_fn,
                "function_name": function_name,
                "task_ops": task_ops,
                "outcome": expected_outcome,
                "read_only": is_read_only,
            }

        # All platforms in this scene done — merge their contexts into context_data for the next scene
        context_data.update(scene_ctx)

    return {
        "task_id": task_id,
        "status": "ok",
        "goal": goal,
        "platforms": platform_results,
        "suggestions": all_suggestions,
    }


# ── Run ────────────────────────────────────────────────────────────────────────

def _all_platforms_verified(task: dict, all_verified_platforms: dict) -> bool:
    vp = all_verified_platforms.get(task["task_id"], {})
    platforms = list(dict.fromkeys(
        p for scene in task.get("metadata", {}).get("scene_platforms", []) for p in scene
    ))
    return bool(platforms) and all(p in vp for p in platforms)


def load_pending_tasks(args: PipelineConfig, in_batch_ids: set[str]) -> list[dict]:
    """Return tasks that still need verification, excluding done and in-batch ones."""
    tasks = load_tasks(args.tasks_output)
    verifiers_gen = load_verifiers_gen(args.verifier_gen_output)
    seeded = load_seeded_task_ids(args.data_records)
    all_verified_platforms = load_verified_platforms(args.verified_platforms_output)

    pending = [
        t for t in tasks
        if t["task_id"] in seeded
        and t["task_id"] in verifiers_gen
        and t["task_id"] not in in_batch_ids
        and not _all_platforms_verified(t, all_verified_platforms)
    ]
    random.shuffle(pending)
    return pending


def process_batch(
    args: PipelineConfig,
    tasks: list[dict],
    *,
    on_task_file: Callable[[str], None] | None = None,
) -> int:
    """Run a batch of tasks; write one {task_id}.json per task that has suggestions.

    Calls on_task_file(file_path) immediately after each file is written.
    Returns the number of files written.
    """
    schemas = load_schemas(args.schemas_output)
    verifiers_gen = load_verifiers_gen(args.verifier_gen_output)
    envs = load_envs(args.envs_output)
    all_verified_platforms = load_verified_platforms(args.verified_platforms_output)
    task_supplements = load_task_supplements(args.task_supplements_output)
    suggestions_dir = Path(args.verified_suggestions_dir)

    # One LLMClient per worker thread — a shared client's single httpx connection
    # pool is not safe under this concurrency and intermittently raises
    # "'_GeneratorContextManager' object has no attribute 'args'". Thread-local
    # clients give each thread its own pool while still reusing connections.
    _local = threading.local()

    def _client() -> LLMClient:
        c = getattr(_local, "client", None)
        if c is None:
            c = LLMClient(api_key=args.api_key, base_url=args.base_url, aws_region=args.aws_region)
            _local.client = c
        return c

    def _process(task: dict) -> dict:
        task_id = task["task_id"]
        return process_task(
            _client(), args.model, args.gen_model,
            task, schemas, verifiers_gen,
            args.databases_dir,
            envs, args.max_retries, args.max_completion_tokens,
            args.agent_run_iterations_multiplier,
            suggestions_path=args.verified_platforms_output,
            verified_platforms=all_verified_platforms.get(task_id),
            task_supplement=task_supplements.get(task_id),
        )

    written = 0
    success = failed = 0
    with ThreadPoolExecutor(max_workers=args.fix_batch_size) as executor:
        futures = {executor.submit(_process, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Verifier step"):
            entry = future.result()
            if entry.get("suggestions"):
                task_id = entry["task_id"]
                file_path = suggestions_dir / f"{task_id}.json"
                tmp_path = str(file_path) + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump({"task_id": task_id, "suggestions": entry["suggestions"]},
                              f, ensure_ascii=False)
                os.replace(tmp_path, file_path)
                written += 1
                if on_task_file:
                    on_task_file(str(file_path))
            if entry["status"] == "ok":
                success += 1
            else:
                failed += 1

    logger.info(f"Batch done. {success} OK, {failed} failed, {written} with suggestions.")
    return written


def run(args: PipelineConfig) -> None:
    """Standalone run: process all pending tasks at once (no batching)."""
    pending = load_pending_tasks(args, in_batch_ids=set())
    if not pending:
        logger.success("All tasks already processed.")
        return
    logger.info(f"{len(pending)} tasks to process")
    process_batch(args, pending)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Simulate agents and run verifiers per task")
    parser.add_argument("--tasks_output", type=str, default=defaults.tasks_output)
    parser.add_argument("--schemas_output", type=str, default=defaults.schemas_output)
    parser.add_argument("--verifier_gen_output", type=str, default=defaults.verifier_gen_output)
    parser.add_argument("--envs_output", type=str, default=defaults.envs_output)
    parser.add_argument("--databases_dir", type=str, default=defaults.databases_dir)
    parser.add_argument("--data_records", type=str, default=defaults.data_records)
    parser.add_argument("--verifier_step_output", type=str, default=defaults.verifier_step_output)
    parser.add_argument("--model", type=str, default=defaults.model)
    parser.add_argument("--gen_model", type=str, default=defaults.gen_model)
    parser.add_argument("--api_key", type=str, default=defaults.api_key)
    parser.add_argument("--base_url", type=str, default=defaults.base_url)
    parser.add_argument("--concurrency", type=int, default=defaults.concurrency)
    parser.add_argument("--max_retries", type=int, default=defaults.max_retries)
    parser.add_argument("--max_completion_tokens", type=int, default=defaults.max_completion_tokens)
    parser.add_argument("--agent_run_max_iterations", type=int, default=defaults.agent_run_max_iterations)
    parsed = parser.parse_args()

    cfg = PipelineConfig()
    for k, v in vars(parsed).items():
        if hasattr(cfg, k) and v is not None:
            setattr(cfg, k, v)
    run(cfg)


if __name__ == "__main__":
    main()
