import os
import re
import json
import time
import signal
import shutil
import subprocess
import sys
import tempfile
import socket
from pathlib import Path

from loguru import logger
from utils.llm import LLMClient

from core.config import PipelineConfig

PYTHON_VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"


# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert FastAPI backend developer. Generate a single, fully self-contained Python file that implements a simplified platform API based on the provided spec and database schema.

Hard Requirements:
1. Implement EVERY endpoint from the spec with complete, working code. The database schema DDL is the ABSOLUTE source of truth — if the spec conflicts with the schema, follow the schema. NEVER invent or add columns/tables not present in the DDL.
2. Use SQLAlchemy ORM with declarative_base — NO raw SQL
3. Use Pydantic v2 (ConfigDict, not orm_mode)
4. Every endpoint handler MUST be async
5. Session lifecycle: session = SessionLocal() at start; for reads: query, read all needed values into plain Python variables, then session.close(); for writes: see rule 6 below. NEVER access ORM object attributes after session.close().
6. Write pattern (create/insert): ALWAYS compute the new id explicitly BEFORE inserting — `new_id = (session.query(func.max(Model.id)).scalar() or 0) + 1` — then set id=new_id on the new object, session.add(obj), session.commit(), session.close(), return {"id": new_id, "status": "..."}. NEVER use session.refresh() — it will fail because id columns have no PRIMARY KEY autoincrement.
   - If a table has a secondary string-formatted ID column (e.g. `job_id TEXT` storing values like `"mon_job_5521"`), NEVER compute it via `func.max(Model.job_id) + 1` — func.max on a String column returns a str, and `str + int` raises TypeError. Instead derive it from the numeric new_id: `job_id = f"<prefix>_{new_id}"` where the prefix matches the naming pattern shown in the schema or spec.
   - If any route uses a string-formatted ID column (e.g. `job_id`) as its path parameter instead of the integer `id`, the write response MUST include that string field too — e.g. return {"id": new_id, "job_id": f"<prefix>_{new_id}", "status": "..."} — so the caller knows the correct identifier for subsequent path calls.
7. Always check query.first() / query.all() results before use — if .first() can return None, guard with `if result is None` before accessing any attribute; never pass None to model_validate()
8. Path parameters in routes MUST appear as function parameters
9. app = FastAPI(...) MUST be defined BEFORE any route decorators
10. NO try/except, NO error handling, NO HTTPException, NO comments, NO placeholders
11. Include uvicorn entry point at the end reading HOST/PORT from environment variables
12. Booleans MUST be real bool fields in Pydantic responses — never use 0/1 for bool
13. There MUST be exactly one route function per (path, HTTP method) pair — NEVER register duplicate handlers
14. STRICTLY PROHIBITED: duplicate route registration, references to undefined models/fields/tables, schema-external helper tables, dynamic or introspective tricks to construct response models

CRITICAL — task_id and user_id isolation:
- Every endpoint MUST read task_id from the HTTP header "X-Task-ID" using: task_id: str = Header(..., alias="X-Task-ID", include_in_schema=False)
- Every READ query MUST filter by task_id: .filter(Model.task_id == task_id)
- Every WRITE must set task_id=task_id on the new row
- user_id is always 1 (the authenticated user); hardcode CURRENT_USER_ID = 1 as a module-level constant

CRITICAL — Atomicity enforcement:
- search/list endpoints: return ONLY the fields listed in the spec's "returns", query with .limit(MAX_RESULTS) where MAX_RESULTS is a module-level constant defined at the top of the file
- get_detail endpoints: return all fields for a single record by id, also filter by task_id
- check/validate endpoints: return ONLY {"eligible": bool, "result_value": ...}
- write endpoints: return ONLY {"id": ..., "status": "..."}
- get_status endpoints: return ONLY status-related fields
- NEVER return extra fields beyond what the spec specifies in "returns"

SQLAlchemy Rules:
- Every table in the schema has an `id INTEGER` column — always map it as `id = Column(Integer, primary_key=True)`. This is the ONLY primary key column; never declare any other column as primary_key=True.
- Column types MUST match Pydantic field types end-to-end: TEXT → `String` ORM / `str` Pydantic; INTEGER → `Integer` ORM / `int` Pydantic; REAL → `Float` ORM / `float` Pydantic. NEVER assign a value of the wrong Python type to a column (e.g. assigning an int to a String column or vice versa).
- Define ORM models for every table in the schema — ONLY the columns listed in the DDL, no extra columns, no invented fields
- Call Base.metadata.create_all(engine) after all models are defined
- NEVER name ORM attributes "metadata", "query", or "query_class" — use trailing underscore (e.g. metadata_) and map via Column("metadata", ...)
- When multiple FK paths exist between tables, ALWAYS specify relationship(..., foreign_keys=[...]) explicitly
- Import ORM helpers (declarative_base, relationship, sessionmaker) from sqlalchemy.orm only — do NOT import non-existent or internal SQLAlchemy symbols
- SQLite column type mapping: TEXT → String, INTEGER → Integer, REAL → Float (NEVER import or use `Real` — it does not exist in sqlalchemy's top-level namespace; always use Float for floating-point columns)

CRITICAL — Request body rules:
- POST / PUT / PATCH endpoints MUST accept all input fields via a Pydantic request body model — NEVER use Query(...) or query parameters for write operations
- Define a dedicated input Pydantic class (e.g. CreateVendorRequest, UpdateOrderRequest) for every POST/PUT/PATCH endpoint
- The only non-body parameters allowed on write endpoints are path parameters and the X-Task-ID header
- For GET/DELETE query parameters, declare each with FastAPI's Query and copy the param's spec description into it, e.g. `brand: Optional[str] = Query(None, description="<exact description from the spec>")` — so every query parameter's meaning shows up in /openapi.json. Path parameters likewise use `Path(..., description="<exact description from the spec>")`.
- POST request body Pydantic models MUST NOT include `id` as a field — the server always assigns the id internally via `new_id = (session.query(func.max(Model.id)).scalar() or 0) + 1`

Pydantic v2 Rules:
- Use model_config = ConfigDict(from_attributes=True) for ORM models
- Use Field for every body field, and set Field(..., description=...) to the EXACT description given for that param in the spec (so it appears in /openapi.json) — NEVER pass `example=` to Field() as it is deprecated in Pydantic v2; use `json_schema_extra={"example": ...}` if an example is needed
- Field names MUST NEVER clash with any Pydantic import name ("Field", "model_config", "model_validator", etc.) OR with any type name or class name used in the same module — if required, use a snake_case alias and Field(..., serialization_alias="original_name")
- Do NOT use Python reserved keywords (return, class, global, etc.) as field names or function parameters — use a trailing underscore and map via Field(..., alias="keyword")
- response_model must be a concrete Pydantic class, not a dynamic expression
- Endpoint return type annotations MUST be consistent with response_model — do NOT use Union[...] or Response types
- Use only: int, float, str, bool, Optional[T], List[T]
- Do NOT use Annotated or other advanced type tricks

Output Format:
- Return ONLY valid Python source code, no markdown fences, no JSON wrapper
- First line MUST be a Python import statement"""


USER_PROMPT_TEMPLATE = """Generate a complete FastAPI implementation for the following platform.

Platform: {name}
Python version: {python_version}

API Spec (implement every endpoint):
{spec_json}

Database Schema (DDL):
{schema_ddl}

Environment & DB Setup:
- Read DATABASE_PATH from environment variable; default to sqlite:///{name_lower}.db
- Read HOST from env (default "127.0.0.1"), PORT from env (default 8000, cast to int)
- Use SQLAlchemy ORM, create engine from DATABASE_PATH, call Base.metadata.create_all(engine)
- Define MAX_RESULTS = {max_results} as a module-level constant; use it for all search/list query limits
- Define CURRENT_USER_ID = 1 as a module-level constant (the authenticated user)

task_id isolation rules (STRICTLY enforce on every endpoint):
- Every endpoint reads task_id from header: task_id: str = Header(..., alias="X-Task-ID", include_in_schema=False)
- Every READ query filters by task_id: session.query(Model).filter(Model.task_id == task_id, ...)
- Every WRITE sets task_id=task_id on the new row

Atomicity rules (STRICTLY follow for every endpoint):
- search/list → query .limit(MAX_RESULTS), return ONLY the fields in spec's "returns"
- get_detail → fetch by id AND task_id, return all schema fields for that record
- check/validate → return {{"eligible": bool, "result_value": <one key value>}} only
- write → return {{"id": <new id>, "status": "<status string>"}} only
- status → return only status-related fields

End the file with:
if __name__ == "__main__":
    import uvicorn, os
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)"""


SEED_DATA_SYSTEM = """You are a database test engineer. Generate SQLite INSERT statements to seed exactly 3 rows per table.

Rules:
- Insert parent tables (no FK columns) before child tables that reference them
- Every row must have task_id = 'test'
- Use sequential IDs: 1, 2, 3 for every table — id values MUST be unique within each table, never repeat the same id in the same table
- FK columns referencing another table's id must use 1 (the first row of the parent)
- CRITICAL — column type determines value (check the DDL):
  - INTEGER or INT column → use integer 1 (NO quotes). NEVER use a string for an INTEGER column.
  - TEXT, VARCHAR, or CHAR column → use 'test_value' (with quotes)
  - REAL or FLOAT column → use 1.0
  - BOOLEAN column → use 1
  - DATE or DATETIME column → use '2024-01-01'
- Use INSERT OR IGNORE so duplicate id conflicts are skipped silently
- Output only valid SQLite INSERT OR IGNORE statements, no markdown, no comments, no explanations

Example:
INSERT OR IGNORE INTO categories (id, name, task_id) VALUES (1, 'test_value', 'test'), (2, 'test_value_2', 'test'), (3, 'test_value_3', 'test');
INSERT OR IGNORE INTO products (id, name, category_id, task_id) VALUES (1, 'test_value', 1, 'test'), (2, 'test_value_2', 1, 'test'), (3, 'test_value_3', 1, 'test');"""

SEED_DATA_USER_TEMPLATE = """Platform: {name}

Database Schema DDL:
{schema_ddl}

Generate INSERT statements to seed 3 rows per table in the correct FK order."""


TEST_CASE_SYSTEM = """You are a test engineer. Given a FastAPI spec and database schema, generate exactly one HTTP test request per endpoint.

Context: The database is pre-seeded with 3 rows per table. Every table has rows with id=1, id=2, id=3 and task_id="test".

Rules:
- POST/PUT/PATCH body: include ALL columns from the relevant table in the schema (realistic simple values)
- GET/search query params: include ALL filterable params mentioned in the spec
- Path params requiring an ID: use 1 (the generic `id` column is INTEGER and rows are seeded with id=1,2,3) — e.g. /vendors/1, /carts/1/items. This refers to the generic integer `id`, not the semantic TEXT id columns in the body.
- Do NOT use any placeholder syntax like __vendor_id__ — always use real values
- Do NOT include X-Task-ID in output — added automatically
- Do NOT include task_id in body or query_params
- CRITICAL: every body/query value MUST match the column's SQL type in the DDL — check each column's type before assigning a value:
    * TEXT / VARCHAR columns → JSON string, even for id-like fields (e.g. "gig_id": "1", "freelancer_id": "1", not 1)
    * INTEGER columns → JSON number (e.g. 1)
    * REAL / FLOAT / NUMERIC columns → JSON number (e.g. 1.0)
    * BOOLEAN (INTEGER 0/1) columns → 1 or 0
- Use simple values: TEXT → "test_value", INTEGER → 1, REAL → 1.0, booleans → 1, dates (TEXT) → "2024-01-01"

Output valid JSON array only, no markdown fences:
[
  {
    "method": "GET",
    "path": "/vendors/1",
    "query_params": {},
    "body": null
  },
  {
    "method": "GET",
    "path": "/vendors/search",
    "query_params": {"name": "test_value", "category": "tech"},
    "body": null
  },
  {
    "method": "POST",
    "path": "/vendors",
    "query_params": {},
    "body": {"name": "new_vendor", "category": "tech", "email": "v@test.com"}
  }
]"""

TEST_CASE_USER_TEMPLATE = """Platform: {name}

API Spec (all endpoints to test):
{spec_json}

Database Schema DDL (use column names for body/query fields):
{schema_ddl}

Generate exactly one test case per endpoint listed above. Use id=1 for any path param requiring an existing resource."""


ANALYZE_REVISION_SYSTEM = """You are a FastAPI code reviewer. Your job is to analyze an existing server implementation and a list of reported issues, then produce a precise change plan.

For each issue, identify exactly:
- Which class, function, or endpoint needs to change
- What the current (broken) code looks like
- What it should be changed to, and why

Be specific and concise. Do NOT output any Python code — output only a structured text analysis."""

ANALYZE_REVISION_USER = """Platform: {name}

Reported issues (from real agent simulation runs):
{issues}

Existing server code:
```python
{existing_code}
```

List the precise changes needed to fix each issue. No code output — describe the changes in plain text."""


APPLY_REVISION_USER = """Apply the following targeted changes to fix bugs in an existing FastAPI server. Output the complete corrected Python file.

Platform: {name}
Python version: {python_version}

API Spec (reference for endpoint contracts):
{spec_json}

Database Schema DDL (source of truth for tables and columns):
{schema_ddl}

Environment & DB Setup:
- Read DATABASE_PATH from environment variable; default to sqlite:///{name_lower}.db
- Read HOST from env (default "127.0.0.1"), PORT from env (default 8000, cast to int)
- Use SQLAlchemy ORM, create engine from DATABASE_PATH, call Base.metadata.create_all(engine)
- Define MAX_RESULTS = {max_results} as a module-level constant; use it for all search/list query limits
- Define CURRENT_USER_ID = 1 as a module-level constant (the authenticated user)

task_id isolation rules (STRICTLY enforce on every endpoint):
- Every endpoint reads task_id from header: task_id: str = Header(..., alias="X-Task-ID", include_in_schema=False)
- Every READ query filters by task_id: session.query(Model).filter(Model.task_id == task_id, ...)
- Every WRITE sets task_id=task_id on the new row

Atomicity rules (STRICTLY follow for every endpoint):
- search/list → query .limit(MAX_RESULTS), return ONLY the fields in spec's "returns"
- get_detail → fetch by id AND task_id, return all schema fields for that record
- check/validate → return {{"eligible": bool, "result_value": <one key value>}} only
- write → return {{"id": <new id>, "status": "<status string>"}} only
- status → return only status-related fields

End the file with:
if __name__ == "__main__":
    import uvicorn, os
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)

Change plan (apply exactly these fixes):
{analysis}

Existing server code to modify:
```python
{existing_code}
```"""


ERROR_SUMMARY_SYSTEM = """You are a FastAPI expert. Analyze an error from generated FastAPI code and provide concise guidance to fix it.

Format your response as:
[Error Cause]: what went wrong
[Problematic Code]:
```python
# the specific code that caused the error
```
[Guidance]: exactly how to fix or avoid this

Keep your response under 800 tokens."""


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_specs(path: str) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("name"):
                    specs[item["name"]] = item
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(specs)} platform specs from {path}")
    return specs


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


def load_existing_results(output_path: str) -> dict[str, dict]:
    existing: dict[str, dict] = {}
    if not os.path.exists(output_path):
        return existing
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if item.get("name") and item.get("server_path"):
                    existing[item["name"]] = item
            except json.JSONDecodeError:
                pass
    logger.info(f"Loaded {len(existing)} existing results from {output_path}")
    return existing


def _read_server_code(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


def append_result(output_path: str, item: dict) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _replace_result(output_path: str, item: dict) -> None:
    """Replace the existing entry for item['name'] in-place; append if not found."""
    name = item.get("name")
    if not name or not os.path.exists(output_path):
        append_result(output_path, item)
        return
    kept_lines: list[str] = []
    replaced = False
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                existing = json.loads(stripped)
                if existing.get("name") == name:
                    kept_lines.append(json.dumps(item, ensure_ascii=False))
                    replaced = True
                else:
                    kept_lines.append(stripped)
            except json.JSONDecodeError:
                kept_lines.append(stripped)
    if not replaced:
        kept_lines.append(json.dumps(item, ensure_ascii=False))
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")
    os.replace(tmp_path, output_path)


def _format_schema_ddl(schema_item: dict) -> str:
    return "\n".join(t.get("ddl", "") for t in schema_item.get("schemas", []))


def _parse_column_types(schema_item: dict) -> dict[str, str]:
    """Parse DDL into {column_name: type_category} where category is 'text', 'int', or 'real'.
    On a column-name conflict across tables, first-seen wins (column naming is usually consistent)."""
    col_types: dict[str, str] = {}
    for t in schema_item.get("schemas", []):
        ddl = t.get("ddl", "")
        m = re.search(r"\((.*)\)", ddl, re.DOTALL)
        if not m:
            continue
        for col_def in m.group(1).split(","):
            parts = col_def.strip().split()
            if len(parts) < 2:
                continue
            col_name = parts[0].strip("`\"'")
            sql_type = parts[1].upper()
            if col_name in col_types:
                continue
            if any(k in sql_type for k in ("INT",)):
                col_types[col_name] = "int"
            elif any(k in sql_type for k in ("REAL", "FLOA", "DOUB", "NUMER", "DECI")):
                col_types[col_name] = "real"
            elif any(k in sql_type for k in ("TEXT", "CHAR", "CLOB", "VARCHAR")):
                col_types[col_name] = "text"
    return col_types


def _coerce_value(value, type_cat: str):
    """Coerce a single value to match its column's SQL type category."""
    if value is None or isinstance(value, bool):
        return value
    if type_cat == "text" and isinstance(value, (int, float)):
        return str(value)
    if type_cat == "int" and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    if type_cat == "real" and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def _coerce_test_cases(test_cases: list[dict], col_types: dict[str, str]) -> list[dict]:
    """Fix body/query values whose JSON type doesn't match the schema column type
    (e.g. a TEXT id field given integer 1 → "1"), so the server's validation passes."""
    for tc in test_cases:
        for field in ("body", "query_params"):
            obj = tc.get(field)
            if not isinstance(obj, dict):
                continue
            for k, v in list(obj.items()):
                if k in col_types:
                    obj[k] = _coerce_value(v, col_types[k])
    return test_cases


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout: float = 30.0) -> bool:
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


# ── DB + test helpers ─────────────────────────────────────────────────────────

def _create_db_from_ddl(db_path: str, schema_item: dict, seed_sql: str = "") -> None:
    import sqlite3
    ddl_sql = _format_schema_ddl(schema_item)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(ddl_sql)
        # Add UNIQUE index on id for each table so INSERT OR IGNORE actually
        # prevents duplicate id rows (no PRIMARY KEY needed).
        table_names = re.findall(
            r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?['\"`]?(\w+)['\"`]?",
            ddl_sql,
            re.IGNORECASE,
        )
        for table in table_names:
            conn.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS _ux_{table}_id ON {table}(id)"
            )
        conn.commit()
        if seed_sql:
            conn.executescript(seed_sql)
        conn.commit()
    finally:
        conn.close()


def _generate_seed_sql(
    client: "LLMClient",
    model: str,
    name: str,
    schema_item: dict,
    max_completion_tokens: int = 16384,
) -> str:
    schema_ddl = _format_schema_ddl(schema_item)
    messages = [
        {"role": "system", "content": SEED_DATA_SYSTEM},
        {"role": "user", "content": SEED_DATA_USER_TEMPLATE.format(name=name, schema_ddl=schema_ddl)},
    ]
    try:
        content = client.complete(model, messages, max_completion_tokens).strip()
        content = re.sub(r"^```[a-z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content).strip()
        return content
    except Exception as e:
        logger.warning(f"[{name}] Failed to generate seed SQL: {e}")
        return ""


def _generate_test_cases(
    client: "LLMClient",
    model: str,
    name: str,
    spec_item: dict,
    schema_item: dict,
    max_completion_tokens: int = 16384,
) -> list[dict]:
    spec_json = json.dumps(spec_item.get("endpoints", []), ensure_ascii=False, indent=2)
    schema_ddl = _format_schema_ddl(schema_item)
    user_content = TEST_CASE_USER_TEMPLATE.format(
        name=name,
        spec_json=spec_json,
        schema_ddl=schema_ddl,
    )
    messages = [
        {"role": "system", "content": TEST_CASE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        content = client.complete(model, messages, max_completion_tokens).strip()
        content = re.sub(r"^```[a-z]*\n?", "", content)
        content = re.sub(r"\n?```$", "", content).strip()
        test_cases = json.loads(content)
        # Deterministically coerce body/query values to match schema column types,
        # so id-like TEXT fields the LLM filled with integer 1 become "1" (avoids 422s).
        return _coerce_test_cases(test_cases, _parse_column_types(schema_item))
    except Exception as e:
        logger.warning(f"[{name}] Failed to generate test cases: {e}")
        return []


def _run_endpoint_tests(port: int, test_cases: list[dict]) -> list[str]:
    import urllib.request as ureq
    import urllib.parse
    import urllib.error

    base_url = f"http://127.0.0.1:{port}"
    errors: list[str] = []

    method_order = {"POST": 0, "PUT": 1, "PATCH": 1, "GET": 2, "DELETE": 3}
    sorted_cases = sorted(test_cases, key=lambda x: method_order.get(x.get("method", "GET").upper(), 2))

    for tc in sorted_cases:
        method = tc.get("method", "GET").upper()
        path = tc.get("path", "/")
        query_params = tc.get("query_params") or {}
        body = tc.get("body")

        url = base_url + path
        if query_params:
            url += "?" + urllib.parse.urlencode(query_params)

        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = ureq.Request(url, data=data, method=method)
        req.add_header("X-Task-ID", "test")
        if data:
            req.add_header("Content-Type", "application/json")

        try:
            with ureq.urlopen(req, timeout=10) as resp:
                status = resp.status
                resp_body = resp.read().decode("utf-8", errors="replace")
        except ureq.HTTPError as e:
            status = e.code
            resp_body = e.read().decode("utf-8", errors="replace")
        except Exception as e:
            errors.append(f"{method} {tc['path']} → connection error: {str(e)[:300]}")
            continue

        if status >= 500:
            errors.append(f"{method} {tc['path']} → {status}: {resp_body[:500]}")
        elif status == 422 and method in ("POST", "PUT", "PATCH"):
            try:
                detail = json.loads(resp_body).get("detail", [])
                query_missing = [d for d in detail if isinstance(d, dict) and d.get("loc", [None])[0] == "query"]
            except Exception:
                query_missing = []
            if query_missing:
                missing_fields = [d["loc"][-1] for d in query_missing]
                errors.append(
                    f"{method} {tc['path']} → 422: server used Query() for write fields {missing_fields} — "
                    f"must use a Pydantic request body model instead of Query() for POST/PUT/PATCH"
                )
            else:
                logger.debug(f"[endpoint test] {method} {tc['path']} → 422 (test data issue): {resp_body[:300]}")
        elif status >= 400:
            logger.debug(f"[endpoint test] {method} {tc['path']} → {status} (test data issue): {resp_body[:300]}")

    return errors


# ── Server startup test ────────────────────────────────────────────────────────

def test_server(
    name: str,
    code: str,
    client: "LLMClient",
    model: str,
    spec_item: dict,
    schema_item: dict,
    max_completion_tokens: int = 16384,
) -> tuple[bool, str]:
    """Start server with DDL-seeded DB, run per-endpoint tests, return (success, error_output)."""
    port = _get_free_port()
    safe_name = name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    temp_dir = tempfile.mkdtemp(prefix=f"env_test_{safe_name}_")
    temp_py = os.path.join(temp_dir, "server.py")
    db_path = os.path.join(temp_dir, f"{safe_name}.db")

    try:
        seed_sql = _generate_seed_sql(client, model, name, schema_item, max_completion_tokens)
        try:
            _create_db_from_ddl(db_path, schema_item, seed_sql)
        except Exception as e:
            logger.warning(f"[{name}] DB setup failed: {e}")

        with open(temp_py, "w", encoding="utf-8") as f:
            f.write(code)

        env = os.environ.copy()
        env["PORT"] = str(port)
        env["DATABASE_PATH"] = f"sqlite:///{db_path}"

        proc = subprocess.Popen(
            [sys.executable, temp_py],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
            env=env,
        )

        time.sleep(2)

        if proc.poll() is not None:
            stdout, _ = proc.communicate()
            return False, (stdout or "")

        if not _wait_for_server(port, timeout=20):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, _ = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, _ = proc.communicate()
            return False, (stdout or "")

        # Generate test cases once, run them
        test_cases = _generate_test_cases(client, model, name, spec_item, schema_item, max_completion_tokens)
        endpoint_errors: list[str] = []
        if test_cases:
            endpoint_errors = _run_endpoint_tests(port, test_cases)

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            server_stdout, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            server_stdout, _ = proc.communicate()

        if endpoint_errors:
            error_msg = "Endpoint test failures:\n" + "\n".join(endpoint_errors)
            if server_stdout and server_stdout.strip():
                error_msg += "\n\nServer process output (tracebacks):\n" + server_stdout.strip()
            return False, error_msg

        return True, ""

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ── LLM calls ─────────────────────────────────────────────────────────────────

def summarize_error(client: LLMClient, model: str, error_output: str, code: str, max_completion_tokens: int = 16384) -> str:
    messages = [
        {"role": "system", "content": ERROR_SUMMARY_SYSTEM},
        {"role": "user", "content": f"Error output:\n{error_output}\n\nGenerated code:\n```python\n{code[:6000]}\n```"},
    ]
    try:
        return client.complete(model, messages, max_completion_tokens)
    except Exception:
        return error_output


def generate_code(
    client: LLMClient,
    model: str,
    name: str,
    spec_item: dict,
    schema_item: dict,
    error_summaries: list[str] | None = None,
    max_completion_tokens: int = 16384,
    max_results: int = 5,
) -> str:
    spec_json = json.dumps(spec_item.get("endpoints", []), ensure_ascii=False, indent=2)
    schema_ddl = _format_schema_ddl(schema_item)

    user_content = USER_PROMPT_TEMPLATE.format(
        name=name,
        python_version=PYTHON_VERSION,
        spec_json=spec_json,
        schema_ddl=schema_ddl,
        name_lower=name.lower().replace(" ", "_").replace("/", "_"),
        max_results=max_results,
    )

    if error_summaries:
        errors_block = "\n\n".join(f"Previous error #{i+1}:\n{s}" for i, s in enumerate(error_summaries))
        user_content += f"\n\nYou MUST avoid these errors from previous attempts:\n{errors_block}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return client.complete(model, messages, max_completion_tokens).strip()


def revise_code(
    client: LLMClient,
    model: str,
    name: str,
    existing_code: str,
    env_suggestions: list[str],
    max_completion_tokens: int = 16384,
) -> str:
    issues_block = "\n".join(f"- {s}" for s in env_suggestions)
    messages = [
        {"role": "system", "content": ANALYZE_REVISION_SYSTEM},
        {"role": "user", "content": ANALYZE_REVISION_USER.format(
            name=name,
            issues=issues_block,
            existing_code=existing_code,
        )},
    ]
    return client.complete(model, messages, max_completion_tokens).strip()


def apply_revision(
    client: LLMClient,
    model: str,
    name: str,
    spec_item: dict,
    schema_item: dict,
    existing_code: str,
    analysis: str,
    error_summaries: list[str] | None = None,
    max_completion_tokens: int = 16384,
    max_results: int = 5,
) -> str:
    apply_user = APPLY_REVISION_USER.format(
        name=name,
        python_version=PYTHON_VERSION,
        spec_json=json.dumps(spec_item.get("endpoints", []), ensure_ascii=False, indent=2),
        schema_ddl=_format_schema_ddl(schema_item),
        name_lower=name.lower().replace(" ", "_").replace("/", "_"),
        max_results=max_results,
        analysis=analysis,
        existing_code=existing_code,
    )
    if error_summaries:
        errors_block = "\n\n".join(f"Previous fix attempt error #{i+1}:\n{s}" for i, s in enumerate(error_summaries))
        apply_user += f"\n\nAlso avoid these errors from previous fix attempts:\n{errors_block}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": apply_user},
    ]
    return client.complete(model, messages, max_completion_tokens).strip()


# ── Per-platform processing ────────────────────────────────────────────────────

def process_platform(
    client: LLMClient,
    model: str,
    name: str,
    spec_item: dict,
    schema_item: dict,
    servers_dir: str,
    max_retries: int,
    max_completion_tokens: int = 16384,
    max_results: int = 5,
    existing_code: str | None = None,
    env_suggestions: list[str] | None = None,
) -> dict | None:
    error_summaries: list[str] = []
    is_revision = bool(existing_code and env_suggestions)
    revision_analysis: str = ""
    current_base_code: str = existing_code or ""
    if is_revision:
        logger.info(f"[{name}] Revising existing code with {len(env_suggestions)} env suggestions")
        try:
            revision_analysis = revise_code(client, model, name, current_base_code, env_suggestions, max_completion_tokens)
            logger.debug(f"[{name}] Revision analysis done")
        except Exception as e:
            logger.warning(f"[{name}] Revision analysis failed: {e}")

    # max_retries = TOTAL attempts (env_max_retries in config); 1 = single shot, no retry.
    for attempt in range(1, max_retries + 1):
        try:
            if is_revision:
                code = apply_revision(client, model, name, spec_item, schema_item, current_base_code,
                                      revision_analysis, error_summaries if error_summaries else None,
                                      max_completion_tokens, max_results)
            else:
                code = generate_code(client, model, name, spec_item, schema_item,
                                     error_summaries if error_summaries else None, max_completion_tokens, max_results)
        except Exception as e:
            logger.warning(f"[{name}] LLM call failed (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            continue

        if code and not code.startswith(("import", "from")):
            # Try to extract from markdown code block
            m = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
            if m:
                 code = m.group(1).strip()

        if not code or not code.startswith(("import", "from")):
            logger.warning(f"[{name}] Invalid code format (attempt {attempt}), {len(code)}")
            if attempt < max_retries:
                error_summaries.append("The response must be pure Python code starting with an import statement. No markdown, no JSON wrapper.")
            continue

        success, error_output = test_server(name, code, client, model, spec_item, schema_item, max_completion_tokens)

        if success:
            server_path = os.path.join(servers_dir, f"{name.lower().replace(' ', '_').replace('/', '_')}_server.py")
            Path(servers_dir).mkdir(parents=True, exist_ok=True)
            tmp_path = server_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(code)
            os.replace(tmp_path, server_path)
            logger.success(f"[{name}] Server generated and tested OK (attempt {attempt})")
            return {
                "name": name,
                "server_path": server_path,
                "status": "ok",
                "ops_fingerprint": spec_item.get("ops_fingerprint", ""),
            }

        logger.warning(f"[{name}] Server test failed (attempt {attempt}):\n{error_output}")
        if attempt < max_retries:
            summary = summarize_error(client, model, error_output, code, max_completion_tokens)
            error_summaries.append(summary)
            if is_revision and code:
                current_base_code = code  # next attempt builds on latest generated code
            time.sleep(2)

    logger.error(f"[{name}] All {max_retries} attempts failed — skipping.")
    return None


# ── Run ────────────────────────────────────────────────────────────────────────

