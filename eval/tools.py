"""Tool executors for eval agents.

Tool calls are parsed by utils.agent_io.parse_tool_calls into
{"name": <tool>, "arguments": {...}}. The http executor is shared by every actor
(single agent, orchestrator, sub-agents); spawn_subagent / get_queue_status /
get_task_info / wait_task are orchestrator-only and live in multi_agent.py.

OpenAPI discovery is TWO-STEP (added 2026-07-14) to keep full specs out of the
agent's context — a single platform's raw openapi.json can be 150k+ chars, and a
single agent that loads several at once blows past the model context window:

  1. GET /openapi.json  (no params)  -> a compact INDEX: one line per endpoint
     (METHOD /path — summary). No parameters, no schemas.
  2. GET /openapi.json  with params {"paths": ["/a", "/b"]} (or {"path": "/a"})
     -> full DETAIL (path/query params + request body fields + 200 response
     fields, with types & required) for ONLY the requested endpoints.

The interception happens here in the http executor, so we do NOT touch the
platform servers — we still fetch the server's real openapi.json, then reshape it.
"""
from __future__ import annotations

import json

from eval.platform import PlatformRuntime


def _is_openapi(path: str) -> bool:
    return path.strip().strip("/").lower() == "openapi.json"


def _type_str(sch: object) -> str:
    """Short human-readable type for an openapi schema node (handles $ref/anyOf/array)."""
    if not isinstance(sch, dict):
        return "any"
    if "$ref" in sch:
        return str(sch["$ref"]).split("/")[-1]
    for key in ("anyOf", "oneOf"):
        if key in sch and isinstance(sch[key], list):
            opts = [o for o in sch[key] if not (isinstance(o, dict) and o.get("type") == "null")]
            if len(opts) == 1:
                return _type_str(opts[0])
            return "|".join(_type_str(o) for o in opts) or "any"
    if "allOf" in sch and isinstance(sch["allOf"], list) and sch["allOf"]:
        return _type_str(sch["allOf"][0])
    t = sch.get("type")
    if t == "array":
        return f"[{_type_str(sch.get('items', {}))}]"
    if t is None:
        return "object" if "properties" in sch else "any"
    return str(t)


def _obj_fields(schema: object, components: dict) -> list[tuple[str, str, bool]]:
    """Resolve a (possibly $ref) object schema -> [(field, type, required), ...]."""
    if isinstance(schema, dict) and "$ref" in schema:
        name = str(schema["$ref"]).split("/")[-1]
        schema = components.get(name, {})
    if not isinstance(schema, dict):
        return []
    # array of objects -> describe the item
    if schema.get("type") == "array":
        return _obj_fields(schema.get("items", {}), components)
    props = schema.get("properties", {})
    req = set(schema.get("required", []) or [])
    out = []
    for k, v in props.items():
        out.append((k, _type_str(v), k in req))
    return out


def _index_view(spec: dict) -> str:
    """L1: compact endpoint index — one line per (method, path)."""
    title = (spec.get("info") or {}).get("title", "API")
    paths = spec.get("paths") or {}
    lines = []
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue
        for m, info in methods.items():
            if not isinstance(info, dict):
                continue
            summary = info.get("summary") or info.get("description") or ""
            summary = summary.strip().splitlines()[0][:100] if summary else ""
            lines.append(f"{m.upper():6} {path}" + (f" — {summary}" if summary else ""))
    header = (
        f'API index for "{title}" — {len(lines)} endpoints. This is a directory only. '
        'For the exact parameters / request body / response of the endpoints you plan to call, '
        'request their details with: GET /openapi.json  params {"paths": ["/first", "/second"]} '
        '(or {"path": "/single"}).'
    )
    return header + "\n" + "\n".join(lines)


def _detail_view(spec: dict, wanted: list[str]) -> str:
    """L2: full parameter/body/response detail for the requested endpoint paths."""
    components = (spec.get("components") or {}).get("schemas", {}) or {}
    paths = spec.get("paths") or {}
    # normalize: ensure a leading slash
    want = []
    for w in wanted:
        w = str(w).strip()
        if w and not w.startswith("/"):
            w = "/" + w
        want.append(w)

    blocks = []
    for w in want:
        methods = paths.get(w)
        if not isinstance(methods, dict):
            blocks.append(f"{w}: not found. Call GET /openapi.json (no params) for the endpoint index.")
            continue
        for m, info in methods.items():
            if not isinstance(info, dict):
                continue
            summary = (info.get("summary") or "").strip()
            lines = [f"{m.upper()} {w}" + (f" — {summary}" if summary else "")]
            # path + query params
            pin: dict[str, list[str]] = {}
            for p in (info.get("parameters") or []):
                if not isinstance(p, dict):
                    continue
                loc = p.get("in", "query")
                nm = p.get("name", "?")
                ty = _type_str(p.get("schema", {}))
                rq = " (required)" if p.get("required") else ""
                pin.setdefault(loc, []).append(f"{nm}: {ty}{rq}")
            if pin.get("path"):
                lines.append("  path params: " + ", ".join(pin["path"]))
            if pin.get("query"):
                lines.append("  query params: " + ", ".join(pin["query"]))
            # request body
            rb = info.get("requestBody") or {}
            body_schema = ((rb.get("content") or {}).get("application/json") or {}).get("schema")
            if body_schema is not None:
                fields = _obj_fields(body_schema, components)
                if fields:
                    lines.append("  request body (JSON):")
                    for nm, ty, rq in fields:
                        lines.append(f"    {nm}: {ty}" + (" (required)" if rq else ""))
                else:
                    lines.append("  request body (JSON): {}")
            # 200 response fields (one level)
            resp = (info.get("responses") or {}).get("200") or {}
            resp_schema = ((resp.get("content") or {}).get("application/json") or {}).get("schema")
            if resp_schema is not None:
                fields = _obj_fields(resp_schema, components)
                if fields:
                    lines.append("  → response 200: " + ", ".join(f"{nm}: {ty}" for nm, ty, _ in fields))
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _openapi_response(raw: str, ctrl: dict) -> str:
    """Reshape a raw openapi.json body into an index (default) or per-endpoint detail."""
    try:
        spec = json.loads(raw)
    except Exception:
        # not JSON (e.g. an HTTP error string) — return as-is so the model sees the real error
        return raw
    if not isinstance(spec, dict) or "paths" not in spec:
        return raw
    # detail mode if the caller named specific endpoint(s)
    wanted = ctrl.get("paths")
    if wanted is None:
        single = ctrl.get("path")
        if isinstance(single, str) and single.strip():
            wanted = [single]
    if isinstance(wanted, str):
        wanted = [wanted]
    if isinstance(wanted, list) and wanted:
        return _detail_view(spec, wanted)
    return _index_view(spec)


def make_http_executor(runtime: PlatformRuntime):
    """Returns an async executor that handles the 'http' tool against the live servers."""
    async def execute(call: dict) -> str:
        if call.get("name") != "http":
            return f"Error: unknown tool '{call.get('name')}'. Available: http."
        args = call.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return "Error: http arguments must be a JSON object"
        method = str(args.get("method", "GET")).upper()
        base_url = args.get("base_url", "")
        path = args.get("path", "/")
        params = args.get("params") or {}
        if not base_url:
            return "Error: http call missing base_url"

        # ── OpenAPI discovery (two-step): intercept, fetch the real spec, return
        #    a compact index by default or per-endpoint detail when params name paths ──
        if method == "GET" and _is_openapi(path):
            raw = runtime.call(base_url, "GET", "/openapi.json")  # params are control, not query
            ctrl = params if isinstance(params, dict) else {}
            return f"[GET /openapi.json] {_openapi_response(raw, ctrl)}"

        # GET/DELETE → query params; otherwise → JSON body
        if method in ("GET", "DELETE"):
            resp = runtime.call(base_url, method, path, params=params)
        else:
            resp = runtime.call(base_url, method, path, body=params)
        return f"[{method} {path}] {resp}"

    return execute
