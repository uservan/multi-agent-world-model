"""Platform server + database lifecycle helpers.

Shared by the verifier pipeline and the eval framework: start/stop a generated
FastAPI platform server pointed at a per-task SQLite copy, wait for readiness,
copy seed databases, and make HTTP calls with the X-Task-ID header injected.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def wait_for_server(port: int, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _server_python() -> str:
    """Interpreter used to run a platform server.

    Defaults to sys.executable (correct for eval, which already runs under the venv).
    Under RL training it must be overridden via AWM_SERVER_PYTHON: there the caller is a
    ray worker running SYSTEM python, because megatron requires numpy 1.x while this
    project's venv carries numpy 2.x (mcp-agent needs >=2.1.3). System python has no
    fastapi/sqlalchemy, so spawning the server with sys.executable dies with
    "ModuleNotFoundError: No module named 'sqlalchemy'" and every platform fails to start.
    Pointing this at the venv python keeps the two interpreters cleanly separated.
    """
    return os.environ.get("AWM_SERVER_PYTHON") or sys.executable


def start_server(server_path: str, db_path: str) -> tuple[subprocess.Popen, int]:
    """Launch a generated FastAPI server bound to db_path on a free port."""
    port = get_free_port()
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["HOST"] = "127.0.0.1"
    env["DATABASE_PATH"] = f"sqlite:///{db_path}"
    proc = subprocess.Popen(
        [_server_python(), server_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc, port


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def copy_seed_db(seed_db: str, dst_db: str) -> str:
    """Copy a seed DB to dst (the agent's writable 'final' DB) and make it writable.

    The chmod is required: verifier sims set seed DBs read-only, and a copy can
    inherit that mode, which would make the platform server fail to write.
    """
    shutil.copy2(seed_db, dst_db)
    os.chmod(dst_db, 0o644)
    return dst_db


def http_call(
    base_url: str,
    task_id: str,
    method: str,
    path: str,
    params: dict | None = None,
    body: dict | None = None,
) -> str:
    """Make one HTTP request to a platform server with X-Task-ID injected."""
    try:
        url = base_url.rstrip("/") + "/" + path.lstrip("/")
        if params:
            if not isinstance(params, dict):
                return f"Error: 'params' must be a JSON object, got {type(params).__name__}"
            url += "?" + urllib.parse.urlencode(params)
        data = json.dumps(body).encode() if body else None
        req = urllib.request.Request(url, data=data, method=method.upper())
        req.add_header("X-Task-ID", task_id)
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}"
    except Exception as e:
        return f"Error: {e}"
