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


def start_server(server_path: str, db_path: str) -> tuple[subprocess.Popen, int]:
    """Launch a generated FastAPI server bound to db_path on a free port."""
    port = get_free_port()
    env = os.environ.copy()
    env["PORT"] = str(port)
    env["HOST"] = "127.0.0.1"
    env["DATABASE_PATH"] = f"sqlite:///{db_path}"
    proc = subprocess.Popen(
        [sys.executable, server_path],
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
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method.upper())
    req.add_header("X-Task-ID", task_id)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}"
    except Exception as e:
        return f"Error: {e}"
