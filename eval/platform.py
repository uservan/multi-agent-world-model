"""Per-task platform runtime: start all of a task's servers, proxy HTTP, tear down.

Reuses the shared lifecycle helpers in utils.server. Each platform gets its own
writable copy of the seed DB (the 'final' DB), started on its own port. The seed
file stays untouched as the verifier's 'initial' DB.
"""
from __future__ import annotations

import os
import tempfile

from loguru import logger

from utils.server import copy_seed_db, http_call, start_server, stop_server, wait_for_server


class PlatformRuntime:
    """Manages the live servers + DBs for a single task run."""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._tmpdir = tempfile.mkdtemp(prefix=f"eval_{task_id[:8]}_")
        # platform -> {"url", "description", "seed_db", "final_db", "proc"}
        self.platforms: dict[str, dict] = {}

    def start(self, platform: str, resource: dict) -> bool:
        """Start one platform server against a fresh writable DB copy. Returns success."""
        safe = platform.lower().replace(" ", "_").replace("/", "_")
        final_db = os.path.join(self._tmpdir, f"{safe}.db")
        copy_seed_db(resource["seed_db"], final_db)

        proc, port = start_server(resource["server_path"], final_db)
        if not wait_for_server(port, timeout=25):
            stop_server(proc)
            logger.warning(f"[{self.task_id}::{platform}] server failed to start")
            return False

        self.platforms[platform] = {
            "url": f"http://127.0.0.1:{port}",
            "description": resource.get("description", ""),
            "seed_db": resource["seed_db"],
            "final_db": final_db,
            "proc": proc,
        }
        return True

    def call(self, base_url: str, method: str, path: str, params: dict | None = None, body: dict | None = None) -> str:
        """HTTP call with this task's X-Task-ID injected."""
        return http_call(base_url, self.task_id, method, path, params, body)

    def platform_map(self) -> dict[str, dict]:
        """{platform: {url, description}} for prompt building."""
        return {p: {"url": v["url"], "description": v["description"]} for p, v in self.platforms.items()}

    def stop_all(self) -> None:
        for v in self.platforms.values():
            try:
                stop_server(v["proc"])
            except Exception:
                pass

    def cleanup(self) -> None:
        self.stop_all()
        import shutil
        try:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            pass
