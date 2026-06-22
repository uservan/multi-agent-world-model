from __future__ import annotations
import threading


class PlatformLocks:
    """Thread-safe per-platform lock registry."""

    def __init__(self) -> None:
        self._meta = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def get(self, platform: str) -> threading.Lock:
        with self._meta:
            if platform not in self._locks:
                self._locks[platform] = threading.Lock()
            return self._locks[platform]


class FixLocks:
    """Shared locks for the fix pipeline workers."""

    def __init__(self) -> None:
        self.goal = threading.Lock()       # task_supplements.jsonl
        self.verifier = threading.Lock()   # verifiers_gen.jsonl
        self.data = PlatformLocks()        # {platform}.db
        self.env = PlatformLocks()         # servers/{platform}.py
