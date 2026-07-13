"""Eval configuration.

Two commands, two simple cases (see eval_main.py):

  --init <template.yml>   NEW run. Creates a result folder under eval_root, copies
                          the yml in as config.yml, and locks the frozen fields with
                          an md5 stored as `frozen_md5`. Then runs.

  --config <result_dir>   RESUME an existing run. Reads <result_dir>/config.yml and
                          verifies frozen_md5 — so you may edit cost_config (and data
                          paths) afterwards, but NOT the model/run fields.

Ready-made templates live in eval/config/: single.yml, multi_same.yml, multi_diff.yml.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

# Fields locked by the md5. Editing any of these after --init breaks resume.
# cost_config, data paths and eval_root are intentionally NOT locked.
_FROZEN_FIELDS = (
    "mode", "orch_model", "orch_api_key", "orch_base_url",
    "sub_model", "sub_api_key", "sub_base_url",
    "n", "base_seed", "temperature", "max_completion_tokens",
    "orch_llm_params", "sub_llm_params", "min_completion_tokens",
    "max_turns", "sub_max_turns", "max_concurrent", "max_queue",
    "orch_style",
)


def _sanitize(name: str) -> str:
    name = (name or "").rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _frozen_md5(data: dict) -> str:
    """Stable md5 over the frozen fields only."""
    frozen = {k: data.get(k) for k in _FROZEN_FIELDS}
    return hashlib.md5(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


@dataclass
class EvalConfig:
    # frozen
    mode: str = "single"
    orch_model: str = ""
    orch_api_key: str | None = None
    orch_base_url: str | None = None
    orch_input_cost: float | None = None     # editable (not locked)
    orch_output_cost: float | None = None
    sub_model: str = ""
    sub_api_key: str | None = None
    sub_base_url: str | None = None
    sub_input_cost: float | None = None
    sub_output_cost: float | None = None
    n: int = 3
    base_seed: int = 0
    temperature: float = 1.0
    max_completion_tokens: int = 1024 * 16
    # Per-role LLM passthrough params (temperature/top_p/extra_body...). orch and sub may be
    # different models needing different keys, e.g. Kimi instant mode:
    #   {"extra_body": {"chat_template_kwargs": {"thinking": False}}}
    orch_llm_params: dict = field(default_factory=dict)
    sub_llm_params: dict = field(default_factory=dict)
    # Floor on max_completion_tokens so reasoning doesn't eat a small cap → empty content (None = off).
    min_completion_tokens: int | None = None
    max_turns: int = 100             # orchestrator / single-agent turn cap
    sub_max_turns: int = 30          # sub-agent turn cap (multi mode only)
    max_concurrent: int = 4
    max_queue: int = 16
    # Orchestrator delegation bias (multi mode only): neutral | delegate | solo
    orch_style: str = "neutral"
    # runtime / editable
    # data_root: dataset root dir (e.g. outputs/eval_gen_claude). When set, the four
    # paths below are derived from it (root/verified/task_final.jsonl, root/platforms.jsonl,
    # root/envs.jsonl, root/databases) unless explicitly given in the yml, and server
    # paths from envs.jsonl are re-rooted to root/servers/<basename>.
    data_root: str = ""
    task_final: str = "outputs/generated/verified/task_final.jsonl"
    platforms_input: str = "outputs/platforms.jsonl"
    envs_input: str = "outputs/generated/envs.jsonl"
    databases_dir: str = "outputs/generated/databases"
    eval_root: str = "outputs/eval"
    # resolved at runtime
    run_dir: str = ""

    @property
    def servers_dir(self) -> str:
        return str(Path(self.data_root) / "servers") if self.data_root else ""

    # ── Build from a plain dict (yml contents) ─────────────────────────────────
    @classmethod
    def _from_dict(cls, d: dict) -> "EvalConfig":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        cfg = cls(**{k: v for k, v in d.items() if k in fields})
        if not cfg.mode or not cfg.orch_model:
            raise ValueError("config needs at least 'mode' and 'orch_model'")
        if cfg.mode == "multi" and not cfg.sub_model:
            raise ValueError("mode 'multi' requires 'sub_model'")
        if cfg.data_root:
            root = Path(cfg.data_root)
            if "task_final" not in d:
                cfg.task_final = str(root / "verified" / "task_final.jsonl")
            if "platforms_input" not in d:
                cfg.platforms_input = str(root / "platforms.jsonl")
            if "envs_input" not in d:
                cfg.envs_input = str(root / "envs.jsonl")
            if "databases_dir" not in d:
                cfg.databases_dir = str(root / "databases")
        return cfg

    # ── Case 1: --init (new run) ───────────────────────────────────────────────
    @classmethod
    def init_run(cls, template_yml: str) -> "EvalConfig":
        with open(template_yml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        cfg = cls._from_dict(data)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sub = _sanitize(cfg.sub_model) if cfg.mode == "multi" else "single"
        name = f"{cfg.mode}__{_sanitize(cfg.orch_model)}__{sub}__{ts}"
        cfg.run_dir = str(Path(cfg.eval_root) / name)
        Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)

        # Persist the yml verbatim + a frozen_md5 lock.
        data["frozen_md5"] = _frozen_md5(data)
        with open(Path(cfg.run_dir) / "config.yml", "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        return cfg

    # ── Case 2: --config (resume) ──────────────────────────────────────────────
    @classmethod
    def load_run(cls, run_dir: str) -> "EvalConfig":
        # Accept either the dir or its config.yml.
        p = Path(run_dir)
        cfg_path = p if p.name == "config.yml" else p / "config.yml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"No config.yml in {run_dir}")
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        stored = data.get("frozen_md5")
        if stored and stored != _frozen_md5(data):
            raise ValueError(
                f"{cfg_path}: a locked field was edited (frozen_md5 mismatch). "
                "Only cost_config / data paths may change after --init; "
                "to change model/run settings, start a new run with --init."
            )
        cfg = cls._from_dict(data)
        cfg.run_dir = str(cfg_path.parent)
        return cfg

    @property
    def traj_dir(self) -> Path:
        return Path(self.run_dir) / "traj"

    @property
    def eval_json(self) -> Path:
        return Path(self.run_dir) / "eval.json"
