"""AWM RL training config — all the *non-model, non-optimizer* knobs.

Pointed at by slime's `--custom-config-path train/awm_config.py`. The rollout reads
these via `load_awm_config(args)`. Training hyperparams (lr / batch / num-rollout /
rollout-temperature / max-response-len ...) come from slime's own CLI flags and are
read off `args` directly — they do NOT live here.

Two orthogonal axes control the policy structure (see train/rollout.py):
  - share_model:  orchestrator and sub-agent are the SAME policy (one ckpt) vs separate.
  - train_roles:  which roles' tokens become trainable samples. Today ["orch"]; later
                  ["orch", "sub"] to co-train the sub-agent — rollout code is unchanged,
                  you only flip this + add the policy to config.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AWMConfig:
    # ── Agent mode / prompt (reused verbatim from eval) ──────────────────────
    mode: str = "multi"                  # we only train multi; single is eval-only
    orch_style: str = "neutral"          # neutral | delegate | solo | a_neutral | a_delegate | a_solo
    max_turns: int = 200                 # orchestrator turn cap
    sub_max_turns: int = 200             # sub-agent turn cap
    max_concurrent: int = 4              # parallel sub-agents
    max_queue: int = 16                  # sub-agent queue limit

    # ── Policy structure (the two axes) ──────────────────────────────────────
    share_model: bool = True             # True: orch & sub use the same engine/policy
    train_roles: list[str] = field(default_factory=lambda: ["orch"])  # later: ["orch","sub"]
    orch_policy: str = "orchestrator"    # policy_name tagged onto orchestrator tokens
    sub_policy: str = "subagent"         # policy_name for sub tokens (== orch_policy if share_model)

    # ── Reward (pluggable; see train/reward.py) ──────────────────────────────
    reward_fn_path: str = "train.reward:compute_reward"   # module:function (the dispatcher)
    reward_type: str = "acc"   # which reward FORM: acc | binary | shaped | per_platform | ...
    reward_kwargs: dict = field(default_factory=dict)     # extra knobs for the chosen form

    # ── Environment / resource bundle (same files eval uses for scoring) ─────
    platforms_input: str = "outputs/platforms.jsonl"
    envs_input: str = "outputs/generated_new/envs_fixed.jsonl"
    databases_dir: str = "outputs/generated_new/databases"


# slime loads this module by path and looks for a module-level `AWM` (a dict) or a
# `get_config()`; we expose both so either convention works.
AWM = AWMConfig()


def get_config() -> AWMConfig:
    return AWM


def load_awm_config(args) -> AWMConfig:
    """Resolve the AWMConfig for a rollout. `args.custom_config` is whatever slime
    loaded from --custom-config-path; fall back to the module default."""
    cfg = getattr(args, "custom_config", None)
    if isinstance(cfg, AWMConfig):
        return cfg
    if isinstance(cfg, dict):
        return AWMConfig(**{k: v for k, v in cfg.items() if k in AWMConfig.__dataclass_fields__})
    return AWM
