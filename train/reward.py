"""Reward interface for AWM multi-agent RL.

Two layers:
  * compute_reward(...)  — the DISPATCHER slime calls (path in awm_config.reward_fn_path).
    It picks a reward FORM by `cfg.reward_type` and returns a per-policy dict.
  * REWARD_FORMS         — registry of forms. Add a new form = add one function +
    one registry entry; select it with `reward_type: <name>` in awm_config. No rollout change.

Per-policy dict is what enables co-training the sub-agent later WITHOUT touching the rollout:
    {policy_name: reward}   (a role that runs but isn't trained simply isn't a key)

Signature of a form:
    form(task, trajectory, verifier_results, acc, **reward_kwargs) -> float   # episode reward (scalar)
The dispatcher then assigns that scalar to the trained policies per train_roles.
"""
from __future__ import annotations


# ── reward FORMS (the "好几种形式") ──────────────────────────────────────────────

def _form_acc(task, trajectory, verifier_results, acc, **kw) -> float:
    """Fraction of platforms whose verifier passed (our standard eval acc)."""
    return float(acc)


def _form_binary(task, trajectory, verifier_results, acc, *, threshold=1.0, **kw) -> float:
    """1.0 only if the whole task passes (acc >= threshold), else 0.0."""
    return 1.0 if acc >= threshold else 0.0


def _form_shaped(task, trajectory, verifier_results, acc, *, turn_penalty=0.0, **kw) -> float:
    """acc minus a small penalty per orchestrator turn (discourage dithering)."""
    n_turns = sum(1 for e in trajectory.get("events", []) if e.get("role") == "orchestrator")
    return float(acc) - turn_penalty * n_turns


REWARD_FORMS = {
    "acc": _form_acc,
    "binary": _form_binary,
    "shaped": _form_shaped,
    # "per_platform": ...   # future: needs per-sub credit, returns per-policy directly
}


# ── dispatcher (what slime calls) ────────────────────────────────────────────────

def compute_reward(task, trajectory, verifier_results, acc, *, cfg=None) -> dict[str, float]:
    reward_type = getattr(cfg, "reward_type", "acc")
    reward_kwargs = getattr(cfg, "reward_kwargs", {}) or {}
    orch_policy = getattr(cfg, "orch_policy", "orchestrator")
    sub_policy = getattr(cfg, "sub_policy", "subagent")
    train_roles = getattr(cfg, "train_roles", ["orch"])

    form = REWARD_FORMS.get(reward_type)
    if form is None:
        raise ValueError(f"unknown reward_type={reward_type!r}; have {list(REWARD_FORMS)}")
    r = form(task, trajectory, verifier_results, acc, **reward_kwargs)

    # assign the episode reward to the trained policies (v0/v1).
    # a per-policy form (v2) would return its own dict and skip this broadcast.
    rewards: dict[str, float] = {}
    if "orch" in train_roles:
        rewards[orch_policy] = r
    if "sub" in train_roles:
        rewards[sub_policy] = r
    return rewards
