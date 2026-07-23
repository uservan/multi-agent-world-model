"""Dynamic-sampling filters for the AWM multi-agent RL rollout (DAPO-style).

Wired in via:  --dynamic-sampling-filter-path  train.filters.check_reward_nonzero_std

slime's rollout controller (slime/rollout/sglang_rollout.py:generate_rollout_async)
over-samples prompts and calls this filter on each completed GRPO group; groups the
filter drops are discarded and the controller keeps sampling until it has
`rollout_batch_size` KEPT groups (batch stays full — the DAPO refill behavior).

Why not slime's built-in `check_reward_nonzero_std`: that one assumes a flat
`list[Sample]` (one Sample per rollout). Our multi-agent rollout fans ONE trajectory
out into many turn-samples (train/rollout.py:_samples_from_turns), so a GRPO group is
`list[list[Sample]]` — one inner list per trajectory, every turn-sample in it sharing
that trajectory's single episode reward. Feeding that to the built-in filter throws
(`list` has no `get_reward_value`). Here we reduce each trajectory to its one reward
first, then apply the same "keep only if the group's rewards vary" rule.
"""
from __future__ import annotations

import torch

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample


def _group_rewards(args, group) -> list[float]:
    """One reward per group member. `group` is `list[list[Sample]]` (multi-agent
    fan-out) or `list[Sample]` (plain rollout). Empty trajectories are skipped."""
    rewards: list[float] = []
    for item in group:
        if isinstance(item, list):
            if not item:                 # a trajectory that emitted no trainable turn
                continue
            sample = item[0]             # all turns of one trajectory share its reward
        else:
            sample = item
        rewards.append(sample.get_reward_value(args))
    return rewards


def check_reward_nonzero_std(args, group, **kwargs) -> DynamicFilterOutput:
    """DAPO dynamic-sampling filter: keep a group only if its per-trajectory rewards
    are not all identical (zero-std groups give advantage=0 → no gradient)."""
    rewards = _group_rewards(args, group)
    if len(rewards) < 2:                 # degenerate group — no usable contrast
        return DynamicFilterOutput(keep=False, reason="insufficient_samples")
    keep = bool(torch.tensor(rewards, dtype=torch.float64).std() > 1e-6)
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 2)}",
    )
