"""AWM multi-agent RL rollout for slime.

Wired in via:  --custom-generate-function-path  train.rollout:generate

Design (see train/awm_config.py for the knobs):
  * We REUSE the eval stack unchanged — eval.data.resolve_resources to mount the
    platform servers + seed DBs, and eval.multi_agent.run_task to drive the whole
    orchestrator + sub-agent episode (tool parsing, http exec, verifier scoring).
  * The ONLY adaptation is the model call: eval talks to a chat API; RL needs
    token-level generation + logprobs. `SlimeModelClient` is a drop-in for eval's
    ModelClient whose `.complete(messages)` routes through slime's sglang engine and
    records, per turn, (prompt_ids, response_ids, response_logprobs). run_task is
    therefore reused with ZERO changes.
  * After the episode we read each role's recorded turns and emit slime Samples,
    tagged with `policy_name` per `train_roles`, carrying the per-policy reward.
"""
from __future__ import annotations

import importlib
from copy import deepcopy

from slime.rollout.sglang_rollout import get_model_url
from slime.utils.http_utils import post
from slime.utils.types import Sample

from eval import data as eval_data
from eval import multi_agent
from eval.agent import ModelClient  # for interface parity / isinstance docs only

from train.awm_config import load_awm_config


# ── slime-backed model client: same interface as eval.agent.ModelClient ──────────

class SlimeModelClient:
    """Drop-in for eval's ModelClient, but generation goes through slime's sglang
    engine (token ids + logprobs) instead of the OpenAI chat API. Records every turn
    so the rollout can build training Samples afterwards.

    One instance == one policy/role. The sub-agent worker-pool shares a single
    sub instance, so ALL sub turns land in one buffer tagged with `policy_name`.
    """

    def __init__(self, args, route_policy: str, tag_policy: str):
        # route_policy → which sglang engine generates (get_model_url).
        # tag_policy   → the policy_name stamped on emitted Samples (which policy learns).
        # share_model: route==tag==orch for both roles. separate: orch/sub each their own.
        self.args = args
        self.route_policy = route_policy
        self.tag_policy = tag_policy
        self.tokenizer = args.tokenizer
        # Each item: {"prompt_ids", "response_ids", "logprobs", "prompt_text"}
        self.turns: list[dict] = []

    async def complete(self, messages: list[dict]) -> tuple[str, dict]:
        args = self.args
        # 1) chat template → prompt token ids (same prompt the model would see)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        # 2) token-level generation via the policy's sglang engine
        sampling_params = deepcopy(args.sampling_params)
        budget = args.rollout_max_context_len - len(prompt_ids)
        sampling_params["max_new_tokens"] = min(sampling_params["max_new_tokens"], max(0, budget))
        if sampling_params["max_new_tokens"] <= 0:
            return "", {"in": len(prompt_ids), "out": 0}

        payload = {"input_ids": prompt_ids, "sampling_params": sampling_params, "return_logprob": True}
        output = await post(get_model_url(args, self.route_policy), payload)

        token_logprobs = output.get("meta_info", {}).get("output_token_logprobs") or []
        response_ids = [t[1] for t in token_logprobs]
        logprobs = [t[0] for t in token_logprobs]
        text = output.get("text", "")

        # 3) record this turn for later Sample-building
        self.turns.append({
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "logprobs": logprobs,
            "prompt_text": prompt_text,
        })
        # usage dict shaped like eval expects (in/out token counts)
        return text, {"in": len(prompt_ids), "out": len(response_ids)}


# ── reward loader (pluggable per awm_config.reward_fn_path) ──────────────────────

def _load_reward_fn(path: str):
    module_path, fn_name = path.split(":")
    return getattr(importlib.import_module(module_path), fn_name)


# ── build slime Samples from one role's recorded turns ───────────────────────────

def _samples_from_turns(client: SlimeModelClient, reward: float, base: Sample) -> list[Sample]:
    """One Sample per generated turn: tokens = prompt_ids + response_ids, with the
    response span as the trainable part (slime derives loss_mask from response_length).
    Outcome reward is broadcast to every turn of this policy (standard GRPO outcome RL;
    swap for per-turn shaping later)."""
    out: list[Sample] = []
    for turn in client.turns:
        if not turn["response_ids"]:
            continue
        s = deepcopy(base)
        s.policy_name = client.tag_policy
        s.prompt = turn["prompt_text"]
        s.tokens = turn["prompt_ids"] + turn["response_ids"]
        s.response = ""                      # text not needed for training; tokens are authoritative
        s.response_length = len(turn["response_ids"])
        s.rollout_log_probs = turn["logprobs"]
        s.reward = reward
        s.status = Sample.Status.COMPLETED
        out.append(s)
    return out


# ── the rollout entrypoint slime calls ───────────────────────────────────────────

async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> list[Sample]:
    cfg = load_awm_config(args)

    # task context straight off the Sample (no data conversion — task_final.jsonl
    # already has prompt=goal, label=task_id, metadata={verifiers, ...})
    verifiers = (sample.metadata or {}).get("verifiers", {})
    platforms = list(verifiers.keys())
    task = {"prompt": sample.prompt, "label": sample.label,
            "task_id": sample.label, "metadata": sample.metadata}

    # mount platform servers + seed DBs exactly like eval does
    descriptions = eval_data.load_platform_descriptions(cfg.platforms_input)
    server_paths = eval_data.load_server_paths(cfg.envs_input)
    resources = eval_data.resolve_resources(platforms, descriptions, server_paths, cfg.databases_dir)
    if not resources:
        return []

    # ALWAYS two separate clients (separate buffers, so train_roles can include/exclude
    # the sub-agent independently). share_model just routes+tags both to the orch policy.
    orch_client = SlimeModelClient(args, route_policy=cfg.orch_policy, tag_policy=cfg.orch_policy)
    if cfg.share_model:
        sub_client = SlimeModelClient(args, route_policy=cfg.orch_policy, tag_policy=cfg.orch_policy)
    else:
        sub_client = SlimeModelClient(args, route_policy=cfg.sub_policy, tag_policy=cfg.sub_policy)

    seed = (sample.index or 0)
    traj = await multi_agent.run_task(
        task, resources, verifiers, orch_client, sub_client, seed,
        cfg.max_turns, cfg.max_concurrent, cfg.max_queue, cfg.sub_max_turns, cfg.orch_style,
    )

    # reward = our verifier acc, assigned per policy (pluggable)
    compute_reward = _load_reward_fn(cfg.reward_fn_path)
    rewards = compute_reward(sample, traj, traj.get("verifier_results", {}), traj.get("acc", 0.0), cfg=cfg)

    # Emit training Samples by role. A role can RUN (provide environment) yet not be
    # trained — it just isn't emitted. tag_policy decides which policy the tokens train:
    #   share_model:  orch + sub  →  orch_policy   (one policy learns the selected roles)
    #   separate:     orch→orch_policy, sub→sub_policy
    samples: list[Sample] = []
    if "orch" in cfg.train_roles:
        samples += _samples_from_turns(orch_client, rewards.get(orch_client.tag_policy, 0.0), sample)
    if "sub" in cfg.train_roles:
        samples += _samples_from_turns(sub_client, rewards.get(sub_client.tag_policy, 0.0), sample)
    return samples
