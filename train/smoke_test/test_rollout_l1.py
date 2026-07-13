"""L1 test: exercise train/rollout.py:generate END-TO-END with ONLY the sglang
generation call mocked. Everything else is real:
  * real tokenizer (Qwen3.6-27B)          → real chat-template + prompt_ids
  * real platform servers (subprocess)    → resolve_resources + PlatformRuntime
  * real tool execution / verifier / reward
  * real Sample building (_samples_from_turns)

The ONLY fake is `train.rollout.post` (the sglang engine call): it returns a
canned model turn so we never touch a GPU or a running sglang. This isolates
"our integration glue" from "the sglang backend".

Run from PROJECT ROOT:
    PYTHONPATH=$PWD:$PWD/slime-n python3 train/test_rollout_l1.py
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from transformers import AutoTokenizer

import train.rollout as R
from train.awm_config import AWMConfig
from train.rollout import generate
from slime.utils.types import Sample

MODEL = os.environ.get("L1_MODEL", "/shared/models/Qwen3.6-27B")
TASKS = "outputs/generated_new/verified/task_final.jsonl"

# What the fake "model" emits every turn. `<done>` makes the orchestrator finish
# after one turn — minimal but drives the whole pipeline once. Swap this for a
# scripted <tool_call> sequence to exercise tools / sub-agents in a richer L1.
CANNED_TURN = "<done>\nL1 smoke: task acknowledged, no actions taken.\n</done>"


def build_args(tokenizer) -> argparse.Namespace:
    """A minimal slime-like args namespace with just what generate/SlimeModelClient read."""
    cfg = AWMConfig()  # defaults: outputs/ paths (relative → run from PROJECT ROOT)
    return argparse.Namespace(
        tokenizer=tokenizer,
        sampling_params={"max_new_tokens": 256, "temperature": 1.0, "top_p": 1.0},
        rollout_max_context_len=131072,
        custom_config=cfg,
    )


def make_fake_post(tokenizer):
    """Async stand-in for slime's sglang POST: returns canned text + real token ids
    (encoded with the real tokenizer) and dummy per-token logprobs, in sglang's
    meta_info.output_token_logprobs shape [[logprob, token_id, None], ...]."""
    ids = tokenizer(CANNED_TURN, add_special_tokens=False)["input_ids"]
    text = tokenizer.decode(ids)
    token_logprobs = [[-0.1, tid, None] for tid in ids]

    async def fake_post(url, payload=None, *a, **k):
        return {"text": text, "meta_info": {"output_token_logprobs": token_logprobs}}

    return fake_post


async def main() -> int:
    # 1) load one real task
    line = open(TASKS).readline()
    import json
    t = json.loads(line)
    sample = Sample(
        index=0,
        prompt=t["prompt"],
        label=t["label"],
        metadata=t.get("metadata", {}),
    )
    platforms = list((sample.metadata or {}).get("verifiers", {}).keys())
    print(f"[L1] task={sample.label}  platforms={platforms}")

    # 2) real tokenizer
    print(f"[L1] loading tokenizer: {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    # 3) mock ONLY the sglang generation call (post + url resolution)
    R.post = make_fake_post(tok)
    R.get_model_url = lambda args, name, endpoint="/generate": "http://fake-sglang"

    args = build_args(tok)

    # 4) run the real generate()
    print("[L1] running generate() — real servers, real reward, fake model …")
    samples = await generate(args, sample, args.sampling_params)

    # 5) assertions / summary
    print(f"\n[L1] generate() returned {len(samples)} training Sample(s)")
    ok = True
    for i, s in enumerate(samples):
        n_resp = s.response_length
        n_lp = len(s.rollout_log_probs or [])
        n_tok = len(s.tokens or [])
        pol = getattr(s, "policy_name", "<none>")
        print(f"  [{i}] policy={pol} reward={s.reward} tokens={n_tok} "
              f"response_len={n_resp} logprobs={n_lp} status={s.status}")
        if n_resp <= 0 or n_lp != n_resp or n_tok <= n_resp:
            print(f"      !! FIELD MISMATCH (expect logprobs==response_len, tokens>response_len)")
            ok = False
        if not isinstance(s.reward, (int, float)):
            print(f"      !! reward not scalar: {type(s.reward)}")
            ok = False

    if not samples:
        print("[L1] WARNING: 0 samples — check train_roles / that orch produced a turn")
        ok = False

    print(f"\n[L1] {'PASS ✅' if ok else 'FAIL ❌'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
