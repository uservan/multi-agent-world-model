"""Convert a slime rollout dump (.pt) into readable message format.

Each dumped sample is ONE orchestrator turn: tokens = prompt_ids + response_ids,
where prompt is the full chat-templated history up to that turn. So within a group
(one rollout execution), the turn with the MOST tokens carries the fullest history;
its `prompt` + decoded response = the complete orchestrator conversation. We parse
the chat-template markers (<|im_start|>role ... <|im_end|>) back into
[{role, content}] messages.

Usage:
    .venv/bin/python train/smoke_test/rollout_to_msg.py \
        [save/qwen3.6-27b/rollout/orchestrator/rollout_data/0.pt]
"""
from __future__ import annotations

import json
import os
import re
import sys

import torch
from transformers import AutoTokenizer

PT = sys.argv[1] if len(sys.argv) > 1 else "save/qwen3.6-27b/rollout/orchestrator/rollout_data/0.pt"
MODEL = os.environ.get("L1_MODEL", "/tmp/instance_storage/models/Qwen3.6-27B")
OUT_DIR = os.path.join(os.path.dirname(PT), "..", "msgs")

_MSG_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)<\|im_end\|>", re.DOTALL)


def parse_messages(text: str) -> list[dict]:
    """Split chat-templated text into [{role, content}]. A trailing open
    <|im_start|>assistant\\n...(no <|im_end|>) is captured as the final turn."""
    msgs = [{"role": r, "content": c} for r, c in _MSG_RE.findall(text)]
    tail = text.rsplit("<|im_end|>", 1)[-1]
    m = re.search(r"<\|im_start\|>(\w+)\n(.*)$", tail, re.DOTALL)
    if m and m.group(2).strip():
        msgs.append({"role": m.group(1), "content": m.group(2)})
    return msgs


def main() -> int:
    d = torch.load(PT, weights_only=False)
    samples = d["samples"]
    print(f"[msg] {PT}: {len(samples)} turn-samples, policy={samples[0].get('policy_name')}")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    # group by group_id (one rollout execution = one episode)
    groups: dict = {}
    for s in samples:
        groups.setdefault(s["group_id"], []).append(s)

    for gid, turns in sorted(groups.items()):
        # fullest-history turn = most tokens; its prompt + its response = whole convo
        top = max(turns, key=lambda x: len(x["tokens"]))
        resp_ids = top["tokens"][-top["response_length"]:] if top["response_length"] else []
        full_text = top["prompt"] + tok.decode(resp_ids)
        msgs = parse_messages(full_text)

        out = os.path.join(OUT_DIR, f"group{gid}.json")
        with open(out, "w") as f:
            json.dump({"group_id": gid, "reward": top.get("reward"),
                       "n_turns": len(turns), "messages": msgs}, f, ensure_ascii=False, indent=2)
        roles = [m["role"] for m in msgs]
        print(f"\n=== group {gid}: reward={top.get('reward'):.3f}  turns={len(turns)}  "
              f"messages={len(msgs)} {roles} ===")
        for m in msgs:
            body = m["content"].strip().replace("\n", " ")
            print(f"  [{m['role']}] {body[:160]}{'…' if len(body) > 160 else ''}")
        print(f"  → full JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
