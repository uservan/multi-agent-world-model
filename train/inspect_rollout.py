#!/usr/bin/env python3
"""Inspect a debug-rollout dump and check the rollout+reward pipeline is sane.

Usage:
    python train/inspect_rollout.py [dump_dir]            # default: /tmp/awm_debug35/dump_details
    python train/inspect_rollout.py <file.pt>
    python train/inspect_rollout.py <dir> --decode /shared/models/Qwen3.6-35B-A3B   # also decode a token snippet

Checks (the 5 things that tell you rollout+reward is OK):
  1. reward is a number in [0,1]  (== verifier acc)
  2. reward has VARIANCE across the GRPO group  (else advantage is 0 — nothing to learn)
  3. policy_name == "orchestrator"  (only orch tokens are trainable, per train_roles=["orch"])
  4. loss_mask covers SOME but not ALL tokens  (orch-generated tokens =1, tool/sub tokens =0)
  5. tokens form a real multi-agent trajectory  (decode to eyeball http/spawn calls)
"""
import sys, glob, os
import torch


def find_pts(arg):
    if arg.endswith(".pt"):
        return [arg]
    pats = [os.path.join(arg, "**", "rollout_data", "*.pt"), os.path.join(arg, "**", "*.pt")]
    for p in pats:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            return hits
    return []


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dump = args[0] if args else "/tmp/awm_debug35/dump_details"
    decode_model = None
    if "--decode" in sys.argv:
        decode_model = sys.argv[sys.argv.index("--decode") + 1]

    files = find_pts(dump)
    if not files:
        print(f"❌ no rollout .pt found under {dump}\n"
              f"   (run train/run_awm_debug35.sh first; it dumps to /tmp/awm_debug35/dump_details)")
        sys.exit(1)

    tok = None
    if decode_model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(decode_model, trust_remote_code=True)

    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        samples = d.get("samples", [])
        pol = d.get("policy_name")
        print(f"\n===== {f}")
        print(f"  policy_name={pol}  rollout_id={d.get('rollout_id')}  n_samples={len(samples)}")
        rewards = []
        for i, s in enumerate(samples):
            r = s.get("reward")
            toks = s.get("tokens") or []
            lm = s.get("loss_mask")
            rl = s.get("response_length")
            spn = s.get("policy_name")
            mask_sum = sum(lm) if lm else None
            rewards.append(r if isinstance(r, (int, float)) else None)
            print(f"  [sample {i}] reward={r}  policy={spn}  n_tokens={len(toks)}  "
                  f"resp_len={rl}  loss_mask_sum={mask_sum}"
                  f"{'/'+str(len(lm)) if lm else ''}")
            # checks
            flags = []
            if not isinstance(r, (int, float)) or not (0.0 <= float(r) <= 1.0):
                flags.append("⚠️ reward not in [0,1]")
            if spn != "orchestrator":
                flags.append(f"⚠️ policy_name != orchestrator (={spn})")
            if lm is not None:
                if mask_sum == 0:
                    flags.append("⚠️ loss_mask all-zero (no trainable tokens!)")
                elif mask_sum == len(lm):
                    flags.append("⚠️ loss_mask all-one (tool/sub tokens also trainable?)")
            else:
                flags.append("⚠️ no loss_mask")
            for fl in flags:
                print(f"        {fl}")
            if tok and i == 0:
                print("        --- decoded token snippet (first 400 chars) ---")
                print("        " + tok.decode(toks)[:400].replace("\n", "\n        "))
                # also show a trainable-only slice to confirm loss_mask aligns with orch output
                if lm:
                    tr = [t for t, m in zip(toks, lm) if m]
                    print("        --- trainable (loss_mask=1) snippet ---")
                    print("        " + tok.decode(tr)[:300].replace("\n", "\n        "))
        clean = [r for r in rewards if r is not None]
        if len(clean) >= 2:
            import statistics
            var = statistics.pvariance(clean)
            print(f"  reward group: mean={statistics.mean(clean):.3f}  var={var:.4f}  values={clean}")
            if var == 0:
                print("        ⚠️ reward has ZERO variance across the group → GRPO advantage = 0, "
                      "nothing to learn (fine for a 1-task smoke test, but watch this at scale)")
        print("  ✅ structural check done" )


if __name__ == "__main__":
    main()
