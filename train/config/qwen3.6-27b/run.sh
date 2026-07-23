#!/bin/bash
# L3 TRAIN: full GRPO step on Qwen3.6-27B (rollout + reward + megatron weight update
# + sync back to sglang). 4+4 non-colocate placement, run WITHOUT --debug-rollout-only
# (actor is real) and WITHOUT --colocate (actor on 4 GPUs,
# sglang on the other 4). Run under the uv venv.
#
# Run from PROJECT ROOT:
#     MASTER_ADDR=<this-node-ip> bash train/config/qwen3.6-27b/run.sh

set -ex
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

MASTER_ADDR=${MASTER_ADDR:?set MASTER_ADDR to this node IP}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}                   # 2 nodes → tp4×cp4=16, dp=1. NOT 4:
                                                        # tp*cp is capped at 16 by linear_num_key_heads=16, so the
                                                        # only way to use 4 nodes is dp=2 — tried 2026-07-18 and it
                                                        # died on the static-schedule divisibility check (rollout
                                                        # emitted 1089 samples, odd). See train.yaml.
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}   # matches config num_gpus_per_node=8 → 2×8=16 GPU
RAY_NUM_GPUS=${RAY_NUM_GPUS:-16}                         # total GPUs ray manages (colocate)
RAY_DASH_PORT=${RAY_DASH_PORT:-8266}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}       # prompts/tasks per rollout step (GRPO batch);
                                                  # single source of truth — also tags the W&B run name.
# read the ACTUAL lr straight out of train.yaml (next to this script) so the W&B run name
# can never drift from the real value. Matches the `lr:` line only, not lr_decay_style.
LR=$(awk '/^[[:space:]]*lr:[[:space:]]/{print $2; exit}' "$(dirname -- "${BASH_SOURCE[0]}")/train.yaml")

# ── W&B logging (override via env before launch) ──────────────────────────────────
#   Live curves need egress + a key:  export WANDB_API_KEY=... before launching.
#   No egress? run offline and sync later:  WANDB_MODE=offline bash train/config/qwen3.6-27b/run.sh
#   Turn it off entirely:  WANDB_MODE=disabled (or delete the --use-wandb block below).
WANDB_PROJECT=${WANDB_PROJECT:-awm-multi-agent}   # W&B project (all AWM runs)
WANDB_GROUP=${WANDB_GROUP:-qwen3.6-27b-lr${LR}-bs${ROLLOUT_BATCH_SIZE}}   # run name: model + rl + lr + batch size
WANDB_MODE=${WANDB_MODE:-online}                  # online | offline | disabled

# SCRIPT_DIR = this model's config folder (train/config/qwen3.6-27b); its train.yaml +
# rollout.yaml live right next to this script, so a new model = copy this whole folder.
# PROJECT_ROOT is 3 levels up (…/train/config/qwen3.6-27b → project root).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime-n"
VENV_PY="${PROJECT_ROOT}/.venv/bin/python"

# ── secrets: source export.sh if present (GITIGNORED — put your WANDB_API_KEY there) ──
[ -f "${PROJECT_ROOT}/export.sh" ] && source "${PROJECT_ROOT}/export.sh"

# ── TWO interpreters, deliberately kept apart (2026-07-18: tried to unify, both ways broke) ──
#   TRAIN_PY (system python, numpy 1.26.4) runs the driver + every ray worker/megatron actor.
#     megatron asserts numpy 1.x; the venv has numpy 2.4.6 (mcp-agent needs >=2.1.3) so the
#     venv can NEVER run megatron → "Megatron does not support numpy 2.x".
#   VENV_PY  runs ONLY the eval platform server subprocesses, which need fastapi/uvicorn/
#     sqlalchemy (venv-only). Passed down as AWM_SERVER_PYTHON so utils/server.py stops
#     using sys.executable → otherwise every platform dies with "No module named 'sqlalchemy'".
#   Consequence: ray must ALSO be started with system python (sh/start_ray.sh).
TRAIN_PY=${TRAIN_PY:-/usr/bin/python3}

# ── auto-detect the NIC carrying MASTER_ADDR (image has no eth0/ip cmd) ───────────
SOCKET_IFNAME=${SOCKET_IFNAME:-$(python3 - "$MASTER_ADDR" <<'PY'
import socket, fcntl, struct, os, sys
tip = sys.argv[1]
for ifn in sorted(os.listdir('/sys/class/net')):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        ip = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', ifn[:15].encode()))[20:24])
    except OSError:
        ip = None
    finally:
        s.close()
    if ip == tip:
        print(ifn); break
PY
)}
SOCKET_IFNAME=${SOCKET_IFNAME:-lo}

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

# ── 27B arch (qwen3_5) ────────────────────────────────────────────────────────────
source "${SLIME_DIR}/scripts/models/qwen3.5-27B.sh"     # → MODEL_ARGS[]

# ── our rollout + reward + data ──────────────────────────────────────────────────
ROLLOUT_ARGS=(
   --custom-generate-function-path train.rollout.generate
   --custom-config-path            "${SCRIPT_DIR}/rollout.yaml"
   --prompt-data  "${PROJECT_ROOT}/outputs/test_gen_by_claude/verified/task_final.jsonl"
   --input-key  prompt
   --label-key  label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout              25         # ~one epoch: 196 tasks / rollout_batch_size 8 = 24.5 steps.
                                         # slime's loop is range(start_rollout_id, num_rollout), and a
                                         # fresh bridge-mode load reports loaded_rollout_id=0 → start=1,
                                         # so this gives 24 real rollout+GRPO steps. (num_rollout=2 was
                                         # the single-step smoke test used to debug the OOM.)
                                         # A step costs ~1.6-2.8h, so a 24h SDB job only finishes ~3-4;
                                         # save_interval=1 makes the next job resume from `load:`.
   --rollout-batch-size       "${ROLLOUT_BATCH_SIZE}"   # prompts/tasks per rollout step (var at top)
   --n-samples-per-prompt     8          # GRPO group size = 8 (→ 8×8 = 64 trajectories/step)
   --rollout-max-context-len  32768      # cap total trajectory length at 32K (was 128K → some rambled to 44K).
                                         # trained via tp4×cp4: 32K sequence → 8K/GPU.
   --rollout-max-response-len 8192
   --rollout-temperature      1.0
   --balance-data
   # ── DAPO dynamic sampling (refill): over-sample prompts, DROP any GRPO group whose
   #    8 trajectories all share one reward (advantage=0 → zero gradient), and keep
   #    sampling until rollout_batch_size(8) VALID groups are collected. The training
   #    step still sees exactly 8 groups (slime asserts len(data)==rollout_batch_size),
   #    so training-side memory is UNCHANGED — the extra cost is generation wall-clock
   #    only (~1.6-2× rollouts/step at the current ~62% keep rate). No OOM risk.
   --dynamic-sampling-filter-path train.filters.check_reward_nonzero_std   # multi-agent
                                         # aware (group is list[list[Sample]]); see train/filters.py.
   --over-sampling-batch-size  16        # submit prompts in waves of 16 (2× batch) so one wave
                                         # usually yields ≥8 keepers. Lower it (→ closer to 1×) to
                                         # waste fewer rollouts at the cost of more sequential waves.
)

TRAIN_ARGS=(
   --config "${SCRIPT_DIR}/train.yaml"                    # 27B, full training (actor real)
   --disable-grpo-std-normalization                       # Dr.GRPO (arXiv:2503.20783): advantage keeps the
                                                          # group-mean baseline (rewards - group_mean) but SKIPS
                                                          # the /std step (ray/rollout.py:916). Removes the
                                                          # difficulty-bias that /std introduces. The other half
                                                          # of Dr.GRPO (constant, length-unbiased loss norm) is
                                                          # already on via calculate_per_token_loss=true in the
                                                          # config, and KL is already off (kl_coef=0).
   --save-interval 1                                      # MUST be passed HERE, not only in the policy's
                                                          # megatron: block. train_multi_policy.py:376 gates
                                                          # saving on the GLOBAL args.save_interval, whose
                                                          # default is None (arguments.py:801), and nothing
                                                          # copies the per-policy value up to it (only
                                                          # n_samples_per_prompt / global_batch_size get copied).
                                                          # With it unset, should_run_periodic_action returns
                                                          # False on its first line — so even the end-of-run
                                                          # "last rollout" save never fires and the job finishes
                                                          # having written NOTHING. Cost us 2 full steps
                                                          # (~3.5h) on 2026-07-18 before it was caught: the
                                                          # policy dump showed save_interval=1 and looked right.
   --qkv-format bshd                                      # non-packed layout: megatron's GDN (gated-delta-net)
                                                          # layers reject packed sequences (qkv_format=thd);
                                                          # bshd requires use_dynamic_batch_size=false (set in config).
   --cp-split-in-model                                    # this checkpoint is built by megatron-bridge as a
                                                          # Qwen3VLModel, whose forward slices the sequence
                                                          # across CP ranks itself when packed_seq_params is
                                                          # None (i.e. under bshd) — so get_batch must hand it
                                                          # the full padded sequence, not a pre-sliced one, or
                                                          # it gets split twice. Same zigzag layout either way.
   --colocate                                             # actor+sglang share the 4 GPUs (time-shared via
                                                          # auto offload); routes weight-sync through the
                                                          # megatron-bridge weight iterator (correct for this
                                                          # VLM/GDN checkpoint — the non-colocate distributed
                                                          # path uses a hand-written converter that lacks the
                                                          # bridge-built vision_model.* / language_model.* naming).
   --dump-details "${PROJECT_ROOT}/save/qwen3.6-27b/rollout"
   # ── W&B: logs reward/grad-norm/loss/entropy per train step + the full arg config
   #    (lr, advantage_estimator, grpo_std_normalization, dynamic-sampling, …) automatically.
   #    Mode/key come from WANDB_MODE / WANDB_API_KEY in the ray runtime env below.
   --use-wandb
   --wandb-project "${WANDB_PROJECT}"
   --wandb-group   "${WANDB_GROUP}"
   --wandb-always-use-train-step        # x-axis = train step (not wall-clock), so resume jobs align
)

# ── ray cluster (start only if not already up) ────────────────────────────────────
export no_proxy="127.0.0.1,${MASTER_ADDR}"
if ! ray status --address "127.0.0.1:6379" >/dev/null 2>&1; then
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${RAY_NUM_GPUS}" \
     --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASH_PORT}"
fi

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{ "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "${PROJECT_ROOT}:${SLIME_DIR}:/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "AWM_SERVER_PYTHON": "${VENV_PY}",
    "WANDB_API_KEY": "${WANDB_API_KEY:-}",
    "WANDB_MODE": "${WANDB_MODE}"
} }
EOF_JSON
)

# ── optional passthrough for one-off experiments; empty by default ───────────────
#   e.g. deterministic train-only A/B replay (skips sglang + generation entirely):
#     EXTRA_ARGS="--load-debug-rollout-data ${PROJECT_ROOT}/save/ab_baseline/rollout_fixed.pt" \
#       bash train/config/qwen3.6-27b/run.sh
#   Setting --load-debug-rollout-data forces debug_train_only + skip_sglang (arguments.py:1540,1849),
#   so both A and B sides consume the SAME 802 samples and the numbers are comparable.
read -r -a EXTRA_ARGS_ARR <<< "${EXTRA_ARGS:-}"

# entrypoint runs in ray-head cwd (PROJECT_ROOT); ABSOLUTE paths, VENV python.
ray job submit --address="http://127.0.0.1:${RAY_DASH_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${TRAIN_PY}" "${SLIME_DIR}/train_multi_policy.py" \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   "${MODEL_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${TRAIN_ARGS[@]}" \
   ${EXTRA_ARGS_ARR[@]+"${EXTRA_ARGS_ARR[@]}"}
