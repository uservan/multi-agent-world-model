#!/bin/bash
# AWM multi-agent RL — train the orchestrator (Qwen3.6-27B) on our generated tasks.
# Reward = our verifier acc (train/reward.py). Rollout = our multi-agent eval wrapped
# for token-level generation (train/rollout.py). Multi-policy entry so the sub-agent
# can be co-trained later by editing train/config.yaml + awm_config.py only.
#
# Run from the PROJECT ROOT:
#     BASE_FOLDER=/shared/models MASTER_ADDR=<this-node-ip> bash train/run_awm_qwen36_27b.sh

set -ex
pkill -9 sglang; sleep 2; ray stop --force; pkill -9 ray; pkill -9 python; sleep 2
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

# ── must-set env ────────────────────────────────────────────────────────────────
BASE_FOLDER=${BASE_FOLDER:-/shared/models}          # where Qwen3.6-27B* live
MASTER_ADDR=${MASTER_ADDR:?set MASTER_ADDR to this node IP}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}               # single node
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}   # 8 H200 (enough for 27B w/ cpu-offload + colocate)
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime-n"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

# ── model arch flags (Qwen3.6-27B == qwen3_5 arch; reuse slime's 27B arch) ───────
source "${SLIME_DIR}/scripts/models/qwen3.5-27B.sh"     # → MODEL_ARGS[]

# ── OUR custom rollout + reward + data (the only AWM-specific wiring) ─────────────
ROLLOUT_ARGS=(
   --custom-generate-function-path train.rollout:generate          # ← our rollout
   --custom-config-path            train/awm_config.py             # ← our knobs (mode/style/paths/policy axes)
   --prompt-data  "${PROJECT_ROOT}/outputs/generated_new/verified/task_final.jsonl"  # already slime-shaped
   --input-key  prompt
   --label-key  label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout              2000
   --rollout-batch-size       8          # tasks per step
   --n-samples-per-prompt     8          # GRPO group (must match config.yaml)
   --rollout-max-context-len  131072     # long: multi-agent trajectories are big
   --rollout-max-response-len 8192       # per-turn generation cap
   --rollout-temperature      1.0
   --balance-data
)

TRAIN_ARGS=(
   --config "${PROJECT_ROOT}/train/config.yaml"   # the policies (1 = orchestrator)
   --colocate
   --dump-details /tmp/awm_qwen36_27b/dump_details
)

# ── ray cluster ──────────────────────────────────────────────────────────────────
export no_proxy="127.0.0.1,${MASTER_ADDR}"
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
# (multi-node: start ray workers on the other node pointing at ${MASTER_ADDR}:6379,
#  same as scripts/run-qwen3.5-27B.sh's HOSTFILE loop)

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{ "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "${PROJECT_ROOT}:${SLIME_DIR}:/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}"
} }
EOF_JSON
)

# train_multi_policy.py: per-policy model/megatron/sglang comes from --config yaml;
# MODEL_ARGS (arch) + ROLLOUT_ARGS are shared on the CLI.
cd "${SLIME_DIR}"
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_multi_policy.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   "${MODEL_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${TRAIN_ARGS[@]}"
