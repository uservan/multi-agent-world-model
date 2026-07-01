#!/bin/bash
# DEBUG: rollout + reward ONLY (no training) on Qwen3.6-35B-A3B.
# Non-destructive: does NOT pkill sglang/python/ray, so a concurrently-running eval
# is left alone. NOTE: it still needs FREE GPUs for its own rollout sglang (tp4) —
# if the other job holds all GPU memory, this will fail to allocate. Free ~4 GPUs first.
#
# Run from PROJECT ROOT:
#     BASE_FOLDER=/shared/models MASTER_ADDR=<this-node-ip> bash train/run_awm_debug35.sh

set -ex
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

BASE_FOLDER=${BASE_FOLDER:-/shared/models}
MASTER_ADDR=${MASTER_ADDR:?set MASTER_ADDR to this node IP}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-1}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-4}   # rollout-only: 4 GPUs is plenty for 35B-A3B
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
RAY_DASH_PORT=${RAY_DASH_PORT:-8266}                    # separate from any existing ray (8265)

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime-n"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)

# ── 35B-A3B arch (qwen3_5 MoE) — the only line changed from the 27B launcher ──────
source "${SLIME_DIR}/scripts/models/qwen3.5-35B-A3B.sh"     # → MODEL_ARGS[]

# ── our rollout + reward + data ──────────────────────────────────────────────────
ROLLOUT_ARGS=(
   --custom-generate-function-path train.rollout:generate
   --custom-config-path            train/awm_config.py
   --prompt-data  "${PROJECT_ROOT}/outputs/generated_new/verified/task_final.jsonl"
   --input-key  prompt
   --label-key  label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout              1          # DEBUG: a single rollout step
   --rollout-batch-size       1          # DEBUG: one task
   --n-samples-per-prompt     2          # DEBUG: tiny GRPO group (enough to see reward variance)
   --rollout-max-context-len  131072
   --rollout-max-response-len 8192
   --rollout-temperature      1.0
   --balance-data
)

TRAIN_ARGS=(
   --config "${PROJECT_ROOT}/train/config_debug35.yaml"   # 35B model, rollout-only
   --debug-rollout-only                                   # ← rollout+reward, NO weight update / no megatron
   --colocate
   --dump-details /tmp/awm_debug35/dump_details           # rollout_data + samples auto-dumped here
)

# ── ray cluster (start only if not already up; never kill an existing one) ────────
export no_proxy="127.0.0.1,${MASTER_ADDR}"
if ! ray status --address "127.0.0.1:6379" >/dev/null 2>&1; then
  ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
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
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}"
} }
EOF_JSON
)

cd "${SLIME_DIR}"
ray job submit --address="http://127.0.0.1:${RAY_DASH_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_multi_policy.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   "${MODEL_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${TRAIN_ARGS[@]}"
