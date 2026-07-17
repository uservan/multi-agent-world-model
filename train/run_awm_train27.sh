#!/bin/bash
# L3 TRAIN: full GRPO step on Qwen3.6-27B (rollout + reward + megatron weight update
# + sync back to sglang). Same 4+4 non-colocate placement as run_awm_debug27.sh, but
# WITHOUT --debug-rollout-only (actor is real) and WITHOUT --colocate (actor on 4 GPUs,
# sglang on the other 4). Run under the uv venv.
#
# Run from PROJECT ROOT:
#     MASTER_ADDR=<this-node-ip> bash train/run_awm_train27.sh

set -ex
export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

MASTER_ADDR=${MASTER_ADDR:?set MASTER_ADDR to this node IP}
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}                   # both nodes
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-8}   # matches config num_gpus_per_node=8 → 2×8=16 GPU, tp4×dp4
RAY_NUM_GPUS=${RAY_NUM_GPUS:-16}                         # total GPUs ray manages across both nodes (colocate)
RAY_DASH_PORT=${RAY_DASH_PORT:-8266}

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &>/dev/null && pwd)"
SLIME_DIR="${PROJECT_ROOT}/slime-n"
VENV_PY="${PROJECT_ROOT}/.venv/bin/python"

# ── use the uv venv for ray + workers + the python that spawns platform servers ──
export PATH="${PROJECT_ROOT}/.venv/bin:$PATH"

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
   --custom-config-path            "${PROJECT_ROOT}/train/awm_config.yaml"
   --prompt-data  "${PROJECT_ROOT}/outputs/test_gen_by_claude/verified/task_final.jsonl"
   --input-key  prompt
   --label-key  label
   --apply-chat-template
   --rollout-shuffle
   --num-rollout              2          # L3: bridge-mode fresh load reports loaded_rollout_id=0 →
                                         # start_rollout_id=1, and slime's loop is range(start, num_rollout).
                                         # num_rollout=1 → range(1,1)=∅ (no train step!). num_rollout=2 →
                                         # range(1,2)=[1] → exactly one real rollout+GRPO step.
   --rollout-batch-size       8          # 8 prompts/tasks per rollout step
   --n-samples-per-prompt     8          # GRPO group size = 8 (→ 8×8 = 64 trajectories/step)
   --rollout-max-context-len  32768      # cap total trajectory length at 32K (was 128K → some rambled to 44K).
                                         # trained via tp4×cp4: 32K sequence → 8K/GPU.
   --rollout-max-response-len 8192
   --rollout-temperature      1.0
   --balance-data
)

TRAIN_ARGS=(
   --config "${PROJECT_ROOT}/train/config_train27.yaml"   # 27B, full training (actor real)
   --qkv-format bshd                                      # non-packed layout: megatron's GDN (gated-delta-net)
                                                          # layers reject packed sequences (qkv_format=thd);
                                                          # bshd requires use_dynamic_batch_size=false (set in config).
   --colocate                                             # actor+sglang share the 4 GPUs (time-shared via
                                                          # auto offload); routes weight-sync through the
                                                          # megatron-bridge weight iterator (correct for this
                                                          # VLM/GDN checkpoint — the non-colocate distributed
                                                          # path uses a hand-written converter that lacks the
                                                          # bridge-built vision_model.* / language_model.* naming).
   --dump-details "${PROJECT_ROOT}/save/qwen3.6-27b/rollout"
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
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}"
} }
EOF_JSON
)

# entrypoint runs in ray-head cwd (PROJECT_ROOT); ABSOLUTE paths, VENV python.
ray job submit --address="http://127.0.0.1:${RAY_DASH_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${VENV_PY}" "${SLIME_DIR}/train_multi_policy.py" \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
   "${MODEL_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${TRAIN_ARGS[@]}"
