#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$SCRIPT_DIR/config.env" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/config.env"
fi

# Enforce WANDB_API_KEY presence
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is required. Export it or set it in sft/config.env." >&2
    exit 1
fi

mkdir -p "$CACHE_DIR/uv" "$CACHE_DIR/pip" "$HF_HOME" "$HF_HUB_CACHE" "$TMP_DIR" "${EXPERIMENT_DIR:-}"

DATASET_CONFIG_PATH="${1:-$TRAIN_DATASET_CONFIG}"

docker run --rm \
  ${DOCKER_GPU_FLAG:-} ${DOCKER_IPC_FLAG:-} ${DOCKER_ULIMIT_FLAGS:-} \
  --shm-size "${SHM_SIZE:-8g}" \
  -e XDG_CACHE_HOME="$CACHE_DIR" \
  -e UV_CACHE_DIR="$UV_CACHE_DIR" \
  -e PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  -e TMPDIR="$TMP_DIR" \
  -e HF_HOME="$HF_HOME" \
  -e HF_HUB_CACHE="$HF_HUB_CACHE" \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e WANDB_ENTITY="$WANDB_ENTITY" \
  -e WANDB_PROJECT="$WANDB_PROJECT" \
  ${EXPERIMENT_DIR:+-e WANDB_DIR=${EXPERIMENT_DIR}} \
  ${EXPERIMENT_DIR:+-e NEMO_LOG_DIR=${EXPERIMENT_DIR}} \
  -v /efs:/efs \
  -v "$PROJECT_HOST_DIR":"$WORKDIR" \
  -w "$WORKDIR" \
  "$IMAGE" bash -lc "python -m bionemo.evo2.run.train \
    --dataset-config '$DATASET_CONFIG_PATH' \
    --devices ${DEVICES:-1} \
    --num-nodes ${NUM_NODES:-1} \
    --seq-length ${SEQ_LENGTH:-8192} \
    --model-size ${MODEL_SIZE:-1b} \
    --micro-batch-size ${MICRO_BATCH_SIZE:-1} \
    --grad-acc-batches ${GRAD_ACC_BATCHES:-1} \
    --tensor-parallel-size ${TP_SIZE:-1} \
    --pipeline-model-parallel-size ${PP_SIZE:-1} \
    --context-parallel-size ${CP_SIZE:-1} \
    --workers ${WORKERS:-8} \
    ${CLIP_GRAD:+--clip-grad ${CLIP_GRAD}} \
    ${DISABLE_CHECKPOINTING:+--disable-checkpointing ${DISABLE_CHECKPOINTING}} \
    ${ACT_CKPT_LAYERS:+--activation-checkpoint-recompute-num-layers ${ACT_CKPT_LAYERS}} \
    ${FP8:+--fp8} \
    ${CKPT_DIR:+--ckpt-dir ${CKPT_DIR}} \
    --wandb-entity \"${WANDB_ENTITY:-}\" \
    --wandb-project \"${WANDB_PROJECT:-}\" \
    --wandb-tags ${WANDB_TAGS:-} \
    --wandb-group \"${WANDB_GROUP:-}\" \
    --wandb-job-type \"${WANDB_JOB_TYPE:-}\" \
    --wandb-run-name \"${WANDB_RUN_NAME:-}\" \
    "

echo "Training launched with dataset config: $DATASET_CONFIG_PATH"

