#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$SCRIPT_DIR/config.env" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/config.env"
fi

mkdir -p "$CACHE_DIR/uv" "$CACHE_DIR/pip" "$HF_HOME" "$HF_HUB_CACHE" "$TMP_DIR" "$OUTPUT_NEMO_DIR"

docker run --rm \
  ${DOCKER_GPU_FLAG:-} ${DOCKER_IPC_FLAG:-} ${DOCKER_ULIMIT_FLAGS:-} \
  --shm-size "${SHM_SIZE:-8g}" \
  -e XDG_CACHE_HOME="$CACHE_DIR" \
  -e UV_CACHE_DIR="$UV_CACHE_DIR" \
  -e PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  -e TMPDIR="$TMP_DIR" \
  -e HF_HOME="$HF_HOME" \
  -e HF_HUB_CACHE="$HF_HUB_CACHE" \
  -v /mcclain:/mcclain \
  -v "$PROJECT_HOST_DIR":"$WORKDIR" \
  -w "$WORKDIR" \
  "$IMAGE" bash -lc "evo2_convert_to_nemo2 --model-path '$MODEL_PATH' --model-size '$MODEL_SIZE' --output-dir '$OUTPUT_NEMO_DIR'"

echo "Converted model saved to: $OUTPUT_NEMO_DIR"
docker run --rm -it --gpus all -e XDG_CACHE_HOME=/mcclain/.cache -e UV_CACHE_DIR=/mcclain/.cache/uv -e PIP_CACHE_DIR=/mcclain/.cache/pip -e TMPDIR=/mcclain/tmp -e HF_HOME=/mcclain/.cache/huggingface -e HF_HUB_CACHE=/mcclain/.cache/huggingface/cache -v /mcclain:/mcclain -v /mcclain/projects/PlasmidRL:/workspace -w /workspace nvcr.io/nvidia/clara/bionemo-framework:2.6.3 bash -lc "uv run --extra sft evo2_convert_to_nemo2 --model-path hf://arcinstitute/savanna_evo2_1b_base --model-size 1b --output-dir /mcclain/models/nemo2_evo2_1b_8k"