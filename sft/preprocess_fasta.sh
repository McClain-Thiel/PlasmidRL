#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$SCRIPT_DIR/config.env" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/config.env"
fi

# Ensure directories exist
mkdir -p "$CACHE_DIR/uv" "$CACHE_DIR/pip" "$HF_HOME" "$HF_HUB_CACHE" "$TMP_DIR" "$PREPROC_OUTPUT_DIR"

# Allow overriding config path via first argument
CONFIG_JSON_PATH="${1:-$PREPROC_JSON}"

# Run inside container
docker run --rm \
  ${DOCKER_GPU_FLAG:-} ${DOCKER_IPC_FLAG:-} ${DOCKER_ULIMIT_FLAGS:-} \
  --shm-size "${SHM_SIZE:-8g}" \
  -e XDG_CACHE_HOME="$CACHE_DIR" \
  -e UV_CACHE_DIR="$UV_CACHE_DIR" \
  -e PIP_CACHE_DIR="$PIP_CACHE_DIR" \
  -e TMPDIR="$TMP_DIR" \
  -e HF_HOME="$HF_HOME" \
  -e HF_HUB_CACHE="$HF_HUB_CACHE" \
  -v /efs:/efs \
  -v "$PROJECT_HOST_DIR":"$WORKDIR" \
  -w "$WORKDIR" \
  "$IMAGE" bash -lc "python - <<'PY'
import json, os, sys
from bionemo.evo2.data.preprocess import Evo2Preprocessor, Evo2PreprocessingConfig
cfg_path = os.environ.get('CFG_JSON', 'sft/preprocess_evo2.json')
print('Loading config from', cfg_path)
with open(cfg_path, 'r') as f:
    cfg_text = f.read()
try:
    cfg = Evo2PreprocessingConfig.model_validate_json(cfg_text)
except Exception:
    cfg = Evo2PreprocessingConfig(**json.loads(cfg_text))
print('Output dir:', cfg.output_dir)
pre = Evo2Preprocessor(cfg)
pre.preprocess_offline(cfg)
print('Done.')
PY
" CFG_JSON="$CONFIG_JSON_PATH"

echo "Preprocessing complete. Output: $PREPROC_OUTPUT_DIR"

