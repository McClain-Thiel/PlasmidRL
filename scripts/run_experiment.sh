#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/run_experiment.sh <experiment-name>
# Reads infra/kuberay/experiments.yaml and renders a RayJob manifest.

EXP_NAME=${1:-}
if [[ -z "$EXP_NAME" ]]; then
  echo "Usage: $0 <experiment-name>" >&2
  exit 1
fi

CFG_FILE=${CFG_FILE:-infra/kuberay/experiments.yaml}
TEMPLATE=infra/kuberay/templates/rayjob.tmpl.yaml
NAMESPACE=${NAMESPACE:-default}
CLUSTER_NAME=${CLUSTER_NAME:-plasmidray}
TTL_SECONDS_DEFAULT=${TTL_SECONDS_DEFAULT:-7200}

if [[ ! -f "$CFG_FILE" ]]; then
  echo "Config file not found: $CFG_FILE" >&2
  exit 1
fi

# Extract fields via Python to avoid external deps
readarray -t KV < <(python3 - <<'PY'
import os,sys,yaml
cfg_file=os.environ.get('CFG_FILE')
name=os.environ.get('EXP_NAME')
with open(cfg_file) as f:
    data=yaml.safe_load(f)
exp=(data.get('experiments') or {}).get(name)
if not exp:
    print(f"ERR\tExperiment '{name}' not found", file=sys.stderr)
    sys.exit(2)
def out(k,v):
    print(f"KV\t{k}\t{v}")
out('ENTRYPOINT', exp.get('entrypoint',''))
out('NEEDS_GPU', str(bool(exp.get('needs_gpu', False))).lower())
out('GPU_FAMILY', exp.get('gpu_family','none'))
out('TTL_SECONDS', str(exp.get('ttl_seconds', os.environ.get('TTL_SECONDS_DEFAULT','7200'))))
# extra env vars block
env=exp.get('env') or {}
if env:
    # format as additional lines under runtimeEnvYAML with correct indentation
    lines=['    '+k+': '+str(v) for k,v in env.items()]
    print('EXTRA_RUNTIME_ENV\n'+'\n'.join(lines))
PY
)

ENTRYPOINT=""
NEEDS_GPU="false"
GPU_FAMILY="none"
TTL_SECONDS="$TTL_SECONDS_DEFAULT"
EXTRA_RUNTIME_ENV=""
for line in "${KV[@]}"; do
  if [[ "$line" == KV$'\t'ERR* ]]; then
    echo "${line#KV\t}" >&2; exit 2
  fi
  key=${line#KV$'\t'}
  k=${key%%$'\t'*}
  v=${key#*$'\t'}
  case "$k" in
    ENTRYPOINT) ENTRYPOINT="$v";;
    NEEDS_GPU) NEEDS_GPU="$v";;
    GPU_FAMILY) GPU_FAMILY="$v";;
    TTL_SECONDS) TTL_SECONDS="$v";;
    EXTRA_RUNTIME_ENV*) EXTRA_RUNTIME_ENV="$v"$'\n'"${key#*$'\n'}";;
  esac
done

WANDB_ENTITY=${WANDB_ENTITY:-mcclain}
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_TAGS=${WANDB_TAGS:-}
WANDB_NOTES=${WANDB_NOTES:-}
HF_TOKEN=${HF_TOKEN:-}

export NAME=$EXP_NAME NAMESPACE CLUSTER_NAME ENTRYPOINT TTL_SECONDS \
  WANDB_ENTITY WANDB_PROJECT WANDB_TAGS WANDB_NOTES HF_TOKEN \
  GPU_FAMILY NEEDS_GPU EXTRA_RUNTIME_ENV

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT
envsubst < "$TEMPLATE" > "$TMP"

echo "Submitting experiment '$EXP_NAME' to Ray cluster '$CLUSTER_NAME' (ns=$NAMESPACE)"
kubectl apply -f "$TMP"
echo "Use: kubectl get rayjobs.ray.io $EXP_NAME -n $NAMESPACE"
