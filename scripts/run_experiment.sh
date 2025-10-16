#!/usr/bin/env sh
set -eu

# Usage: scripts/run_experiment.sh <experiment-name>
# Reads infra/kuberay/experiments.yaml and renders a RayJob manifest.

EXP_NAME=${1:-}
if [ -z "$EXP_NAME" ]; then
  echo "Usage: $0 <experiment-name>" >&2
  exit 1
fi

CFG_FILE=${CFG_FILE:-infra/kuberay/experiments.yaml}
NAMESPACE=${NAMESPACE:-default}
CLUSTER_LABEL=${CLUSTER_LABEL:-plasmidray}
TTL_SECONDS_DEFAULT=${TTL_SECONDS_DEFAULT:-0}

if [ ! -f "$CFG_FILE" ]; then
  echo "Config file not found: $CFG_FILE" >&2
  exit 1
fi

# Optional env passthrough
WANDB_ENTITY=${WANDB_ENTITY:-}
WANDB_PROJECT=${WANDB_PROJECT:-}
WANDB_TAGS=${WANDB_TAGS:-}
WANDB_NOTES=${WANDB_NOTES:-}
HF_TOKEN=${HF_TOKEN:-}

TMP=$(mktemp)
trap 'rm -f "$TMP"' EXIT

EXP_NAME="$EXP_NAME" CFG_FILE="$CFG_FILE" CLUSTER_LABEL="$CLUSTER_LABEL" \
WANDB_ENTITY="$WANDB_ENTITY" WANDB_PROJECT="$WANDB_PROJECT" WANDB_TAGS="$WANDB_TAGS" WANDB_NOTES="$WANDB_NOTES" HF_TOKEN="$HF_TOKEN" \
python3 - <<'PY' > "$TMP"
import os, sys, yaml, textwrap

exp_name = os.environ.get("EXP_NAME")
cfg_file = os.environ.get("CFG_FILE")
cluster_label = os.environ.get("CLUSTER_LABEL", "plasmidray")
if not exp_name or not cfg_file:
    print("Missing EXP_NAME or CFG_FILE", file=sys.stderr)
    sys.exit(2)

with open(cfg_file) as f:
    data = yaml.safe_load(f) or {}
exp = (data.get("experiments") or {}).get(exp_name)
if not exp:
    print(f"Experiment '{exp_name}' not found in {cfg_file}", file=sys.stderr)
    sys.exit(3)

entrypoint = exp.get("entrypoint", "")
ttl = int(exp.get("ttl_seconds", 0) or 0)
env_map = exp.get("env") or {}

# Inject optional WANDB/HF envs if provided
for k in ("WANDB_ENTITY","WANDB_PROJECT","WANDB_TAGS","WANDB_NOTES","HF_TOKEN"):
    v = os.environ.get(k)
    if v:
        env_map[k] = v

runtime_env_yaml = ""
if env_map:
    lines = [f"  {k}: {v}" for k,v in env_map.items()]
    runtime_env_yaml = "\n".join(lines)

doc = f"""
apiVersion: ray.io/v1
kind: RayJob
metadata:
  name: {exp_name}
  namespace: default
spec:
  entrypoint: |-
{textwrap.indent(entrypoint.strip(), '    ')}
  clusterSelector:
    ray.io/cluster: {cluster_label}
  submissionMode: K8sJobMode
  ttlSecondsAfterFinished: {ttl}
""".rstrip() + "\n"

if runtime_env_yaml:
    doc += "  runtimeEnvYAML: |\n" + runtime_env_yaml + "\n"

sys.stdout.write(doc)
PY

echo "Submitting experiment '$EXP_NAME'"
kubectl apply -f "$TMP"
echo "Use: kubectl get rayjobs.ray.io $EXP_NAME -n $NAMESPACE"
