#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-infra/eks/cluster.yaml}
CLUSTER=${2:-plasmidrl-eks}
REGION=${3:-us-east-1}

echo "Adding nodegroups to cluster $CLUSTER in $REGION using $CFG"
eksctl create nodegroup --config-file "$CFG" --include gpu-g6-ondemand --region "$REGION"
eksctl create nodegroup --config-file "$CFG" --include gpu-p6-spot --region "$REGION"
