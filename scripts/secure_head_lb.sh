#!/usr/bin/env bash
set -euo pipefail

# Restrict the Ray head LoadBalancer to specific CIDR(s)
# Usage: scripts/secure_head_lb.sh <cidr1>[,<cidr2>...] [namespace]

CIDRS=${1:-}
NS=${2:-default}
if [[ -z "$CIDRS" ]]; then
  echo "Usage: $0 <cidr1>[,<cidr2>...] [namespace]" >&2
  exit 1
fi

IFS="," read -r -a ARR <<< "$CIDRS"
J=$(printf '"%s",' "${ARR[@]}")
J="[${J%,}]"

echo "Patching service plasmidray-head-svc in $NS with loadBalancerSourceRanges=$J"
kubectl -n "$NS" patch svc plasmidray-head-svc -p '{"spec":{"loadBalancerSourceRanges":'"$J"'}}'
kubectl -n "$NS" get svc plasmidray-head-svc -o wide
