#!/usr/bin/env sh
set -eu

# Usage: scripts/build_and_push_images.sh <aws-account-id> <region> <repo-prefix>
# Builds docker/ray-*-uv images and pushes to ECR: <repo-prefix>-ray-cpu-uv and <repo-prefix>-ray-gpu-uv

ACCOUNT_ID=${1:-}
REGION=${2:-us-east-1}
PREFIX=${3:-plasmidrl}
if [ -z "$ACCOUNT_ID" ]; then
  echo "Usage: $0 <aws-account-id> [region] [repo-prefix]" >&2
  exit 1
fi

CPU_REPO=${PREFIX}-ray-cpu-uv
GPU_REPO=${PREFIX}-ray-gpu-uv
CPU_URI=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${CPU_REPO}
GPU_URI=${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${GPU_REPO}

echo "Ensure ECR repos exist: $CPU_REPO, $GPU_REPO"
aws ecr describe-repositories --repository-names "$CPU_REPO" --region "$REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$CPU_REPO" --image-scanning-configuration scanOnPush=true --region "$REGION"
aws ecr describe-repositories --repository-names "$GPU_REPO" --region "$REGION" >/dev/null 2>&1 || \
  aws ecr create-repository --repository-name "$GPU_REPO" --image-scanning-configuration scanOnPush=true --region "$REGION"

echo "Login to ECR"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

echo "Build CPU image"
docker build -f docker/ray-cpu-uv.Dockerfile -t ${CPU_URI}:latest .
echo "Build GPU image"
docker build -f docker/ray-gpu-uv.Dockerfile -t ${GPU_URI}:latest .

echo "Push images"
docker push ${CPU_URI}:latest
docker push ${GPU_URI}:latest

echo "Images pushed:"
echo "  CPU: ${CPU_URI}:latest"
echo "  GPU: ${GPU_URI}:latest"

