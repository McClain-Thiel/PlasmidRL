#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME=${1:-plasmidrl-eks}
REGION=${2:-us-east-1}

echo "Using cluster: $CLUSTER_NAME in $REGION"

echo "Ensuring kubectl points to EKS cluster..."
aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION"

echo "Adding Helm repos (kuberay, nvidia, autoscaler)"
helm repo add kuberay https://ray-project.github.io/kuberay-helm/ >/dev/null 2>&1 || true
helm repo add nvidia https://nvidia.github.io/k8s-device-plugin >/dev/null 2>&1 || true
helm repo add autoscaler https://kubernetes.github.io/autoscaler >/dev/null 2>&1 || true
helm repo update

echo "Installing KubeRay operator..."
kubectl create namespace kuberay-system >/dev/null 2>&1 || true
helm upgrade --install kuberay-operator kuberay/kuberay-operator \
  --namespace kuberay-system

echo "Installing NVIDIA device plugin..."
helm upgrade --install nvidia-device-plugin nvidia/nvidia-device-plugin \
  --namespace kube-system \
  --set tolerations[0].key=nvidia.com/gpu \
  --set tolerations[0].operator=Exists \
  --set tolerations[0].effect=NoSchedule || true

echo "Installing Cluster Autoscaler..."
# Use SA created by eksctl with IRSA
helm upgrade --install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system \
  --set autoDiscovery.clusterName="$CLUSTER_NAME" \
  --set awsRegion="$REGION" \
  --set rbac.serviceAccount.create=false \
  --set rbac.serviceAccount.name=cluster-autoscaler \
  --set extraArgs.balance-similar-node-groups=true \
  --set extraArgs.skip-nodes-with-local-storage=false \
  --set extraArgs.skip-nodes-with-system-pods=false \
  --set extraArgs.scale-down-unneeded-time=5m \
  --set extraArgs.scale-down-delay-after-add=2m \
  --set image.tag=v1.29.3

echo "All addons installed. You can now apply the RayCluster:"
echo "  kubectl apply -f infra/kuberay/raycluster.yaml"
