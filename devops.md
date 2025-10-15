Ray on EKS (quick start)

- Cluster: `plasmidrl-eks` (us-east-1). KubeRay operator, Cluster Autoscaler, and NVIDIA device plugin are installed.
- Ray service account: `default/ray` with IRSA to S3 (AmazonS3FullAccess). Ray pods can use `boto3` without keys.
- Shared memory: head/worker pods mount `/dev/shm` as an in‑memory volume sized to the pod memory (2–16Gi).

Connect kubectl
- `aws eks update-kubeconfig --name plasmidrl-eks --region us-east-1`
- `kubectl config use-context arn:aws:eks:us-east-1:<account>:cluster/plasmidrl-eks`

Open Ray dashboard (local)
- `kubectl -n default port-forward svc/plasmidray-head-svc 8265:8265`
- Visit http://localhost:8265

Connect Ray client (local)
- `kubectl -n default port-forward svc/plasmidray-head-svc 10001:10001`
- Python: `ray.init("ray://localhost:10001")`

Expose head via LoadBalancer (optional)
- Already enabled in our manifest: `serviceType: LoadBalancer` on the head service.
- Restrict access: `scripts/secure_head_lb.sh <your.ip.addr>/32 [default]`
  - Example: `scripts/secure_head_lb.sh 203.0.113.45/32`
  - This sets `loadBalancerSourceRanges` so the LB only accepts from your IP(s).
  - For VPC-only access, set the service to internal with the AWS LB annotation (ask if you want this).
  - Connect: `ray.init("ray://<EXTERNAL-IP>:10001")`; dashboard on `:8265`.

Verify S3 access from a Ray pod
- `kubectl -n default exec -it deploy/plasmidray-head -- python - <<'PY'`
  `import boto3; s3=boto3.client('s3'); print([b['Name'] for b in s3.list_buckets()['Buckets']][:5])`
  `PY`

GPU and autoscaling
- Submit tasks that request CPU/GPU; KubeRay scales worker groups, and Cluster Autoscaler adds EC2 nodes (spot for workers).
- GPU pool defaults: g5/g6. Availability varies per AZ; scale may queue briefly.

Experiment workflow
- Define experiments in one config: `infra/kuberay/experiments.yaml` (keeps CLI simple).
  - Examples included: `grpo-trl`, `es`, `verl-ppo` with entrypoint, GPU need, TTL, and W&B tags.
- Submit by name:
  - CPU: `scripts/run_experiment.sh es`
  - GPU (G6 on-demand): `scripts/run_experiment.sh grpo-trl`
  - GPU (P6 spot): set `gpu_family: p6` in config and `num_gpus: 1`, then `scripts/run_experiment.sh <name>`
- In Ray code, you can read hints:
  - `os.getenv('EXP_GPU_FAMILY')` and `os.getenv('EXP_NEEDS_GPU')`
  - Or simply rely on `@ray.remote(num_gpus=...)` and the cluster will scale accordingly.

Build a new experiment (detailed)
- Copy an entry in `infra/kuberay/experiments.yaml` and change:
  - `entrypoint`: your command (e.g., `uv run python -m src.runners.my_exp`)
  - `needs_gpu`: true/false; set `gpu_family: g6` (on-demand) or `p6` (spot, expensive, burst only)
  - `num_gpus`: 0/1/2… as needed; adjust `ttl_seconds` (auto cleanup)
  - `env`: W&B names/tags, notes, tokens as needed (these are injected only into the driver)
- Run it: `scripts/run_experiment.sh <your-experiment-name>`
- Observe:
  - `kubectl get pods -l ray.io/cluster=plasmidray -n default`
  - `kubectl logs -f job/<rayjob-driver> -n default`
- Cleanup happens automatically after `ttlSecondsAfterFinished`.

GPU pools
- G6 on-demand pool (default): moderate cost, reliable capacity.
- P6 spot pool (optional): very high performance, low cost but preemptible; keep `maxSize` small.
- We can further split Ray worker groups and advertise custom resources (e.g., `{g6:1}`, `{p6:1}`) if you want code-level selection.

Notes
- By default, the cluster uses a single GPU worker pool. For strict GPU family pinning (g5 vs g6), we can split node groups and Ray worker groups per family and tag Ray workers with custom resources (e.g., `{"g5":1}` / `{"g6":1}`). Then experiments select with `resources={"g5":1}`.
- To monitor jobs: `kubectl get pods -l ray.io/cluster=plasmidray -A` and `kubectl logs -f job/<rayjob-driver>`.
Overview
- This repo runs Ray on Amazon EKS with KubeRay for on-demand, autoscaling compute (CPU + GPUs). It lets you keep a small, cheap head node online and burst up to G6/P6 GPUs only when experiments need them, then scale back down to save cost. Prometheus + Grafana are included for metrics, and a simple config file defines experiments you can submit quickly.

Operate
- Context: `aws eks update-kubeconfig --name plasmidrl-eks --region us-east-1`
- Ray head (client/dashboard):
  - Client: `ray.init("ray://<LB_HOST>:10001")`
  - Dashboard: `http://<LB_HOST>:8265`
  - Current: see README for the active `<LB_HOST>`; allowed IPs can be restricted or opened.
- Security: we can lock the head LB to specific CIDRs (`loadBalancerSourceRanges`) using `scripts/secure_head_lb.sh`; open/close as needed.
- Autoscaling: GPU/CPU workers scale from 0 to meet Ray demands; nodes terminate a few minutes after idle. For instant GPU start, temporarily set `gpu-g6-workers` minReplicas to 1, then revert.
- Experiments: define in `infra/kuberay/experiments.yaml`, submit with `scripts/run_experiment.sh <name>`; WANDB/HF envs pass via runtime env.
- Metrics: kube-prometheus-stack deployed in `monitoring`; Grafana LB is restricted to your IP; Ray head is configured with Prometheus/Grafana so time-series charts render in the dashboard.
