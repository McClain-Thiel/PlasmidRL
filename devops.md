Ray on EKS (Ops Guide)

- What/Why: Ray on Amazon EKS via KubeRay for on-demand CPU/GPU autoscaling. Keep a small head node online; burst to G6/P6 GPUs only when needed; scale back to save cost. Prometheus + Grafana included for metrics.

- Connect kubectl: `aws eks update-kubeconfig --name plasmidrl-eks --region us-east-1`

- Ray endpoints (current):
  - Client: `ray.init("ray://ac898ed8d0f3441a29315b985b15a9fa-897444232.us-east-1.elb.amazonaws.com:10001")`
  - Dashboard: `http://ac898ed8d0f3441a29315b985b15a9fa-897444232.us-east-1.elb.amazonaws.com:8265` (HTTP)
  - Debug NodePort: `30643` on any node external IP

- Security: Head LB allowlist is configurable (`loadBalancerSourceRanges`).
  - Tighten: `scripts/secure_head_lb.sh <ip-or-cidr>`
  - Currently open for troubleshooting; re‑lock when done.

- Autoscaling: Workers scale from 0; nodes terminate ~5–7 min after idle.
  - Fast start tip: temporarily set `gpu-g6-workers` minReplicas=1, then revert to 0.

- Experiments: define in `infra/kuberay/experiments.yaml`; submit via `scripts/run_experiment.sh <name>`.
  - WANDB/HF envs pass via runtime env; see examples in the file.

How to
- 1) Set up a new run
  - Edit `infra/kuberay/experiments.yaml` and add under `experiments:`:
    - `entrypoint`: e.g., `uv sync --frozen && uv run python -m src.runners.my_exp`
    - `needs_gpu`: true/false; `gpu_family`: g6 or p6; `num_gpus`: 0/1/2…
    - `ttl_seconds`: auto cleanup; `env`: WANDB project/tags/notes
  - Submit: `scripts/run_experiment.sh <your-name>`
  - Watch: `kubectl get pods -l ray.io/cluster=plasmidray -n default` and the Ray dashboard.

- 2) Run a current project
  - CPU (ES): `scripts/run_experiment.sh es`
  - GPU (TRL GRPO on G6): `scripts/run_experiment.sh grpo-trl`
  - GPU (VERL PPO): `scripts/run_experiment.sh verl-ppo`
  - Logs: `kubectl logs -f job/<rayjob-driver> -n default`

- Metrics: kube-prometheus-stack in `monitoring` (Grafana exposed; Prometheus internal).
  - Ray head is configured to point to Grafana/Prometheus so charts render in the dashboard.
