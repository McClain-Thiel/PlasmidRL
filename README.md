PlasmidRL — Experiments and Runners

This repository collects several training/optimization entrypoints for plasmid-related RL experiments. All jobs run via Docker Compose and log to separate Weights & Biases (W&B) projects with clear tags and notes.

Prerequisites
- NVIDIA GPU host with Docker + Compose
- `.env` file at repo root with secrets:
  - `WANDB_API_KEY=...`
  - `HF_TOKEN=...` (optional)

Services
- `dev`
  - Interactive shell in the project container.
  - Run: `docker compose run --rm dev`

- `grpo-trl`
  - TRL GRPO training loop using this repo’s TRL runner.
  - W&B: project `plasmidrl-trl-grpo`, tags `[plasmid, rl, trl, grpo]`.
  - Run: `docker compose up grpo-trl`
  - Resume: `RESUME=true docker compose up grpo-trl`

- `es`
  - TRL Evolution Strategies optimization.
  - W&B: project `plasmidrl-trl-es`, tags `[plasmid, rl, trl, es]`.
  - Run: `docker compose up es`

- `verl-ppo`
  - VERL PPO trainer using `docker/verl.Dockerfile` (base: `verlai/verl`).
  - W&B: project `plasmidrl-verl-ppo`, tags `[plasmid, rl, verl, ppo]`.
  - Config: `config/verl_ppo.yaml` mapped into VERL’s expected path.
  - Run: `docker compose up verl-ppo`

- `verl-grpo`
  - VERL GRPO trainer using `docker/verl.Dockerfile`.
  - W&B: project `plasmidrl-verl-grpo`, tags `[plasmid, rl, verl, grpo]`.
  - Config: `config/verl_grpo.yaml` mapped into VERL’s expected path.
  - Run: `docker compose up verl-grpo`

- `eval`
  - Optional evaluation entrypoint if `src/cli` supports it.
  - W&B: project `plasmidrl-eval`, tags `[eval]`.
  - Run: `docker compose up eval`

Notes
- All services inherit environment from `.env` and expose all GPUs.
- W&B entity is hard-coded to `mcclain`; adjust in `docker-compose.yaml` if needed.
- VERL services mount config YAMLs into `/opt/verl/verl/trainer/config/`.

Ray on EKS (external access)
- Ray cluster head service (LoadBalancer):
  - Host: ac898ed8d0f3441a29315b985b15a9fa-897444232.us-east-1.elb.amazonaws.com
  - Client: `ray.init("ray://ac898ed8d0f3441a29315b985b15a9fa-897444232.us-east-1.elb.amazonaws.com:10001")`
  - Dashboard: `http://ac898ed8d0f3441a29315b985b15a9fa-897444232.us-east-1.elb.amazonaws.com:8265`
- IP restrictions: LoadBalancer is locked to specific IPs via `loadBalancerSourceRanges`.
  - Your current IP has been allowlisted.
  - To add another IP: `scripts/secure_head_lb.sh <NEW_IP>/32`
  - To allow multiple: `scripts/secure_head_lb.sh 198.51.100.10/32,203.0.113.25/32`
- For private-only access (optional): switch to an internal LB or use a VPN/bastion; see `devops.md`.
