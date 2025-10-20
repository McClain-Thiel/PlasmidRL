PlasmidRL — Experiments and Runners

This repository collects several entrypoints for plasmid-related RL experiments. There are two ways to interact with the code 1.) run via Docker Compose (preffered) and  2.) uv cli. 

## Docker Compose Entrypoints
When using docker entry points all config is done in config files (config.py or ~/config/*.yaml)

Long running processes are kicked off with `docker compose up {service_name}`

### Services

* dev
  * simple interactive bash shell just for debugging

* grpo-trl
  * runs src/runners/grpo. GRPO implemented using huggingface TRL library

* es
  * evoutionary strategy from that one paper. runs src/runners/es.py

* verl-ppo
  * uses the VERL docker container to run PPO. No code just the config file in config/verl_ppo.yaml
  * docker compose env var args include [WANDB_ENTITY, WANDB_PROJECT, WANDB_TAGS, WANDB_NOTES, HF_TOKEN, WANDB_API_KEY]

* verl-grpo
  * uses the VERL docker container to run GRPO. No code just the config file in config/verl_ppo.yaml
  * docker compose env var args include [WANDB_ENTITY, WANDB_PROJECT, WANDB_TAGS, WANDB_NOTES, HF_TOKEN, WANDB_API_KEY]

* infer
  * takes a model defined in config.py (sample_model) and prompts it for samples then uploads to s3. 



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

