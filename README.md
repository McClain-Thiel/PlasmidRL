# PlasmidRL

Reinforcement learning experiments for plasmid design. Train language models to generate functional DNA sequences using various RL algorithms.

## Quick Start

**Prerequisites:**
- NVIDIA GPU with Docker + Compose
- `.env` file with `WANDB_API_KEY` and `HF_TOKEN`

Run training with Docker Compose:
```bash
docker compose up {service_name}
```

Or use the CLI directly:
```bash
python -m src.main --help
```

## Docker Services

- **`dev`** – Interactive shell for debugging
- **`grpo-trl`** – GRPO training using HuggingFace TRL
- **`es`** – Evolution Strategies training
- **`verl-ppo`** – PPO training via VERL (config in `config/verl_ppo.yaml`)
- **`verl-grpo`** – GRPO training via VERL (config in `config/verl_grpo.yaml`)
- **`infer`** – Generate samples and upload to S3

## CLI Commands

```bash
# Train with different algorithms
python -m src.main train-es         # Evolution Strategies
python -m src.main train-grpo       # GRPO with TRL

# Generate samples
python -m src.main generate-samples

# Convert checkpoints to HuggingFace format
python -m src.main convert-checkpoint \
  --checkpoint-path s3://bucket/path \
  --hf-repo username/model-name
```

## Config

- Environment variables in `.env`
- Python config in `src/config.py`
- VERL configs in `config/*.yaml`

