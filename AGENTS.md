# AGENTS.md – PlasmidRL Briefing for AI Coding Agents

This document provides context and conventions for AI agents working on PlasmidRL. It complements the [README.md](README.md) for human contributors.

---

## Project Overview

**PlasmidRL** applies modern LLM techniques to biological language models. The core goal: train DNA language models using Reinforcement Learning (RL) to generate functional plasmids, progressing from unconditional to conditional generation.

### Domain Context

- **Plasmid**: Circular DNA molecules, often used in synthetic biology and genetic engineering
- **DNA Language Model**: A transformer-based model that learns to predict and generate DNA sequences (treating DNA like a language with 4-letter alphabet: A, T, G, C)
- **RL for Plasmid Design**: Uses reward signals (e.g., informatics-based metrics) to steer generation toward functional sequences
- **Key Metrics**:
  - GFP cassette presence/functionality
  - Sequence validity (no premature stops, proper annotations)
  - Plasmid size and structural properties

---

## Environment Setup & Dependencies

### Package Manager

**UV** (Rust-based Python package manager) is the primary dependency manager. All Python dependencies are frozen in `uv.lock`.

- **Frozen install**: `uv sync --frozen`
- **Interactive install** (development): `uv sync`
- **Running commands**: `uv run <command>`

### Required Environment Variables

Create a `.env` file in the repository root with:

```bash
HF_TOKEN=<your_huggingface_api_token>        # Required for model downloads/uploads
WANDB_API_KEY=<your_wandb_api_key>           # Required for experiment tracking
```

Optional (already in docker-compose):
- `WANDB_ENTITY` – Weights & Biases team/entity
- `WANDB_PROJECT` – Weights & Biases project name
- `RESUME` – Set to `true` to resume training from checkpoint

### Python Version

- **Required**: Python ≥ 3.11
- **Specified in**: `.python-version` (for pyenv/asdf)

### System-Level Dependencies

The main `Dockerfile` and `docker/verl.Dockerfile` handle most system setup, including:

- **PyTorch with CUDA 12.1** support (GPU required)
- **Micromamba** environment for bioinformatics tools (BLAST, DIAMOND, Infernal)
- **uv** for Python package management
- **Git, curl, build-essential** for compilation

**Bioinformatics tools** (BLAST, DIAMOND, Infernal) are installed via conda in a separate micromamba environment to avoid Python dependency conflicts.

---

## Build & Test

### Local Development (Outside Docker)

```bash
# Sync frozen dependencies
uv sync --frozen

# Run linting
uv run ruff check src/ tests/

# Run type checking
uv run mypy src/

# Run tests
uv run pytest
```

### Docker Development

```bash
# Enter interactive dev shell with all tools installed
docker compose up dev

# Run training inside container (TRL GRPO)
docker compose up grpo-trl

# Run VERL PPO training
docker compose up verl-ppo
```

### Quality Checks (CI/CD)

Before committing:

1. **Linting** – `uv run ruff check src/`
2. **Type checking** – `uv run mypy src/`
3. **Tests** – `uv run pytest` (if available)
4. **Black formatting** – `uv run black --check src/`

All checks must pass before creating a PR.

---

## Architecture & Code Conventions

### Project Layout

```
PlasmidRL/
├── src/
│   ├── main.py                # Click CLI entry point
│   ├── config.py              # Pydantic config (loads .env)
│   ├── runners/               # Training/inference entry points
│   │   ├── grpo.py            # TRL GRPO trainer
│   │   ├── grpo_sweep.py      # W&B sweep runner for GRPO
│   │   ├── es.py              # TRL Evolution Strategies trainer
│   │   └── generate_samples.py # vLLM-based inference
│   ├── rewards/               # Reward computation & scoring
│   │   ├── rewards.py         # Reward signal definitions
│   │   ├── plasmid_informatics.py  # Plasmid analysis (validity, GFP, etc.)
│   │   ├── verl_reward.py     # VERL-compatible reward wrapper
│   │   └── bioinformatics/    # Bioinformatics-based reward system
│   │       ├── scorer.py      # Core scoring logic
│   │       ├── reward_config.py # Reward configuration
│   │       └── logger.py      # W&B logging callback
│   ├── eval/                  # Evaluation utilities
│   └── utils/                 # Helper functions (model utils, S3 access)
├── sweeps/                    # Hyperparameter sweep configurations
│   ├── configs/               # W&B sweep YAML configs
│   │   ├── sweep_config.yaml            # Full sweep (training + reward)
│   │   ├── sweep_config_training.yaml   # Training hyperparameters only
│   │   └── sweep_config_refined.yaml    # Refined ranges (500 steps)
│   ├── run_sweep_agent.py     # W&B agent wrapper script
│   ├── README.md              # Quick reference
│   └── SWEEPS.md              # Detailed sweep documentation
├── config/                    # YAML configs for VERL trainers
│   ├── verl_ppo.yaml
│   ├── verl_grpo.yaml
│   ├── verl_naive_ppo.yaml
│   └── naive_grpo.yaml
├── docker/                    # Specialized Dockerfiles
│   ├── verl.Dockerfile        # VERL training environment
│   └── verl-monkey.patch      # VERL patches/customizations
├── Dockerfile                 # Main development/TRL container
├── docker-compose.yaml        # Multi-container orchestration
├── pyproject.toml             # UV/Python project metadata
├── uv.lock                    # Frozen dependency lock file
├── AGENTS.md                  # This file - AI agent briefing
└── README.md                  # User-facing documentation
```

### Code Style

- **Python**: Follow PEP 8; formatted with **Black**
- **Linting**: **Ruff** (fast, strict)
- **Type hints**: Required for function signatures; MyPy validates
- **Naming**:
  - `snake_case` for functions/variables
  - `UPPER_CASE` for constants
  - `PascalCase` for classes
- **Config management**: Use Pydantic `BaseSettings` + `.env` for env vars (see `src/config.py`)

### Entry Points

#### CLI Entry Points (Click)

**File**: `src/main.py`

Define new commands as Click group decorators:

```python
@cli.command("my-command")
@click.option("--my-flag", default=False)
def my_command(my_flag):
    """Description of what my_command does."""
    from src.runners.my_runner import main
    main()
```

Then run via `uv run python -m src.main my-command --my-flag`.

#### Docker Compose Entry Points

**File**: `docker-compose.yaml`

Add a new service by defining a section under `services:`:

```yaml
my-trainer:
  <<: *common          # Inherit base config (GPU, volumes, env_file)
  command: >
    bash -lc "
      uv sync --frozen && uv run python -m src.runners.my_trainer
    "
  environment:
    - WANDB_PROJECT=plasmidrl-my-trainer
    - WANDB_TAGS=[\"my-tag\"]
```

Then run via `docker compose up my-trainer`.

**Notes**:
- Use `<<: *common` to inherit GPU setup, volumes, env_file, and base image
- Mount datasets/models via `volumes:` (e.g., S3 paths)
- Set `WANDB_*` environment variables for experiment tracking

---

## Configuration Management

### Runtime Config (`src/config.py`)

All runtime settings are defined in `Config` class (Pydantic `BaseSettings`). Loads from:

1. `.env` file (highest priority)
2. Environment variables
3. Hardcoded defaults in the class

**Key settings**:
- `model` – Base HF model ID (default: `McClain/plasmidgpt-addgene-gpt2`)
- `huggingface_token` – HF API token (from env: `HF_TOKEN`, `HUGGINGFACE_TOKEN`)
- `wandb_api_key` – Weights & Biases token
- `s3_bucket` – S3 path for checkpoints/outputs
- `default_query` – GFP cassette sequence (hardcoded reference sequence)
- `informatics_server_url` – External informatics service endpoint

When modifying config, update `src/config.py` directly (not hardcoded in runners).

### Training Config (YAML)

**VERL trainers** use YAML configs:
- `config/verl_ppo.yaml` – PPO hyperparams for VERL
- `config/verl_grpo.yaml` – GRPO hyperparams for VERL
- `config/naive_grpo.yaml` – Experimental configs

**Note**: Reward functions and configs change frequently. When modifying reward logic or hyperparameters, update the respective YAML or runner file, then document changes in commit messages.

---

## Reward Functions & Scoring

**Location**: `src/rewards/`

The reward system is modular and subject to frequent changes. Here's the structure:

- **`rewards.py`** – Core reward signal definitions (high-level)
- **`plasmid_informatics.py`** – Plasmid validity checks (GFP detection, stops, annotations)
- **`verl_reward.py`** – VERL-compatible reward wrapper

**Key concepts**:
- Rewards are computed per-sequence during training
- Signals may include: GFP cassette presence, sequence validity, evolutionary fitness
- Changes to reward logic are decoupled from trainer code (easier iteration)

When adding new reward functions:
1. Implement in `src/rewards/rewards.py` or `plasmid_informatics.py`
2. Update trainer configs or runner code to use the new reward function
3. Document intent and any new dependencies in commit message

---

## External Services & Integrations

### Weights & Biases (W&B)

- **Purpose**: Experiment tracking, logging metrics, model versioning
- **Setup**: Add `WANDB_API_KEY` to `.env`
- **Configuration**: Set `WANDB_ENTITY` and `WANDB_PROJECT` per runner in `docker-compose.yaml`
- **Scope**: Each runner logs to a different project for organization

### Hugging Face Hub

- **Purpose**: Model hosting, dataset management, tokenizer downloads
- **Setup**: Add `HF_TOKEN` to `.env`
- **Usage**: `transformers` and `trl` libraries automatically use `HF_TOKEN`
- **Scope**: Download pretrained models, upload fine-tuned checkpoints

### S3 Storage

- **Purpose**: Checkpoint storage, large dataset I/O
- **Setup**: Mount via `/mnt/s3/phd-research-storage-1758274488` (in docker-compose)
- **Config**: `s3_bucket`, `region_name` in `src/config.py`
- **Usage**: `boto3` client handles S3 access; see `src/utils/` for helpers

### Informatics Server

- **Purpose**: External service for plasmid validation/scoring
- **Endpoint**: `informatics_server_url` in `src/config.py` (default: `http://server:8080`)
- **Note**: May not always be running; check `plasmid_informatics.py` for graceful fallbacks

---

## Testing & Validation

### Unit Tests

Place tests in `tests/` directory (if available) matching source structure:

```
tests/
├── test_rewards.py        # Test reward computation
├── test_config.py         # Test config loading
└── test_runners.py        # Test runner initialization
```

Run with:

```bash
uv run pytest -v
```

### Integration Testing

Docker Compose services can be tested in isolation:

```bash
# Start dev shell, run a quick trainer step
docker compose up dev
# Inside container: uv run python -m src.main train-grpo --max-steps 1
```

### Pre-Commit Hooks

If `.pre-commit-config.yaml` is present, install hooks:

```bash
pre-commit install
```

Hooks run linting/type checks before each commit.

---

## Troubleshooting for Agents

### Common Issues

1. **`ModuleNotFoundError` when running locally**
   - Run `uv sync --frozen` first
   - Ensure you're using `uv run` to execute scripts

2. **Docker build fails**
   - Check GPU availability: `docker compose config | grep -A 5 nvidia`
   - Ensure `.env` exists (dummy values OK for build)
   - VERL Dockerfile may require extra build context; check `docker/verl.Dockerfile`

3. **CUDA out of memory**
   - Reduce batch size in runner config
   - Check `docker-compose.yaml` for `per_device_train_batch_size`

4. **Missing S3 access**
   - Verify `/mnt/s3/` mount is available in docker-compose
   - Check AWS credentials if using external S3

### Debug Mode

```bash
# Enter interactive shell with all deps installed
docker compose up dev

# Inside container
uv run python -c "from src.config import Config; print(Config())"
uv run python -m src.main --help
```

---

## Common Workflow

### Adding a New Trainer

1. **Create runner**: `src/runners/my_trainer.py` with `main()` function
2. **Add CLI command**: Register in `src/main.py` with Click decorator
3. **Add Docker service**: Define in `docker-compose.yaml` using `<<: *common`
4. **Test locally**: `uv run python -m src.main my-command`
5. **Test in Docker**: `docker compose up my-trainer`
6. **Commit**: Include runner file, CLI update, and docker-compose changes in one PR

### Modifying Reward Functions

1. **Edit reward logic**: Update `src/rewards/rewards.py` or `plasmid_informatics.py`
2. **Test in isolation**: Create a quick test script or notebook
3. **Update trainer config** (if hyperparams change): Modify YAML in `config/`
4. **Re-run training**: Launch via docker-compose or CLI
5. **Track results**: W&B automatically logs metrics for comparison

### Debugging Training

1. **Check W&B dashboard** for live metrics, loss curves, gradients
2. **Inspect checkpoints** in S3 or local output directory
3. **Review logs** from docker-compose or runner stderr
4. **Run with smaller dataset** or fewer steps for quick iteration

### Running Hyperparameter Sweeps

**Location**: `sweeps/`

W&B sweeps enable automated hyperparameter optimization. The project includes three pre-configured sweep strategies:

1. **Refined sweep** (`sweeps/configs/sweep_config_refined.yaml`) - **Recommended**
   - 500 steps per trial for stable evaluation
   - Narrow ranges around best performers from initial exploration
   - Fixed batch_size=16, num_generations=8 (proven best combo)
   
2. **Training-only sweep** (`sweeps/configs/sweep_config_training.yaml`)
   - 100 steps per trial for quick iteration
   - Explores all training hyperparameters
   - Fixed reward configurations
   
3. **Full sweep** (`sweeps/configs/sweep_config.yaml`)
   - 100 steps per trial
   - Includes both training and reward parameter tuning
   - Most comprehensive but slower convergence

**Workflow:**

```bash
# 1. Initialize sweep (from project root)
wandb sweep sweeps/configs/sweep_config_refined.yaml

# 2. Run agent(s) with returned SWEEP_ID
SWEEP_ID=<sweep-id> docker compose up grpo-sweep

# 3. Optional: Run multiple parallel agents for faster sweeps
SWEEP_ID=<sweep-id> docker compose up --scale grpo-sweep=3

# 4. Monitor at https://wandb.ai/mcclain/plasmidrl-grpo-sweeps
```

**Key files:**
- `sweeps/configs/` - Sweep configuration YAMLs
- `sweeps/run_sweep_agent.py` - W&B agent wrapper
- `src/runners/grpo_sweep.py` - Main training script called by W&B
- `sweeps/SWEEPS.md` - Detailed sweep documentation

**Tips:**
- Start with `sweep_config_refined.yaml` if you've done initial exploration
- Use multiple agents (`--scale grpo-sweep=3`) to parallelize trials
- Each 100-step trial takes ~5-10 minutes; 500-step trials take ~25-50 minutes
- Document findings in `sweeps/README.md` for future reference

---

## References & Links

- [Weights & Biases Docs](https://docs.wandb.ai/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [TRL (Transformers Reinforcement Learning)](https://huggingface.co/docs/trl/)
- [VERL Framework](https://github.com/volcengine/verl)
- [vLLM Inference](https://docs.vllm.ai/)
- [uv Package Manager](https://docs.astral.sh/uv/)

---

**Last Updated**: October 2025  
**Maintainer**: McClain Thiel
