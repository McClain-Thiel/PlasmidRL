# Hyperparameter Sweeps (code-driven)

The legacy YAML-based sweep configs have been retired in favor of the code-defined `SweepConfig`
inside `src/config.py`. The current workflow runs `src.runners.grpo_sweep` via Docker so that the
sweep behaves like any other service (no manual CLI `wandb sweep` step required).

## Getting started

1. Confirm your `.env` includes `WANDB_ENTITY`, `WANDB_PROJECT`, and `WANDB_API_KEY`.
2. Start the sweep service:
   ```bash
   docker compose up grpo-sweep
   ```
3. The runner launches an Optuna study, logs each trial to W&B (`PlasmidRL` project), and uses
   the `SweepConfig` ranges to sample training, generation, and reward parameters.

You can scale the service just like other entries:
```bash
docker compose up --scale grpo-sweep=3
```
Each container contributes to separate Optuna trials but writes all checkpoints to `/mnt/s3/.../checkpoints/grpo-sweeps/`.

## SweepConfig summary

Managed inside `src/config.py`, these knobs dictate every trial:

- **Schedule**
  - `n_trials`, `timeout_minutes`
  - `training_steps` (set on each GRPO config via `max_steps`)
  - `eval_strategy`, `eval_steps`, `log_frequency`
- **Training search space**
  - `learning_rate_range` (log uniform 1e-6 → 1e-4)
  - `per_device_train_batch_size_choices` (e.g., 8, 16, 32)
  - `num_generations_choices` (e.g., 4, 8, 16)
  - `temperature_range` (0.7 → 1.3)
  - `top_p_range` (0.85 → 0.95)
  - `beta_range` (1e-4 → 1e-2) and `epsilon_range` (0.1 → 0.3)
- **Reward search space**
  - `reward_*_weight_choices`: `ori`, `promoter`, `terminator`, `marker`, `cds`
  - `reward_length_reward_mode_options`: toggles length bonuses
  - `reward_length_configs`: two tuples for small + large plasmids
  - `sampling_params`: `max_tokens=256`, `temperature=0.95`, `top_p=0.9`
- **Logging & objective**
  - `direction="maximize"`
  - Metric: `reward_components/total_reward/mean`
  - Each trial logs to W&B with its checkpoint path in `/mnt/s3/.../grpo-sweeps/{run_name}/trial-{trial}`

Editing the ranges inside `SweepConfig` is the primary way to customize a sweep. The runner is
hard-coded to save best checkpoints every 100 steps and keep the last 2.

## Optimization strategy

- **Optimizer**: Optuna (Bayesian sampling) running for `n_trials` or until `timeout_minutes`.
- **Metric**: `reward_components/total_reward/mean` (maximizes mean reward per training step).
- **Duration**: ~100 GRPO steps per trial for fast feedback; evaluation runs every 50 steps by default.
- **Reward variations**: each trial samples component weights and, optionally, one of the two length bonus scenarios.

## Checkpoint management

- **Target**: `/mnt/s3/phd-research-storage-1758274488/checkpoints/grpo-sweeps/{run_name}/trial-{trial}`
- **Frequency**: Saves every 100 steps, keeps the last 2 checkpoints in each trial directory.
- **Contents**: Model weights and tokenizer (same as regular training).
- **Best run**: Identify the best trial in W&B and read its run name; checkpoints live under that directory.

## Troubleshooting

- **W&B X failures**: Ensure the `.env` tokens are valid and `uv sync --frozen` has run successfully.
- **GPU OOM**: Reduce `per_device_train_batch_size_choices` or `num_generations_choices` in `SweepConfig`.
- **Sweep stuck**: Restart the `grpo-sweep` container; Optuna writes study state under `~/.optuna`.
- **Need shorter runs**: Lower `training_steps` or `eval_steps` in `SweepConfig`.

