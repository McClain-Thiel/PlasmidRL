# PlasmidRL Hyperparameter Sweeps

This folder now documents the code-driven sweep runner. Metadata (parameters, ranges, steps)
now live in `src/config.py` via `SweepConfig`, and the entry point is `src.runners.grpo_sweep`.

## Structure

```
sweeps/
├── SWEEPS.md        # Detailed documentation (this file mirrors the original content)
└── README.md        # This file (entry doc)
```

## Quick Start

```bash
docker compose up grpo-sweep
```

The `grpo-sweep` service syncs dependencies, starts `python -m src.runners.grpo_sweep`, and
uses the hard-coded `SweepConfig` ranges (learning rate, generations, reward weights, etc.)
to run Optuna trials that maximize `reward_components/total_reward/mean` over ~100 steps per trial.

Each trial automatically logs to W&B under the `PlasmidRL` project. No sweep CLI setup is needed.

## Sweep Options

Update `src/config.py` → `SweepConfig` to:

- adjust `n_trials`, `timeout_minutes`, or `training_steps`
- change each search space (`learning_rate_range`, `per_device_train_batch_size_choices`, `temperature_range`, etc.)
- tweak reward weight choices (`reward_ori_weight_choices`, …)
- choose between the two baked-in length bonus settings under `reward_length_configs`
- control logging frequency (`log_frequency`) or evaluation cadence (`eval_steps`, `eval_strategy`)

## Related Files

- **Sweep runner:** `src/runners/grpo_sweep.py` – builds trials from `SweepConfig`, logs per-trial W&B runs, and saves checkpoints to `/s3/.../grpo-sweeps/`
- **Reward logger:** `src/rewards/bioinformatics/logger.py`
- **Docker service:** `docker-compose.yaml` (`grpo-sweep`)
- **Doc:** `SWEEPS.md` (detailed narrative below)

