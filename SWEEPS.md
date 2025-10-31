# Hyperparameter Sweeps with W&B

This directory contains configuration for running hyperparameter sweeps using Weights & Biases.

## Quick Start

### 1. Initialize a Sweep

```bash
wandb sweep sweep_config.yaml
```

This will output a sweep ID like: `mcclain/plasmidrl-grpo-sweeps/abc123xyz`

### 2. Run Sweep Agent(s)

**Single agent:**
```bash
SWEEP_ID=mcclain/plasmidrl-grpo-sweeps/abc123xyz docker compose up grpo-sweep
```

**Multiple parallel agents (recommended for faster sweeps):**
```bash
SWEEP_ID=mcclain/plasmidrl-grpo-sweeps/abc123xyz docker compose up --scale grpo-sweep=3
```

### 3. Monitor Progress

Visit your W&B dashboard:
- https://wandb.ai/mcclain/plasmidrl-grpo-sweeps

## Sweep Configuration

The sweep is configured in `sweep_config.yaml` with the following parameters:

### Training Hyperparameters
- `learning_rate`: 1e-6 to 1e-4 (log uniform)
- `per_device_train_batch_size`: 8, 16, or 32
- `num_generations`: 4, 8, or 16
- `temperature`: 0.7 to 1.4
- `top_p`: 0.85 to 0.95

### GRPO Parameters
- `beta`: 1e-4 to 1e-2 (KL penalty coefficient)
- `epsilon`: 0.1 to 0.3 (PPO-style clipping)

### Reward Configuration
- **Constraints**: min/max lengths, counts for each component
- **Weights**: Can be set to 0.0 to disable a component
  - `reward_ori_weight`: 0.0, 0.5, 1.0, or 2.0
  - `reward_promoter_weight`: 0.0, 0.5, 1.0, or 2.0
  - `reward_terminator_weight`: 0.0, 0.25, 0.5, or 1.0
  - `reward_marker_weight`: 0.0, 0.5, 1.0, or 2.0
  - `reward_cds_weight`: 0.0, 0.5, 1.0, or 2.0
- **Penalties**: `reward_punish_mode`, `reward_length_penalty`
- **Features**: `reward_location_aware` (cassette bonuses)

## Optimization Strategy

The sweep uses **Bayesian optimization** to intelligently explore the hyperparameter space:
- **Maximizes**: `reward_components/total_reward/mean` - mean reward across all training steps
- **Quick evaluation**: Each run completes in ~100 steps for fast iteration
- **Early termination**: Hyperband stops poorly performing runs after 20 steps
- **Smart search**: Focuses on promising hyperparameter regions

## Customizing Sweeps

Edit `sweep_config.yaml` to:
1. Change the search method (`bayes`, `grid`, `random`)
2. Add/remove parameters
3. Adjust parameter ranges
4. Change the optimization metric
5. Modify early termination settings

## Best Practices

1. **Start small**: Run a few trials manually to validate config
2. **Use multiple agents**: Parallel agents speed up sweeps significantly
3. **Monitor early**: Check results after 5-10 runs to ensure valid config
4. **Stop bad sweeps**: If all runs fail, stop and fix the config
5. **Save good configs**: Note the best hyperparameters for future runs

## Troubleshooting

**Sweep not starting:**
- Verify W&B login: `wandb login`
- Check SWEEP_ID format includes entity/project/id

**All runs failing:**
- Check logs: `docker compose logs grpo-sweep`
- Verify reward config constraints are achievable

**Out of memory:**
- Reduce `per_device_train_batch_size` in sweep config
- Reduce `num_generations` parameter range

