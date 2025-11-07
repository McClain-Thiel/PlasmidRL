# Hyperparameter Sweeps with W&B

This directory contains configuration for running hyperparameter sweeps using Weights & Biases.

## Quick Start

### 1. Initialize a Sweep

```bash
# From project root
wandb sweep sweeps/configs/sweep_config_refined.yaml
```

This will output a sweep ID like: `mcclain/plasmidrl-grpo-sweeps/abc123xyz`

**Copy this sweep ID** - you'll need it for the next step.

### 2. Run Sweep Agent(s) in Docker

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

## Available Sweep Configurations

All sweep configs are in `configs/`:

### 1. `sweep_config_refined.yaml` (Recommended)
**Best for:** Finding optimal hyperparameters based on initial sweep insights
- **Duration:** 500 steps per trial (stable evaluation)
- **Focus:** Refined ranges around best performers
- **Fixed:** Batch size 16, 8 generations (best combo)
- **Tuned:** Learning rate, beta, epsilon, temperature, top_p

### 2. `sweep_config_training_with_length.yaml` (RECOMMENDED FOR NEXT SWEEP)
**Best for:** Full hyperparameter search with length rewards
- **Duration:** 500 steps per trial
- **Focus:** All training hyperparameters + length rewards
- **Fixed:** Standard reward component weights
- **Tuned:** LR, batch size, generations, temperature, top_p, beta, epsilon, length bonus
- **Length configs:** Tests 2 combinations:
  - Smaller plasmids: min=2000, ideal=3000-12000, max=15000
  - Larger plasmids: min=5000, ideal=7000-20000, max=30000

### 3. `sweep_config_length_reward.yaml`
**Best for:** Testing length-based reward strategies only
- **Duration:** 500 steps per trial
- **Focus:** Length reward parameters (ideal ranges, bonus multipliers)
- **Fixed:** Best training hyperparameters from previous sweeps
- **Tuned:** Ideal length ranges, length reward bonus
- **Feature:** Rewards sequences within ideal length ranges with bonuses

### 4. `sweep_config_training.yaml`
**Best for:** Broad exploration of training hyperparameters
- **Duration:** 100 steps per trial (quick evaluation)
- **Focus:** Training parameters only
- **Fixed:** All reward configurations
- **Tuned:** Full ranges for LR, batch size, generations, etc.

### 5. `sweep_config.yaml`
**Best for:** Full exploration including reward weights
- **Duration:** 100 steps per trial
- **Focus:** Both training and reward parameters
- **Tuned:** Everything (training + reward configs)

## Sweep Configuration Details

The sweep configurations include the following parameters:

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
- **Penalties**: `reward_punish_mode`
- **Features**: 
  - `reward_location_aware` (cassette bonuses)
  - `reward_length_reward_mode` (length-based rewards)

### Length Reward System (NEW)
When `reward_length_reward_mode` is enabled:
- **Base reward (1.0)**: Sequences within `[min_length, max_length]`
- **Bonus reward**: Sequences within `[ideal_min_length, ideal_max_length]` get up to `1.0 + length_reward_bonus`
- **Partial bonus**: Sequences between min/ideal or ideal/max get proportional bonus
- **Penalty**: Sequences outside acceptable range get `violation_penalty_factor` or 0.5

Example configuration:
```yaml
reward_length_reward_mode: true
reward_min_length: 1000      # Minimum acceptable
reward_max_length: 30000     # Maximum acceptable
reward_ideal_min_length: 3000  # Ideal minimum
reward_ideal_max_length: 10000 # Ideal maximum
reward_length_reward_bonus: 0.5  # +50% bonus for ideal range
```

## Checkpoint Management

Each sweep run automatically saves checkpoints to S3:
- **Location**: `/mnt/s3/phd-research-storage-1758274488/checkpoints/grpo-sweeps/{run_name}/`
- **Naming**: Run name matches W&B run name (e.g., `grpo-sweep-20251103_153516`)
- **Strategy**: Saves every 100 steps, keeps last 2 checkpoints
- **Contents**: Model weights + tokenizer

To find the best checkpoint from a sweep:
1. Identify the best run in W&B dashboard
2. Note the run name (visible in W&B UI)
3. Find checkpoint at `/mnt/s3/.../checkpoints/grpo-sweeps/{run_name}/`

## Optimization Strategy

The sweep uses **Bayesian optimization** to intelligently explore the hyperparameter space:
- **Maximizes**: `reward_components/total_reward/mean` - mean reward across all training steps
- **Quick evaluation**: Each run completes in ~100 steps for fast iteration
- **Early termination**: Hyperband stops poorly performing runs after 20 steps
- **Smart search**: Focuses on promising hyperparameter regions

## Directory Structure

```
sweeps/
├── configs/                                        # Sweep configuration files
│   ├── sweep_config.yaml                          # Full sweep (training + reward)
│   ├── sweep_config_training.yaml                 # Training hyperparameters only
│   ├── sweep_config_refined.yaml                  # Refined ranges (500 steps)
│   ├── sweep_config_training_with_length.yaml     # Training + length rewards (RECOMMENDED)
│   └── sweep_config_length_reward.yaml            # Length reward testing only
├── run_sweep_agent.py                             # Wrapper script for W&B agent
├── README.md                                      # Quick reference
└── SWEEPS.md                                      # This file - detailed documentation
```

## Customizing Sweeps

Edit any config in `configs/` to:
1. Change the search method (`bayes`, `grid`, `random`)
2. Add/remove parameters
3. Adjust parameter ranges
4. Change the optimization metric
5. Modify early termination settings

## Best Practices

1. **Start small**: Run a few trials manually to validate config
2. **Use multiple agents**: Parallel agents speed up sweeps significantly
   - Each run takes ~5-10 minutes (100 steps)
   - 3 parallel agents = ~50 trials in 3-4 hours
3. **Monitor early**: Check results after 5-10 runs to ensure valid config
4. **Stop bad sweeps**: If all runs fail, stop and fix the config
5. **Save good configs**: Note the best hyperparameters for future runs
6. **Full training**: Once you find best params, run full training (5000+ steps) with those settings

## Troubleshooting

**Sweep initialization fails:**
- Check W&B login: `wandb login`
- Verify you're logged into the correct W&B entity

**Sweep not starting:**
- Verify SWEEP_ID format includes entity/project/id (e.g., `mcclain/plasmidrl-grpo-sweeps/abc123`)
- Check you copied the full sweep ID from initialization output
- Ensure `.env` file has `WANDB_API_KEY`

**All runs failing:**
- Check logs: `docker compose logs grpo-sweep`
- Verify reward config constraints are achievable
- Run a single step manually: `docker compose run --rm grpo-sweep`

**Out of memory:**
- Reduce `per_device_train_batch_size` in sweep config
- Reduce `num_generations` parameter range
- Stop other running containers first

