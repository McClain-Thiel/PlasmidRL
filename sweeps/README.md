# PlasmidRL Hyperparameter Sweeps

This directory contains all files related to W&B hyperparameter sweeps for GRPO training.

## Structure

```
sweeps/
├── configs/                                   # Sweep configuration files
│   ├── sweep_config.yaml                     # Full sweep (training + reward)
│   ├── sweep_config_training.yaml            # Training hyperparameters only
│   ├── sweep_config_refined.yaml             # Refined ranges (500 steps)
│   ├── sweep_config_training_with_length.yaml # Training + length rewards ⭐ NEW
│   └── sweep_config_length_reward.yaml       # Length reward testing only
├── run_sweep_agent.py                        # Wrapper script for W&B agent
├── SWEEPS.md                                 # Detailed documentation
└── README.md                                 # This file
```

## Quick Start

```bash
# 1. Initialize sweep (from project root)
wandb sweep sweeps/configs/sweep_config_training_with_length.yaml

# 2. Run agent with the returned SWEEP_ID
SWEEP_ID=<your-sweep-id> docker compose up grpo-sweep

# Or run multiple agents in parallel for faster sweeps:
SWEEP_ID=<your-sweep-id> docker compose up --scale grpo-sweep=3

# 3. Monitor at https://wandb.ai/mcclain/plasmidrl-grpo-sweeps
```

## Which Config to Use?

- **`sweep_config_training_with_length.yaml`** ⭐ **RECOMMENDED FOR NEXT SWEEP**
  - 500 steps per trial for stable evaluation
  - Broad training hyperparameter search
  - Tests 2 length reward configurations:
    - Smaller plasmids: 2-15kb (ideal: 3-12kb)
    - Larger plasmids: 5-30kb (ideal: 7-20kb)
  - Great for finding optimal hyperparameters with length rewards

- **`sweep_config_refined.yaml`** - Use for fine-tuning without length rewards
  - 500 steps per trial for stable results
  - Narrow ranges around best performers from initial sweep
  - Fixed: batch_size=16, num_generations=8
  
- **`sweep_config_length_reward.yaml`** - Use to explore length reward ranges only
  - 500 steps per trial
  - Fixed: best training hyperparameters
  - Variable: ideal length ranges and bonus multipliers
  
- **`sweep_config_training.yaml`** - Use for broad exploration (no length rewards)
  - 100 steps per trial for quick iteration
  - Wide ranges for all training params
  - Fixed reward configs
  
- **`sweep_config.yaml`** - Use to explore reward weights
  - 100 steps per trial
  - Includes reward weight tuning
  - Most comprehensive but slowest convergence

## Related Files

- **Sweep runner:** `src/runners/grpo_sweep.py` - Main training script called by W&B agent
- **Reward logger:** `src/rewards/bioinformatics/logger.py` - Logs detailed metrics to W&B
- **Docker service:** `docker-compose.yaml` (grpo-sweep service)

## Documentation

See [SWEEPS.md](./SWEEPS.md) for detailed documentation on:
- Configuration options
- Optimization strategy
- Best practices
- Troubleshooting

## Sweep Results

After running sweeps, document key findings here:

### Sweep 1: Initial Exploration (Complete)
- **Best config:** Batch 16 + 8 generations
- **Learning rate:** 4e-05 to 8e-05 range optimal
- **Epsilon:** 0.27-0.30 (high exploration helps)
- **Beta:** 0.0008-0.0023 range works well
- **Success rate:** 84% (127/151 runs)
- **Best reward:** 0.86

### Sweep 2: Refined Training (In Progress)
- **Status:** Running
- **Config:** `sweep_config_refined.yaml`
- **Notes:** 500 steps per trial for more stable evaluation

