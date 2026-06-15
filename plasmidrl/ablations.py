"""Reward-ablation configurations for the GRPO trainer.

Each entry is a RewardConfig that differs from `full_reward` only in which
reward components are enabled. All training hyperparameters (learning rate,
batch size, generations per prompt, ...) remain identical across ablations.
"""

from plasmidrl.rewards.bioinformatics.reward_config import RewardConfig


# Production reward weights selected by the Optuna sweep. Do not edit without
# rerunning the sweep — the ablations are defined relative to these values.
_PRODUCTION_BASE = dict(
    violation_penalty_factor=0,
    punish_mode=True,
    length_reward_mode=True,
    min_length=2000,
    max_length=30000,
    ideal_min_length=3000,
    ideal_max_length=6000,
    length_reward_bonus=0.7085046275614012,  # exact Optuna output; do not round
    ori_min=1,
    ori_max=1,
    ori_weight=1.0,
    promoter_min=1,
    promoter_max=5,
    promoter_weight=1.0,
    terminator_min=0,
    terminator_max=2,
    terminator_weight=0.5,
    marker_min=1,
    marker_max=2,
    marker_weight=1.0,
    cds_min=1,
    cds_max=2,
    cds_weight=1.0,
    location_aware=True,
    repeat_penalty_enabled=True,
    repeat_min_length=50,
    repeat_penalty_per_region=0.1,
)

# Weight used for "turned off but not removed" components in the *_only configs,
# so shapes of the reward match the full reward (avoids divide-by-zero etc.).
_EPSILON = 0.001


def _config(**overrides) -> RewardConfig:
    return RewardConfig(**{**_PRODUCTION_BASE, **overrides})


_ABLATION_CONFIGS = {
    "full_reward": _config(),
    "no_repeat_penalty": _config(repeat_penalty_enabled=False),
    "no_length_prior": _config(length_reward_mode=False),
    "no_cassette_bonus": _config(location_aware=False),
    "cds_only": _config(
        ori_weight=_EPSILON,
        promoter_weight=_EPSILON,
        terminator_weight=_EPSILON,
        marker_weight=_EPSILON,
        cds_weight=1.0,
        length_reward_mode=False,
        repeat_penalty_enabled=False,
        location_aware=False,
    ),
    "length_only": _config(
        ori_weight=_EPSILON,
        promoter_weight=_EPSILON,
        terminator_weight=_EPSILON,
        marker_weight=_EPSILON,
        cds_weight=_EPSILON,
        length_reward_mode=True,
        repeat_penalty_enabled=False,
        location_aware=False,
    ),
}

ABLATION_NAMES: list[str] = list(_ABLATION_CONFIGS.keys())


def get_ablation_config(name: str) -> RewardConfig:
    if name not in _ABLATION_CONFIGS:
        raise KeyError(
            f"Unknown ablation config {name!r}. Valid: {', '.join(ABLATION_NAMES)}"
        )
    return _ABLATION_CONFIGS[name]
