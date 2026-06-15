from .bioinformatics.scorer import Scorer
from .bioinformatics.reward_config import RewardConfig

# Convenience alias: call as `from plasmidrl.rewards import score` if needed
score = Scorer(RewardConfig()).score