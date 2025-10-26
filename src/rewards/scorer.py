from rewards.reward_config import RewardConfig
import plasmidkit as pk
import pandas as pd
from typing import Any
from concurrent.futures import ThreadPoolExecutor


class Scorer:
    def __init__(self, reward_config: RewardConfig):
        self.reward_config = reward_config
        self.score_functions = [self.score_ori, self.score_promoter, self.score_terminator, self.score_marker, self.score_cds]
        self.weights = [self.reward_config.ori_weight, self.reward_config.promoter_weight, self.reward_config.terminator_weight, self.reward_config.marker_weight, self.reward_config.cds_weight]
        total_weight = sum(self.weights)
        if total_weight:
            self.weights = [w / total_weight for w in self.weights]
        else:
            uniform_weight = 1.0 / len(self.score_functions)
            self.weights = [uniform_weight for _ in self.score_functions]

    def annotate(self, sequence: str) -> Any:
        return pk.annotate(sequence, is_sequence=True)

    def runner(self, sequence: str, annotations: Any) -> list[float]:
        """
        Takes a list of runnable functions with the same signature and runs them in parallel.

        Args:
            sequence: The sequence to score
            annotations: The annotations to score

        Returns:
            A list of scores in the same order as `self.score_functions`.
        """
        with ThreadPoolExecutor(max_workers=len(self.score_functions)) as executor:
            futures = [executor.submit(func, sequence, annotations) for func in self.score_functions]
            return [future.result() for future in futures]

    def score_ori(self, seq: str, annotations: pd.DataFrame) -> float:
        return 0.0

    def score_promoter(self, seq: str, annotations: pd.DataFrame) -> float:
        return 0.0

    def score_terminator(self, seq: str, annotations: pd.DataFrame) -> float:
        return 0.0

    def score_marker(self, seq: str, annotations: pd.DataFrame) -> float:
        return 0.0

    def score_cds(self, seq: str, annotations: pd.DataFrame) -> float:
        return 0.0

    def score_length(self, seq: str, annotations: pd.DataFrame) -> float:
        return 0.0

    def score(self, seq: str, annotations: pd.DataFrame) -> float:
        results = self.runner(seq, annotations)
        return sum(w * r for w, r in zip(self.weights, results))