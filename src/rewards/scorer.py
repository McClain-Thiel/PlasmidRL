from rewards.reward_config import RewardConfig
import plasmidkit as pk
import pandas as pd


class Scorer:
    def __init__(self, reward_config: RewardConfig):
        self.reward_config = reward_config
        self.score_functions = [self.score_ori, self.score_promoter, self.score_terminator, self.score_marker, self.score_cds]
        self.weights = [self.reward_config.ori_weight, self.reward_config.promoter_weight, self.reward_config.terminator_weight, self.reward_config.marker_weight, self.reward_config.cds_weight]

    def annotate(self, sequence: str) -> Any:
        return pk.annotate(sequence, is_sequence=True)

    def runner(self, sequence: str, annotations: Any) -> float:
        """
        Takes a list of runnable functions with the same signature and runs them in parallel.

        Args:
            sequence: The sequence to score
            annotations: The annotations to score

        Returns:
            The score
        """
        with ThreadPoolExecutor(max_workers=len(self.score_functions)) as executor:
            futures = [executor.submit(func, sequence, annotations) for func in self.score_functions]
            return [future.result() for future in futures]
        pass

    def score_ori(self, annotations: pd.DataFrame) -> float:
        return 0.0

    def score_promoter(self, annotations: pd.DataFrame) -> float:
        return 0.0

    def score_terminator(self, annotations: pd.DataFrame) -> float:
        return 0.0

    def score_marker(self, annotations: pd.DataFrame) -> float:
        return 0.0

    def score_cds(self, annotations: pd.DataFrame) -> float:
        return 0.0