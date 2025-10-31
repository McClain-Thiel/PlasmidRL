from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, List
import logging
import time

import plasmidkit as pk

from src.rewards.scorer import Scorer
from src.rewards.reward_config import RewardConfig


logger = logging.getLogger("reward_logger")

# Toggle detailed timing logs
try:
    from src.config import Config
    _REWARD_LOG_TIMINGS = bool(Config().reward_log_timings)
except Exception:
    _REWARD_LOG_TIMINGS = False

_SCORER = Scorer(RewardConfig())


def annotate_completions(completions: list[str]) -> list[Any]:
    """Annotate a flat list of completions using threads; strips spaces from sequences."""
    if _REWARD_LOG_TIMINGS:
        t0 = time.perf_counter()
    sequences = [s.replace(" ", "") for s in completions]
    with ThreadPoolExecutor() as executor:
        annotate = partial(pk.annotate, is_sequence=True)
        annotations = list(executor.map(annotate, sequences))
    if _REWARD_LOG_TIMINGS:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"reward.annotate n={len(completions)} time_ms={dt_ms:.2f}")
    return annotations


def score_sequence(sequence: str) -> float:
    """Config-driven score in [0, 100] using Scorer (auto-annotates)."""
    if not sequence or not str(sequence).strip():
        return 0.0
    score01, _ = _SCORER.score(sequence)
    return float(max(0.0, min(100.0, 100.0 * score01)))


def score_completions(completions: list[str]) -> list[float]:
    if _REWARD_LOG_TIMINGS:
        t0 = time.perf_counter()
    if not completions:
        logger.warning("reward.score_completions called with empty completions list")
        return []

    scores: List[float] = []
    non_empty_indices: List[int] = []
    non_empty_completions: List[str] = []

    for i, c in enumerate(completions):
        if not c or len(c.strip()) == 0:
            scores.append(0.0)
        else:
            non_empty_indices.append(i)
            non_empty_completions.append(c)
            scores.append(0.0)  # placeholder

    if non_empty_completions:
        for idx in non_empty_indices:
            scores[idx] = score_sequence(completions[idx])

    if _REWARD_LOG_TIMINGS:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        n = len(scores)
        mean_score = sum(scores) / n if n else 0.0
        logger.info(
            f"reward.score n={n} mean={mean_score:.2f} time_ms={dt_ms:.2f}"
        )
    return scores
