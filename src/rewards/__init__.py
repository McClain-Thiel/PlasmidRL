from .rewards import score_completions
# The reward functions are expected to accept:

# completions: list of model outputs for the batch
# other keyword arguments corresponding to columns in your dataset (e.g. solution, ground_truth, image, etc.)
# Optionally, **kwargs to absorb extra parameters (like trainer_state)
# this always has the same signature so I just define it here once.
def Score(
    completions: list,
    **kwargs,
) -> list[float] | tuple[list[float], list[dict]]:
    return_breakdown = kwargs.get('return_breakdown', False)
    
    if return_breakdown:
        raw_scores, breakdowns = score_completions(completions, return_breakdown=True)
        # Map the heuristic 0-100 score to 0-1 so higher is better while preserving scale
        scaled_scores = [max(0.0, min(1.0, score / 100.0)) for score in raw_scores]
        return scaled_scores, breakdowns
    else:
        raw_scores = score_completions(completions)
        # Map the heuristic 0-100 score to 0-1 so higher is better while preserving scale
        return [max(0.0, min(1.0, score / 100.0)) for score in raw_scores]
