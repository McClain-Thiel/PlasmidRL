"""
VERL-compatible reward function using local plasmidkit annotations.
Adapted from src/rewards/rewards.py to work with VERL's expected interface.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Iterable, List, Tuple, Optional, Dict

try:
    import plasmidkit as pk
except ImportError:
    pk = None

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)

# Toggle detailed timing logs
try:
    from src.config import Config
    _REWARD_LOG_TIMINGS = bool(Config().reward_log_timings)
except Exception:
    _REWARD_LOG_TIMINGS = False


def annotate_completions(completions: List[str]) -> List[Any]:
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


def score_sequence(
    sequence: str,
    annotations: Any,
    target_keywords: Iterable[str] | None = None,
    return_components: bool = False,
) -> float | Tuple[float, Dict[str, float]]:
    """
    Heuristic backbone score for GRPO.

    Rewards:
      • ORI present: +20 for first ORI, -10 for each additional ORI.
      • Marker present: +10 for having at least one marker.
      • Up to two highest-scoring cassettes with partial credit:
          - promoter→CDS: order (+5) + proximity (≤100bp:+5, ≤300:+3, ≤500:+2, ≤1000:+1).
          - CDS→terminator: order (+5) + proximity (same as above).
          - Out-of-order legs get proximity-only partial (+≤3).
          - +2 if CDS is a marker and promoter is within 300 bp.
      • Standalone promoters (+1 each up to +5).
      • Standalone terminators (+1 each up to +5).
      • Payload (GOI) CDS anywhere: +8, plus +4 if any promoter within 500 bp.
      • Length bonus: shorter sequences favored (linear from +10 at ≤5kb to 0 at ≥30kb).

    Args:
        sequence: DNA sequence to score
        annotations: Plasmidkit annotations
        target_keywords: Keywords for identifying payload genes
        return_components: If True, return (score, components_dict)

    Returns:
        float score [0, 100] if return_components=False
        Tuple[float, Dict[str, float]] if return_components=True
    """
    # ---- collect features (case-insensitive 'type') ----
    def T(x): return (x.type or "").lower()
    feats = list(annotations)
    oris        = [x for x in feats if T(x) in ("rep_origin", "ori", "origin_of_replication")]
    promoters   = [x for x in feats if T(x) == "promoter"]
    cdss        = [x for x in feats if T(x) == "cds"]
    terminators = [x for x in feats if T(x) == "terminator"]
    markers     = [x for x in feats if T(x) == "marker"]

    # ---- small helpers ----
    def strand(x): return getattr(x, "strand", "+")
    def start(x):  return int(getattr(x, "start", 0))
    def end(x):    return int(getattr(x, "end", 0))
    def role(x):   return (getattr(x, "evidence", {}) or {}).get("role", "").lower()

    def in_order_same_strand(a, b) -> bool:
        if strand(a) != strand(b): return False
        if strand(a) == "+": return start(a) <= start(b)
        else:                return end(a)   >= end(b)

    def distance(a, b) -> int:
        if end(a) < start(b): return start(b) - end(a)
        if end(b) < start(a): return start(a) - end(b)
        return 0  # overlapping/adjacent

    def prox_points(d: int, max_pts: int = 5) -> int:
        if d <= 100:  return max_pts
        if d <= 300:  return max_pts - 2   # 3
        if d <= 500:  return max_pts - 3   # 2
        if d <= 1000: return 1
        return 0

    # ---- ORI scoring ----
    score = 0.0
    components = {}
    
    ori_score = 0.0
    if len(oris) >= 1:
        ori_score += 20.0  # +20 for first ORI
        if len(oris) > 1:
            ori_score -= 10.0 * (len(oris) - 1)  # -10 for each additional ORI
    score += ori_score
    components["ori"] = ori_score
    components["ori_count"] = float(len(oris))
    
    # ---- Marker bonus ----
    marker_score = 0.0
    if len(markers) >= 1:
        marker_score = 10.0  # +10 for having at least one marker
    score += marker_score
    components["marker"] = marker_score
    components["marker_count"] = float(len(markers))

    # ---- Cassette scoring (incl. out-of-order partials) ----
    def best_cassettes() -> List[Tuple[Any, Any, Any, int]]:
        triples = []
        for p in promoters:
            cds_same = [c for c in cdss if strand(c) == strand(p)]
            if not cds_same: continue
            cds_same.sort(key=lambda c: distance(p, c))
            c = cds_same[0]

            term_cands = [t for t in terminators if strand(t) == strand(c)]
            # prefer ordered downstream terminators, but allow out-of-order partial credit
            term_cands.sort(key=lambda t: distance(c, t))
            t = term_cands[0] if term_cands else None

            pts = 0
            # promoter -> CDS
            if in_order_same_strand(p, c):
                pts += 5
                pts += prox_points(distance(p, c), 5)
            else:
                pts += min(3, prox_points(distance(p, c), 5))  # out-of-order partial

            # CDS -> terminator
            if t is not None:
                if in_order_same_strand(c, t):
                    pts += 5
                    pts += prox_points(distance(c, t), 5)
                else:
                    pts += min(3, prox_points(distance(c, t), 5))  # out-of-order partial

            if role(c) == "marker" and distance(p, c) <= 300:
                pts += 2

            triples.append((p, c, t, pts))
        triples.sort(key=lambda x: x[3], reverse=True)
        return triples[:2]

    cassette_score = 0.0
    for (_, _, _, pts) in best_cassettes():
        cassette_score += float(min(20, pts))
    score += cassette_score
    components["cassette"] = cassette_score

    # ---- Standalone promoter & terminator credit ----
    promoter_score = 0.0
    if promoters:
        promoter_score = float(min(5.0, 1.0 * len(promoters)))   # +1 each up to +5
    score += promoter_score
    components["promoter"] = promoter_score
    components["promoter_count"] = float(len(promoters))
    
    terminator_score = 0.0
    if terminators:
        terminator_score = float(min(5.0, 1.0 * len(terminators))) # +1 each up to +5
    score += terminator_score
    components["terminator"] = terminator_score
    components["terminator_count"] = float(len(terminators))

    # ---- Payload (GOI) anywhere ----
    target_keywords = [k.lower() for k in (target_keywords or [])]
    def is_payload(c) -> bool:
        r = role(c)
        id_l = (getattr(c, "id", "") or "").lower()
        return r in {"payload", "goi", "reporter"} or (target_keywords and any(k in id_l for k in target_keywords))

    payloads = [c for c in cdss if is_payload(c)]
    payload_score = 0.0
    if payloads:
        payload_score = 8.0
        # +4 if any promoter within 500 bp (either direction; same strand preferred implicitly by proximity)
        if any(distance(p, c) <= 500 for c in payloads for p in promoters):
            payload_score += 4.0
    score += payload_score
    components["payload"] = payload_score
    components["payload_count"] = float(len(payloads))
    components["cds_count"] = float(len(cdss))

    # ---- Length reward: shorter sequences get bonus ----
    L = max(0, len(sequence or ""))

    def length_reward(L: int) -> float:
        # Linear reward from 10 pts at 5kb down to 0 pts at 30kb
        # Sequences > 30kb get 0 reward
        # Sequences <= 5kb get max 10 pts
        if L >= 30000:
            return 0.0
        if L <= 5000:
            return 10.0
        # Linear interpolation between 5kb and 30kb
        return 10.0 * (30000 - L) / (30000 - 5000)

    len_score = length_reward(L)
    score += len_score
    components["length_bonus"] = len_score
    components["sequence_length"] = float(L)

    # ---- Clamp ----
    final_score = float(max(0.0, min(100.0, score)))
    components["total_score"] = final_score
    
    if return_components:
        return final_score, components
    return final_score


def extract_dna_sequence(solution_str: str) -> str:
    """
    Extract DNA sequence from the generated solution string.
    Assumes the model generates DNA sequences (ACGT characters).
    Strips whitespace and non-ACGT characters.
    """
    if not solution_str:
        return ""
    
    # Remove whitespace
    seq = solution_str.strip().replace(" ", "").replace("\n", "").replace("\r", "")
    
    # Filter to only valid DNA bases (case-insensitive)
    valid_bases = set("ACGTacgt")
    seq = "".join(c for c in seq if c in valid_bases)
    
    return seq.upper()


def compute_score(
    data_source: Optional[str],
    solution_str: Optional[str],
    ground_truth: Optional[str],
    extra_info: Optional[Dict] = None,
) -> float:
    """
    VERL-compatible reward function.
    
    This is the main entry point called by VERL's RewardManager.
    
    Args:
        data_source: Dataset name (unused in this implementation)
        solution_str: Generated text from the model (should contain DNA sequence)
        ground_truth: Ground truth from the dataset (unused in this implementation)
        extra_info: Additional info dict (unused in this implementation)
    
    Returns:
        float: Reward score, normalized to [0, 1] range (from 0-100 internal score)
    """
    if _REWARD_LOG_TIMINGS:
        t0 = time.perf_counter()
    
    # Extract DNA sequence from solution
    sequence = extract_dna_sequence(solution_str or "")
    
    if not sequence:
        logger.warning("Empty or invalid DNA sequence in solution_str")
        return 0.0
    

    annotations = pk.annotate(sequence, is_sequence=True)
        
    # Score the sequence (returns 0-100) with component breakdown
    raw_score, components = score_sequence(sequence, annotations, return_components=True)
        
    # Normalize to [0, 1] for VERL
    normalized_score = raw_score / 100.0
    
    # Log reward components for monitoring (at INFO level so they appear in console)
    # Sample logging: only log ~1% of the time to avoid flooding
    import random
    if random.random() < 0.01 or _REWARD_LOG_TIMINGS:
        logger.info(
            f"REWARD_BREAKDOWN: total={raw_score:.2f} | "
            f"ori={components.get('ori', 0):.1f}(n={int(components.get('ori_count', 0))}) "
            f"marker={components.get('marker', 0):.1f}(n={int(components.get('marker_count', 0))}) "
            f"cassette={components.get('cassette', 0):.1f} "
            f"prom={components.get('promoter', 0):.1f}(n={int(components.get('promoter_count', 0))}) "
            f"term={components.get('terminator', 0):.1f}(n={int(components.get('terminator_count', 0))}) "
            f"payload={components.get('payload', 0):.1f}(n={int(components.get('payload_count', 0))}) "
            f"len_bonus={components.get('length_bonus', 0):.1f}({int(components.get('sequence_length', 0))}bp)"
        )
        
    if _REWARD_LOG_TIMINGS:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"reward.compute_score seq_len={len(sequence)} "
            f"raw_score={raw_score:.2f} normalized={normalized_score:.4f} "
            f"time_ms={dt_ms:.2f}"
        )
        
    return float(normalized_score)
        



# Backward compatibility alias
def get_plasmid_reward(
    plasmid: Optional[str] = None,
    *,
    solution_str: Optional[str] = None,
    data_source: Optional[str] = None,
    ground_truth: Optional[str] = None,
    extra_info: Optional[Dict] = None,
    **kwargs,
) -> float:
    """
    Backward compatibility wrapper.
    Delegates to compute_score.
    """
    seq = plasmid or solution_str or ""
    return compute_score(data_source, seq, ground_truth, extra_info)
