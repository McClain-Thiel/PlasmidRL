from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Iterable, List, Tuple
import logging
import time
import os
import wandb

import plasmidkit as pk

logger = logging.getLogger("reward_logger")

# Toggle detailed timing logs from Config if available
try:
    from src.config import Config  # local import to avoid heavy deps at import time
    _REWARD_LOG_TIMINGS = bool(Config().reward_log_timings)
    _REWARD_LOG_BREAKDOWN = bool(getattr(Config(), "reward_log_breakdown", False))
except Exception:
    _REWARD_LOG_TIMINGS = False
    _REWARD_LOG_BREAKDOWN = False

# Allow env to enable breakdown logs without code changes
if os.getenv("REWARD_LOG_BREAKDOWN"):
    v = os.getenv("REWARD_LOG_BREAKDOWN", "").strip().lower()
    _REWARD_LOG_BREAKDOWN = v in ("1", "true", "yes", "on")


_CURRENT_ITER: int | None = None

def set_reward_iter(step: int | None) -> None:
    global _CURRENT_ITER
    _CURRENT_ITER = int(step) if step is not None else None


def annotate_completions(completions: list[str]) -> list[Any]:
    """Annotate a flat list of completions using threads; strips spaces from sequences."""
    if _REWARD_LOG_TIMINGS:
        t0 = time.perf_counter()
    sequences = ["".join(s.split()).upper() for s in completions]
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
    return_breakdown: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """
    Heuristic backbone score for GRPO.

    Rewards:
      • Single ORI (+20; multiple get +10 then −5 per extra, capped).
      • Up to two highest-scoring cassettes with partial credit:
          - promoter→CDS: order (+5) + proximity (≤100bp:+5, ≤300:+3, ≤500:+2, ≤1000:+1).
          - CDS→terminator: order (+5) + proximity (same as above).
          - Out-of-order legs get proximity-only partial (+≤3).
          - +2 if CDS is a marker and promoter is within 300 bp.
      • Standalone promoters (+1 each up to +5).
      • Standalone terminators (+1 each up to +5).
      • Payload (GOI) CDS anywhere: +8, plus +4 if any promoter within 500 bp.

    Penalty:
      • Length penalty: No penalty up to 5kb, then linear penalty from 0 to -100 over 5kb-60kb.
        Anything above 60kb gets -100 penalty (very bad).

    Returns [0, 100] or tuple (score, breakdown) if return_breakdown=True.
    """
    # ---- collect features (case-insensitive 'type') ----
    def T(x): return (x.type or "").lower()
    feats = list(annotations)
    
    # Deduplicate ORIs by location (more robust than ID alone)
    # Two ORIs are considered duplicates if they overlap significantly
    ori_candidates = [x for x in feats if T(x) in ("rep_origin", "ori", "origin_of_replication")]
    
    def overlaps(ori1, ori2, threshold=0.8):
        """Check if two features overlap by more than threshold fraction"""
        s1, e1 = getattr(ori1, "start", 0), getattr(ori1, "end", 0)
        s2, e2 = getattr(ori2, "start", 0), getattr(ori2, "end", 0)
        if s1 == 0 and e1 == 0 or s2 == 0 and e2 == 0:
            return False  # No location info
        overlap_start = max(s1, s2)
        overlap_end = min(e1, e2)
        overlap_len = max(0, overlap_end - overlap_start)
        len1 = max(1, e1 - s1)
        len2 = max(1, e2 - s2)
        return (overlap_len / len1 >= threshold) or (overlap_len / len2 >= threshold)
    
    oris = []
    for ori in ori_candidates:
        # Check if this ORI overlaps with any already added
        is_duplicate = any(overlaps(ori, existing_ori) for existing_ori in oris)
        if not is_duplicate:
            oris.append(ori)
    
    promoters   = [x for x in feats if T(x) == "promoter"]
    cdss        = [x for x in feats if T(x) == "cds"]
    terminators = [x for x in feats if T(x) == "terminator"]
    
    # Debug logging for all features
    if _REWARD_LOG_BREAKDOWN:
        all_types = {}
        for feat in feats:
            feat_type = T(feat)
            all_types[feat_type] = all_types.get(feat_type, 0) + 1
        logger.info(f"reward.features total={len(feats)} types={all_types}")
        logger.info(f"reward.ori_dedup before={len(ori_candidates)} after={len(oris)}")

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
    parts = {
        "ori": 0.0,
        "cassettes": 0.0,
        "standalone_promoters": 0.0,
        "standalone_terminators": 0.0,
        "payload": 0.0,
        "length_penalty": 0.0,
    }
    
    # Debug logging for ORIs
    if _REWARD_LOG_BREAKDOWN and oris:
        ori_ids = [getattr(ori, "id", "unknown") for ori in oris]
        logger.info(f"reward.ori_debug count={len(oris)} ids={ori_ids}")
    
    if len(oris) == 0:
        pass
    elif len(oris) == 1:
        score += 20.0
        parts["ori"] += 20.0
    else:
        score += 10.0
        parts["ori"] += 10.0
        penalty = min(15.0, 5.0 * (len(oris) - 1))
        score -= penalty
        parts["length_penalty"] -= 0.0  # keep structure consistent (no length penalty here)

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

    cassettes_total = 0.0
    for (_, _, _, pts) in best_cassettes():
        add = float(min(20, pts))
        score += add
        cassettes_total += add
    parts["cassettes"] += cassettes_total

    # ---- Standalone promoter & terminator credit ----
    if promoters:
        add = float(min(5.0, 1.0 * len(promoters)))   # +1 each up to +5
        score += add
        parts["standalone_promoters"] += add
    if terminators:
        add = float(min(5.0, 1.0 * len(terminators))) # +1 each up to +5
        score += add
        parts["standalone_terminators"] += add

    # ---- Payload (GOI) and Marker CDS ----
    target_keywords = [k.lower() for k in (target_keywords or [])]
    def is_payload(c) -> bool:
        r = role(c)
        id_l = (getattr(c, "id", "") or "").lower()
        return r in {"payload", "goi", "reporter"} or (target_keywords and any(k in id_l for k in target_keywords))
    
    def is_marker(c) -> bool:
        r = role(c)
        return r == "marker"

    payloads = [c for c in cdss if is_payload(c)]
    markers = [c for c in cdss if is_marker(c)]
    
    # Debug logging for CDS and payload detection
    if _REWARD_LOG_BREAKDOWN:
        cds_info = []
        for c in cdss:
            cds_id = getattr(c, "id", "unknown")
            cds_role = role(c)
            cds_info.append(f"{cds_id}(role:{cds_role})")
        logger.info(f"reward.cds_debug count={len(cdss)} cds={cds_info[:5]}")  # Log first 5
        if payloads:
            payload_ids = [getattr(p, "id", "unknown") for p in payloads]
            logger.info(f"reward.payload_debug count={len(payloads)} ids={payload_ids}")
        if markers:
            logger.info(f"reward.marker_debug count={len(markers)}")
    
    # Payload CDS: high reward
    if payloads:
        add = 8.0
        score += add
        parts["payload"] += add
        # +4 if any promoter within 500 bp (either direction; same strand preferred implicitly by proximity)
        if any(distance(p, c) <= 500 for c in payloads for p in promoters):
            score += 4.0
            parts["payload"] += 4.0
    
    # Marker CDS: reward for having selection markers
    # Give +2 per marker up to +6 to encourage selection markers
    if markers:
        add = float(min(6.0, 2.0 * len(markers)))
        score += add
        parts["payload"] += add

    # ---- Length penalty: kicks in at 5kb, linear to 60kb ----
    L = max(0, len(sequence or ""))

    def length_penalty(L: int) -> float:
        # No penalty up to 5kb
        # Linear penalty from 5kb to 60kb: 0 → -100
        # Above 60kb: just return -100 (very bad)
        if L <= 5000:
            return 0.0
        if L <= 60000:
            # Linear from 0 to -100 over 55kb range
            return 10.0 * (L - 5000) / 55000.0
        return 10.0  # Cap at -100 for anything above 60kb

    lp = length_penalty(L)
    score -= lp
    parts["length_penalty"] -= float(lp)

    # ---- Clamp ----
    total = float(max(0.0, min(100.0, score)))

    if _REWARD_LOG_BREAKDOWN:
        try:
            logger.info(
                "reward.parts L=%d ori=%.2f cassettes=%.2f promoters=%.2f terminators=%.2f payload=%.2f length_penalty=%.2f total=%.2f",
                L,
                parts["ori"],
                parts["cassettes"],
                parts["standalone_promoters"],
                parts["standalone_terminators"],
                parts["payload"],
                parts["length_penalty"],
                total,
            )
        except Exception:
            pass

    if return_breakdown:
        parts["L"] = float(L)
        return total, parts
    return total



def score_completions(completions: list[str], return_breakdown: bool = False) -> list[float] | tuple[list[float], list[dict[str, float]]]:
    if _REWARD_LOG_TIMINGS:
        t0 = time.perf_counter()
    if not completions:
        logger.warning("reward.score_completions called with empty completions list")
        return [] if not return_breakdown else ([], [])
    annotations = annotate_completions(completions)
    
    if return_breakdown:
        results = [score_sequence(c, a, return_breakdown=True) for c, a in zip(completions, annotations)]
        scores = [r[0] for r in results]
        breakdowns = [r[1] for r in results]
    else:
        scores = [score_sequence(c, a) for c, a in zip(completions, annotations)]
        breakdowns = []
    
    if _REWARD_LOG_TIMINGS:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        n = len(scores)
        mean_score = sum(scores) / n if n else 0.0
        min_score = min(scores) if n else 0.0
        max_score = max(scores) if n else 0.0
        logger.info(
            f"reward.score n={n} mean={mean_score:.2f} min={min_score:.2f} max={max_score:.2f} time_ms={dt_ms:.2f}"
        )
    
    if return_breakdown:
        return scores, breakdowns
    return scores
