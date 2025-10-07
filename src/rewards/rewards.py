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
) -> float:
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
      • Piecewise length penalty (1–10 kb, max −15) with **gentler slope at 3–5 kb**:
          1–3 kb: up to −3; 3–5 kb: to −5; 5–7.5 kb: to −9; 7.5–10 kb: to −15.
        Ensures removing useful features can’t increase the score.

    Returns [0, 100].
    """
    # ---- collect features (case-insensitive 'type') ----
    def T(x): return (x.type or "").lower()
    feats = list(annotations)
    oris        = [x for x in feats if T(x) in ("rep_origin", "ori", "origin_of_replication")]
    promoters   = [x for x in feats if T(x) == "promoter"]
    cdss        = [x for x in feats if T(x) == "cds"]
    terminators = [x for x in feats if T(x) == "terminator"]

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

    # ---- Payload (GOI) anywhere ----
    target_keywords = [k.lower() for k in (target_keywords or [])]
    def is_payload(c) -> bool:
        r = role(c)
        id_l = (getattr(c, "id", "") or "").lower()
        return r in {"payload", "goi", "reporter"} or (target_keywords and any(k in id_l for k in target_keywords))

    payloads = [c for c in cdss if is_payload(c)]
    if payloads:
        add = 8.0
        score += add
        parts["payload"] += add
        # +4 if any promoter within 500 bp (either direction; same strand preferred implicitly by proximity)
        if any(distance(p, c) <= 500 for c in payloads for p in promoters):
            score += 4.0
            parts["payload"] += 4.0

    # ---- Length penalty: gentler at 3–5 kb ----
    L = max(0, len(sequence or ""))

    def length_penalty(L: int) -> float:
        # piecewise linear:
        # 1–3 kb:   0 → -3
        # 3–5 kb:  -3 → -5
        # 5–7.5 kb:-5 → -9
        # 7.5–10 kb:-9 → -15
        if L <= 1000:
            return 0.0
        if L <= 3000:
            # span 2kb to -3
            return 3.0 * (L - 1000) / 2000.0
        if L <= 5000:
            # next 2kb adds -2 (to -5)
            return 3.0 + 2.0 * (L - 3000) / 2000.0
        if L <= 7500:
            # next 2.5kb adds -4 (to -9)
            return 5.0 + 4.0 * (L - 5000) / 2500.0
        if L <= 10000:
            # final 2.5kb adds -6 (to -15)
            return 9.0 + 6.0 * (L - 7500) / 2500.0
        return 15.0

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

            if wandb is not None:
                wandb.log(
                    {
                        **({"iter": _CURRENT_ITER} if _CURRENT_ITER is not None else {}),
                        "reward/parts/ori": parts["ori"],
                        "reward/parts/cassettes": parts["cassettes"],
                        "reward/parts/promoters": parts["standalone_promoters"],
                        "reward/parts/terminators": parts["standalone_terminators"],
                        "reward/parts/payload": parts["payload"],
                        "reward/parts/length_penalty": parts["length_penalty"],
                        "reward/total": total,
                        "reward/L": float(L),
                    },
                    commit=True,
                )
        except Exception:
            pass

    return total



def score_completions(completions: list[str]) -> list[float]:
    if _REWARD_LOG_TIMINGS:
        t0 = time.perf_counter()
    if not completions:
        logger.warning("reward.score_completions called with empty completions list")
        return []
    annotations = annotate_completions(completions)
    scores = [score_sequence(c, a) for c, a in zip(completions, annotations)]
    if _REWARD_LOG_TIMINGS:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        n = len(scores)
        mean_score = sum(scores) / n if n else 0.0
        min_score = min(scores) if n else 0.0
        max_score = max(scores) if n else 0.0
        logger.info(
            f"reward.score n={n} mean={mean_score:.2f} min={min_score:.2f} max={max_score:.2f} time_ms={dt_ms:.2f}"
        )
    return scores
