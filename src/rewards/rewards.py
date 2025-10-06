from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any
import plasmidkit as pk
import logging
from typing import List, Tuple, Any, Iterable

logger = logging.getLogger("reward_logger")


def annotate_completions(completions: list[str]) -> list[Any]:
    """Annotate a flat list of completions using threads; strips spaces from sequences."""
    sequences = [s.replace(" ", "") for s in completions]
    with ThreadPoolExecutor() as executor:
        annotate = partial(pk.annotate, is_sequence=True)
        return list(executor.map(annotate, sequences))

from typing import Any, List, Tuple

from typing import Any, List, Tuple, Iterable

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
    if len(oris) == 0:
        pass
    elif len(oris) == 1:
        score += 20.0
    else:
        score += 10.0
        score -= min(15.0, 5.0 * (len(oris) - 1))

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

    for (_, _, _, pts) in best_cassettes():
        score += float(min(20, pts))

    # ---- Standalone promoter & terminator credit ----
    if promoters:
        score += float(min(5.0, 1.0 * len(promoters)))   # +1 each up to +5
    if terminators:
        score += float(min(5.0, 1.0 * len(terminators))) # +1 each up to +5

    # ---- Payload (GOI) anywhere ----
    target_keywords = [k.lower() for k in (target_keywords or [])]
    def is_payload(c) -> bool:
        r = role(c)
        id_l = (getattr(c, "id", "") or "").lower()
        return r in {"payload", "goi", "reporter"} or (target_keywords and any(k in id_l for k in target_keywords))

    payloads = [c for c in cdss if is_payload(c)]
    if payloads:
        score += 8.0
        # +4 if any promoter within 500 bp (either direction; same strand preferred implicitly by proximity)
        if any(distance(p, c) <= 500 for c in payloads for p in promoters):
            score += 4.0

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

    score -= length_penalty(L)

    # ---- Clamp ----
    return float(max(0.0, min(100.0, score)))



def score_completions(completions: list[str]) -> list[float]:
    annotations = annotate_completions(completions)
    return [score_sequence(c, a) for c, a in zip(completions, annotations)]
