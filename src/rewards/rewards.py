from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any
import plasmidkit as pk
import logging
from typing import List, Tuple, Any

logger = logging.getLogger("reward_logger")


def annotate_completions(completions: list[str]) -> list[Any]:
    """Annotate a flat list of completions using threads; strips spaces from sequences."""
    sequences = [s.replace(" ", "") for s in completions]
    with ThreadPoolExecutor() as executor:
        annotate = partial(pk.annotate, is_sequence=True)
        return list(executor.map(annotate, sequences))

from typing import Any, List, Tuple

def score_sequence(sequence: str, annotations: Any) -> float:
    """
    Compute a heuristic backbone design score for use as a reward signal
    in GRPO or other optimization loops.

    The score rewards the presence of essential plasmid features and their
    arrangement, while applying a mild length penalty to discourage overly
    large backbones without allowing length reduction to inflate the score.

    Scoring components:
    - **Origin of replication (ORI)**:
      +20 points for exactly one origin (preferred).
      +10 for multiple origins with a small penalty (−5 per extra, capped).
      0 if no origin is found.

    - **Promoter→CDS→Terminator cassettes**:
      Up to two of the best-scoring cassettes are considered.
      Each cassette earns partial credit for:
        • Correct promoter→CDS order (+5) and proximity (up to +5),
        • Correct CDS→terminator order (+5) and proximity (up to +5),
        • Small bonus (+2) if the CDS has evidence.role == "marker"
          and its promoter is nearby (≤300 bp).
      A tight, well-formed cassette scores ~20 points; looser arrangements
      get proportionally fewer points.

    - **Marker genes**:
      CDSs annotated with evidence.role == "marker" receive a small
      standalone bonus (up to +10 total) even if they are not in a full
      cassette, encouraging inclusion of selectable markers.

    - **Length penalty**:
      A linear penalty from 0 at 1 kb to −15 at 10 kb discourages
      unnecessary length but cannot exceed the value of important
      features. This ensures that removing useful features never
      increases the score (monotonicity).

    The final score is clipped to [0, 100] for stability. It is dense
    (most designs get partial credit) and shaped to provide smooth,
    monotonic gradients for GRPO-based optimization.

    Args:
        sequence: The raw nucleotide sequence (used for length penalty).
        annotations: Iterable of feature objects with at least:
            - type: "rep_origin", "promoter", "CDS", or "terminator"
            - start, end, strand
            - evidence.role (optional): "marker" for AMR/selectable markers

    Returns:
        float: A heuristic design score in [0, 100].
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
        # 0 if overlapping/adjacent; otherwise gap between them
        if end(a) < start(b): return start(b) - end(a)
        if end(b) < start(a): return start(a) - end(b)
        return 0

    # score proximity: closer → more points
    def prox_points(d: int, max_pts: int = 5) -> int:
        if d <= 100:  return max_pts
        if d <= 300:  return max_pts - 2   # 3
        if d <= 500:  return max_pts - 3   # 2
        if d <= 1000: return 1
        return 0

    # ---- ORI scoring (max 20; penalize extras) ----
    score = 0.0
    ori_points = 0.0
    if len(oris) == 0:
        ori_points = 0.0
    elif len(oris) == 1:
        ori_points = 20.0  # single ori preferred
    else:
        penalty = min(15.0, 5.0 * (len(oris) - 1))  # light penalty per extra
        ori_points = 10.0 - penalty  # has replication, but multiple origins can conflict
    print(f"[score] ORI count={len(oris)} -> ori_points={ori_points:.2f}")
    score += ori_points

    # ---- Cassette scoring (max 2 best cassettes × 20 = 40) ----
    # Build candidate triplets by greedy nearest neighbors on same strand, in order.
    # Partial credit: order (+5 each leg) + proximity (up to +5 each leg).
    # Total per full, tight cassette ≈ 20.
    def best_cassettes() -> List[Tuple[Any, Any, Any, int]]:
        triples = []
        for p in promoters:
            # find nearest CDS downstream on same strand
            cds_cands = [c for c in cdss if strand(c) == strand(p) and in_order_same_strand(p, c)]
            cds_cands.sort(key=lambda c: distance(p, c))
            if not cds_cands: continue
            c = cds_cands[0]

            # find nearest terminator downstream of CDS on same strand
            term_cands = [t for t in terminators if strand(t) == strand(c) and in_order_same_strand(c, t)]
            term_cands.sort(key=lambda t: distance(c, t))
            if not term_cands: continue
            t = term_cands[0]

            # score legs
            pts = 0
            # promoter -> CDS
            if in_order_same_strand(p, c):
                pts += 5  # order
                pts += prox_points(distance(p, c), 5)
            # CDS -> terminator
            if in_order_same_strand(c, t):
                pts += 5  # order
                pts += prox_points(distance(c, t), 5)

            # small bonus if CDS is a marker and promoter likely drives it (upstream & close)
            if role(c) == "marker" and distance(p, c) <= 300:
                pts += 2

            triples.append((p, c, t, pts))
        # keep the top-scoring cassettes (non-overlap not enforced for simplicity)
        triples.sort(key=lambda x: x[3], reverse=True)
        return triples[:2]

    for (_, _, _, pts) in best_cassettes():
        contrib = float(min(20, pts))  # cap per-cassette contribution at 20
        print(f"[score] Cassette points={pts} -> contrib={contrib:.2f}")
        score += contrib

    # ---- Standalone marker presence (small, separate from cassette) ----
    # Encourages having a selectable marker even if not paired perfectly.
    markers = [c for c in cdss if role(c) == "marker"]
    marker_bonus = 0.0
    if markers:
        marker_bonus = float(min(10, 5 + 2 * min(2, len(markers))))  # +5 base, +2 per (up to 2)
        score += marker_bonus
    print(f"[score] Markers count={len(markers)} -> marker_bonus={marker_bonus:.2f}")

    # ---- Length penalty: 1 kb → 10 kb maps to 0 → -15; <1 kb: 0; >10 kb: -15
    L = max(0, len(sequence or ""))
    if L <= 1000:
        length_penalty = 0.0
    elif L >= 10000:
        length_penalty = 15.0
    else:
        # linear between 1kb and 10kb
        length_penalty = 15.0 * (L - 1000) / (9000)
    print(f"[score] Length={L} bp -> length_penalty={length_penalty:.2f}")
    score -= length_penalty

    # ---- Clamp to 0..100
    final_score = float(max(0.0, min(100.0, score)))
    print(f"[score] Total unclamped={score:.2f} -> final={final_score:.2f}")
    return final_score


def score_completions(completions: list[str]) -> list[float]:
    annotations = annotate_completions(completions)
    return [score_sequence(c, a) for c, a in zip(completions, annotations)]


seq = "GGTCTGCTATGTGGTGCTATCTGACTTTTTGCTGTTCAGCAGTTCCTGCCCTCTGATTTTCCAGTCTGACCACTTCGGATTATCCCGTGACAGGTCATTCAGACTGGCTAATGCACCCAGTAAGGCAGCGGTATCATCAACGGGGTCTGACGCTCAGTGGAACGAAAACTCACGTTAAGGGATTTTGGTCATGAGATTATCAAAAAGGATCTTCACCTAGATCCGTTATGCAGCGGAAAGTAAAAAATTTTTAGTTTATTAGACATCTCCACAAAAGGCGTAGTGTACAGTGACAAATTATCTGTCGTCGGTGACAGATTAATGTCATTGTGACTATTTAATTGTCGTCGTGACCCATCAGCGTTGCTTAATTAATTGATGACAAATTAAATGTCATCAATATAATATGCTCTGCAATTATTATACAAAGCAATTAAAACAAGCGGATAAAAGGACTTGCTTTCAACCCACCCCTAAGTTTAATAGTTACTGAGGGGGATCCACTAGTGAGCTCATGCATGATCTCGAATTAGCTTCAAAAGCGCTCTGAAGTTCCTATACTTTCTAGAGAATAGGAACTTCGGAATAGGAACTTCAAGATCCCCTGATTCCCTTTGTCAACAGCAATGGATAATTCGATTTAACAAATGCATGGCGCAAGGGCTGCTAAAGGAAGCGGAACACGTAGAAAGCCAGTCCGCAGAAACGGTGCTGACCCCGGATGAATGTCAGCTACTGGGCTATCTGGACAAGGGAAAACGCAAGCGCAAAGAGAAAGCAGGTAGCTTGCAGTGGGCTTACATGGCGATAGCTAGACTGGGCGGTTTTATGGACAGCAAGCGAACCGGAATTGCCAGCTGGGGCGCCCTCTGGTAAGGTTGGGAAGCCCTGCAAAGTAAACTGGATGGCTTTCTTGCCGCCAAGGATCTGATGGCGCAGGGGATCAAGATCTGATCAAGAGACAGGATGAGGATCGTTTCGCATGATTGAACAAGATGGATTGCACGCAGGTTCTCCGGCCGCTTGGGTGGAGAGGCTATTCGGCTATGACTGGGCACAACAGACAATCGGCTGCTCTGATGCCGCCGTGTTCCGGCTGTCAGCGCAGGGGCGCCCGGTTCTTTTTGTCAAGACCGACCTGTCCGGTGCCCTGAATGAACTGCAGGACGAGGCAGCGCGGCTATCGTGGCTGGCCACGACGGGCGTTCCTTGCGCAGCTGTGCTCGACGTTGTCACTGAAGCGGGAAGGGACTGGCTGCTATTGGGCGAAGTGCCGGGGCAGGATCTCCTGTCATCCCACCTTGCTCCTGCCGAGAAAGTATCCATCATGGCTGATGCAATGCGGCGGCTGCATACGCTTGATCCGGCTACCTGCCCATTCGACCACCAAGCGAAACATCGCATCGAGCGAGCACGTACTCGGATGGAAGCCGGTCTTGTCGATCAGGATGATCTGGACGAAGAGCATCAGGGGCTCGCGCCAGCCGAACTGTTCGCCAGGCTCAAGGCGCGCATGCCCGACGGCGAGGATCTCGTCGTGACCCATGGCGATGCCTGCTTGCCGAATATCATGGTGGAAAATGGCCGCTTTTCTGGATTCATCGACTGTGGCCGGCTGGGTGTGGCGGACCGCTATCAGGACATAGCGTTGGCTACCCGTGATATTGCTGAAGAGCTTGGCGGCGAATGGGCTGACCGCTTCCTCGTGCTTTACGGTATCGCCGCTCCCGATTCGCAGCGCATCGCCTTCTATCGCCTTCTTGACGAGTTCTTCTGAATTGAAAAAGGAAGAGTATGAGGATCCAACATTTCCAATCACTAGTGAATTATCTAGAATTATTCCATTGAGTAAGTTTTTAAGCACATCAGCTTCAAAAGCGCTCTGAAGTTCCTATACTTTCTAGAGAATAGGAACTTCGGAATAGGTACTTCAAGATCCCCAATTCGAGATCGTCCGGGCCGCAAGCTCCTAGCGGCGGATTTGTCCTACTCAGGAGAGCGTTCACCGACAAACAACAGATAAAACGAAAGGCCCAGTCTTTCGACTGAGCCTTTCGTTTTATTTGATGCCTCAAGCTAGAGAGTCATTACCCCAGGCGTTTAAGGGCACCAATAACTGCCTTAAAAAAATTACGCCCCGCCCTGCCACTCATCGCAGTCTAGCTTGGATTCTCACCAATAAAAAACGCCCGGCGGCAACCGAGCGTTCTGAACAAATCCAGATGGAGTTCTGAGGTCATTACTGGATCTATCAACAGGAGTCCAAGCTCAGCTAATTAAGGCGACAGTCAATTTGTCATTATGAAAATACACAAAAGCTTTTTCCTATCTTGCAAAGCGACAGCTAATTTGTCACAATCACGGACAACGACATCTATTTTGTCACTGCAAAGAGGTTATGCTAAAACTGCCAAAGCGCTATAATCTATACTGTATAAGGATTTTACTGATGACAATAATTTGTCACAACGACATATAATTAGTCACTGTACACGTAGAGACGTAGCAATGCTACCTCTCTACAATGGTTTTGTGTTAGTCTTGATGCTTCACTGATAGATACAAGAGCCATAAGAACCTCAGATCCTTCCGTATTTAGCCAGTATGTTCTCTAGTGTGGTTCGTTGTTTTTGCGTGAGCCATGAGAACGAACCATTGAGATCATACTTACTTTGCATGTCACTCAAAAATTTTGCCTCAAAACTGGTGAGCTGAATTTTTGCAGTTAAAGCATCGTGTAGTGTTTTTCTTAGTCCGTTATGTAGGTAGGAATCTGATGTAATGGTTGTTGGTATTTTGTCACCATTCATTTTTATCTGGTTGTTCTCAAGTTCGGTTACGAGATCCATTTGTCTATCTAGTTCAACTTGGAAAATCAACGTATCAGTCGGGCGGCCTCGCTTATCAACCACCAATTTCATATTGCTGTAAGTGTTTAAATCTTTACTTATTGGTTTCAAAACCCATTGGTTAAGCCTTTTAAACTCATGGTAGTTATTTTCAAGCATTAACATGAACTTAAATTCATCAAGGCTAATCTCTATATTTGCCTTGTGAGTTTTCTTTTGTGTTAGTTCTTTTAATAACCACTCATAAATCCTCATAGAGTATTTGTTTTCAAAAGACTTAACATGTTCCAGATTATATTTTATGAATTTTTTTAACTGGAAAAGATAAGGCAATATCTCTTCACTAAAAACTAATTCTAATTTTTCGCTTGAGAACTTGGCATAGTTTGTCCACTGGAAAATCTCAAAGCCTTTAACCAAAGGATTCCTGATTTCCACAGTTCTCGTCATCAGCTCTCTGGTTGCTTTAGCTAATACACCATAAGCATTTTCCCTACTGATGTTCATCATCTGAACGTATTGGTTATAAGTGAACGATACCGTCCGTTCTTTCCTTGTAGGGTTTTCAATCGTGGGGTTGAGTAGTGCCACACAGCATAAAATTAGCTTGGTTTCATGCTCCGTTAAGTCATAGCGACTAATCGCTAGTTCATTTGCTTTGAAAACAACTAATTCAGACATACATCTCAATTGGTCTAGGTGATTTTAATCACTATACCAATTGAGATGGGCTAGTCAATGATAATTACTAGTCCTTTTCCTTTGAGTTGTGGGTATCTGTAAATTCTGCTAGACCTTTGCTGGAAAACTTGTAAATTCTGCTAGACCCTCTGTAAATTCCGCTAGACCTTTGTGTGTTTTTTTTGTTTATATTCAAGTGGTTATAATTTATAGAATAAAGAAAGAATAAAAAAAGATAAAAAGAATAGATCCCAGCCCTGTGTATAACTCACTACTTTAGTCAGTTCCGCAGTATTACAAAAGGATGTCGCAAACGCTGTTTGCTCCTCTACAAAACAGACCTTAAAACCCTAAAGGCTTAAGTAGCACCCTCGCAAGCTCGGTTGCGGCCGCAATCGGGCAAATCGCTGAATATTCCTTTTGTCTCCGACCATCAGGCACCTGAGTCGCTGTCTTTTTCGTGACATTCAGTTCGCTGCGCTCACGGCTCTGGCAGTGAATGGGGGTAAATGGCACTACAGGCGCCTTTTATGGATTCATGCAAGGAAACTACCCATAATACAAGAAAAGCCCGTCACGGGCTTCTCAGGGCGTTTTATGGCG"
annotation = pk.annotate(seq)

for anno in annotation:
    if anno.type != "restriction_site":
        print(f"{anno.type} {anno.id} {anno.evidence}")

score = score_sequence(seq, annotation)
print(score)