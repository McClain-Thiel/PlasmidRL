import pandas as pd
import numpy as np
import random
from typing import Tuple, Dict, Any, Callable

# -------------------------
# Utility functions
# -------------------------

def _to_0based(start_1b: int, end_1b: int) -> Tuple[int, int]:
    """Convert 1-based inclusive [start,end] to 0-based half-open [start,end)."""
    if start_1b <= 0 or end_1b <= 0:
        raise ValueError("Indices must be 1-based positive.")
    if end_1b < start_1b:
        # handle reversed spans by swapping (your table seems forward-only, but be safe)
        start_1b, end_1b = end_1b, start_1b
    return start_1b - 1, end_1b  # end is already inclusive; half-open wants end

def _shift_annotations(ann: pd.DataFrame, cut_start0: int, delta_len: int) -> pd.DataFrame:
    """
    Shift qstart/qend after an insertion/deletion starting at cut_start0.
    delta_len < 0 for deletions (sequence shrinks), > 0 for insertions (sequence grows).
    Flags overlapping features as 'broken'.
    """
    ann = ann.copy()
    ann["broken"] = ann.get("broken", False)

    # Convert to 0-based to reason, then convert back to 1-based at the end.
    q0 = ann["qstart"].astype(int) - 1
    q1 = ann["qend"].astype(int)      # half-open

    # Features entirely after the edit start get shifted
    after_mask = q0 >= cut_start0
    ann.loc[after_mask, "qstart"] = (q0[after_mask] + delta_len) + 1
    ann.loc[after_mask, "qend"]   = (q1[after_mask] + delta_len)

    # Features that overlap the edited region are marked broken.
    # For deletions: region [cut_start0, cut_start0 - delta_len)
    # For insertions: treat the insertion point as a zero-length overlap
    if delta_len < 0:
        del_len = -delta_len
        edit_lo, edit_hi = cut_start0, cut_start0 + del_len
        overlap = (q0 < edit_hi) & (q1 > edit_lo)
        ann.loc[overlap, "broken"] = True
        # Optionally, clip overlapping features to boundaries (simple choice):
        ann.loc[overlap, "qstart"] = np.maximum(ann.loc[overlap, "qstart"], edit_lo + 1)
        ann.loc[overlap, "qend"]   = np.maximum(ann.loc[overlap, "qend"],   ann.loc[overlap, "qstart"])
    else:
        # Insertion: features that start at/after insertion were shifted already.
        # A feature spanning the insertion point is still valid; no need to break it.
        pass

    return ann

def _delete_interval(seq: str, start0: int, end0: int) -> Tuple[str, int]:
    """Delete seq[start0:end0] and return (new_seq, delta_len). delta_len is negative."""
    new_seq = seq[:start0] + seq[end0:]
    return new_seq, -(end0 - start0)

def _insert_interval(seq: str, pos0: int, ins: str) -> Tuple[str, int]:
    """Insert 'ins' at position pos0 (0-based). Return (new_seq, delta_len)."""
    new_seq = seq[:pos0] + ins + seq[pos0:]
    return new_seq, len(ins)

def _reverse_complement(dna: str) -> str:
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return dna.translate(comp)[::-1]

def _pick_feature_row(ann: pd.DataFrame,
                      predicate: Callable[[pd.Series], bool]) -> pd.Series | None:
    """Return the first row matching the predicate, or None."""
    for _, row in ann.iterrows():
        if predicate(row):
            return row
    return None

# -------------------------
# Transformations (return new_seq, new_ann, log)
# -------------------------

def remove_feature_by_kind(seq: str, ann: pd.DataFrame, kind: str) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    """
    Delete the first feature with ann['kind'] == kind (e.g., 'ori', 'CDS', 'promoter').
    Good for creating 'missing essential part' negatives (e.g., remove ori or resistance CDS).
    """
    row = _pick_feature_row(ann, lambda r: str(r.get("kind","")).lower() == kind.lower())
    if row is None:
        return seq, ann, {"op":"remove_feature_by_kind","kind":kind,"status":"no_feature_found"}

    s0, e1 = _to_0based(int(row["qstart"]), int(row["qend"]))
    new_seq, delta = _delete_interval(seq, s0, e1)
    new_ann = _shift_annotations(ann, s0, delta)
    log = {"op":"remove_feature_by_kind","kind":kind,"deleted_span":[s0,e1],"delta":delta}
    return new_seq, new_ann, log

def corrupt_ori_with_non_ecoli(seq: str, ann: pd.DataFrame) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    """
    Replace the ori region with a same-length 'non-E.coli-like' placeholder (random GC-heavy block).
    This mimics swapping in an incompatible origin.
    """
    row = _pick_feature_row(ann, lambda r: ("ori" in str(r.get("sseqid","")).lower()) or (str(r.get("kind","")).lower() == "ori"))
    if row is None:
        return seq, ann, {"op":"corrupt_ori","status":"no_ori_found"}

    s0, e1 = _to_0based(int(row["qstart"]), int(row["qend"]))
    span_len = e1 - s0
    # Make a GC-heavy block that looks unlike common E. coli oris
    rnd = random.Random(42 + s0 + span_len)  # deterministic per call
    gc_pool = "GCGCGCGCGCGCGCGCTTCCGGCCGCGGCGGCGCGGCGCGCCG"
    filler = "".join(rnd.choice(gc_pool) for _ in range(span_len))

    new_seq = seq[:s0] + filler + seq[e1:]
    new_ann = ann.copy()
    # Mark ori as broken + relabel
    idx = ann.index[ann["qstart"].eq(int(row["qstart"])) & ann["qend"].eq(int(row["qend"]))][0]
    new_ann.loc[idx, "broken"] = True
    new_ann.loc[idx, "note"] = "ori_replaced_with_incompatible_sequence"
    log = {"op":"corrupt_ori","replaced_span":[s0,e1],"len":span_len}
    return new_seq, new_ann, log

def frameshift_marker_cds(seq: str, ann: pd.DataFrame, marker_names=("AmpR","KanR","HygR","CmR")) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    """
    Introduce a 1-bp insertion near the start codon of a resistance CDS (AmpR, KanR, etc.) to inactivate selection.
    """
    row = _pick_feature_row(
        ann,
        lambda r: (str(r.get("kind","")).lower() == "cds") and any(name.lower() in str(r.get("sseqid","")).lower() for name in marker_names)
    )
    if row is None:
        return seq, ann, {"op":"frameshift_marker_cds","status":"no_marker_found"}

    s0, e1 = _to_0based(int(row["qstart"]), int(row["qend"]))
    insert_pos0 = s0 + 3  # within first codon neighborhood
    ins_base = "A"  # single-base insertion to cause frameshift

    new_seq, delta = _insert_interval(seq, insert_pos0, ins_base)
    new_ann = _shift_annotations(ann, insert_pos0, delta)
    # mark this feature as broken
    idx = ann.index[ann["qstart"].eq(int(row["qstart"])) & ann["qend"].eq(int(row["qend"]))][0]
    new_ann.loc[idx, "broken"] = True
    new_ann.loc[idx, "note"] = "frameshift_in_marker_CDS"
    log = {"op":"frameshift_marker_cds","pos0":insert_pos0,"delta":delta,"marker":row.get("sseqid")}
    return new_seq, new_ann, log

def insert_tandem_repeat(seq: str, ann: pd.DataFrame, unit_len: int = 50, copies: int = 6, at: str = "random") -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    """
    Insert a tandem repeat block (instability bait).
    """
    rnd = random.Random(1337 + len(seq) + unit_len + copies)
    unit = "".join(rnd.choice("ACGT") for _ in range(unit_len))
    block = unit * copies

    if at == "random":
        pos0 = rnd.randrange(0, len(seq)+1)
    else:
        pos0 = max(0, min(int(at), len(seq)))

    new_seq, delta = _insert_interval(seq, pos0, block)
    new_ann = _shift_annotations(ann, pos0, delta)

    # record a synthetic feature row (optional)
    syn_row = {
        "qstart": pos0 + 1,
        "qend": pos0 + delta,
        "sseqid": f"TANDEM_REPEAT_{unit_len}x{copies}",
        "kind": "repeat",
        "broken": False,
        "note": "synthetic_repeat_insert"
    }
    new_ann = pd.concat([new_ann, pd.DataFrame([syn_row])], ignore_index=True)

    log = {"op":"insert_tandem_repeat","pos0":pos0,"unit_len":unit_len,"copies":copies,"delta":delta}
    return new_seq, new_ann, log

def insert_hairpin_palindrome(seq: str, ann: pd.DataFrame, arm_len: int = 40, loop_len: int = 6, at: str = "random") -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    """
    Insert a strong inverted repeat (palindrome) -> hairpin-forming element.
    """
    rnd = random.Random(9001 + len(seq) + arm_len + loop_len)
    left_arm = "".join(rnd.choice("ACGT") for _ in range(arm_len))
    right_arm = _reverse_complement(left_arm)
    loop = "".join(rnd.choice("ACGT") for _ in range(loop_len))
    insert = left_arm + loop + right_arm

    pos0 = rnd.randrange(0, len(seq)+1) if at == "random" else max(0, min(int(at), len(seq)))
    new_seq, delta = _insert_interval(seq, pos0, insert)
    new_ann = _shift_annotations(ann, pos0, delta)

    syn_row = {
        "qstart": pos0 + 1,
        "qend": pos0 + delta,
        "sseqid": f"INVERTED_REPEAT_{arm_len}+{loop_len}+{arm_len}",
        "kind": "palindrome",
        "broken": False,
        "note": "synthetic_hairpin_insert"
    }
    new_ann = pd.concat([new_ann, pd.DataFrame([syn_row])], ignore_index=True)

    log = {"op":"insert_hairpin_palindrome","pos0":pos0,"arm_len":arm_len,"loop_len":loop_len,"delta":delta}
    return new_seq, new_ann, log

def scramble_promoter(seq: str, ann: pd.DataFrame, promoter_hint=("promoter","T7","lac","ara","CMV")) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    """
    Destroy the first promoter-like feature by shuffling bases (keeps length, ruins motif).
    """
    row = _pick_feature_row(
        ann,
        lambda r: (str(r.get("kind","")).lower() == "promoter") or any(h.lower() in str(r.get("sseqid","")).lower() for h in promoter_hint)
    )
    if row is None:
        return seq, ann, {"op":"scramble_promoter","status":"no_promoter_found"}

    s0, e1 = _to_0based(int(row["qstart"]), int(row["qend"]))
    frag = list(seq[s0:e1])
    rnd = random.Random(777 + s0 + e1)
    rnd.shuffle(frag)
    scrambled = "".join(frag)
    new_seq = seq[:s0] + scrambled + seq[e1:]
    new_ann = ann.copy()
    idx = ann.index[ann["qstart"].eq(int(row["qstart"])) & ann["qend"].eq(int(row["qend"]))][0]
    new_ann.loc[idx, "broken"] = True
    new_ann.loc[idx, "note"] = "promoter_scrambled"
    log = {"op":"scramble_promoter","span":[s0,e1],"len":e1-s0}
    return new_seq, new_ann, log

def oversize_plasmid(seq: str, ann: pd.DataFrame, target_kb: int = 25) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    """
    Inflate plasmid size to ~target_kb by inserting neutral random DNA (size burden).
    """
    if len(seq) >= target_kb * 1000:
        return seq, ann, {"op":"oversize_plasmid","status":"already_large","size":len(seq)}

    rnd = random.Random(24601 + len(seq))
    need = target_kb * 1000 - len(seq)
    # Insert in two chunks to avoid one giant contiguous block
    chunk1 = need // 2
    chunk2 = need - chunk1
    def rand_block(n): return "".join(rnd.choice("ACGT") for _ in range(n))

    pos1 = rnd.randrange(0, len(seq)+1)
    seq1, d1 = _insert_interval(seq, pos1, rand_block(chunk1))
    ann1 = _shift_annotations(ann, pos1, d1)

    pos2 = rnd.randrange(0, len(seq1)+1)
    seq2, d2 = _insert_interval(seq1, pos2, rand_block(chunk2))
    ann2 = _shift_annotations(ann1, pos2, d2)

    syn_rows = [
        {"qstart": pos1 + 1, "qend": pos1 + d1, "sseqid": f"NEUTRAL_INSERT_{chunk1}", "kind":"cargo", "broken": False, "note": "size_inflation"},
        {"qstart": pos2 + 1, "qend": pos2 + d2, "sseqid": f"NEUTRAL_INSERT_{chunk2}", "kind":"cargo", "broken": False, "note": "size_inflation"}
    ]
    ann2 = pd.concat([ann2, pd.DataFrame(syn_rows)], ignore_index=True)
    log = {"op":"oversize_plasmid","target_bp":target_kb*1000,"delta":(d1+d2)}
    return seq2, ann2, log
