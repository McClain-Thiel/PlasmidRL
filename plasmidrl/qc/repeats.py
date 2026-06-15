#!/usr/bin/env python3
"""
Repeated Region Detection

Original logic developed by Angus Cunningham (University College London).
Repository: https://github.com/angusgcunningham/plasmidbackbonedesign/tree/main

This script identifies the longest exact repeated regions within DNA sequences.
It employs suffix array construction (prefix-doubling) and Longest Common Prefix (LCP) 
arrays (Kasai's algorithm) to efficiently locate direct and inverted repeats in 
circular genomes.
"""

import argparse
import csv
import gzip
import os
from pathlib import Path
from typing import Iterator, Tuple, List, Dict

# ── FASTA reader (minimal) ───────────────────────────────────────────────
def read_fasta_simple(path: str):
    hdr, buf = None, []
    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if hdr is not None:
                    yield hdr, "".join(buf)
                hdr = line[1:].strip()
                buf = []
            else:
                buf.append(line)
    if hdr is not None:
        yield hdr, "".join(buf)

# ── Suffix array (prefix-doubling) & LCP (Kasai) ─────────────────────────
def suffix_array(s: str) -> List[int]:
    n = len(s)
    sa = list(range(n))
    # initial ranks by character
    rank = [ord(c) for c in s]
    tmp  = [0] * n
    k = 1
    while True:
        sa.sort(key=lambda i: (rank[i], rank[i + k] if i + k < n else -1))
        tmp[sa[0]] = 0
        for i in range(1, n):
            a, b = sa[i-1], sa[i]
            prev = (rank[a], rank[a + k] if a + k < n else -1)
            curr = (rank[b], rank[b + k] if b + k < n else -1)
            tmp[b] = tmp[a] + (prev != curr)
        rank, tmp = tmp, rank
        if rank[sa[-1]] == n - 1:
            break
        k <<= 1
    return sa

def lcp_array(s: str, sa: List[int]) -> List[int]:
    n = len(s)
    rank = [0] * n
    for i, si in enumerate(sa):
        rank[si] = i
    lcp = [0] * n
    h = 0
    for i in range(n):
        r = rank[i]
        if r == 0:
            h = 0
            continue
        j = sa[r - 1]
        while i + h < n and j + h < n and s[i + h] == s[j + h]:
            h += 1
        lcp[r] = h
        if h:
            h -= 1
    return lcp

# ── Helpers to extract repeat blocks from SA/LCP ─────────────────────────
def _collect_block(sa, lcp, idx, L):
    """
    For a given index 'idx' where lcp[idx] == L, collect the maximal
    contiguous block [L..R] of LCP >= L, and return the suffix starts
    in sa[L-1 .. R] that share at least L characters.
    """
    n = len(sa)
    left = idx
    while left - 1 >= 1 and lcp[left - 1] >= L:
        left -= 1
    right = idx
    while right + 1 < n and lcp[right + 1] >= L:
        right += 1
    starts = sa[left - 1: right + 1]
    return starts

def _norm_positions_for_circular(starts, n0):
    """Map positions from doubled string back into [0, n0), dedupe while keeping order."""
    seen = set()
    out = []
    for p in starts:
        q = p % n0
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out

def find_longest_repeats(seq: str, circular: bool = False, min_len: int = 2, top_n: int = 10) -> Dict:
    """
    Return:
      {
        "longest": {
            "length": L, "pattern": str, "positions": [p1, p2, ...], "count": k
        },
        "top": [ { "length": L, "pattern": str, "positions": [...], "count": k }, ... ]
      }
    Notes:
      - Exact repeats only (no mismatches).
      - For circular=True, repeats that wrap the origin are found and positions are modulo len(seq).
      - min_len filters what goes into the 'top' list; 'longest' is always returned if any repeat exists.
    """
    s = seq.upper().replace("U", "T")
    n0 = len(s)
    if n0 < 2:
        return {"longest": None, "top": []}

    # Double sequence if circular to capture wrap-around
    s2 = s + s if circular else s
    n = len(s2)

    sa = suffix_array(s2)
    lcp = lcp_array(s2, sa)

    # Find the single longest repeated substring (global max of LCP)
    # Collect only if both starts map into unique positions under circular constraints.
    best_len = 0
    best_block_starts = []

    for i in range(1, n):
        L = lcp[i]
        if L <= 0:
            continue
        if L < best_len:
            continue
        # For circular, we allow starts anywhere in s2 but normalize mod n0 and dedupe.
        starts = _collect_block(sa, lcp, i, L)
        norm = _norm_positions_for_circular(starts, n0) if circular else sorted({p for p in starts if p < n0})
        if len(norm) >= 2:
            if L > best_len:
                best_len = L
                best_block_starts = norm
            elif L == best_len and len(norm) > len(best_block_starts):
                # prefer more occurrences at same length
                best_block_starts = norm

    longest = None
    if best_len > 0:
        # choose the first representative to extract the pattern
        rep_start = best_block_starts[0]
        pattern = (s + s)[rep_start:rep_start + best_len] if circular else s[rep_start:rep_start + best_len]
        longest = {
            "length": best_len,
            "pattern": pattern,
            "positions": best_block_starts,
            "count": len(best_block_starts),
        }

    # Build 'top' list (unique patterns >= min_len), limited by top_n
    # We iterate LCP entries in descending L, harvesting unique (pattern, positions) tuples.
    items = []
    seen_keys = set()
    # Make a list of (L, i) pairs where L >= min_len
    pairs = [(lcp[i], i) for i in range(1, n) if lcp[i] >= max(min_len, 1)]
    pairs.sort(key=lambda x: (-x[0], x[1]))

    for L, i in pairs:
        starts = _collect_block(sa, lcp, i, L)
        norm = _norm_positions_for_circular(starts, n0) if circular else sorted({p for p in starts if p < n0})
        if len(norm) < 2:
            continue
        rep_start = norm[0]
        patt = (s + s)[rep_start:rep_start + L] if circular else s[rep_start:rep_start + L]
        key = (L, tuple(norm))  # using positions avoids storing huge pattern texts in the key
        if key in seen_keys:
            continue
        seen_keys.add(key)
        items.append({
            "length": L,
            "pattern": patt,
            "positions": norm,
            "count": len(norm),
        })
        if len(items) >= top_n:
            break

    return {"longest": longest, "top": items}

def read_fasta_any(path: Path) -> Iterator[Tuple[str, str]]:
    """
    Minimal FASTA reader for uncompressed or .gz files.
    Yields (header, sequence). Header excludes the leading '>'.
    """
    if not path.exists():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        header, chunks = None, []
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            yield header, "".join(chunks)

def find_files(root: Path, suffixes: List[str]) -> List[Path]:
    if root.is_file():
        return [root] if root.exists() else []
    files = []
    for suf in suffixes:
        files.extend(root.rglob(f"*{suf}"))
    return sorted({p for p in files if p.exists() and p.is_file()})

def standard_plasmid_id(fp: Path) -> str:
    """
    Return a stable ID from the filename, matching how the rest of your QC uses IDs.
    Strips double extensions like *.fasta.gz, *.fa.gz, etc.
    """
    name = fp.name
    if name.endswith(".gz"):
        name = name[:-3]               # drop .gz
    return Path(name).stem             # drop .fa/.fasta/.fna/.fas

# ── Reverse-complement support (no external deps) ────────────────────────
_RC = str.maketrans("ACGTUNacgtun", "TGCAANtgcaan")

def reverse_complement(s: str) -> str:
    return s.translate(_RC)[::-1]

def map_rc_positions_to_fwd(rc_positions: List[int], match_len: int, n: int, circular: bool) -> List[int]:
    """
    Map start positions from the reverse-complement string back to forward-strand
    0-based coordinates. For a match of length L that begins at p_rc in RC(s),
    the forward coordinate is: p_fwd = n - (p_rc + L). Circular wraps modulo n.
    """
    if n <= 0:
        return []
    out = []
    seen = set()
    for p_rc in rc_positions:
        p = n - (p_rc + match_len)
        if circular:
            p %= n
        if 0 <= p < n and p not in seen:
            seen.add(p)
            out.append(p)
    out.sort()
    return out

def find_direct_and_inverted_longest(seq: str, circular: bool, min_len: int):
    # Direct (same-orientation) exact repeats — unchanged:
    direct = find_longest_repeats(seq, circular=circular, min_len=min_len, top_n=0)["longest"]

    # Inverted (reverse-complement) exact repeats — cross-compare s vs rc(s):
    inverted = find_longest_inverted_repeat(seq, circular=circular, min_len=min_len)

    return direct, inverted

def find_longest_inverted_repeat(seq: str, circular: bool = False, min_len: int = 2):
    """
    Find the longest exact inverted (reverse-complement) repeat by computing the
    longest common substring between s (or s+s if circular) and rc(s) (or rc+rc if circular).

    Returns a dict like:
      {
        "length": L,
        "pattern": forward_oriented_pattern,    # from the s-side
        "positions": [p1, p2, ...],             # all forward-strand starts for both arms
        "count": K
      }
    or None if no inverted repeat exists.
    """
    s = seq.upper().replace("U", "T")
    if len(s) < 2:
        return None

    n0 = len(s)
    s2  = s + s if circular else s
    rc2 = reverse_complement(s)
    rc2 = rc2 + rc2 if circular else rc2

    SEP = "#"  # not in DNA alphabet
    T = s2 + SEP + rc2
    sa = suffix_array(T)
    lcp = lcp_array(T, sa)

    # Tag each suffix as coming from s2 (0) or rc2 (1) or SEP (-1)
    len_s2 = len(s2)
    len_sep = 1
    def origin(idx):
        if idx < len_s2:
            return 0
        elif idx == len_s2:
            return -1
        else:
            return 1

    best_len = 0
    best_block_idx = None

    # scan LCP, but only consider adjacent suffixes from different origins (0 vs 1)
    for i in range(1, len(T)):
        o1 = origin(sa[i-1]); o2 = origin(sa[i])
        if o1 == -1 or o2 == -1 or o1 == o2:
            continue
        L = lcp[i]
        if L >= max(min_len, 1) and L >= best_len:
            best_len = L
            best_block_idx = i

    if best_len == 0 or best_block_idx is None:
        return None

    # collect the whole block around best_block_idx with LCP >= best_len
    starts_block = _collect_block(sa, lcp, best_block_idx, best_len)

    # split starts into s2-side and rc2-side, map to forward coords
    s2_starts  = []
    rc2_starts = []
    for st in starts_block:
        o = origin(st)
        if o == 0:
            s2_starts.append(st)                  # forward orientation directly
        elif o == 1:
            rc2_starts.append(st - (len_s2 + len_sep))

    # normalise to [0, n0)
    def norm_positions_forward(ps):
        seen = set(); out = []
        for p in ps:
            q = p % n0 if circular else p
            if 0 <= q < n0 and q not in seen:
                seen.add(q); out.append(q)
        out.sort()
        return out

    s2_norm  = norm_positions_forward(s2_starts)
    # map rc positions to forward coords for the *other arm* starts
    rc2_norm = map_rc_positions_to_fwd(rc2_starts, best_len, n0, circular)

    # if no valid pair across sides after normalisation, bail
    if not s2_norm or not rc2_norm:
        return None

    # pattern: take a representative from s2 side
    rep = s2_norm[0]
    pattern = (s + s)[rep:rep + best_len] if circular else s[rep:rep + best_len]

    # merge both arms' forward coordinates for reporting (dedup/sort)
    merged = sorted(set(s2_norm) | set(rc2_norm))

    return {
        "length": best_len,
        "pattern": pattern,       # forward-oriented arm
        "positions": merged,      # all forward starts of both arms
        "count": len(merged),
    }

def main():
    ap = argparse.ArgumentParser(description="Batch longest repeated region scan for FASTA files.")
    ap.add_argument("path", type=str, help="Path to a FASTA file or a directory of FASTA files")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out", type=str, default=None, help="Output CSV path")
    ap.add_argument("--suffixes", type=str, default=".fa,.fasta,.fna,.fas,.fa.gz,.fasta.gz",
                    help="Comma-separated list of file extensions to include")
    ap.add_argument("--circular", action="store_true", help="Treat sequences as circular (wrap-around repeats)")
    ap.add_argument("--min-len", type=int, default=2, help="Min length used inside finder for its 'top' list")
    ap.add_argument("--omit-seq", action="store_true", help="Exclude the repeat sequence text from CSV")
    ap.add_argument("--include-reverse", action="store_true",
                    help="Also detect inverted (reverse-complement) exact repeats")
    args = ap.parse_args()

    root = Path(args.path)
    suffixes = [s.strip() for s in args.suffixes.split(",") if s.strip()]
    files = find_files(root, suffixes)
    if not files:
        raise SystemExit(f"No FASTA files found under: {root}")

    if args.out:
        out_path = Path(args.out)
    elif args.run_name:
        out_path = Path("runs") / args.run_name / "qc" / "repeats.csv"
    else:
        out_path = Path("longest_repeats.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "plasmid_id", "seq_length", "circular",
        "longest_len", "longest_count", "longest_positions", "longest_fraction",
    ]
    if not args.omit_seq:
        fieldnames.append("longest_seq")

    # Extra columns for inverted (reverse-complement) repeats
    if args.include_reverse:
        fieldnames.extend([
            "inv_longest_len", "inv_longest_count", "inv_longest_positions", "inv_longest_fraction",
        ])
        if not args.omit_seq:
            fieldnames.append("inv_longest_seq")

    with open(out_path, "w", newline="") as csvfh:
        w = csv.DictWriter(csvfh, fieldnames=fieldnames)
        w.writeheader()

        for fp in files:
            # Some files may contain multiple records; handle all.
            for header, seq in read_fasta_any(fp):
                seq = seq.upper().replace("U", "T")
                n0 = len(seq)
                pid = standard_plasmid_id(fp)

                # Run dual-scan (direct + inverted)
                direct, inverted = find_direct_and_inverted_longest(
                    seq, circular=args.circular, min_len=args.min_len
                )

                # ----- DIRECT (forward) -----
                if direct is None:
                    row = {
                        "plasmid_id": pid,
                        "seq_length": n0,
                        "circular": bool(args.circular),
                        "longest_len": 0,
                        "longest_count": 0,
                        "longest_positions": "",
                        "longest_fraction": 0.0,
                    }
                    if not args.omit_seq:
                        row["longest_seq"] = ""
                else:
                    Ld = int(direct["length"])
                    positions_d = direct["positions"]
                    count_d = int(direct["count"])
                    frac_d = (Ld / n0) if n0 else 0.0
                    row = {
                        "plasmid_id": pid,
                        "seq_length": n0,
                        "circular": bool(args.circular),
                        "longest_len": Ld,
                        "longest_count": count_d,
                        "longest_positions": ";".join(map(str, positions_d)),
                        "longest_fraction": f"{frac_d:.6f}",
                    }
                    if not args.omit_seq:
                        row["longest_seq"] = direct["pattern"]

                # ----- INVERTED (reverse-complement), optional -----
                if args.include_reverse:
                    if inverted is None:
                        row.update({
                            "inv_longest_len": 0,
                            "inv_longest_count": 0,
                            "inv_longest_positions": "",
                            "inv_longest_fraction": 0.0,
                        })
                        if not args.omit_seq:
                            row["inv_longest_seq"] = ""
                    else:
                        Li = int(inverted["length"])
                        positions_i = inverted["positions"]       # mapped to forward coords
                        count_i = int(inverted["count"])
                        frac_i = (Li / n0) if n0 else 0.0
                        row.update({
                            "inv_longest_len": Li,
                            "inv_longest_count": count_i,
                            "inv_longest_positions": ";".join(map(str, positions_i)),
                            "inv_longest_fraction": f"{frac_i:.6f}",
                        })
                        if not args.omit_seq:
                            # keep the motif as returned from RC pass (RC orientation)
                            row["inv_longest_seq"] = inverted["pattern"]

                w.writerow(row)

    print(f"Wrote: {out_path}  (records: {sum(1 for _ in open(out_path)) - 1})")

if __name__ == "__main__":
    main()
