"""Strict QC for generated plasmid sequences.

The public entry point is :func:`run_strict_qc`. It chains four steps:

1. Bulk BLAST of generated sequences against a curated origin-of-replication
   database, then non-overlapping highest-identity selection per sequence.
2. AMRFinderPlus on the same FASTA in batches of 500 sequences (one
   ``amrfinder`` invocation per batch — the underlying ``blastx`` step
   loads all hits at once and exhausts memory on larger inputs).
3. Longest exact-repeat detection per sequence with a multiprocessing pool.
4. Two-stage filter at paper thresholds: each sequence must have exactly one
   strict ORI hit (≥99% identity, ≥99% subject coverage), at least one
   strict AMR hit (100% identity, 100% coverage), and no exact repeat
   ≥50 bp.

System tools required on ``$PATH``: ``blastn``, ``makeblastdb``, ``amrfinder``.
The conda environment in ``environment.yml`` provides them.

The reference ORI database used by the paper is at ``data/canonical_oris.fasta``;
:func:`run_strict_qc` builds the BLAST database on demand if it does not
already exist next to the FASTA.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from .blast_ori import (
    amrfinder_nucl,
    blast_oris,
    choose_non_overlapping_highest_identity,
    filter_ori_hits,
    standardize_amr_df,
)
from .filter import two_stage_filter
from .repeats import find_longest_repeats

log = logging.getLogger(__name__)

DEFAULT_ORI_FASTA = Path(__file__).resolve().parents[2] / "data" / "canonical_oris.fasta"

# Paper thresholds — see Section 3 of the paper. Don't change without
# re-deriving the headline pass-rate numbers.
ORI_STRICT_ID = 99.0
ORI_STRICT_COV = 99.0
ORI_MIN_LEN = 100
AMR_STRICT_ID = 100.0
AMR_STRICT_COV = 100.0
REPEAT_MAX_LEN = 50
AMR_BATCH_SIZE = 500


def run_strict_qc(
    input_csv: Path,
    out_dir: Path,
    *,
    ori_fasta: Path = DEFAULT_ORI_FASTA,
    n_repeat_workers: int = 14,
) -> dict:
    """Run strict QC on the generated sequences in ``input_csv``.

    ``input_csv`` must have ``id`` and ``full`` columns (the cleaned ACGT
    sequence). Outputs to ``out_dir``:

    - ``passed.csv`` and ``failed.csv`` from the two-stage filter
    - ``aggregate_ori_calls.csv`` and ``aggregate_amr_calls.csv``
    - ``repeats.csv``

    Returns a dict with ``n_total``, ``n_passed`` and ``pass_rate_pct``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _require_tool("blastn")
    _require_tool("makeblastdb")
    _require_tool("amrfinder")

    df = pd.read_csv(input_csv)
    if "id" not in df.columns or "full" not in df.columns:
        raise ValueError(
            f"{input_csv} must have 'id' and 'full' columns; got {list(df.columns)}"
        )

    fa = out_dir / "combined.fasta"
    n_written = _write_combined_fasta(df, fa)
    log.info("wrote %d sequences to %s", n_written, fa)

    db_prefix = _ensure_blast_db(ori_fasta)
    df_blast = blast_oris(fa, db_prefix, out_dir / "ori.blast.tsv", threads=8)
    ori_calls = _aggregate_ori_per_seq(df_blast)
    ori_csv = out_dir / "aggregate_ori_calls.csv"
    ori_calls.to_csv(ori_csv, index=False)
    log.info("ori: %d calls across %d sequences",
             len(ori_calls), ori_calls["sequence"].nunique() if not ori_calls.empty else 0)

    amr_calls = _run_amr_batched(fa, out_dir / "amr_batches", out_dir / "amr_batch_out")
    amr_csv = out_dir / "aggregate_amr_calls.csv"
    amr_calls.to_csv(amr_csv, index=False)
    log.info("amr: %d calls across %d sequences",
             len(amr_calls), amr_calls["sequence"].nunique() if not amr_calls.empty else 0)

    repeats = _run_repeats_parallel(df, n_workers=n_repeat_workers)
    repeats_csv = out_dir / "repeats.csv"
    repeats.to_csv(repeats_csv, index=False)

    passed_csv = out_dir / "passed.csv"
    failed_csv = out_dir / "failed.csv"
    two_stage_filter(
        qc_out=out_dir,
        out_pass_csv=passed_csv,
        out_fail_csv=failed_csv,
        ori_low_id=85.0, ori_low_cov=80.0,
        amr_low_id=85.0, amr_low_cov=80.0,
        ori_low_count_min=1, ori_low_count_max=1,
        amr_low_count_min=1, amr_low_count_max=1,
        ori_strict_id=ORI_STRICT_ID, ori_strict_cov=ORI_STRICT_COV,
        amr_strict_id=AMR_STRICT_ID, amr_strict_cov=AMR_STRICT_COV,
        amr_strict_min=None, amr_strict_all=False,
        repeats_csv=repeats_csv,
        repeat_max_len=REPEAT_MAX_LEN, repeat_ge=True,
    )
    n_total = len(df)
    n_passed = sum(1 for _ in open(passed_csv)) - 1 if passed_csv.exists() else 0
    pass_rate = round(100 * n_passed / n_total, 3) if n_total else 0.0
    log.info("strict QC: %d/%d = %.2f%%", n_passed, n_total, pass_rate)
    return {"n_total": n_total, "n_passed": n_passed, "pass_rate_pct": pass_rate}


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(
            f"required tool '{name}' is not on $PATH; "
            "install via the conda environment in environment.yml"
        )


def _write_combined_fasta(df: pd.DataFrame, fa_path: Path) -> int:
    n = 0
    with open(fa_path, "w") as f:
        for sid, seq in zip(df["id"], df["full"]):
            if not isinstance(seq, str):
                continue
            seq = seq.upper().replace(" ", "").translate(
                str.maketrans("", "", "RYSWKMBDHVN"))
            if len(seq) < 50:
                continue
            f.write(f">{sid}\n{seq}\n")
            n += 1
    return n


def _ensure_blast_db(ori_fasta: Path) -> Path:
    db_prefix = ori_fasta.with_suffix("")
    if not (ori_fasta.parent / f"{db_prefix.name}.nhr").exists():
        log.info("building BLAST database at %s", db_prefix)
        subprocess.run(
            ["makeblastdb", "-in", str(ori_fasta), "-dbtype", "nucl",
             "-out", str(db_prefix)],
            check=True, capture_output=True,
        )
    return db_prefix


def _aggregate_ori_per_seq(df_blast: pd.DataFrame) -> pd.DataFrame:
    cols = ["sequence", "ori_type", "pct_identity", "pct_cov_subject",
            "q_start", "q_end", "strand", "qlen", "sstart", "send", "slen",
            "length", "bitscore", "evalue"]
    df_low = filter_ori_hits(df_blast, min_pident=85.0, min_scovs=80.0,
                             min_len=ORI_MIN_LEN)
    if df_low.empty:
        return pd.DataFrame(columns=cols)
    parts = []
    for sid, sub in df_low.groupby("qseqid", sort=False):
        chosen = choose_non_overlapping_highest_identity(sub.copy())
        if chosen.empty:
            continue
        chosen = chosen.rename(columns={
            "sseqid": "ori_type", "pident": "pct_identity",
            "scovs": "pct_cov_subject", "q_from": "q_start", "q_to": "q_end",
        })
        chosen.insert(0, "sequence", sid)
        if "qseqid" in chosen.columns:
            chosen = chosen.drop(columns=["qseqid"])
        parts.append(chosen[[c for c in cols if c in chosen.columns]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def _split_fasta(fa_path: Path, batch_dir: Path, batch_size: int) -> list[Path]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    batches: list[Path] = []
    fh = None
    cur_count = 0
    cur_idx = 0
    with open(fa_path) as src:
        for line in src:
            if line.startswith(">"):
                if cur_count >= batch_size or fh is None:
                    if fh is not None:
                        fh.close()
                    cur_path = batch_dir / f"batch_{cur_idx:04d}.fasta"
                    batches.append(cur_path)
                    fh = open(cur_path, "w")
                    cur_idx += 1
                    cur_count = 0
                cur_count += 1
            fh.write(line)
    if fh is not None:
        fh.close()
    return batches


def _run_amr_batched(fa_path: Path, batch_dir: Path, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    batches = _split_fasta(fa_path, batch_dir, AMR_BATCH_SIZE)
    log.info("amrfinder on %d batches of <=%d sequences", len(batches), AMR_BATCH_SIZE)
    parts = []
    for i, b in enumerate(batches):
        out_tsv = out_dir / f"batch_{i:04d}.amr.tsv"
        if not out_tsv.exists() or out_tsv.stat().st_size == 0:
            raw = amrfinder_nucl(b, out_tsv, threads=8)
        else:
            raw = pd.read_csv(out_tsv, sep="\t", comment="#") if out_tsv.stat().st_size else pd.DataFrame()
        if raw is not None and not raw.empty:
            parts.append(raw)
    cols = ["sequence", "symbol", "name", "start", "end", "strand",
            "pct_identity", "pct_cov"]
    if not parts:
        return pd.DataFrame(columns=cols)
    raw_all = pd.concat(parts, ignore_index=True)
    std = standardize_amr_df(raw_all)
    if std.empty:
        return pd.DataFrame(columns=cols)
    cid_col = next(
        (c for c in raw_all.columns if "contig" in c.strip().lower()), None)
    if cid_col is None:
        raise RuntimeError(
            f"AMRFinder output has no Contig column; got {list(raw_all.columns)}")
    std.insert(0, "sequence", raw_all[cid_col].astype(str).values)
    return std


def _repeats_worker(args):
    sid, seq = args
    if not isinstance(seq, str) or len(seq) < 2:
        return {"plasmid_id": sid, "seq_length": 0, "circular": False,
                "longest_len": 0, "longest_count": 0,
                "longest_positions": "", "longest_fraction": 0.0}
    res = find_longest_repeats(seq, circular=False, min_len=2, top_n=0)
    longest = res["longest"]
    n0 = len(seq)
    if longest is None:
        return {"plasmid_id": sid, "seq_length": n0, "circular": False,
                "longest_len": 0, "longest_count": 0,
                "longest_positions": "", "longest_fraction": 0.0}
    return {"plasmid_id": sid, "seq_length": n0, "circular": False,
            "longest_len": int(longest["length"]),
            "longest_count": int(longest["count"]),
            "longest_positions": ";".join(map(str, longest["positions"])),
            "longest_fraction": float(longest["length"] / n0)}


def _run_repeats_parallel(df: pd.DataFrame, n_workers: int) -> pd.DataFrame:
    work = list(zip(df["id"].tolist(), df["full"].tolist()))
    log.info("repeats: %d sequences across %d workers", len(work), n_workers)
    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for r in pool.imap_unordered(_repeats_worker, work, chunksize=8):
            rows.append(r)
    return pd.DataFrame(rows)


__all__ = ["run_strict_qc", "DEFAULT_ORI_FASTA"]
