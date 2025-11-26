from __future__ import annotations

import os
import shutil
import subprocess as sp
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
import numpy as np
import pandas as pd


# -----------------------------
# Generic utils
# -----------------------------

def which(tool: str) -> Optional[str]:
    return shutil.which(tool)


def run(cmd: List[str], check: bool = True) -> sp.CompletedProcess:
    print("[RUN]", " ".join(cmd), flush=True)
    return sp.run(cmd, check=check)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_fasta(p: Path) -> bool:
    return p.suffix.lower() in {".fa", ".fna", ".fasta", ".ffn"}


def list_fastas(in_path: Path) -> List[Path]:
    if in_path.is_dir():
        files = [p for p in in_path.rglob("*") if p.is_file() and is_fasta(p)]
        return sorted(files)
    if in_path.is_file() and is_fasta(in_path):
        return [in_path]
    raise ValueError(f"Input {in_path} is not a FASTA or directory of FASTAs.")


# -----------------------------
# BLAST (ori DB)
# -----------------------------

def ensure_blast_db(oridb_ref: Optional[Path], db_prefix: Path) -> Optional[Path]:
    """Ensure a BLAST DB exists at db_prefix.* (nucl). Build if ref provided and missing.
    Returns the db_prefix or None if BLAST tools are unavailable and cannot build.
    """
    db_exists = all((db_prefix.with_suffix(ext)).exists() for ext in (".nhr", ".nin", ".nsq"))
    if db_exists:
        return db_prefix
    if oridb_ref is None or not oridb_ref.exists():
        print(f"[WARN] ori DB missing at {db_prefix}.* and no valid --oridb-ref provided.")
        return None
    if not which("makeblastdb"):
        print("[WARN] makeblastdb not found on PATH; skipping ori BLAST DB build.")
        return None
    run(["makeblastdb", "-in", str(oridb_ref), "-dbtype", "nucl", "-out", str(db_prefix)])
    return db_prefix


def blast_oris(
    fasta: Path,
    db_prefix: Optional[Path],
    out_tsv: Path,
    task: str = "dc-megablast",
    evalue: str = "1e-20",
    max_hits: int = 2000,
    threads: int = 1,
) -> pd.DataFrame:
    if not which("blastn") or db_prefix is None:
        print("[WARN] blastn or DB not available; skipping ori BLAST.")
        cols = [
            "qseqid",
            "sseqid",
            "pident",
            "length",
            "evalue",
            "bitscore",
            "qstart",
            "qend",
            "qlen",
            "sstart",
            "send",
            "slen",
        ]
        return pd.DataFrame(columns=cols + ["qcov", "scovs", "q_from", "q_to", "strand"])

    outfmt = (
        "6 qseqid sseqid pident length evalue bitscore "
        "qstart qend qlen sstart send slen"
    )
    cmd = [
        "blastn",
        "-task",
        task,
        "-query",
        str(fasta),
        "-db",
        str(db_prefix),
        "-outfmt",
        outfmt,
        "-evalue",
        evalue,
        "-max_target_seqs",
        str(max_hits),
        "-soft_masking",
        "true",
        "-dust",
        "yes",
        "-num_threads",
        str(threads),
        "-out",
        str(out_tsv),
    ]
    run(cmd)

    cols = [
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "evalue",
        "bitscore",
        "qstart",
        "qend",
        "qlen",
        "sstart",
        "send",
        "slen",
    ]
    if not out_tsv.exists() or out_tsv.stat().st_size == 0:
        return pd.DataFrame(columns=cols + ["qcov", "scovs", "q_from", "q_to", "strand"])

    df = pd.read_csv(out_tsv, sep="\t", header=None, names=cols)

    # numerics
    for c in [
        "pident",
        "length",
        "evalue",
        "bitscore",
        "qstart",
        "qend",
        "qlen",
        "sstart",
        "send",
        "slen",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # compute coverages robustly
    df["qcov"] = 100.0 * df["length"] / df["qlen"].replace(0, np.nan)
    df["scovs"] = 100.0 * df["length"] / df["slen"].replace(0, np.nan)

    # normalize intervals/strand
    df["q_from"] = df[["qstart", "qend"]].min(axis=1).astype("Int64")
    df["q_to"] = df[["qstart", "qend"]].max(axis=1).astype("Int64")
    df["strand"] = np.where(df["sstart"] <= df["send"], "+", "-")

    return df


def filter_ori_hits(
    df: pd.DataFrame, min_pident: float, min_scovs: float, min_len: int
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "qseqid",
                "sseqid",
                "pident",
                "length",
                "evalue",
                "bitscore",
                "qstart",
                "qend",
                "qlen",
                "sstart",
                "send",
                "slen",
                "qcovs",
                "scovs",
                "q_from",
                "q_to",
                "strand",
            ]
        )
    keep = (
        (df["pident"] >= min_pident)
        & (df["scovs"] >= min_scovs)
        & (df["length"] >= min_len)
    )
    df2 = df.loc[keep].copy()
    if df2.empty:
        return df2
    df2 = df2.sort_values(
        by=["pident", "bitscore", "scovs", "length"],
        ascending=[False, False, False, False],
    )
    return df2


def choose_non_overlapping_highest_identity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(
        by=["pident", "bitscore", "scovs", "length"],
        ascending=[False, False, False, False],
    )
    chosen: List[pd.Series] = []
    intervals: List[Tuple[int, int]] = []
    for _, r in df.iterrows():
        s, e = int(r.q_from), int(r.q_to)
        overlap = any(not (e < cs or s > ce) for cs, ce in intervals)
        if not overlap:
            chosen.append(r)
            intervals.append((s, e))
    return pd.DataFrame(chosen)


# -----------------------------
# AMRFinder (ARGs)
# -----------------------------

def amrfinder_nucl(fasta: Path, out_tsv: Path, threads: int = 1) -> pd.DataFrame:
    tool = which("amrfinder")
    if not tool:
        print("[WARN] amrfinder not found; skipping ARG detection.")
        return pd.DataFrame()
    cmd = [tool, "-n", str(fasta), "-o", str(out_tsv)]
    if threads and threads > 1:
        cmd += ["--threads", str(threads)]
    run(cmd)
    if not out_tsv.exists() or out_tsv.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(out_tsv, sep="\t", comment="#")


def standardize_amr_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "start",
                "end",
                "strand",
                "pct_identity",
                "pct_cov",
            ]
        )

    def grab(*cands: str) -> Optional[str]:
        for c in df.columns:
            cl = c.strip().lower()
            for t in cands:
                if cl == t or t in cl:
                    return c
        return None

    col_symbol = grab("element symbol", "gene symbol", "symbol")
    col_name = grab("element name", "name")
    col_start = grab("start")
    col_end = grab("end", "stop")
    col_strand = grab("strand")
    col_pid = grab("% identity to reference", "identity")
    col_pcov = grab("% coverage of reference", "coverage")

    out = pd.DataFrame(
        {
            "symbol": df[col_symbol] if col_symbol in df else "",
            "name": df[col_name] if col_name in df else "",
            "start": df[col_start] if col_start in df else "",
            "end": df[col_end] if col_end in df else "",
            "strand": df[col_strand] if col_strand in df else "",
            "pct_identity": df[col_pid] if col_pid in df else "",
            "pct_cov": df[col_pcov] if col_pcov in df else "",
        }
    )
    for c in ["start", "end", "pct_identity", "pct_cov"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# -----------------------------
# Prodigal (genes)
# -----------------------------

def _fasta_len_bp(fasta: Path) -> int:
    n = 0
    with fasta.open() as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            n += len(line.strip())
    return n


def run_prodigal(
    fasta: Path, out_prefix: Path, closed_circular: bool = True
) -> None:
    tool = which("prodigal")
    if not tool:
        print("[WARN] prodigal not found; skipping gene prediction.")
        return

    gff = out_prefix.with_suffix(".gff")
    faa = out_prefix.with_suffix(".faa")
    fna = out_prefix.with_suffix(".fna")
    gbk = out_prefix.with_suffix(".gbk")

    L = _fasta_len_bp(fasta)
    mode = "single" if L >= 20000 else "meta"

    base = [tool, "-i", str(fasta), "-p", mode]
    if closed_circular:
        base += ["-c"]

    run(base + ["-o", str(gbk), "-a", str(faa), "-d", str(fna)])
    run([tool, "-i", str(fasta), "-p", mode, "-f", "gff", "-o", str(gff)] + (["-c"] if closed_circular else []))


# -----------------------------
# S3 helpers
# -----------------------------

def parse_s3_uri(uri: str) -> Tuple[str, str]:
    s = uri.replace("s3://", "")
    parts = s.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key.rstrip("/") + "/"


def s3_list_fastas(client, bucket: str, prefix: str) -> List[str]:
    suffixes = (".fa", ".fna", ".fasta", ".ffn")
    keys: List[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            k = obj["Key"]
            if k.endswith("/"):
                continue
            if k.lower().endswith(suffixes):
                keys.append(k)
    return keys


def s3_download_files(client, bucket: str, keys: List[str], dest_dir: Path) -> List[Path]:
    ensure_dir(dest_dir)
    out: List[Path] = []
    for k in keys:
        local = dest_dir / Path(k).name
        ensure_dir(local.parent)
        client.download_file(bucket, k, str(local))
        out.append(local)
    return out


def s3_upload_dir(client, local_dir: Path, bucket: str, prefix: str) -> None:
    for root, _, files in os.walk(local_dir):
        for fname in files:
            fpath = Path(root) / fname
            rel = fpath.relative_to(local_dir).as_posix()
            key = prefix.rstrip("/") + "/" + rel
            with open(fpath, "rb") as fh:
                client.put_object(Bucket=bucket, Key=key, Body=fh.read())


# -----------------------------
# Orchestration per FASTA
# -----------------------------

def process_one(
    fasta: Path,
    outdir: Path,
    db_prefix: Optional[Path],
    min_pident: float,
    min_scovs: float,
    min_len: int,
    threads: int,
    skip_prodigal: bool = False,
) -> Dict[str, Any]:
    sample = fasta.stem
    sdir = outdir / sample
    ensure_dir(sdir)

    # 1) ori BLAST
    raw_blast_tsv = sdir / f"{sample}.ori.blast.tsv"
    df_blast = blast_oris(fasta, db_prefix, raw_blast_tsv, threads=threads)
    df_blast_f = filter_ori_hits(df_blast, min_pident, min_scovs, min_len)

    df_ori_final = pd.DataFrame()
    if not df_blast_f.empty:
        df_ori_final = choose_non_overlapping_highest_identity(df_blast_f).copy()
        if not df_ori_final.empty:
            keep_cols = [
                "qseqid",
                "sseqid",
                "pident",
                "scovs",
                "q_from",
                "q_to",
                "strand",
                "qlen",
                "sstart",
                "send",
                "slen",
                "length",
                "bitscore",
                "evalue",
            ]
            for c in keep_cols:
                if c not in df_ori_final.columns:
                    df_ori_final[c] = pd.NA
            df_ori_final = df_ori_final[keep_cols]
            df_ori_final.insert(0, "sequence", sample)
            df_ori_final.rename(
                columns={
                    "sseqid": "ori_type",
                    "pident": "pct_identity",
                    "scovs": "pct_cov_subject",
                    "q_from": "q_start",
                    "q_to": "q_end",
                },
                inplace=True,
            )

    ori_csv = sdir / f"{sample}.ori_calls.csv"
    (
        df_ori_final
        if not df_ori_final.empty
        else pd.DataFrame(
            columns=[
                "sequence",
                "ori_type",
                "pct_identity",
                "pct_cov_subject",
                "q_start",
                "q_end",
                "strand",
                "qlen",
                "sstart",
                "send",
                "slen",
                "length",
                "bitscore",
                "evalue",
            ]
        )
    ).to_csv(ori_csv, index=False)

    # 2) AMRFinder (nucleotide)
    amr_raw_tsv = sdir / f"{sample}.amrfinder.tsv"
    df_amr_raw = amrfinder_nucl(fasta, amr_raw_tsv, threads=threads)
    df_amr_std = standardize_amr_df(df_amr_raw)
    if not df_amr_std.empty:
        df_amr_std.insert(0, "sequence", sample)
    amr_csv = sdir / f"{sample}.amr_calls.csv"
    (
        df_amr_std
        if not df_amr_std.empty
        else pd.DataFrame(
            columns=[
                "sequence",
                "symbol",
                "name",
                "start",
                "end",
                "strand",
                "pct_identity",
                "pct_cov",
            ]
        )
    ).to_csv(amr_csv, index=False)

    # 3) Prodigal
    prodigal_done = False
    if not skip_prodigal:
        run_prodigal(fasta, sdir / sample, closed_circular=True)
        prodigal_done = True

    return {
        "sample": sample,
        "ori_csv": str(ori_csv),
        "amr_csv": str(amr_csv),
        "prodigal": prodigal_done,
        "n_ori_kept": int(0 if df_ori_final is None or df_ori_final.empty else len(df_ori_final)),
        "n_amr": int(0 if df_amr_std is None or df_amr_std.empty else len(df_amr_std)),
    }


# -----------------------------
# Entrypoint
# -----------------------------

def main(
    input_path: str,
    oridb_prefix: Optional[str] = None,
    oridb_ref: Optional[str] = None,
    threads: int = 1,
    skip_prodigal: bool = False,
    out_subdir: str = "qc_out",
) -> Dict[str, Any]:
    """Run QC on a local directory or an S3 prefix of FASTA files.

    Returns a dictionary with locations of the written reports.
    """

    is_s3 = input_path.startswith("s3://")
    s3 = boto3.client("s3") if is_s3 else None

    if is_s3:
        bucket, prefix = parse_s3_uri(input_path)
        keys = s3_list_fastas(s3, bucket, prefix)
        if not keys:
            raise SystemExit(f"No FASTA files found under: {input_path}")
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            in_dir = tdir / "inputs"
            out_dir_local = tdir / "out"
            ensure_dir(in_dir)
            ensure_dir(out_dir_local)
            local_fastas = s3_download_files(s3, bucket, keys, in_dir)

            # Prepare BLAST DB locally
            db_prefix_local = Path(oridb_prefix) if oridb_prefix else Path(tdir / "oridb")
            db_prefix_ready = ensure_blast_db(
                Path(oridb_ref) if oridb_ref else None, db_prefix_local
            )

            summaries: List[Dict[str, Any]] = []
            all_ori_rows: List[pd.DataFrame] = []
            all_amr_rows: List[pd.DataFrame] = []

            print(f"[INFO] Found {len(local_fastas)} FASTA(s) in {input_path}.")
            for fa in local_fastas:
                try:
                    res = process_one(
                        fasta=fa,
                        outdir=out_dir_local,
                        db_prefix=db_prefix_ready,
                        min_pident=85.0,
                        min_scovs=80.0,
                        min_len=100,
                        threads=threads,
                        skip_prodigal=skip_prodigal,
                    )
                    summaries.append(res)
                    ori_csv = Path(res["ori_csv"]) if "ori_csv" in res else None
                    amr_csv = Path(res["amr_csv"]) if "amr_csv" in res else None
                    if ori_csv and ori_csv.exists() and ori_csv.stat().st_size > 0:
                        df = pd.read_csv(ori_csv)
                        if not df.empty:
                            all_ori_rows.append(df)
                    if amr_csv and amr_csv.exists() and amr_csv.stat().st_size > 0:
                        df = pd.read_csv(amr_csv)
                        if not df.empty:
                            all_amr_rows.append(df)
                except sp.CalledProcessError as e:
                    print(f"[ERROR] Tool failed on {fa.name}: {e}", flush=True)
                except Exception as e:
                    print(f"[ERROR] {fa.name}: {e}", flush=True)

            # Aggregates
            agg_ori = (
                pd.concat(all_ori_rows, ignore_index=True)
                if all_ori_rows
                else pd.DataFrame(
                    columns=[
                        "sequence",
                        "ori_type",
                        "pct_identity",
                        "pct_cov_subject",
                        "q_start",
                        "q_end",
                        "strand",
                        "qlen",
                        "sstart",
                        "send",
                        "slen",
                        "length",
                        "bitscore",
                        "evalue",
                    ]
                )
            )
            agg_amr = (
                pd.concat(all_amr_rows, ignore_index=True)
                if all_amr_rows
                else pd.DataFrame(
                    columns=[
                        "sequence",
                        "symbol",
                        "name",
                        "start",
                        "end",
                        "strand",
                        "pct_identity",
                        "pct_cov",
                    ]
                )
            )
            agg_ori_path = out_dir_local / "aggregate_ori_calls.csv"
            agg_amr_path = out_dir_local / "aggregate_amr_calls.csv"
            summary_csv = out_dir_local / "qc_summary.csv"
            agg_ori.to_csv(agg_ori_path, index=False)
            agg_amr.to_csv(agg_amr_path, index=False)
            pd.DataFrame(summaries).to_csv(summary_csv, index=False)

            # Upload to S3 under the same prefix/out_subdir
            out_prefix = prefix.rstrip("/") + f"/{out_subdir}"
            s3_upload_dir(s3, out_dir_local, bucket, out_prefix)

            s3_base = f"s3://{bucket}/{out_prefix.strip('/')}/"
            print("[DONE] QC complete.")
            print(f"  - Aggregate ORIs: {s3_base}aggregate_ori_calls.csv")
            print(f"  - Aggregate AMRs: {s3_base}aggregate_amr_calls.csv")
            print(f"  - Summary:        {s3_base}qc_summary.csv")

            return {
                "location": s3_base,
                "aggregate_ori": s3_base + "aggregate_ori_calls.csv",
                "aggregate_amr": s3_base + "aggregate_amr_calls.csv",
                "summary": s3_base + "qc_summary.csv",
            }
    else:
        in_path = Path(input_path)
        fastas = list_fastas(in_path)
        if not fastas:
            raise SystemExit(f"No FASTA files found under: {in_path}")
        outdir = in_path / out_subdir
        ensure_dir(outdir)

        db_prefix_ready = ensure_blast_db(
            Path(oridb_ref) if oridb_ref else None, Path(oridb_prefix) if oridb_prefix else outdir / "oridb"
        )

        summaries: List[Dict[str, Any]] = []
        all_ori_rows: List[pd.DataFrame] = []
        all_amr_rows: List[pd.DataFrame] = []

        print(f"[INFO] Found {len(fastas)} FASTA(s).")
        for fa in fastas:
            try:
                res = process_one(
                    fasta=fa,
                    outdir=outdir,
                    db_prefix=db_prefix_ready,
                    min_pident=85.0,
                    min_scovs=80.0,
                    min_len=100,
                    threads=threads,
                    skip_prodigal=skip_prodigal,
                )
                summaries.append(res)
                ori_csv = Path(res["ori_csv"]) if "ori_csv" in res else None
                amr_csv = Path(res["amr_csv"]) if "amr_csv" in res else None
                if ori_csv and ori_csv.exists() and ori_csv.stat().st_size > 0:
                    df = pd.read_csv(ori_csv)
                    if not df.empty:
                        all_ori_rows.append(df)
                if amr_csv and amr_csv.exists() and amr_csv.stat().st_size > 0:
                    df = pd.read_csv(amr_csv)
                    if not df.empty:
                        all_amr_rows.append(df)
            except sp.CalledProcessError as e:
                print(f"[ERROR] Tool failed on {fa.name}: {e}", flush=True)
            except Exception as e:
                print(f"[ERROR] {fa.name}: {e}", flush=True)

        agg_ori = (
            pd.concat(all_ori_rows, ignore_index=True)
            if all_ori_rows
            else pd.DataFrame(
                columns=[
                    "sequence",
                    "ori_type",
                    "pct_identity",
                    "pct_cov_subject",
                    "q_start",
                    "q_end",
                    "strand",
                    "qlen",
                    "sstart",
                    "send",
                    "slen",
                    "length",
                    "bitscore",
                    "evalue",
                ]
            )
        )
        agg_amr = (
            pd.concat(all_amr_rows, ignore_index=True)
            if all_amr_rows
            else pd.DataFrame(
                columns=[
                    "sequence",
                    "symbol",
                    "name",
                    "start",
                    "end",
                    "strand",
                    "pct_identity",
                    "pct_cov",
                ]
            )
        )
        agg_ori_path = outdir / "aggregate_ori_calls.csv"
        agg_amr_path = outdir / "aggregate_amr_calls.csv"
        summary_csv = outdir / "qc_summary.csv"
        agg_ori.to_csv(agg_ori_path, index=False)
        agg_amr.to_csv(agg_amr_path, index=False)
        pd.DataFrame(summaries).to_csv(summary_csv, index=False)

        print("[DONE] QC complete.")
        print(f"  - Aggregate ORIs: {agg_ori_path}")
        print(f"  - Aggregate AMRs: {agg_amr_path}")
        print(f"  - Summary:        {summary_csv}")

        return {
            "location": str(outdir.resolve()),
            "aggregate_ori": str(agg_ori_path.resolve()),
            "aggregate_amr": str(agg_amr_path.resolve()),
            "summary": str(summary_csv.resolve()),
        }
