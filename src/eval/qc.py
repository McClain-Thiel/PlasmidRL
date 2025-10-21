import os
import sys
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import click
import boto3
import pandas as pd
import numpy as np


# -----------------------------
# Utilities
# -----------------------------


def is_s3_uri(uri: str) -> bool:
    return uri.strip().lower().startswith("s3://")


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    raw = uri.strip()
    assert raw.startswith("s3://"), f"Not an s3 URI: {uri}"
    body = raw[5:]
    parts = body.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket, prefix.rstrip("/")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def which(tool: str) -> Optional[str]:
    return shutil.which(tool)


def run(cmd: List[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    print("[RUN]", " ".join(cmd), flush=True)
    if capture:
        return subprocess.run(cmd, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return subprocess.run(cmd, check=check)


def is_fasta_path(p: Path) -> bool:
    return p.suffix.lower() in {".fa", ".fna", ".fasta", ".ffn"} or any(
        str(p).lower().endswith(s) for s in [".fa.gz", ".fasta.gz", ".fna.gz"]
    )


def list_local_fastas(root: Path) -> List[Path]:
    if root.is_file() and is_fasta_path(root):
        return [root]
    if root.is_dir():
        return sorted([p for p in root.rglob("*") if p.is_file() and is_fasta_path(p)])
    raise FileNotFoundError(f"Path not found or not FASTA: {root}")


def list_s3_fastas(s3: boto3.client, bucket: str, prefix: str) -> List[str]:
    keys: List[str] = []
    cont_token: Optional[str] = None
    while True:
        kwargs: Dict[str, str] = {"Bucket": bucket, "Prefix": prefix}
        if cont_token:
            kwargs["ContinuationToken"] = cont_token
        resp = s3.list_objects_v2(**kwargs)
        for obj in resp.get("Contents", []):
            k = obj["Key"]
            lk = k.lower()
            if lk.endswith((".fa", ".fna", ".fasta", ".ffn", ".fa.gz", ".fasta.gz", ".fna.gz")):
                keys.append(k)
        if not resp.get("IsTruncated"):
            break
        cont_token = resp.get("NextContinuationToken")
    return keys


def download_s3_key(s3: boto3.client, bucket: str, key: str, dest_dir: Path) -> Path:
    ensure_dir(dest_dir)
    local = dest_dir / Path(key).name
    s3.download_file(bucket, key, str(local))
    return local


def upload_tree_to_s3(s3: boto3.client, local_dir: Path, bucket: str, prefix: str) -> None:
    for root, _, files in os.walk(local_dir):
        for name in files:
            fp = Path(root) / name
            rel = fp.relative_to(local_dir)
            key = "/".join([prefix.rstrip("/"), str(rel).replace(os.sep, "/")]) if prefix else str(rel).replace(os.sep, "/")
            print(f"[S3 PUT] s3://{bucket}/{key}")
            s3.upload_file(str(fp), bucket, key)


# -----------------------------
# BLAST / AMRFinder / Prodigal
# -----------------------------


def ensure_blast_db(oridb_prefix: Path, oridb_ref: Optional[Path]) -> Path:
    nhr, nin, nsq = [oridb_prefix.with_suffix(ext) for ext in (".nhr", ".nin", ".nsq")]
    if nhr.exists() and nin.exists() and nsq.exists():
        return oridb_prefix
    if oridb_ref is None or not oridb_ref.exists():
        raise FileNotFoundError(
            f"BLAST DB missing at {oridb_prefix}.* and no valid --oridb-ref provided."
        )
    if not which("makeblastdb"):
        raise RuntimeError("makeblastdb not found on PATH.")
    run(["makeblastdb", "-in", str(oridb_ref), "-dbtype", "nucl", "-out", str(oridb_prefix)])
    return oridb_prefix


def blast_oris(
    fasta: Path,
    db_prefix: Path,
    out_tsv: Path,
    task: str = "dc-megablast",
    evalue: str = "1e-20",
    max_hits: int = 2000,
    threads: int = 1,
) -> pd.DataFrame:
    if not which("blastn"):
        raise RuntimeError("blastn not found on PATH.")
    outfmt = "6 qseqid sseqid pident length evalue bitscore qstart qend qlen sstart send slen"
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
    df["qcov"] = 100.0 * df["length"] / df["qlen"].replace(0, np.nan)
    df["scovs"] = 100.0 * df["length"] / df["slen"].replace(0, np.nan)
    df["q_from"] = df[["qstart", "qend"]].min(axis=1).astype("Int64")
    df["q_to"] = df[["qstart", "qend"]].max(axis=1).astype("Int64")
    df["strand"] = np.where(df["sstart"] <= df["send"], "+", "-")
    return df


def filter_ori_hits(df: pd.DataFrame, min_pident: float, min_scovs: float, min_len: int) -> pd.DataFrame:
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
                "qcov",
                "scovs",
                "q_from",
                "q_to",
                "strand",
            ]
        )
    keep = (df["pident"] >= min_pident) & (df["scovs"] >= min_scovs) & (df["length"] >= min_len)
    df2 = df.loc[keep].copy()
    if df2.empty:
        return df2
    return df2.sort_values(by=["pident", "bitscore", "scovs", "length"], ascending=[False, False, False, False])


def choose_non_overlapping_highest_identity(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values(by=["pident", "bitscore", "scovs", "length"], ascending=[False, False, False, False])
    chosen = []
    intervals: List[Tuple[int, int]] = []
    for _, r in df.iterrows():
        s, e = int(r.q_from), int(r.q_to)
        overlap = any(not (e < cs or s > ce) for (cs, ce) in intervals)
        if not overlap:
            chosen.append(r)
            intervals.append((s, e))
    return pd.DataFrame(chosen)


def amrfinder_nucl(fasta: Path, out_tsv: Path, threads: int = 1) -> pd.DataFrame:
    if not which("amrfinder"):
        raise RuntimeError("amrfinder (AMRFinderPlus) not found on PATH.")
    cmd = ["amrfinder", "-n", str(fasta), "-o", str(out_tsv)]
    if threads and threads > 1:
        cmd += ["--threads", str(threads)]
    run(cmd)
    if not out_tsv.exists() or out_tsv.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(out_tsv, sep="\t", comment="#")


def standardize_amr_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["symbol", "name", "start", "end", "strand", "pct_identity", "pct_cov"])

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


def run_prodigal(fasta: Path, out_prefix: Path, closed_circular: bool = True) -> None:
    if not which("prodigal"):
        raise RuntimeError("prodigal not found on PATH.")

    def fasta_len(fp: Path) -> int:
        n = 0
        with fp.open() as fh:
            for line in fh:
                if line.startswith(">"):
                    continue
                n += len(line.strip())
        return n

    gff = out_prefix.with_suffix(".gff")
    faa = out_prefix.with_suffix(".faa")
    fna = out_prefix.with_suffix(".fna")
    gbk = out_prefix.with_suffix(".gbk")

    L = fasta_len(fasta)
    mode = "single" if L >= 20000 else "meta"
    base = ["prodigal", "-i", str(fasta), "-p", mode]
    if closed_circular:
        base += ["-c"]
    run(base + ["-o", str(gbk), "-a", str(faa), "-d", str(fna)])
    run(["prodigal", "-i", str(fasta), "-p", mode, "-f", "gff", "-o", str(gff)] + (["-c"] if closed_circular else []))


# -----------------------------
# QC per FASTA and aggregation
# -----------------------------


@dataclass
class PerSampleResult:
    sample: str
    ori_csv: Path
    amr_csv: Path
    prodigal_done: bool
    n_ori_kept: int
    n_amr: int


def process_one(
    fasta: Path,
    outdir: Path,
    db_prefix: Path,
    min_pident: float,
    min_scovs: float,
    min_len: int,
    threads: int,
    skip_prodigal: bool,
) -> PerSampleResult:
    sample = fasta.stem
    sdir = outdir / sample
    ensure_dir(sdir)

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

    amr_raw_tsv = sdir / f"{sample}.amrfinder.tsv"
    df_amr_raw = (
        amrfinder_nucl(fasta, amr_raw_tsv, threads=threads) if which("amrfinder") else pd.DataFrame()
    )
    df_amr_std = standardize_amr_df(df_amr_raw)
    if not df_amr_std.empty:
        df_amr_std.insert(0, "sequence", sample)
    amr_csv = sdir / f"{sample}.amr_calls.csv"
    (
        df_amr_std
        if not df_amr_std.empty
        else pd.DataFrame(columns=["sequence", "symbol", "name", "start", "end", "strand", "pct_identity", "pct_cov"])
    ).to_csv(amr_csv, index=False)

    prodigal_done = False
    if not skip_prodigal:
        run_prodigal(fasta, sdir / sample, closed_circular=True)
        prodigal_done = True

    return PerSampleResult(
        sample=sample,
        ori_csv=ori_csv,
        amr_csv=amr_csv,
        prodigal_done=prodigal_done,
        n_ori_kept=int(0 if df_ori_final is None or df_ori_final.empty else len(df_ori_final)),
        n_amr=int(0 if df_amr_std is None or df_amr_std.empty else len(df_amr_std)),
    )


def aggregate_and_report(outdir: Path, per_sample: List[PerSampleResult], report_path: Path,
                         low_id: float, low_cov: float,
                         strict_id: float, strict_cov: float,
                         require_ori: Tuple[int, int], require_amr: Tuple[int, int]) -> Tuple[Path, Path, Path]:
    ori_rows, amr_rows = [], []
    for res in per_sample:
        if res.ori_csv.exists() and res.ori_csv.stat().st_size > 0:
            df = pd.read_csv(res.ori_csv)
            if not df.empty:
                ori_rows.append(df)
        if res.amr_csv.exists() and res.amr_csv.stat().st_size > 0:
            df = pd.read_csv(res.amr_csv)
            if not df.empty:
                amr_rows.append(df)

    agg_ori = (
        pd.concat(ori_rows, ignore_index=True)
        if ori_rows
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
        pd.concat(amr_rows, ignore_index=True)
        if amr_rows
        else pd.DataFrame(columns=["sequence", "symbol", "name", "start", "end", "strand", "pct_identity", "pct_cov"])
    )

    agg_ori_path = outdir / "aggregate_ori_calls.csv"
    agg_amr_path = outdir / "aggregate_amr_calls.csv"
    agg_ori.to_csv(agg_ori_path, index=False)
    agg_amr.to_csv(agg_amr_path, index=False)

    # Build pass/fail with reasons (two-stage simplified)
    plasmids = sorted(set(agg_ori.get("sequence", pd.Series([], dtype=str))).union(
        set(agg_amr.get("sequence", pd.Series([], dtype=str)))
    ))

    low_min_ori, low_max_ori = require_ori
    low_min_amr, low_max_amr = require_amr

    passed_rows, failed_rows = [], []
    lines: List[str] = []

    for pid in plasmids:
        o_all = agg_ori.loc[agg_ori["sequence"] == pid].copy() if not agg_ori.empty else pd.DataFrame(columns=agg_ori.columns)
        a_all = agg_amr.loc[agg_amr["sequence"] == pid].copy() if not agg_amr.empty else pd.DataFrame(columns=agg_amr.columns)
        if not o_all.empty and {"q_start", "q_end"}.issubset(o_all.columns):
            o_all = o_all.sort_values(["q_start", "q_end"])  # stable presentation

        o_low = o_all[(o_all["pct_identity"] >= low_id) & (o_all["pct_cov_subject"] >= low_cov)] if not o_all.empty else o_all
        a_low = a_all[(a_all["pct_identity"] >= low_id) & (a_all["pct_cov"] >= low_cov)] if not a_all.empty else a_all
        n_ori_low, n_amr_low = len(o_low), len(a_low)

        reasons: List[str] = []
        if not (low_min_ori <= n_ori_low <= low_max_ori):
            reasons.append(f"ORI low-threshold count {n_ori_low} outside [{low_min_ori},{low_max_ori}]")
        if not (low_min_amr <= n_amr_low <= low_max_amr):
            reasons.append(f"ARG low-threshold count {n_amr_low} outside [{low_min_amr},{low_max_amr}]")

        # Strict: if singletons required, ensure at least one passes strict
        if not reasons:
            need_ori = (low_min_ori == low_max_ori == 1)
            need_amr = (low_min_amr == low_max_amr == 1)
            if need_ori:
                o_strict = o_all[(o_all["pct_identity"] >= strict_id) & (o_all["pct_cov_subject"] >= strict_cov)] if not o_all.empty else o_all
                if len(o_strict) < 1:
                    reasons.append(f"ORI strict threshold not met (ID≥{strict_id}, Cov≥{strict_cov})")
            if need_amr:
                a_strict = a_all[(a_all["pct_identity"] >= strict_id) & (a_all["pct_cov"] >= strict_cov)] if not a_all.empty else a_all
                if len(a_strict) < 1:
                    reasons.append(f"ARG strict threshold not met (ID≥{strict_id}, Cov≥{strict_cov})")

        if reasons:
            failed_rows.append({"Plasmid_ID": pid, "reason failed": "; ".join(reasons)})
            lines.append(f"[FAIL] {pid}: {'; '.join(reasons)}")
        else:
            # Passed; show strict-detail if available else low-detail
            detail = o_low if 'o_strict' not in locals() or o_strict.empty else o_strict
            ori_names = (detail["ori_type"].fillna("").astype(str).tolist()) if not detail.empty else []
            lines.append(f"[PASS] {pid}: ORIs={','.join(ori_names)}; ARGs={len(a_low)}")
            passed_rows.append({
                "Plasmid_ID": pid,
                "Ori's present": ",".join(ori_names),
                "Identity of each ori": ",".join([f"{x:.2f}" for x in detail.get('pct_identity', pd.Series([], dtype=float)).fillna(0).tolist()]) if not detail.empty else "",
                "Cov of each ori": ",".join([f"{x:.2f}" for x in detail.get('pct_cov_subject', pd.Series([], dtype=float)).fillna(0).tolist()]) if not detail.empty else "",
                "ARG's present": ",".join(a_low.get("symbol", pd.Series([], dtype=str)).fillna("").astype(str).tolist()) if not a_low.empty else "",
                "Identity of ARGs": ",".join([f"{x:.2f}" for x in a_low.get('pct_identity', pd.Series([], dtype=float)).fillna(0).tolist()]) if not a_low.empty else "",
                "Cov of ARGs": ",".join([f"{x:.2f}" for x in a_low.get('pct_cov', pd.Series([], dtype=float)).fillna(0).tolist()]) if not a_low.empty else "",
            })

    passed_df = pd.DataFrame(passed_rows).sort_values("Plasmid_ID")
    failed_df = pd.DataFrame(failed_rows).sort_values("Plasmid_ID")
    passed_path = outdir / "passed.csv"
    failed_path = outdir / "failed.csv"
    passed_df.to_csv(passed_path, index=False)
    failed_df.to_csv(failed_path, index=False)

    report_text = "\n".join(lines)
    report_path.write_text(report_text)
    print("\n=== QC Report ===\n" + report_text + "\n=================\n", flush=True)

    summary = pd.DataFrame([{ "n_plasmids": len(plasmids), "n_passed": len(passed_df), "n_failed": len(failed_df) }])
    summary.to_csv(outdir / "qc_summary.csv", index=False)

    return agg_ori_path, agg_amr_path, report_path


# -----------------------------
# CLI
# -----------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--input", "input_uri", required=True, help="Input FASTA file/dir or s3://bucket/prefix")
@click.option("--output", "output_uri", required=True, help="Output directory (local or s3://bucket/prefix)")
@click.option("--oridb-prefix", required=True, help="BLAST DB prefix (local path or s3://bucket/prefix)")
@click.option("--oridb-ref", default=None, help="FASTA references to build DB if missing (local or s3)")
@click.option("--min-pident", type=float, default=85.0, show_default=True)
@click.option("--min-scovs", type=float, default=80.0, show_default=True)
@click.option("--min-len", type=int, default=100, show_default=True)
@click.option("--threads", type=int, default=1, show_default=True)
@click.option("--skip-prodigal", is_flag=True, default=False)
@click.option("--low-id", type=float, default=85.0, show_default=True, help="Stage-A: identity")
@click.option("--low-cov", type=float, default=80.0, show_default=True, help="Stage-A: subject coverage")
@click.option("--strict-id", type=float, default=99.0, show_default=True, help="Stage-B: identity")
@click.option("--strict-cov", type=float, default=99.0, show_default=True, help="Stage-B: subject coverage")
@click.option("--need-ori", type=str, default="1..1", show_default=True, help="Allowed ORI count range, e.g., 1..1 or 1..2")
@click.option("--need-amr", type=str, default="1..1", show_default=True, help="Allowed ARG count range")
def main(
    input_uri: str,
    output_uri: str,
    oridb_prefix: str,
    oridb_ref: Optional[str],
    min_pident: float,
    min_scovs: float,
    min_len: int,
    threads: int,
    skip_prodigal: bool,
    low_id: float,
    low_cov: float,
    strict_id: float,
    strict_cov: float,
    need_ori: str,
    need_amr: str,
):
    """Run ORI/ARG QC on FASTA files from local disk or S3.

    Outputs per-sample CSVs, aggregate tables, a pass/fail report, and a summary.
    If --output is an s3:// URI, results are uploaded there after completion.
    """
    # Prepare working dirs
    work = Path(tempfile.mkdtemp(prefix="plasmidrl_qc_"))
    local_in = work / "input"
    local_out = work / "out"
    local_db = work / "db"
    ensure_dir(local_in)
    ensure_dir(local_out)
    ensure_dir(local_db)

    s3 = boto3.client("s3")

    # Resolve ORI DB locally (download/build if s3)
    if is_s3_uri(oridb_prefix):
        b, p = parse_s3_uri(oridb_prefix)
        # try download .nhr/.nin/.nsq
        base = local_db / "oridb"
        for ext in (".nhr", ".nin", ".nsq"):
            key = f"{p}{ext}" if not p.endswith(ext) else p
            try:
                s3.download_file(b, key, str(base.with_suffix(ext)))
            except Exception:
                pass
        local_db_prefix = base
    else:
        local_db_prefix = Path(oridb_prefix)

    local_ref: Optional[Path] = None
    if oridb_ref:
        if is_s3_uri(oridb_ref):
            b, p = parse_s3_uri(oridb_ref)
            local_ref = local_db / Path(p).name
            s3.download_file(b, p, str(local_ref))
        else:
            local_ref = Path(oridb_ref)

    db_prefix_path = ensure_blast_db(local_db_prefix, local_ref)

    # Collect input FASTAs locally
    if is_s3_uri(input_uri):
        bucket, prefix = parse_s3_uri(input_uri)
        keys = list_s3_fastas(s3, bucket, prefix)
        if not keys:
            raise SystemExit(f"No FASTA files found under s3://{bucket}/{prefix}")
        for k in keys:
            download_s3_key(s3, bucket, k, local_in)
        fastas = list_local_fastas(local_in)
    else:
        fastas = list_local_fastas(Path(input_uri))

    print(f"[INFO] Found {len(fastas)} FASTA(s).", flush=True)

    # Process
    results: List[PerSampleResult] = []
    for fa in fastas:
        try:
            res = process_one(
                fasta=fa,
                outdir=local_out,
                db_prefix=db_prefix_path,
                min_pident=min_pident,
                min_scovs=min_scovs,
                min_len=min_len,
                threads=threads,
                skip_prodigal=skip_prodigal,
            )
            results.append(res)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Tool failed on {fa.name}: {e}", flush=True)
        except Exception as e:
            print(f"[ERROR] {fa.name}: {e}", flush=True)

    # Aggregates + report
    def parse_range(spec: str) -> Tuple[int, int]:
        spec = spec.strip()
        if ".." in spec:
            a, b = spec.split("..", 1)
            return int(a), int(b)
        val = int(spec)
        return val, val

    req_ori = parse_range(need_ori)
    req_amr = parse_range(need_amr)
    report_path = local_out / "qc_report.txt"
    aggregate_and_report(local_out, results, report_path, low_id, low_cov, strict_id, strict_cov, req_ori, req_amr)

    # Upload results if requested
    if is_s3_uri(output_uri):
        b, p = parse_s3_uri(output_uri)
        upload_tree_to_s3(s3, local_out, b, p)
        print(f"[OK] Uploaded results to s3://{b}/{p}")
    else:
        out_dir = Path(output_uri)
        ensure_dir(out_dir)
        # Copy local_out → out_dir
        for root, _, files in os.walk(local_out):
            for name in files:
                src = Path(root) / name
                rel = src.relative_to(local_out)
                dst = out_dir / rel
                ensure_dir(dst.parent)
                shutil.copy2(src, dst)
        print(f"[OK] Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
