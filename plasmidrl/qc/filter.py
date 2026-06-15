#!/usr/bin/env python3
"""
Two-Stage Quality Control Filtering

Original logic developed by Angus Cunningham (University College London).
Repository: https://github.com/angusgcunningham/plasmidbackbonedesign/tree/main

This script implements a hierarchical filtering strategy for generated plasmids:
1. Stage A (Detection): Identifies presence of Origins and ARGs using relaxed thresholds.
2. Stage B (Validation): Enforces strict identity and coverage thresholds (default >99%).
3. Structural Integrity: Filters sequences containing exact repeats exceeding a specified length.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

# ---------- ID normalization (robust across files/headers/paths) ----------
_EXT_RE = re.compile(r"\.(fa|fasta|fna|fas|gbk|gb|fa\.gz|fasta\.gz|fna\.gz|fas\.gz)$", re.IGNORECASE)

def _basename_no_ext(x: str) -> str:
    x = str(x).strip()
    x = Path(x).name
    if x.lower().endswith(".gz"):
        x = x[:-3]
    x = _EXT_RE.sub("", x)
    return x

def _norm_id(x: str) -> str:
    return _basename_no_ext(x).lower()

def _guess_id_col(df: pd.DataFrame, prefs=("sample","Plasmid_ID","plasmid_id","sequence","id","name")) -> str:
    """Guess an ID column in qc_summary or similar."""
    cols_lower = {c.lower(): c for c in df.columns}
    for p in prefs:
        if p.lower() in cols_lower:
            return cols_lower[p.lower()]
    # fallback: first column
    return df.columns[0]

# ---------- loaders ----------
def _load_oris(ori_csv: Path) -> pd.DataFrame:
    cols = ["sequence","ori_type","pct_identity","pct_cov_subject","q_start","q_end"]
    if not ori_csv.exists() or ori_csv.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(ori_csv)
    if "ori_type" not in df.columns and "sseqid" in df.columns:
        df = df.rename(columns={"sseqid":"ori_type"})
    if "pct_identity" not in df.columns and "pident" in df.columns:
        df = df.rename(columns={"pident":"pct_identity"})
    if "pct_cov_subject" not in df.columns and "scovs" in df.columns:
        df = df.rename(columns={"scovs":"pct_cov_subject"})
    for c in ("pct_identity","pct_cov_subject","q_start","q_end"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _load_amrs(amr_csv: Path) -> pd.DataFrame:
    cols = ["sequence","symbol","name","pct_identity","pct_cov"]
    if not amr_csv.exists() or amr_csv.stat().st_size == 0:
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(amr_csv)
    for c in ("pct_identity","pct_cov"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _load_repeats_map(repeats_csv: Path) -> dict[str, float]:
    """
    Build {normalized_id -> longest_len}. Accepts columns:
      plasmid_id (preferred) and/or file; uses whichever maps.
    """
    if repeats_csv is None or not repeats_csv.exists():
        return {}
    df = pd.read_csv(repeats_csv)
    cols_lower = {c.lower(): c for c in df.columns}
    if "longest_len" not in df.columns and "longest_len" not in cols_lower:
        raise ValueError("Repeats CSV must contain a 'longest_len' column.")
    if "longest_len" not in df.columns and "longest_len" in cols_lower:
        df = df.rename(columns={cols_lower["longest_len"]: "longest_len"})

    df["longest_len"] = pd.to_numeric(df["longest_len"], errors="coerce")

    # ID candidates
    id_col = cols_lower.get("plasmid_id") or cols_lower.get("sequence")
    df["_idnorm_from_col"]  = df[id_col].astype(str).map(_norm_id) if id_col else ""
    df["_idnorm_from_file"] = df[cols_lower["file"]].astype(str).map(_norm_id) if "file" in cols_lower else ""

    # Merge into one map (prefer explicit id column; fall back to file stem)
    rep_map: dict[str, float] = {}
    for _, r in df.iterrows():
        L = r.get("longest_len", np.nan)
        if pd.isna(L):
            continue
        for k in {str(r.get("_idnorm_from_col","")), str(r.get("_idnorm_from_file",""))} - {""}:
            if (k not in rep_map) or (L > rep_map[k]):
                rep_map[k] = float(L)
    return rep_map

# ---------- formatting ----------
def _fmt_list(vals, digits):
    out = []
    for v in vals:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append("")
        elif isinstance(v, (int, np.integer)):
            out.append(str(v))
        elif isinstance(v, (float, np.floating)):
            out.append(f"{v:.{digits}f}")
        else:
            out.append(str(v))
    return ",".join(out)

# ---------- core ----------
def two_stage_filter(
    qc_out: Path,
    out_pass_csv: Path,
    out_fail_csv: Path,
    # Stage-A (low) thresholds:
    ori_low_id: float, ori_low_cov: float,
    amr_low_id: float, amr_low_cov: float,
    # Stage-A count rules:
    ori_low_count_min: int, ori_low_count_max: int,
    amr_low_count_min: int, amr_low_count_max: int,
    # Stage-B (strict) thresholds:
    ori_strict_id: float, ori_strict_cov: float,
    amr_strict_id: float, amr_strict_cov: float,
    amr_strict_min: int | None,
    amr_strict_all: bool,
    # Repeats gate:
    repeats_csv: Path | None,
    repeat_max_len: int,
    repeat_ge: bool,   # True => fail if longest_len >= repeat_max_len, False => >
    digits: int = 2,
):
    ori_csv = qc_out / "aggregate_ori_calls.csv" if qc_out.is_dir() else qc_out
    amr_csv = (qc_out / "aggregate_amr_calls.csv") if qc_out.is_dir() else Path(str(qc_out).replace("ori","amr"))

    odf = _load_oris(ori_csv)
    adf = _load_amrs(amr_csv)
    rep_map = _load_repeats_map(repeats_csv) if repeats_csv else {}

    # Base set of plasmids: from ORI/AMR aggregates
    plasmids = set(odf.get("sequence", pd.Series([], dtype=str))).union(
               set(adf.get("sequence", pd.Series([], dtype=str))))

    # Also include any samples from qc_summary that had no ORI/ARG calls
    if qc_out.is_dir():
        summary_csv = qc_out / "qc_summary.csv"
        if summary_csv.exists() and summary_csv.stat().st_size > 0:
            sdf = pd.read_csv(summary_csv)
            id_col = _guess_id_col(sdf)
            extra_ids = set(sdf[id_col].astype(str))
            plasmids |= extra_ids

    plasmids = sorted(plasmids)

    passed_rows, failed_rows = [], []

    for pid in plasmids:
        o_all = odf.loc[odf["sequence"] == pid].copy()
        a_all = adf.loc[adf["sequence"] == pid].copy()

        if not o_all.empty and {"q_start","q_end"}.issubset(o_all.columns):
            o_all = o_all.sort_values(["q_start","q_end"])

        # ---------- Stage-A (low thresholds) ----------
        if not o_all.empty:
            o_low = o_all[(o_all["pct_identity"] >= ori_low_id) &
                          (o_all["pct_cov_subject"] >= ori_low_cov)]
        else:
            o_low = o_all

        if not a_all.empty:
            a_low = a_all[(a_all["pct_identity"] >= amr_low_id) &
                          (a_all["pct_cov"]      >= amr_low_cov)]
        else:
            a_low = a_all

        n_ori_low = len(o_low)
        n_amr_low = len(a_low)

        reasons = []

        # If absolutely no ORI and no ARG at all (even before thresholds), flag explicitly
        if o_all.empty and a_all.empty:
            reasons.append("No ORI; No ARG")
        else:
            # apply low-stage count window rules
            if not (ori_low_count_min <= n_ori_low <= ori_low_count_max):
                reasons.append(
                    f"ORI low-threshold count {n_ori_low} outside "
                    f"[{ori_low_count_min},{ori_low_count_max}]"
                )
            if not (amr_low_count_min <= n_amr_low <= amr_low_count_max):
                reasons.append(
                    f"ARG low-threshold count {n_amr_low} outside "
                    f"[{amr_low_count_min},{amr_low_count_max}]"
                )

        # ---------- Repeats gate (always evaluated) ----------
        rep_reason = ""
        if rep_map:
            rid = _norm_id(pid)
            L = rep_map.get(rid, None)
            if L is not None:
                if (repeat_ge and L >= repeat_max_len) or ((not repeat_ge) and L > repeat_max_len):
                    rep_reason = f"repeat {'≥' if repeat_ge else '>'} {repeat_max_len}"

        # If Stage-A already failed or no ORI/ARG, record + continue
        if reasons:
            if rep_reason:
                reasons.append(rep_reason)
            failed_rows.append({"Plasmid_ID": pid, "reason failed": "; ".join(reasons)})
            continue

        # ---------- Stage-B (strict thresholds) ----------
        if not o_all.empty:
            o_strict = o_all[(o_all["pct_identity"] >= ori_strict_id) &
                             (o_all["pct_cov_subject"] >= ori_strict_cov)]
        else:
            o_strict = o_all

        if not a_all.empty:
            a_strict = a_all[(a_all["pct_identity"] >= amr_strict_id) &
                             (a_all["pct_cov"]      >= amr_strict_cov)]
        else:
            a_strict = a_all

        n_ori_strict = len(o_strict)
        n_amr_strict = len(a_strict)

        # ORI strict rule: at least low-stage minimum must survive strict
        if n_ori_strict < ori_low_count_min:
            reasons.append(
                f"ORI strict count {n_ori_strict} < required {ori_low_count_min} "
                f"(ID≥{ori_strict_id}, Cov≥{ori_strict_cov})"
            )

        # ARG base strict rule: at least low-stage minimum must survive strict
        if n_amr_strict < amr_low_count_min:
            reasons.append(
                f"ARG strict count {n_amr_strict} < required {amr_low_count_min} "
                f"(ID≥{amr_strict_id}, Cov≥{amr_strict_cov})"
            )

        # ARG strict policy (optional extra constraints)
        if amr_strict_all:
            if n_amr_strict != n_amr_low:
                reasons.append(
                    f"ARG strict policy 'all' failed: {n_amr_strict}/{n_amr_low} meet strict "
                    f"(ID≥{amr_strict_id}, Cov≥{amr_strict_cov})"
                )
        elif amr_strict_min is not None:
            if n_amr_strict < amr_strict_min:
                reasons.append(
                    f"ARG strict policy 'min {amr_strict_min}' failed: only {n_amr_strict} meet strict "
                    f"(ID≥{amr_strict_id}, Cov≥{amr_strict_cov})"
                )

        # If strict fail OR repeats fail, record fail
        if reasons or rep_reason:
            if rep_reason and rep_reason not in reasons:
                reasons.append(rep_reason)
            failed_rows.append({"Plasmid_ID": pid, "reason failed": "; ".join(reasons)})
            continue

        # ---------- Passed (use strict-filtered details) ----------
        ori_names = o_strict["ori_type"].fillna("").astype(str).tolist() if not o_strict.empty else []
        ori_ids   = o_strict["pct_identity"].tolist() if not o_strict.empty else []
        ori_covs  = o_strict["pct_cov_subject"].tolist() if not o_strict.empty else []

        if not a_strict.empty:
            labels = a_strict["symbol"] if "symbol" in a_strict.columns else a_strict.get("name", pd.Series([], dtype=str))
            labels = labels.fillna("").astype(str).tolist()
            amr_ids = a_strict["pct_identity"].tolist()
            amr_cov = a_strict["pct_cov"].tolist()
        else:
            labels, amr_ids, amr_cov = [], [], []

        passed_rows.append({
            "Plasmid_ID": pid,
            "Ori's present": _fmt_list(ori_names, digits),
            "Identity of each ori": _fmt_list(ori_ids, digits),
            "Cov of each ori": _fmt_list(ori_covs, digits),
            "ARG's present": _fmt_list(labels, digits),
            "Identity of ARGs": _fmt_list(amr_ids, digits),
            "Cov of ARGs": _fmt_list(amr_cov, digits),
        })

    df_pass = pd.DataFrame(passed_rows)
    if not df_pass.empty:
        df_pass = df_pass.sort_values("Plasmid_ID")
    else:
        # Create empty df with expected columns
        df_pass = pd.DataFrame(columns=["Plasmid_ID", "Ori's present", "Identity of each ori", "Cov of each ori", "ARG's present", "Identity of ARGs", "Cov of ARGs"])
    df_pass.to_csv(out_pass_csv, index=False)

    df_fail = pd.DataFrame(failed_rows)
    if not df_fail.empty:
        df_fail = df_fail.sort_values("Plasmid_ID")
    else:
        df_fail = pd.DataFrame(columns=["Plasmid_ID", "reason failed"])
    df_fail.to_csv(out_fail_csv, index=False)

# ---------- CLI ----------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Two-stage QC with optional repeat gate: (A) count with low thresholds, "
                    "(B) validate with strict thresholds, (C) fail on long repeats."
    )
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--qc_out", required=False,
                    help="Folder with aggregate_ori_calls.csv & aggregate_amr_calls.csv, or a single ori CSV path")
    ap.add_argument("--out_pass", required=False)
    ap.add_argument("--out_fail", required=False)

    # Stage-A (low) thresholds for counting
    ap.add_argument("--ori_low_identity", type=float, default=85.0)
    ap.add_argument("--ori_low_cov",      type=float, default=80.0)
    ap.add_argument("--amr_low_identity", type=float, default=85.0)
    ap.add_argument("--amr_low_cov",      type=float, default=80.0)

    # Stage-A count windows (use min=max for exact)
    ap.add_argument("--ori_low_count_min", type=int, default=1)
    ap.add_argument("--ori_low_count_max", type=int, default=1)
    ap.add_argument("--amr_low_count_min", type=int, default=1)
    ap.add_argument("--amr_low_count_max", type=int, default=1)

    # ARG strict policies (on top of base rule)
    ap.add_argument("--amr_strict_min", type=int, default=None,
                    help="Require at least this many ARGs to meet strict thresholds (unset = no extra minimum)")
    ap.add_argument("--amr_strict_all", action="store_true",
                    help="Require ALL low-threshold ARGs to meet strict thresholds")

    # Stage-B (strict) thresholds to validate hits
    ap.add_argument("--ori_strict_identity", type=float, default=99.0)
    ap.add_argument("--ori_strict_cov",      type=float, default=99.0)
    ap.add_argument("--amr_strict_identity", type=float, default=100.0)
    ap.add_argument("--amr_strict_cov",      type=float, default=100.0)

    # Repeats gate (optional)
    ap.add_argument("--repeats_csv", type=str, default=None,
                    help="CSV with columns including longest_len and plasmid_id/file")
    ap.add_argument("--repeat_max_len", type=int, default=50,
                    help="Fail if longest repeat meets/exceeds this")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--repeat_ge", action="store_true",
                    help="Fail if longest_len ≥ repeat_max_len (default)")
    grp.add_argument("--repeat_gt", action="store_true",
                    help="Fail if longest_len > repeat_max_len")

    ap.add_argument("--digits", type=int, default=2)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.qc_out:
        qc_out = Path(args.qc_out)
    elif args.run_name:
        qc_out = Path("runs") / args.run_name / "qc"
    else:
        raise SystemExit("Provide --qc_out or --run-name.")

    if args.out_pass:
        out_pass = Path(args.out_pass)
    elif args.run_name:
        out_pass = Path("runs") / args.run_name / "qc" / "passed.csv"
    else:
        raise SystemExit("Provide --out_pass or --run-name.")

    if args.out_fail:
        out_fail = Path(args.out_fail)
    elif args.run_name:
        out_fail = Path("runs") / args.run_name / "qc" / "failed.csv"
    else:
        raise SystemExit("Provide --out_fail or --run-name.")

    repeat_ge = True if (args.repeat_ge or not args.repeat_gt) else False  # default to ≥
    two_stage_filter(
        qc_out=qc_out,
        out_pass_csv=out_pass,
        out_fail_csv=out_fail,
        ori_low_id=args.ori_low_identity,
        ori_low_cov=args.ori_low_cov,
        amr_low_id=args.amr_low_identity,
        amr_low_cov=args.amr_low_cov,
        ori_low_count_min=args.ori_low_count_min,
        ori_low_count_max=args.ori_low_count_max,
        amr_low_count_min=args.amr_low_count_min,
        amr_low_count_max=args.amr_low_count_max,
        ori_strict_id=args.ori_strict_identity,
        ori_strict_cov=args.ori_strict_cov,
        amr_strict_id=args.amr_strict_identity,
        amr_strict_cov=args.amr_strict_cov,
        amr_strict_min=args.amr_strict_min,
        amr_strict_all=args.amr_strict_all,
        repeats_csv=Path(args.repeats_csv) if args.repeats_csv else None,
        repeat_max_len=args.repeat_max_len,
        repeat_ge=repeat_ge,
        digits=args.digits,
    )
    passed_ids_path = out_pass.parent / "passed_ids.txt"
    if out_pass.exists():
        df_pass = pd.read_csv(out_pass)
        if "Plasmid_ID" in df_pass.columns:
            with open(passed_ids_path, "w") as fh:
                for pid in df_pass["Plasmid_ID"].dropna().astype(str):
                    fh.write(f"{pid}\n")
