"""Read-only loader for the public artefact bucket.

All experiment outputs, generated samples, QC tables, MFE results, and
likelihood scores referenced by the notebooks live at:

    https://huggingface.co/datasets/UCL-CSSB/PlasmidRL-ICML

Files are downloaded on first access and cached under
``~/.cache/plasmidrl/`` (override with ``PLASMIDRL_CACHE_DIR``). The bucket
is treated as read-only — there is no write or delete API in this module.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi

BUCKET = "UCL-CSSB/PlasmidRL-ICML"
CACHE_DIR = Path(os.environ.get("PLASMIDRL_CACHE_DIR",
                                Path.home() / ".cache" / "plasmidrl"))


def load(remote_path: str, *, force_refresh: bool = False) -> Path:
    """Download ``remote_path`` from the public bucket and return its local path.

    Cached at ``CACHE_DIR / remote_path``. Set ``force_refresh=True`` to
    re-download.
    """
    local = CACHE_DIR / remote_path
    if local.exists() and not force_refresh:
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.download_bucket_files(BUCKET, files=[(remote_path, str(local))])
    return local


def load_csv(remote_path: str, **read_csv_kwargs) -> pd.DataFrame:
    """Download a CSV from the bucket and return a DataFrame."""
    return pd.read_csv(load(remote_path), **read_csv_kwargs)


def load_json(remote_path: str) -> Any:
    """Download a JSON file from the bucket and return its parsed contents."""
    return json.loads(load(remote_path).read_text())


def load_fasta(remote_path: str) -> dict[str, str]:
    """Download a FASTA from the bucket and return ``{accession: sequence}``."""
    text = load(remote_path).read_text()
    out: dict[str, str] = {}
    cur_id = None
    cur_seq: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur_id is not None:
                out[cur_id] = "".join(cur_seq)
            cur_id = line[1:].split()[0].split("|")[0]
            cur_seq = []
        else:
            cur_seq.append(line.strip())
    if cur_id is not None:
        out[cur_id] = "".join(cur_seq)
    return out


def list_tree(prefix: str = "") -> list[str]:
    """List all paths under ``prefix`` in the bucket (read-only)."""
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    return sorted(item.path for item in api.list_bucket_tree(
        BUCKET, prefix=prefix, recursive=True))


__all__ = ["BUCKET", "CACHE_DIR", "load", "load_csv", "load_json",
           "load_fasta", "list_tree"]
