import argparse
import sys
from pathlib import Path


def read_prompts(parquet_path: Path, prompt_key: str):
    try:
        import pyarrow.parquet as pq
    except Exception as e:
        print("ERROR: pyarrow is required to read parquet (pip install pyarrow)")
        raise
    table = pq.read_table(str(parquet_path))
    if prompt_key not in table.column_names:
        raise KeyError(f"Column '{prompt_key}' not found in {parquet_path} (available: {table.column_names})")
    col = table[prompt_key]
    return [str(x.as_py()) if x is not None else "" for x in col]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="Parquet file paths")
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    for p in args.paths:
        path = Path(p)
        prompts = read_prompts(path, args.prompt_key)
        lengths = [len(s.strip()) for s in prompts]
        total = len(lengths)
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        num_too_short = sum(1 for L in lengths if L < args.k)
        pct_short = (100.0 * num_too_short / total) if total else 0.0
        print(f"File: {path}")
        print(f"  total={total} min_len={min_len} max_len={max_len} <{args.k}={num_too_short} ({pct_short:.2f}%)")
        if num_too_short:
            print("  examples (first 5 too-short prompts):")
            shown = 0
            for s in prompts:
                if len(s.strip()) < args.k:
                    print(f"    len={len(s.strip())!s} text={s!r}")
                    shown += 1
                    if shown >= 5:
                        break


if __name__ == "__main__":
    main()


