"""
Print columns, dtypes, row count and a few sample rows for a raw data file.
Run this BEFORE build_dataset.py on any new download — the loaders assume
specific column names, and guessing wrong fails silently, not loudly.

Usage:
    python scripts/inspect_raw.py data/counsel-chat.parquet
"""

import sys

import pandas as pd


def main(path: str):
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif path.endswith(".tsv"):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    print(f"file: {path}")
    print(f"rows: {len(df)}")
    print(f"columns: {df.columns.tolist()}")
    print("\ndtypes:")
    print(df.dtypes)
    print("\nfirst 3 rows:")
    with pd.option_context("display.max_colwidth", 80):
        print(df.head(3))

    # If there's an integer-looking label column, show its unique values —
    # this is what you need to fill in the int_to_name mapping correctly.
    for col in df.columns:
        if "label" in col.lower() or "emotion" in col.lower():
            print(f"\nunique values in '{col}': {sorted(df[col].unique().tolist())[:20]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/inspect_raw.py <path-to-file>")
    main(sys.argv[1])