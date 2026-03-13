#!/usr/bin/env python3
"""Load QA JSON using the `datasets` library and show some examples.

Usage:
  python app/load_qa.py --file app/qa.json
"""
from __future__ import annotations

import argparse
import sys
from typing import Any


def load_and_inspect(path: str) -> None:
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover - helpful error message
        print("The `datasets` library is required. Install with: pip install datasets")
        raise

    ds = load_dataset("json", data_files=path, split="train")
    print(f"Loaded dataset from {path!s}: {len(ds)} examples")
    print("Columns:", ds.column_names)

    if len(ds) > 0:
        print("\nFirst record:")
        first = ds[0]
        for k, v in first.items():
            print(f"- {k}: {v}")

    # Show only question and answer
    keep = [c for c in ds.column_names if c in ("question", "answer")]
    if keep:
        subset = ds.remove_columns([c for c in ds.column_names if c not in keep])
        print("\nFirst QA pair:")
        print(subset[0])
        return subset

    # Try converting to pandas (optional)
    try:
        df = ds.to_pandas()
        print(f"\nConverted to pandas DataFrame with shape: {df.shape}")
    except Exception as e:
        print("Could not convert to pandas (pandas may not be installed):", e)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load QA JSON with datasets")
    parser.add_argument(
        "--file", "-f", default="qa.json", help="Path to the JSON file or glob"
    )
    args = parser.parse_args(argv)

    try:
        load_and_inspect(args.file)
    except Exception as exc:
        print("Error:", exc)
        return 2
    return 0


# if __name__ == "__main__":
#     raise SystemExit(main())
