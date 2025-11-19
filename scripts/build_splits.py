# scripts/build_splits.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_csv", type=str, required=True,
                   help="Path to metadata.csv produced by preprocess_wimans.py")
    p.add_argument("--out_csv", type=str, required=True,
                   help="Output path for splits.csv")
    p.add_argument("--train_envs", nargs="+", required=True,
                   help="List of environments used for training")
    p.add_argument("--val_envs", nargs="+", required=True,
                   help="List of environments used for validation")
    p.add_argument("--test_envs", nargs="+", required=True,
                   help="List of environments used for testing")
    p.add_argument("--val_ratio", type=float, default=0.2,
                   help="Validation ratio inside train_envs")
    return p.parse_args()


def main():
    args = parse_args()

    meta_path = Path(args.metadata_csv)
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.csv not found: {meta_path}")

    df = pd.read_csv(meta_path)

    if "sample_id" not in df.columns:
        raise KeyError("'sample_id' column not found in metadata.csv")
    if "environment" not in df.columns:
        raise KeyError("'environment' column not found in metadata.csv")

    df["split"] = "ignore"

    train_envs = set(args.train_envs)
    val_envs = set(args.val_envs)
    test_envs = set(args.test_envs)

    env_col = "environment"

    train_mask = df[env_col].isin(train_envs)
    val_mask   = df[env_col].isin(val_envs)
    test_mask  = df[env_col].isin(test_envs)

    df.loc[test_mask, "split"] = "test"

    candidate = df[train_mask & ~test_mask].copy()
    if candidate.empty:
        raise RuntimeError("No candidate samples for train/val after filtering.")

    train_ids, val_ids = train_test_split(
        candidate["sample_id"].values,
        test_size=args.val_ratio,
        random_state=42,
        shuffle=True,
    )

    df.loc[df["sample_id"].isin(train_ids), "split"] = "train"
    df.loc[df["sample_id"].isin(val_ids),   "split"] = "val"

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[INFO] Saved splits to {out_path}")
    print(df["split"].value_counts())


if __name__ == "__main__":
    main()
