# scripts/run_cross_env_experiments.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_config_sadu", type=str, required=True,
                   help="Base YAML config for SADU")
    p.add_argument("--base_config_baseline", type=str, required=True,
                   help="Base YAML config for baseline")
    p.add_argument("--metadata_csv", type=str, required=True,
                   help="Path to metadata.csv from preprocess_wimans.py")
    p.add_argument("--processed_dir", type=str, required=True,
                   help="Directory where splits_*.csv and configs will be created")
    p.add_argument("--envs", nargs="+", required=True,
                   help="List of all environments, e.g. classroom empty_room meeting_room")
    return p.parse_args()


def build_split_for_env(
    metadata_csv: str,
    out_split_csv: str,
    train_envs,
    val_envs,
    test_envs,
    val_ratio: float = 0.2,
):
    df = pd.read_csv(metadata_csv)

    if "sample_id" not in df.columns:
        raise KeyError("'sample_id' column not found in metadata.csv")
    if "environment" not in df.columns:
        raise KeyError("'environment' column not found in metadata.csv")

    df["split"] = "ignore"

    train_envs = set(train_envs)
    val_envs = set(val_envs)
    test_envs = set(test_envs)

    env_col = "environment"

    train_mask = df[env_col].isin(train_envs)
    val_mask = df[env_col].isin(val_envs)
    test_mask = df[env_col].isin(test_envs)

    df.loc[test_mask, "split"] = "test"

    candidate = df[train_mask & ~test_mask].copy()
    if candidate.empty:
        raise RuntimeError(
            f"No candidate samples for train/val in train_envs={train_envs} excluding test_envs={test_envs}"
        )

    train_ids, val_ids = train_test_split(
        candidate["sample_id"].values,
        test_size=val_ratio,
        random_state=42,
        shuffle=True,
    )

    df.loc[df["sample_id"].isin(train_ids), "split"] = "train"
    df.loc[df["sample_id"].isin(val_ids), "split"] = "val"

    out_path = Path(out_split_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"[INFO] Saved cross-env split to {out_path}")
    print(df["split"].value_counts())


def run_training_script(script_path: str, config_path: str):
    cmd = ["python", script_path, "--config", config_path]
    print(f"[INFO] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    base_config_sadu = Path(args.base_config_sadu)
    base_config_baseline = Path(args.base_config_baseline)
    metadata_csv = Path(args.metadata_csv)
    processed_dir = Path(args.processed_dir)

    with open(base_config_sadu, "r") as f:
        cfg_sadu_base = yaml.safe_load(f)
    with open(base_config_baseline, "r") as f:
        cfg_base_base = yaml.safe_load(f)

    all_envs = args.envs

    experiments = []

    # 1) Train and test in each env individually
    for env in all_envs:
        experiments.append((
            f"{env}_only",
            [env],
            [env],
            [env],
        ))

    # 2) Cross-env: train on all except last, test on last
    if len(all_envs) >= 2:
        last_env = all_envs[-1]
        train_envs = all_envs[:-1]
        experiments.append((
            f"train_{'_'.join(train_envs)}_test_{last_env}",
            train_envs,
            train_envs,
            [last_env],
        ))

    for exp_name, train_envs, val_envs, test_envs in experiments:
        print(f"\n=========================================")
        print(f"[EXPERIMENT] {exp_name}")
        print(f"  Train envs: {train_envs}")
        print(f"  Val envs:   {val_envs}")
        print(f"  Test envs:  {test_envs}")
        print(f"=========================================\n")

        split_csv = processed_dir / f"splits_{exp_name}.csv"
        build_split_for_env(
            metadata_csv=str(metadata_csv),
            out_split_csv=str(split_csv),
            train_envs=train_envs,
            val_envs=val_envs,
            test_envs=test_envs,
            val_ratio=0.2,
        )

        cfg_sadu = cfg_sadu_base.copy()
        cfg_base = cfg_base_base.copy()

        cfg_sadu["data"]["split_csv"] = str(split_csv)
        cfg_base["data"]["split_csv"] = str(split_csv)

        for cfg, tag in [(cfg_sadu, "sadu"), (cfg_base, "baseline")]:
            ckpt_dir = cfg["train"]["ckpt_dir"]
            new_ckpt_dir = os.path.join(ckpt_dir, exp_name)
            cfg["train"]["ckpt_dir"] = new_ckpt_dir

            log_dir = cfg["train"].get("log_dir", os.path.join(ckpt_dir, "logs"))
            new_log_dir = os.path.join(log_dir, exp_name)
            cfg["train"]["log_dir"] = new_log_dir

        cfg_sadu_path = processed_dir / f"config_sadu_{exp_name}.yaml"
        cfg_base_path = processed_dir / f"config_baseline_{exp_name}.yaml"

        with open(cfg_sadu_path, "w") as f:
            yaml.safe_dump(cfg_sadu, f)
        with open(cfg_base_path, "w") as f:
            yaml.safe_dump(cfg_base, f)

        run_training_script("scripts/train_sadu.py", str(cfg_sadu_path))
        run_training_script("scripts/train_baseline_cnn.py", str(cfg_base_path))


if __name__ == "__main__":
    main()
