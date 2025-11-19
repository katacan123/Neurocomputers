# scripts/preprocess_wimans.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import pandas as pd
import torch

from wimans.transforms import preprocess_csi_sample
from wimans.labels import build_sample_labels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default="training_dataset",
                   help="Root folder containing annotation.csv and wifi_csi/mat/")
    p.add_argument("--band", type=float, choices=[2.4, 5.0], required=True,
                   help="WiFi band to preprocess (2.4 or 5.0 GHz)")
    p.add_argument("--target_T", type=int, default=3000,
                   help="Target time length for CSI sequences")
    p.add_argument("--max_users", type=int, default=5,
                   help="Maximum number of users per sample (slots)")
    p.add_argument("--use_wavelet_pp", action="store_true",
                   help="Use wavelet denoising as in SADU paper")
    p.add_argument("--out_dir", type=str, default="training_dataset/processed",
                   help="Output directory for tensors + metadata.csv")
    p.add_argument("--csi_key", type=str, default="csi",
                   help="Key name of CSI inside .mat file")
    return p.parse_args()


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotation_csv = dataset_root / "annotation.csv"
    csi_mat_root = dataset_root / "wifi_csi" / "mat"

    if not annotation_csv.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_csv}")
    if not csi_mat_root.exists():
        raise FileNotFoundError(f"CSI .mat folder not found: {csi_mat_root}")

    df_raw = pd.read_csv(annotation_csv)

    SAMPLE_ID_COL = "label"
    ENV_COL = "environment"
    BAND_COL = "wifi_band"

    for col in (SAMPLE_ID_COL, ENV_COL, BAND_COL):
        if col not in df_raw.columns:
            raise KeyError(f"'{col}' column not found in {annotation_csv}")

    df_band = df_raw[df_raw[BAND_COL] == args.band].reset_index(drop=True)
    print(f"[INFO] Total samples in band {args.band} GHz: {len(df_band)}")

    rows = []

    for idx, row in df_band.iterrows():
        sample_id = row[SAMPLE_ID_COL]
        env = row[ENV_COL]
        band = row[BAND_COL]

        csi_mat_path = csi_mat_root / f"{sample_id}.mat"
        if not csi_mat_path.exists():
            print(f"[WARN] CSI mat file not found for {sample_id}: {csi_mat_path}")
            continue

        x = preprocess_csi_sample(
            csi_mat_path=csi_mat_path,
            target_T=args.target_T,
            use_wavelet_pp=args.use_wavelet_pp,
            csi_key=args.csi_key,
        )

        tensor_path = out_dir / f"{sample_id}.pt"
        torch.save(x, tensor_path)

        label_info = build_sample_labels(
            raw_row=row,
            max_users=args.max_users
        )

        rows.append({
            "sample_id": sample_id,
            "tensor_path": str(tensor_path),
            "environment": env,
            "wifi_band": band,
            **label_info,
        })

        if (idx + 1) % 100 == 0:
            print(f"[INFO] Processed {idx + 1}/{len(df_band)} samples...")

    meta_df = pd.DataFrame(rows)
    meta_csv_path = out_dir / "metadata.csv"
    meta_df.to_csv(meta_csv_path, index=False)
    print(f"[INFO] Saved metadata to {meta_csv_path}")
    print(f"[INFO] Total processed samples: {len(meta_df)}")


if __name__ == "__main__":
    main()
