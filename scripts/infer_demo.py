# scripts/infer_demo.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
import os

import torch
import yaml
import pandas as pd

from models.sadu import SADUWiMANS
from wimans.labels import idx_to_activity
from training.metrics import derive_counts_from_activity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint path (default: best.pt from cfg)")
    p.add_argument("--sample_id", type=str, required=True,
                   help="Sample ID to run inference on")
    return p.parse_args()


def load_model(cfg, device, ckpt_path):
    model = SADUWiMANS(
        in_channels=cfg["model"]["in_channels"],
        num_classes_activity=cfg["model"]["num_classes_activity"],
        max_users=cfg["model"]["max_users"],
        backbone_cfg=cfg["model"]["backbone"],
        attention_cfg=cfg["model"]["attention"],
    ).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    splits_csv = cfg["data"]["split_csv"]
    df = pd.read_csv(splits_csv)

    if "sample_id" not in df.columns:
        raise KeyError(f"'sample_id' column not found in {splits_csv}")

    row = df[df["sample_id"] == args.sample_id]
    if row.empty:
        raise ValueError(f"Sample ID '{args.sample_id}' not found in {splits_csv}")

    row = row.iloc[0]
    tensor_path = Path(row["tensor_path"])
    env = row["environment"]
    band = row["wifi_band"]

    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {tensor_path}")

    print(f"[INFO] Sample '{args.sample_id}' | env={env} | band={band}")
    print(f"[INFO] Loading tensor from: {tensor_path}")

    x = torch.load(tensor_path)  # (C, T)
    target_T = cfg["data"]["target_T"]
    if x.shape[1] > target_T:
        start = (x.shape[1] - target_T) // 2
        x = x[:, start:start + target_T]
    elif x.shape[1] < target_T:
        pad_len = target_T - x.shape[1]
        pad = x[:, -1:].repeat(1, pad_len)
        x = torch.cat([x, pad], dim=1)

    x = x.unsqueeze(0).to(device)

    ckpt_dir = cfg["train"]["ckpt_dir"]
    default_ckpt = os.path.join(ckpt_dir, "best.pt")
    ckpt_path = args.ckpt if args.ckpt is not None else default_ckpt
    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    model = load_model(cfg, device, ckpt_path)

    act_nothing_class = cfg["train"]["act_nothing_class"]
    count_threshold = cfg["train"]["count_threshold"]

    with torch.no_grad():
        out = model(x)
        logits = out["activity_logits"]
        pred_counts, pred_classes, pred_confs = derive_counts_from_activity(
            logits,
            act_nothing_class=act_nothing_class,
            threshold=count_threshold,
        )

    max_users = cfg["model"]["max_users"]
    pred_counts = pred_counts.cpu()[0].item()
    pred_classes = pred_classes.cpu()[0]
    pred_confs = pred_confs.cpu()[0]

    print("\n[Inference results]")
    print(f"  Predicted number of people in the room: {pred_counts}")
    print("\n  Per-user slots:")
    for slot in range(max_users):
        cls_idx = int(pred_classes[slot].item())
        conf = float(pred_confs[slot].item())
        act_name = idx_to_activity.get(cls_idx, f"cls_{cls_idx}")

        is_active = (cls_idx != act_nothing_class) and (conf > count_threshold)
        status = "ACTIVE" if is_active else "inactive"

        print(
            f"    Slot {slot+1}: "
            f"pred_activity={act_name} (class={cls_idx}), "
            f"conf={conf:.3f}, status={status}"
        )


if __name__ == "__main__":
    main()
