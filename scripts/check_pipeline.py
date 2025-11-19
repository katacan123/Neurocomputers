# scripts/check_pipeline.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import torch
import yaml

from models.sadu import SADUWiMANS
from wimans.dataset import build_dataloaders
from training.metrics import derive_counts_from_activity


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    dl_train, _, _ = build_dataloaders(
        splits_csv=cfg["data"]["split_csv"],
        target_T=cfg["data"]["target_T"],
        max_users=cfg["data"]["max_users"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_memory"],
    )

    batch = next(iter(dl_train))

    x = batch["x"].to(device)
    y_act = batch["y_act"]
    slot_mask = batch["slot_mask"]
    gt_counts = batch["gt_count"]

    print(f"[INFO] x shape:        {tuple(x.shape)}")
    print(f"[INFO] y_act shape:    {tuple(y_act.shape)}")
    print(f"[INFO] slot_mask shape:{tuple(slot_mask.shape)}")
    print(f"[INFO] gt_counts shape:{tuple(gt_counts.shape)}")

    model = SADUWiMANS(
        in_channels=cfg["model"]["in_channels"],
        num_classes_activity=cfg["model"]["num_classes_activity"],
        max_users=cfg["model"]["max_users"],
        backbone_cfg=cfg["model"]["backbone"],
        attention_cfg=cfg["model"]["attention"],
    ).to(device)

    with torch.no_grad():
        out = model(x)
        h = out["h"]
        z_f = out["z_f"]
        logits = out["activity_logits"]

    print(f"[INFO] h shape:        {tuple(h.shape)}")
    print(f"[INFO] z_f shape:      {tuple(z_f.shape)}")
    print(f"[INFO] logits shape:   {tuple(logits.shape)}")

    act_nothing_class = cfg["train"]["act_nothing_class"]
    count_threshold = cfg["train"]["count_threshold"]

    pred_counts, _, _ = derive_counts_from_activity(
        logits,
        act_nothing_class=act_nothing_class,
        threshold=count_threshold,
    )

    print(f"[INFO] gt_counts:      {gt_counts[:8].tolist()}")
    print(f"[INFO] pred_counts:    {pred_counts.cpu()[:8].tolist()}")
    print("[INFO] check_pipeline.py completed without shape errors.")


if __name__ == "__main__":
    main()
