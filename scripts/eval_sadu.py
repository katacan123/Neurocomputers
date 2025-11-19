# scripts/eval_sadu.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os

import torch
import yaml

from models.sadu_full import SADUWiMANSFull as SADUWiMANS

from wimans.dataset import build_dataloaders
from training.loop import eval_one_epoch
from training.metrics import (
    derive_counts_from_activity,
    count_metrics,
    activity_accuracy,
    accumulate_env_metrics,
    finalize_env_metrics,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="Path to YAML config file")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Checkpoint path (default: best.pt from cfg)")
    p.add_argument("--per_env", action="store_true",
                   help="Compute per-environment metrics on test set")
    return p.parse_args()


def load_model_from_cfg(cfg, device, ckpt_path):
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

    train_loader, val_loader, test_loader = build_dataloaders(
        splits_csv=cfg["data"]["split_csv"],
        target_T=cfg["data"]["target_T"],
        max_users=cfg["data"]["max_users"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_memory"],
    )

    ckpt_dir = cfg["train"]["ckpt_dir"]
    default_ckpt = os.path.join(ckpt_dir, "best.pt")
    ckpt_path = args.ckpt if args.ckpt is not None else default_ckpt
    print(f"[INFO] Loading checkpoint from: {ckpt_path}")

    model = load_model_from_cfg(cfg, device, ckpt_path)

    act_nothing_class = cfg["train"]["act_nothing_class"]
    count_threshold = cfg["train"]["count_threshold"]

    print("\n[INFO] Evaluating on validation set...")
    val_stats = eval_one_epoch(
        model=model,
        dataloader=val_loader,
        device=device,
        act_nothing_class=act_nothing_class,
        count_threshold=count_threshold,
    )
    print(
        f"  Val:  loss={val_stats['loss']:.4f}, "
        f"act_acc={val_stats['act_acc']:.4f}, "
        f"count_acc={val_stats['count_acc']:.4f}, "
        f"count_mae={val_stats['count_mae']:.4f}"
    )

    print("\n[INFO] Evaluating on test set...")
    test_stats = eval_one_epoch(
        model=model,
        dataloader=test_loader,
        device=device,
        act_nothing_class=act_nothing_class,
        count_threshold=count_threshold,
    )
    print(
        f"  Test: loss={test_stats['loss']:.4f}, "
        f"act_acc={test_stats['act_acc']:.4f}, "
        f"count_acc={test_stats['count_acc']:.4f}, "
        f"count_mae={test_stats['count_mae']:.4f}"
    )

    if args.per_env:
        print("\n[INFO] Computing per-environment metrics on test set...")
        env_to_stats = {}

        with torch.no_grad():
            for batch in test_loader:
                x = batch["x"].to(device)
                y_act = batch["y_act"].to(device)
                slot_mask = batch["slot_mask"].to(device)
                gt_counts = batch["gt_count"].to(device)
                envs = batch["environment"]

                out = model(x)
                logits = out["activity_logits"]

                pred_counts, _, _ = derive_counts_from_activity(
                    logits,
                    act_nothing_class=act_nothing_class,
                    threshold=count_threshold,
                )

                # use standard activity_accuracy and count_metrics
                act_acc = activity_accuracy(logits, y_act, slot_mask)
                c_acc, c_mae = count_metrics(pred_counts, gt_counts)
                loss_val = 0.0  # you can compute a full loss per env if you want

                for env in envs:
                    accumulate_env_metrics(
                        env_to_stats,
                        env,
                        loss=loss_val,
                        act_acc=act_acc,
                        count_acc=c_acc,
                        count_mae=c_mae,
                        weight=1,
                    )

        env_results = finalize_env_metrics(env_to_stats)

        print("\n[Per-environment test metrics]")
        for env, stats in env_results.items():
            print(
                f"  Env={env}: "
                f"loss={stats['loss']:.4f}, "
                f"act_acc={stats['act_acc']:.4f}, "
                f"count_acc={stats['count_acc']:.4f}, "
                f"count_mae={stats['count_mae']:.4f}"
            )


if __name__ == "__main__":
    main()
