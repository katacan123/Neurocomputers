# scripts/train_sadu.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import os

import torch
import yaml
from torch.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from models.sadu_full import SADUWiMANSFull as SADUWiMANS

from wimans.dataset import build_dataloaders
from training.loop import train_one_epoch, eval_one_epoch


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
    print(f"[INFO] Using device: {device}")

    dl_train, dl_val, dl_test = build_dataloaders(
        splits_csv=cfg["data"]["split_csv"],
        target_T=cfg["data"]["target_T"],
        max_users=cfg["data"]["max_users"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
        pin_memory=cfg["train"]["pin_memory"],
    )

    model = SADUWiMANS(
        in_channels=cfg["model"]["in_channels"],
        num_classes_activity=cfg["model"]["num_classes_activity"],
        max_users=cfg["model"]["max_users"],
        backbone_cfg=cfg["model"]["backbone"],
        attention_cfg=cfg["model"]["attention"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    use_amp = cfg["train"].get("amp", True) and (device == "cuda")
    scaler = GradScaler() if use_amp else None

    ckpt_dir = cfg["train"]["ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "best.pt")

    log_dir = cfg["train"].get("log_dir", os.path.join(ckpt_dir, "logs"))
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    act_nothing_class = cfg["train"]["act_nothing_class"]
    count_threshold = cfg["train"]["count_threshold"]

    best_val_metric = -1.0
    epochs_no_improve = 0
    patience = cfg["train"]["patience"]
    num_epochs = cfg["train"]["epochs"]

    loss_weights = cfg["train"].get("loss_weights", None)
    ignore_index_id = cfg["train"].get("ignore_index_id", -1)
    ignore_index_loc = cfg["train"].get("ignore_index_loc", -1)

    for epoch in range(1, num_epochs + 1):
        print(f"\n[Epoch {epoch}/{num_epochs}] (SADU)")

        train_stats = train_one_epoch(
            model=model,
            dataloader=dl_train,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            act_nothing_class=act_nothing_class,
            count_threshold=count_threshold,
            loss_weights=loss_weights,
            ignore_index_id=ignore_index_id,
            ignore_index_loc=ignore_index_loc,
        )

        print(
            f"  Train: "
            f"loss={train_stats['loss_total']:.4f}, "
            f"act_acc={train_stats['act_acc']:.3f}, "
            f"loc_acc={train_stats['loc_acc']:.3f}, "
            f"id_acc={train_stats['id_acc']:.3f}, "
            f"count_acc={train_stats['count_acc']:.3f}, "
            f"MAE={train_stats['count_mae']:.3f}"
        )

        writer.add_scalar("train/loss", train_stats["loss"], epoch)
        writer.add_scalar("train/act_acc", train_stats["act_acc"], epoch)
        writer.add_scalar("train/count_acc", train_stats["count_acc"], epoch)
        writer.add_scalar("train/count_mae", train_stats["count_mae"], epoch)

        val_stats = eval_one_epoch(
            model=model,
            dataloader=dl_val,
            device=device,
            act_nothing_class=act_nothing_class,
            count_threshold=count_threshold,
            loss_weights=loss_weights,
            ignore_index_id=ignore_index_id,
            ignore_index_loc=ignore_index_loc,
        )

        print(
            f"  Val:   "
            f"loss={val_stats['loss_total']:.4f}, "
            f"act_acc={val_stats['act_acc']:.3f}, "
            f"loc_acc={val_stats['loc_acc']:.3f}, "
            f"id_acc={val_stats['id_acc']:.3f}, "
            f"count_acc={val_stats['count_acc']:.3f}, "
            f"MAE={val_stats['count_mae']:.3f}"
        )

        writer.add_scalar("val/loss", val_stats["loss"], epoch)
        writer.add_scalar("val/act_acc", val_stats["act_acc"], epoch)
        writer.add_scalar("val/count_acc", val_stats["count_acc"], epoch)
        writer.add_scalar("val/count_mae", val_stats["count_mae"], epoch)

        val_metric = 0.5 * (val_stats["act_acc"] + val_stats["count_acc"])

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  [INFO] New best model saved to {best_ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"  [INFO] No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print("[INFO] Early stopping triggered.")
            break

    writer.close()

    if os.path.exists(best_ckpt_path):
        print("\n[INFO] Evaluating best checkpoint on test set...")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        test_stats = eval_one_epoch(
            model=model,
            dataloader=dl_test,
            device=device,
            act_nothing_class=act_nothing_class,
            count_threshold=count_threshold,
        )
        print(
            f"  Test:  loss={test_stats['loss']:.4f}, "
            f"act_acc={test_stats['act_acc']:.4f}, "
            f"count_acc={test_stats['count_acc']:.4f}, "
            f"count_mae={test_stats['count_mae']:.4f}"
        )


if __name__ == "__main__":
    main()
