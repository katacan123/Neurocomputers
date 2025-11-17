# eval/sweep_thresholds.py

import argparse
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

from data.splits import split_by_env_with_val
from data.wimans_dataset import WiMansActivityDataset, ACTIVITIES
from data.preprocess_csi import compute_static_average
from models.wimuar_hstnn import WiMUAR_HSTNN
from utils.metrics import (
    compute_activity_metrics,
    compute_count_metrics,
    logits_to_activity_matrix,
    derive_count_from_pred,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep activity/count thresholds on validation set"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g., configs/base_wimans.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (e.g., runs/.../best_model.pt)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size for eval (default: use train.batch_size from config)",
    )
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)

    device = cfg["train"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --------------------------------------------------------
    # Paths (mirrors what train_wimuar.py does)
    # --------------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "training_dataset"

    annotation_csv = dataset_root / "annotation.csv"
    csi_amp_root = dataset_root / "wifi_csi" / "amp"

    print(f"[INFO] Annotation CSV: {annotation_csv}")
    print(f"[INFO] CSI amp root:   {csi_amp_root}")

    # --------------------------------------------------------
    # Env split (same as training)
    # --------------------------------------------------------
    train_envs = tuple(cfg["data"]["env_train"])
    test_env = cfg["data"]["env_test"]
    val_ratio = cfg["data"].get("val_ratio", 0.2)

    print(f"[INFO] Train envs: {train_envs}, Test env: {test_env}, val_ratio={val_ratio}")

    split_ids = split_by_env_with_val(
        annotation_csv=str(annotation_csv),
        train_envs=train_envs,
        test_env=test_env,
        val_ratio=val_ratio,
        random_state=42,
    )
    print(
        f"[INFO] Split sizes: train={len(split_ids['train_ids'])}, "
        f"val={len(split_ids['val_ids'])}, test={len(split_ids['test_ids'])}"
    )

    # --------------------------------------------------------
    # Static averages H_AVGS per (env, wifi_band)
    # Use train + val ids, same as in training
    # --------------------------------------------------------
    all_trainval_ids = split_ids["train_ids"] + split_ids["val_ids"]
    T_target = cfg["preprocess"]["T_target"]
    print(
        f"[INFO] Computing static averages H_AVGS for T_target={T_target} "
        f"over {len(all_trainval_ids)} train+val samples..."
    )
    H_avgs_dict = compute_static_average(
        sample_ids=all_trainval_ids,
        annotation_csv=str(annotation_csv),
        csi_amp_root=str(csi_amp_root),
        T_target=T_target,
    )
    print(f"[INFO] Computed static averages for {len(H_avgs_dict)} (env, band) pairs")

    # --------------------------------------------------------
    # Build validation dataset and loader
    # --------------------------------------------------------
    alpha = cfg["preprocess"]["alpha"]
    noise_power = cfg["preprocess"]["noise_power"]
    C_tx = cfg["preprocess"]["C_tx"]
    preprocessed_dir = cfg["preprocess"].get("preprocessed_dir", None)

    ds_val = WiMansActivityDataset(
        annotation_csv=str(annotation_csv),
        csi_amp_root=str(csi_amp_root),
        sample_ids=split_ids["val_ids"],
        H_avgs_dict=H_avgs_dict,
        T_target=T_target,
        alpha=alpha,
        noise_power=noise_power,
        C_tx=C_tx,
        cache_preprocessed=True,
        preprocessed_dir=preprocessed_dir,
    )

    batch_size = args.batch_size or cfg["train"]["batch_size"]
    dl_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(
        f"[INFO] Validation DataLoader: {len(ds_val)} samples, "
        f"{len(dl_val)} batches, batch_size={batch_size}"
    )

    # --------------------------------------------------------
    # Build model & load checkpoint
    # --------------------------------------------------------
    model = WiMUAR_HSTNN(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        gru_hidden=cfg["model"]["gru_hidden"],
        num_teachers=2,
        dropout_p=cfg["model"].get("dropout_p", 0.3),
    ).to(device)

    print(f"[INFO] Loading checkpoint from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --------------------------------------------------------
    # Run model on validation set and collect logits/labels
    # --------------------------------------------------------
    all_logits = []
    all_y_act = []
    all_y_count = []

    with torch.no_grad():
        for x, y_act, y_count, meta in dl_val:
            x = x.to(device)
            y_act = y_act.to(device)
            y_count = y_count.to(device)

            student_logits, _ = model(x)  # (B, 54)
            all_logits.append(student_logits.cpu())
            all_y_act.append(y_act.cpu())
            all_y_count.append(y_count.cpu())

    logits_all = torch.cat(all_logits, dim=0)        # (N, 54)
    y_act_all = torch.cat(all_y_act, dim=0)          # (N, 54)
    y_count_all = torch.cat(all_y_count, dim=0)      # (N,)

    true_6x9 = y_act_all.view(-1, 6, len(ACTIVITIES))

    print(f"[INFO] Collected logits for {logits_all.shape[0]} validation samples")

    # --------------------------------------------------------
    # Threshold sweep
    # --------------------------------------------------------
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    best_thr = None
    best_f1 = -1.0
    best_act = None
    best_cnt = None

    print("\n[RESULTS] Threshold sweep on validation set:")
    print("thr\tact_F1\tact_P\tact_R\tcnt_acc\tcnt_MAE")

    for thr in thresholds:
        pred_6x9 = logits_to_activity_matrix(logits_all, threshold=thr)
        pred_counts = derive_count_from_pred(pred_6x9)

        act_metrics = compute_activity_metrics(pred_6x9, true_6x9)
        cnt_metrics = compute_count_metrics(pred_counts, y_count_all)

        f1 = act_metrics["act_micro_f1"]
        print(
            f"{thr:.2f}\t"
            f"{f1:.4f}\t"
            f"{act_metrics['act_micro_precision']:.4f}\t"
            f"{act_metrics['act_micro_recall']:.4f}\t"
            f"{cnt_metrics['count_acc']:.4f}\t"
            f"{cnt_metrics['count_mae']:.4f}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_act = act_metrics
            best_cnt = cnt_metrics

    print("\n[SUMMARY] Best threshold on validation:")
    print(f"Best thr = {best_thr:.2f}")
    print(f"Activity micro-F1      = {best_act['act_micro_f1']:.4f}")
    print(f"Activity micro-Prec    = {best_act['act_micro_precision']:.4f}")
    print(f"Activity micro-Recall  = {best_act['act_micro_recall']:.4f}")
    print(f"Activity exact-match   = {best_act['act_exact_match']:.4f}")
    print(f"Count accuracy         = {best_cnt['count_acc']:.4f}")
    print(f"Count MAE              = {best_cnt['count_mae']:.4f}")


if __name__ == "__main__":
    main()
