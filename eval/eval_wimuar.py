# eval/eval_wimuar.py

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from utils.io_utils import load_config
from utils.seed_utils import set_seed
from utils.logging_utils import setup_logger

from data.splits import split_by_env_with_val
from data.preprocess_csi import compute_static_average
from data.wimans_dataset import WiMansActivityDataset

from models.wimuar_hstnn import WiMUAR_HSTNN
from utils.metrics import (
    logits_to_activity_matrix,
    derive_count_from_pred,
    compute_count_metrics,
    compute_activity_metrics,
)


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate WiMUAR-style model on WiMANS test set")
    ap.add_argument(
        "--config",
        type=str,
        default="configs/base_wimans.yaml",
        help="Path to YAML config file",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (best_model.pt)",
    )
    ap.add_argument(
        "--run_name",
        type=str,
        default="wimans_eval",
        help="Name for eval run (for logs)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(123)

    run_root = Path("runs") / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    log_file = run_root / "eval.log"
    logger = setup_logger(log_file=str(log_file))

    device = torch.device(cfg["train"]["device"])

    annotation_csv = cfg["data"]["annotation_csv"]
    csi_amp_root = cfg["data"]["csi_amp_root"]
    preprocessed_dir = cfg["data"]["preprocessed_dir"]
    T_target = cfg["data"]["T"]

    env_train = tuple(cfg["data"]["env_train"])
    test_env = cfg["data"]["env_test"]

    logger.info(f"Eval config: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Train envs (for split only): {env_train}, Test env: {test_env}")

    split_ids = split_by_env_with_val(
        annotation_csv=annotation_csv,
        train_envs=env_train,
        test_env=test_env,
        val_ratio=0.2,
        random_state=42,
    )

    logger.info(f"Test set size: {len(split_ids['test_ids'])}")

    # We can recompute H_AVGS over all IDs, same as in training
    all_ids_for_static = (
        split_ids["train_ids"]
        + split_ids["val_ids"]
        + split_ids["test_ids"]
    )
    logger.info("Computing static averages H_AVGS per (env, band) for eval...")
    H_avgs_dict = compute_static_average(
        sample_ids=all_ids_for_static,
        annotation_csv=annotation_csv,
        csi_amp_root=csi_amp_root,
        T_target=T_target,
    )

    alpha = cfg["preprocess"]["alpha"]
    noise_power = cfg["preprocess"]["noise_power"]
    C_tx = cfg["preprocess"]["C_tx"]

    ds_test = WiMansActivityDataset(
        annotation_csv=annotation_csv,
        csi_amp_root=csi_amp_root,
        sample_ids=split_ids["test_ids"],
        H_avgs_dict=H_avgs_dict,
        T_target=T_target,
        alpha=alpha,
        noise_power=noise_power,
        C_tx=C_tx,
        cache_preprocessed=True,
        preprocessed_dir=preprocessed_dir,
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ---------------- Model ----------------
    model = WiMUAR_HSTNN(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        gru_hidden=cfg["model"]["gru_hidden"],
        num_teachers=2,
        dropout_p=0.5,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded checkpoint from epoch={ckpt.get('epoch', 'N/A')}, "
                f"val_f1={ckpt.get('val_f1', 'N/A')}")

    model.eval()

    all_pred_6x9 = []
    all_true_6x9 = []
    all_pred_counts = []
    all_true_counts = []

    with torch.no_grad():
        for x, y_act, y_count, meta in dl_test:
            x = x.to(device)
            y_act = y_act.to(device)
            y_count = y_count.to(device)

            student_logits, _ = model(x)

            pred_6x9 = logits_to_activity_matrix(student_logits)  # (B,6,9)
            true_6x9 = y_act.view(-1, 6, 9)

            pred_counts = derive_count_from_pred(pred_6x9)        # (B,)

            all_pred_6x9.append(pred_6x9.cpu())
            all_true_6x9.append(true_6x9.cpu())
            all_pred_counts.append(pred_counts.cpu())
            all_true_counts.append(y_count.cpu())

    pred_6x9 = torch.cat(all_pred_6x9, dim=0)
    true_6x9 = torch.cat(all_true_6x9, dim=0)
    pred_counts = torch.cat(all_pred_counts, dim=0)
    true_counts = torch.cat(all_true_counts, dim=0)

    act_metrics = compute_activity_metrics(pred_6x9, true_6x9)
    count_metrics = compute_count_metrics(pred_counts, true_counts)

    logger.info("=== Test Results (Unseen Environment) ===")
    logger.info(
        "Activity metrics: "
        f"micro_F1={act_metrics['act_micro_f1']:.4f}, "
        f"micro_P={act_metrics['act_micro_precision']:.4f}, "
        f"micro_R={act_metrics['act_micro_recall']:.4f}, "
        f"exact_match={act_metrics['act_exact_match']:.4f}"
    )
    logger.info(
        "Count metrics: "
        f"accuracy={count_metrics['count_acc']:.4f}, "
        f"MAE={count_metrics['count_mae']:.4f}"
    )

    # quick console summary
    print("\n=== TEST SUMMARY (unseen env) ===")
    print(f"Activity micro F1:    {act_metrics['act_micro_f1']:.4f}")
    print(f"Activity micro Prec.: {act_metrics['act_micro_precision']:.4f}")
    print(f"Activity micro Rec.:  {act_metrics['act_micro_recall']:.4f}")
    print(f"Activity exact match: {act_metrics['act_exact_match']:.4f}")
    print(f"Count accuracy:       {count_metrics['count_acc']:.4f}")
    print(f"Count MAE:            {count_metrics['count_mae']:.4f}")


if __name__ == "__main__":
    main()
