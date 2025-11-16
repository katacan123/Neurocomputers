# train/train_wimuar.py

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.io_utils import load_config, ensure_dir
from utils.seed_utils import set_seed
from utils.logging_utils import setup_logger

from data.splits import split_by_env_with_val
from data.preprocess_csi import compute_static_average
from data.wimans_dataset import WiMansActivityDataset

from models.wimuar_hstnn import WiMUAR_HSTNN
from losses.aokd import AOKDLoss

from utils.metrics import (
    logits_to_activity_matrix,
    derive_count_from_pred,
    compute_count_metrics,
    compute_activity_metrics,
)

from data.wimans_dataset import WiMansActivityDataset, ACTIVITIES



def parse_args():
    ap = argparse.ArgumentParser(description="Train WiMUAR-style model on WiMANS dataset")
    ap.add_argument(
        "--config",
        type=str,
        default="configs/base_wimans.yaml",
        help="Path to YAML config file",
    )
    ap.add_argument(
        "--run_name",
        type=str,
        default="wimans_run",
        help="Name for this run (used for checkpoints/logs directory)",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # ------------------ Setup ------------------
    set_seed(42)

    run_root = ensure_dir(Path("runs") / args.run_name)
    ckpt_dir = ensure_dir(run_root / "checkpoints")
    log_file = run_root / "train.log"

    logger = setup_logger(log_file=str(log_file))
    logger.info(f"Using config: {args.config}")
    logger.info(f"Run directory: {run_root}")

    device = torch.device(cfg["train"]["device"])

    # ------------------ Splits ------------------
    annotation_csv = cfg["data"]["annotation_csv"]
    csi_amp_root = cfg["data"]["csi_amp_root"]
    preprocessed_dir = cfg["data"]["preprocessed_dir"]
    T_target = cfg["data"]["T"]

    env_train = tuple(cfg["data"]["env_train"])
    test_env = cfg["data"]["env_test"]

    logger.info(f"Train envs: {env_train}, Test env: {test_env}")

    split_ids = split_by_env_with_val(
        annotation_csv=annotation_csv,
        train_envs=env_train,
        test_env=test_env,
        val_ratio=0.2,
        random_state=42,
    )

    logger.info(
        f"Split sizes: train={len(split_ids['train_ids'])}, "
        f"val={len(split_ids['val_ids'])}, "
        f"test={len(split_ids['test_ids'])}"
    )

    # ------------------ Static averages (H_AVGS) ------------------
    all_ids_for_static = (
        split_ids["train_ids"]
        + split_ids["val_ids"]
        + split_ids["test_ids"]
    )

    logger.info("Computing static averages H_AVGS per (env, band)...")
    H_avgs_dict = compute_static_average(
        sample_ids=all_ids_for_static,
        annotation_csv=annotation_csv,
        csi_amp_root=csi_amp_root,
        T_target=T_target,
    )
    logger.info(f"Computed static averages for {len(H_avgs_dict)} (env, band) pairs")

    # ------------------ Datasets & Loaders ------------------
    alpha = cfg["preprocess"]["alpha"]
    noise_power = cfg["preprocess"]["noise_power"]
    C_tx = cfg["preprocess"]["C_tx"]

    ds_train = WiMansActivityDataset(
        annotation_csv=annotation_csv,
        csi_amp_root=csi_amp_root,
        sample_ids=split_ids["train_ids"],
        H_avgs_dict=H_avgs_dict,
        T_target=T_target,
        alpha=alpha,
        noise_power=noise_power,
        C_tx=C_tx,
        cache_preprocessed=True,
        preprocessed_dir=preprocessed_dir,
    )
    ds_val = WiMansActivityDataset(
        annotation_csv=annotation_csv,
        csi_amp_root=csi_amp_root,
        sample_ids=split_ids["val_ids"],
        H_avgs_dict=H_avgs_dict,
        T_target=T_target,
        alpha=alpha,
        noise_power=noise_power,
        C_tx=C_tx,
        cache_preprocessed=True,
        preprocessed_dir=preprocessed_dir,
    )

        # ------------------ pos_weight for class imbalance ------------------
    # Using your sanity-check stats: mean(y) ≈ 0.0483
    # pos_weight ~= (1 - p) / p  -> about 19.7
    num_classes = cfg["model"]["num_classes"]
    base_pos_weight = (1.0 - 0.0483) / 0.0483  # ~19.7
    pos_weight = torch.full((num_classes,), base_pos_weight, dtype=torch.float32)
    logger.info(
        f"Using constant pos_weight={base_pos_weight:.2f} for all {num_classes} classes"
    )


    dl_train = DataLoader(
        ds_train,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Train loader: {len(dl_train)} batches, Val loader: {len(dl_val)} batches")

    # ------------------ Model, Loss, Optim ------------------
    model = WiMUAR_HSTNN(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        gru_hidden=cfg["model"]["gru_hidden"],
        num_teachers=2,
        dropout_p=0.5,
    ).to(device)

    pos_weight = pos_weight.to(device)

    criterion = AOKDLoss(
        temperature=cfg["aokd"]["temperature"],
        beta=cfg["aokd"]["beta"],      # currently 0.0 (no KD)
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )


    # ------------------ Training Loop ------------------
    best_val_f1 = -1.0
    best_ckpt_path = ckpt_dir / "best_model.pt"

    num_epochs = cfg["train"]["epochs"]
    logger.info(f"Starting training for {num_epochs} epochs")

    patience = cfg["train"].get("patience", 20)
    min_delta = cfg["train"].get("min_delta", 1e-4)
    val_target = cfg["train"].get("val_target", None)
    epochs_no_improve = 0

    try:
        for epoch in range(1, num_epochs + 1):

            # ----- TRAIN -----
            model.train()
            train_loss = 0.0

            pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{num_epochs} [train]", leave=False)
            for x, y_act, y_count, meta in pbar:
                x = x.to(device)
                y_act = y_act.to(device)

                optimizer.zero_grad()
                student_logits, teacher_logits_list = model(x)
                loss = criterion(student_logits, teacher_logits_list, y_act)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * x.size(0)
                pbar.set_postfix(loss=loss.item())

            train_loss /= len(ds_train)

            # ----- VALIDATION -----
            model.eval()
            val_loss = 0.0

            all_pred_6x9, all_true_6x9 = [], []
            all_pred_counts, all_true_counts = [], []

            with torch.no_grad():
                first = True
                for x, y_act, y_count, meta in dl_val:
                    x = x.to(device)
                    y_act = y_act.to(device)
                    y_count = y_count.to(device)
                    if first:
                        # DEBUG: Are there any positives in the labels?
                        print("DEBUG y_act: mean over all labels:", y_act.mean().item())
                        print("DEBUG y_act: avg number of 1s per sample:",
                            y_act.sum(dim=1).mean().item())
                        first = False


                    student_logits, teacher_logits_list = model(x)
                    loss = criterion(student_logits, teacher_logits_list, y_act)
                    val_loss += loss.item() * x.size(0)

                    pred_6x9 = logits_to_activity_matrix(student_logits)
                    true_6x9 = y_act.view(-1, 6, 9)
                    pred_counts = derive_count_from_pred(pred_6x9)

                    all_pred_6x9.append(pred_6x9.cpu())
                    all_true_6x9.append(true_6x9.cpu())
                    all_pred_counts.append(pred_counts.cpu())
                    all_true_counts.append(y_count.cpu())

            val_loss /= len(ds_val)

            pred_6x9 = torch.cat(all_pred_6x9, dim=0)
            true_6x9 = torch.cat(all_true_6x9, dim=0)
            pred_counts = torch.cat(all_pred_counts, dim=0)
            true_counts = torch.cat(all_true_counts, dim=0)

            act_metrics = compute_activity_metrics(pred_6x9, true_6x9)
            count_metrics = compute_count_metrics(pred_counts, true_counts)

            val_f1 = act_metrics["act_micro_f1"]

            logger.info(
                f"Epoch {epoch}/{num_epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"act_f1={val_f1:.4f} | act_exact={act_metrics['act_exact_match']:.4f} | "
                f"cnt_acc={count_metrics['count_acc']:.4f} | cnt_mae={count_metrics['count_mae']:.4f}"
            )

            # ----- SAVE BEST CHECKPOINT -----
            if val_f1 > best_val_f1 + min_delta:
                best_val_f1 = val_f1
                epochs_no_improve = 0  # reset patience counter

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_f1": val_f1,
                        "config": cfg,
                    },
                    best_ckpt_path,
                )
                logger.info(f"New best model saved to {best_ckpt_path} (val_f1={val_f1:.4f})")
            else:
                epochs_no_improve += 1

            # ----- EARLY STOP: patience -----
            if epochs_no_improve >= patience:
                logger.info(
                    f"Early stopping triggered (patience={patience}), best_f1={best_val_f1:.4f}"
                )
                break

            # ----- EARLY STOP: reaching validation target -----
            if val_target is not None and val_f1 >= val_target:
                logger.info(
                    f"Target val_f1 reached: {val_f1:.4f} >= {val_target}. Stopping early."
                )
                break

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user (CTRL+C).")


        # =========================
    # After training: load best model and evaluate on TEST environment
    # =========================
    logger.info(f"Training finished. Best val_f1={best_val_f1:.4f}")
    logger.info(f"Best checkpoint: {best_ckpt_path}")

    # ---- Load best checkpoint ----
    logger.info(f"Loading best checkpoint from {best_ckpt_path}")
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    # ---- Build test dataset & dataloader (unseen environment) ----
    logger.info(f"Preparing test set for environment: {test_env}")
    alpha = cfg["preprocess"]["alpha"]
    noise_power = cfg["preprocess"]["noise_power"]
    C_tx = cfg["preprocess"]["C_tx"]

    ds_test = WiMansActivityDataset(
        annotation_csv=annotation_csv,
        csi_amp_root=csi_amp_root,
        sample_ids=split_ids["test_ids"],   # <-- unseen env samples
        H_avgs_dict=H_avgs_dict,            # same static averages used in train/val
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

    logger.info(f"Test loader: {len(dl_test)} batches, {len(ds_test)} samples")

    # ---- Inference on test set ----
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
            probs = torch.sigmoid(student_logits)
            print("logits mean:", student_logits.mean().item())
            print("probs min/max:", probs.min().item(), probs.max().item())
            print("fraction of probs >= 0.5:", (probs >= 0.5).float().mean().item())


            # activities -> 6x9 matrix
            pred_6x9 = logits_to_activity_matrix(student_logits, threshold=0.3) # (B, 6, 9)
            true_6x9 = y_act.view(-1, 6, len(ACTIVITIES))         # (B, 6, 9)

            # derived counts from predicted activities
            pred_counts = derive_count_from_pred(pred_6x9)        # (B,)

            all_pred_6x9.append(pred_6x9.cpu())
            all_true_6x9.append(true_6x9.cpu())
            all_pred_counts.append(pred_counts.cpu())
            all_true_counts.append(y_count.cpu())

    pred_6x9 = torch.cat(all_pred_6x9, dim=0)
    true_6x9 = torch.cat(all_true_6x9, dim=0)
    pred_counts = torch.cat(all_pred_counts, dim=0)
    true_counts = torch.cat(all_true_counts, dim=0)

    act_metrics_test = compute_activity_metrics(pred_6x9, true_6x9)
    count_metrics_test = compute_count_metrics(pred_counts, true_counts)

    logger.info("=== TEST RESULTS (unseen environment) ===")
    logger.info(
        "Activity metrics: "
        f"micro_F1={act_metrics_test['act_micro_f1']:.4f}, "
        f"micro_P={act_metrics_test['act_micro_precision']:.4f}, "
        f"micro_R={act_metrics_test['act_micro_recall']:.4f}, "
        f"exact_match={act_metrics_test['act_exact_match']:.4f}"
    )
    logger.info(
        "Count metrics: "
        f"accuracy={count_metrics_test['count_acc']:.4f}, "
        f"MAE={count_metrics_test['count_mae']:.4f}"
    )

    print("\n=== TEST SUMMARY (unseen environment) ===")
    print(f"Activity micro F1:    {act_metrics_test['act_micro_f1']:.4f}")
    print(f"Activity micro Prec.: {act_metrics_test['act_micro_precision']:.4f}")
    print(f"Activity micro Rec.:  {act_metrics_test['act_micro_recall']:.4f}")
    print(f"Activity exact match: {act_metrics_test['act_exact_match']:.4f}")
    print(f"Count accuracy:       {count_metrics_test['count_acc']:.4f}")
    print(f"Count MAE:            {count_metrics_test['count_mae']:.4f}")



if __name__ == "__main__":
    main()
