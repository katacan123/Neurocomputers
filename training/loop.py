# training/loop.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .losses import activity_loss
from .metrics import (
    derive_counts_from_activity,
    count_metrics,
    activity_accuracy,
)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: GradScaler,
    act_nothing_class: int,
    count_threshold: float,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_act_acc = 0.0
    total_count_acc = 0.0
    total_count_mae = 0.0
    n_batches = 0

    for batch in dataloader:
        x = batch["x"].to(device)
        y_act = batch["y_act"].to(device)
        slot_mask = batch["slot_mask"].to(device)
        gt_counts = batch["gt_count"].to(device)

        optimizer.zero_grad(set_to_none=True)

        use_amp = (device == "cuda") and (scaler is not None)
        with autocast("cuda", enabled=use_amp):
            out = model(x)
            logits = out["activity_logits"]
            loss = activity_loss(logits, y_act, slot_mask)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_counts, _, _ = derive_counts_from_activity(
                logits,
                act_nothing_class=act_nothing_class,
                threshold=count_threshold,
            )
            act_acc = activity_accuracy(logits, y_act, slot_mask)
            c_acc, c_mae = count_metrics(pred_counts, gt_counts)

        total_loss += loss.item()
        total_act_acc += act_acc
        total_count_acc += c_acc
        total_count_mae += c_mae
        n_batches += 1

    if n_batches == 0:
        return {"loss": 0.0, "act_acc": 0.0, "count_acc": 0.0, "count_mae": 0.0}

    return {
        "loss": total_loss / n_batches,
        "act_acc": total_act_acc / n_batches,
        "count_acc": total_count_acc / n_batches,
        "count_mae": total_count_mae / n_batches,
    }


def eval_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    act_nothing_class: int,
    count_threshold: float,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_act_acc = 0.0
    total_count_acc = 0.0
    total_count_mae = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            y_act = batch["y_act"].to(device)
            slot_mask = batch["slot_mask"].to(device)
            gt_counts = batch["gt_count"].to(device)

            out = model(x)
            logits = out["activity_logits"]
            loss = activity_loss(logits, y_act, slot_mask)

            pred_counts, _, _ = derive_counts_from_activity(
                logits,
                act_nothing_class=act_nothing_class,
                threshold=count_threshold,
            )
            act_acc = activity_accuracy(logits, y_act, slot_mask)
            c_acc, c_mae = count_metrics(pred_counts, gt_counts)

            total_loss += loss.item()
            total_act_acc += act_acc
            total_count_acc += c_acc
            total_count_mae += c_mae
            n_batches += 1

    if n_batches == 0:
        return {"loss": 0.0, "act_acc": 0.0, "count_acc": 0.0, "count_mae": 0.0}

    return {
        "loss": total_loss / n_batches,
        "act_acc": total_act_acc / n_batches,
        "count_acc": total_count_acc / n_batches,
        "count_mae": total_count_mae / n_batches,
    }
