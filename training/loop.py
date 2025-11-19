# training/loop.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from .losses import multitask_loss
from .metrics import (
    derive_counts_from_activity,
    count_metrics,
    per_slot_accuracy,
)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: Optional[GradScaler],
    act_nothing_class: int,
    count_threshold: float,
    loss_weights: Optional[Dict[str, float]] = None,
    ignore_index_id: int = -1,
    ignore_index_loc: int = -1,
) -> Dict[str, float]:
    """
    One training epoch with multi-task SADU loss.

    Metrics returned (all averaged over batches):
      - loss_total
      - loss_activity
      - loss_identity
      - loss_location
      - act_acc
      - id_acc
      - loc_acc
      - count_acc
      - count_mae
    """
    model.train()

    # running sums
    sums = {
        "loss_total": 0.0,
        "loss_activity": 0.0,
        "loss_identity": 0.0,
        "loss_location": 0.0,
        "act_acc": 0.0,
        "id_acc": 0.0,
        "loc_acc": 0.0,
        "count_acc": 0.0,
        "count_mae": 0.0,
    }
    n_batches = 0

    for batch in dataloader:
        x = batch["x"].to(device)              # (B, C, T)
        y_act = batch["y_act"].to(device)      # (B, U)
        y_id  = batch.get("y_id", None)
        y_loc = batch.get("y_loc", None)
        if y_id is not None:
            y_id = y_id.to(device)
        if y_loc is not None:
            y_loc = y_loc.to(device)

        slot_mask = batch.get("slot_mask", None)
        if slot_mask is not None:
            slot_mask = slot_mask.to(device)

        gt_counts = batch["gt_count"].to(device)  # (B,)

        optimizer.zero_grad(set_to_none=True)

        use_amp = (device == "cuda") and (scaler is not None)
        with autocast("cuda", enabled=use_amp):
            out = model(x)  # SADUWiMANSFull outputs dict with all logits

            total_loss, loss_dict = multitask_loss(
                out=out,
                y_act=y_act,
                y_id=y_id,
                y_loc=y_loc,
                slot_mask=slot_mask,
                loss_weights=loss_weights,
                ignore_index_id=ignore_index_id,
                ignore_index_loc=ignore_index_loc,
            )

        if use_amp:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # ---- metrics (no grad) ----
        with torch.no_grad():
            act_logits = out["activity_logits"]               # (B, U, C_act)
            pred_counts, _, _ = derive_counts_from_activity(
                act_logits,
                act_nothing_class=act_nothing_class,
                threshold=count_threshold,
            )
            c_acc, c_mae = count_metrics(pred_counts, gt_counts)

            act_acc = per_slot_accuracy(act_logits, y_act, slot_mask, ignore_index=None)

            # Identity / location accuracies only if logits and labels present
            if "identity_logits" in out and (y_id is not None):
                id_acc = per_slot_accuracy(out["identity_logits"], y_id, slot_mask,
                                           ignore_index=ignore_index_id)
            else:
                id_acc = 0.0

            if "location_logits" in out and (y_loc is not None):
                loc_acc = per_slot_accuracy(out["location_logits"], y_loc, slot_mask,
                                            ignore_index=ignore_index_loc)
            else:
                loc_acc = 0.0

        # accumulate
        sums["loss_total"]    += float(total_loss.item())
        sums["loss_activity"] += loss_dict["activity"]
        sums["loss_identity"] += loss_dict["identity"]
        sums["loss_location"] += loss_dict["location"]
        sums["act_acc"]       += act_acc
        sums["id_acc"]        += id_acc
        sums["loc_acc"]       += loc_acc
        sums["count_acc"]     += c_acc
        sums["count_mae"]     += c_mae
        n_batches += 1

    if n_batches == 0:
        return {k: 0.0 for k in sums.keys()}

    return {k: v / n_batches for k, v in sums.items()}


def eval_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    act_nothing_class: int,
    count_threshold: float,
    loss_weights: Optional[Dict[str, float]] = None,
    ignore_index_id: int = -1,
    ignore_index_loc: int = -1,
) -> Dict[str, float]:
    """
    Evaluation loop (no gradient). Same metrics as train_one_epoch.
    """
    model.eval()

    sums = {
        "loss_total": 0.0,
        "loss_activity": 0.0,
        "loss_identity": 0.0,
        "loss_location": 0.0,
        "act_acc": 0.0,
        "id_acc": 0.0,
        "loc_acc": 0.0,
        "count_acc": 0.0,
        "count_mae": 0.0,
    }
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x = batch["x"].to(device)
            y_act = batch["y_act"].to(device)
            y_id  = batch.get("y_id", None)
            y_loc = batch.get("y_loc", None)
            if y_id is not None:
                y_id = y_id.to(device)
            if y_loc is not None:
                y_loc = y_loc.to(device)

            slot_mask = batch.get("slot_mask", None)
            if slot_mask is not None:
                slot_mask = slot_mask.to(device)

            gt_counts = batch["gt_count"].to(device)

            out = model(x)

            total_loss, loss_dict = multitask_loss(
                out=out,
                y_act=y_act,
                y_id=y_id,
                y_loc=y_loc,
                slot_mask=slot_mask,
                loss_weights=loss_weights,
                ignore_index_id=ignore_index_id,
                ignore_index_loc=ignore_index_loc,
            )

            act_logits = out["activity_logits"]
            pred_counts, _, _ = derive_counts_from_activity(
                act_logits,
                act_nothing_class=act_nothing_class,
                threshold=count_threshold,
            )
            c_acc, c_mae = count_metrics(pred_counts, gt_counts)

            act_acc = per_slot_accuracy(act_logits, y_act, slot_mask, ignore_index=None)

            if "identity_logits" in out and (y_id is not None):
                id_acc = per_slot_accuracy(out["identity_logits"], y_id, slot_mask,
                                           ignore_index=ignore_index_id)
            else:
                id_acc = 0.0

            if "location_logits" in out and (y_loc is not None):
                loc_acc = per_slot_accuracy(out["location_logits"], y_loc, slot_mask,
                                            ignore_index=ignore_index_loc)
            else:
                loc_acc = 0.0

            sums["loss_total"]    += float(total_loss.item())
            sums["loss_activity"] += loss_dict["activity"]
            sums["loss_identity"] += loss_dict["identity"]
            sums["loss_location"] += loss_dict["location"]
            sums["act_acc"]       += act_acc
            sums["id_acc"]        += id_acc
            sums["loc_acc"]       += loc_acc
            sums["count_acc"]     += c_acc
            sums["count_mae"]     += c_mae
            n_batches += 1

    if n_batches == 0:
        return {k: 0.0 for k in sums.keys()}

    return {k: v / n_batches for k, v in sums.items()}