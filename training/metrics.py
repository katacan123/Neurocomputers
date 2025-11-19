# training/metrics.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# training/metrics.py

import torch
from torch import Tensor
from typing import Tuple, Dict, Optional
from collections import defaultdict


def per_slot_accuracy(
    logits: Tensor,              # (B, U, C)
    targets: Tensor,             # (B, U)
    slot_mask: Optional[Tensor] = None,  # (B, U)
    ignore_index: Optional[int] = None,
) -> float:
    """
    General per-slot accuracy used for activity / identity / location.

    - Flatten to (B*U,) and mask invalid entries
      (slot_mask == 0 or target == ignore_index).
    """
    B, U, C = logits.shape
    preds = logits.argmax(dim=-1)  # (B, U)

    preds_flat = preds.view(-1)
    t_flat = targets.view(-1)
    valid_mask = torch.ones_like(t_flat, dtype=torch.bool, device=logits.device)

    if slot_mask is not None:
        sm_flat = slot_mask.view(-1).to(logits.device)
        valid_mask = valid_mask & (sm_flat > 0.5)

    if ignore_index is not None:
        valid_mask = valid_mask & (t_flat != ignore_index)

    if valid_mask.sum() == 0:
        return 0.0

    correct = (preds_flat == t_flat) & valid_mask
    acc = correct.sum().float() / valid_mask.sum().float()
    return float(acc.item())


def derive_counts_from_activity(
    activity_logits: Tensor,   # (B, max_users, num_classes)
    act_nothing_class: int,
    threshold: float,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Derive human count from per-slot activity predictions:

      slot is active if:
        pred_class != ACT_NOTHING and pred_conf > threshold
    """
    probs = activity_logits.softmax(dim=-1)
    pred_classes = probs.argmax(dim=-1)           # (B, U)
    pred_confs = probs.max(dim=-1).values         # (B, U)

    active = (pred_classes != act_nothing_class) & (pred_confs > threshold)
    pred_counts = active.sum(dim=-1)              # (B,)

    return pred_counts, pred_classes, pred_confs


def count_metrics(pred_counts: Tensor, gt_counts: Tensor):
    """
    Simple count metrics:
      - accuracy: P(pred_count == gt_count)
      - MAE: mean(|pred_count - gt_count|)
    """
    pred_counts = pred_counts.to(gt_counts.device)
    gt_counts = gt_counts.to(pred_counts.device)

    correct = (pred_counts == gt_counts).float().mean().item()
    mae = (pred_counts.float() - gt_counts.float()).abs().mean().item()
    return correct, mae
