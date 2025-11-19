# training/metrics.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import Tensor
from typing import Tuple, Dict
from collections import defaultdict


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


def activity_accuracy(
    activity_logits: Tensor,  # (B, max_users, num_classes)
    y_act: Tensor,            # (B, max_users)
    slot_mask: Tensor = None, # (B, max_users)
) -> float:
    """
    Per-slot activity accuracy (optionally masked).
    """
    B, U, C = activity_logits.shape
    preds = activity_logits.argmax(dim=-1)
    correct = (preds == y_act)

    if slot_mask is not None:
        mask = slot_mask.bool()
        correct = correct & mask
        denom = mask.sum()
    else:
        denom = torch.numel(y_act)

    if denom.item() == 0:
        return 0.0

    acc = correct.sum().float() / denom.float()
    return acc.item()


def accumulate_env_metrics(
    env_to_stats: Dict[str, Dict[str, float]],
    env: str,
    loss: float,
    act_acc: float,
    count_acc: float,
    count_mae: float,
    weight: int = 1,
):
    """
    Accumulate environment-level metric sums.
    """
    if env not in env_to_stats:
        env_to_stats[env] = {
            "loss_sum": 0.0,
            "act_acc_sum": 0.0,
            "count_acc_sum": 0.0,
            "count_mae_sum": 0.0,
            "weight_sum": 0,
        }

    env_to_stats[env]["loss_sum"] += loss * weight
    env_to_stats[env]["act_acc_sum"] += act_acc * weight
    env_to_stats[env]["count_acc_sum"] += count_acc * weight
    env_to_stats[env]["count_mae_sum"] += count_mae * weight
    env_to_stats[env]["weight_sum"] += weight


def finalize_env_metrics(env_to_stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Normalize environment-level metric sums to averages.
    """
    out = {}
    for env, d in env_to_stats.items():
        w = max(d["weight_sum"], 1)
        out[env] = {
            "loss": d["loss_sum"] / w,
            "act_acc": d["act_acc_sum"] / w,
            "count_acc": d["count_acc_sum"] / w,
            "count_mae": d["count_mae_sum"] / w,
        }
    return out
