# utils/metrics.py

from typing import Dict

import torch


def logits_to_activity_matrix(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Convert activity logits to binary 6x9 matrix per sample.

    Inputs:
        logits: (B, 54)
        threshold: sigmoid threshold (default 0.5)

    Returns:
        pred_bin: (B, 6, 9) in {0,1}
    """
    probs = torch.sigmoid(logits)          # (B, 54)
    bin_ = (probs >= threshold).float()    # (B, 54)
    pred_6x9 = bin_.view(-1, 6, 9)        # (B, 6, 9)
    return pred_6x9


def derive_count_from_pred(pred_6x9: torch.Tensor) -> torch.Tensor:
    """
    Derive human count from predicted 6x9 activity matrix.

    pred_6x9: (B, 6, 9), each row corresponds to a user.

    Strategy:
      - A user is present if any activity in that row is active (sum > 0).
      - Count = number of present users.

    Returns:
      counts: (B,) int tensor
    """
    # sum over activity dimension -> (B, 6)
    present_mask = (pred_6x9.sum(dim=2) > 0)
    counts = present_mask.sum(dim=1)  # (B,)
    return counts


def compute_count_metrics(
    pred_counts: torch.Tensor,
    true_counts: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute simple metrics for derived human count.

    Inputs:
        pred_counts: (N,) int tensor
        true_counts: (N,) int tensor

    Returns:
        {
          "count_acc": float,  # exact match accuracy
          "count_mae": float,  # mean absolute error
        }
    """
    if pred_counts.shape != true_counts.shape:
        raise ValueError(
            f"Shape mismatch: pred_counts {pred_counts.shape}, "
            f"true_counts {true_counts.shape}"
        )

    pred_counts = pred_counts.to(true_counts.device)

    acc = (pred_counts == true_counts).float().mean().item()
    mae = (pred_counts.to(torch.float32) - true_counts.to(torch.float32)).abs().mean().item()

    return {"count_acc": acc, "count_mae": mae}


def compute_activity_metrics(
    pred_6x9: torch.Tensor,
    true_6x9: torch.Tensor,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Multi-label activity metrics (micro-averaged):

      - micro_precision, micro_recall, micro_f1
      - exact_match: all 54 labels correct for a sample

    pred_6x9: (B, 6, 9) in {0,1}
    true_6x9: (B, 6, 9) in {0,1}
    """
    if pred_6x9.shape != true_6x9.shape:
        raise ValueError(
            f"Shape mismatch: pred_6x9 {pred_6x9.shape}, true_6x9 {true_6x9.shape}"
        )

    B = pred_6x9.shape[0]
    pred_flat = pred_6x9.view(B, -1)   # (B,54)
    true_flat = true_6x9.view(B, -1)   # (B,54)

    # TP, FP, FN (micro)
    tp = (pred_flat * true_flat).sum().item()
    fp = (pred_flat * (1 - true_flat)).sum().item()
    fn = ((1 - pred_flat) * true_flat).sum().item()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # exact match: all 54 labels correct
    exact_match = (pred_flat == true_flat).all(dim=1).float().mean().item()

    return {
        "act_micro_precision": float(precision),
        "act_micro_recall": float(recall),
        "act_micro_f1": float(f1),
        "act_exact_match": float(exact_match),
    }
