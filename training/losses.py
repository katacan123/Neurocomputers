# training/losses.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# training/losses.py
#
# Multi-task loss for SADU:
#   - Activity classification (per user slot)
#   - Identity classification (per user slot)
#   - Location classification (per user slot)
#
# Each head uses standard softmax + cross-entropy.
# We support:
#   - slot_mask: mask out empty user slots
#   - ignore_index: per-task "no label" value (e.g., -1)
#
# Total loss:
#   L_total = w_act * L_act + w_id * L_id + w_loc * L_loc
#
# In the paper they do not explicitly state non-unit weights,
# so we use all weights = 1.0 by default. You can control them
# via the config (see below).

from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _task_ce_loss(
    logits: Tensor,              # (B, U, C)
    targets: Tensor,             # (B, U)
    slot_mask: Optional[Tensor], # (B, U) or None
    ignore_index: int = -1,
) -> Tensor:
    """
    Generic cross-entropy loss for one per-slot classification task.

    - Flattens (B, U, C) -> (B*U, C)
    - Applies slot_mask (if provided) to keep only active user slots.
    - Applies ignore_index (e.g., -1) to skip unknown labels.

    Returns scalar loss tensor on same device as logits.
    If no valid elements remain, returns 0.0 on logits.device.
    """
    B, U, C = logits.shape

    logits_flat = logits.view(B * U, C)        # (BU, C)
    targets_flat = targets.view(B * U)         # (BU,)

    valid_mask = torch.ones_like(targets_flat, dtype=torch.bool, device=logits.device)

    # Mask by slot presence
    if slot_mask is not None:
        sm_flat = slot_mask.view(B * U).to(logits.device)
        valid_mask = valid_mask & (sm_flat > 0.5)

    # Mask by ignore_index
    if ignore_index is not None:
        valid_mask = valid_mask & (targets_flat != ignore_index)

    if valid_mask.sum() == 0:
        # No valid entries -> zero loss (but differentiable)
        return torch.zeros((), dtype=torch.float32, device=logits.device)

    logits_valid = logits_flat[valid_mask]
    targets_valid = targets_flat[valid_mask]

    return F.cross_entropy(logits_valid, targets_valid)


def multitask_loss(
    out: Dict[str, Tensor],
    y_act: Optional[Tensor] = None,
    y_id: Optional[Tensor] = None,
    y_loc: Optional[Tensor] = None,
    slot_mask: Optional[Tensor] = None,
    loss_weights: Optional[Dict[str, float]] = None,
    ignore_index_id: int = -1,
    ignore_index_loc: int = -1,
) -> Tuple[Tensor, Dict[str, float]]:
    """
    Multi-task loss for SADU.

    Inputs:
      out:
        - "activity_logits": (B, U, C_act)
        - "identity_logits": (B, U, C_id)  [optional]
        - "location_logits": (B, U, C_loc) [optional]
      y_act: (B, U)
      y_id:  (B, U) with -1 for unknown IDs
      y_loc: (B, U) with -1 for unknown locations
      slot_mask: (B, U) with 1 for real slots, 0 for empty
      loss_weights: dict with keys "activity", "identity", "location"
        If None -> all weights = 1.0
      ignore_index_id: value meaning "no identity label" (default -1)
      ignore_index_loc: value meaning "no location label" (default -1)

    Returns:
      total_loss (scalar tensor), loss_dict (python floats)
    """

    if loss_weights is None:
        loss_weights = {"activity": 1.0, "identity": 1.0, "location": 1.0}

    device = next(iter(out.values())).device

    total_loss = torch.zeros((), dtype=torch.float32, device=device)
    loss_dict: Dict[str, float] = {
        "activity": 0.0,
        "identity": 0.0,
        "location": 0.0,
    }

    # Activity loss (always present for SADU)
    if "activity_logits" in out and y_act is not None:
        L_act = _task_ce_loss(out["activity_logits"], y_act, slot_mask, ignore_index=None)
        total_loss = total_loss + loss_weights.get("activity", 1.0) * L_act
        loss_dict["activity"] = float(L_act.item())

    # Identity loss (optional, depending on dataset/annotation)
    if "identity_logits" in out and y_id is not None:
        L_id = _task_ce_loss(out["identity_logits"], y_id, slot_mask, ignore_index=ignore_index_id)
        total_loss = total_loss + loss_weights.get("identity", 1.0) * L_id
        loss_dict["identity"] = float(L_id.item())

    # Location loss (optional, depending on dataset/annotation)
    if "location_logits" in out and y_loc is not None:
        L_loc = _task_ce_loss(out["location_logits"], y_loc, slot_mask, ignore_index=ignore_index_loc)
        total_loss = total_loss + loss_weights.get("location", 1.0) * L_loc
        loss_dict["location"] = float(L_loc.item())

    return total_loss, loss_dict
