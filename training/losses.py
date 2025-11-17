# training/losses.py

import torch
import torch.nn.functional as F
from torch import Tensor


def activity_loss(
    activity_logits: Tensor,  # (B, max_users, num_classes)
    y_act: Tensor,            # (B, max_users)
    slot_mask: Tensor = None, # (B, max_users)
) -> Tensor:
    """
    Cross-entropy loss over per-user activity slots.

    If slot_mask is provided:
      - Only slots with mask == 1 contribute.
    """
    B, U, C = activity_logits.shape
    logits_flat = activity_logits.view(B * U, C)
    targets_flat = y_act.view(B * U)

    if slot_mask is not None:
        mask_flat = slot_mask.view(B * U).bool()
        logits_flat = logits_flat[mask_flat]
        targets_flat = targets_flat[mask_flat]

        if logits_flat.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=activity_logits.device)

    return F.cross_entropy(logits_flat, targets_flat)
