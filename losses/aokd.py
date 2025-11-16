# losses/aokd.py

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AOKDLoss(nn.Module):
    """
    Assisted Online Knowledge Distillation (AOKD) loss
    for multi-label activity prediction (54-dim).

    Total loss:
        L = (1 - beta) * L_task + beta * L_kd

    where:
        L_task = BCEWithLogits(student_logits, target) with optional pos_weight
        L_kd   = average BCE between softened student + teacher predictions
    """

    def __init__(
        self,
        temperature: float = 2.0,
        beta: float = 0.5,
        pos_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.beta = float(beta)
        # pos_weight shape: (num_classes,) or None
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.clone().detach())
        else:
            self.pos_weight = None

    def _bce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.pos_weight is not None:
            return F.binary_cross_entropy_with_logits(
                logits, target, pos_weight=self.pos_weight
            )
        else:
            return F.binary_cross_entropy_with_logits(logits, target)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: List[torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        student_logits: (B, 54)
        teacher_logits_list: list of (B, 54)
        target: (B, 54) binary {0,1}
        """
        Tem = self.temperature
        beta = self.beta

        # ----- task loss -----
        loss_task = self._bce(student_logits, target)

        # If KD is disabled or no teachers, just return task loss
        if len(teacher_logits_list) == 0 or beta == 0.0:
            return loss_task

        # ----- KD loss -----
        s_soft = torch.sigmoid(student_logits / Tem)
        loss_kd = 0.0
        for t_logits in teacher_logits_list:
            t_soft = torch.sigmoid(t_logits / Tem)
            loss_kd = loss_kd + F.binary_cross_entropy(
                s_soft, t_soft, reduction="mean"
            )

        loss_kd = loss_kd / len(teacher_logits_list)

        # ----- combine -----
        total_loss = (1.0 - beta) * loss_task + beta * loss_kd
        return total_loss
