# models/wimuar_hstnn.py

import torch
import torch.nn as nn

from .mdc import MDC
from .agm import AGM


class WiMUAR_HSTNN(nn.Module):
    """
    Full WiMUAR core network:

        F_theta(A) = AGM( MDC(A) )

    - MDC: Multi-scale Dilated Conv + Downsampling
    - AGM: ABGRU + Temporal Attention + multi-branch heads

    Input:
        x: (B, C_in=270, T=3000)

    Outputs:
        student_logits: (B, num_classes=54)
        teacher_logits_list: list of (B, 54)
    """

    def __init__(
        self,
        in_channels: int = 270,
        num_classes: int = 54,
        gru_hidden: int = 256,
        num_teachers: int = 2,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.mdc = MDC(in_channels=in_channels, out_channels=256)
        self.agm = AGM(
            in_channels=256,
            gru_hidden=gru_hidden,
            num_classes=num_classes,
            num_teachers=num_teachers,
            dropout_p=dropout_p,
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, C_in, T)
        """
        x = self.mdc(x)
        student_logits, teacher_logits_list = self.agm(x)
        return student_logits, teacher_logits_list
