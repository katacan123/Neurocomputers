# models/agm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class TemporalAttention(nn.Module):
    """
    Simple temporal attention over sequence length T_out.

    Input:
        h: (B, T_out, D)

    Output:
        context: (B, D)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T_out, D)
        scores = self.leaky_relu(self.fc(h))  # (B, T_out, 1)
        alpha = torch.softmax(scores, dim=1)  # attention over time axis
        context = (alpha * h).sum(dim=1)      # (B, D)
        return context


class AGM(nn.Module):
    """
    AGM module from WiMUAR:
      - ABGRU (BiGRU)
      - Temporal attention
      - Multi-branch MLP (student + teacher heads)
    Only outputs activity logits (54-dim), no separate count head.

    Input:
        x: (B, 256, T_out)  from MDC

    Outputs:
        student_logits: (B, num_classes)
        teacher_logits_list: list of (B, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 256,
        gru_hidden: int = 256,
        num_classes: int = 54,
        num_teachers: int = 2,
        dropout_p: float = 0.5,
    ):
        super().__init__()
        self.gru_hidden = gru_hidden
        self.num_classes = num_classes

        # BiGRU (ABGRU in the paper)
        self.bigru = nn.GRU(
            input_size=in_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Temporal attention over GRU outputs
        self.attn = TemporalAttention(d_model=2 * gru_hidden)

        # Student + teachers
        self.student_head = self._make_head(2 * gru_hidden, num_classes, dropout_p)
        self.teacher_heads = nn.ModuleList(
            [self._make_head(2 * gru_hidden, num_classes, dropout_p)
             for _ in range(num_teachers)]
        )

        self._init_weights()

    def _make_head(self, in_dim: int, out_dim: int, dropout_p: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout_p),
            nn.Linear(256, out_dim),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        x: (B, 256, T_out) -> (B, T_out, 256) for GRU
        """
        x = x.permute(0, 2, 1)  # (B, T_out, 256)

        h, _ = self.bigru(x)    # (B, T_out, 2*gru_hidden)

        context = self.attn(h)  # (B, 2*gru_hidden)

        student_logits = self.student_head(context)
        teacher_logits_list = [head(context) for head in self.teacher_heads]

        return student_logits, teacher_logits_list
