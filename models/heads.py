# models/heads.py

import torch
import torch.nn as nn


class PerUserActivityHead(nn.Module):
    """
    Per-user activity head.

    Takes global feature h (B, D) and outputs per-slot activity logits:
      (B, max_users, num_classes_activity)
    """

    def __init__(
        self,
        feature_dim: int,
        max_users: int,
        num_classes_activity: int,
        hidden_dim: int = None,
    ):
        super().__init__()
        self.max_users = max_users
        self.num_classes_activity = num_classes_activity

        if hidden_dim is None:
            hidden_dim = feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, max_users * num_classes_activity),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, D)
        returns:
          activity_logits: (B, max_users, num_classes_activity)
        """
        B = h.size(0)
        out = self.mlp(h)
        out = out.view(B, self.max_users, self.num_classes_activity)
        return out
