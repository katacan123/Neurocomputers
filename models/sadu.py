# models/sadu.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
from .backbone import SADUBackbone
from .heads import PerUserActivityHead


class SADUWiMANS(nn.Module):
    """
    Full WiMANS model with SADU backbone + per-user activity head.

    Forward(x) returns:
      {
        "h": (B, 256),
        "z_f": (B, 256, L),
        "activity_logits": (B, max_users, num_classes_activity),
      }
    """

    def __init__(
        self,
        in_channels: int,
        num_classes_activity: int,
        max_users: int,
        backbone_cfg: dict,
        attention_cfg: dict,
    ):
        super().__init__()

        self.backbone = SADUBackbone(
            in_channels=in_channels,
            base_channels=backbone_cfg["base_channels"],
            k1=backbone_cfg["k1"],
            k2=backbone_cfg["k2"],
            k3=backbone_cfg["k3"],
            s1=backbone_cfg["s1"],
            s2=backbone_cfg["s2"],
            s3=backbone_cfg["s3"],
            deep_unfold_K=backbone_cfg["deep_unfold_K"],
            attention_heads=attention_cfg["num_heads"],
            attention_layers=attention_cfg["num_layers"],
        )

        self.activity_head = PerUserActivityHead(
            feature_dim=256,
            max_users=max_users,
            num_classes_activity=num_classes_activity,
        )

    def forward(self, x):
        h, z_f = self.backbone(x)
        activity_logits = self.activity_head(h)
        return {
            "h": h,
            "z_f": z_f,
            "activity_logits": activity_logits,
        }
