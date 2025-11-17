# models/baseline_cnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple1DCNNBackbone(nn.Module):
    """
    Simple 1D CNN baseline backbone for WiMANS:
      - Input BN
      - 3 conv blocks: Conv1d -> BN -> ReLU -> MaxPool1d
      - Global mean pooling -> (B, D)
    """

    def __init__(
        self,
        in_channels: int,
        channels: int = 64,
        num_layers: int = 3,
        kernel_size: int = 9,
        pool_stride: int = 2,
    ):
        super().__init__()

        self.bn_in = nn.BatchNorm1d(in_channels)

        layers = []
        c_in = in_channels
        for _ in range(num_layers):
            layers.append(
                nn.Conv1d(
                    c_in,
                    channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(pool_stride))
            c_in = channels

        self.conv = nn.Sequential(*layers)
        self.out_dim = channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.bn_in(x)
        z_f = self.conv(x)
        h = z_f.mean(dim=-1)
        return h, z_f


class PerUserActivityHead(nn.Module):
    """
    Same as SADU head, redefined here for convenience.
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
        B = h.size(0)
        out = self.mlp(h)
        out = out.view(B, self.max_users, self.num_classes_activity)
        return out


class BaselineCNNWiMANS(nn.Module):
    """
    Baseline model:
      - Simple1DCNNBackbone
      - Per-user activity head
    """

    def __init__(
        self,
        in_channels: int,
        num_classes_activity: int,
        max_users: int,
        backbone_cfg: dict,
    ):
        super().__init__()

        self.backbone = Simple1DCNNBackbone(
            in_channels=in_channels,
            channels=backbone_cfg.get("channels", 64),
            num_layers=backbone_cfg.get("num_layers", 3),
            kernel_size=backbone_cfg.get("kernel_size", 9),
            pool_stride=backbone_cfg.get("pool_stride", 2),
        )

        self.activity_head = PerUserActivityHead(
            feature_dim=self.backbone.out_dim,
            max_users=max_users,
            num_classes_activity=num_classes_activity,
        )

    def forward(self, x: torch.Tensor):
        h, z_f = self.backbone(x)
        activity_logits = self.activity_head(h)
        return {
            "h": h,
            "z_f": z_f,
            "activity_logits": activity_logits,
        }
