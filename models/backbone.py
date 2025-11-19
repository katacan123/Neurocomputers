# models/backbone.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleDepthwiseCNN(nn.Module):
    """
    Multi-scale depthwise separable CNN:
      - Three branches with different kernel/stride.
      - Depthwise conv -> pointwise conv -> BN -> LeakyReLU
      - Concatenate and project to 256 channels.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        k1: int,
        k2: int,
        k3: int,
        s1: int,
        s2: int,
        s3: int,
    ):
        super().__init__()

        self.branch1 = self._make_branch(in_channels, base_channels, k1, s1)
        self.branch2 = self._make_branch(in_channels, base_channels, k2, s2)
        self.branch3 = self._make_branch(in_channels, base_channels, k3, s3)

        self.out_conv = nn.Conv1d(base_channels * 3, 256, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm1d(256)

    def _make_branch(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
    ) -> nn.Sequential:
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv1d(
                in_ch,
                in_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_ch,
                bias=False,
            ),
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        returns:
          z: (B, 256, L)
        """
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        min_L = min(b1.size(-1), b2.size(-1), b3.size(-1))
        if b1.size(-1) != min_L:
            b1 = F.adaptive_max_pool1d(b1, min_L)
        if b2.size(-1) != min_L:
            b2 = F.adaptive_max_pool1d(b2, min_L)
        if b3.size(-1) != min_L:
            b3 = F.adaptive_max_pool1d(b3, min_L)

        z = torch.cat([b1, b2, b3], dim=1)
        z = self.out_conv(z)
        z = self.out_bn(z)
        return z


class DeepUnfoldingModule(nn.Module):
    """
    Deep iteratively unfolded structure:
      u^{k+1} = ReLU(u^k - alpha_k * (W u^k + b))
    """

    def __init__(self, feature_dim: int, K: int):
        super().__init__()
        self.K = K
        self.W = nn.Linear(feature_dim, feature_dim, bias=True)
        self.alpha = nn.Parameter(torch.ones(K))

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        s: (B, D)
        returns:
          u_K: (B, D)
        """
        u = s
        for k in range(self.K):
            step = self.W(u)
            u = F.relu(u - self.alpha[k] * step)
        return u


class ManifoldAttentionFusion(nn.Module):
    """
    Manifold attention + gated fusion module.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.proj_u = nn.Linear(feature_dim, feature_dim)
        self.proj_s = nn.Linear(feature_dim, feature_dim)
        self.att_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        s: torch.Tensor,
        uK: torch.Tensor,
        z_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        s: (B, D)
        uK: (B, D)
        z_mean: (B, D)
        """
        p = self.proj_u(uK)
        q = self.proj_s(s)

        att_input = q + z_mean
        alpha = self.att_mlp(att_input)  # (B, D)

        h = alpha * p + (1.0 - alpha) * s
        return h, alpha


class SelfAttentionRefinement(nn.Module):
    """
    Self-attention refinement using TransformerEncoder over a length-1 sequence.
    """

    def __init__(self, feature_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, D)
        returns:
          h_refined: (B, D)
        """
        x = h.unsqueeze(1)  # (B, 1, D)
        x = self.encoder(x)
        return x.squeeze(1)


class SADUBackbone(nn.Module):
    """
    Full SADU-style backbone:
      - Input BN
      - Multi-scale depthwise CNN -> Z_f
      - Global mean pooling -> s
      - Deep unfolding -> u_K
      - Manifold attention + gated fusion -> h
      - Self-attention refinement -> h_refined
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        k1: int,
        k2: int,
        k3: int,
        s1: int,
        s2: int,
        s3: int,
        deep_unfold_K: int,
        attention_heads: int,
        attention_layers: int,
    ):
        super().__init__()

        self.bn_in = nn.BatchNorm1d(in_channels)

        self.ms_cnn = MultiScaleDepthwiseCNN(
            in_channels=in_channels,
            base_channels=base_channels,
            k1=k1,
            k2=k2,
            k3=k3,
            s1=s1,
            s2=s2,
            s3=s3,
        )

        self.deep_unfold = DeepUnfoldingModule(feature_dim=256, K=deep_unfold_K)
        self.fusion = ManifoldAttentionFusion(feature_dim=256)
        self.self_attn = SelfAttentionRefinement(
            feature_dim=256,
            num_heads=attention_heads,
            num_layers=attention_layers,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, T)

        returns:
          h_refined: (B, 256)
          z_f: (B, 256, L)
        """
        x = self.bn_in(x)
        z_f = self.ms_cnn(x)
        s = z_f.mean(dim=-1)
        z_mean = s
        uK = self.deep_unfold(s)
        h_fused, _ = self.fusion(s, uK, z_mean)
        h_refined = self.self_attn(h_fused)
        return h_refined, z_f
