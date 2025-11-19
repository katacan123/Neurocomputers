# models/sadu_full.py
#
# Complete SADU architecture for WiMANS, matching the paper’s components:
#   - Multi-scale depthwise separable CNN feature extractor
#   - Deep unfolding module (Eq. 7)
#   - Manifold attention with row-normalized W_m (Eqs. 8–10)
#   - Gated fusion (Eq. 11)
#   - Enhanced multi-head self-attention refinement (Sec. II-E, Eq. 12)
#   - Multi-task heads for activity, identity, and location (per user slot)
#
# Shapes (WiMANS configuration):
#   Input x:           (B, 270, T)
#   Z_f:               (B, 256, L)
#   s, u0, u_K, h_*:   (B, 256)
#   MHSA tokens:       (B, 2, 256)
#   activity_logits:   (B, max_users, num_activities)
#   identity_logits:   (B, max_users, num_identities)
#   location_logits:   (B, max_users, num_locations)
#
# You can plug this class into your training code instead of the simpler model.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Multi-scale depthwise separable CNN front-end (Sec. II-A, Eqs. 2–4)
# ---------------------------------------------------------------------------

class MultiScaleDWSeparableCNN(nn.Module):
    """
    Multi-scale depthwise-separable CNN front-end.

    Paper text / equations:
      - Three parallel branches with:
          branch 1: kernel=128, stride=8
          branch 2: kernel=64,  stride=4
          branch 3: kernel=32,  stride=2
      - Each branch:
          depthwise Conv1d (groups=in_channels)
          pointwise Conv1d (1x1) -> 32 channels
          BatchNorm1d + LeakyReLU
      - Outputs are pooled to a common temporal length L = min_i T_i via
        AdaptiveMaxPool1d, concatenated to 96 channels, then a 1x1 conv
        + BN produces Z_f ∈ R^{B×256×L}.

    Returns:
      Z_f: (B, 256, L)
      s:   (B, 256)  # "salient" summary
      u0:  (B, 256)  # "average" summary, used as deep-unfolding input
    """

    def __init__(
        self,
        in_channels: int,
        branch_channels: int = 32,
        kernels: Tuple[int, int, int] = (128, 64, 32),
        strides: Tuple[int, int, int] = (8, 4, 2),
    ):
        super().__init__()
        assert len(kernels) == 3 and len(strides) == 3

        self.in_channels = in_channels
        self.branch_channels = branch_channels

        # Per-channel (subcarrier) normalization
        self.input_bn = nn.BatchNorm1d(in_channels)

        # 3 depthwise + pointwise branches
        self.branches = nn.ModuleList()
        for k, s in zip(kernels, strides):
            padding = k // 2
            branch = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    in_channels,
                    kernel_size=k,
                    stride=s,
                    padding=padding,
                    groups=in_channels,  # depthwise
                    bias=False,
                ),
                nn.Conv1d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,      # pointwise
                    bias=False,
                ),
                nn.BatchNorm1d(branch_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
            self.branches.append(branch)

        # 96 (= 3 * 32) -> 256 projection (Eq. 4)
        self.proj = nn.Conv1d(3 * branch_channels, 256, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm1d(256)

        # Temporal pooling for s and u0.  The paper text/exact notation is a bit
        # inconsistent (it says AdaptiveAvgPool1d for both Eq. (5) and (6), but
        # also talks about "salient" vs "average" behavior). Here we follow the
        # *spirit*:
        #   s  = max-pool over time (salient)
        #   u0 = avg-pool over time (average behavior)
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.pool_avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, F, T)
        """
        x = self.input_bn(x)

        # Multi-scale branches
        feats = []
        lengths = []
        for branch in self.branches:
            z = branch(x)          # (B, 32, T_i)
            feats.append(z)
            lengths.append(z.size(-1))

        # L = min_i T_i
        min_L = min(lengths)
        feats_pooled = []
        for z in feats:
            if z.size(-1) != min_L:
                z = F.adaptive_max_pool1d(z, min_L)
            feats_pooled.append(z)

        # Concatenate along channel dimension: (B, 96, L)
        z = torch.cat(feats_pooled, dim=1)

        # Project 96 -> 256 channels (Eq. 4)
        Z_f = self.proj_bn(self.proj(z))  # (B, 256, L)

        # Temporal summaries from Z_f:
        # s: salient (max over time), u0: average behavior (avg over time)
        s = self.pool_max(Z_f).squeeze(-1)  # (B, 256)
        u0 = self.pool_avg(Z_f).squeeze(-1) # (B, 256)

        return Z_f, s, u0


# ---------------------------------------------------------------------------
#  Deep Unfolding Module (Sec. II-C, Eq. 7)
# ---------------------------------------------------------------------------

class DeepUnfoldingModule(nn.Module):
    """
    Deep unfolding module implementing K unrolled iterations of:

        u^{k+1} = ReLU( u^k - α_k * (W u^k + β) )       (Eq. 7)

    where:
      - W ∈ R^{D×D} is learnable (implemented via nn.Linear with bias)
      - α_k ≥ 0 is a learnable step size for each layer k
      - β is included as the linear bias term
    """

    def __init__(self, dim: int = 256, steps: int = 5):
        super().__init__()
        self.dim = dim
        self.steps = steps

        # W and β via a linear layer
        self.linear = nn.Linear(dim, dim, bias=True)

        # Raw step sizes; we enforce α_k >= 0 via softplus
        self.alpha_raw = nn.Parameter(torch.zeros(steps))

    def forward(self, u0: torch.Tensor) -> torch.Tensor:
        """
        u0: (B, D) initial vector (u^(0))

        returns:
          u_K: (B, D) after K unfolding steps
        """
        u = u0
        # Enforce non-negative step sizes
        alphas = F.softplus(self.alpha_raw)  # (K,)

        for k in range(self.steps):
            step = self.linear(u)              # (B, D)
            u = u - alphas[k] * step
            u = F.relu(u)                      # non-negativity constraint

        return u  # u^(K)


# ---------------------------------------------------------------------------
#  Manifold Attention + Gated Fusion (Sec. II-D, Eqs. 8–11)
# ---------------------------------------------------------------------------

class ManifoldAttentionGatedFusion(nn.Module):
    """
    Manifold attention with row-normalized W_m and gated fusion.

    Given:
      Z_f: (B, 256, L)
      s:   (B, 256)   # skip vector
      u_K: (B, 256)   # unfolded output

    Steps (paper):

      1) Temporal averaging to form manifold anchor:
         z_bar = mean over L -> (B, 256)

      2) Row-normalized W_m (Eq. 8):
         W_m_tilde = diag( ||W_m||_2,rows )^{-1} W_m

      3) Project to query vector q (Eq. 9):
         q = z_bar W_m_tilde ∈ R^{B×256}

      4) Bottleneck MLP (Eq. 10):
         α = σ( W2 ReLU(W1 q^T + b1) + b2 ) ∈ (0,1)^{B×256}

      5) Gated fusion (Eq. 11):
         h_fused = α ⊙ p + (1 − α) ⊙ s,
         where here we set p := u_K (unfolding projection).
    """

    def __init__(self, dim: int = 256, bottleneck_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.bottleneck_dim = bottleneck_dim

        # W_m ∈ R^{256×256}, row-normalized at each forward pass
        self.Wm = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.Wm)

        # Bottleneck MLP: 256 -> R -> 256
        self.mlp1 = nn.Linear(dim, bottleneck_dim)
        self.mlp2 = nn.Linear(bottleneck_dim, dim)

    def forward(
        self,
        Z_f: torch.Tensor,  # (B, 256, L)
        s: torch.Tensor,    # (B, 256)
        u_K: torch.Tensor,  # (B, 256)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        returns:
          h_fused: (B, 256)
          alpha:   (B, 256)  # attention weights
        """
        B, D, L = Z_f.shape
        assert D == self.dim

        # 1) Temporal averaging: z_bar ∈ R^{B×256}
        z_bar = Z_f.mean(dim=-1)  # (B, 256)

        # 2) Row-normalize W_m (Eq. 8)
        #    Wm_tilde[i,:] = Wm[i,:] / ||Wm[i,:]||_2
        row_norms = self.Wm.norm(p=2, dim=1, keepdim=True) + 1e-6
        Wm_tilde = self.Wm / row_norms  # (256, 256)

        # 3) q = z_bar Wm_tilde (Eq. 9)
        q = z_bar @ Wm_tilde  # (B, 256)

        # 4) Bottleneck MLP (Eq. 10)
        #    α = σ( W2 ReLU(W1 q^T + b1) + b2 )
        h = self.mlp1(q)                  # (B, R)
        h = F.relu(h, inplace=True)
        h = self.mlp2(h)                  # (B, 256)
        alpha = torch.sigmoid(h)          # (B, 256), in (0,1)

        # 5) Gated fusion (Eq. 11), using p = u_K
        p = u_K
        h_fused = alpha * p + (1.0 - alpha) * s  # (B, 256)

        return h_fused, alpha


# ---------------------------------------------------------------------------
#  Multi-Head Self-Attention Refinement (Sec. II-E, Eq. 12)
# ---------------------------------------------------------------------------

class MHSABlock(nn.Module):
    """
    One MHSA refinement block.

    We treat the SADU features as a length-2 sequence:
      token 0: fused feature (h_fused)
      token 1: skip feature (s)

    Each block:
      - LayerNorm
      - MultiHeadAttention (H heads, embed_dim = dim)
      - Residual connection
      - LayerNorm + 2-layer FFN (dim -> 4*dim -> dim)
      - Residual connection
    """

    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,  # x: (B, N, D)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, D)
        """
        # MHSA with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # FFN with residual
        x_norm2 = self.norm2(x)
        x = x + self.ffn(x_norm2)

        return x


class MHSARefinement(nn.Module):
    """
    Cascade of three MHSA blocks as in Sec. II-E.

    For WiMANS, the paper states that using a constant number of heads
    in all three blocks (H = 2) works well, whereas MM-Fi uses a
    coarse-to-fine schedule {16, 8, 4} with different dropouts.

    Here we implement the WiMANS configuration by default:
      - 3 blocks
      - 2 heads each
      - dropout = 0 (WiMANS) unless specified otherwise
    """

    def __init__(
        self,
        dim: int = 256,
        num_blocks: int = 3,
        num_heads: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MHSABlock(dim=dim, num_heads=num_heads, dropout=dropout)
             for _ in range(num_blocks)]
        )

    def forward(
        self,
        h_fused: torch.Tensor,  # (B, 256)
        s: torch.Tensor,        # (B, 256)
    ) -> torch.Tensor:
        """
        Construct a 2-token sequence [h_fused, s] and refine it via MHSA.

        returns:
          h_out: (B, 256)  # pooled representation after MHSA
        """
        # tokens: (B, 2, 256)
        tokens = torch.stack([h_fused, s], dim=1)

        for blk in self.blocks:
            tokens = blk(tokens)  # (B, 2, 256)

        # Pool over the 2 tokens (mean pooling)
        h_out = tokens.mean(dim=1)  # (B, 256)
        return h_out


# ---------------------------------------------------------------------------
#  Multi-task per-user heads (activity, identity, location)
# ---------------------------------------------------------------------------

class PerUserHead(nn.Module):
    """
    Per-user classification head.

    Maps a global feature h ∈ R^{B×D} to per-slot logits:
      logits ∈ R^{B×U×C}

    Implementation: MLP with 1 hidden layer (optional) followed by
    a final linear that outputs U*C logits which are then reshaped.
    """

    def __init__(
        self,
        feature_dim: int,
        max_users: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.max_users = max_users
        self.num_classes = num_classes

        if hidden_dim is None:
            hidden_dim = feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, max_users * num_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, D)

        returns:
          logits: (B, max_users, num_classes)
        """
        B, D = h.shape
        out = self.mlp(h)                       # (B, U*C)
        out = out.view(B, self.max_users, self.num_classes)
        return out


# ---------------------------------------------------------------------------
#  Full SADU model (WiMANS configuration)
# ---------------------------------------------------------------------------

class SADUWiMANSFull(nn.Module):
    """
    Full SADU model for WiMANS with all components:

      - MultiScaleDWSeparableCNN
      - DeepUnfoldingModule
      - ManifoldAttentionGatedFusion
      - MHSARefinement
      - Multi-task per-user heads:
          * activity
          * identity
          * location

    Args (common WiMANS config):
      in_channels        = 270  (3x3x30 CSI subcarriers flattened)
      max_users          = 5
      num_activities     = 9    (depends on your annotation)
      num_identities     = N_id (set from dataset)
      num_locations      = N_loc (e.g., grid cells)
    """

    def __init__(
        self,
        in_channels: int = 270,
        max_users: int = 5,
        num_activities: int = 9,
        num_identities: int = 16,
        num_locations: int = 15,
        # Front-end CNN settings
        branch_channels: int = 32,
        kernels: Tuple[int, int, int] = (128, 64, 32),
        strides: Tuple[int, int, int] = (8, 4, 2),
        # Deep unfolding
        unfold_steps: int = 5,
        # Manifold attention
        bottleneck_dim: int = 64,
        # MHSA refinement
        mhsa_blocks: int = 3,
        mhsa_heads: int = 2,
        mhsa_dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.max_users = max_users
        self.num_activities = num_activities
        self.num_identities = num_identities
        self.num_locations = num_locations

        # 1) Multi-scale depthwise separable CNN
        self.frontend = MultiScaleDWSeparableCNN(
            in_channels=in_channels,
            branch_channels=branch_channels,
            kernels=kernels,
            strides=strides,
        )

        # 2) Deep unfolding module
        self.deep_unfold = DeepUnfoldingModule(dim=256, steps=unfold_steps)

        # 3) Manifold attention + gated fusion
        self.manifold_fusion = ManifoldAttentionGatedFusion(
            dim=256,
            bottleneck_dim=bottleneck_dim,
        )

        # 4) Enhanced MHSA refinement
        self.mhsa_refine = MHSARefinement(
            dim=256,
            num_blocks=mhsa_blocks,
            num_heads=mhsa_heads,
            dropout=mhsa_dropout,
        )

        # 5) Multi-task per-user heads
        self.activity_head = PerUserHead(
            feature_dim=256,
            max_users=max_users,
            num_classes=num_activities,
        )
        self.identity_head = PerUserHead(
            feature_dim=256,
            max_users=max_users,
            num_classes=num_identities,
        )
        self.location_head = PerUserHead(
            feature_dim=256,
            max_users=max_users,
            num_classes=num_locations,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, in_channels, T)

        returns dict with:
          - Z_f:              (B, 256, L)
          - s:                (B, 256)
          - u0:               (B, 256)
          - u_K:              (B, 256)
          - alpha_manifold:   (B, 256)
          - h_fused:          (B, 256)
          - h_refined:        (B, 256)
          - activity_logits:  (B, max_users, num_activities)
          - identity_logits:  (B, max_users, num_identities)
          - location_logits:  (B, max_users, num_locations)
        """

        # 1) Multi-scale depthwise CNN front-end
        Z_f, s, u0 = self.frontend(x)        # Z_f: (B, 256, L), s/u0: (B, 256)

        # 2) Deep unfolding refinement
        u_K = self.deep_unfold(u0)           # (B, 256)

        # 3) Manifold attention + gated fusion
        h_fused, alpha = self.manifold_fusion(Z_f, s, u_K)  # (B, 256), (B, 256)

        # 4) MHSA refinement over tokens [h_fused, s]
        h_refined = self.mhsa_refine(h_fused, s)            # (B, 256)

        # 5) Multi-task heads (per-user)
        activity_logits = self.activity_head(h_refined)     # (B, U, C_act)
        identity_logits = self.identity_head(h_refined)     # (B, U, C_id)
        location_logits = self.location_head(h_refined)     # (B, U, C_loc)

        return {
            "Z_f": Z_f,
            "s": s,
            "u0": u0,
            "u_K": u_K,
            "alpha_manifold": alpha,
            "h_fused": h_fused,
            "h_refined": h_refined,
            "activity_logits": activity_logits,
            "identity_logits": identity_logits,
            "location_logits": location_logits,
        }


# ---------------------------------------------------------------------------
#  Backwards-compat alias (if your code expects models.sadu.SADUWiMANS)
# ---------------------------------------------------------------------------

# If you want to drop this file in and keep the old import:
#
#   from models.sadu_full import SADUWiMANSFull as SADUWiMANS
#
# or you can re-export it here:
#
# class SADUWiMANS(SADUWiMANSFull):
#     pass
