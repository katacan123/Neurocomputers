# models/mdc.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDC(nn.Module):
    """
    Multi-scale Dilated Convolution + Downsampling module (MDC)
    Implements F_MDC(A) as in the WiMUAR paper.

    Input:
        x: (B, C_in=270, T=3000)

    Output:
        x_out: (B, C_out=256, T_out≈22)
    """

    def __init__(self, in_channels: int = 270, out_channels: int = 256):
        super().__init__()

        # --- Stage 1: Multi-scale dilated conv branches ---
        # 4 branches, 64 channels each, kernel=3, dilations 1/2/4/8
        self.branch1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1, dilation=1)
        self.branch2 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=2, dilation=2)
        self.branch3 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=4, dilation=4)
        self.branch4 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=8, dilation=8)

        # fuse branches to 256 channels
        self.fuse = nn.Conv1d(64 * 4, out_channels, kernel_size=1)
        self.bn_fuse = nn.BatchNorm1d(out_channels)

        # --- Stage 3: CNN downsampling stack ---
        # shapes roughly (3000 -> ~360 -> ~75 -> ~22)
        self.conv_ds1 = nn.Conv1d(out_channels, 64, kernel_size=128, stride=8)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv_ds2 = nn.Conv1d(64, 128, kernel_size=64, stride=4)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv_ds3 = nn.Conv1d(128, out_channels, kernel_size=32, stride=2)
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.leaky_relu = nn.LeakyReLU(0.01)

        self._init_weights()

    def _init_weights(self):
        # Xavier init to match paper’s typical setup
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T) -> x_out: (B, 256, T_out)
        """
        # --- Multi-scale dilated branches ---
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        x = torch.cat([b1, b2, b3, b4], dim=1)  # (B, 256, T)
        x = self.fuse(x)                        # (B, 256, T)

        # --- Gated activation: tanh * sigmoid (Stage 2 in paper) ---
        x = self.bn_fuse(x)
        g = torch.sigmoid(x)
        f = torch.tanh(x)
        x = g * f

        # --- Downsampling CNN stack (Stage 3) ---
        x = self.conv_ds1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.conv_ds2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        x = self.conv_ds3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)

        # shape should now be (B, 256, ~22)
        return x
