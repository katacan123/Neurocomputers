import torch
import torch.nn as nn
import torch.nn.functional as F


class ChSE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.fc1 = nn.Conv1d(c, c // r, 1)
        self.fc2 = nn.Conv1d(c // r, c, 1)
    def forward(self, x):
        w = x.mean(dim=-1, keepdim=True)
        w = F.silu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class TCNBlock(nn.Module):
    def __init__(self, c, dilation=1, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c, c, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(c),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Conv1d(c, c, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(c),
            ChSE(c)
        )
    def forward(self, x):
        return F.silu(x + self.net(x))


class CSI1DTCNCount(nn.Module):
    def __init__(self, in_channels=540, n_classes=6, depth=5, stem_channels=256, p_drop=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, 3, padding=1),
            nn.BatchNorm1d(stem_channels),
            nn.SiLU(),
        )
        blocks, dil = [], 1
        for _ in range(depth):
            blocks.append(TCNBlock(stem_channels, dilation=dil, p_drop=p_drop))
            dil *= 2
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(stem_channels, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.tcn(x)
        return self.head(x)
