# model_amp_count.py
import torch
import torch.nn as nn

class AmpCountNet(nn.Module):
    """
    (B,T,D) amplitude -> Conv1d temporal encoder (InstanceNorm1d + Dropout)
                      -> GlobalAvgPool -> Linear -> count logits
    """
    def __init__(self, in_dim, num_counts=6, p_drop=0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_dim, 128, kernel_size=7, padding=3),
            nn.InstanceNorm1d(128, affine=True),
            nn.ReLU(), nn.Dropout(p_drop),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=7, padding=3),
            nn.InstanceNorm1d(128, affine=True),
            nn.ReLU(), nn.Dropout(p_drop),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.InstanceNorm1d(256, affine=True),
            nn.ReLU(), nn.Dropout(p_drop),
        )
        self.gpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, num_counts)
        )

    def forward(self, x):
        # x: (B,T,D)
        x = x.permute(0,2,1)            # (B,D,T)
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.gpool(h).squeeze(-1)   # (B,256)
        return self.head(h)             # (B,num_counts)
