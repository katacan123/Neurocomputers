# model_csi.py
import torch
import torch.nn as nn

class CSI1DCNNCount(nn.Module):
    """
    1D CNN for CSI human counting (0..5 people)
    Input: (B, C, T) where C can be 270 (amp) or >270 (amp+DS)
    Output: (B, 6)
    """
    def __init__(self, in_channels=270, n_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=9, padding=4),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=9, padding=4),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        # x: (B, C, T)
        x = self.features(x)
        x = self.gap(x).squeeze(-1)  # (B, 256)
        x = self.fc(x)
        logits = self.classifier(x)  # (B, 6)
        return logits
