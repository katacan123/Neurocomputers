# model_csi.py
import torch
import torch.nn as nn

class CSI1DCNNCount(nn.Module):
    """
    1D CNN for CSI human counting (0..5 people)
    Input: (B, 270, 3000)
    Output: (B, 6)
    """
    def __init__(self, in_channels=270, n_classes=6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 3000 -> 1500

            nn.Conv1d(64, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 1500 -> 750

            nn.Conv1d(128, 256, kernel_size=9, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 750 -> 375
        )

        self.gap = nn.AdaptiveAvgPool1d(1)  # (B,256,1) -> (B,256)
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