import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Import preprocessing helpers and dataset
from wimans_preprocess import (
    build_sample_table,
    split_by_environment,
    WiMANSCountDataset,
    ENV_COL,
)

# ============================================================
# CONFIG
# ============================================================

# Environment to use: must match values in annotations.csv (e.g. "classroom")
ENV_NAME = "classroom"      # change to "meeting_room", "empty_room", or "all" as needed
USE_BAND = "5G"             # "5G", "2.4G", or None for both

TARGET_T = 3000             # pad/trim target time length before downsampling
DOWNSAMPLE_FACTOR = 5       # e.g. 1 (no downsample), 5 (keep every 5th sample)
BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 6             # 0..5 users

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 50            # print train progress every N batches


# ============================================================
# MODEL: CSI-PCNH style (3D CNN + CAM + 2D CNN + LSTM)
# ============================================================

class ChannelAttention3D(nn.Module):
    """
    Channel Attention Module (CAM) for 3D feature maps.
    Input: (B, C, D, H, W)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x):
        B, C, D, H, W = x.shape

        # Global average pooling
        avg_pool = F.adaptive_avg_pool3d(x, 1).view(B, C)
        # Global max pooling
        max_pool = F.adaptive_max_pool3d(x, 1).view(B, C)

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        attn = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1, 1)
        return x * attn


class Global3DBranch(nn.Module):
    """
    Global spatial-temporal branch:
      3D CNN stacks + Channel Attention + global pooling
    """
    def __init__(self, in_channels=1):
        super().__init__()
        # CSI-PCNH uses multiple 3D conv blocks; we keep it reasonably light
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.dropout = nn.Dropout3d(p=0.3)

        self.cam = ChannelAttention3D(256)

    def forward(self, x):
        # x: (B, 1, T, H, W)
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.dropout(x)

        # Channel Attention
        x = self.cam(x)

        # Global average pool over (D,H,W)
        x = F.adaptive_avg_pool3d(x, 1).view(x.size(0), -1)  # (B, 256)
        return x  # global feature vector


class Local2DTemporalBranch(nn.Module):
    """
    Local branch:
      For each time step:
        2D CNN -> FC -> sequence of features
      Then LSTM over sequence -> local temporal feature.
    """
    def __init__(self, in_channels=1, lstm_hidden=128, lstm_layers=1):
        super().__init__()
        # 2D CNN blocks
        self.conv2d_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2d = nn.MaxPool2d(kernel_size=2)
        self.dropout2d = nn.Dropout2d(p=0.3)

        # We'll infer flattened feature dim at runtime
        self.fc1 = None
        self.fc2 = None

        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.lstm = None  # will be created after we know feature dim

    def _init_fc_and_lstm(self, feat_map_shape):
        """
        Initialize FC and LSTM layers once we know the feature map shape:
        feat_map_shape: (C_out, H_out, W_out)
        """
        C_out, H_out, W_out = feat_map_shape
        feat_dim = C_out * H_out * W_out

        self.fc1 = nn.Linear(feat_dim, 256)
        self.fc2 = nn.Linear(256, 128)

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=self.lstm_hidden,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x3d):
        """
        x3d: (B, 1, T, H, W)
        We treat each time slice as a 2D "image":
          X_seq: (B, T, C=1, H, W)
        """
        B, C_in, T, H, W = x3d.shape

        # (B, 1, T, H, W) -> (B, T, 1, H, W)
        x_seq = x3d.permute(0, 2, 1, 3, 4)
        # Merge batch and time: (B*T, 1, H, W)
        x_2d = x_seq.reshape(B * T, C_in, H, W)

        # 2D CNN
        x_2d = F.relu(self.conv2d_1(x_2d))
        x_2d = self.pool2d(x_2d)

        x_2d = F.relu(self.conv2d_2(x_2d))
        x_2d = self.pool2d(x_2d)
        x_2d = self.dropout2d(x_2d)

        # Initialize FC + LSTM lazily based on the CNN output shape
        if self.fc1 is None or self.lstm is None:
            C_out, H_out, W_out = x_2d.shape[1:]
            self._init_fc_and_lstm((C_out, H_out, W_out))
            # Move newly created parameters to same device as x_2d
            self.fc1.to(x_2d.device)
            self.fc2.to(x_2d.device)
            self.lstm.to(x_2d.device)

        # Flatten spatial dims
        x_2d = x_2d.view(B * T, -1)            # (B*T, feat_dim)
        x_2d = F.relu(self.fc1(x_2d))          # (B*T, 256)
        x_2d = F.relu(self.fc2(x_2d))          # (B*T, 128)

        # Restore time dimension: (B, T, 128)
        x_seq_feat = x_2d.view(B, T, 128)

        # LSTM over time
        out, (h_n, c_n) = self.lstm(x_seq_feat)
        # Use last hidden state as local feature: (B, lstm_hidden)
        local_feat = h_n[-1]  # (num_layers * num_directions, B, hidden) -> (B, hidden)

        return local_feat


class CSIParallelCountNet(nn.Module):
    """
    CSI-PCNH-style parallel network for user-count prediction.

    Input: X of shape (B, 270, T)
    Internally reshaped to (B, 1, T, 9, 30).
    Output: logits for 6 classes (0..5 users).
    """
    def __init__(self, num_classes=6, T_to_HW=(9, 30)):
        super().__init__()
        self.num_classes = num_classes
        self.H, self.W = T_to_HW  # how we reshape 270 -> H*W

        self.global_branch = Global3DBranch(in_channels=1)
        self.local_branch = Local2DTemporalBranch(in_channels=1, lstm_hidden=128)

        # 256 from global branch + 128 from local branch
        self.fc = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        x: (B, 270, T)
        """
        B, C, T = x.shape
        if C != self.H * self.W:
            raise ValueError(
                f"Expected C={self.H*self.W} (H*W), got {C}. "
                "Adjust H,W in CSIParallelCountNet or reshape X differently."
            )

        # Reshape to 3D conv input: (B, 1, T, H, W)
        x_3d = x.view(B, 1, T, self.H, self.W)

        g = self.global_branch(x_3d)   # (B, 256)
        l = self.local_branch(x_3d)    # (B, 128)

        z = torch.cat([g, l], dim=1)   # (B, 384)
        logits = self.fc(z)            # (B, num_classes)
        return logits


# ============================================================
# TRAIN / EVAL HELPERS
# ============================================================

def get_dataloaders(
    env_name: str,
    use_band: str,
    target_T: int,
    downsample_factor: int,
    batch_size: int,
):
    """
    Build train/val/test DataLoaders for a given environment and band.
    - env_name: one of the environments (e.g. "classroom") or "all"
    """
    df_all = build_sample_table()
    env_splits = split_by_environment(df_all)

    if env_name == "all" or ENV_COL not in df_all.columns:
        # Use the first (and only) split if no env info
        key = list(env_splits.keys())[0]
        base_train_df = env_splits[key]["train"]
        test_df = env_splits[key]["test"]
    else:
        if env_name not in env_splits:
            raise ValueError(
                f"Environment '{env_name}' not found. Available: {list(env_splits.keys())}"
            )
        base_train_df = env_splits[env_name]["train"]
        test_df = env_splits[env_name]["test"]

    # Further split base_train_df into train / val
    train_df, val_df = train_test_split(
        base_train_df,
        test_size=0.2,
        random_state=123,
        stratify=base_train_df["num_users"],
    )

    train_ds = WiMANSCountDataset(
        train_df,
        target_T=target_T,
        downsample_factor=downsample_factor,
        use_band=use_band,
    )
    val_ds = WiMANSCountDataset(
        val_df,
        target_T=target_T,
        downsample_factor=downsample_factor,
        use_band=use_band,
    )
    test_ds = WiMANSCountDataset(
        test_df,
        target_T=target_T,
        downsample_factor=downsample_factor,
        use_band=use_band,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"[INFO] Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Infer T_proc from one batch (after downsampling)
    X_example, _ = next(iter(train_loader))
    _, C, T_proc = X_example.shape
    print(f"[INFO] Example batch X shape: {X_example.shape} (C={C}, T={T_proc})")

    return train_loader, val_loader, test_loader, T_proc


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (X, y) in enumerate(loader):
        X = X.to(device)  # (B, C, T)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

        if (i + 1) % PRINT_EVERY == 0:
            print(f"  [batch {i+1}/{len(loader)}] loss={loss.item():.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            running_loss += loss.item() * X.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return epoch_loss, epoch_acc, all_preds, all_labels


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Using device: {DEVICE}")

    # Data
    train_loader, val_loader, test_loader, T_proc = get_dataloaders(
        env_name=ENV_NAME,
        use_band=USE_BAND,
        target_T=TARGET_T,
        downsample_factor=DOWNSAMPLE_FACTOR,
        batch_size=BATCH_SIZE,
    )

    # Model (we know C=270; we use H=9, W=30 -> 9*30 = 270)
    model = CSIParallelCountNet(num_classes=NUM_CLASSES, T_to_HW=(9, 30)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc, _, _ = eval_model(
            model, val_loader, criterion, DEVICE
        )

        print(
            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            print(f"[INFO] New best val acc: {best_val_acc:.4f}")

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on test set
    test_loss, test_acc, test_preds, test_labels = eval_model(
        model, test_loader, criterion, DEVICE
    )
    print("\n=== Test Results ===")
    print(f"Test loss={test_loss:.4f}, Test acc={test_acc:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(test_labels, test_preds))

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, digits=4))


if __name__ == "__main__":
    main()
