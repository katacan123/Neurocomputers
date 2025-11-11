import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from tqdm.auto import tqdm

from wimans_preprocess import (
    build_sample_table,
    WiMANSCountDataset,
    ENV_COL,
    BAND_COL,
)

from torch.amp import autocast      # keep autocast from here
from torch.amp import GradScaler         # GradScaler from new namespace
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

TRAIN_ENVS = ("empty_room", "classroom")
TEST_ENV = "meeting_room"       # unseen test environment

USE_BAND = None                  # "2.4", "5" or None

TARGET_T = 3000
DOWNSAMPLE_FACTOR = 2

BATCH_SIZE = 16
EPOCHS = 40
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MODEL (3D CNN + CAM + 2D CNN + LSTM)
# ============================================================

class ChannelAttention3D(nn.Module):
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
        avg_pool = F.adaptive_avg_pool3d(x, 1).view(B, C)
        max_pool = F.adaptive_max_pool3d(x, 1).view(B, C)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        attn = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1, 1)
        return x * attn


class Global3DBranch(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.dropout = nn.Dropout3d(p=0.3)
        self.cam = ChannelAttention3D(256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.conv4(x))
        x = self.dropout(x)

        x = self.cam(x)
        x = F.adaptive_avg_pool3d(x, 1).view(x.size(0), -1)  # (B, 256)
        return x


class Local2DTemporalBranch(nn.Module):
    def __init__(self, in_channels=1, lstm_hidden=128, lstm_layers=1):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2d = nn.MaxPool2d(kernel_size=2)
        self.dropout2d = nn.Dropout2d(p=0.3)

        self.fc1 = None
        self.fc2 = None
        self.lstm = None
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers

    def _init_fc_and_lstm(self, feat_map_shape):
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
        B, C_in, T, H, W = x3d.shape
        x_seq = x3d.permute(0, 2, 1, 3, 4)         # (B, T, C, H, W)
        x_2d = x_seq.reshape(B * T, C_in, H, W)    # (B*T, C, H, W)

        x_2d = F.relu(self.conv2d_1(x_2d))
        x_2d = self.pool2d(x_2d)

        x_2d = F.relu(self.conv2d_2(x_2d))
        x_2d = self.pool2d(x_2d)
        x_2d = self.dropout2d(x_2d)

        if self.fc1 is None or self.lstm is None:
            C_out, H_out, W_out = x_2d.shape[1:]
            self._init_fc_and_lstm((C_out, H_out, W_out))
            self.fc1.to(x_2d.device)
            self.fc2.to(x_2d.device)
            self.lstm.to(x_2d.device)

        x_2d = x_2d.view(B * T, -1)
        x_2d = F.relu(self.fc1(x_2d))
        x_2d = F.relu(self.fc2(x_2d))              # (B*T, 128)

        x_seq_feat = x_2d.view(B, T, 128)          # (B, T, 128)
        out, (h_n, c_n) = self.lstm(x_seq_feat)
        local_feat = h_n[-1]                       # (B, hidden)
        return local_feat


class CSIParallelCountNet(nn.Module):
    """
    STEM-like parallel network for user-count prediction.

    Input: X of shape (B, C_in, T)
    Internally reshaped to (B, 1, T, H, W) where H*W = C_in (or H=1, W=C_in).
    """
    def __init__(self, num_classes=6, in_channels=270, H: int = None, W: int = None):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        if H is not None and W is not None:
            assert H * W == in_channels, "H * W must equal in_channels"
            self.H, self.W = H, W
        else:
            # Default: try to use H=9 if possible; otherwise flatten into 1xC
            if in_channels % 9 == 0:
                self.H = 9
                self.W = in_channels // 9
            else:
                self.H = 1
                self.W = in_channels

        self.global_branch = Global3DBranch(in_channels=1)
        self.local_branch = Local2DTemporalBranch(in_channels=1, lstm_hidden=128)

        self.fc = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        B, C, T = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected C={self.in_channels}, got {C}")

        x_3d = x.view(B, 1, T, self.H, self.W)  # (B,1,T,H,W)

        g = self.global_branch(x_3d)            # (B,256)
        l = self.local_branch(x_3d)             # (B,128)

        z = torch.cat([g, l], dim=1)            # (B,384)
        logits = self.fc(z)                     # (B, num_classes)
        return logits


# ============================================================
# DATASET / DATALOADER SETUP (2 envs train+val, 1 env test)
# ============================================================

def get_dataloaders_two_envs(
    train_envs,
    test_env,
    use_band,
    target_T,
    downsample_factor,
    batch_size,
):
    df_all = build_sample_table()

    if ENV_COL not in df_all.columns:
        raise ValueError(f"ENV_COL '{ENV_COL}' not in annotations.csv")

    # Train/val environments
    df_trainval = df_all[df_all[ENV_COL].isin(train_envs)].reset_index(drop=True)
    if df_trainval.empty:
        raise ValueError(f"No samples found for train_envs={train_envs}")

    # Test (unseen env)
    df_test_meeting = df_all[df_all[ENV_COL] == test_env].reset_index(drop=True)
    if df_test_meeting.empty:
        print(f"[WARN] No samples found for TEST_ENV='{test_env}'.")
    else:
        print(f"[INFO] Meeting-room (unseen) samples: {len(df_test_meeting)}")

    # Band filtering
    if use_band is not None:
        target_val = float(use_band)
        band_trainval = pd.to_numeric(df_trainval[BAND_COL], errors="coerce")
        mask_tv = np.isclose(band_trainval, target_val)
        df_trainval = df_trainval[mask_tv].reset_index(drop=True)

        band_test = pd.to_numeric(df_test_meeting[BAND_COL], errors="coerce")
        mask_te = np.isclose(band_test, target_val)
        df_test_meeting = df_test_meeting[mask_te].reset_index(drop=True)

        print(
            f"[INFO] After band filter wifi_band≈{target_val}: "
            f"{len(df_trainval)} train+val samples, "
            f"{len(df_test_meeting)} meeting_room samples"
        )

    # Train/val split
    train_df, val_df = train_test_split(
        df_trainval,
        test_size=0.2,
        random_state=123,
        stratify=df_trainval["num_users"],
    )
    print(
        f"[INFO] Train+Val from envs={train_envs}: "
        f"{len(train_df)} train, {len(val_df)} val samples"
    )

    # Use DWT + Doppler here
    train_ds = WiMANSCountDataset(
        train_df,
        target_T=target_T,
        downsample_factor=downsample_factor,
        use_band=None,        # band already filtered
        use_dwt=True,
        use_doppler=True,
    )
    val_ds = WiMANSCountDataset(
        val_df,
        target_T=target_T,
        downsample_factor=downsample_factor,
        use_band=None,
        use_dwt=True,
        use_doppler=True,
    )
    test_meeting_ds = WiMANSCountDataset(
        df_test_meeting,
        target_T=target_T,
        downsample_factor=downsample_factor,
        use_band=None,
        use_dwt=True,
        use_doppler=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_meeting_loader = DataLoader(test_meeting_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"[INFO] Training DataLoader: {len(train_ds)} samples")
    print(f"[INFO] Validation DataLoader: {len(val_ds)} samples")
    print(f"[INFO] Meeting-room Test DataLoader: {len(test_meeting_ds)} samples")

    # Inspect shape after full preprocessing
    X_example, _ = next(iter(train_loader))
    print(f"[INFO] Example batch X shape: {X_example.shape}")  # (B, C_in, T_proc)

    return train_loader, val_loader, test_meeting_loader


# ============================================================
# TRAIN / EVAL UTILITIES
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = (device == "cuda") and (scaler is not None)

    for X, y in tqdm(loader, desc="Train", leave=False):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # ---- Forward + loss with mixed precision ----
        if use_amp:
            with autocast(device_type="cuda", dtype=torch.float16):
                logits = model(X)
                loss = distance_aware_loss(logits, y, alpha=1.0, beta=0.2)
        else:
            logits = model(X)
            loss = distance_aware_loss(logits, y, alpha=1.0, beta=0.2)

        # ---- Backward + optimizer step ----
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * X.size(0)

        # NOTE: this acc only makes sense if you're using classification (CrossEntropy).
        # If you're using MSELoss/regression, you may want a different metric here.
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



def eval_model(model, loader, criterion, device, desc="Eval"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in tqdm(loader, desc=desc, leave=False):
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = distance_aware_loss(logits, y, alpha=1.0, beta=0.2)

            running_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = correct / total if total > 0 else 0.0

    if all_preds:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
    else:
        all_preds = np.array([])
        all_labels = np.array([])

    return epoch_loss, epoch_acc, all_preds, all_labels

def distance_aware_loss(logits, targets, alpha=1.0, beta=0.2):
    """
    logits: (B, num_classes)
    targets: (B,) integer labels in [0, num_classes-1]
    alpha: weight for cross-entropy part
    beta:  weight for distance-aware regression part
    """
    # 1) Standard cross-entropy on logits + integer labels
    ce = F.cross_entropy(logits, targets)

    # 2) Distance-aware term on expected count
    num_classes = logits.size(1)
    device = logits.device

    # class indices: [0,1,2,3,4,5]
    class_indices = torch.arange(num_classes, device=device, dtype=torch.float32)

    # probabilities via softmax
    probs = F.softmax(logits, dim=1)                  # (B, C)

    # expected predicted count: sum_k p_k * k
    pred_count = (probs * class_indices.unsqueeze(0)).sum(dim=1)  # (B,)

    # true count as float
    true_count = targets.to(torch.float32)            # (B,)

    mse = F.mse_loss(pred_count, true_count)          # scalar

    return alpha * ce + beta * mse

# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Using device: {DEVICE}")

    train_loader, val_loader, test_meeting_loader = get_dataloaders_two_envs(
        train_envs=TRAIN_ENVS,
        test_env=TEST_ENV,
        use_band=USE_BAND,
        target_T=TARGET_T,
        downsample_factor=DOWNSAMPLE_FACTOR,
        batch_size=BATCH_SIZE,
    )

    # Infer number of channels after full preprocessing
    X_example, _ = next(iter(train_loader))
    C_in = X_example.shape[1]
    print(f"[INFO] Model input channels (C_in): {C_in}")

    # Model
    model = CSIParallelCountNet(num_classes=NUM_CLASSES, in_channels=C_in).to(DEVICE)
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.MSELoss()
    criterion = None
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    # ---- ADD THIS: GradScaler for mixed precision ----
    scaler = GradScaler("cuda") if DEVICE == "cuda" else None


    patience = 5
    patience_counter = 0
    best_val_acc = 0.0
    best_state = None

    try:
        for epoch in range(1, EPOCHS + 1):
            print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE, scaler=scaler
            )
            val_loss, val_acc, _, _ = eval_model(
                model, val_loader, criterion, DEVICE, desc="Val"
            )

            print(
                f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
                f"Val (empty_room+classroom): loss={val_loss:.4f}, acc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict()
                patience_counter = 0
                print(f"[INFO] New best val acc: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                print(f"[INFO] No improvement for {patience_counter} epochs.")

            # Early stopping
            if patience_counter >= patience:
                print(f"[EARLY STOP] Validation accuracy has not improved for {patience} epochs.")
                break

    except KeyboardInterrupt:
        print("\n[INTERRUPT] KeyboardInterrupt received. "
              "Stopping training early and proceeding to test evaluation...")
        
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_preds, test_labels = eval_model(
        model, test_meeting_loader, criterion, DEVICE, desc="Test (meeting_room)"
    )

    print("\n=== Unseen Environment Test: meeting_room ===")
    print(f"Test loss={test_loss:.4f}, Test acc={test_acc:.4f}")
    if test_labels.size > 0:
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(confusion_matrix(test_labels, test_preds))

        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, digits=4))
    else:
        print("[WARN] No meeting_room samples in test set.")


if __name__ == "__main__":
    main()
