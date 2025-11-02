# train.py
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from prepare_amp_data import CSIAmpDataset   # your class
from model_csi import CSI1DCNNCount

# ---------- CONFIG ----------
DATA_DIR = r"D:\ee543\data"     # change to your path
CSV_PATH = os.path.join(DATA_DIR, "annotation.csv")
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3
MAX_LEN = 3000
USE_BAND = 2.4   # set to 5.0 or None

def split_by_env(df, train_env="classroom", val_env=None):
    """
    HW wants env/location generalization.
    If val_env is given, we do env-based split.
    Otherwise fallback to 80/20 random.
    """
    if val_env is not None:
        train_df = df[df["environment"] == train_env].reset_index(drop=True)
        val_df = df[df["environment"] == val_env].reset_index(drop=True)
    else:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n = len(df)
        n_train = int(0.8 * n)
        train_df = df.iloc[:n_train].reset_index(drop=True)
        val_df = df.iloc[n_train:].reset_index(drop=True)
    return train_df, val_df

def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y_count in loader:
        x = x.to(device)          # (B, 270, 3000)
        y_count = y_count.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y_count)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y_count).sum().item()
        total += y_count.size(0)

    return total_loss / total, total_correct / total

@torch.no_grad()
def eval_one_epoch(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y_count in loader:
        x = x.to(device)
        y_count = y_count.to(device)

        logits = model(x)
        loss = loss_fn(logits, y_count)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y_count).sum().item()
        total += y_count.size(0)

    return total_loss / total, total_correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) read CSV
    df = pd.read_csv(CSV_PATH)

    # 2) filter by wifi band
    if USE_BAND is not None:
        df = df[df["wifi_band"] == USE_BAND].reset_index(drop=True)

    # 3) split
    # change to split_by_env(df, train_env="classroom", val_env="meeting_room") if you know env names
    train_df, val_df = split_by_env(df, val_env=None)

    # 4) datasets
    train_ds = CSIAmpDataset(
        csv_path=CSV_PATH,
        data_root=DATA_DIR,
        max_len=MAX_LEN,
        ids=train_df["label"].tolist(),
        wifi_band=USE_BAND,
    )
    val_ds = CSIAmpDataset(
        csv_path=CSV_PATH,
        data_root=DATA_DIR,
        max_len=MAX_LEN,
        ids=val_df["label"].tolist(),
        wifi_band=USE_BAND,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    # 5) model
    model = CSI1DCNNCount(in_channels=270, n_classes=6).to(device)

    # 6) loss + optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 7) train
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, loss_fn
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, device, loss_fn
        )
        print(f"Epoch {epoch:02d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

    # 8) save
    torch.save(model.state_dict(), "csi_count_amp_24ghz.pth")
    print("Saved to csi_count_amp_24ghz.pth")

if __name__ == "__main__":
    main()
