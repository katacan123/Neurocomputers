# train.py
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from prepare_amp_data import CSIAmpDataset   # your dataset class
from model_csi import CSI1DCNNCount         # your model class

# ---------- CONFIG ----------
DATA_DIR = r"D:\ee543\data"     # change to your path
CSV_PATH = os.path.join(DATA_DIR, "annotation.csv")
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3
MAX_LEN = 3000
USE_BAND = 2.4   # set to 5.0 or None

TRAIN_ENVS = ("classroom", "meeting_room")
TEST_ENV = "empty_room"   # check exact name in your csv! maybe "empty" or "emptyroom"


def split_env_with_val(df, train_envs, test_env, val_ratio=0.2):
    """
    1) train+val = rows whose environment is in train_envs
    2) test       = rows whose environment == test_env
    3) train+val are split randomly (NOT by location) just to get a val set
    """
    train_all = df[df["environment"].isin(train_envs)].reset_index(drop=True)
    test_df = df[df["environment"] == test_env].reset_index(drop=True)

    # random split inside train_all
    train_df, val_df = train_test_split(
        train_all, test_size=val_ratio, random_state=42, shuffle=True
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df, test_df


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

    # 3) split by environment
    train_df, val_df, test_df = split_env_with_val(
        df,
        train_envs=TRAIN_ENVS,
        test_env=TEST_ENV,
        val_ratio=0.2,
    )

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples:   {len(val_df)}")
    print(f"Test samples:  {len(test_df)}")

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
    test_ds = CSIAmpDataset(
        csv_path=CSV_PATH,
        data_root=DATA_DIR,
        max_len=MAX_LEN,
        ids=test_df["label"].tolist(),
        wifi_band=USE_BAND,
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
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

    # 8) AUTO TEST AFTER TRAIN
    test_loss, test_acc = eval_one_epoch(
        model, test_loader, device, loss_fn
    )
    print(f"\n=== FINAL TEST ON '{TEST_ENV}' (band={USE_BAND}) ===")
    print(f"test loss {test_loss:.4f} | test acc {test_acc:.3f}")

    # 9) save
    torch.save(model.state_dict(), "csi_count_amp_24ghz.pth")
    print("Saved to csi_count_amp_24ghz.pth")


if __name__ == "__main__":
    main()
