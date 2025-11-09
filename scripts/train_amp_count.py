# train_amp_count.py
import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from data_amp import AmpCSIDataset
from model_amp_count import AmpCountNet

# ---- PATHS ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "training_dataset"))
AMP_DIR   = os.path.join(DATASET_BASE_DIR, "wifi_csi", "amp")
ANNOT_CSV = os.path.join(DATASET_BASE_DIR, "annotation.csv")

# ---- HYPERPARAMS ----
SEED = 42
BATCH = 8
EPOCHS = 30
LR = 2e-4
TARGET_LEN = 3000
USE_SVD = True
USE_LOWPASS = True
N_PCA = 5
P_DROP = 0.3
EARLY_STOP_PATIENCE = 7
USE_AMP = True

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def split_env_e1_train_e2_val_e3_test(df, val_ratio=0.2, random_state=42):
    """
    E1 -> %100 train
    E2 -> stratified split: train (%1 - val_ratio), val (%val_ratio)
    E3 -> %100 test
    Stratify kriteri: number_of_users (count dağılımını korur)
    """
    # Filtreler
    df_e1 = df[df["environment"] == "classroom"]
    df_e2 = df[df["environment"] == "meeting_room"]
    df_e3 = df[df["environment"] == "empty_room"]

    # E1 tamamen train
    train_labels = df_e1["label"].tolist()

    # E2'yi stratified train/val böl
    val_labels = []
    if len(df_e2) > 0:
        y = df_e2["number_of_users"].astype(int).values
        idx = df_e2.index.values
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
        tr_idx, vl_idx = next(splitter.split(np.zeros_like(y), y))
        e2_train_idx = idx[tr_idx]
        e2_val_idx   = idx[vl_idx]
        train_labels += df.loc[e2_train_idx, "label"].tolist()
        val_labels    = df.loc[e2_val_idx,   "label"].tolist()

    # E3 tamamen test
    test_labels = df_e3["label"].tolist()

    # küçük kontrol çıktısı (isteğe bağlı)
    print(f"Split -> E1 train: {len(df_e1)} | E2 train: {len(train_labels)-len(df_e1)} | "
          f"E2 val: {len(val_labels)} | E3 test: {len(test_labels)}")

    return train_labels, val_labels, test_labels

def compute_class_weights(df, ids):
    sub = df[df["label"].isin(ids)]
    counts = sub["number_of_users"].value_counts().sort_index()
    maxc = counts.max()
    weights = torch.tensor([maxc / counts.get(i, 1) for i in range(int(df["number_of_users"].max())+1)], dtype=torch.float32)
    return weights

def make_loaders():
    df = pd.read_csv(ANNOT_CSV)
    for col in ["label","number_of_users","environment"]:
        assert col in df.columns, f"{col} missing in annotation.csv"

    # E1 train, E2 (kalan) val, E3 test
    train_ids, val_ids, test_ids = split_env_e1_train_e2_val_e3_test(df, val_ratio=0.2, random_state=42)

    train_ds = AmpCSIDataset(ANNOT_CSV, AMP_DIR, ids=train_ids,
                             target_len=TARGET_LEN, use_svd=USE_SVD, use_lowpass=USE_LOWPASS, n_pca=N_PCA)
    val_ds   = AmpCSIDataset(ANNOT_CSV, AMP_DIR, ids=val_ids,
                             target_len=TARGET_LEN, use_svd=USE_SVD, use_lowpass=USE_LOWPASS, n_pca=N_PCA)
    test_ds  = AmpCSIDataset(ANNOT_CSV, AMP_DIR, ids=test_ids,
                             target_len=TARGET_LEN, use_svd=USE_SVD, use_lowpass=USE_LOWPASS, n_pca=N_PCA)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, df, train_ids


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader, df, train_ids = make_loaders()

    x0, y0c, env0 = next(iter(train_loader))
    _, T, D = x0.shape
    num_counts = int(df["number_of_users"].max()) + 1

    model = AmpCountNet(in_dim=D, num_counts=num_counts, p_drop=P_DROP).to(device)
    cls_w = compute_class_weights(df, train_ids).to(device)
    crit = nn.CrossEntropyLoss(weight=cls_w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    def run_epoch(loader, train=True):
        model.train(train)
        total, loss_sum, correct = 0, 0.0, 0
        for x, y_cnt, env in tqdm(loader, ncols=100):
            x = x.to(device); y_cnt = y_cnt.to(device)
            if train:
                opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(x)
                loss = crit(logits, y_cnt)
            if train:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            bs = x.size(0); total += bs
            loss_sum += loss.item() * bs
            correct += (logits.argmax(1) == y_cnt).sum().item()
        return loss_sum/total, correct/total

    best_acc, patience = -1, 0
    for ep in range(1, EPOCHS+1):
        tr_loss, tr_acc = run_epoch(train_loader, True)
        vl_loss, vl_acc = run_epoch(val_loader, False)
        print(f"[{ep:02d}] train {tr_loss:.4f}/{tr_acc:.3f} | val {vl_loss:.4f}/{vl_acc:.3f}")
        if vl_acc > best_acc:
            best_acc = vl_acc; patience = 0
            torch.save(model.state_dict(), "amp_count_best.pt")
            print("  -> saved amp_count_best.pt")
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print("Early stopping triggered."); break

    model.load_state_dict(torch.load("amp_count_best.pt", map_location=device))
    ts_loss, ts_acc = run_epoch(test_loader, False)
    print(f"TEST (unseen env): loss={ts_loss:.4f} acc={ts_acc:.3f}")

if __name__ == "__main__":
    main()
