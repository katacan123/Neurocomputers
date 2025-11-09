# train_amp_count.py
import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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

def split_env_ids(df, test_env_key="E3"):
    envs = sorted(df["environment"].unique().tolist())
    test_envs = [e for e in envs if str(test_env_key) in str(e)]
    if not test_envs: test_envs = [envs[-1]]
    trainval_envs = [e for e in envs if e not in test_envs]

    idx_trainval = df.index[df["environment"].isin(trainval_envs)].tolist()
    idx_test = df.index[df["environment"].isin(test_envs)].tolist()
    random.shuffle(idx_trainval)
    n_val = max(1, int(0.1 * len(idx_trainval)))
    idx_val = idx_trainval[:n_val]
    idx_train = idx_trainval[n_val:]

    return (
        df.loc[idx_train, "label"].tolist(),
        df.loc[idx_val,   "label"].tolist(),
        df.loc[idx_test,  "label"].tolist(),
    )

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

    train_ids, val_ids, test_ids = split_env_ids(df, test_env_key="E3")

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
