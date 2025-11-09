import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from prepare_phase_data import CSIPhaseDataset, compute_channel_stats
from model_csi import CSI1DTCNCount

# ---------- CONFIG ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "training_dataset"))
CSV_PATH = os.path.join(DATA_DIR, "annotation.csv")

BATCH_SIZE = 24
EPOCHS = 15
LR = 3e-4
MAX_LEN = 3000
USE_BAND = 5.0           # 5.0 veya None
USE_DPHASE = True
USE_AMP_FEAT = True
AUGMENT = True

E1_NAME = "classroom"
E2_NAME = "meeting_room"
E3_NAME = "empty_room"
SEED = 42

# Early stop & checkpoint
PATIENCE = 8
CKPT_DIR = Path("./checkpoints"); CKPT_DIR.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def split_e1e2_train_e2val_e3test(
    df: pd.DataFrame,
    env_col: str = "environment",
    id_col: str = "label",
    y_col: str = "number_of_users",
    e1_name: str = E1_NAME,
    e2_name: str = E2_NAME,
    e3_name: str = E3_NAME,
    val_ratio: float = 0.2,
    seed: int = SEED,
):
    df_e1 = df[df[env_col] == e1_name]
    df_e2 = df[df[env_col] == e2_name]
    df_e3 = df[df[env_col] == e3_name]

    ids = df_e2[id_col].tolist()
    y   = df_e2[y_col].to_numpy()
    strat = y if (pd.Series(y).value_counts().min() >= 2) else None

    e2_train_ids, e2_val_ids = train_test_split(
        ids, test_size=val_ratio, random_state=seed, shuffle=True, stratify=strat
    )

    train_df = pd.concat(
        [df_e1, df_e2[df_e2[id_col].isin(e2_train_ids)]],
        ignore_index=True
    ).reset_index(drop=True)

    val_df   = df_e2[df_e2[id_col].isin(e2_val_ids)].reset_index(drop=True)
    test_df  = df_e3.reset_index(drop=True)

    print(f"[split] Train: {len(train_df)}  (E1={len(df_e1)}, E2_train={len(e2_train_ids)})")
    print(f"[split]  Val : {len(val_df)}   (E2_val={len(e2_val_ids)})")
    print(f"[split]  Test: {len(test_df)}  (E3 all)")
    return train_df, val_df, test_df

@torch.no_grad()
def compute_bg_means_per_env(df, env_names, csv_path, data_root, wifi_band, max_len):
    """Return dict: env -> (270,) mean computed from 0-human samples only."""
    bg_means = {}
    for env in env_names:
        rows = df[(df["environment"] == env) & (df["number_of_users"] == 0)]
        if len(rows) == 0:
            continue
        tmp_ds = CSIPhaseDataset(
            csv_path=csv_path, data_root=data_root,
            ids=rows["label"].tolist(), wifi_band=wifi_band,
            max_len=max_len, use_dphase=False, use_amp=False,
            augment=False, enable_cache=False
        )
        tmp_ld = DataLoader(tmp_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
        s = torch.zeros(270); c = 0
        for x, _, _ in tmp_ld:  # x: (B, 270, T)
            s += x.mean(dim=2).mean(dim=0)
            c += 1
        if c > 0:
            bg_means[env] = (s / c).numpy().astype(np.float32)
    return bg_means

def get_class_weights(df):
    counts = df["number_of_users"].value_counts().reindex(range(6), fill_value=0).sort_index()
    freq = counts.to_numpy(dtype=np.float32) + 1e-6
    inv = 1.0 / freq
    w = inv / inv.sum() * 6.0
    return torch.tensor(w, dtype=torch.float32)

def train_one_epoch(model, loader, optimizer, device, loss_fn, scaler):
    model.train()
    total_loss = 0.0; total_correct = 0; total = 0
    for x, y_count, _ in tqdm(loader, desc="train", ncols=100):
        x = x.to(device, non_blocking=True)
        y_count = y_count.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = loss_fn(logits, y_count)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y_count).sum().item()
        total += y_count.size(0)
    return total_loss / total, total_correct / total

@torch.no_grad()
def eval_one_epoch(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0; total_correct = 0; total = 0
    for x, y_count, _ in tqdm(loader, desc="eval", ncols=100):
        x = x.to(device, non_blocking=True)
        y_count = y_count.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(x)
            loss = loss_fn(logits, y_count)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y_count).sum().item()
        total += y_count.size(0)
    return total_loss / total, total_correct / total

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) CSV & band filtre
    df = pd.read_csv(CSV_PATH)
    if USE_BAND is not None:
        df = df[df["wifi_band"] == USE_BAND].reset_index(drop=True)

    # 2) env split (Train = E1 + part of E2; Val = E2-only; Test = E3)
    train_df, val_df, test_df = split_e1e2_train_e2val_e3test(df, val_ratio=0.2, seed=SEED)
    print(f"Train samples: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 3) background means ONLY from TRAIN ENVS (E1 & E2), and only 0-human rows
    bg_means = compute_bg_means_per_env(
        df, env_names=[E1_NAME, E2_NAME],
        csv_path=CSV_PATH, data_root=DATA_DIR, wifi_band=USE_BAND, max_len=MAX_LEN
    )

    # 4) kanal istatistikleri (augment kapalı) — use same bg_means
    train_ds_noaug = CSIPhaseDataset(
        csv_path=CSV_PATH, data_root=DATA_DIR,
        ids=train_df["label"].tolist(), wifi_band=USE_BAND,
        max_len=MAX_LEN, use_dphase=USE_DPHASE, use_amp=USE_AMP_FEAT,
        augment=False, background_phase_mean=bg_means, enable_cache=False
    )
    mean, std = compute_channel_stats(train_ds_noaug, batch_size=16, num_workers=0)
    print("Channel mean/std computed:", mean.shape, std.shape)

    # 5) gerçek dataset'ler
    common = dict(
        csv_path=CSV_PATH, data_root=DATA_DIR, max_len=MAX_LEN,
        use_dphase=USE_DPHASE, use_amp=USE_AMP_FEAT,
        mean=mean, std=std, background_phase_mean=bg_means,
        enable_cache=True
    )
    train_ds = CSIPhaseDataset(ids=train_df["label"].tolist(), wifi_band=USE_BAND,
                               augment=AUGMENT, seed=SEED, **common)
    val_ds   = CSIPhaseDataset(ids=val_df["label"].tolist(),   wifi_band=USE_BAND,
                               augment=False, seed=SEED, **common)
    test_ds  = CSIPhaseDataset(ids=test_df["label"].tolist(),  wifi_band=USE_BAND,
                               augment=False, seed=SEED, **common)

    # Windows-friendly loader settings
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, prefetch_factor=6, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, prefetch_factor=4, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, prefetch_factor=4, persistent_workers=True
    )

    # 6) model
    in_channels = 270 + (270 if USE_DPHASE else 0) + (270 if USE_AMP_FEAT else 0)
    model = CSI1DTCNCount(in_channels=in_channels, n_classes=6, depth=5).to(device)

    # 7) loss + optimizer + scheduler
    class_weights = get_class_weights(train_df).to(device)
    try:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    except TypeError:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, threshold=1e-3, verbose=True
    )
    scaler = torch.cuda.amp.GradScaler()

    # 8) train loop (early stopping + ckpt)
    best_val_loss = float('inf'); best_state = None; epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, loss_fn, scaler)
        val_loss, val_acc     = eval_one_epoch(model, val_loader, device, loss_fn)

        scheduler.step(val_loss)

        # checkpoint (per-epoch)
        torch.save(model.state_dict(), CKPT_DIR / f"epoch_{epoch:02d}.pth")

        print(f"Epoch {epoch:02d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

        # early stop takip
        if val_loss + 1e-6 < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, CKPT_DIR / "best_val_loss.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improve {PATIENCE})")
                break

    # 9) test (best by val loss)
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = eval_one_epoch(model, test_loader, device, loss_fn)
    print(f"\n=== FINAL TEST ON '{E3_NAME}' (band={USE_BAND}) ===")
    print(f"test loss {test_loss:.4f} | test acc {test_acc:.3f}")

    # 10) save final name
    suffix = f"{' _dphase' if USE_DPHASE else ''}{'_amp' if USE_AMP_FEAT else ''}_{str(USE_BAND).replace('.','')}"
    out_name = f"csi_count_phase{suffix}.pth".replace(" ", "")
    torch.save(model.state_dict(), out_name)
    print("Saved to", out_name)

if __name__ == "__main__":
    main()
