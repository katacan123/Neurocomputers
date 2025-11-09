import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from prepare_phase_data import CSIPhaseDataset
from model_csi import CSI1DTCNCount

# ---------- PATHS ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "training_dataset"))
CSV_PATH = os.path.join(DATA_DIR, "annotation.csv")
CKPT_DIR = Path("./checkpoints")

# pick a checkpoint to load
BUNDLE_CKPT = CKPT_DIR / "best_bundle.pth"     # preferred (if you saved it)
FALLBACK_CKPT = CKPT_DIR / "best_val_loss.pth" # plain state_dict
ART_PATH = CKPT_DIR / "artifacts.npz"
CFG_PATH = CKPT_DIR / "config.json"

E3_NAME = "empty_room"   # test environment

def set_seed(seed=42):
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

@torch.no_grad()
def eval_one_epoch(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0; total_correct = 0; total = 0
    all_preds = []; all_targets = []
    for x, y_count, _ in tqdm(loader, desc="test", ncols=100):
        x = x.to(device, non_blocking=True)
        y = y_count.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += y.size(0)
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())
    test_loss = total_loss / max(total, 1)
    test_acc = total_correct / max(total, 1)
    all_preds = torch.cat(all_preds) if all_preds else torch.empty(0, dtype=torch.long)
    all_targets = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)
    return test_loss, test_acc, all_preds.numpy(), all_targets.numpy()

def load_artifacts_and_config():
    # try bundle first
    if BUNDLE_CKPT.exists():
        bundle = torch.load(BUNDLE_CKPT, map_location="cpu")
        mean = bundle.get("mean", None)
        std  = bundle.get("std", None)
        bg_store = bundle.get("bg_means", None)
        cfg = bundle.get("config", None)
        return mean, std, bg_store, cfg, bundle["state_dict"]

    # fallback to separate files
    assert ART_PATH.exists() and CFG_PATH.exists(), \
        "Missing artifacts.npz or config.json. Re-run train.py with the save patch."
    art = np.load(ART_PATH, allow_pickle=False)
    mean = art["mean"]; std = art["std"]

    # reconstruct bg dict (keys like 'bg__classroom', 'bg__meeting_room', or '_global')
    bg_store = {}
    for k in art.files:
        if k.startswith("bg__"):
            bg_store[k[4:]] = art[k]
        elif k == "bg__" or k == "_global":
            bg_store["_global"] = art[k]

    with open(CFG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # load plain state_dict
    state_dict = torch.load(FALLBACK_CKPT, map_location="cpu")
    return mean, std, bg_store if bg_store else None, cfg, state_dict

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Load CSV and filter TEST env (E3 only)
    df = pd.read_csv(CSV_PATH)
    # (optional) filter band to match training:
    with open(CFG_PATH, "r", encoding="utf-8") as f:
        tmp_cfg = json.load(f)
    use_band = tmp_cfg.get("wifi_band", None)
    if use_band is not None:
        df = df[df["wifi_band"] == use_band].reset_index(drop=True)

    test_df = df[df["environment"] == E3_NAME].reset_index(drop=True)
    print(f"Test samples (E3='{E3_NAME}'): {len(test_df)}")

    # 2) Load artifacts + config + weights
    mean, std, bg_means, cfg, state_dict = load_artifacts_and_config()
    use_dphase = bool(cfg["use_dphase"])
    use_amp    = bool(cfg["use_amp"])
    in_ch = int(cfg["in_channels"])
    max_len = int(cfg["max_len"])

    # 3) Build dataset/loader that uses EXACT SAME normalization & bg
    test_ds = CSIPhaseDataset(
        csv_path=CSV_PATH, data_root=DATA_DIR,
        ids=test_df["label"].tolist(),
        wifi_band=use_band,
        max_len=max_len,
        use_dphase=use_dphase,
        use_amp=use_amp,
        mean=mean, std=std,
        background_phase_mean=bg_means,  # dict or vector or None
        augment=False,
        enable_cache=False
    )
    test_loader = DataLoader(test_ds, batch_size=24, shuffle=False, num_workers=0, pin_memory=True)

    # 4) Model -> load weights
    model = CSI1DTCNCount(in_channels=in_ch, n_classes=6, depth=5).to(device)
    try:
        model.load_state_dict(state_dict if isinstance(state_dict, dict) else state_dict["state_dict"], strict=True)
    except Exception as e:
        # if you accidentally changed in_channels, fall back but warn
        print("Strict load failed:", e)
        print("Retrying with strict=False (check your in_channels/config!)")
        model.load_state_dict(state_dict if isinstance(state_dict, dict) else state_dict["state_dict"], strict=False)

    # 5) Eval
    loss_fn = nn.CrossEntropyLoss()
    test_loss, test_acc, preds, targets = eval_one_epoch(model, test_loader, device, loss_fn)
    print(f"\n=== FINAL TEST on {E3_NAME} ===")
    print(f"test loss {test_loss:.4f} | test acc {test_acc:.3f}")

    # Confusion matrix (quick)
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(targets, preds, labels=list(range(6)))
        print("\nConfusion Matrix (rows=true, cols=pred):\n", cm)
    except Exception:
        pass

if __name__ == "__main__":
    main()
