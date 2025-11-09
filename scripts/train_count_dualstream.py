import os, math, json, random, argparse, signal, sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------- Repro --------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# -------------------- DSP helpers --------------------
from scipy.signal import butter, filtfilt
def butter_bandpass_filter(x, fs=1000.0, low=0.5, high=50.0, axis=0):
    # x: (..., T, ...)
    b, a = butter(N=4, Wn=[low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x, axis=axis, method="gust")

def standardize_per_channel(x, eps=1e-6):
    # x: (T, C)
    m = x.mean(axis=0, keepdims=True)
    s = x.std(axis=0, keepdims=True)
    return (x - m) / (s + eps)

def pad_or_crop(x, T):
    t = x.shape[0]
    if t >= T:
        return x[:T]
    out = np.zeros((T, x.shape[1]), dtype=x.dtype)
    out[:t] = x
    return out

# -------------------- Δphase from .mat --------------------
import scipy.io

def _extract_csi_complex_from_mat(mat_path):
    """
    Returns array of shape (t, 3, 3, 30) complex, or None.
    Compatible with WiMANS 'trace' cell/struct layout.
    """
    try:
        file = scipy.io.loadmat(mat_path)
        trace = file["trace"]
        t = trace.shape[0]
        mats = []
        for i in range(t):
            try:
                csi = trace[i, 0]["csi"][0, 0]  # (3,3,30) complex
                if csi.shape == (3,3,30):
                    mats.append(csi)
            except Exception:
                continue
        if not mats:
            return None
        arr = np.array(mats)  # (t,3,3,30)
        # Some dumps are real-imag pairs; ensure complex dtype:
        if not np.iscomplexobj(arr):
            # try to view as complex if saved separately; here we assume already complex
            arr = arr.astype(np.complex64)
        return arr
    except Exception:
        return None

def compute_dphase_from_mat(mat_path):
    """
    Δphase pipeline:
      - Phase = angle(complex)
      - Unwrap along time axis
      - Δphase = diff over time
      - Pad one row of zeros to keep length T
      - Reshape to (T, 270)
    """
    csi = _extract_csi_complex_from_mat(mat_path)
    if csi is None:
        raise RuntimeError(f"Cannot read CSI from {mat_path}")
    phase = np.angle(csi)                 # (t,3,3,30)
    phase = np.unwrap(phase, axis=0)      # unwrap over time
    dphi  = np.diff(phase, axis=0)        # (t-1,3,3,30)
    # pad one zero at start to align length:
    dphi  = np.concatenate([np.zeros_like(dphi[:1]), dphi], axis=0)  # (t,3,3,30)
    t, a, b, s = dphi.shape
    dphi = dphi.reshape(t, a*b*s).astype(np.float32)  # (t,270)
    return dphi

# -------------------- Dataset --------------------
class CSIDualStreamCount(Dataset):
    """
    Dual-stream dataset (Amplitude + ΔPhase).
    - annotation.csv columns: '#','label','environment','wifi_band','number_of_users', ...
    - amp files in data_root/wifi_csi/amp/{label}.npy  (shape: (t,270) or (t,3,3,30))
    - mat files in data_root/wifi_csi/mat/{label}.mat  (to compute Δphase)
    - band_mode: '2.4' | '5' | 'both'  (pair rows by label)
    """
    def __init__(self, df, data_root, max_len=3000, band_mode="both",
                 use_filter=True, fs=1000.0, bp_low=0.5, bp_high=50.0,
                 cache_items=64):
        self.df = df.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.max_len = max_len
        assert band_mode in {"2.4","5","both"}
        self.band_mode = band_mode
        self.use_filter = use_filter
        self.fs = fs; self.bp_low = bp_low; self.bp_high = bp_high

        # Build map: label -> rows { "2.4": idx, "5": idx }
        self.by_label = {}
        for i, r in self.df.iterrows():
            lab = r["label"]
            b = str(r["wifi_band"]).strip()
            self.by_label.setdefault(lab, {})[b] = i

        # Final index depending on band_mode
        self.index = []
        if self.band_mode == "both":
            for lab, bands in self.by_label.items():
                if "2.4" in bands and "5" in bands:
                    self.index.append(("both", lab, bands["2.4"], bands["5"]))
        else:
            need = self.band_mode
            for lab, bands in self.by_label.items():
                if need in bands:
                    self.index.append((need, lab, bands[need], None))

        # Simple LRU cache for Δphase arrays to avoid re-reading MAT every time
        self._phase_cache = OrderedDict()
        self._max_cache = cache_items

    def __len__(self):
        return len(self.index)

    def _lru_get(self, key):
        if key in self._phase_cache:
            val = self._phase_cache.pop(key)
            self._phase_cache[key] = val
            return val
        return None

    def _lru_put(self, key, val):
        if key in self._phase_cache:
            self._phase_cache.pop(key)
        elif len(self._phase_cache) >= self._max_cache:
            self._phase_cache.pop(next(iter(self._phase_cache)))
        self._phase_cache[key] = val

    def _amp_path(self, label):
        return self.data_root / "wifi_csi" / "amp" / f"{label}.npy"

    def _mat_path(self, label):
        return self.data_root / "wifi_csi" / "mat" / f"{label}.mat"

    def _load_amp(self, path):
        arr = np.load(str(path))  # (t,270) or (t,3,3,30)
        if arr.ndim == 4:
            t, a, b, s = arr.shape
            arr = arr.reshape(t, a*b*s)
        return arr.astype(np.float32)

    def _load_dphase(self, label):
        cached = self._lru_get(label)
        if cached is not None:
            return cached
        matp = self._mat_path(label)
        dphi = compute_dphase_from_mat(str(matp))  # (t,270)
        self._lru_put(label, dphi)
        return dphi

    def _prep_pair(self, lab_row_idx):
        r = self.df.iloc[lab_row_idx]
        label = r["label"]
        amp = self._load_amp(self._amp_path(label))
        dph = self._load_dphase(label)

        # length alignment
        t = min(len(amp), len(dph))
        amp = amp[:t]; dph = dph[:t]

        # (optional) band-pass to suppress static and high-freq noise
        if self.use_filter:
            amp = butter_bandpass_filter(amp, fs=self.fs, low=self.bp_low, high=self.bp_high, axis=0)
            dph = butter_bandpass_filter(dph, fs=self.fs, low=self.bp_low, high=self.bp_high, axis=0)

        # per-channel standardization (instance-wise) ➜ location invariance
        amp = standardize_per_channel(amp)
        dph = standardize_per_channel(dph)

        # pad/crop to fixed T
        amp = pad_or_crop(amp, self.max_len)      # (T,C)
        dph = pad_or_crop(dph, self.max_len)

        # to tensors (C,T) for Conv1d
        x_amp = torch.from_numpy(amp).float().permute(1,0)  # (C,T)
        x_dph = torch.from_numpy(dph).float().permute(1,0)  # (C,T)

        y = int(r["number_of_users"])
        meta = {"environment": r["environment"], "label": label, "wifi_band": str(r["wifi_band"])}
        return x_amp, x_dph, y, meta

    def __getitem__(self, i):
        mode, lab, i24, i5 = self.index[i]
        if mode == "both":
            # Pair the *same label* across 2.4 and 5; concat channels
            xA_24, xP_24, y, meta = self._prep_pair(i24)
            xA_5,  xP_5,  y2, _   = self._prep_pair(i5)
            # safety: labels should match
            assert y == y2, "Paired bands disagree in number_of_users"

            x_amp = torch.cat([xA_24, xA_5], dim=0)  # (540, T)
            x_dph = torch.cat([xP_24, xP_5], dim=0)  # (540, T)
            meta["wifi_band"] = "both"
        else:
            x_amp, x_dph, y, meta = self._prep_pair(i24)
        return x_amp, x_dph, torch.tensor(y, dtype=torch.long), meta

# -------------------- Model --------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=5, d=1):
        super().__init__()
        pad = d*(k//2)
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=d, bias=False),
            nn.InstanceNorm1d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class DualStreamCNNBiLSTM(nn.Module):
    """
    Two branches (amp & dphase) -> temporal CNN stacks -> BiLSTM per branch -> fusion FC.
    """
    def __init__(self, in_ch_amp, in_ch_dph, num_classes=6, lstm_hidden=256, lstm_layers=1):
        super().__init__()
        # amplitude branch
        self.amp_cnn = nn.Sequential(
            ConvBlock(in_ch_amp, 256, k=5, d=1),
            ConvBlock(256, 256, k=5, d=2),
            ConvBlock(256, 256, k=7, d=4),
        )
        # dphase branch
        self.ph_cnn = nn.Sequential(
            ConvBlock(in_ch_dph, 256, k=5, d=1),
            ConvBlock(256, 256, k=5, d=2),
            ConvBlock(256, 256, k=7, d=4),
        )
        # BiLSTMs (input = 256 features over time)
        self.amp_lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden, num_layers=lstm_layers,
                                batch_first=True, bidirectional=True)
        self.ph_lstm  = nn.LSTM(input_size=256, hidden_size=lstm_hidden, num_layers=lstm_layers,
                                batch_first=True, bidirectional=True)
        # Fusion head
        fusion_in = 2*(2*lstm_hidden)  # amp(2H) + ph(2H)
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward_branch(self, x, cnn, lstm):
        # x: (B,C,T) -> CNN: (B,256,T) -> permute -> LSTM over T
        z = cnn(x)                       # (B,256,T)
        z = z.permute(0,2,1).contiguous()# (B,T,256)
        out, _ = lstm(z)                 # (B,T,2H)
        # temporal global average pooling
        feat = out.mean(dim=1)           # (B,2H)
        return feat

    def forward(self, x_amp, x_dph):
        fa = self.forward_branch(x_amp, self.amp_cnn, self.amp_lstm)
        fp = self.forward_branch(x_dph, self.ph_cnn,  self.ph_lstm)
        f = torch.cat([fa, fp], dim=1)
        return self.head(f)

# -------------------- Splits --------------------
def split_env(df, train_env="classroom", val_env="meeting_room", test_env="empty_room"):
    tr = df[df.environment==train_env]
    va = df[df.environment==val_env]
    te = df[df.environment==test_env]
    return tr, va, te

# -------------------- Training / Eval --------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot=0; correct=0
    for xa, xp, y, _ in loader:
        xa=xa.to(device, non_blocking=True); xp=xp.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
        logits = model(xa, xp)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        tot += y.numel()
    return correct/max(1,tot)

def make_class_weights(ds, device):
    # quick 1-pass over dataset to count
    counts = np.zeros(6, dtype=np.int64)
    for i in range(len(ds)):
        _, _, y, _ = ds[i]
        counts[int(y)] += 1
    tot = counts.sum()
    w = (tot/(counts+1e-6))
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="annotation.csv")
    ap.add_argument("--data_root", type=str, default=".")
    ap.add_argument("--band", type=str, default="both", choices=["2.4","5","both"])
    ap.add_argument("--max_len", type=int, default=3000)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", type=str, default="count_dualstream.pt")
    ap.add_argument("--train_env", type=str, default="classroom")
    ap.add_argument("--val_env", type=str, default="meeting_room")
    ap.add_argument("--test_env", type=str, default="empty_room")
    ap.add_argument("--no_filter", action="store_true", help="disable band-pass filtering")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv)
    # keep only needed columns
    need_cols = {"label","environment","wifi_band","number_of_users"}
    miss = need_cols - set(df.columns)
    if miss:
        raise ValueError(f"annotation.csv missing columns: {miss}")

    # filter valid counts and known envs
    df = df[df["number_of_users"].between(0,5)]
    df = df[df["environment"].isin(["classroom","meeting_room","empty_room"])]

    tr_df, va_df, te_df = split_env(df, args.train_env, args.val_env, args.test_env)
    print(f"Rows → train {len(tr_df)} | val {len(va_df)} | test {len(te_df)}")

    # datasets
    ds_kwargs = dict(
        data_root=args.data_root,
        max_len=args.max_len,
        band_mode=args.band,
        use_filter=not args.no_filter,
        fs=1000.0, bp_low=0.5, bp_high=50.0,
        cache_items=64
    )
    train_set = CSIDualStreamCount(tr_df, **ds_kwargs)
    val_set   = CSIDualStreamCount(va_df, **ds_kwargs)
    test_set  = CSIDualStreamCount(te_df, **ds_kwargs)

    in_ch = 270 if args.band in {"2.4","5"} else 540

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True, persistent_workers=args.num_workers>0)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = DualStreamCNNBiLSTM(in_ch_amp=in_ch, in_ch_dph=in_ch, num_classes=6).to(device)
    weights = make_class_weights(train_set, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = 0.0

    def save_ckpt(tag="best"):
        torch.save({"model": model.state_dict()}, args.ckpt if tag=="best" else args.ckpt+".last")

    try:
        for epoch in range(1, args.epochs+1):
            model.train()
            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", ncols=100)
            run_loss, seen = 0.0, 0
            for xa, xp, y, _ in pbar:
                xa=xa.to(device, non_blocking=True); xp=xp.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    logits = model(xa, xp)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                bs = y.size(0)
                run_loss += loss.item()*bs; seen += bs
                pbar.set_postfix(loss=f"{run_loss/max(1,seen):.4f}")

            sched.step()
            val_acc = evaluate(model, val_loader, device)
            print(f"val_acc: {val_acc:.4f}")
            if val_acc > best_val:
                best_val = val_acc
                save_ckpt("best")

    except KeyboardInterrupt:
        print("\nCtrl-C detected → saving last and evaluating...")
        save_ckpt("last")

    # Load best (fallback to last)
    try:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"])
        print("Loaded BEST checkpoint.")
    except Exception:
        state = torch.load(args.ckpt+".last", map_location=device)
        model.load_state_dict(state["model"])
        print("Loaded LAST checkpoint.")

    val_acc = evaluate(model, val_loader, device)
    test_acc = evaluate(model, test_loader, device)
    print(f"[FINAL] val_acc={val_acc:.4f} | test_acc={test_acc:.4f}")

if __name__ == "__main__":
    main()
