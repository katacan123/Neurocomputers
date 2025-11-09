#prepare_ phase_data

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import OrderedDict

# ----- activities (multi-label) -----
activities_list = [
    "nothing", "walk", "rotation", "jump", "wave",
    "lie_down", "pick_up", "sit_down", "stand_up"
]
activity_to_label = {activity: i for i, activity in enumerate(activities_list)}


class CSIPhaseDataset(Dataset):
    """
    Faz + opsiyonel Δfaz (+opsiyonel amplitude) dataset'i.
    Çıkış: x: (C, T)  -> Conv1d girişi için
           y_count: Long (0..5)
           y_act: multi-hot vektör (9)
    """
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        ids=None,
        wifi_band: float | None = None,
        max_len: int = 3000,
        use_dphase: bool = True,
        use_amp: bool = True,
        phase_dirname: str = "phase",
        amp_dirname: str = "amp_from_mat",
        mean: np.ndarray | None = None,   # (C,)
        std:  np.ndarray | None = None,   # (C,)
        augment: bool = False,
        seed: int = 42,
        # background subtraction:
        # can be np.ndarray (270,) for global or dict[str, np.ndarray] per-env
        background_phase_mean=None,
        # küçük RAM cache
        enable_cache: bool = False,
        max_cache_items: int = 128,
    ):
        self.rng = np.random.RandomState(seed)

        df = pd.read_csv(csv_path)
        if wifi_band is not None:
            df = df[df["wifi_band"] == wifi_band]
        if ids is not None:
            df = df[df["label"].isin(ids)]

        self.df = df.reset_index(drop=True)
        self.root = Path(data_root)
        self.phase_dir = self.root / "wifi_csi" / phase_dirname
        self.amp_dir   = self.root / "wifi_csi" / amp_dirname
        self.max_len = max_len

        self.use_dphase = use_dphase
        self.use_amp = use_amp
        self.mean = mean
        self.std = std
        self.augment = augment

        self.bg_phase_mean = background_phase_mean  # None | np.ndarray | dict
        self.enable_cache = enable_cache
        self.max_cache_items = max_cache_items
        self._cache = OrderedDict()  # path -> np.ndarray

    def __len__(self):
        return len(self.df)

    # ---------------- utils ----------------
    def _pad_or_crop(self, x: np.ndarray) -> np.ndarray:
        """x: (T, C) -> (max_len, C)"""
        T, C = x.shape
        if T > self.max_len:
            return x[: self.max_len, :]
        if T < self.max_len:
            pad = np.zeros((self.max_len - T, C), dtype=x.dtype)
            return np.vstack([x, pad])
        return x

    def _cache_put(self, path: Path, arr: np.ndarray):
        """LRU RAM cache: memmap yerine gerçek ndarray sakla."""
        if not self.enable_cache:
            return
        if path in self._cache:
            self._cache.move_to_end(path)
            return
        arr = np.ascontiguousarray(arr)
        self._cache[path] = arr
        if len(self._cache) > self.max_cache_items:
            self._cache.popitem(last=False)

    def _np_load_cached(self, path: Path) -> np.ndarray:
        """np.load + LRU cache (mmap KAPALI)."""
        if self.enable_cache and (path in self._cache):
            self._cache.move_to_end(path)
            return self._cache[path]
        arr = np.load(path, allow_pickle=False)  # mmap_mode=None
        arr = np.ascontiguousarray(arr)
        self._cache_put(path, arr)
        return arr

    def _get_bg(self, env: str):
        if self.bg_phase_mean is None:
            return None
        if isinstance(self.bg_phase_mean, dict):
            return self.bg_phase_mean.get(env, None)
        return self.bg_phase_mean  # single vector

    def _load_phase(self, base: str, env: str) -> np.ndarray:
        """phase npy'yi yükler, (T,270) hale getirir ve (max_len,270) döndürür."""
        fp = self.phase_dir / f"{base}.npy"
        p = self._np_load_cached(fp)  # (T,270) veya (T,3,3,30)
        if p.ndim == 4:
            T, tx, rx, sc = p.shape
            p = p.reshape(T, tx * rx * sc)
        p = self._pad_or_crop(p.astype(np.float32))
        # background subtraction (sadece faz)
        bg = self._get_bg(env)
        if bg is not None:
            p = p - bg[None, :]  # (T,270)
        return p

    def _load_amp(self, base: str) -> np.ndarray:
        fa = self.amp_dir / f"{base}.npy"
        a = self._np_load_cached(fa)
        if a.ndim == 4:
            T, tx, rx, sc = a.shape
            a = a.reshape(T, tx * rx * sc)
        a = self._pad_or_crop(a.astype(np.float32))
        return a

    def _build_feats(self, phase_Tx: np.ndarray, base: str) -> np.ndarray:
        """phase_Tx: (T,270) -> concat([phase, dphase, amp?]) along channel -> (T, C)"""
        feats = [phase_Tx]
        if self.use_dphase:
            dphase = np.diff(phase_Tx, axis=0, prepend=phase_Tx[:1, :])
            feats.append(dphase)
        if self.use_amp:
            amp = self._load_amp(base)
            feats.append(amp)
        X = np.concatenate(feats, axis=1).astype(np.float32)  # (T, C)
        return X

    # ------------- simple augmentations (train only) -------------
    def _augment_inplace(self, x: np.ndarray):
        """time mask, channel dropout, jitter, tiny shift"""
        T, C = x.shape
        for _ in range(2):
            if self.rng.rand() < 0.8:
                t0 = self.rng.randint(0, max(1, T - 16))
                t1 = min(T, t0 + 16)
                x[t0:t1, :] *= 0.0
        if self.rng.rand() < 0.8:
            k = max(1, int(0.03 * C))
            idx = self.rng.choice(C, size=k, replace=False)
            x[:, idx] = 0.0
        if self.rng.rand() < 0.8:
            noise = self.rng.normal(0.0, 0.01, size=x.shape).astype(np.float32)
            x += noise
        if self.rng.rand() < 0.8:
            shift = int(self.rng.randint(-8, 9))
            if shift > 0:
                x[shift:, :] = x[:-shift, :]
                x[:shift, :] = 0.0
            elif shift < 0:
                s = -shift
                x[:-s, :] = x[s:, :]
                x[-s:, :] = 0.0

    # ---------------- main ----------------
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        base = row["label"]
        env  = row["environment"]

        # 1) phase yükle (T,270) (+ optional per-env bg subtract)
        phase_Tx = self._load_phase(base, env)

        # 2) özellik: phase [+ dphase] [+ amp]
        X = self._build_feats(phase_Tx, base)   # (T, C)

        # 3) augment (sadece train)
        if self.augment:
            self._augment_inplace(X)

        # 4) (C, T) -> torch
        x = torch.from_numpy(X).permute(1, 0).contiguous()  # (C, T)

        # 5) normalize (kanal bazlı)
        if (self.mean is not None) and (self.std is not None):
            mean_t = torch.from_numpy(self.mean).view(-1, 1)
            std_t  = torch.from_numpy(self.std).view(-1, 1)
            x = (x - mean_t) / (std_t + 1e-8)

        # --------- labels ---------
        y_count = torch.tensor(int(row["number_of_users"]), dtype=torch.long)

        # multi-label aktiviteler
        num_users = int(row["number_of_users"])
        all_acts = []
        for i in range(1, 7):
            col = f"user_{i}_activity"
            val = row[col]
            if isinstance(val, str) and val.strip() != "":
                all_acts.append(val.strip())
        if len(all_acts) > num_users:
            all_acts = all_acts[:num_users]

        y_act = torch.zeros(len(activities_list), dtype=torch.float32)
        for a in all_acts:
            if a in activity_to_label:
                y_act[activity_to_label[a]] = 1.0

        return x, y_count, y_act


# ---------- kanal istatistikleri (mean/std) ----------
@torch.no_grad()
def compute_channel_stats(dataset: Dataset, batch_size: int = 32, num_workers: int = 0):
    """
    Per-channel mean/std (C,) döndürür. Augment KAPALI dataset ver.
    Dataset'in __getitem__ çıktısı (x,y,_) -> x: (C,T)
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    cnt = 0
    mean = None
    M2 = None  # approximate streaming variance

    for x, _, _ in loader:
        # x: (B, C, T)
        x_mean_c = x.mean(dim=2).mean(dim=0).numpy()                 # (C,)
        x_var_c  = x.var(dim=2, unbiased=False).mean(dim=0).numpy()  # (C,)
        if mean is None:
            mean = x_mean_c
            M2 = x_var_c
            cnt = 1
        else:
            cnt += 1
            mean = mean + (x_mean_c - mean) / cnt
            M2 = M2 + (x_var_c - M2) / cnt

    std = np.sqrt(M2 + 1e-8)
    return mean.astype(np.float32), std.astype(np.float32)
