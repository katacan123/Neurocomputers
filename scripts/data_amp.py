# data_amp.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from preprocess_amp import build_amp_features

class AmpCSIDataset(Dataset):
    """
    annotation.csv: id, activity, count, environment
    AMP_DIR/{id}.npy: (T,3,3,30) veya (T,270) amplitude
    """
    def __init__(self, annotation_csv, amp_dir, ids=None, target_len=3000,
                 use_svd=True, use_lowpass=True, n_pca=5):
        self.df = pd.read_csv(annotation_csv)
        self.amp_dir = amp_dir
        self.target_len = target_len
        self.use_svd = use_svd
        self.use_lowpass = use_lowpass
        self.n_pca = n_pca

        need = ["id","count","environment"]
        miss = [c for c in need if c not in self.df.columns]
        if miss:
            raise RuntimeError(f"annotation.csv missing columns: {miss}")

        if ids is not None:
            self.df = self.df[self.df["id"].isin(ids)].reset_index(drop=True)

        envs = sorted(self.df["environment"].unique().tolist())
        self.env_to_idx = {e:i for i,e in enumerate(envs)}

    def __len__(self): return len(self.df)

    def _load_amp(self, rec_id):
        f = os.path.join(self.amp_dir, f"{rec_id}.npy")
        if not os.path.exists(f):
            raise FileNotFoundError(f"Amplitude file not found: {f}")
        return np.load(f, allow_pickle=False)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rec_id = row["id"]
        amp_raw = self._load_amp(rec_id)
        X = build_amp_features(
            amp_raw, target_len=self.target_len,
            use_svd=self.use_svd, use_lowpass=self.use_lowpass, n_pca=self.n_pca
        )  # (T, D)
        y_cnt = int(row["count"])
        env_id = self.env_to_idx[row["environment"]]
        return torch.from_numpy(X), torch.tensor(y_cnt).long(), torch.tensor(env_id).long()
