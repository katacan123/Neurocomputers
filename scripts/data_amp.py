# data_amp.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from preprocess_amp import build_amp_features

class AmpCSIDataset(Dataset):
    """
    WiMANS-format annotation.csv:
    #, label, environment, wifi_band, number_of_users, user_X_activity, ...
    AMP_DIR/{label}.npy  -> (T,3,3,30) veya (T,270)
    Output: (T, D), count, env_id
    """
    def __init__(self, annotation_csv, amp_dir, ids=None,
                 target_len=3000, use_svd=True, use_lowpass=True, n_pca=5):
        self.df = pd.read_csv(annotation_csv)
        self.amp_dir = amp_dir
        self.target_len = target_len
        self.use_svd = use_svd
        self.use_lowpass = use_lowpass
        self.n_pca = n_pca

        need_cols = ["label", "environment", "number_of_users"]
        for c in need_cols:
            assert c in self.df.columns, f"Column '{c}' missing in annotation.csv"

        if ids is not None:
            self.df = self.df[self.df["label"].isin(ids)].reset_index(drop=True)

        envs = sorted(self.df["environment"].unique().tolist())
        self.env_to_idx = {e:i for i,e in enumerate(envs)}

    def __len__(self): 
        return len(self.df)

    def _load_amp(self, label):
        f = os.path.join(self.amp_dir, f"{label}.npy")
        if not os.path.exists(f):
            raise FileNotFoundError(f"Amplitude file not found: {f}")
        return np.load(f, allow_pickle=False)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row["label"]
        amp_raw = self._load_amp(label)

        X = build_amp_features(
            amp_raw,
            target_len=self.target_len,
            use_svd=self.use_svd,
            use_lowpass=self.use_lowpass,
            n_pca=self.n_pca,
        )  # (T, D)

        y_cnt = int(row["number_of_users"])
        env_id = self.env_to_idx[row["environment"]]

        return torch.from_numpy(X), torch.tensor(y_cnt).long(), torch.tensor(env_id).long()
