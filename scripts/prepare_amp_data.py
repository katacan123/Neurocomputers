import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

torch.cuda.is_available(), torch.cuda.get_device_name(0)

activities_list = [
    "nothing", "walk", "rotation", "jump", "wave",
    "lie_down", "pick_up", "sit_down", "stand_up"
]

activity_to_label = {activity: i for i, activity in enumerate(activities_list)}

print(activity_to_label)

class CSIAmpDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        max_len: int = 3000,
        ids=None,
        wifi_band = None,
    ):
        """
        csv_path: path to annotation.csv
        data_root: folder that contains wifi_csi/amp/
        max_len: pad/crop time dimension to this
        ids: optional list of labels (act_1_1, act_1_2, ...)
        wifi_band: 2.4 or 5.0 or None (means use all)
        """
        df = pd.read_csv(csv_path)

        # filter by band if wanted
        if wifi_band is not None:
            df = df[df["wifi_band"] == wifi_band]

        # filter by ids if wanted
        if ids is not None:
            df = df[df["label"].isin(ids)]

        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root + "/wifi_csi/amp")
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _pad_or_crop(self, x: np.ndarray) -> np.ndarray:
        """
        x: (T, C)
        return: (max_len, C)
        """
        T, C = x.shape
        if T > self.max_len:
            x = x[: self.max_len, :]
        elif T < self.max_len:
            pad = np.zeros((self.max_len - T, C), dtype=x.dtype)
            x = np.vstack([x, pad])
        return x

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --------- load amplitude array ---------
        label = row["label"]  # e.g. act_1_1
        fpath = self.data_root / f"{label}.npy"
        amp = np.load(fpath)  # (T,3,3,30) or (T,270)

        # (T, 3, 3, 30) -> (T, 270)
        if amp.ndim == 4:
            T, tx, rx, sc = amp.shape
            amp = amp.reshape(T, tx * rx * sc)

        # pad/crop time dim
        amp = self._pad_or_crop(amp)  # (max_len, 270)

        # --------- STEP 2 & 3: log, domain randomization, normalization ---------
        # ensure float32
        amp = amp.astype(np.float32)

        # 1) log-amplitude to compress dynamic range
        amp = np.log1p(amp)

        # 2) domain randomization (applied to all samples here)
        #    (a) random global gain & offset
        gain = np.random.uniform(0.9, 1.1)
        offset = np.random.normal(0.0, 0.02)
        amp = gain * amp + offset

        #    (b) feature dropout (channel masking)
        mask_prob = 0.02  # 2% of channels zeroed
        mask = (np.random.rand(1, amp.shape[1]) > mask_prob).astype(np.float32)
        amp = amp * mask

        # 3) sample-wise standardization (global mean/std over this example)
        mean = amp.mean()
        std = amp.std() + 1e-6
        amp = (amp - mean) / std

        # to torch, and put channels first for Conv1d
        x = torch.from_numpy(amp).permute(1, 0)  # (270, max_len)

        # --------- labels ---------
        # human count:
        y_count = int(row["number_of_users"])
        num_users = y_count
        y_count = torch.tensor(y_count, dtype=torch.long)

        # 1) collect ALL non-empty activities from 6 slots
        all_acts = []
        for i in range(1, 7):   # 1..6
            col = f"user_{i}_activity"
            val = row[col]
            if isinstance(val, str) and val.strip() != "":
                all_acts.append(val.strip())

        # 2) respect number_of_users (dataset says “there were N people”)
        #    so if we somehow got more acts than N, cut it
        if len(all_acts) > num_users:
            all_acts = all_acts[:num_users]

        # 3) build multi-hot vector
        y_act = torch.zeros(len(activities_list), dtype=torch.float32)
        for a in all_acts:
            if a in activity_to_label:
                y_act[activity_to_label[a]] = 1.0

        return x, y_count, y_act
