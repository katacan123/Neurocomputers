# wimans/dataset.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# wimans/dataset.py

from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from .labels import decode_list


class WiMANSDataset(Dataset):
    """
    Loads tensors + per-user labels (activity, location, identity dummy).
    """

    def __init__(self, splits_csv: str, split: str, target_T: int, max_users: int):
        df = pd.read_csv(splits_csv)
        if "split" not in df.columns:
            raise KeyError("'split' column missing in CSV")

        df = df[df["split"] == split].reset_index(drop=True)
        if df.empty:
            raise RuntimeError(f"No samples for split={split}")

        self.df = df
        self.target_T = target_T
        self.max_users = max_users

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- Load CSI tensor ----
        tensor_path = Path(row["tensor_path"])
        x = torch.load(tensor_path)  # (C, T)

        # crop/pad to target_T
        T = x.shape[1]
        if T > self.target_T:
            start = (T - self.target_T) // 2
            x = x[:, start:start+self.target_T]
        elif T < self.target_T:
            pad = x[:, -1:].repeat(1, self.target_T - T)
            x = torch.cat([x, pad], dim=1)

        # ---- Parse label lists ----
        y_act = torch.tensor(decode_list(row["y_act"]), dtype=torch.long)
        y_loc = torch.tensor(decode_list(row["y_loc"]), dtype=torch.long)
        y_id  = torch.tensor(decode_list(row["y_id"]),  dtype=torch.long)
        slot_mask = torch.tensor(decode_list(row["slot_mask"]),
                                 dtype=torch.float32)
        gt_count = torch.tensor(int(row["gt_count"]), dtype=torch.long)

        return {
            "x": x,                # (C,T)
            "y_act": y_act,        # (U)
            "y_loc": y_loc,        # (U)
            "y_id": y_id,          # (U)
            "slot_mask": slot_mask,# (U)
            "gt_count": gt_count,  # ()
            "sample_id": row["label"],
            "environment": row["environment"],
            "wifi_band": row["wifi_band"],
        }


def build_dataloaders(
    splits_csv: str,
    target_T: int,
    max_users: int,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    train_ds = WiMANSDataset(splits_csv, "train", target_T, max_users)
    val_ds   = WiMANSDataset(splits_csv, "val",   target_T, max_users)
    test_ds  = WiMANSDataset(splits_csv, "test",  target_T, max_users)

    def collate(batch):
        out = {}
        for k in batch[0].keys():
            v0 = batch[0][k]
            if isinstance(v0, torch.Tensor):
                out[k] = torch.stack([b[k] for b in batch], dim=0)
            else:
                out[k] = [b[k] for b in batch]
        return out

    return (
        DataLoader(train_ds, batch_size, True,  num_workers=num_workers,
                   pin_memory=pin_memory, collate_fn=collate),
        DataLoader(val_ds,   batch_size, False, num_workers=num_workers,
                   pin_memory=pin_memory, collate_fn=collate),
        DataLoader(test_ds,  batch_size, False, num_workers=num_workers,
                   pin_memory=pin_memory, collate_fn=collate),
    )
