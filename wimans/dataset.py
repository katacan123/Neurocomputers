# wimans/dataset.py

from pathlib import Path
from typing import Dict, Any, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from .labels import decode_list


class WiMANSDataset(Dataset):
    """
    WiMANS dataset for SADU-style HAR + derived count.

    Expects splits CSV with columns:
      - sample_id
      - tensor_path
      - environment
      - wifi_band
      - y_act (JSON list)
      - y_id (JSON list)
      - slot_mask (JSON list)
      - gt_count (int)
      - split ("train"/"val"/"test")
    """

    def __init__(
        self,
        splits_csv: str,
        split: str,
        target_T: int,
        max_users: int
    ):
        self.df = pd.read_csv(splits_csv)
        if "split" not in self.df.columns:
            raise KeyError("'split' column not found in splits CSV.")

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError(f"No samples found for split='{split}' in {splits_csv}")

        self.target_T = target_T
        self.max_users = max_users

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        tensor_path = Path(row["tensor_path"])
        if not tensor_path.exists():
            raise FileNotFoundError(f"Tensor file not found: {tensor_path}")

        x: torch.Tensor = torch.load(tensor_path)  # (C, T)
        # Safety: enforce target_T
        if x.shape[1] > self.target_T:
            start = (x.shape[1] - self.target_T) // 2
            x = x[:, start:start + self.target_T]
        elif x.shape[1] < self.target_T:
            pad_len = self.target_T - x.shape[1]
            pad = x[:, -1:].repeat(1, pad_len)
            x = torch.cat([x, pad], dim=1)

        y_act_list = decode_list(row["y_act"])
        y_id_list = decode_list(row["y_id"])
        slot_mask_list = decode_list(row["slot_mask"])

        if len(y_act_list) != self.max_users:
            raise ValueError(f"y_act length {len(y_act_list)} != max_users={self.max_users}")
        if len(y_id_list) != self.max_users:
            raise ValueError(f"y_id length {len(y_id_list)} != max_users={self.max_users}")
        if len(slot_mask_list) != self.max_users:
            raise ValueError(f"slot_mask length {len(slot_mask_list)} != max_users={self.max_users}")

        y_act = torch.tensor(y_act_list, dtype=torch.long)           # (max_users,)
        y_id = torch.tensor(y_id_list, dtype=torch.long)             # (max_users,)
        slot_mask = torch.tensor(slot_mask_list, dtype=torch.float32)  # (max_users,)
        gt_count = int(row["gt_count"])

        return {
            "x": x,  # (C, T)
            "y_act": y_act,
            "y_id": y_id,
            "slot_mask": slot_mask,
            "gt_count": torch.tensor(gt_count, dtype=torch.long),
            "sample_id": row["sample_id"],
            "environment": row["environment"],
            "wifi_band": row["wifi_band"],
        }


def build_dataloaders(
    splits_csv: str,
    target_T: int,
    max_users: int,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders from a splits CSV.
    """

    train_ds = WiMANSDataset(splits_csv, "train", target_T, max_users)
    val_ds   = WiMANSDataset(splits_csv, "val",   target_T, max_users)
    test_ds  = WiMANSDataset(splits_csv, "test",  target_T, max_users)

    def default_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        keys = batch[0].keys()
        for k in keys:
            v0 = batch[0][k]
            if isinstance(v0, torch.Tensor):
                out[k] = torch.stack([b[k] for b in batch], dim=0)
            else:
                out[k] = [b[k] for b in batch]
        return out

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=default_collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=default_collate,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=default_collate,
    )

    return train_loader, val_loader, test_loader
