# data/wimans_dataset.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocess_csi import preprocess_sample

# 9 activity classes (adjust if your labels differ)
ACTIVITIES = [
    "nothing",
    "walk",
    "rotation",
    "jump",
    "wave",
    "lie_down",
    "pick_up",
    "sit_down",
    "stand_up",
]

ACT_TO_IDX = {a: i for i, a in enumerate(ACTIVITIES)}



class WiMansActivityDataset(Dataset):
    """
    Returns:
      x:        (C, T) float32   # model input, e.g. (270, 3000)
      y_act:    (54,) float32    # 6 x 9 flattened multi-label
      y_count:  scalar int (0..5)  # derived from GT activities, for eval only
      meta:     dict with label, environment, wifi_band
    """

    def __init__(
        self,
        annotation_csv: str | Path,
        csi_amp_root: str | Path,
        sample_ids: List[str],
        H_avgs_dict: Dict[Tuple[str, float], np.ndarray],
        T_target: int = 3000,
        alpha: float = 1.0,
        noise_power: float = 1.0,
        C_tx: float = 1.0,
        cache_preprocessed: bool = False,
        preprocessed_dir: Optional[str] = None,
    ):
        self.annotation_csv = Path(annotation_csv)
        self.csi_amp_root = Path(csi_amp_root)
        self.df = pd.read_csv(self.annotation_csv)

        # here sample_ids are actually 'label' values
        self.sample_ids = list(sample_ids)
        self.H_avgs_dict = H_avgs_dict

        self.T_target = T_target
        self.alpha = alpha
        self.noise_power = noise_power
        self.C_tx = C_tx

        self.cache_preprocessed = cache_preprocessed
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None

        # basic checks
        required_cols = {
            "label", "environment", "wifi_band",
            "user_1_activity", "user_2_activity", "user_3_activity",
            "user_4_activity", "user_5_activity", "user_6_activity",
        }
        missing = required_cols.difference(self.df.columns)
        if missing:
            raise ValueError(f"annotation.csv is missing columns: {missing}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    # -----------------------------------------------------------------
    # Label helpers
    # -----------------------------------------------------------------

    def _build_y_6x9(self, row: pd.Series) -> np.ndarray:
        """
        Build 6x9 binary label matrix from user_1_activity..user_6_activity.

        Each user row:
          - if NaN: all zeros (user absent)
          - else:   one-hot at activity index k

        Returns:
          y_6x9: np.ndarray of shape (6, 9), float32 in {0,1}
        """
        y = np.zeros((6, len(ACTIVITIES)), dtype=np.float32)

        for ui in range(6):
            col = f"user_{ui + 1}_activity"
            act = row[col]
            if pd.isna(act):
                # user absent; row stays all-zero
                continue

            act_str = str(act)
            if act_str not in ACT_TO_IDX:
                raise ValueError(
                    f"Unknown activity '{act_str}' in column '{col}'. "
                    f"Expected one of {list(ACT_TO_IDX.keys())}"
                )
            k = ACT_TO_IDX[act_str]
            y[ui, k] = 1.0

        return y

    def _derive_count(self, y_6x9: np.ndarray) -> int:
        """
        True count = number of users with any non-zero activity
        (row-sum > 0).
        """
        present_mask = (y_6x9.sum(axis=1) > 0)
        return int(present_mask.sum())

    # -----------------------------------------------------------------
    # Optional disk caching of preprocessed A (C, T)
    # -----------------------------------------------------------------

    def _cache_path(self, sample_id: str) -> Optional[Path]:
        if not (self.cache_preprocessed and self.preprocessed_dir):
            return None
        return self.preprocessed_dir / f"{sample_id}.npy"

    def _maybe_load_cached(self, sample_id: str) -> Optional[np.ndarray]:
        path = self._cache_path(sample_id)
        if path is None:
            return None
        if path.exists():
            return np.load(path)
        return None

    def _maybe_save_cached(self, sample_id: str, A: np.ndarray) -> None:
        path = self._cache_path(sample_id)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, A)

    # -----------------------------------------------------------------
    # Main access
    # -----------------------------------------------------------------

    def __getitem__(self, idx: int):
        sid = self.sample_ids[idx]  # this is a 'label'

        # get row for this label
        rows = self.df[self.df["label"] == sid]
        if rows.empty:
            raise KeyError(f"label '{sid}' not found in annotation.csv")
        row = rows.iloc[0]

        # ---------- labels ----------
        y_6x9 = self._build_y_6x9(row)
        y_act = y_6x9.reshape(-1)      # (54,)
        y_count = self._derive_count(y_6x9)

        # ---------- CSI -> model input ----------
        A = self._maybe_load_cached(sid)
        if A is None:
            A = preprocess_sample(
                sample_id=sid,
                df_row=row,
                csi_amp_root=self.csi_amp_root,
                H_avgs_dict=self.H_avgs_dict,
                T_target=self.T_target,
                alpha=self.alpha,
                noise_power=self.noise_power,
                C_tx=self.C_tx,
            )
            self._maybe_save_cached(sid, A)

        x = torch.from_numpy(A).float()          # (C, T)
        y_act_t = torch.from_numpy(y_act).float()  # BCEWithLogits
        y_count_t = torch.tensor(y_count, dtype=torch.long)

        meta = {
            "label": sid,
            "environment": row["environment"],
            "wifi_band": float(row["wifi_band"]),
        }

        return x, y_act_t, y_count_t, meta
