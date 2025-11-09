# scripts/wimans_preprocess.py

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset


# ============================================================
# CONFIG (paths for your structure)
# ============================================================

# This script lives in: top/scripts/
# Dataset is in: top/training_dataset/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "training_dataset"

ANNOTATION_CSV = DATASET_ROOT / "annotation.csv"
CSI_AMP_ROOT = DATASET_ROOT / "wifi_csi" / "amp"

# Column names in annotations.csv (as you provided)
INDEX_COL = "#"                # optional, we ignore it
SAMPLE_ID_COL = "label"        # e.g. "act_1_1" -> act_1_1.npy
ENV_COL = "environment"        # e.g. "classroom", "meeting_room", "empty_room"
BAND_COL = "wifi_band"         # e.g. "2.4G", "5G" (adjust values to your CSV)
NUM_USERS_COL = "number_of_users"  # 0..5


# ============================================================
# 1. Build sample table with paths + labels
# ============================================================

def build_sample_table(
    annotation_csv: Path = ANNOTATION_CSV,
    csi_amp_root: Path = CSI_AMP_ROOT,
    num_users_col: Optional[str] = NUM_USERS_COL,
) -> pd.DataFrame:
    """
    Load annotations.csv and create a DataFrame with at least:
        - label        (string like act_1_1)
        - environment
        - wifi_band
        - number_of_users  (0..5)
        - amp_path     (path to *.npy amplitude file)

    Assumes:
      amp file path = training_dataset/wifi_csi/amp/{label}.npy
    """

    df = pd.read_csv(annotation_csv)

    # Basic checks
    for col in [SAMPLE_ID_COL, ENV_COL, BAND_COL, num_users_col]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' column in {annotation_csv}")

    # Keep only relevant columns (and whatever extras you want)
    df["num_users"] = df[num_users_col].astype(int)

    # Build absolute path to the amplitude .npy file
    def _amp_path(label: str) -> str:
        # label expected like "act_1_1"
        return str(csi_amp_root / f"{label}.npy")

    df["amp_path"] = df[SAMPLE_ID_COL].astype(str).apply(_amp_path)

    # Optional sanity check
    missing = [p for p in df["amp_path"] if not Path(p).is_file()]
    if missing:
        print(f"[WARN] {len(missing)} amplitude .npy files missing. Example: {missing[0]}")

    return df


# ============================================================
# 2. Split per environment (80/20)
# ============================================================

def split_by_environment(
    df: pd.DataFrame,
    env_col: str = ENV_COL,
    train_ratio: float = 0.8,
    random_state: int = 42,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Split each environment subset into 80% train / 20% test.

    Returns:
        {
          env_name: {
             "train": df_env_train,
             "test":  df_env_test,
          },
          ...
        }
    """
    splits: Dict[str, Dict[str, pd.DataFrame]] = {}

    if env_col not in df.columns:
        # No environment info, treat all as single env
        print(f"[WARN] '{env_col}' not found; treating as single environment 'all'.")
        df_train, df_test = train_test_split(
            df,
            test_size=1 - train_ratio,
            random_state=random_state,
            stratify=df["num_users"],
        )
        splits["all"] = {
            "train": df_train.reset_index(drop=True),
            "test": df_test.reset_index(drop=True),
        }
        return splits

    for env_name, df_env in df.groupby(env_col):
        df_train, df_test = train_test_split(
            df_env,
            test_size=1 - train_ratio,
            random_state=random_state,
            stratify=df_env["num_users"],
        )
        splits[env_name] = {
            "train": df_train.reset_index(drop=True),
            "test": df_test.reset_index(drop=True),
        }
        print(
            f"[INFO] Env='{env_name}': "
            f"{len(df_train)} train samples, {len(df_test)} test samples"
        )

    return splits


# ============================================================
# 3. PyTorch Dataset for user-count prediction
# ============================================================

class WiMANSCountDataset(Dataset):
    """
    Dataset for predicting number of users (0..5)
    from preprocessed CSI amplitude .npy files.

    Expected raw amplitude shape per file: (T_raw, 3, 3, 30)
    Output shape for model: (C=270, T_proc)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_T: int = 3000,
        downsample_factor: int = 1,
        use_band: Optional[str] = None,   # e.g. "5G" / "2.4G" depending on CSV
    ):
        """
        Args:
            df: DataFrame with at least columns 'amp_path' and 'num_users'
            target_T: pad/trim CSI time dimension to this length
            downsample_factor: keep every Nth sample in time
            use_band: filter to specific wifi_band value (exact match)
        """
        if use_band is not None:
            if BAND_COL not in df.columns:
                raise ValueError(f"BAND_COL '{BAND_COL}' missing in df.")
            df = df[df[BAND_COL] == use_band].reset_index(drop=True)
            print(f"[INFO] Filtered to wifi_band={use_band}: {len(df)} samples")

        self.df = df.reset_index(drop=True)
        self.target_T = target_T
        self.downsample_factor = downsample_factor

    def __len__(self) -> int:
        return len(self.df)

    def _load_amp(self, idx: int) -> np.ndarray:
        path = Path(self.df.loc[idx, "amp_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Amplitude file not found: {path}")
        amp = np.load(path)  # expected shape: (T_raw, 3, 3, 30)
        if amp.ndim != 4:
            raise ValueError(f"{path}: expected 4D array (T,3,3,30), got {amp.shape}")
        return amp

    def _pad_or_trim_time(self, amp: np.ndarray) -> np.ndarray:
        """
        Handle packet loss and variable T_raw by padding or trimming
        time dimension to self.target_T.
        """
        T_raw = amp.shape[0]
        if T_raw > self.target_T:
            amp = amp[: self.target_T]
        elif T_raw < self.target_T:
            pad_len = self.target_T - T_raw
            pad_shape = (pad_len,) + amp.shape[1:]
            pad = np.zeros(pad_shape, dtype=amp.dtype)
            amp = np.concatenate([amp, pad], axis=0)
        return amp

    def _downsample_time(self, amp: np.ndarray) -> np.ndarray:
        if self.downsample_factor <= 1:
            return amp
        return amp[:: self.downsample_factor]

    def _reshape_to_channels_time(self, amp: np.ndarray) -> np.ndarray:
        """
        (T_proc, 3, 3, 30) -> (C=270, T_proc)
        """
        T_proc = amp.shape[0]
        amp_flat = amp.reshape(T_proc, -1)  # (T_proc, 270)
        X = amp_flat.T                      # (270, T_proc)
        return X

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.loc[idx]
        y = int(row["num_users"])

        amp = self._load_amp(idx)                    # (T_raw, 3, 3, 30)
        amp = self._pad_or_trim_time(amp)            # (target_T, 3, 3, 30)
        amp = self._downsample_time(amp)             # (T_proc, 3, 3, 30)
        X_np = self._reshape_to_channels_time(amp)   # (270, T_proc)

        X = torch.from_numpy(X_np).float()
        y = torch.tensor(y, dtype=torch.long)

        return X, y


# ============================================================
# 4. Example usage
# ============================================================

if __name__ == "__main__":
    df_all = build_sample_table()
    splits = split_by_environment(df_all)

    # Pick an environment (e.g. "classroom") or first available
    env_name = list(splits.keys())[0]
    train_df = splits[env_name]["train"]
    test_df = splits[env_name]["test"]

    train_ds = WiMANSCountDataset(
        train_df,
        target_T=3000,
        downsample_factor=1,
        use_band=None,  # or e.g. "5G" if your wifi_band values are "5G"
    )
    test_ds = WiMANSCountDataset(
        test_df,
        target_T=3000,
        downsample_factor=1,
        use_band=None,
    )

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")

    X, y = train_ds[0]
    print("X shape:", X.shape, "label:", y.item())
