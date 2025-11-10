# scripts/wimans_preprocess.py

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import torch
from torch.utils.data import Dataset

from scipy.io import loadmat
from scipy.signal import butter, filtfilt, stft
import pywt


# ============================================================
# CONFIG (paths for your structure)
# ============================================================

# This script lives in: top/scripts/
# Dataset is in: top/training_dataset/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "training_dataset"

ANNOTATION_CSV = DATASET_ROOT / "annotation.csv"
CSI_AMP_ROOT = DATASET_ROOT / "wifi_csi" / "amp"
CSI_MAT_ROOT = DATASET_ROOT / "wifi_csi" / "mat"

# Column names in annotations.csv
INDEX_COL = "#"                # optional, we ignore it
SAMPLE_ID_COL = "label"        # e.g. "act_1_1" -> act_1_1.npy / .mat
ENV_COL = "environment"        # e.g. "classroom", "meeting_room", "empty_room"
BAND_COL = "wifi_band"         # e.g. "2.4", "5"
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
        - label
        - environment
        - wifi_band
        - number_of_users  (0..5)
        - amp_path   (path to *.npy amplitude file)
        - mat_path   (path to *.mat complex CSI file)
    """

    df = pd.read_csv(annotation_csv)

    # Basic checks
    for col in [SAMPLE_ID_COL, ENV_COL, BAND_COL, num_users_col]:
        if col not in df.columns:
            raise ValueError(f"Missing '{col}' column in {annotation_csv}")

    df["num_users"] = df[num_users_col].astype(int)

    def _amp_path(label: str) -> str:
        return str(csi_amp_root / f"{label}.npy")

    def _mat_path(label: str) -> str:
        return str(CSI_MAT_ROOT / f"{label}.mat")

    df["amp_path"] = df[SAMPLE_ID_COL].astype(str).apply(_amp_path)
    df["mat_path"] = df[SAMPLE_ID_COL].astype(str).apply(_mat_path)

    # Optional sanity checks
    missing_amp = [p for p in df["amp_path"] if not Path(p).is_file()]
    if missing_amp:
        print(f"[WARN] {len(missing_amp)} amplitude .npy files missing. Example: {missing_amp[0]}")

    missing_mat = [p for p in df["mat_path"] if not Path(p).is_file()]
    if missing_mat:
        print(f"[WARN] {len(missing_mat)} CSI .mat files missing. Example: {missing_mat[0]}")

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
# 3. Helper functions for preprocessing
# ============================================================

def dwt_denoise_1d(x: np.ndarray, wavelet: str = "db4") -> np.ndarray:
    """
    1D DWT denoising with soft thresholding on detail coefficients.
    """
    coeffs = pywt.wavedec(x, wavelet, mode="symmetric")
    cA, *details = coeffs
    if not details or details[-1].size == 0:
        return x

    detail = details[-1]
    sigma = np.median(np.abs(detail)) / 0.6745 if detail.size > 0 else 0.0
    if sigma <= 0:
        return x

    thr = sigma * np.sqrt(2 * np.log(len(x)))
    new_coeffs = [cA]
    for d in details:
        new_coeffs.append(pywt.threshold(d, thr, mode="soft"))
    x_rec = pywt.waverec(new_coeffs, wavelet, mode="symmetric")
    return x_rec[: len(x)]


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Band-pass filter along axis=0 (time).
    data: (T, D) or (T,) real.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0)


# ============================================================
# 4. PyTorch Dataset for user-count prediction
# ============================================================

class WiMANSCountDataset(Dataset):
    """
    Dataset for predicting number of users (0..5)
    from preprocessed CSI data.

    Raw amplitude: .npy, shape (T_raw, 3, 3, 30)
    Raw complex CSI: .mat, expected key 'csi' with shape (T_raw, 3, 3, 30) (complex)

    Output X:
      - If use_doppler=False: amplitude only, shape (270, T_proc)
      - If use_doppler=True:  joint [amp + Doppler], shape (C_total, T_proc)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_T: int = 3000,
        downsample_factor: int = 1,
        use_band: Optional[str] = None,   # "2.4" / "5" or None
        use_dwt: bool = True,             # amplitude DWT denoising
        use_doppler: bool = False,        # Doppler extraction
        fs: float = 100.0,                # sampling rate [Hz] (adjust to your data)
        doppler_pca_components: int = 3,
        stft_nperseg: int = 256,
        stft_noverlap: int = 128,
        bpf_low: float = 0.5,
        bpf_high: float = 20.0,
    ):
        """
        Args:
            df: DataFrame with at least 'amp_path', 'mat_path', 'num_users'
            target_T: pad/trim CSI time dimension to this length
            downsample_factor: keep every Nth packet in time
            use_band: filter by wifi_band (numeric-like strings "2.4"/"5"), or None
            use_dwt: apply DWT denoising on amplitude
            use_doppler: compute Doppler spectrogram features and concatenate
        """
        df = df.copy()

        if use_band is not None:
            if BAND_COL not in df.columns:
                raise ValueError(f"BAND_COL '{BAND_COL}' missing in df.")
            band_vals = pd.to_numeric(df[BAND_COL], errors="coerce")
            target_val = float(use_band)
            mask = np.isclose(band_vals, target_val)
            df = df[mask].reset_index(drop=True)
            print(
                f"[INFO] Filtered to wifi_band≈{target_val}: "
                f"{len(df)} samples"
            )

        self.df = df.reset_index(drop=True)
        self.target_T = target_T
        self.downsample_factor = downsample_factor

        self.use_dwt = use_dwt
        self.use_doppler = use_doppler
        self.fs = fs
        self.doppler_pca_components = doppler_pca_components
        self.stft_nperseg = stft_nperseg
        self.stft_noverlap = stft_noverlap
        self.bpf_low = bpf_low
        self.bpf_high = bpf_high

    def __len__(self) -> int:
        return len(self.df)

    # ---------- Loading ----------

    def _load_amp(self, idx: int) -> np.ndarray:
        path = Path(self.df.loc[idx, "amp_path"])
        if not path.is_file():
            raise FileNotFoundError(f"Amplitude file not found: {path}")
        amp = np.load(path)  # expected shape: (T_raw, 3, 3, 30)
        if amp.ndim != 4:
            raise ValueError(f"{path}: expected 4D array (T,3,3,30), got {amp.shape}")
        return amp

    def _load_csi_complex(self, idx: int) -> np.ndarray:
        """
        Load complex CSI from Intel 5300-style .mat file.

        File structure (after loadmat):

            mat['trace'] -> ndarray of shape (T, 1)
              each element trace[t, 0] is a (1, 1) structured array
              the actual record is trace[t,0][0,0], with fields:
                ('timestamp_low', ..., 'csi')

        We extract 'csi' for each packet and stack into
        Hc of shape (T, 3, 3, 30) (complex).
        """
        path = Path(self.df.loc[idx, "mat_path"])
        if not path.is_file():
            raise FileNotFoundError(f"CSI .mat file not found: {path}")

        mat = loadmat(path)
        if "trace" not in mat:
            raise KeyError(
                f"'trace' key not found in {path.name}. "
                f"Available keys: {list(mat.keys())}"
            )

        trace = mat["trace"]          # shape (T, 1)
        T = trace.shape[0]
        csi_list = []

        for t in range(T):
            # trace[t, 0] is a (1,1) structured array
            entry = trace[t, 0]
            if not isinstance(entry, np.ndarray) or entry.dtype.fields is None:
                raise ValueError(
                    f"{path.name}: unexpected trace element type "
                    f"{type(entry)}, expected structured ndarray."
                )

            # get scalar structured record (np.void) with fields incl. 'csi'
            rec = entry[0, 0]

            if "csi" not in rec.dtype.names:
                raise KeyError(
                    f"'csi' field not found in trace element in {path.name}. "
                    f"Fields: {rec.dtype.names}"
                )

            # rec['csi'] is already ndarray, e.g. (3,3,30)
            csi = np.array(rec["csi"])

            # in case there are extra singleton dims, squeeze once
            csi = np.squeeze(csi)

            if csi.ndim != 3:
                raise ValueError(
                    f"{path.name}: unexpected CSI shape {csi.shape} "
                    f"(expected 3D like (3,3,30))"
                )

            csi_list.append(csi)

        if not csi_list:
            raise ValueError(f"No CSI entries extracted from {path.name}")

        # Stack over time dimension: (T_raw, 3, 3, 30)
        Hc = np.stack(csi_list, axis=0).astype(np.complex64)
        return Hc

    # ---------- Time handling ----------

    def _pad_or_trim_time(self, x: np.ndarray) -> np.ndarray:
        """
        Pad or trim along axis=0 (time) to self.target_T.
        x: shape (T_raw, ...)
        """
        T_raw = x.shape[0]
        if T_raw > self.target_T:
            x = x[: self.target_T]
        elif T_raw < self.target_T:
            pad_len = self.target_T - T_raw
            pad_shape = (pad_len,) + x.shape[1:]
            pad = np.zeros(pad_shape, dtype=x.dtype)
            x = np.concatenate([x, pad], axis=0)
        return x

    def _downsample_time_array(self, x: np.ndarray) -> np.ndarray:
        if self.downsample_factor <= 1:
            return x
        return x[:: self.downsample_factor]

    # ---------- Amplitude denoising + reshape ----------

    def _dwt_denoise_amp(self, amp: np.ndarray) -> np.ndarray:
        """
        Apply DWT denoising along time for each (rx, tx, subcarrier) stream.
        amp: (T, 3, 3, 30)
        """
        T, nr, nt, ns = amp.shape
        amp_denoised = np.zeros_like(amp)
        for r in range(nr):
            for t in range(nt):
                for s in range(ns):
                    ts = amp[:, r, t, s]
                    amp_denoised[:, r, t, s] = dwt_denoise_1d(ts)
        return amp_denoised

    def _reshape_amp_to_channels_time(self, amp: np.ndarray) -> np.ndarray:
        """
        (T_proc, 3, 3, 30) -> (C=270, T_proc)
        """
        T_proc = amp.shape[0]
        amp_flat = amp.reshape(T_proc, -1)  # (T_proc, 270)
        X_amp = amp_flat.T                  # (270, T_proc)
        return X_amp

    # ---------- Doppler Shift (DS) extraction ----------

    def _compute_doppler_features(self, idx: int, target_T_proc: int) -> np.ndarray:
        """
        Compute Doppler spectrogram features for sample idx and
        return as (C_ds, T_proc) so it can be concatenated with amplitude.

        Steps:
          - Load complex CSI: Hc(t, rx, tx, sc)
          - Pad/trim + downsample in time
          - Conjugate multiplication across rx antenna pairs
          - Phase extraction + unwrapping
          - Band-pass filter (0.5–20 Hz default)
          - PCA across links -> (T_proc, n_components)
          - STFT on each component -> time-frequency magnitude
          - Resample STFT time dimension to T_proc
        """
        Hc = self._load_csi_complex(idx)      # (T_raw, nr, nt, ns)
        #print("Hc shape:", Hc.shape)  
        Hc = self._pad_or_trim_time(Hc)
        Hc = self._downsample_time_array(Hc)
        T_proc = Hc.shape[0]

        # Conjugate multiplication across rx antenna pairs
        T, nr, nt, ns = Hc.shape
        pair_signals = []
        for r1 in range(nr - 1):
            for r2 in range(r1 + 1, nr):
                # shape (T, nt, ns)
                c = Hc[:, r1, :, :] * np.conj(Hc[:, r2, :, :])
                pair_signals.append(c)
        if not pair_signals:
            # Fallback: no antenna pairs -> just use raw phase
            pair_stack = Hc
        else:
            # (n_pairs, T, nt, ns) -> (T, n_pairs, nt, ns)
            pair_stack = np.stack(pair_signals, axis=0).transpose(1, 0, 2, 3)

        # Phase extraction and unwrapping
        phase = np.angle(pair_stack)        # (T, n_pairs, nt, ns)
        phase = np.unwrap(phase, axis=0)
        phase_flat = phase.reshape(T, -1)   # (T, D_links)

        # Band-pass filter in time
        phase_filt = butter_bandpass_filter(
            phase_flat,
            lowcut=self.bpf_low,
            highcut=self.bpf_high,
            fs=self.fs,
            order=4,
        )  # (T, D_links)

        # PCA across link dimension
        pca = PCA(n_components=self.doppler_pca_components)
        phase_pca = pca.fit_transform(phase_filt)  # (T, n_components), real

        # STFT for each PCA component
        spec_list = []
        for i in range(self.doppler_pca_components):
            x_i = phase_pca[:, i]
            f, t_spec, Zxx = stft(
                x_i,
                fs=self.fs,
                nperseg=self.stft_nperseg,
                noverlap=self.stft_noverlap,
                padded=False,
                boundary=None,
            )
            # Zxx: (F, T_frames)
            mag = np.abs(Zxx).T   # (T_frames, F)
            spec_list.append(mag)

        # Make all components same time length and concatenate in feature dim
        min_T_frames = min(s.shape[0] for s in spec_list)
        spec_trim = [s[:min_T_frames] for s in spec_list]
        D = np.concatenate(spec_trim, axis=1)  # (T_frames, F * n_components)

        # Resample DS time dimension to match T_proc / target_T_proc
        T_frames, D_feat = D.shape
        if T_frames != target_T_proc:
            old_idx = np.linspace(0.0, 1.0, T_frames)
            new_idx = np.linspace(0.0, 1.0, target_T_proc)
            D_resampled = np.empty((target_T_proc, D_feat), dtype=D.dtype)
            for j in range(D_feat):
                D_resampled[:, j] = np.interp(new_idx, old_idx, D[:, j])
            D = D_resampled

        # Return (C_ds, T_proc)
        return D.T

    # ---------- __getitem__ ----------

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.loc[idx]
        y = int(row["num_users"])

        # 1) Amplitude
        amp = self._load_amp(idx)          # (T_raw, 3, 3, 30)
        amp = self._pad_or_trim_time(amp)  # (target_T, 3, 3, 30)

        if self.use_dwt:
            amp = self._dwt_denoise_amp(amp)

        amp = self._downsample_time_array(amp)  # (T_proc, 3, 3, 30)
        T_proc = amp.shape[0]
        X_amp = self._reshape_amp_to_channels_time(amp)   # (270, T_proc)

        # 2) Doppler features (optional)
        if self.use_doppler:
            X_ds = self._compute_doppler_features(idx, T_proc)  # (C_ds, T_proc)
            X_np = np.concatenate([X_amp, X_ds], axis=0)        # (C_total, T_proc)
        else:
            X_np = X_amp

        X = torch.from_numpy(X_np).float()
        y = torch.tensor(y, dtype=torch.long)

        return X, y


# ============================================================
# 5. Example usage
# ============================================================

if __name__ == "__main__":
    df_all = build_sample_table()
    splits = split_by_environment(df_all)

    env_name = list(splits.keys())[0]
    train_df = splits[env_name]["train"]
    test_df = splits[env_name]["test"]

    train_ds = WiMANSCountDataset(
        train_df,
        target_T=3000,
        downsample_factor=1,
        use_band=None,
        use_dwt=True,
        use_doppler=True,   # turn on Doppler here to test
    )
    test_ds = WiMANSCountDataset(
        test_df,
        target_T=3000,
        downsample_factor=1,
        use_band=None,
        use_dwt=True,
        use_doppler=True,
    )

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    X, y = train_ds[0]
    print("X shape:", X.shape, "label:", y.item())
