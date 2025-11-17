# data/preprocess_csi.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1) Load raw CSI amplitude for one sample
# ---------------------------------------------------------------------

def load_csi_amp(
    sample_id: str,
    csi_amp_root: str | Path,
) -> np.ndarray:
    """
    Load WiMANS CSI amplitude for a single sample.

    We assume that 'label' from annotation.csv is used as the file stem.

    Example placeholder implementation (adjust to your real structure):

        file_path = Path(csi_amp_root) / f"{sample_id}.npy"
        C_raw = np.load(file_path)  # should have shape (T_raw, 3, 3, 30)

    If your data is in .mat files, use scipy.io.loadmat, etc.
    """
    csi_amp_root = Path(csi_amp_root)

    file_path = csi_amp_root / f"{sample_id}.npy"
    if not file_path.exists():
        raise FileNotFoundError(f"CSI file not found for sample '{sample_id}': {file_path}")

    C_raw = np.load(file_path)  # e.g. (T_raw, 3, 3, 30)
    if C_raw.ndim != 4:
        raise ValueError(f"Expected 4D CSI amplitude (T, Nt, Nr, Nsc), got shape {C_raw.shape}")
    return C_raw


# ---------------------------------------------------------------------
# 2) Time normalization to T_target
# ---------------------------------------------------------------------

def normalize_time_axis(
    C_raw: np.ndarray,
    T_target: int,
) -> np.ndarray:
    """
    Normalize time dimension to T_target via center-crop or zero-pad.

    Input:  C_raw (T_raw, Nt, Nr, Nsc)
    Output: C      (T_target, Nt, Nr, Nsc)
    """
    T_raw = C_raw.shape[0]

    if T_raw > T_target:
        start = (T_raw - T_target) // 2
        C = C_raw[start:start + T_target]
    elif T_raw < T_target:
        pad_len = T_target - T_raw
        pad_shape = (pad_len,) + C_raw.shape[1:]
        pad = np.zeros(pad_shape, dtype=C_raw.dtype)
        C = np.concatenate([C_raw, pad], axis=0)
    else:
        C = C_raw

    return C


# ---------------------------------------------------------------------
# 3) Scaling (Scaled(C)) – simplified amplitude scaling
# ---------------------------------------------------------------------

def scale_csi(
    C: np.ndarray,
    alpha: float,
    noise_power: float,
    C_tx: float,
) -> np.ndarray:
    """
    Apply CSI amplitude scaling.

    Here we approximate with a scalar factor:

        |H_scaled| = |H| * sqrt(alpha / P_noise) * C_tx

    Input/Output: (T, Nt, Nr, Nsc)
    """
    if noise_power <= 0:
        raise ValueError("noise_power must be > 0 for scaling")

    scale_factor = np.sqrt(alpha / noise_power) * C_tx
    C_scaled = C * scale_factor
    return C_scaled


# ---------------------------------------------------------------------
# 4) Static average per (env, wifi_band)
# ---------------------------------------------------------------------

def compute_static_average(
    sample_ids: List[str],
    annotation_csv: str | Path,
    csi_amp_root: str | Path,
    T_target: int,
    alpha: float,
    noise_power: float,
    C_tx: float,
) -> Dict[Tuple[str, float], np.ndarray]:
    """
    Compute H_AVGS per (environment, wifi_band) using 0-user samples.

    annotation.csv columns (relevant):
      'label', 'environment', 'wifi_band', 'number_of_users',
      'user_1_activity', ..., 'user_6_activity'

    0-user samples: number_of_users == 0

    Returns dict:
        (env, band) -> H_AVGS with shape (T_target, Nt, Nr, Nsc)
    """
    csi_amp_root = Path(csi_amp_root)
    df = pd.read_csv(annotation_csv)

    required_cols = {
        "label", "environment", "wifi_band", "number_of_users",
        "user_1_activity", "user_2_activity", "user_3_activity",
        "user_4_activity", "user_5_activity", "user_6_activity",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"annotation.csv is missing columns: {missing}")

    # Map (env, band) -> list of 0-user labels
    mapping: Dict[Tuple[str, float], List[str]] = {}

    df_sub = df[df["label"].isin(sample_ids)].copy()

    for _, row in df_sub.iterrows():
        sid = row["label"]
        env = row["environment"]
        band = float(row["wifi_band"])
        num_users = int(row["number_of_users"])

        # 0-user samples only
        if num_users != 0:
            continue

        key = (env, band)
        mapping.setdefault(key, []).append(sid)

    static_avgs: Dict[Tuple[str, float], np.ndarray] = {}

    for key, sids in mapping.items():
        if len(sids) == 0:
            continue

        accum = []
        for sid in sids:
            C_raw = load_csi_amp(sid, csi_amp_root)
            C_norm = normalize_time_axis(C_raw, T_target)
            C_scaled = scale_csi(C_norm, alpha=alpha, noise_power=noise_power, C_tx=C_tx)
            accum.append(C_scaled)

        static_avgs[key] = np.mean(np.stack(accum, axis=0), axis=0)

    if not static_avgs:
        raise RuntimeError(
            "compute_static_average: no 0-user samples found. "
            "Check your annotation.csv or sample_ids."
        )

    return static_avgs


# ---------------------------------------------------------------------
# 5) Dynamic component + reshape to (C, T)
# ---------------------------------------------------------------------

def compute_scaled_dynamic(
    C_scaled: np.ndarray,
    H_avgs_env_band: np.ndarray,
) -> np.ndarray:
    """
    Compute dynamic component:

        C_dyn = |C_scaled - H_AVGS|

    Inputs:
        C_scaled, H_AVGS: (T, Nt, Nr, Nsc)

    Output:
        C_dyn: (T, Nt, Nr, Nsc)
    """
    if C_scaled.shape != H_avgs_env_band.shape:
        raise ValueError(
            f"Shape mismatch between C_scaled {C_scaled.shape} "
            f"and H_AVGS {H_avgs_env_band.shape}"
        )
    C_dyn = np.abs(C_scaled - H_avgs_env_band)
    return C_dyn


def reshape_to_model_input(C_dyn: np.ndarray) -> np.ndarray:
    """
    Convert (T, Nt, Nr, Nsc) -> (C, T) with C = Nt * Nr * Nsc.

    This matches WiMUAR input shape (270, T).
    """
    T, Nt, Nr, Nsc = C_dyn.shape
    C = Nt * Nr * Nsc
    A = C_dyn.reshape(T, C).transpose(1, 0)  # (C, T)
    return A


# ---------------------------------------------------------------------
# 6) End-to-end preprocess for one sample
# ---------------------------------------------------------------------

def preprocess_sample(
    sample_id: str,
    df_row: pd.Series,
    csi_amp_root: str | Path,
    H_avgs_dict: Dict[Tuple[str, float], np.ndarray],
    T_target: int,
    alpha: float,
    noise_power: float,
    C_tx: float,
) -> np.ndarray:
    """
    End-to-end preprocessing pipeline P(C):

        1) load raw CSI amplitude
        2) normalize time axis to T_target
        3) scaling: Scaled(C)
        4) subtract environment+band static avg: H_AVGS
        5) take |dynamic| and reshape to (C, T)

    Returns:
        A: np.ndarray with shape (C, T_target)
    """
    csi_amp_root = Path(csi_amp_root)

    # 1) load raw CSI   (sample_id is the 'label')
    C_raw = load_csi_amp(sample_id, csi_amp_root)

    # 2) time normalization
    C_norm = normalize_time_axis(C_raw, T_target)

    # 3) scaling
    C_scaled = scale_csi(C_norm, alpha=alpha, noise_power=noise_power, C_tx=C_tx)

    # 4) get H_AVGS for this (env, band)
    env = df_row["environment"]
    band = float(df_row["wifi_band"])
    key = (env, band)
    if key not in H_avgs_dict:
        raise KeyError(
            f"No static average found for env='{env}', wifi_band={band}. "
            "Check compute_static_average input sample_ids."
        )
    H_avgs = H_avgs_dict[key]

    # 5) dynamic + reshape
    C_dyn = compute_scaled_dynamic(C_scaled, H_avgs)
    A = reshape_to_model_input(C_dyn)  # (C, T_target)
    return A
