# wimans/transforms.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from typing import Tuple
import numpy as np
from scipy.io import loadmat
import pywt
import torch


def load_csi_mat(path: Path, key: str = "csi") -> np.ndarray:
    """
    Load CSI from a WiMANS .mat file.

    Two possible formats:

    1) Top-level 'csi' variable: csi.shape == (T, 3, 3, 30)
    2) Intel 5300-like 'trace' array of structs:
         mat['trace'] is (T,) object array
         mat['trace'][i].csi is (3, 3, 30) complex

    This function handles both and returns:
      csi: (T, 3, 3, 30) as a **real** numpy array (amplitude).
    """
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)

    # Case 1: direct 'csi' variable
    if key in mat:
        csi = mat[key]
        # Ensure temporal dimension is first
        if csi.ndim == 3 and csi.shape == (3, 3, 30):
            csi = csi[np.newaxis, ...]  # (1, 3, 3, 30)
        elif csi.ndim != 4:
            raise ValueError(f"Unexpected 'csi' shape {csi.shape} in {path}")
    # Case 2: WiMANS / Intel 5300 format with 'trace' struct array
    elif "trace" in mat:
        trace = mat["trace"]  # (T,) array of mat_struct
        # make sure it's 1D
        trace = np.atleast_1d(trace)
        T = trace.shape[0]
        csi_list = []
        for i in range(T):
            rec = trace[i]
            if not hasattr(rec, "csi"):
                raise ValueError(f"'trace[{i}]' has no 'csi' field in {path}")
            cs = rec.csi  # expected (3, 3, 30)
            cs = np.asarray(cs)
            if cs.ndim == 3:
                csi_list.append(cs[np.newaxis, ...])  # (1, 3, 3, 30)
            elif cs.ndim == 4 and cs.shape[0] == 1:
                csi_list.append(cs)  # already (1, 3, 3, 30)
            else:
                raise ValueError(f"Unexpected 'trace[{i}].csi' shape {cs.shape} in {path}")
        csi = np.concatenate(csi_list, axis=0)  # (T, 3, 3, 30)
    else:
        raise KeyError(
            f"Neither key '{key}' nor 'trace' found in {path}. "
            f"Available keys: {list(mat.keys())}"
        )

    # Convert complex CSI to amplitude (real)
    if np.iscomplexobj(csi):
        csi = np.abs(csi)

    return csi  # (T, 3, 3, 30)


def reshape_csi(csi: np.ndarray) -> np.ndarray:
    """
    Reshape CSI from (T, 3, 3, 30) to (T, 270).
    """
    T = csi.shape[0]
    return csi.reshape(T, -1)  # (T, 270)


def wavelet_denoise_channel(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int = 2
) -> np.ndarray:
    """
    Wavelet denoise a single 1D signal:
      - DWT (db4, 2 levels)
      - Estimate noise via MAD
      - Soft-threshold
      - Inverse DWT
    """
    coeffs = pywt.wavedec(x, wavelet, level=level)
    detail_coeffs = coeffs[-1]
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    tau = sigma * np.sqrt(2 * np.log(len(x)))

    coeffs_thresh = [coeffs[0]] + [
        pywt.threshold(c, tau, mode="soft") for c in coeffs[1:]
    ]
    x_denoised = pywt.waverec(coeffs_thresh, wavelet)

    # Ensure same length
    if len(x_denoised) < len(x):
        pad_len = len(x) - len(x_denoised)
        x_denoised = np.concatenate([x_denoised, np.repeat(x_denoised[-1], pad_len)])
    return x_denoised[:len(x)]


def wavelet_denoise_csi(csi_2d: np.ndarray) -> np.ndarray:
    """
    Apply wavelet denoising channel-wise.

    csi_2d: (T, 270)
    """
    T, C = csi_2d.shape
    out = np.empty_like(csi_2d)
    for ch in range(C):
        out[:, ch] = wavelet_denoise_channel(csi_2d[:, ch])
    return out


def pad_or_crop(csi_2d: np.ndarray, target_T: int) -> np.ndarray:
    """
    Center-crop or pad along time dimension to target_T.

    csi_2d: (T, 270) -> (target_T, 270)
    """
    T, C = csi_2d.shape
    if T == target_T:
        return csi_2d
    if T > target_T:
        start = (T - target_T) // 2
        return csi_2d[start:start + target_T, :]
    else:
        pad_len = target_T - T
        pad = np.repeat(csi_2d[-1:, :], pad_len, axis=0)
        return np.vstack([csi_2d, pad])


def preprocess_csi_sample(
    csi_mat_path: Path,
    target_T: int,
    use_wavelet_pp: bool = True,
    csi_key: str = "csi"
) -> torch.Tensor:
    """
    Full preprocessing for one sample:
      1) Load mat (T, 3, 3, 30)
      2) Reshape to (T, 270)
      3) Wavelet denoise (optional)
      4) Pad/crop to (target_T, 270)
      5) Transpose to (270, target_T)
    """
    csi = load_csi_mat(csi_mat_path, key=csi_key)
    csi_2d = reshape_csi(csi)
    if use_wavelet_pp:
        csi_2d = wavelet_denoise_csi(csi_2d)
    csi_2d = pad_or_crop(csi_2d, target_T)
    x = torch.from_numpy(csi_2d.T).float()  # (270, target_T)
    return x
