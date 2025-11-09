# preprocess_amp.py
import numpy as np
import scipy.signal as sig
from sklearn.decomposition import PCA

FS = 1000
LOWPASS_MAX_HZ = 20   # jump ağır basıyorsa 30–40 da deneyebilirsin
CHEBY_ORDER = 5
N_PCA = 5             # her link için 30 -> N_PCA; toplam D = 9*N_PCA

def cheby2_lowpass(x, fs=FS, cutoff=LOWPASS_MAX_HZ, order=CHEBY_ORDER):
    sos = sig.cheby2(order, rs=20, Wn=cutoff, btype='low', fs=fs, output='sos')
    return sig.sosfiltfilt(sos, x, axis=0)

def svd_background_remove(H):
    U, s, Vt = np.linalg.svd(H, full_matrices=False)
    if s.size > 0:
        s[0] = 0.0
    return (U * s) @ Vt

def pca_per_link(amp_link, n_pca=N_PCA):
    p = PCA(n_components=min(n_pca, amp_link.shape[1]))
    return p.fit_transform(amp_link)

def pad_or_crop(x, target_len):
    T, C = x.shape
    if T == target_len:
        return x
    if T > target_len:
        return x[:target_len]
    pad = np.zeros((target_len - T, C), dtype=x.dtype)
    return np.vstack([x, pad])

def ensure_T_9_30(amp):
    if amp.ndim == 4 and amp.shape[1:4] == (3,3,30):
        T = amp.shape[0]; return amp.reshape(T, 9, 30)
    if amp.ndim == 2 and amp.shape[1] == 270:
        T = amp.shape[0]; return amp.reshape(T, 9, 30)
    raise ValueError(f"Amplitude shape not recognized: {amp.shape}")

def build_amp_features(amp_array, target_len=3000, use_svd=True, use_lowpass=True, n_pca=N_PCA):
    amp = ensure_T_9_30(amp_array)  # (T,9,30)

    if use_svd:
        amp2 = np.empty_like(amp)
        for li in range(9):
            A = amp[:, li, :] - amp[:, li, :].mean(axis=0, keepdims=True)
            amp2[:, li, :] = svd_background_remove(A)
        amp = amp2

    if use_lowpass:
        for li in range(9):
            amp[:, li, :] = cheby2_lowpass(amp[:, li, :])

    feats = [pca_per_link(amp[:, li, :], n_pca=n_pca) for li in range(9)]
    X = np.concatenate(feats, axis=1)  # (T, 9*n_pca)
    X = pad_or_crop(X, target_len)
    return X.astype(np.float32)
