"""
WiMANS: .mat -> phase (and amplitude) .npy exporter
- Extracts complex CSI (T, 3, 3, 30) from each .mat in wifi_csi/mat
- Computes amplitude and phase
- Phase post-processing: unwrap (time), detrend (time)
- (Optional) pad/crop to fixed length (e.g., 3000)
- Saves:
    wifi_csi/phase/act_x_y.npy         # processed phase, shape (T, 3, 3, 30) or (T, 270)
    wifi_csi/amp_from_mat/act_x_y.npy  # optional amplitude generated from the same pipeline
Notes:
- Keeping (T, 3, 3, 30) is nice for future methods; for 1D-CNN you’ll reshape to (T, 270) later.
"""

import os
import numpy as np
import scipy.io
import scipy.signal
from multiprocessing import Pool, cpu_count

# ====================== CONFIG ======================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "training_dataset"))
MAT_DIR   = os.path.join(DATASET_BASE_DIR, "wifi_csi", "mat")
PHASE_DIR = os.path.join(DATASET_BASE_DIR, "wifi_csi", "phase")
AMP2_DIR  = os.path.join(DATASET_BASE_DIR, "wifi_csi", "amp_from_mat")  # opsiyonel
MAKE_AMP_TOO = True               # we also want to save amplitude again
APPLY_UNWRAP = True
APPLY_DETREND = True              
FIX_LEN = True                    # we need fixed-length inputs for training
TARGET_LEN = 3000                 
SAVE_FLATTENED = False            #  False means save (T,3,3,30)

# ====================================================

def ensure_dirs():
    os.makedirs(PHASE_DIR, exist_ok=True)
    if MAKE_AMP_TOO:
        os.makedirs(AMP2_DIR, exist_ok=True)

def extract_csi_complex_from_mat(mat_path):
    """
    Returns:
        csi: np.ndarray, shape (T, 3, 3, 30), dtype=complex
        or None if fails
    """
    try:
        file = scipy.io.loadmat(mat_path)
        trace = file["trace"]
    except Exception as e:
        print(f"[ERROR] loadmat failed: {mat_path} -> {e}")
        return None

    T = trace.shape[0]  # number of time steps
    csi_list = []
    for i in range(T):
        try:
            csi = trace[i, 0]["csi"][0, 0]  # (3,3,30) complex
            if csi.shape == (3, 3, 30):
                csi_list.append(csi)
            else:
                print(f"[WARN] {os.path.basename(mat_path)} step {i}: shape {csi.shape}, skip")
        except Exception:
            # malformed step
            continue

    if not csi_list:
        return None

    csi_arr = np.asarray(csi_list)  # (T_valid, 3, 3, 30)
    # cast to complex64 to save space
    if csi_arr.dtype != np.complex64:
        csi_arr = csi_arr.astype(np.complex64, copy=False)
    return csi_arr

def complex_to_amp_phase(csi_complex):
    amp = np.abs(csi_complex).astype(np.float32, copy=False)
    phase = np.angle(csi_complex).astype(np.float32, copy=False)
    return amp, phase

def phase_postprocess(phase):
    """
    phase: (T, 3, 3, 30), float32 (radians)
    Applies unwrap + detrend along time axis.
    """
    out = phase
    if APPLY_UNWRAP:
        # unwrap expects radians; axis=0 is time
        out = np.unwrap(out, axis=0)
        out = out.astype(np.float32, copy=False)
    if APPLY_DETREND:
        out = scipy.signal.detrend(out, axis=0, type="linear")
        out = out.astype(np.float32, copy=False)
    return out

def pad_or_crop_time(x, target_len):
    """
    x: (T, ...), returns (target_len, ...)
    """
    T = x.shape[0]
    if T == target_len:
        return x
    if T > target_len:
        return x[:target_len]
    # pad with zeros on time axis
    pad_shape = (target_len - T,) + x.shape[1:]
    pad = np.zeros(pad_shape, dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)

def maybe_fix_len(amp, phase):
    if not FIX_LEN:
        return amp, phase
    amp2 = pad_or_crop_time(amp, TARGET_LEN)
    phase2 = pad_or_crop_time(phase, TARGET_LEN)
    return amp2, phase2

def maybe_flatten_last(amp, phase):
    """
    Flatten (3,3,30) -> 270 on the channel axis, keeping time first.
    Outputs shape: (T, 270)
    """
    if not SAVE_FLATTENED:
        return amp, phase
    T = phase.shape[0]
    amp2   = amp.reshape(T, -1)   # (T, 270)
    phase2 = phase.reshape(T, -1) # (T, 270)
    return amp2, phase2

def save_npy_safe(path, arr):
    np.save(path, arr)
    print(f"[OK] saved -> {path}  shape={arr.shape}  dtype={arr.dtype}")

def process_one(mat_fname):
    if not mat_fname.endswith(".mat"):
        return
    base = os.path.splitext(mat_fname)[0]  # e.g., act_1_1
    phase_out = os.path.join(PHASE_DIR, f"{base}.npy")
    amp_out   = os.path.join(AMP2_DIR,  f"{base}.npy")

    # skip if already exists (idempotent)
    if os.path.exists(phase_out) and (not MAKE_AMP_TOO or os.path.exists(amp_out)):
        return

    mat_path = os.path.join(MAT_DIR, mat_fname)
    csi = extract_csi_complex_from_mat(mat_path)
    if csi is None:
        print(f"[WARN] no valid CSI in {mat_fname}")
        return

    amp, phase = complex_to_amp_phase(csi)
    phase = phase_postprocess(phase)
    amp, phase = maybe_fix_len(amp, phase)
    amp, phase = maybe_flatten_last(amp, phase)

    # save
    save_npy_safe(phase_out, phase)
    if MAKE_AMP_TOO:
        save_npy_safe(amp_out, amp)

def main():
    ensure_dirs()
    files = [f for f in os.listdir(MAT_DIR) if f.endswith(".mat")]
    if not files:
        print(f"[ERROR] no .mat files in {MAT_DIR}")
        return
    with Pool(cpu_count()) as pool:
        pool.map(process_one, files)

if __name__ == "__main__":
    main()
