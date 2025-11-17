# WiMANS SADU: Multi-user Activity + Human Count from WiFi CSI

This repo implements a SADU-style model on the WiMANS dataset for:
- Multi-user activity recognition (per-user activities).
- Human count estimation **derived from activities** (no separate count head).

## Folder structure

```text
wimans_sadu/
  ├─ configs/
  │   ├─ wimans_5g_pp.yaml
  │   └─ wimans_5g_pp_baseline.yaml
  ├─ scripts/
  │   ├─ preprocess_wimans.py
  │   ├─ build_splits.py
  │   ├─ train_sadu.py
  │   ├─ eval_sadu.py
  │   ├─ infer_demo.py
  │   ├─ train_baseline_cnn.py
  │   ├─ run_cross_env_experiments.py
  │   └─ check_pipeline.py
  ├─ wimans/
  │   ├─ transforms.py
  │   ├─ labels.py
  │   └─ dataset.py
  ├─ models/
  │   ├─ backbone.py
  │   ├─ heads.py
  │   ├─ sadu.py
  │   └─ baseline_cnn.py
  ├─ training/
  │   ├─ losses.py
  │   ├─ metrics.py
  │   └─ loop.py
  ├─ training_dataset/
  │   ├─ annotation.csv
  │   └─ wifi_csi/mat/*.mat
  └─ experiments/
      └─ ...
