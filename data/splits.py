# data/splits.py

from typing import Tuple, Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_env_with_val(
    annotation_csv: str,
    train_envs: Tuple[str, ...],
    test_env: str,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Dict[str, List[str]]:
    """
    Split WiMANS samples into train/val/test based on environment.

    Your annotation.csv has columns:
      '#', 'label', 'environment', 'wifi_band', 'number_of_users',
      'user_1_location', ..., 'user_6_location',
      'user_1_activity', ..., 'user_6_activity'

    We treat 'label' as the sample ID.

    Returns a dict of ID lists (labels):
        {
          "train_ids": [...],
          "val_ids": [...],
          "test_ids": [...]
        }

    - train + val come only from train_envs
    - test comes only from test_env
    """
    df = pd.read_csv(annotation_csv)

    if "label" not in df.columns:
        raise ValueError("annotation.csv must contain a 'label' column")

    if "environment" not in df.columns:
        raise ValueError("annotation.csv must contain an 'environment' column")

    # Train+Val subset
    df_trainval = df[df["environment"].isin(train_envs)].copy()
    if df_trainval.empty:
        raise ValueError(f"No samples found for train_envs {train_envs}")

    # Test subset
    df_test = df[df["environment"] == test_env].copy()
    if df_test.empty:
        raise ValueError(f"No samples found for test_env '{test_env}'")

    train_ids, val_ids = train_test_split(
        df_trainval["label"].tolist(),
        test_size=val_ratio,
        random_state=random_state,
        stratify=df_trainval["environment"],
    )

    split_ids = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": df_test["label"].tolist(),
    }
    return split_ids
