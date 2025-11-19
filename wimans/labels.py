# wimans/labels.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any, List
import json

import numpy as np
import pandas as pd

# Define activity vocabulary: 0 MUST be "nothing" for count rule
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

activity_to_idx = {act: i for i, act in enumerate(ACTIVITIES)}
idx_to_activity = {i: act for act, i in activity_to_idx.items()}
ACT_NOTHING = activity_to_idx["nothing"]


def encode_list(lst: List[int]) -> str:
    """Encode list as JSON string for CSV."""
    return json.dumps(lst)


def decode_list(s: str) -> List[int]:
    """Decode JSON string list from CSV."""
    return json.loads(s)


def build_sample_labels(raw_row: pd.Series, max_users: int) -> Dict[str, Any]:
    """
    Build per-user labels for one sample from a row of annotation.csv.

    Expected columns (adapt if needed):
      - user1_activity, user2_activity, ..., userN_activity
      - user1_id, user2_id, ..., userN_id (optional)

    If userX_activity is missing or NaN:
      - that slot is treated as empty:
          y_act = ACT_NOTHING
          y_id = -1
          slot_mask = 0
    """
    y_act: List[int] = []
    y_id: List[int] = []
    slot_mask: List[int] = []

    for slot in range(1, max_users + 1):
        act_col = f"user{slot}_activity"
        id_col = f"user{slot}_id"

        # Activity
        if (act_col in raw_row.index) and (not pd.isna(raw_row[act_col])):
            act_name = str(raw_row[act_col]).strip()
            act_idx = activity_to_idx.get(act_name, ACT_NOTHING)
            mask_val = 1
        else:
            act_idx = ACT_NOTHING
            mask_val = 0

        # Identity (optional)
        if (id_col in raw_row.index) and (not pd.isna(raw_row[id_col])):
            try:
                id_val = int(raw_row[id_col])
            except Exception:
                id_val = -1
        else:
            id_val = -1

        y_act.append(act_idx)
        y_id.append(id_val)
        slot_mask.append(mask_val)

    gt_count = int(sum(1 for a in y_act if a != ACT_NOTHING))

    return {
        "y_act": encode_list(y_act),
        "y_id": encode_list(y_id),
        "slot_mask": encode_list(slot_mask),
        "gt_count": gt_count,
    }
