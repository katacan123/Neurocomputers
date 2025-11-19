# wimans/labels.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# wimans/labels.py

import json
from typing import Dict, Any, List
import pandas as pd

# -------------------------
# Activity vocabulary
# -------------------------
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

# -------------------------
# Helper JSON encode/decode
# -------------------------
def encode_list(lst: List[int]) -> str:
    return json.dumps(lst)

def decode_list(s: str) -> List[int]:
    return json.loads(s)

# -------------------------
# Main label builder
# -------------------------
def build_sample_labels(row: pd.Series, max_users: int) -> Dict[str, Any]:
    """
    Builds y_act, y_loc, y_id (dummy = -1), slot_mask, gt_count
    from columns like:
      user_1_activity
      user_1_location
      ...
    """

    y_act = []
    y_loc = []
    y_id  = []     # ALWAYS unknown → -1
    slot_mask = []

    # ----- Iterate over user slots -----
    for k in range(1, max_users + 1):
        act_col = f"user_{k}_activity"
        loc_col = f"user_{k}_location"

        # ---- Activity ----
        if act_col in row and not pd.isna(row[act_col]):
            act_name = str(row[act_col]).strip()
            act_idx = activity_to_idx.get(act_name, ACT_NOTHING)
            user_present = True
        else:
            act_idx = ACT_NOTHING
            user_present = False

        # ---- Location ----
        if loc_col in row and not pd.isna(row[loc_col]) and user_present:
            try:
                loc_idx = int(row[loc_col])
            except Exception:
                loc_idx = -1
        else:
            loc_idx = -1   # ignored during loss if -1

        # ---- Identity (you don't have IDs) ----
        id_idx = -1  # always unknown

        # ---- Slot mask ----
        # User is considered "present" if activity column exists and not nothing.
        if user_present and act_idx != ACT_NOTHING:
            mask_val = 1
        else:
            mask_val = 0

        y_act.append(act_idx)
        y_loc.append(loc_idx)
        y_id.append(id_idx)
        slot_mask.append(mask_val)

    # ----- Count ground truth -----
    gt_count = sum(slot_mask)

    return {
        "y_act": encode_list(y_act),
        "y_loc": encode_list(y_loc),
        "y_id": encode_list(y_id),
        "slot_mask": encode_list(slot_mask),
        "gt_count": gt_count,
    }
