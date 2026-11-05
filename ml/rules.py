"""The simple-rules baseline: what a first fraud system might do before any ML.

Rule 1  big transfer to a stranger: a TRANSFER at least 5x the sender's average, to a receiver the
        sender has not paid before (in its last 64 payments).
Rule 2  quick cash-out: a CASH_OUT of at least 80% of the money received in the last 24 hours by
        an account first seen less than 48 hours ago (typical mule behaviour).

The rules give a yes/no answer. Their "score" is the number of rules that fired (0, 1 or 2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BIG_TRANSFER_RATIO = 5.0
CASH_OUT_SHARE = 0.8
NEW_ACCOUNT_HOURS = 48


def rule_hits(features: pd.DataFrame) -> pd.DataFrame:
    f = features
    big_transfer = ((f["type_TRANSFER"] == 1) & (f["amount_vs_sender_avg"] >= BIG_TRANSFER_RATIO)
                    & (f["pair_seen_before"] == 0))
    first_seen = f["hours_since_sender_first_seen"]
    quick_cash_out = ((f["type_CASH_OUT"] == 1) & (f["sender_inflow_24h_ratio"] >= CASH_OUT_SHARE)
                      & (first_seen >= 0) & (first_seen < NEW_ACCOUNT_HOURS))
    return pd.DataFrame({"big_transfer_to_stranger": big_transfer,
                         "quick_cash_out": quick_cash_out})


def rule_score(features: pd.DataFrame) -> np.ndarray:
    return rule_hits(features).sum(axis=1).to_numpy(dtype=float)
