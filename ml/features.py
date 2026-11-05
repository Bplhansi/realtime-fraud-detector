"""Per-transaction features, computed from each account's history BEFORE the transaction.

This is the reference implementation. The C++ engine (engine/src/features.cpp) computes the same
numbers, and a parity test checks they match on thousands of transactions. The feature list and
its order live in one place: config/features.txt.

Each transaction is scored first and only then added to the history, so a feature can never see
the transaction itself or anything after it.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import TYPES, is_merchant

FEATURES_FILE = Path(__file__).resolve().parent.parent / "config" / "features.txt"
HISTORY = 64  # transactions remembered per account and direction (same as the C++ ring buffers)
WINDOW_1H = 1  # steps (1 step = 1 hour)
WINDOW_24H = 24


def load_feature_names(path: Path = FEATURES_FILE) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


FEATURE_NAMES = load_feature_names()


@dataclass
class AccountState:
    out: deque = field(default_factory=lambda: deque(maxlen=HISTORY))  # (dest, step, amount)
    inc: deque = field(default_factory=lambda: deque(maxlen=HISTORY))  # (orig, step, amount)
    out_total: float = 0.0
    out_count: int = 0
    first_seen: int = -1
    last_out_step: int = -1


class FeatureStore:
    """Remembers recent activity per account. compute() then update() for each transaction."""

    def __init__(self) -> None:
        self.accounts: dict[int, AccountState] = {}

    def __len__(self) -> int:
        return len(self.accounts)

    def compute(self, step: int, type_code: int, amount: float, orig: int,
                dest: int) -> list[float]:
        s = self.accounts.get(orig)
        r = self.accounts.get(dest)
        f: dict[str, float] = {}
        f["amount_log"] = math.log1p(amount)
        if s is not None and s.out_count > 0:
            f["amount_vs_sender_avg"] = amount / (s.out_total / s.out_count)
        else:
            f["amount_vs_sender_avg"] = 1.0
        n1 = n24 = 0
        seen_pair = 0.0
        inflow = 0.0
        if s is not None:
            for d, st, _ in s.out:
                if st > step - WINDOW_1H:
                    n1 += 1
                if st > step - WINDOW_24H:
                    n24 += 1
                if d == dest:
                    seen_pair = 1.0
            for _, st, amt in s.inc:
                if st > step - WINDOW_24H:
                    inflow += amt
        f["sender_tx_1h"] = float(n1)
        f["sender_tx_24h"] = float(n24)
        f["hours_since_sender_last"] = (
            float(step - s.last_out_step) if s is not None and s.last_out_step >= 0 else -1.0)
        f["sender_is_new"] = 1.0 if s is None or s.out_count == 0 else 0.0
        f["pair_seen_before"] = seen_pair
        senders: set[int] = set()
        if r is not None:
            for o, st, _ in r.inc:
                if st > step - WINDOW_24H:
                    senders.add(o)
        f["receiver_distinct_senders_24h"] = float(len(senders))
        f["receiver_is_new"] = 1.0 if r is None else 0.0
        f["sender_inflow_24h_ratio"] = amount / inflow if inflow > 0.0 else 0.0
        f["hours_since_sender_first_seen"] = (
            float(step - s.first_seen) if s is not None and s.first_seen >= 0 else -1.0)
        f["receiver_is_merchant"] = 1.0 if is_merchant(dest) else 0.0
        for i, t in enumerate(TYPES):
            f[f"type_{t}"] = 1.0 if type_code == i else 0.0
        return [f[name] for name in FEATURE_NAMES]

    def update(self, step: int, amount: float, orig: int, dest: int) -> None:
        s = self.accounts.setdefault(orig, AccountState())
        r = self.accounts.setdefault(dest, AccountState())
        s.out.append((dest, step, amount))
        s.out_total += amount
        s.out_count += 1
        s.last_out_step = step
        if s.first_seen < 0:
            s.first_seen = step
        r.inc.append((orig, step, amount))
        if r.first_seen < 0:
            r.first_seen = step

    def process(self, step: int, type_code: int, amount: float, orig: int,
                dest: int) -> list[float]:
        row = self.compute(step, type_code, amount, orig, dest)
        self.update(step, amount, orig, dest)
        return row


def compute_features(df: pd.DataFrame, store: FeatureStore | None = None) -> np.ndarray:
    """Features for every row of a time-sorted frame (from ml.data.load_transactions)."""
    store = store or FeatureStore()
    rows = _iter_rows(df)
    out = np.empty((len(df), len(FEATURE_NAMES)), dtype=np.float64)
    for i, (step, t, amount, orig, dest) in enumerate(rows):
        out[i] = store.process(step, t, amount, orig, dest)
    return out


def _iter_rows(df: pd.DataFrame) -> Iterable[tuple[int, int, float, int, int]]:
    return zip(df["step"].astype(int).tolist(), df["type_code"].astype(int).tolist(),
               df["amount"].astype(float).tolist(), df["orig_id"].astype("uint64").tolist(),
               df["dest_id"].astype("uint64").tolist(), strict=True)


def signed_log(x: np.ndarray) -> np.ndarray:
    """log1p that keeps the sign; squashes big ratios for the logistic regression. Lives here
    (not in ml/train.py) so a saved model can be loaded from any script."""
    return np.sign(x) * np.log1p(np.abs(x))


def feature_frame(df: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    """Features plus the ids and label, ready to save as CSV."""
    out = pd.DataFrame(X, columns=FEATURE_NAMES)
    out.insert(0, "tx_id", df["tx_id"].to_numpy())
    out.insert(1, "step", df["step"].to_numpy())
    out["amount"] = df["amount"].to_numpy()
    out["isFraud"] = df["isFraud"].to_numpy()
    return out
