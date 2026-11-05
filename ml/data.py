"""Load PaySim-format transactions, drop leaky columns, split by time, write engine files.

Works with the real PaySim CSV and with sim/generate.py's output (same columns).

Leaky columns, dropped on load (see docs/data.md):
  * newbalanceOrig / newbalanceDest: balances AFTER the payment, not known at payment time.
    In PaySim a fraud usually empties the account, so "new balance = 0" gives the label away.
  * oldbalanceOrg / oldbalanceDest: PaySim's balance bookkeeping is inconsistent for fraud and
    merchants (often both 0 for mules), which also hints at the label.
  * isFlaggedFraud: the output of PaySim's own rule, not an input a model should copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]  # type code = index
LEAKY_COLUMNS = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                 "isFlaggedFraud"]
REQUIRED_COLUMNS = ["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]
MERCHANT_BIT = 1 << 62  # account ids: customers = the digits, merchants = digits | MERCHANT_BIT

# One transaction in the engine's binary file: 40 bytes, little-endian.
TX_DTYPE = np.dtype([
    ("orig", "<u8"), ("dest", "<u8"), ("amount", "<f8"), ("tx_id", "<u8"),
    ("step", "<i4"), ("type", "u1"), ("is_fraud", "u1"), ("pad", "<u2"),
])
assert TX_DTYPE.itemsize == 40


def account_id(name: str) -> int:
    """'C1231006815' -> 1231006815; 'M1979787155' -> 1979787155 | MERCHANT_BIT."""
    if len(name) < 2 or name[0] not in "CM" or not name[1:].isdigit():
        raise ValueError(f"bad account name: {name!r}")
    number = int(name[1:])
    if number >= MERCHANT_BIT:
        raise ValueError(f"account number too large: {name!r}")
    return number | MERCHANT_BIT if name[0] == "M" else number


def is_merchant(acc: int) -> bool:
    return bool(acc & MERCHANT_BIT)


def load_transactions(path: Path) -> pd.DataFrame:
    """Read a PaySim CSV, keep only what is known at payment time, sort by time (stable)."""
    df = pd.read_csv(path, usecols=lambda c: c not in LEAKY_COLUMNS)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    unknown = set(df["type"].unique()) - set(TYPES)
    if unknown:
        raise ValueError(f"{path}: unknown transaction types {sorted(unknown)}")
    if (df["amount"] < 0).any():
        raise ValueError(f"{path}: negative amounts")
    # Stable sort keeps the file's order inside a step (PaySim is already in step order).
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    df["tx_id"] = np.arange(len(df), dtype=np.uint64)
    df["type_code"] = df["type"].map({t: i for i, t in enumerate(TYPES)}).astype(np.uint8)
    df["orig_id"] = np.array([account_id(n) for n in df["nameOrig"]], dtype=np.uint64)
    df["dest_id"] = np.array([account_id(n) for n in df["nameDest"]], dtype=np.uint64)
    df["isFraud"] = df["isFraud"].astype(np.uint8)
    return df


@dataclass(frozen=True)
class SplitSteps:
    train_end: int  # last step in train
    val_end: int  # last step in validation; test = everything after


def time_split_steps(steps: pd.Series, train: float = 0.70, val: float = 0.15) -> SplitSteps:
    """Cut points on whole steps, so no step (hour) is shared by two sets."""
    counts = steps.value_counts().sort_index()
    cum = counts.cumsum() / counts.sum()
    train_end = int(cum.index[np.searchsorted(cum.values, train)])
    val_end = int(cum.index[np.searchsorted(cum.values, train + val)])
    if not (train_end < val_end < int(counts.index[-1])):
        raise ValueError("not enough distinct steps to split train / validation / test")
    return SplitSteps(train_end, val_end)


def split_by_time(df: pd.DataFrame, cut: SplitSteps) -> dict[str, pd.DataFrame]:
    return {
        "train": df[df["step"] <= cut.train_end],
        "validation": df[(df["step"] > cut.train_end) & (df["step"] <= cut.val_end)],
        "test": df[df["step"] > cut.val_end],
    }


def to_tx_array(df: pd.DataFrame) -> np.ndarray:
    arr = np.zeros(len(df), dtype=TX_DTYPE)
    arr["orig"] = df["orig_id"].to_numpy()
    arr["dest"] = df["dest_id"].to_numpy()
    arr["amount"] = df["amount"].to_numpy(dtype=np.float64)
    arr["tx_id"] = df["tx_id"].to_numpy()
    arr["step"] = df["step"].to_numpy(dtype=np.int32)
    arr["type"] = df["type_code"].to_numpy()
    arr["is_fraud"] = df["isFraud"].to_numpy()
    return arr


def write_tx_file(df: pd.DataFrame, path: Path) -> None:
    """The engine's input: 40-byte records (engine/include/fd/transaction.hpp)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    to_tx_array(df).tofile(path)


def read_tx_file(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype=TX_DTYPE)
