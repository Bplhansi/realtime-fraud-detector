"""Explain why a transaction was flagged, in plain words.

SHAP (TreeExplainer) splits the model's output for one transaction into a contribution per
feature. The features that pushed the score UP the most become the reasons, written as phrases a
reviewer understands ("amount is 12.4x the sender's average").

This runs in Python and only for flagged transactions (review or block), so it never slows
down normal scoring.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ml.features import FEATURE_NAMES


def _type_phrase(name: str) -> Callable[[float], str]:
    return lambda v: (f"transaction type is {name}" if v >= 0.5
                      else f"transaction type is not {name}")


def _hours(v: float, never: str, template: str) -> str:
    return never if v < 0 else template.format(v=v)


PHRASES: dict[str, Callable[[float], str]] = {
    "amount_log": lambda v: f"large amount ({math.expm1(v):,.0f})",
    "amount_vs_sender_avg": lambda v: f"amount is {v:.1f}x the sender's average",
    "sender_tx_1h": lambda v: f"{v:.0f} earlier payments by the sender in the same hour",
    "sender_tx_24h": lambda v: f"{v:.0f} payments by the sender in the last 24 hours",
    "hours_since_sender_last": lambda v: _hours(
        v, "sender has never paid anyone before", "{v:.0f} hours since the sender's last payment"),
    "sender_is_new": lambda v: ("sender has never paid anyone before" if v >= 0.5
                                else "sender has a payment history"),
    "pair_seen_before": lambda v: ("sender has paid this receiver before" if v >= 0.5
                                   else "new receiver for this sender"),
    "receiver_distinct_senders_24h": lambda v: (
        f"receiver was paid by {v:.0f} different senders in the last 24 hours"),
    "receiver_is_new": lambda v: ("receiver account never seen before" if v >= 0.5
                                  else "receiver account seen before"),
    "sender_inflow_24h_ratio": lambda v: (
        f"amount is {v:.0%} of what the sender received in the last 24 hours" if v > 0
        else "sender received no money in the last 24 hours"),
    "hours_since_sender_first_seen": lambda v: _hours(
        v, "sender account never seen before", "sender account first seen {v:.0f} hours ago"),
    "receiver_is_merchant": lambda v: ("receiver is a merchant" if v >= 0.5
                                       else "receiver is a person, not a merchant"),
    **{f"type_{t}": _type_phrase(t) for t in
       ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]},
}
assert set(PHRASES) == set(FEATURE_NAMES), "every feature needs a phrase"


@dataclass(frozen=True)
class Reason:
    feature: str
    value: float
    contribution: float  # in log-odds; > 0 pushed the score towards fraud
    text: str


class Explainer:
    def __init__(self, model: object) -> None:
        import shap  # imported here: it is slow to import and only needed for flagged cases

        self._explainer = shap.TreeExplainer(model)

    def contributions(self, row: Sequence[float]) -> np.ndarray:
        x = np.asarray(row, dtype=np.float32).reshape(1, -1)
        values = np.asarray(self._explainer.shap_values(x))
        return values.reshape(-1)[-len(FEATURE_NAMES):]

    def reasons(self, row: Sequence[float], top: int = 3) -> list[Reason]:
        contrib = self.contributions(row)
        order = np.argsort(-contrib, kind="stable")
        out: list[Reason] = []
        for j in order:
            if contrib[j] <= 0 or len(out) == top:
                break
            name = FEATURE_NAMES[j]
            if name.startswith("type_") and row[j] < 0.5:
                continue  # "type is not PAYMENT" doesn't help a reviewer
            out.append(Reason(name, float(row[j]), float(contrib[j]), PHRASES[name](float(row[j]))))
        return out
