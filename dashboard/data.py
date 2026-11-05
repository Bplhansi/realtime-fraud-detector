"""Data helpers for the dashboard, kept apart from Streamlit so they are easy to test."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from api.store import DecisionStore

QUEUE_COLUMNS = ["id", "score", "decision", "step", "type", "amount", "name_orig", "name_dest",
                 "review_label"]


def queue_frame(store: DecisionStore, pending_only: bool, limit: int = 500) -> pd.DataFrame:
    rows = store.flagged(pending_only=pending_only, limit=limit)
    return pd.DataFrame(rows, columns=[*QUEUE_COLUMNS, "reasons"])


def activity_frame(store: DecisionStore, name: str, before_step: int) -> pd.DataFrame:
    rows = store.account_activity(name, before_step=before_step, limit=15)
    cols = ["step", "type", "amount", "name_orig", "name_dest", "decision", "score"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["direction"] = ["sent" if o == name else "received" for o in df["name_orig"]]
    return df


def alerts_frame(store: DecisionStore) -> pd.DataFrame:
    df = pd.DataFrame(store.alerts_by_step(), columns=["step", "review", "block", "total"])
    return df.melt(id_vars="step", value_vars=["review", "block"], var_name="decision",
                   value_name="alerts")


def reviewer_precision(stats: dict) -> float | None:
    fraud, false_alarm = stats["reviews"]["fraud"], stats["reviews"]["false_alarm"]
    total = fraud + false_alarm
    return fraud / total if total else None


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
