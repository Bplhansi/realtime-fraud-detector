"""Check for model drift: has recent data moved away from what the model was trained on?

Fraud changes over time, so a model can get worse without any error appearing. Two checks:

1. Feature drift (PSI, population stability index) for a few key features, recent data vs
   training data. Rule of thumb: PSI < 0.1 stable, 0.1-0.25 worth watching, > 0.25 a real shift.
2. Reviewer-confirmed precision: of the flagged cases reviewers have decided, how many were
   really fraud? A big drop below the validation precision is a warning too.

Usage:
    python -m ml.drift --reference data/features_train.csv --current data/features_test.csv
    python -m ml.drift --reference data/features_train.csv --current-csv data/shifted.csv
    (add --db data/decisions.db to include reviewer precision; writes data/drift_report.json)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import load_transactions
from ml.features import compute_features, feature_frame

MONITORED = ["amount_log", "amount_vs_sender_avg", "sender_tx_24h",
             "receiver_distinct_senders_24h", "type_CASH_OUT", "type_PAYMENT", "type_TRANSFER"]
PSI_WATCH = 0.10
PSI_ALERT = 0.25
MIN_REVIEWS = 20  # too few reviewer decisions say nothing
PRECISION_DROP = 0.20  # warn if confirmed precision falls this far below validation precision
BINS = 10


def psi(reference: np.ndarray, current: np.ndarray, bins: int = BINS) -> float:
    """Population stability index with bins from the reference's quantiles."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("psi needs data on both sides")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) <= 2:  # (nearly) constant or binary feature: compare the values directly
        values = np.unique(np.concatenate([reference, current]))
        ref = np.array([np.mean(reference == v) for v in values])
        cur = np.array([np.mean(current == v) for v in values])
    else:
        inner = edges[1:-1]
        ref = np.bincount(np.searchsorted(inner, reference, side="right"),
                          minlength=len(inner) + 1) / len(reference)
        cur = np.bincount(np.searchsorted(inner, current, side="right"),
                          minlength=len(inner) + 1) / len(current)
    eps = 1e-4  # avoids log(0) for empty bins
    ref = np.clip(ref, eps, None)
    cur = np.clip(cur, eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def feature_drift(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    return {f: round(psi(reference[f].to_numpy(), current[f].to_numpy()), 4) for f in MONITORED}


def reviewer_precision(db_path: Path) -> tuple[int, int]:
    """(confirmed fraud, reviewed) using each decision's latest review."""
    if not db_path.exists():
        return 0, 0
    with closing(sqlite3.connect(db_path)) as con:
        row = con.execute(
            """SELECT COALESCE(SUM(label = 'fraud'), 0), COUNT(*) FROM (
                 SELECT label FROM reviews r
                 WHERE id = (SELECT MAX(id) FROM reviews WHERE decision_id = r.decision_id))"""
        ).fetchone()
    return int(row[0]), int(row[1])


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 confirmed: int = 0, reviewed: int = 0,
                 expected_precision: float | None = None) -> dict:
    scores = feature_drift(reference, current)
    shifted = [f for f, v in scores.items() if v > PSI_ALERT]
    watch = [f for f, v in scores.items() if PSI_WATCH < v <= PSI_ALERT]
    warnings = []
    if shifted:
        warnings.append(f"feature drift: {', '.join(shifted)} (PSI > {PSI_ALERT})")
    precision = confirmed / reviewed if reviewed else None
    if (precision is not None and reviewed >= MIN_REVIEWS and expected_precision is not None
            and precision < expected_precision - PRECISION_DROP):
        warnings.append(f"reviewer-confirmed precision {precision:.2f} is well below the "
                        f"validation precision {expected_precision:.2f}")
    return {"psi": scores, "shifted": shifted, "watch": watch,
            "reviewed": reviewed, "confirmed_fraud": confirmed,
            "confirmed_precision": None if precision is None else round(precision, 4),
            "expected_precision": expected_precision,
            "warning": bool(warnings), "warnings": warnings}


def features_from_csv(csv_path: Path) -> pd.DataFrame:
    df = load_transactions(csv_path)
    return feature_frame(df, compute_features(df))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Check recent data for drift.")
    p.add_argument("--reference", type=Path, default=Path("data/features_train.csv"))
    src = p.add_mutually_exclusive_group()
    src.add_argument("--current", type=Path, help="a features CSV (from ml.prepare)")
    src.add_argument("--current-csv", type=Path, help="a raw PaySim-format CSV")
    p.add_argument("--db", type=Path, help="decisions database, for reviewer precision")
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--out", type=Path, default=Path("data/drift_report.json"))
    args = p.parse_args(argv)
    reference = pd.read_csv(args.reference)
    if args.current_csv:
        current = features_from_csv(args.current_csv)
    else:
        current = pd.read_csv(args.current or Path("data/features_test.csv"))
    confirmed, reviewed = reviewer_precision(args.db) if args.db else (0, 0)
    expected = None
    meta_path = args.models / "model_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = meta["validation"]["gradient_boosting"]["precision"]
    report = drift_report(reference, current, confirmed, reviewed, expected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for f, v in report["psi"].items():
        mark = "SHIFTED" if f in report["shifted"] else "watch" if f in report["watch"] else "ok"
        print(f"{f:<32} PSI {v:>7.4f}  {mark}")
    print("WARNING: " + "; ".join(report["warnings"]) if report["warning"] else "no drift warning")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
