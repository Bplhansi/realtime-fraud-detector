"""Turn a PaySim CSV into features and time-based train / validation / test sets.

Features are computed over the whole stream in time order (each from the past only), then the
rows are split by time: earliest 70% train, next 15% validation, final 15% test.

Writes to --out (default data/):
    features_train.csv, features_validation.csv, features_test.csv   (for ml.train / ml.evaluate)
    transactions.tx.bin                                              (all rows, for the C++ engine)
    split.json                                                       (cut steps and row counts)

Usage:
    python -m ml.prepare --csv data/transactions.csv --out data
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

from ml.data import load_transactions, split_by_time, time_split_steps, write_tx_file
from ml.features import compute_features, feature_frame

SETS = ("train", "validation", "test")


def prepare(csv_path: Path, out: Path) -> dict:
    t0 = time.perf_counter()
    df = load_transactions(csv_path)
    X = compute_features(df)
    feats = feature_frame(df, X)
    cut = time_split_steps(df["step"])
    parts = split_by_time(feats, cut)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"source": str(csv_path), "train_end_step": cut.train_end,
                     "validation_end_step": cut.val_end, "sets": {}}
    for name in SETS:
        part = parts[name]
        part.to_csv(out / f"features_{name}.csv", index=False)
        summary["sets"][name] = {
            "rows": len(part), "fraud": int(part["isFraud"].sum()),
            "first_step": int(part["step"].min()), "last_step": int(part["step"].max()),
        }
    write_tx_file(df, out / "transactions.tx.bin")
    (out / "split.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build features and a time-based split.")
    p.add_argument("--csv", type=Path, default=Path("data/transactions.csv"))
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args(argv)
    s = prepare(args.csv, args.out)
    for name in SETS:
        info = s["sets"][name]
        print(f"{name:<10} steps {info['first_step']:>4}-{info['last_step']:<4} "
              f"{info['rows']:>9,} rows {info['fraud']:>6,} fraud")
    print(f"done in {s['seconds']} s; files in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
