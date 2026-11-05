"""Explore a PaySim-format file: how much fraud, where, when, how big, and which columns leak.

Prints a short report and saves a figure (fraud over time and by amount).

Usage:
    python -m ml.explore --csv data/transactions.csv --figure reports/explore.png
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def explore(df: pd.DataFrame) -> dict:
    fraud = df[df["isFraud"] == 1]
    by_type = df.groupby("type")["isFraud"].agg(["count", "sum"])
    by_type["fraud_share"] = by_type["sum"] / by_type["count"]
    out = {
        "rows": len(df),
        "fraud": len(fraud),
        "fraud_share": len(fraud) / len(df),
        "steps": (int(df["step"].min()), int(df["step"].max())),
        "by_type": by_type.round(5).to_dict("index"),
        "median_amount": {"genuine": float(df.loc[df.isFraud == 0, "amount"].median()),
                          "fraud": float(fraud["amount"].median())},
    }
    if "newbalanceOrig" in df:
        # The leak: how often does the payment leave the sender with exactly 0?
        drained = df["newbalanceOrig"] == 0
        out["sender_left_with_zero"] = {
            "genuine": float(drained[df.isFraud == 0].mean()),
            "fraud": float(drained[df.isFraud == 1].mean())}
    if "isFlaggedFraud" in df:
        out["isFlaggedFraud_rows"] = int(df["isFlaggedFraud"].sum())
    return out


def save_figure(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    per_step = df.groupby("step")["isFraud"].sum()
    a.plot(per_step.index, per_step.to_numpy(), lw=0.8)
    a.set(title="Fraud transactions per step (hour)", xlabel="step", ylabel="fraud transactions")
    bins = np.logspace(0, np.log10(df["amount"].max() + 1), 40)
    b.hist(df.loc[df.isFraud == 0, "amount"] + 1, bins=bins, alpha=0.6, density=True,
           label="genuine")
    b.hist(df.loc[df.isFraud == 1, "amount"] + 1, bins=bins, alpha=0.6, density=True,
           label="fraud")
    b.set(xscale="log", title="Amount (share of each group)", xlabel="amount")
    b.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Explore a PaySim-format dataset.")
    p.add_argument("--csv", type=Path, default=Path("data/transactions.csv"))
    p.add_argument("--figure", type=Path, default=Path("reports/explore.png"))
    args = p.parse_args(argv)
    df = pd.read_csv(args.csv)
    r = explore(df)
    print(f"{r['rows']:,} transactions, steps {r['steps'][0]}-{r['steps'][1]}, "
          f"{r['fraud']:,} fraud ({r['fraud_share']:.3%})")
    print("fraud by type:")
    for t, v in r["by_type"].items():
        print(f"  {t:<9} {int(v['count']):>9,} rows {int(v['sum']):>6,} fraud "
              f"({v['fraud_share']:.3%})")
    m = r["median_amount"]
    print(f"median amount: genuine {m['genuine']:,.2f}, fraud {m['fraud']:,.2f}")
    if "sender_left_with_zero" in r:
        z = r["sender_left_with_zero"]
        print(f"LEAK: sender left with a balance of exactly 0 after the payment: "
              f"{z['genuine']:.1%} of genuine vs {z['fraud']:.1%} of fraud")
    save_figure(df, args.figure)
    print(f"figure saved to {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
