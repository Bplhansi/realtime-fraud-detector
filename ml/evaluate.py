"""Score the TEST set once, with everything frozen on validation, and write the results.

Compares the three methods on the final 15% of time (data/features_test.csv):
rules, logistic regression and the gradient boosted model, each at the threshold chosen on
validation. If the C++ engine's scores for the same period are given (fd_replay --scores),
it also checks that C++ and Python agree.

The test set is meant to be used once. If reports/test_results.json already exists the script
stops, unless --force is given (and then the README should say the test set was reused).

Usage:
    python -m ml.evaluate --data data --models models --engine-scores data/engine_scores.csv
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.features import FEATURE_NAMES
from ml.metrics import summary
from ml.rules import rule_score


def evaluate(data: Path, models: Path, engine_scores: Path | None = None) -> dict:
    test = pd.read_csv(data / "features_test.csv")
    meta = json.loads((models / "model_meta.json").read_text(encoding="utf-8"))
    X = test[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = test["isFraud"].to_numpy(dtype=int)
    amount = test["amount"].to_numpy()
    k = int(meta["review_limit"])

    out: dict = {"model_version": meta["model_version"], "test_rows": len(test),
                 "test_fraud": int(y.sum()), "review_limit": k, "methods": {}}
    rules = rule_score(test)
    out["methods"]["rules"] = summary(y, rules, rules > 0, amount, k)

    logistic = joblib.load(models / "logistic.joblib")
    p_lr = logistic.predict_proba(X)[:, 1]
    t_lr = meta["validation"]["logistic"]["threshold"]
    out["methods"]["logistic"] = summary(y, p_lr, p_lr >= t_lr, amount, k)

    gbm = joblib.load(models / "fraud_model.joblib")
    p = gbm.predict_proba(X)[:, 1]
    review, block = meta["thresholds"]["review"], meta["thresholds"]["block"]
    out["methods"]["gradient_boosting"] = summary(y, p, p >= review, amount, k)
    out["methods"]["gradient_boosting_block_only"] = summary(y, p, p >= block, amount, k)

    if engine_scores is not None:
        eng = pd.read_csv(engine_scores)
        merged = test[["tx_id"]].assign(python=p).merge(eng, on="tx_id", how="inner")
        if len(merged) != len(test):
            raise ValueError(f"engine scores cover {len(merged)} of {len(test)} test rows")
        expected = np.where(merged["python"] >= block, "block",
                            np.where(merged["python"] >= review, "review", "approve"))
        out["engine_parity"] = {
            "rows": len(merged),
            "max_abs_score_diff": float(np.max(np.abs(merged["score"] - merged["python"]))),
            "decisions_agree": int(np.sum(expected == merged["decision"].to_numpy())),
        }
    return out


def to_markdown(r: dict) -> str:
    k = r["review_limit"]
    lines = [f"| Method | Precision | Recall | PR-AUC | Recall@{k} | Fraud value caught "
             "| Flagged |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    names = {"rules": "Rules only", "logistic": "Logistic regression",
             "gradient_boosting": "Gradient boosted model (review or block)",
             "gradient_boosting_block_only": "Gradient boosted model (block only)"}
    for key, label in names.items():
        m = r["methods"][key]
        lines.append(f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | {m['pr_auc']:.3f} | "
                     f"{m[f'recall_at_{k}']:.3f} | {m['fraud_value_caught']:.3f} | "
                     f"{m['flagged']} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score the test set once.")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--engine-scores", type=Path, help="CSV from fd_replay --scores")
    p.add_argument("--out", type=Path, default=Path("reports/test_results.json"))
    p.add_argument("--force", action="store_true", help="overwrite an earlier test result")
    args = p.parse_args(argv)
    if args.out.exists() and not args.force:
        print(f"{args.out} already exists: the test set has been used. Use --force to redo it.")
        return 1
    r = evaluate(args.data, args.models, args.engine_scores)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
    print(f"test set: {r['test_rows']:,} transactions, {r['test_fraud']} fraud")
    print(to_markdown(r))
    if "engine_parity" in r:
        e = r["engine_parity"]
        print(f"C++ vs Python: max score difference {e['max_abs_score_diff']:.2e}; "
              f"{e['decisions_agree']:,} of {e['rows']:,} decisions agree")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
