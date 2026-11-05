"""Train the baseline and the main model, choose thresholds, save everything together.

1. Rules baseline (ml/rules.py).
2. Logistic regression with class weights (rare fraud counts more; no copying of fraud rows).
3. Gradient boosted trees (scikit-learn's HistGradientBoostingClassifier, same family as
   LightGBM) with class weights. A few settings are tried; the best on VALIDATION PR-AUC wins.
4. Thresholds, chosen on validation only:
     review  the score that minimises  cost = missed fraud amount + FALSE_ALARM_COST per false alarm
     block   the lowest score whose validation precision is at least BLOCK_PRECISION
             (blocking a genuine customer is worse than reviewing one)

The test set is not touched here; ml/evaluate.py scores it once at the end.

Writes to --models (default models/):
    fraud_model.joblib, logistic.joblib, model_meta.json, thresholds.ini (read by the C++ engine)

Usage:
    python -m ml.train --data data --models models
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ml.features import FEATURE_NAMES, signed_log
from ml.metrics import pr_auc, summary
from ml.rules import rule_score

MODEL_VERSION = "v1"
FALSE_ALARM_COST = 25.0  # cost of one reviewer check of a genuine payment, in the data's currency
BLOCK_PRECISION = 0.90
REVIEW_LIMIT = 100  # reviewers can check this many cases per validation/test period
SEED = 0
GRID = [
    {"learning_rate": 0.1, "max_leaf_nodes": 15, "max_iter": 150},
    {"learning_rate": 0.1, "max_leaf_nodes": 31, "max_iter": 150},
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "max_iter": 300},
]


def load_set(data: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(data / f"features_{name}.csv")


def xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df[FEATURE_NAMES].to_numpy(dtype=np.float32), df["isFraud"].to_numpy(dtype=int)


def make_logistic() -> object:
    return make_pipeline(FunctionTransformer(signed_log), StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=2000))


def make_gbm(params: dict) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(class_weight="balanced", early_stopping=False,
                                          random_state=SEED, **params)


def choose_review_threshold(y: np.ndarray, score: np.ndarray, amount: np.ndarray) -> float:
    """Threshold with the lowest total cost on validation."""
    best_t, best_cost = 0.5, float("inf")
    for t in np.unique(np.round(score, 4)):
        flagged = score >= t
        cost = float(amount[(y == 1) & ~flagged].sum()) + FALSE_ALARM_COST * int(
            ((y == 0) & flagged).sum())
        if cost < best_cost:
            best_t, best_cost = float(t), cost
    return best_t


def choose_block_threshold(y: np.ndarray, score: np.ndarray, review: float) -> float:
    """Lowest threshold (>= review) whose precision reaches BLOCK_PRECISION; 1.0 = never block."""
    for t in np.unique(np.round(score[score >= review], 4)):
        flagged = score >= t
        if flagged.sum() >= 5 and y[flagged].mean() >= BLOCK_PRECISION:
            return float(t)
    return 1.0


def train(data: Path, models: Path) -> dict:
    tr, va = load_set(data, "train"), load_set(data, "validation")
    Xtr, ytr = xy(tr)
    Xva, yva = xy(va)
    amount = va["amount"].to_numpy()
    results: dict = {"validation": {}}

    rules = rule_score(va)
    results["validation"]["rules"] = summary(yva, rules, rules > 0, amount, REVIEW_LIMIT)

    logistic = make_logistic().fit(Xtr, ytr)
    p_lr = logistic.predict_proba(Xva)[:, 1]
    t_lr = choose_review_threshold(yva, p_lr, amount)
    results["validation"]["logistic"] = summary(yva, p_lr, p_lr >= t_lr, amount, REVIEW_LIMIT)
    results["validation"]["logistic"]["threshold"] = t_lr

    tried = []
    best = None
    for params in GRID:
        model = make_gbm(params).fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]
        auc = pr_auc(yva, p)
        tried.append({**params, "validation_pr_auc": round(auc, 4)})
        if best is None or auc > best[0]:
            best = (auc, params, model, p)
    _, params, gbm, p_gbm = best
    review = choose_review_threshold(yva, p_gbm, amount)
    block = choose_block_threshold(yva, p_gbm, review)
    results["validation"]["gradient_boosting"] = summary(yva, p_gbm, p_gbm >= review, amount,
                                                          REVIEW_LIMIT)
    results["grid"] = tried

    meta = {
        "model_version": MODEL_VERSION,
        "features": FEATURE_NAMES,
        "params": params,
        "thresholds": {"review": review, "block": block},
        "false_alarm_cost": FALSE_ALARM_COST,
        "block_precision": BLOCK_PRECISION,
        "review_limit": REVIEW_LIMIT,
        "train_rows": len(tr),
        "validation": results["validation"],
        "grid": tried,
    }
    models.mkdir(parents=True, exist_ok=True)
    joblib.dump(gbm, models / "fraud_model.joblib")
    joblib.dump(logistic, models / "logistic.joblib")
    (models / "model_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    write_thresholds_ini(models / "thresholds.ini", review, block, MODEL_VERSION)
    return meta


def write_thresholds_ini(path: Path, review: float, block: float, version: str) -> None:
    path.write_text(
        "# Written by ml/train.py; read by the C++ engine. Scores are P(fraud) from 0 to 1.\n"
        f"model_version={version}\nreview={review!r}\nblock={block!r}\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the fraud models.")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--models", type=Path, default=Path("models"))
    args = p.parse_args(argv)
    meta = train(args.data, args.models)
    print(f"{'validation':<18} {'precision':>9} {'recall':>7} {'PR-AUC':>7} {'value':>7}")
    for name, m in meta["validation"].items():
        print(f"{name:<18} {m['precision']:>9.3f} {m['recall']:>7.3f} {m['pr_auc']:>7.3f} "
              f"{m['fraud_value_caught']:>7.3f}")
    t = meta["thresholds"]
    print(f"chosen settings: {meta['params']}; thresholds review={t['review']:.4f} "
          f"block={t['block']:.4f}; saved to {args.models}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
