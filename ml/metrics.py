"""Fraud metrics. Accuracy is useless here (99.8% of rows are genuine), so we report:

precision          of flagged transactions, how many were fraud
recall             of real fraud, how much was flagged
PR-AUC             quality across all thresholds (average precision); better than ROC-AUC
                   when fraud is rare
recall@k           fraud caught if reviewers can only check the k highest-scoring transactions
fraud value caught share of fraud MONEY that was flagged, not just the count
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def precision_recall(y: np.ndarray, flagged: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=bool)
    flagged = np.asarray(flagged, dtype=bool)
    tp = int(np.sum(y & flagged))
    precision = tp / int(flagged.sum()) if flagged.any() else 0.0
    recall = tp / int(y.sum()) if y.any() else 0.0
    return precision, recall


def pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    return float(average_precision_score(y, score)) if y.any() else 0.0


def recall_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    """Share of fraud among the k highest scores (ties broken by row order)."""
    y = np.asarray(y, dtype=bool)
    if not y.any() or k <= 0:
        return 0.0
    top = np.argsort(-np.asarray(score, dtype=float), kind="stable")[:k]
    return float(y[top].sum() / y.sum())


def fraud_value_caught(y: np.ndarray, flagged: np.ndarray, amount: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    fraud_value = float(np.sum(np.asarray(amount)[y]))
    if fraud_value == 0.0:
        return 0.0
    return float(np.sum(np.asarray(amount)[y & np.asarray(flagged, dtype=bool)]) / fraud_value)


def summary(y: np.ndarray, score: np.ndarray, flagged: np.ndarray, amount: np.ndarray,
            k: int) -> dict[str, float]:
    p, r = precision_recall(y, flagged)
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "pr_auc": round(pr_auc(y, score), 4),
        f"recall_at_{k}": round(recall_at_k(y, score, k), 4),
        "fraud_value_caught": round(fraud_value_caught(y, flagged, amount), 4),
        "flagged": int(np.sum(flagged)),
    }
