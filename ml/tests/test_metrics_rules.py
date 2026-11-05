import numpy as np
import pandas as pd
import pytest

from ml.features import FEATURE_NAMES
from ml.metrics import fraud_value_caught, pr_auc, precision_recall, recall_at_k, summary
from ml.rules import rule_hits, rule_score


def test_precision_recall():
    y = np.array([1, 1, 0, 0, 0])
    flagged = np.array([1, 0, 1, 0, 0])
    assert precision_recall(y, flagged) == (0.5, 0.5)
    assert precision_recall(y, np.zeros(5)) == (0.0, 0.0)


def test_accuracy_would_lie_but_these_do_not():
    y = np.array([0] * 999 + [1])
    nothing = np.zeros(1000)
    assert precision_recall(y, nothing)[1] == 0.0  # 99.9% accurate, catches nothing


def test_pr_auc_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1])
    assert pr_auc(y, np.array([0.1, 0.2, 0.3, 0.8, 0.9])) == 1.0
    assert pr_auc(np.zeros(3), np.array([0.1, 0.2, 0.3])) == 0.0


def test_recall_at_k():
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.8, 0.1, 0.0])
    assert recall_at_k(y, score, 1) == 0.5
    assert recall_at_k(y, score, 4) == 1.0
    assert recall_at_k(y, score, 0) == 0.0


def test_fraud_value_caught_counts_money_not_cases():
    y = np.array([1, 1, 0])
    amount = np.array([900.0, 100.0, 50.0])
    assert fraud_value_caught(y, np.array([1, 0, 1]), amount) == pytest.approx(0.9)
    assert fraud_value_caught(np.zeros(3), np.ones(3), amount) == 0.0


def test_summary_keys():
    s = summary(np.array([1, 0]), np.array([0.9, 0.1]), np.array([1, 0]), np.array([5.0, 1.0]), 1)
    assert s == {"precision": 1.0, "recall": 1.0, "pr_auc": 1.0, "recall_at_1": 1.0,
                 "fraud_value_caught": 1.0, "flagged": 1}


def _frame(**overrides):
    row = {name: 0.0 for name in FEATURE_NAMES}
    row.update(overrides)
    return pd.DataFrame([row])


def test_rule_big_transfer_to_stranger():
    hit = _frame(type_TRANSFER=1, amount_vs_sender_avg=6.0, pair_seen_before=0)
    known = _frame(type_TRANSFER=1, amount_vs_sender_avg=6.0, pair_seen_before=1)
    small = _frame(type_TRANSFER=1, amount_vs_sender_avg=2.0, pair_seen_before=0)
    assert rule_hits(hit)["big_transfer_to_stranger"].item()
    assert not rule_hits(known)["big_transfer_to_stranger"].item()
    assert rule_score(small)[0] == 0


def test_rule_quick_cash_out():
    hit = _frame(type_CASH_OUT=1, sender_inflow_24h_ratio=1.0, hours_since_sender_first_seen=3)
    old = _frame(type_CASH_OUT=1, sender_inflow_24h_ratio=1.0, hours_since_sender_first_seen=300)
    never = _frame(type_CASH_OUT=1, sender_inflow_24h_ratio=1.0, hours_since_sender_first_seen=-1)
    assert rule_hits(hit)["quick_cash_out"].item()
    assert not rule_hits(old)["quick_cash_out"].item()
    assert not rule_hits(never)["quick_cash_out"].item()
