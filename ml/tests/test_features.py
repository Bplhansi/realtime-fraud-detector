import math

import numpy as np
import pandas as pd
import pytest

from ml.data import MERCHANT_BIT, TYPES, load_transactions
from ml.features import (
    FEATURE_NAMES,
    FEATURES_FILE,
    HISTORY,
    FeatureStore,
    compute_features,
    load_feature_names,
)

T = {name: i for i, name in enumerate(TYPES)}


def feat(row, name):
    return row[FEATURE_NAMES.index(name)]


def test_feature_list_comes_from_the_shared_file():
    assert FEATURE_NAMES == load_feature_names(FEATURES_FILE)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 17


def test_new_accounts():
    row = FeatureStore().compute(10, T["TRANSFER"], 100.0, 1, 2)
    assert feat(row, "amount_log") == pytest.approx(math.log1p(100.0))
    assert feat(row, "sender_is_new") == 1.0
    assert feat(row, "receiver_is_new") == 1.0
    assert feat(row, "hours_since_sender_last") == -1.0
    assert feat(row, "type_TRANSFER") == 1.0
    assert sum(feat(row, f"type_{t}") for t in TYPES) == 1.0


def test_history_features():
    s = FeatureStore()
    s.process(1, T["PAYMENT"], 100.0, 1, 50 | MERCHANT_BIT)
    s.process(5, T["PAYMENT"], 300.0, 1, 50 | MERCHANT_BIT)
    row = s.process(5, T["TRANSFER"], 1000.0, 1, 2)
    assert feat(row, "amount_vs_sender_avg") == pytest.approx(5.0)
    assert feat(row, "sender_tx_1h") == 1.0
    assert feat(row, "sender_tx_24h") == 2.0
    assert feat(row, "hours_since_sender_first_seen") == 4.0
    assert feat(row, "pair_seen_before") == 0.0
    assert feat(s.compute(6, T["TRANSFER"], 1.0, 1, 2), "pair_seen_before") == 1.0


def test_mule_cash_out():
    s = FeatureStore()
    s.process(10, T["TRANSFER"], 400.0, 1, 900)
    s.process(11, T["TRANSFER"], 600.0, 2, 900)
    row = s.compute(12, T["CASH_OUT"], 1000.0, 900, 7 | MERCHANT_BIT)
    assert feat(row, "sender_inflow_24h_ratio") == pytest.approx(1.0)
    assert feat(row, "receiver_is_merchant") == 1.0
    assert feat(s.compute(12, T["TRANSFER"], 5.0, 3, 900), "receiver_distinct_senders_24h") == 2.0


def test_history_is_bounded():
    s = FeatureStore()
    for i in range(HISTORY + 10):
        s.process(1, T["PAYMENT"], 1.0, 1, 1000 + i)
    assert len(s.accounts[1].out) == HISTORY
    assert feat(s.compute(1, T["PAYMENT"], 1.0, 1, 1000), "pair_seen_before") == 0.0


def test_features_never_use_future_transactions(small_csv):
    """Changing or deleting everything after row k must not change rows 0..k."""
    df = load_transactions(small_csv)
    full = compute_features(df)
    k = len(df) // 2
    future_changed = df.copy()
    future_changed.loc[k + 1:, "amount"] *= 50
    future_changed.loc[k + 1:, "type_code"] = T["TRANSFER"]
    np.testing.assert_array_equal(compute_features(future_changed)[: k + 1], full[: k + 1])
    np.testing.assert_array_equal(compute_features(df.iloc[: k + 1])[: k + 1], full[: k + 1])


def test_a_transaction_does_not_see_itself():
    df = pd.DataFrame({"step": [1], "type_code": [T["TRANSFER"]], "amount": [500.0],
                       "orig_id": [1], "dest_id": [2]})
    row = compute_features(df)[0]
    assert feat(row, "sender_is_new") == 1.0
    assert feat(row, "sender_tx_1h") == 0.0
