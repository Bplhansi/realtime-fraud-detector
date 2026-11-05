import json

import numpy as np
import pandas as pd
import pytest

from ml.data import (
    LEAKY_COLUMNS,
    MERCHANT_BIT,
    TX_DTYPE,
    account_id,
    load_transactions,
    read_tx_file,
    split_by_time,
    time_split_steps,
    write_tx_file,
)


def test_account_ids():
    assert account_id("C1231006815") == 1231006815
    assert account_id("M1979787155") == 1979787155 | MERCHANT_BIT
    for bad in ["", "C", "X123", "C12a", "M-1"]:
        with pytest.raises(ValueError):
            account_id(bad)


def test_leaky_columns_are_dropped(small_csv):
    raw = pd.read_csv(small_csv)
    assert set(LEAKY_COLUMNS) <= set(raw.columns)
    df = load_transactions(small_csv)
    assert not set(LEAKY_COLUMNS) & set(df.columns)


def test_load_sorts_by_step_and_numbers_rows(tmp_path):
    path = tmp_path / "t.csv"
    pd.DataFrame({"step": [2, 1, 2], "type": ["PAYMENT", "TRANSFER", "CASH_IN"],
                  "amount": [1.0, 2.0, 3.0], "nameOrig": ["C1", "C2", "C3"],
                  "nameDest": ["M9", "C4", "M9"], "isFraud": [0, 1, 0]}).to_csv(path, index=False)
    df = load_transactions(path)
    assert df["step"].tolist() == [1, 2, 2]
    assert df["amount"].tolist() == [2.0, 1.0, 3.0], "stable: file order kept within a step"
    assert df["tx_id"].tolist() == [0, 1, 2]


@pytest.mark.parametrize("change, message", [
    ({"type": ["PAYMENT", "BITCOIN"]}, "unknown transaction types"),
    ({"amount": [1.0, -5.0]}, "negative"),
])
def test_load_rejects_bad_files(tmp_path, change, message):
    data = {"step": [1, 2], "type": ["PAYMENT", "PAYMENT"], "amount": [1.0, 2.0],
            "nameOrig": ["C1", "C2"], "nameDest": ["M1", "M2"], "isFraud": [0, 0], **change}
    path = tmp_path / "bad.csv"
    pd.DataFrame(data).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_transactions(path)


def test_load_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"step": [1], "amount": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_transactions(path)


def test_time_split_has_no_overlap_in_time(small_csv):
    df = load_transactions(small_csv)
    cut = time_split_steps(df["step"])
    parts = split_by_time(df, cut)
    assert parts["train"]["step"].max() < parts["validation"]["step"].min()
    assert parts["validation"]["step"].max() < parts["test"]["step"].min()
    assert sum(len(p) for p in parts.values()) == len(df)
    share = len(parts["train"]) / len(df)
    assert 0.65 < share < 0.75


def test_split_needs_enough_steps():
    with pytest.raises(ValueError):
        time_split_steps(pd.Series([1, 1, 1, 2]))


def test_prepared_files_match_the_split(prepared):
    summary = json.loads((prepared / "split.json").read_text())
    for name in ("train", "validation", "test"):
        part = pd.read_csv(prepared / f"features_{name}.csv")
        assert len(part) == summary["sets"][name]["rows"]
    tr = pd.read_csv(prepared / "features_train.csv")
    te = pd.read_csv(prepared / "features_test.csv")
    assert tr["step"].max() <= summary["train_end_step"] < te["step"].min()


def test_tx_file_round_trip(tmp_path, small_csv):
    df = load_transactions(small_csv).head(100)
    write_tx_file(df, tmp_path / "x.tx.bin")
    back = read_tx_file(tmp_path / "x.tx.bin")
    assert TX_DTYPE.itemsize == 40
    assert (tmp_path / "x.tx.bin").stat().st_size == 40 * len(df)
    np.testing.assert_array_equal(back["amount"], df["amount"].to_numpy())
    np.testing.assert_array_equal(back["orig"], df["orig_id"].to_numpy())
    np.testing.assert_array_equal(back["is_fraud"], df["isFraud"].to_numpy())
