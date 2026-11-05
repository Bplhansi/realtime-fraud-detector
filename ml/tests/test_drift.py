import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from ml.drift import (
    MIN_REVIEWS,
    PSI_ALERT,
    drift_report,
    features_from_csv,
    main,
    psi,
    reviewer_precision,
)
from sim.generate import SimConfig, simulate, write_csv


def test_psi_same_and_different():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, rng.normal(size=5000)) < 0.02
    assert psi(a, rng.normal(loc=2.0, size=5000)) > PSI_ALERT
    assert psi(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == pytest.approx(0.0)
    assert psi(np.array([0, 0, 0, 1]), np.array([1, 1, 1, 0])) > PSI_ALERT
    with pytest.raises(ValueError):
        psi(np.array([]), a)


def test_no_warning_on_the_test_period(prepared):
    ref = pd.read_csv(prepared / "features_train.csv")
    cur = pd.read_csv(prepared / "features_test.csv")
    assert not drift_report(ref, cur)["warning"]


def test_shifted_file_triggers_the_warning(prepared, tmp_path):
    path = tmp_path / "shifted.csv"
    write_csv(simulate(SimConfig(customers=800, merchants=80, days=7, shift=3.0, seed=5)), path)
    report = drift_report(pd.read_csv(prepared / "features_train.csv"), features_from_csv(path))
    assert report["warning"]
    assert "amount_log" in report["shifted"]


def test_reviewer_precision_warning(prepared):
    ref = pd.read_csv(prepared / "features_train.csv")
    few = drift_report(ref, ref, confirmed=1, reviewed=MIN_REVIEWS - 1, expected_precision=0.8)
    assert not few["warning"], "too few reviews to judge"
    low = drift_report(ref, ref, confirmed=5, reviewed=MIN_REVIEWS, expected_precision=0.8)
    assert low["warning"] and "precision" in low["warnings"][0]


def test_reviewer_precision_uses_latest_verdict(tmp_path):
    db = tmp_path / "d.db"
    assert reviewer_precision(db) == (0, 0)
    with closing(sqlite3.connect(db)) as con, con:
        con.execute("CREATE TABLE reviews (id INTEGER PRIMARY KEY, decision_id INT, label TEXT)")
        con.executemany("INSERT INTO reviews (decision_id, label) VALUES (?, ?)",
                        [(1, "fraud"), (1, "false_alarm"), (2, "fraud")])
    assert reviewer_precision(db) == (1, 2)


def test_cli_writes_report(prepared, tmp_path, capsys):
    out = tmp_path / "drift.json"
    assert main(["--reference", str(prepared / "features_train.csv"),
                 "--current", str(prepared / "features_test.csv"), "--out", str(out)]) == 0
    assert out.exists() and "no drift warning" in capsys.readouterr().out
