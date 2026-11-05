import csv

import pytest

from sim.generate import COLUMNS, FLAG_LIMIT, SimConfig, main, simulate, write_csv

TINY = SimConfig(customers=300, merchants=30, days=5, fraud_case_rate=0.05, seed=3)


def test_same_seed_same_data():
    a, b = simulate(TINY), simulate(TINY)
    assert [(t.t, t.amount, t.orig) for t in a] == [(t.t, t.amount, t.orig) for t in b]


def test_sorted_in_time_and_inside_the_period():
    txs = simulate(TINY)
    times = [t.t for t in txs]
    assert times == sorted(times)
    assert 0 <= times[0] and times[-1] < TINY.days * 24


def test_fraud_is_transfer_then_cash_out_by_the_mule():
    txs = simulate(TINY)
    fraud = [t for t in txs if t.fraud]
    assert fraud, "some fraud is simulated"
    assert {t.type for t in fraud} <= {"TRANSFER", "CASH_OUT"}
    mules = {t.dest for t in fraud if t.type == "TRANSFER"}
    for t in fraud:
        if t.type == "CASH_OUT":
            assert t.orig in mules
            first_in = min(x.t for x in fraud if x.type == "TRANSFER" and x.dest == t.orig)
            assert t.t > first_in, "a mule cashes out only after receiving money"


def test_fraud_is_rare_but_present():
    txs = simulate(TINY)
    share = sum(t.fraud for t in txs) / len(txs)
    assert 0.001 < share < 0.05


def test_shift_makes_amounts_bigger_and_transfers_commoner():
    base = simulate(TINY)
    shifted = simulate(SimConfig(**{**TINY.__dict__, "shift": 3.0}))

    def avg(txs):
        return sum(t.amount for t in txs) / len(txs)

    def transfers(txs):
        return sum(t.type == "TRANSFER" for t in txs) / len(txs)

    assert avg(shifted) > 2 * avg(base)
    assert transfers(shifted) > transfers(base)


def test_csv_has_paysim_columns(tmp_path):
    path = tmp_path / "t.csv"
    txs = simulate(TINY)
    write_csv(txs, path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0]) == COLUMNS
    assert len(rows) == len(txs)
    assert min(int(r["step"]) for r in rows) >= 1, "PaySim steps start at 1"
    for r in rows:
        assert r["isFlaggedFraud"] == str(int(r["type"] == "TRANSFER" and
                                              float(r["amount"]) > FLAG_LIMIT))
        assert r["nameOrig"][0] == "C"


def test_balance_leak_exists_in_the_raw_file():
    """The raw data really has PaySim's leak, so dropping it (ml/data.py) matters."""
    txs = simulate(TINY)
    fraud_transfers = [t for t in txs if t.fraud and t.type == "TRANSFER"]
    emptied = sum(t.new_orig == 0.0 for t in fraud_transfers) / len(fraud_transfers)
    normal = [t for t in txs if not t.fraud and t.type != "CASH_IN"]
    emptied_normal = sum(t.new_orig == 0.0 for t in normal) / len(normal)
    assert emptied > 0.5 > emptied_normal


def test_cli(tmp_path, capsys):
    out = tmp_path / "x.csv"
    assert main(["--out", str(out), "--customers", "50", "--days", "3"]) == 0
    assert out.exists()
    assert "transactions" in capsys.readouterr().out


def test_cli_rejects_bad_arguments(tmp_path):
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "x.csv"), "--shift", "0"])
