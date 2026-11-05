import pandas as pd

from ml.explore import explore, main


def test_explore_finds_where_fraud_is_and_the_leak(small_csv):
    r = explore(pd.read_csv(small_csv))
    assert r["fraud"] > 0
    fraud_types = {t for t, v in r["by_type"].items() if v["sum"] > 0}
    assert fraud_types == {"TRANSFER", "CASH_OUT"}
    z = r["sender_left_with_zero"]
    assert z["fraud"] > z["genuine"]
    assert r["median_amount"]["fraud"] > r["median_amount"]["genuine"]


def test_cli_saves_a_figure(small_csv, tmp_path, capsys):
    fig = tmp_path / "f.png"
    assert main(["--csv", str(small_csv), "--figure", str(fig)]) == 0
    assert fig.stat().st_size > 1000
    assert "LEAK" in capsys.readouterr().out
