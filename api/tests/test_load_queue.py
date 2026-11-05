from api.load_queue import main
from api.store import DecisionStore


def test_load_queue_scores_the_test_period(small_csv, prepared, trained, tmp_path, capsys):
    db = tmp_path / "q.db"
    args = ["--csv", str(small_csv), "--split", str(prepared / "split.json"), "--db", str(db),
            "--models", str(trained), "--engine", "python", "--max-rows", "800"]
    assert main(args) == 0
    assert "scored 800" in capsys.readouterr().out
    store = DecisionStore(db)
    assert sum(store.stats()["decisions"].values()) == 800
    for d in store.flagged():
        assert d["reasons"], "every flagged case has reasons"
        assert d["is_fraud"] in (0, 1)
    assert main(args) == 0  # runs again from a fresh database
    assert sum(DecisionStore(db).stats()["decisions"].values()) == 800
