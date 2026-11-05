import pytest

from api.engine import PythonEngine, load_engine, read_thresholds
from api.store import DecisionStore
from ml.data import load_transactions, read_tx_file


def _row(**kw):
    base = {"step": 1, "type": "PAYMENT", "amount": 10.0, "name_orig": "C1", "name_dest": "M1",
            "score": 0.1, "decision": "approve", "reasons": [], "model_version": "v1",
            "engine": "python", "latency_us": 5.0}
    return {**base, **kw}


def test_store_round_trip_and_queue(tmp_path):
    store = DecisionStore(tmp_path / "d.db")
    ids = store.add_decisions([_row(), _row(score=0.9, decision="block", reasons=["a", "b"]),
                               _row(score=0.5, decision="review", name_orig="C2")])
    assert store.get(ids[1])["reasons"] == ["a", "b"]
    assert [d["id"] for d in store.flagged()] == [ids[1], ids[2]], "highest score first"
    store.add_review(ids[1], "false_alarm", "me")
    store.add_review(ids[1], "fraud", "me")  # changed mind: a new row, latest wins
    assert [d["id"] for d in store.flagged(pending_only=True)] == [ids[2]]
    assert store.flagged()[0]["review_label"] == "fraud"
    s = store.stats()
    assert s["decisions"] == {"approve": 1, "review": 1, "block": 1}
    assert s["reviews"] == {"fraud": 1, "false_alarm": 0}
    assert s["latency_us_p50"] == 5.0
    assert len(store.account_activity("C1")) == 2
    with pytest.raises(KeyError):
        store.add_review(999, "fraud", "me")
    with pytest.raises(ValueError):
        store.add_review(ids[0], "unsure", "me")


def test_python_engine_decides_with_thresholds(trained):
    eng = PythonEngine(trained / "fraud_model.onnx", trained / "thresholds.ini")
    t = read_thresholds(trained / "thresholds.ini")
    r = eng.score(1, "PAYMENT", 12.0, 1, 2)
    assert r["decision"] == ("block" if r["score"] >= float(t["block"]) else
                             "review" if r["score"] >= float(t["review"]) else "approve")
    assert eng.accounts == 2


def test_engine_choice(trained):
    with pytest.raises(ValueError):
        load_engine(trained / "fraud_model.onnx", trained / "thresholds.ini", prefer="gpu")
    eng = load_engine(trained / "fraud_model.onnx", trained / "thresholds.ini", prefer="python")
    assert eng.kind == "python"


def test_cpp_and_python_engines_agree(trained, prepared, small_csv):
    """Same scores from the C++ engine (if built) and the Python engine."""
    try:
        cpp = load_engine(trained / "fraud_model.onnx", trained / "thresholds.ini", prefer="cpp")
    except RuntimeError:
        pytest.skip("C++ engine module not built")
    py = load_engine(trained / "fraud_model.onnx", trained / "thresholds.ini", prefer="python")
    df = load_transactions(small_csv)
    for r in df.head(3000).itertuples(index=False):
        a = cpp.score(int(r.step), r.type, float(r.amount), int(r.orig_id), int(r.dest_id))
        b = py.score(int(r.step), r.type, float(r.amount), int(r.orig_id), int(r.dest_id))
        assert a["features"] == pytest.approx(b["features"], rel=1e-9)
        assert a["score"] == pytest.approx(b["score"], abs=1e-6)
        assert a["decision"] == b["decision"]
    assert cpp.remember_file(prepared / "transactions.tx.bin") == len(
        read_tx_file(prepared / "transactions.tx.bin"))
