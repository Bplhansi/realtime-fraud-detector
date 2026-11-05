import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from api.store import DecisionStore
from dashboard.data import activity_frame, alerts_frame, queue_frame, read_json, reviewer_precision

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _row(**kw):
    base = {"step": 5, "type": "TRANSFER", "amount": 900.0, "name_orig": "C1", "name_dest": "C2",
            "score": 0.95, "decision": "block", "reasons": ["new receiver for this sender"],
            "model_version": "v1", "engine": "cpp", "latency_us": 8.0}
    return {**base, **kw}


@pytest.fixture()
def db(tmp_path):
    store = DecisionStore(tmp_path / "d.db")
    store.add_decisions([_row(), _row(score=0.4, decision="review", name_orig="C3"),
                         _row(score=0.01, decision="approve", step=4)])
    return tmp_path / "d.db"


def test_data_helpers(db):
    store = DecisionStore(db)
    q = queue_frame(store, pending_only=True)
    assert q["score"].tolist() == [0.95, 0.4]
    act = activity_frame(store, "C1", before_step=5)
    assert set(act["direction"]) == {"sent"}
    alerts = alerts_frame(store)
    assert alerts[(alerts.step == 5) & (alerts.decision == "block")]["alerts"].item() == 1
    assert reviewer_precision(store.stats()) is None
    store.add_review(1, "fraud", "t")
    assert reviewer_precision(store.stats()) == 1.0


def test_read_json(tmp_path):
    assert read_json(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{not json")
    assert read_json(tmp_path / "bad.json") is None


def test_app_shows_queue_and_records_a_verdict(db, tmp_path, monkeypatch):
    drift = tmp_path / "drift.json"
    drift.write_text(json.dumps({"warning": True, "warnings": ["feature drift: amount_log"],
                                 "psi": {"amount_log": 1.2}}))
    monkeypatch.setenv("FD_DB_PATH", str(db))
    monkeypatch.setenv("FD_DRIFT_REPORT", str(drift))
    monkeypatch.setenv("FD_TEST_RESULTS", str(tmp_path / "none.json"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("drift" in e.value for e in at.error)
    assert [b.label for b in at.button] == ["Confirm fraud", "False alarm"]
    at.button[0].click().run()
    assert not at.exception
    assert DecisionStore(db).stats()["reviews"]["fraud"] == 1


def test_app_without_database(tmp_path, monkeypatch):
    monkeypatch.setenv("FD_DB_PATH", str(tmp_path / "missing.db"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "No decisions yet" in at.warning[0].value
