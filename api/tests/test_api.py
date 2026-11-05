import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.app import create_app
from api.schemas import TransactionIn
from api.service import Settings

GOOD = {"step": 400, "type": "TRANSFER", "amount": 950.0, "nameOrig": "C123", "nameDest": "C456"}


@pytest.fixture()
def client(trained, prepared, tmp_path):
    settings = Settings(models_dir=trained, db_path=tmp_path / "decisions.db",
                        warm_file=prepared / "transactions.tx.bin", engine="python")
    with TestClient(create_app(settings)) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["engine"] == "python"
    assert r["warmed_transactions"] > 1000 and r["accounts_in_memory"] > 100


def test_score_returns_a_decision(client):
    r = client.post("/score", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["score"] <= 1
    assert body["decision"] in ("approve", "review", "block")
    assert body["model_version"] == "v1"
    assert body["latency_us"] > 0
    assert (body["reasons"] == []) == (body["decision"] == "approve")


def test_mule_pattern_is_flagged_with_reasons(client):
    """Money from several new victims into a new account, then cashed out at once."""
    for i, victim in enumerate(["C9001", "C9002", "C9003"]):
        client.post("/score", json={"step": 400, "type": "TRANSFER", "amount": 3000.0 + i,
                                    "nameOrig": victim, "nameDest": "C777777"})
    r = client.post("/score", json={"step": 401, "type": "CASH_OUT", "amount": 9003.0,
                                    "nameOrig": "C777777", "nameDest": "M5"}).json()
    assert r["decision"] in ("review", "block")
    assert 1 <= len(r["reasons"]) <= 3


@pytest.mark.parametrize("change", [
    {"amount": -1}, {"amount": "lots"}, {"type": "BITCOIN"}, {"nameOrig": "X1"},
    {"nameDest": "C"}, {"step": 0}, {"extra": "field"},
])
def test_bad_input_is_rejected(client, change):
    assert client.post("/score", json={**GOOD, **change}).status_code == 422


def test_infinite_amount_is_rejected():
    with pytest.raises(ValidationError):
        TransactionIn(**{**GOOD, "amount": float("inf")})


def test_decisions_are_logged_and_reviewable(client):
    for _ in range(3):
        client.post("/score", json={"step": 401, "type": "CASH_OUT", "amount": 20000.0,
                                    "nameOrig": "C31337", "nameDest": "M5"})
    flagged = client.get("/decisions").json()
    assert flagged, "the big cash-outs were flagged"
    first = flagged[0]
    r = client.post(f"/decisions/{first['id']}/review", json={"label": "fraud"})
    assert r.status_code == 200 and r.json()["label"] == "fraud"
    pending = client.get("/decisions", params={"pending": True}).json()
    assert first["id"] not in [d["id"] for d in pending]
    assert client.post("/decisions/999999/review", json={"label": "fraud"}).status_code == 404
    assert client.post(f"/decisions/{first['id']}/review",
                       json={"label": "maybe"}).status_code == 422
