import json

import joblib
import numpy as np
import pytest

from ml.evaluate import evaluate, to_markdown
from ml.evaluate import main as evaluate_main
from ml.train import BLOCK_PRECISION, choose_block_threshold, choose_review_threshold, load_set, xy


def test_review_threshold_balances_costs():
    y = np.array([0, 0, 1, 0, 1])
    score = np.array([0.1, 0.2, 0.5, 0.6, 0.7])
    # Catching the 0.5 fraud costs one false alarm (25); missing it costs its amount.
    big = np.array([0, 0, 1000.0, 0, 1000.0])
    assert choose_review_threshold(y, score, big) == pytest.approx(0.5)
    small = np.array([0, 0, 10.0, 0, 1000.0])
    assert choose_review_threshold(y, score, small) == pytest.approx(0.7)


def test_block_threshold_needs_high_precision():
    y = np.array([0] * 10 + [1] * 10)
    score = np.linspace(0, 1, 20)
    t = choose_block_threshold(y, score, review=0.0)
    flagged = score >= t
    assert y[flagged].mean() >= BLOCK_PRECISION
    assert choose_block_threshold(np.zeros(20, dtype=int), score, 0.0) == 1.0


def test_training_writes_a_consistent_model(trained, prepared):
    meta = json.loads((trained / "model_meta.json").read_text())
    t = meta["thresholds"]
    assert 0 <= t["review"] <= t["block"] <= 1
    ini = (trained / "thresholds.ini").read_text()
    assert f"review={t['review']!r}" in ini and f"model_version={meta['model_version']}" in ini
    model = joblib.load(trained / "fraud_model.joblib")
    assert model.n_features_in_ == len(meta["features"])
    v = meta["validation"]
    assert v["gradient_boosting"]["pr_auc"] > v["logistic"]["pr_auc"], "beats the baseline"
    assert v["gradient_boosting"]["pr_auc"] > v["rules"]["pr_auc"]


def test_training_never_reads_the_test_set(prepared, tmp_path, monkeypatch):
    import ml.train as train_mod

    seen = []
    real = train_mod.load_set
    monkeypatch.setattr(train_mod, "load_set", lambda d, n: seen.append(n) or real(d, n))
    train_mod.train(prepared, tmp_path / "m")
    assert "test" not in seen


def test_evaluate_reports_every_method(prepared, trained):
    r = evaluate(prepared, trained)
    assert set(r["methods"]) == {"rules", "logistic", "gradient_boosting",
                                 "gradient_boosting_block_only"}
    assert r["test_fraud"] == int(xy(load_set(prepared, "test"))[1].sum())
    assert "Rules only" in to_markdown(r)


def test_evaluate_refuses_to_reuse_the_test_set(prepared, trained, tmp_path):
    out = tmp_path / "r.json"
    args = ["--data", str(prepared), "--models", str(trained), "--out", str(out)]
    assert evaluate_main(args) == 0
    assert evaluate_main(args) == 1
    assert evaluate_main([*args, "--force"]) == 0


def test_evaluate_checks_engine_scores(prepared, trained, tmp_path):
    test = load_set(prepared, "test")
    model = joblib.load(trained / "fraud_model.joblib")
    p = model.predict_proba(xy(test)[0])[:, 1]
    meta = json.loads((trained / "model_meta.json").read_text())
    dec = np.where(p >= meta["thresholds"]["block"], "block",
                   np.where(p >= meta["thresholds"]["review"], "review", "approve"))
    path = tmp_path / "scores.csv"
    test[["tx_id"]].assign(score=p + 1e-7, decision=dec).to_csv(path, index=False)
    r = evaluate(prepared, trained, path)
    assert r["engine_parity"]["max_abs_score_diff"] < 1e-6
    assert r["engine_parity"]["decisions_agree"] == len(test)
    test[["tx_id"]].head(5).assign(score=0.0, decision="approve").to_csv(path, index=False)
    with pytest.raises(ValueError, match="cover"):
        evaluate(prepared, trained, path)
