import joblib
import numpy as np
import pandas as pd
import pytest

from ml.explain import PHRASES, Explainer
from ml.export_onnx import onnx_fraud_scores, write_parity_files
from ml.features import FEATURE_NAMES


def test_onnx_matches_scikit_learn(prepared, trained):
    test = pd.read_csv(prepared / "features_test.csv")
    X = test[FEATURE_NAMES].to_numpy(dtype=np.float32)
    model = joblib.load(trained / "fraud_model.joblib")
    want = model.predict_proba(X)[:, 1]
    got = onnx_fraud_scores(trained / "fraud_model.onnx", X)
    assert np.max(np.abs(got - want)) < 1e-5


def test_parity_files(tmp_path):
    write_parity_files(tmp_path)
    n = int(dict(line.split("=") for line in
                 (tmp_path / "parity.meta.txt").read_text().split())["rows"])
    assert n > 1000
    assert (tmp_path / "parity.tx.bin").stat().st_size == 40 * n
    assert (tmp_path / "parity.features.bin").stat().st_size == 8 * n * len(FEATURE_NAMES)
    assert (tmp_path / "parity.scores.bin").stat().st_size == 8 * n
    assert "review=" in (tmp_path / "parity.thresholds.ini").read_text()


def test_every_feature_has_a_readable_phrase():
    assert set(PHRASES) == set(FEATURE_NAMES)
    for name, phrase in PHRASES.items():
        for v in (-1.0, 0.0, 1.0, 7.5):
            text = phrase(v)
            assert text and "{" not in text, name


@pytest.mark.parametrize("name, value, expected", [
    ("amount_vs_sender_avg", 12.4, "amount is 12.4x the sender's average"),
    ("pair_seen_before", 0.0, "new receiver for this sender"),
    ("sender_inflow_24h_ratio", 0.98,
     "amount is 98% of what the sender received in the last 24 hours"),
    ("hours_since_sender_first_seen", -1.0, "sender account never seen before"),
])
def test_phrases(name, value, expected):
    assert PHRASES[name](value) == expected


def test_reasons_for_flagged_fraud(prepared, trained):
    model = joblib.load(trained / "fraud_model.joblib")
    explainer = Explainer(model)
    val = pd.read_csv(prepared / "features_validation.csv")
    row = val[val["isFraud"] == 1].iloc[0][FEATURE_NAMES].to_numpy(dtype=float)
    reasons = explainer.reasons(row, top=3)
    assert 1 <= len(reasons) <= 3
    contribs = [r.contribution for r in reasons]
    assert all(c > 0 for c in contribs) and contribs == sorted(contribs, reverse=True)
    assert not any(r.feature.startswith("type_") and r.value == 0 for r in reasons)
    # SHAP values add up to the model's raw output (log-odds).
    raw = model._raw_predict(row.reshape(1, -1).astype(np.float32))[0, 0]
    base = float(np.ravel(explainer._explainer.expected_value)[-1])
    total = explainer.contributions(row).sum() + base
    assert total == pytest.approx(raw, abs=1e-3)
