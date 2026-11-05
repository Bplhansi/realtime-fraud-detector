"""Export the trained model to ONNX for the C++ engine, and write parity files for its tests.

The ONNX model takes float32 rows [N, n_features] (config/features.txt order) and returns
"probabilities" [N, 2]; the fraud score is column 1.

Usage:
    python -m ml.export_onnx --model models/fraud_model.joblib --out models/fraud_model.onnx
    python -m ml.export_onnx --parity-dir <dir>   # small model, transactions, expected features
                                                  # and scores, for the C++ parity tests
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import HistGradientBoostingClassifier

from ml.data import load_transactions, to_tx_array
from ml.features import FEATURE_NAMES, compute_features

PARITY_CUSTOMERS = 400  # small simulated dataset: a few thousand transactions
PARITY_DAYS = 10


def export_onnx(model: HistGradientBoostingClassifier, path: Path) -> None:
    onx = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, model.n_features_in_]))],
        options={id(model): {"zipmap": False}},  # a plain probability tensor
        target_opset={"": 17, "ai.onnx.ml": 3},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onx.SerializeToString())


def onnx_fraud_scores(onnx_path: Path, X: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return sess.run(["probabilities"], {"features": X.astype(np.float32)})[0][:, 1]


def write_parity_files(out: Path) -> None:
    """parity.tx.bin (transactions), parity.features.bin (float64 [N, F]), parity.onnx,
    parity.scores.bin (float64 [N], Python's P(fraud) for the float32 features) and
    parity.thresholds.ini."""
    from sim.generate import SimConfig, simulate, write_csv

    out.mkdir(parents=True, exist_ok=True)
    cfg = SimConfig(customers=PARITY_CUSTOMERS, merchants=40, days=PARITY_DAYS,
                    fraud_case_rate=0.05, seed=123)
    csv_path = out / "parity.csv"
    write_csv(simulate(cfg), csv_path)
    df = load_transactions(csv_path)
    X = compute_features(df)
    to_tx_array(df).tofile(out / "parity.tx.bin")
    X.astype("<f8").tofile(out / "parity.features.bin")
    model = HistGradientBoostingClassifier(max_iter=40, class_weight="balanced",
                                           early_stopping=False, random_state=0)
    model.fit(X.astype(np.float32), df["isFraud"].to_numpy())
    export_onnx(model, out / "parity.onnx")
    scores = model.predict_proba(X.astype(np.float32))[:, 1]
    scores.astype("<f8").tofile(out / "parity.scores.bin")
    (out / "parity.thresholds.ini").write_text("model_version=parity\nreview=0.5\nblock=0.9\n")
    (out / "parity.meta.txt").write_text(f"rows={len(df)}\nfeatures={len(FEATURE_NAMES)}\n")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export the trained model to ONNX.")
    p.add_argument("--model", type=Path, default=Path("models/fraud_model.joblib"))
    p.add_argument("--out", type=Path, default=Path("models/fraud_model.onnx"))
    p.add_argument("--parity-dir", type=Path, help="write the parity test files instead")
    args = p.parse_args(argv)
    if args.parity_dir:
        write_parity_files(args.parity_dir)
        print(f"parity files written to: {args.parity_dir}")
        return 0
    model = joblib.load(args.model)
    export_onnx(model, args.out)
    print(f"ONNX model written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
