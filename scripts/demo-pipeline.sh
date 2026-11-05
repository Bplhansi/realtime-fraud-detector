#!/bin/sh
# The whole pipeline, start to finish. Docker runs this; it also works in Git Bash or Linux
# from the project folder (set FD_REPLAY to the engine program if it isn't on the PATH).
set -eu
FD_REPLAY="${FD_REPLAY:-fd_replay}"

echo "== 1. simulate transactions (PaySim format)"
python -m sim.generate --out data/transactions.csv

echo "== 2. features + time-based split"
python -m ml.prepare --csv data/transactions.csv --out data

echo "== 3. train rules / logistic regression / gradient boosting, choose thresholds"
python -m ml.train --data data --models models

echo "== 4. export the model to ONNX"
python -m ml.export_onnx --model models/fraud_model.joblib --out models/fraud_model.onnx

echo "== 5. replay every transaction through the C++ engine (test period timed)"
VAL_END=$(python -c "import json; print(json.load(open('data/split.json'))['validation_end_step'])")
"$FD_REPLAY" data/transactions.tx.bin --model models/fraud_model.onnx \
    --thresholds models/thresholds.ini --from-step "$VAL_END" --repeat 3 \
    --scores data/engine_scores.csv

echo "== 6. score the test set once"
python -m ml.evaluate --data data --models models --engine-scores data/engine_scores.csv \
    --out reports/test_results.json --force

echo "== 7. fill the review queue (test period, with reasons)"
python -m api.load_queue --csv data/transactions.csv --split data/split.json \
    --db data/decisions.db --models models

echo "== 8. drift check: test period, then a deliberately shifted file"
python -m ml.drift --reference data/features_train.csv --current data/features_test.csv \
    --db data/decisions.db --out data/drift_report.json
python -m sim.generate --out data/shifted.csv --shift 3 --seed 99 --customers 5000 --days 7
python -m ml.drift --reference data/features_train.csv --current-csv data/shifted.csv \
    --out data/drift_report_shifted.json

echo "== done"
