# Results

All numbers below come from `scripts/demo-pipeline.sh` on the default simulated file (seed 7),
measured on 2026-11-05. The Docker demo produced the same model, thresholds and test results.

**Machine:** 4-vCPU Intel Xeon @ 2.30 GHz cloud VM, Linux, GCC 13.3 Release `-O3`, one thread.

## Why these metrics

Fraud is 0.25% of rows, so accuracy says nothing. We report precision (of flagged, how many
were fraud), recall (of fraud, how much was flagged), PR-AUC (quality across all thresholds,
better than ROC-AUC when the positive class is rare), recall@100 (fraud caught if reviewers can
only check the 100 highest scores in the period) and the share of fraud *money* caught.

## Validation (used for every choice)

Model settings tried (gradient boosting, class weights, 17 features), best by validation PR-AUC:

| Learning rate | Leaves | Trees | Validation PR-AUC |
| --- | --- | --- | --- |
| 0.1 | 15 | 150 | 0.782 |
| 0.1 | 31 | 150 | 0.781 |
| **0.05** | **31** | **300** | **0.806** |

Thresholds chosen on validation:

- **review = 0.1421**: lowest total cost, where cost = missed fraud amount + 25 per false alarm.
- **block = 0.9854**: lowest score with at least 90% validation precision.

| Validation | Precision | Recall | PR-AUC | Recall@100 | Fraud value caught |
| --- | --- | --- | --- | --- | --- |
| Rules only | 0.147 | 0.324 | 0.050 | 0.073 | 0.436 |
| Logistic regression | 0.088 | 0.831 | 0.082 | 0.037 | 0.982 |
| Gradient boosted model | 0.382 | 0.863 | 0.806 | 0.457 | 0.970 |

The model beats both baselines on validation, so it went to the test set.

## Test (scored once)

78,504 transactions, 161 fraud (`reports/test_results.json`):

| Method | Precision | Recall | PR-AUC | Recall@100 | Fraud value caught | Flagged |
| --- | --- | --- | --- | --- | --- | --- |
| Rules only | 0.128 | 0.280 | 0.037 | 0.106 | 0.443 | 352 |
| Logistic regression | 0.066 | 0.764 | 0.055 | 0.006 | 0.977 | 1,872 |
| Gradient boosted model (review or block) | 0.374 | 0.851 | 0.748 | 0.528 | 0.979 | 366 |
| Gradient boosted model (block only) | 0.847 | 0.515 | 0.748 | 0.528 | 0.742 | 98 |

PR-AUC dropped from 0.806 (validation) to 0.748 (test), a normal gap for a later period.
Reading the table: the model sends 366 of 78,504 transactions (0.47%) to review or block, and
those contain 85% of the fraud cases and 98% of the fraud money. Rules alone catch 28%.

## C++ engine

`fd_replay data/transactions.tx.bin --model models/fraud_model.onnx --thresholds models/thresholds.ini --from-step 614 --repeat 3`

Every transaction updates the feature store; the 78,504 test-period transactions are timed
(features + ONNX model + decision, one at a time), pooled over 3 runs:

| Metric | Value |
| --- | --- |
| Throughput (all 527,372 transactions) | 111,333 tx/s (best run 114,288) |
| Latency median | 7.7 µs |
| Latency 99th percentile | 25.4 µs |
| Decisions on test | 78,138 approve, 268 review, 98 block |
| C++ vs Python scores | max difference 5.05e-07; 78,504 / 78,504 decisions agree |

Through the API path (Python → pybind11 → C++), latency logged per decision is 9 µs median,
35 µs 99th percentile. SHAP reasons add about 2 ms, for flagged transactions only.

## Drift check

PSI against the training data (above 0.25 = shifted):

| Feature | Test period | Shifted file (`--shift 3`) |
| --- | --- | --- |
| amount_log | 0.0002 | **1.546** |
| amount_vs_sender_avg | 0.056 | 0.103 |
| sender_tx_24h | 0.0007 | 0.0008 |
| receiver_distinct_senders_24h | 0.003 | **0.899** |
| type_CASH_OUT | 0.000 | 0.001 |
| type_PAYMENT | 0.000 | **0.452** |
| type_TRANSFER | 0.000 | **0.423** |

The test period raises no warning; the deliberately shifted file does.

## Tests

- C++: 37 tests (GoogleTest + CTest), also clean under AddressSanitizer + UBSan.
- Python: 78 tests, 96% line coverage (`pytest --cov`).
