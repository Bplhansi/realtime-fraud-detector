# Commands (Windows, Git Bash)

Run these from the project folder. Requires Python 3.11+, Visual Studio 2022 ("Desktop
development with C++"), CMake 3.20+ and Git.

## One-time setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt
```

## Python tests

```bash
.venv/Scripts/python -m pytest
.venv/Scripts/python -m pytest --cov=sim --cov=ml --cov=api --cov=dashboard   # with coverage
```

## C++ engine: build and test

```bash
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$PWD/.venv/Scripts/python.exe"
cmake --build engine/build --config Release
ctest --test-dir engine/build -C Release --output-on-failure
```

The first configure downloads GoogleTest and ONNX Runtime (both pinned and checked by SHA-256).
The engine program ends up at `engine/build/Release/fd_replay.exe` and the Python module in
`engine/build/python/`.

## The whole pipeline

```bash
source .venv/Scripts/activate
FD_REPLAY=engine/build/Release/fd_replay.exe sh scripts/demo-pipeline.sh
```

Or step by step:

```bash
python -m sim.generate --out data/transactions.csv
python -m ml.explore --csv data/transactions.csv
python -m ml.prepare --csv data/transactions.csv --out data
python -m ml.train
python -m ml.export_onnx
engine/build/Release/fd_replay.exe data/transactions.tx.bin --model models/fraud_model.onnx --thresholds models/thresholds.ini --from-step 614 --repeat 3 --scores data/engine_scores.csv
python -m ml.evaluate --engine-scores data/engine_scores.csv
python -m api.load_queue
python -m ml.drift --current data/features_test.csv
```

`--from-step` is the `validation_end_step` in `data/split.json` (614 for the default data).
`ml.evaluate` refuses to run twice; add `--force` only if you mean to reuse the test set.

## Re-measure latency on your machine

```bash
engine/build/Release/fd_replay.exe data/transactions.tx.bin --model models/fraud_model.onnx --thresholds models/thresholds.ini --from-step 614 --repeat 5
```

Put the throughput and latency lines, and your CPU, RAM and Windows version, in
`docs/results.md` and the README. Only quote numbers you measured.

## Run the services

```bash
python -m uvicorn api.app:app --port 8000          # API: http://127.0.0.1:8000/docs
python -m streamlit run dashboard/app.py           # dashboard: http://localhost:8501
```

The API uses the C++ engine if `engine/build/python` has the module, otherwise the Python one
(same results, slower). To give it account history: `FD_WARM_FILE=data/history.tx.bin`.

## Drift warning demo

```bash
python -m sim.generate --out data/shifted.csv --shift 3 --seed 99 --customers 5000 --days 7
python -m ml.drift --current-csv data/shifted.csv --out data/drift_report.json
```

Reload the dashboard: a red drift warning appears at the top. Re-run
`python -m ml.drift --current data/features_test.csv` to clear it.
