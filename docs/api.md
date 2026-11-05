# Scoring API

Start it (after the pipeline has made `models/`):

```bash
python -m uvicorn api.app:app --port 8000
```

Interactive docs: http://127.0.0.1:8000/docs

Settings (environment variables):

| Variable | Default | Meaning |
| --- | --- | --- |
| `FD_MODELS_DIR` | `models` | ONNX model, joblib model, thresholds, metadata |
| `FD_DB_PATH` | `data/decisions.db` | SQLite decision log |
| `FD_WARM_FILE` | (none) | past transactions (`.tx.bin`) loaded into account history at start |
| `FD_ENGINE` | `auto` | `cpp` (pybind11 module from `engine/build/python`), `python`, or `auto` |
| `FD_ENGINE_DIR` | `engine/build/python` | where to find the C++ module |

Account history lives in the process's memory, so run one worker.

## POST /score

```json
{"step": 620, "type": "TRANSFER", "amount": 9500, "nameOrig": "C1231006815", "nameDest": "C9999999"}
```

Response (example):

```json
{
  "decision_id": 1,
  "score": 0.9958,
  "decision": "block",
  "reasons": ["large amount (9,500)", "receiver is a person, not a merchant", "transaction type is TRANSFER"],
  "model_version": "v1",
  "engine": "cpp",
  "latency_us": 175.5
}
```

`reasons` is empty for approved transactions (SHAP only runs for flagged ones).
`latency_us` is the engine's time for features + model; the first call is slower.

Input checks (Pydantic): `step` 1 to 1,000,000; `type` one of the five PaySim types; `amount`
finite, 0 to 1e10; account names like `C123` or `M123`; no extra fields. Anything else gets
HTTP 422 with the reason.

## GET /decisions?pending=true&limit=100

Flagged decisions (review or block), highest score first, each with its latest reviewer verdict
(`review_label`). `pending=true` returns only those without a verdict.

## POST /decisions/{id}/review

```json
{"label": "fraud", "reviewer": "hansika"}
```

`label` is `fraud` or `false_alarm`. Verdicts are appended, never edited; the latest one counts.
Unknown id: HTTP 404.

## GET /health

Engine in use, model version, accounts in memory, transactions loaded at start.
