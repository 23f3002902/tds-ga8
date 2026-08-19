# TDS 2026 May GA8

Deterministic solutions for the MLOps and Fine-Tuning graded assignment.

## Run locally

```bash
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8765
```

Implemented routes:

- `POST /build-corpus` - immutable, leakage-safe corpus builder
- `POST /bqml` - stateful BigQuery ML selection/evaluation gate
