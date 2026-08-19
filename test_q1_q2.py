from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient

from main import app
from q1_corpus import crc32c_hex
from q2_bqml import reset_runs_for_tests


client = TestClient(app)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def test_q1_happy_path_and_digest():
    row = {
        "id": "r1",
        "entity": "  ACME\u00a0Corp ",
        "eventTime": "2026-01-01T05:30:00+05:30",
        "revision": 1,
        "text": " Hello   WORLD ",
    }
    content = compact(row) + "\n"
    payload = {
        "policy": {
            "minTime": "2025-12-31T00:00:00Z",
            "maxTime": "2026-12-31T00:00:00Z",
            "contaminationThreshold": 1,
        },
        "objects": [
            {
                "uri": "gs://bucket/data.jsonl",
                "generation": "7",
                "fetchedGeneration": "7",
                "crc32c": crc32c_hex(content.encode()),
                "schemaId": "training-v1",
                "content": content,
            }
        ],
    }
    response = client.post("/build-corpus", json=payload)
    assert response.status_code == 200
    data = response.json()
    all_rows = data["splits"]["train"] + data["splits"]["validation"] + data["splits"]["test"]
    assert len(all_rows) == 1
    output = all_rows[0]
    assert output["entity"] == "acme corp"
    assert output["eventTime"] == "2026-01-01T00:00:00.000Z"
    assert output["text"] == "hello world"
    split = next(name for name, rows in data["splits"].items() if rows)
    expected = hashlib.sha256((compact(output) + "\n").encode()).hexdigest()
    assert data["digests"][split] == expected


def test_q1_independent_object_codes():
    response = client.post(
        "/build-corpus",
        json={
            "policy": {"minTime": "bad", "maxTime": "bad", "contaminationThreshold": 2},
            "objects": [
                {
                    "uri": 4,
                    "generation": "x",
                    "fetchedGeneration": "y",
                    "crc32c": "BAD",
                    "schemaId": "wrong",
                    "content": "not json",
                }
            ],
        },
    )
    assert response.status_code == 200
    assert response.json()["rejectedObjects"][0]["reasonCodes"] == sorted(
        [
            "URI_INVALID",
            "GENERATION_INVALID",
            "GENERATION_MISMATCH",
            "CRC32C_INVALID",
            "SCHEMA_INVALID",
            "JSONL_INVALID",
        ]
    )


def test_q2_select_replay_conflict_and_evaluate():
    reset_runs_for_tests()
    select = {
        "phase": "select",
        "runId": "run-1",
        "forbiddenFeatures": ["future"],
        "numTrialsLimit": 3,
        "rows": [
            {
                "id": "train",
                "entity": "e1",
                "eventTime": "2026-01-01T00:00:00Z",
                "predictionTime": "2026-01-01T00:00:01Z",
                "version": 1,
                "split": "TRAIN",
                "features": {
                    "safe": {"value": 1, "availableAt": "2026-01-01T00:00:00Z"},
                    "future": {"value": 9, "availableAt": "2026-01-01T00:00:02Z"},
                },
            },
            {
                "id": "eval",
                "entity": "e2",
                "eventTime": "2026-01-02T00:00:00Z",
                "predictionTime": "2026-01-02T00:00:01Z",
                "version": 1,
                "split": "EVAL",
                "features": {"safe": {"value": 2, "availableAt": "2026-01-02T00:00:00Z"}},
            },
        ],
        "trials": [
            {"trialId": 9, "status": "SUCCEEDED", "evalMetric": 0.9},
            {"trialId": 4, "status": "SUCCEEDED", "evalMetric": 0.9},
        ],
    }
    response = client.post("/bqml", json=select)
    assert response.status_code == 200
    chosen = response.json()
    assert chosen["selectedTrialId"] == 4
    assert chosen["featureNames"] == ["safe"]
    assert client.post("/bqml", json=select).json() == chosen
    changed = dict(select, numTrialsLimit=4)
    assert client.post("/bqml", json=changed).status_code == 409

    evaluate = {
        "phase": "evaluate",
        "runId": "run-1",
        "selectedTrialId": chosen["selectedTrialId"],
        "datasetDigest": chosen["datasetDigest"],
        "metricFloor": 0.5,
        "requiredSlices": {"critical": 0.75},
        "rows": [
            {"label": 1, "prediction": 1, "slice": "critical"},
            {"label": 0, "prediction": 0, "slice": "critical"},
        ],
        "bytesProcessed": 100,
        "maxBytes": 100,
    }
    result = client.post("/bqml", json=evaluate).json()
    assert result["decision"] == "admit"
    assert result["criticalSlicePass"] is True
    assert result["testMetric"] == 1.0
