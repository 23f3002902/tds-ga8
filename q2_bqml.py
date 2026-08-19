from __future__ import annotations

import copy
import hashlib
import threading
from typing import Any

from ga8_utils import (
    compact_json,
    is_finite_number,
    is_safe_integer,
    parse_timestamp,
    sorted_codes,
    utf8_key,
)


_RUNS: dict[str, tuple[str, dict[str, Any]]] = {}
_LOCK = threading.Lock()


def _valid_run_id(value: Any) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 128


def _valid_selection(payload: dict[str, Any]) -> bool:
    forbidden = payload.get("forbiddenFeatures")
    rows = payload.get("rows")
    trials = payload.get("trials")
    limit = payload.get("numTrialsLimit")
    if (
        not _valid_run_id(payload.get("runId"))
        or not isinstance(forbidden, list)
        or any(not isinstance(name, str) for name in forbidden)
        or not is_safe_integer(limit, positive=True)
        or not isinstance(rows, list)
        or not rows
        or not isinstance(trials, list)
    ):
        return False

    row_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        row_id = row.get("id")
        features = row.get("features")
        if (
            not isinstance(row_id, str)
            or row_id in row_ids
            or not isinstance(row.get("entity"), str)
            or parse_timestamp(row.get("eventTime")) is None
            or parse_timestamp(row.get("predictionTime")) is None
            or not is_safe_integer(row.get("version"))
            or row.get("split") not in {"TRAIN", "EVAL"}
            or not isinstance(features, dict)
            or any(not isinstance(name, str) for name in features)
        ):
            return False
        row_ids.add(row_id)
        for feature in features.values():
            if not isinstance(feature, dict) or parse_timestamp(feature.get("availableAt")) is None or "value" not in feature:
                return False

    trial_ids: set[int] = set()
    for trial in trials:
        if not isinstance(trial, dict):
            return False
        trial_id = trial.get("trialId")
        if (
            not is_safe_integer(trial_id)
            or trial_id in trial_ids
            or trial.get("status") not in {"SUCCEEDED", "FAILED"}
        ):
            return False
        trial_ids.add(trial_id)
    return True


def _select(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    run_id = payload.get("runId")
    if not _valid_run_id(run_id):
        return 200, {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": ["INVALID_INPUT"],
        }

    try:
        fingerprint = hashlib.sha256(compact_json(payload, sort_keys=True).encode("utf-8")).hexdigest()
    except (TypeError, ValueError):
        fingerprint = "invalid-json-value"

    with _LOCK:
        previous = _RUNS.get(run_id)
        if previous is not None:
            if previous[0] != fingerprint:
                return 409, {"error": "RUN_ID_CONFLICT"}
            return 200, copy.deepcopy(previous[1])

    if not _valid_selection(payload):
        response = {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": [],
            "evalRowIds": [],
            "featureNames": [],
            "datasetDigest": None,
            "reasonCodes": ["INVALID_INPUT"],
        }
    elif len(payload["trials"]) > payload["numTrialsLimit"]:
        response = {
            "runId": run_id, "selectedTrialId": None, "trainRowIds": [], "evalRowIds": [],
            "featureNames": [], "datasetDigest": None, "reasonCodes": ["TRIAL_LIMIT_EXCEEDED"],
        }
    else:
        retained_by_key: dict[tuple[str, Any], dict[str, Any]] = {}
        for row in payload["rows"]:
            event_time = parse_timestamp(row["eventTime"])
            key = (row["entity"], event_time)
            current = retained_by_key.get(key)
            if current is None or row["version"] > current["version"] or (
                row["version"] == current["version"] and utf8_key(row["id"]) < utf8_key(current["id"])
            ):
                retained_by_key[key] = row
        retained = list(retained_by_key.values())

        forbidden = set(payload["forbiddenFeatures"])
        common_features = set(retained[0]["features"])
        for row in retained[1:]:
            common_features &= set(row["features"])
        eligible_features: list[str] = []
        for name in common_features:
            if name in forbidden:
                continue
            if all(
                parse_timestamp(row["features"][name]["availableAt"])
                <= parse_timestamp(row["predictionTime"])
                for row in retained
            ):
                eligible_features.append(name)
        eligible_features.sort(key=utf8_key)

        train_ids = sorted((row["id"] for row in retained if row["split"] == "TRAIN"), key=utf8_key)
        eval_ids = sorted((row["id"] for row in retained if row["split"] == "EVAL"), key=utf8_key)
        digest_payload = {
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": eligible_features,
        }
        dataset_digest = hashlib.sha256(compact_json(digest_payload).encode("utf-8")).hexdigest()

        eligible_trials = [
            trial
            for trial in payload["trials"]
            if trial["status"] == "SUCCEEDED" and is_finite_number(trial.get("evalMetric"))
        ]
        eligible_trials.sort(key=lambda trial: (-float(trial["evalMetric"]), trial["trialId"]))
        selected = eligible_trials[0]["trialId"] if eligible_trials else None
        response = {
            "runId": run_id,
            "selectedTrialId": selected,
            "trainRowIds": train_ids,
            "evalRowIds": eval_ids,
            "featureNames": eligible_features,
            "datasetDigest": dataset_digest,
            "reasonCodes": [] if selected is not None else ["NO_SUCCESSFUL_TRIAL"],
        }

    with _LOCK:
        _RUNS[run_id] = (fingerprint, copy.deepcopy(response))
    return 200, response


def _evaluate(payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    codes: list[str] = []
    run_id = payload.get("runId")
    selected_trial = payload.get("selectedTrialId")
    digest = payload.get("datasetDigest")
    metric_floor = payload.get("metricFloor")
    required_slices = payload.get("requiredSlices")
    rows = payload.get("rows")
    bytes_processed = payload.get("bytesProcessed")
    max_bytes = payload.get("maxBytes")

    valid_floors = (
        is_finite_number(metric_floor)
        and 0 <= float(metric_floor) <= 1
        and isinstance(required_slices, dict)
        and all(
            isinstance(name, str)
            and name != ""
            and is_finite_number(floor)
            and 0 <= float(floor) <= 1
            for name, floor in required_slices.items()
        )
    )
    valid_counts = is_safe_integer(bytes_processed) and is_safe_integer(max_bytes)
    if not valid_floors or not valid_counts or not isinstance(rows, list):
        codes.append("INVALID_INPUT")

    with _LOCK:
        stored = _RUNS.get(run_id) if _valid_run_id(run_id) else None
    stored_response = stored[1] if stored is not None else None
    lineage_valid = (
        stored_response is not None
        and stored_response.get("selectedTrialId") is not None
        and is_safe_integer(selected_trial)
        and selected_trial == stored_response.get("selectedTrialId")
        and isinstance(digest, str)
        and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest)
        and digest == stored_response.get("datasetDigest")
    )
    if not lineage_valid:
        codes.append("INVALID_LINEAGE")

    valid_rows = isinstance(rows, list) and bool(rows)
    if valid_rows:
        for row in rows:
            if (
                not isinstance(row, dict)
                or isinstance(row.get("label"), bool)
                or row.get("label") not in {0, 1}
                or isinstance(row.get("prediction"), bool)
                or row.get("prediction") not in {0, 1}
                or not isinstance(row.get("slice"), str)
                or not row.get("slice")
            ):
                valid_rows = False
                break
    if not valid_rows:
        codes.append("INVALID_TEST_ROW")

    test_metric: float | None = None
    slice_pass = False
    if valid_rows and valid_floors:
        correct = sum(row["label"] == row["prediction"] for row in rows)
        test_metric = round(correct / len(rows), 12)
        if test_metric < float(metric_floor):
            codes.append("AGGREGATE_FLOOR")

        slice_pass = True
        for name, floor in required_slices.items():
            selected_rows = [row for row in rows if row["slice"] == name]
            if not selected_rows:
                codes.append(f"MISSING_SLICE:{name}")
                slice_pass = False
                continue
            accuracy = round(
                sum(row["label"] == row["prediction"] for row in selected_rows) / len(selected_rows),
                12,
            )
            if accuracy < float(floor):
                codes.append(f"SLICE_FLOOR:{name}")
                slice_pass = False

    if valid_counts and bytes_processed > max_bytes:
        codes.append("BYTE_LIMIT")

    if "INVALID_INPUT" in codes or "INVALID_LINEAGE" in codes or "INVALID_TEST_ROW" in codes:
        slice_pass = False
    reason_codes = sorted_codes(codes)
    return 200, {
        "runId": run_id,
        "selectedTrialId": selected_trial,
        "datasetDigest": digest,
        "testMetric": test_metric,
        "criticalSlicePass": slice_pass,
        "decision": "admit" if not reason_codes else "reject",
        "bytesProcessed": bytes_processed,
        "reasonCodes": reason_codes,
    }


def handle_bqml(payload: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(payload, dict):
        return 400, {"error": "INVALID_INPUT"}
    phase = payload.get("phase")
    if phase == "select":
        return _select(payload)
    if phase == "evaluate":
        return _evaluate(payload)
    return 400, {"error": "INVALID_INPUT"}


def reset_runs_for_tests() -> None:
    with _LOCK:
        _RUNS.clear()
