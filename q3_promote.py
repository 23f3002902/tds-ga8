from __future__ import annotations

import re
from typing import Any

from ga8_utils import is_finite_number, is_safe_integer, parse_timestamp, sorted_codes


CANONICAL_VERSION = re.compile(r"^[1-9][0-9]*$")
SAFE_MAX = 9_007_199_254_740_991


def _canonical_version(value: Any) -> bool:
    return (
        isinstance(value, str)
        and CANONICAL_VERSION.fullmatch(value) is not None
        and int(value) <= SAFE_MAX
    )


def _display_key(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _policy_valid(policy: dict[str, Any]) -> bool:
    required = policy.get("requiredSlices")
    return (
        isinstance(policy.get("datasetDigest"), str)
        and policy["datasetDigest"] != ""
        and isinstance(policy.get("schemaDigest"), str)
        and policy["schemaDigest"] != ""
        and is_safe_integer(policy.get("maxAgeSeconds"))
        and is_finite_number(policy.get("accuracyFloor"))
        and 0 <= float(policy["accuracyFloor"]) <= 1
        and is_finite_number(policy.get("maxLatencyMs"))
        and float(policy["maxLatencyMs"]) >= 0
        and is_finite_number(policy.get("maxSizeBytes"))
        and float(policy["maxSizeBytes"]) >= 0
        and is_finite_number(policy.get("minImprovement"))
        and float(policy["minImprovement"]) >= 0
        and isinstance(required, dict)
        and all(
            isinstance(name, str)
            and name != ""
            and is_finite_number(floor)
            and 0 <= float(floor) <= 1
            for name, floor in required.items()
        )
    )


def handle_promote(payload: Any) -> tuple[int, dict[str, Any]]:
    # The contract reserves transport-level rejection for these three shapes.
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("policy"), dict)
        or not isinstance(payload.get("versions"), list)
        or not isinstance(payload.get("championVersion"), str)
    ):
        return 400, {"error": "INVALID_INPUT"}

    policy = payload["policy"]
    versions = payload["versions"]
    champion = payload["championVersion"]
    as_of = parse_timestamp(payload.get("asOf"))
    policy_ok = _policy_valid(policy)
    required_slices = policy.get("requiredSlices") if isinstance(policy.get("requiredSlices"), dict) else {}

    # Count duplicate version strings without attempting to hash malformed JSON values.
    occurrences: dict[str, int] = {}
    for item in versions:
        value = item.get("version") if isinstance(item, dict) else None
        marker = _display_key(value)
        occurrences[marker] = occurrences.get(marker, 0) + 1

    failed_gates: dict[str, list[str]] = {}
    eligible: list[dict[str, Any]] = []

    for item in versions:
        version = item.get("version") if isinstance(item, dict) else None
        gate_key = _display_key(version)
        codes: list[str] = []

        if not _canonical_version(version):
            codes.append("INVALID_VERSION")
        if occurrences.get(gate_key, 0) > 1:
            codes.append("DUPLICATE_VERSION")
        if not policy_ok:
            codes.append("INVALID_POLICY")
        if as_of is None:
            codes.append("INVALID_TIMESTAMP")

        evaluation = item.get("evaluation") if isinstance(item, dict) else None
        if not isinstance(evaluation, dict):
            codes.append("MISSING_EVALUATION")
        else:
            created_at = parse_timestamp(evaluation.get("createdAt"))
            if created_at is None:
                codes.append("INVALID_TIMESTAMP")
            elif as_of is not None and policy_ok:
                age_seconds = (as_of - created_at).total_seconds()
                if age_seconds < 0:
                    codes.append("FUTURE_EVALUATION")
                elif age_seconds > policy["maxAgeSeconds"]:
                    codes.append("STALE_EVALUATION")

            artifact_digest = item.get("artifactDigest") if isinstance(item, dict) else None
            if evaluation.get("artifactDigest") != artifact_digest:
                codes.append("ARTIFACT_MISMATCH")
            if policy_ok:
                if evaluation.get("datasetDigest") != policy["datasetDigest"]:
                    codes.append("DATASET_MISMATCH")
                if evaluation.get("schemaDigest") != policy["schemaDigest"]:
                    codes.append("SCHEMA_MISMATCH")

            accuracy = evaluation.get("accuracy")
            latency = evaluation.get("latencyMs")
            size = evaluation.get("sizeBytes")
            if not all(is_finite_number(value) for value in (accuracy, latency, size)):
                codes.append("NON_FINITE")
            if is_finite_number(accuracy) and not 0 <= float(accuracy) <= 1:
                codes.append("METRIC_RANGE")
            if is_finite_number(latency) and float(latency) < 0:
                codes.append("METRIC_RANGE")
            if is_finite_number(size) and float(size) < 0:
                codes.append("METRIC_RANGE")

            if policy_ok:
                if is_finite_number(accuracy) and float(accuracy) < policy["accuracyFloor"]:
                    codes.append("ACCURACY_FLOOR")
                if is_finite_number(latency) and float(latency) > policy["maxLatencyMs"]:
                    codes.append("LATENCY_LIMIT")
                if is_finite_number(size) and float(size) > policy["maxSizeBytes"]:
                    codes.append("SIZE_LIMIT")

            slices = evaluation.get("slices") if isinstance(evaluation.get("slices"), dict) else {}
            for name, floor in required_slices.items():
                if name not in slices:
                    codes.append(f"MISSING_SLICE:{name}")
                elif not is_finite_number(slices[name]) or not 0 <= float(slices[name]) <= 1:
                    codes.append(f"SLICE_RANGE:{name}")
                elif policy_ok and float(slices[name]) < floor:
                    codes.append(f"SLICE_FLOOR:{name}")

        codes = sorted_codes(codes)
        failed_gates[gate_key] = codes
        if not codes and isinstance(item, dict):
            eligible.append(item)

    eligible.sort(
        key=lambda item: (
            -float(item["evaluation"]["accuracy"]),
            float(item["evaluation"]["latencyMs"]),
            float(item["evaluation"]["sizeBytes"]),
            int(item["version"]),
        )
    )

    champion_item = next((item for item in eligible if item["version"] == champion), None)
    action = "block"
    selected_version = None
    alias_mutation = None
    evidence = None

    if champion_item is not None:
        winner = eligible[0]
        improvement = round(
            float(winner["evaluation"]["accuracy"])
            - float(champion_item["evaluation"]["accuracy"]),
            12,
        )
        if winner["version"] != champion and improvement >= float(policy["minImprovement"]):
            action = "promote"
            selected_version = winner["version"]
            alias_mutation = {"alias": "champion", "version": selected_version}
        else:
            action = "retain"
            selected_version = champion
        evidence = next(item["evaluation"] for item in eligible if item["version"] == selected_version)

    return 200, {
        "action": action,
        "championVersion": champion,
        "selectedVersion": selected_version,
        "eligibleVersions": [item["version"] for item in eligible],
        "failedGates": failed_gates,
        "aliasMutation": alias_mutation,
        "evidence": evidence,
    }
