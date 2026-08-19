from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

from ga8_utils import (
    compact_json,
    format_utc_millis,
    is_finite_number,
    is_safe_integer,
    parse_timestamp,
    sha256_hex,
    sorted_codes,
    utf8_key,
)


ROW_KEYS = {"id", "entity", "eventTime", "revision", "text"}
URI_RE = re.compile(r"\Ags://[^/]+/[^/]+\Z")
GENERATION_RE = re.compile(r"^[0-9]+$")
CRC32C_RE = re.compile(r"^[0-9a-f]{8}$")


def crc32c_hex(data: bytes) -> str:
    crc = 0xFFFFFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return f"{(crc ^ 0xFFFFFFFF) & 0xFFFFFFFF:08x}"


def canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    return " ".join(normalized.split())


def token_jaccard(left: str, right: str) -> float:
    def words(value: str) -> set[str]:
        result, current = set(), []
        for ch in value.lower():
            if unicodedata.category(ch)[0] in {"L", "N"}: current.append(ch)
            elif current: result.add("".join(current)); current = []
        if current: result.add("".join(current))
        return result
    left_tokens, right_tokens = words(left), words(right)
    union = left_tokens | right_tokens
    if not union:
        return 1.0
    return len(left_tokens & right_tokens) / len(union)


def _policy(body: dict[str, Any]) -> tuple[bool, Any, Any, float | None]:
    policy = body.get("policy")
    if not isinstance(policy, dict):
        return False, None, None, None
    minimum = parse_timestamp(policy.get("minTime"))
    maximum = parse_timestamp(policy.get("maxTime"))
    threshold = policy.get("contaminationThreshold")
    valid_threshold = is_finite_number(threshold) and 0 <= float(threshold) <= 1
    valid = minimum is not None and maximum is not None and minimum <= maximum and valid_threshold
    return valid, minimum, maximum, float(threshold) if valid_threshold else None


def _object_reason_codes(item: Any) -> tuple[list[str], list[dict[str, Any]], Any]:
    obj = item if isinstance(item, dict) else {}
    codes: list[str] = []
    uri = obj.get("uri") if isinstance(obj.get("uri"), str) else None
    raw_uri = obj.get("uri")
    generation = obj.get("generation")
    fetched_generation = obj.get("fetchedGeneration")
    crc = obj.get("crc32c")
    content = obj.get("content")

    if not isinstance(raw_uri, str) or URI_RE.fullmatch(raw_uri) is None:
        codes.append("URI_INVALID")
    generation_valid = (
        isinstance(generation, str)
        and GENERATION_RE.fullmatch(generation) is not None
        and isinstance(fetched_generation, str)
        and GENERATION_RE.fullmatch(fetched_generation) is not None
    )
    if not generation_valid:
        codes.append("GENERATION_INVALID")
    if generation != fetched_generation:
        codes.append("GENERATION_MISMATCH")

    crc_valid = isinstance(crc, str) and CRC32C_RE.fullmatch(crc) is not None
    if not crc_valid:
        codes.append("CRC32C_INVALID")
    if isinstance(content, str) and crc_valid and crc32c_hex(content.encode("utf-8")) != crc:
        codes.append("CRC32C_MISMATCH")

    parsed_rows: list[dict[str, Any]] = []
    if not isinstance(content, str):
        codes.append("SCHEMA_INVALID")
    else:
        nonblank = [line for line in content.splitlines() if line.strip()]
        if not nonblank:
            codes.append("SCHEMA_INVALID")
        for line in nonblank:
            try:
                # Match standards-compliant JSON.parse: NaN and infinities are
                # syntax errors, not parsed values that later fail the schema.
                row = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        json.JSONDecodeError("invalid constant", value, 0)
                    ),
                )
            except (json.JSONDecodeError, UnicodeError):
                codes.append("JSONL_INVALID")
                continue
            if (
                not isinstance(row, dict)
                or set(row) != ROW_KEYS
                or not all(isinstance(row.get(name), str) for name in ("id", "entity", "eventTime", "text"))
                or not is_safe_integer(row.get("revision"))
                or parse_timestamp(row.get("eventTime")) is None
            ):
                codes.append("SCHEMA_INVALID")
                continue
            parsed_rows.append(row)

    if obj.get("schemaId") != "training-v1":
        codes.append("SCHEMA_INVALID")
    return sorted_codes(codes), parsed_rows, uri


def _sort_with_json_tie(items: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    def key(item: dict[str, Any]) -> tuple[bytes, bytes]:
        primary = item.get(field)
        first = primary.encode("utf-8") if isinstance(primary, str) else b""
        return first, compact_json(item).encode("utf-8")

    return sorted(items, key=key)


def build_corpus(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("policy"), dict) or not isinstance(payload.get("objects"), list):
        return None

    policy_valid, minimum, maximum, threshold = _policy(payload)
    rejected_objects: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for item in payload["objects"]:
        codes, rows, rejected_uri = _object_reason_codes(item)
        if codes:
            rejected_objects.append({"uri": rejected_uri, "reasonCodes": codes})
            continue

        assert isinstance(item, dict)
        lineage.append(
            {
                "uri": item["uri"],
                "generation": item["generation"],
                "crc32c": item["crc32c"],
                "schemaId": item["schemaId"],
            }
        )
        for row in rows:
            parsed_time = parse_timestamp(row["eventTime"])
            assert parsed_time is not None
            candidates.append(
                {
                    "id": row["id"],
                    "entity": canonical_text(row["entity"]),
                    "eventTime": format_utc_millis(parsed_time),
                    "revision": row["revision"],
                    "text": canonical_text(row["text"]),
                    "_time": parsed_time,
                }
            )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        key = (row["entity"], row["eventTime"], row["text"])
        grouped.setdefault(key, []).append(row)

    retained: list[dict[str, Any]] = []
    for rows in grouped.values():
        ranked = sorted(rows, key=lambda row: (-row["revision"], utf8_key(row["id"])))
        retained.append(ranked[0])
        for loser in ranked[1:]:
            rejected_rows.append({"id": loser["id"], "reasonCodes": ["DUPLICATE"]})

    eligible: list[dict[str, Any]] = []
    for row in retained:
        codes: list[str] = []
        if not policy_valid:
            codes.append("POLICY_INVALID")
        elif not (minimum <= row["_time"] <= maximum):
            codes.append("OUT_OF_WINDOW")
        if codes:
            rejected_rows.append({"id": row["id"], "reasonCodes": sorted_codes(codes)})
        else:
            eligible.append(row)

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in eligible:
        bucket = hashlib.sha256(row["entity"].encode("utf-8")).digest()[0] % 10
        name = "train" if bucket <= 5 else "validation" if bucket <= 7 else "test"
        split_rows[name].append(row)

    train_rows = split_rows["train"]
    if policy_valid and threshold is not None:
        for split_name in ("validation", "test"):
            clean: list[dict[str, Any]] = []
            for row in split_rows[split_name]:
                contaminated = any(token_jaccard(row["text"], train["text"]) >= threshold for train in train_rows)
                if contaminated:
                    rejected_rows.append({"id": row["id"], "reasonCodes": ["TRAIN_CONTAMINATION"]})
                else:
                    clean.append(row)
            split_rows[split_name] = clean

    output_splits: dict[str, list[dict[str, Any]]] = {}
    digests: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        public_rows = [
            {key: row[key] for key in ("id", "entity", "eventTime", "revision", "text")}
            for row in split_rows[split_name]
        ]
        public_rows.sort(key=lambda row: (utf8_key(row["id"]), compact_json(row).encode("utf-8")))
        output_splits[split_name] = public_rows
        artifact = b"".join((compact_json(row) + "\n").encode("utf-8") for row in public_rows)
        digests[split_name] = sha256_hex(artifact)

    for rejected in rejected_rows:
        rejected["reasonCodes"] = sorted_codes(rejected["reasonCodes"])

    return {
        "splits": output_splits,
        "rejectedObjects": _sort_with_json_tie(rejected_objects, "uri"),
        "rejectedRows": _sort_with_json_tie(rejected_rows, "id"),
        "digests": digests,
        "lineage": _sort_with_json_tie(lineage, "uri"),
    }
