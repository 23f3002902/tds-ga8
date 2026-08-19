from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any


SAFE_INTEGER_MAX = 9_007_199_254_740_991
TIMESTAMP_RE = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})"
    r"(?:\.(\d{1,3}))?(Z|[+-]\d{2}:\d{2})$"
)


def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


def compact_json(value: Any, *, sort_keys: bool = False) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=sort_keys,
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_safe_integer(value: Any, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    lower = 1 if positive else 0
    return lower <= value <= SAFE_INTEGER_MAX


def is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    match = TIMESTAMP_RE.fullmatch(value)
    if match is None:
        return None

    year, month, day, hour, minute, second = map(int, match.group(1, 2, 3, 4, 5, 6))
    fraction = match.group(7) or ""
    zone = match.group(8)

    if second > 59:
        return None
    if zone != "Z":
        offset_hour = int(zone[1:3])
        offset_minute = int(zone[4:6])
        if offset_hour > 14 or offset_minute > 59:
            return None
        if offset_hour == 14 and offset_minute != 0:
            return None

    iso_value = value[:-1] + "+00:00" if zone == "Z" else value
    try:
        parsed = datetime.fromisoformat(iso_value)
    except ValueError:
        return None

    if parsed.year != year or parsed.month != month or parsed.day != day:
        return None
    if parsed.hour != hour or parsed.minute != minute or parsed.second != second:
        return None
    if fraction and parsed.microsecond != int(fraction.ljust(6, "0")):
        return None
    return parsed.astimezone(timezone.utc)


def format_utc_millis(value: datetime) -> str:
    utc = value.astimezone(timezone.utc)
    millis = utc.microsecond // 1_000
    return utc.strftime("%Y-%m-%dT%H:%M:%S") + f".{millis:03d}Z"


def sorted_codes(codes: list[str] | set[str]) -> list[str]:
    return sorted(set(codes), key=utf8_key)
