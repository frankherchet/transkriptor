from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


MAX_UNCHUNKED_SECONDS = 60 * 60


@dataclass(frozen=True)
class ChunkSpec:
    index: int
    start: float
    end: float | None


_HHMMSS_RE = re.compile(r"^\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?$")
_UNIT_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>s|m|h)?$")


def parse_time_marker(value: str) -> float:
    marker = value.strip().lower()
    if not marker:
        raise ValueError("empty chunk marker")

    if _HHMMSS_RE.match(marker):
        parts = [float(part) for part in marker.split(":")]
        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds

    match = _UNIT_RE.match(marker)
    if not match:
        raise ValueError(f"invalid chunk marker: {value!r}")

    seconds = float(match.group("value"))
    unit = match.group("unit") or "s"
    if unit == "m":
        seconds *= 60
    elif unit == "h":
        seconds *= 3600
    return seconds


def parse_chunk_markers(value: str | None) -> list[float]:
    if value is None or not value.strip():
        return []

    markers = [parse_time_marker(part) for part in value.split(",")]
    if any(marker <= 0 for marker in markers):
        raise ValueError("chunk markers must be greater than 0 seconds")
    if markers != sorted(markers):
        raise ValueError("chunk markers must be sorted in ascending order")
    if len(set(markers)) != len(markers):
        raise ValueError("chunk markers must not contain duplicates")
    return markers


def build_chunk_specs(markers: list[float]) -> list[ChunkSpec]:
    starts = [0.0, *markers]
    ends: list[float | None] = [*markers, None]
    return [
        ChunkSpec(index=index, start=start, end=ends[index])
        for index, start in enumerate(starts)
    ]


def validate_markers_with_duration(markers: list[float], duration_seconds: float | None) -> None:
    if duration_seconds is None:
        return
    if markers and markers[-1] >= duration_seconds:
        raise ValueError("last chunk marker must be before the media duration")


def source_name(path: Path) -> str:
    return path.name
