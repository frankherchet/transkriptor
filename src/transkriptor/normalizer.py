from __future__ import annotations

from typing import Any


def _first_present(data: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def normalize_segments(
    raw_segments: list[dict[str, Any]],
    *,
    chunk_index: int,
    offset_seconds: float,
    first_id: int = 1,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    for offset, raw in enumerate(raw_segments):
        local_speaker = str(
            _first_present(raw, ("speaker", "speaker_id", "role"), "UNKNOWN")
        )
        local_start = float(_first_present(raw, ("start", "start_time", "begin"), 0.0))
        local_end = float(_first_present(raw, ("end", "end_time", "finish"), local_start))
        text = str(_first_present(raw, ("text", "transcript", "content"), "")).strip()

        normalized.append(
            {
                "id": first_id + offset,
                "speaker": f"chunk{chunk_index}:{local_speaker}",
                "speaker_local": local_speaker,
                "chunk_index": chunk_index,
                "start": round(offset_seconds + local_start, 3),
                "end": round(offset_seconds + local_end, 3),
                "text": text,
            }
        )

    return normalized


def extract_raw_segments(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        return [segment for segment in result if isinstance(segment, dict)]

    if isinstance(result, dict):
        for key in ("segments", "transcription", "results"):
            value = result.get(key)
            if isinstance(value, list):
                return [segment for segment in value if isinstance(segment, dict)]

    raise ValueError("VibeVoice-ASR result did not contain a segment list")
