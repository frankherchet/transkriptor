from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .asr import ASRBackend, DEFAULT_MODEL_PATH, VibeVoiceASRBackend
from .chunks import (
    MAX_UNCHUNKED_SECONDS,
    build_chunk_specs,
    parse_chunk_markers,
    source_name,
    validate_markers_with_duration,
)
from .media import extract_chunk, probe_duration_seconds
from .normalizer import extract_raw_segments, normalize_segments


@dataclass(frozen=True)
class TranscriptionOptions:
    model_path: str = DEFAULT_MODEL_PATH
    device: str = "auto"
    chunk_markers: str | None = None
    hotwords: list[str] = field(default_factory=list)


class TranscriptionService:
    def __init__(self, backend: ASRBackend | None = None) -> None:
        self._backend = backend

    def transcribe_file(
        self,
        source: Path,
        options: TranscriptionOptions | None = None,
    ) -> dict[str, Any]:
        options = options or TranscriptionOptions()
        source = source.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        if not source.is_file():
            raise ValueError(f"source is not a file: {source}")

        markers = parse_chunk_markers(options.chunk_markers)
        duration = probe_duration_seconds(source)
        validate_markers_with_duration(markers, duration)

        warnings: list[str] = []
        if not markers and duration is not None and duration > MAX_UNCHUNKED_SECONDS:
            warnings.append(
                "media is longer than 60 minutes and no chunk markers were provided"
            )

        backend = self._backend or VibeVoiceASRBackend(
            model_path=options.model_path,
            device=options.device,
        )
        chunks = build_chunk_specs(markers)
        segments: list[dict[str, Any]] = []

        with tempfile.TemporaryDirectory(prefix="transkriptor-") as temp_dir:
            temp_path = Path(temp_dir)
            for spec in chunks:
                media_path = source
                if markers:
                    media_path = temp_path / f"chunk-{spec.index:04d}.wav"
                    extract_chunk(source, media_path, spec)

                raw_result = backend.transcribe(media_path, hotwords=options.hotwords)
                raw_segments = extract_raw_segments(raw_result)
                normalized = normalize_segments(
                    raw_segments,
                    chunk_index=spec.index,
                    offset_seconds=spec.start,
                    first_id=len(segments) + 1,
                )
                segments.extend(normalized)

        return {
            "metadata": {
                "source_file": source_name(source),
                "model": backend.model_path,
                "duration_seconds": duration,
                "chunking": {
                    "enabled": bool(markers),
                    "markers_seconds": markers,
                },
                "hotwords": options.hotwords,
                "warnings": warnings,
            },
            "segments": segments,
        }
