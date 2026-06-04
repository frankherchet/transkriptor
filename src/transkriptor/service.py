from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .asr import (
    ASRBackend,
    DEFAULT_MODEL_PATH,
    DEFAULT_QUANTIZATION,
    VibeVoiceASRBackend,
)
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
    quantization: str = DEFAULT_QUANTIZATION
    chunk_markers: str | None = None
    hotwords: list[str] = field(default_factory=list)
    context: str | None = None


@dataclass(frozen=True)
class ChunkProgress:
    chunk_index: int
    total_chunks: int
    start: float
    end: float | None
    segments_done: int
    result: dict[str, Any] | None = None


class TranscriptionService:
    def __init__(self, backend: ASRBackend | None = None) -> None:
        self._backend = backend

    def transcribe_file(
        self,
        source: Path,
        options: TranscriptionOptions | None = None,
        on_chunk_start: Callable[[ChunkProgress], None] | None = None,
        on_chunk_complete: Callable[[ChunkProgress], None] | None = None,
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
            quantization=options.quantization,
        )
        chunks = build_chunk_specs(markers)
        segments: list[dict[str, Any]] = []
        result = self._build_result(
            source=source,
            backend=backend,
            options=options,
            duration=duration,
            markers=markers,
            warnings=warnings,
            segments=segments,
        )

        with tempfile.TemporaryDirectory(prefix="transkriptor-") as temp_dir:
            temp_path = Path(temp_dir)
            for spec in chunks:
                if on_chunk_start is not None:
                    on_chunk_start(
                        ChunkProgress(
                            chunk_index=spec.index,
                            total_chunks=len(chunks),
                            start=spec.start,
                            end=spec.end,
                            segments_done=len(segments),
                        )
                    )

                media_path = source
                if markers:
                    media_path = temp_path / f"chunk-{spec.index:04d}.wav"
                    extract_chunk(source, media_path, spec)

                raw_result = backend.transcribe(
                    media_path,
                    hotwords=options.hotwords,
                    context=options.context,
                )
                raw_segments = extract_raw_segments(raw_result)
                normalized = normalize_segments(
                    raw_segments,
                    chunk_index=spec.index,
                    offset_seconds=spec.start,
                    first_id=len(segments) + 1,
                )
                segments.extend(normalized)
                result = self._build_result(
                    source=source,
                    backend=backend,
                    options=options,
                    duration=duration,
                    markers=markers,
                    warnings=warnings,
                    segments=segments,
                )
                if on_chunk_complete is not None:
                    on_chunk_complete(
                        ChunkProgress(
                            chunk_index=spec.index,
                            total_chunks=len(chunks),
                            start=spec.start,
                            end=spec.end,
                            segments_done=len(segments),
                            result=result,
                        )
                    )

        return result

    def _build_result(
        self,
        *,
        source: Path,
        backend: ASRBackend,
        options: TranscriptionOptions,
        duration: float | None,
        markers: list[float],
        warnings: list[str],
        segments: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "metadata": {
                "source_file": source_name(source),
                "model": backend.model_path,
                "quantization": options.quantization,
                "duration_seconds": duration,
                "chunking": {
                    "enabled": bool(markers),
                    "markers_seconds": markers,
                },
                "hotwords": options.hotwords,
                "context_provided": bool(options.context and options.context.strip()),
                "warnings": warnings,
            },
            "segments": list(segments),
        }
