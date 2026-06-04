from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import transkriptor.service as service_module
from transkriptor.service import ChunkProgress, TranscriptionOptions, TranscriptionService


class FakeBackend:
    model_path = "fake-model"

    def __init__(self) -> None:
        self.calls: list[tuple[Path, list[str], str | None]] = []

    def transcribe(
        self,
        media_path: Path,
        *,
        hotwords: list[str],
        context: str | None,
    ) -> list[dict[str, Any]]:
        self.calls.append((media_path, hotwords, context))
        return [{"speaker": "SPEAKER_00", "start": 1.0, "end": 2.0, "text": "hello"}]


def test_transcribe_file_without_chunking_warns_for_long_media(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    monkeypatch.setattr(service_module, "probe_duration_seconds", lambda path: 3700.0)

    backend = FakeBackend()
    result = TranscriptionService(backend).transcribe_file(
        source,
        TranscriptionOptions(hotwords=["Ada"], context="Moderation: Markus Lanz."),
    )

    assert backend.calls == [(source.resolve(), ["Ada"], "Moderation: Markus Lanz.")]
    assert result["metadata"]["context_provided"] is True
    assert result["metadata"]["warnings"] == [
        "media is longer than 60 minutes and no chunk markers were provided"
    ]
    assert result["segments"][0]["speaker"] == "chunk0:SPEAKER_00"


def test_transcribe_file_with_chunking_offsets_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    monkeypatch.setattr(service_module, "probe_duration_seconds", lambda path: 30.0)

    def fake_extract(source_path: Path, destination: Path, spec: Any) -> None:
        destination.write_bytes(b"chunk")

    monkeypatch.setattr(service_module, "extract_chunk", fake_extract)

    backend = FakeBackend()
    result = TranscriptionService(backend).transcribe_file(
        source,
        TranscriptionOptions(chunk_markers="10s,20s"),
    )

    assert len(backend.calls) == 3
    assert [segment["start"] for segment in result["segments"]] == [1.0, 11.0, 21.0]
    assert [segment["id"] for segment in result["segments"]] == [1, 2, 3]
    assert result["metadata"]["chunking"]["markers_seconds"] == [10.0, 20.0]
    assert result["metadata"]["context_provided"] is False


def test_transcribe_file_reports_chunk_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"fake")
    monkeypatch.setattr(service_module, "probe_duration_seconds", lambda path: 30.0)
    monkeypatch.setattr(
        service_module,
        "extract_chunk",
        lambda source_path, destination, spec: destination.write_bytes(b"chunk"),
    )

    starts: list[ChunkProgress] = []
    completes: list[ChunkProgress] = []

    TranscriptionService(FakeBackend()).transcribe_file(
        source,
        TranscriptionOptions(chunk_markers="10s,20s"),
        on_chunk_start=starts.append,
        on_chunk_complete=completes.append,
    )

    assert [(item.chunk_index, item.total_chunks, item.start, item.end) for item in starts] == [
        (0, 3, 0.0, 10.0),
        (1, 3, 10.0, 20.0),
        (2, 3, 20.0, None),
    ]
    assert [item.segments_done for item in completes] == [1, 2, 3]
    assert completes[-1].result is not None
    assert len(completes[-1].result["segments"]) == 3
