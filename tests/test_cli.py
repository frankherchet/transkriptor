from __future__ import annotations

import json
from pathlib import Path

from transkriptor import cli
from transkriptor.service import ChunkProgress


class FakeService:
    def transcribe_file(
        self,
        source,
        options,
        on_chunk_start=None,
        on_chunk_complete=None,
    ):
        partial_result = {
            "metadata": {"source_file": source.name},
            "segments": [{"id": 1, "text": "partial"}],
        }
        final_result = {
            "metadata": {"source_file": source.name},
            "segments": [{"id": 1, "text": "final"}],
        }
        if on_chunk_start is not None:
            on_chunk_start(
                ChunkProgress(
                    chunk_index=0,
                    total_chunks=1,
                    start=0,
                    end=None,
                    segments_done=0,
                )
            )
        if on_chunk_complete is not None:
            on_chunk_complete(
                ChunkProgress(
                    chunk_index=0,
                    total_chunks=1,
                    start=0,
                    end=None,
                    segments_done=1,
                    result=partial_result,
                )
            )
        return final_result


def test_cli_writes_partial_during_run_and_removes_it_after_success(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.json"
    input_path.write_bytes(b"fake")
    monkeypatch.setattr(cli, "TranscriptionService", lambda: FakeService())

    args = cli.build_parser().parse_args([str(input_path), "-o", str(output_path)])
    result = cli.run(args)

    assert result["segments"][0]["text"] == "final"
    assert json.loads(output_path.read_text())["segments"][0]["text"] == "final"
    assert not (tmp_path / "output.json.partial").exists()
    stderr = capsys.readouterr().err
    assert "Transcribing chunk 1/1: 00:00:00-end" in stderr
    assert "wrote partial result" in stderr
