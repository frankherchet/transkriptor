from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .asr import DEFAULT_MODEL_PATH, DEFAULT_QUANTIZATION, SUPPORTED_QUANTIZATIONS
from .service import ChunkProgress, TranscriptionOptions, TranscriptionService


def _read_hotwords_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_context_file(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.read_text(encoding="utf-8").strip()


def _combine_context(inline_context: str | None, file_context: str | None) -> str | None:
    parts = [
        value.strip()
        for value in (inline_context, file_context)
        if value is not None and value.strip()
    ]
    return "\n\n".join(parts) if parts else None


def _partial_output_path(output: Path) -> Path:
    return output.with_name(output.name + ".partial")


def _format_time(seconds: float | None) -> str:
    if seconds is None:
        return "end"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _write_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transkriptor",
        description="Transcribe an MP4 file to normalized JSON with VibeVoice-ASR.",
    )
    parser.add_argument("input", type=Path, help="Input MP4/audio/video file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output JSON file")
    parser.add_argument("--chunks", help="Absolute chunk markers, e.g. 100s,23m,59m")
    parser.add_argument(
        "--hotword",
        action="append",
        default=[],
        help="Hotword/context term. Can be passed multiple times.",
    )
    parser.add_argument("--hotwords-file", type=Path, help="Text file with one hotword per line")
    parser.add_argument(
        "--context",
        help="Free-form video metadata/background information for VibeVoice-ASR.",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        help="UTF-8 text file with video metadata/background information.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu", "mps", "xpu"))
    parser.add_argument(
        "--quantization",
        default=DEFAULT_QUANTIZATION,
        choices=SUPPORTED_QUANTIZATIONS,
        help="Model quantization. Use 4bit or 4bit-nf4 for 16 GB VRAM.",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    hotwords = [*args.hotword, *_read_hotwords_file(args.hotwords_file)]
    context = _combine_context(args.context, _read_context_file(args.context_file))
    options = TranscriptionOptions(
        model_path=args.model_path,
        device=args.device,
        quantization=args.quantization,
        chunk_markers=args.chunks,
        hotwords=hotwords,
        context=context,
    )
    partial_output = _partial_output_path(args.output)

    def on_chunk_start(progress: ChunkProgress) -> None:
        print(
            "Transcribing chunk "
            f"{progress.chunk_index + 1}/{progress.total_chunks}: "
            f"{_format_time(progress.start)}-{_format_time(progress.end)}",
            file=sys.stderr,
            flush=True,
        )

    def on_chunk_complete(progress: ChunkProgress) -> None:
        if progress.result is None:
            return
        _write_json(partial_output, progress.result)
        print(
            "Finished chunk "
            f"{progress.chunk_index + 1}/{progress.total_chunks}; "
            f"wrote partial result to {partial_output}",
            file=sys.stderr,
            flush=True,
        )

    result = TranscriptionService().transcribe_file(
        args.input,
        options,
        on_chunk_start=on_chunk_start,
        on_chunk_complete=on_chunk_complete,
    )
    _write_json(args.output, result)
    if partial_output.exists():
        partial_output.unlink()
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
