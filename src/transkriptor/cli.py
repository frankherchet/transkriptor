from __future__ import annotations

import argparse
import json
from pathlib import Path

from .asr import DEFAULT_MODEL_PATH
from .service import TranscriptionOptions, TranscriptionService


def _read_hotwords_file(path: Path | None) -> list[str]:
    if path is None:
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu", "mps", "xpu"))
    return parser


def run(args: argparse.Namespace) -> dict:
    hotwords = [*args.hotword, *_read_hotwords_file(args.hotwords_file)]
    options = TranscriptionOptions(
        model_path=args.model_path,
        device=args.device,
        chunk_markers=args.chunks,
        hotwords=hotwords,
    )
    result = TranscriptionService().transcribe_file(args.input, options)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
