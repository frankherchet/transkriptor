from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from .chunks import ChunkSpec


class MediaToolError(RuntimeError):
    pass


def ensure_media_tools_available() -> None:
    missing = [name for name in ("ffmpeg", "ffprobe") if shutil.which(name) is None]
    if missing:
        joined = ", ".join(missing)
        raise MediaToolError(f"missing required media tool(s): {joined}")


def probe_duration_seconds(path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None

    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, check=False, text=True)
    if completed.returncode != 0:
        raise MediaToolError(completed.stderr.strip() or "ffprobe failed")

    try:
        data = json.loads(completed.stdout)
        duration = data.get("format", {}).get("duration")
        return float(duration) if duration is not None else None
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MediaToolError("could not parse ffprobe duration output") from exc


def extract_chunk(source: Path, destination: Path, spec: ChunkSpec) -> None:
    ensure_media_tools_available()

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{spec.start:.3f}",
        "-i",
        str(source),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
    ]
    if spec.end is not None:
        command.extend(["-t", f"{spec.end - spec.start:.3f}"])
    command.append(str(destination))

    completed = subprocess.run(command, capture_output=True, check=False, text=True)
    if completed.returncode != 0:
        raise MediaToolError(completed.stderr.strip() or "ffmpeg failed")
