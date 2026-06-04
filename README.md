# transkriptor

MP4-to-JSON transcription with Microsoft VibeVoice-ASR.

The project exposes the same transcription core through a CLI and a FastAPI job API. It converts an uploaded or local `.mp4` file into normalized JSON segments with speaker labels, timestamps, and text.

VibeVoice-ASR is imported lazily, so tests and API startup do not require the large model dependencies until a transcription job actually runs.

## Requirements

- Python `>=3.11,<3.13`
- `ffmpeg` and `ffprobe` on `PATH`
- A runtime supported by VibeVoice-ASR; CUDA is recommended for practical long-form inference
- `uv`

If your system Python is newer than the supported range, install a compatible interpreter with uv:

```bash
uv python install 3.12
```

Install the app, ASR dependencies, and test dependencies:

```bash
uv sync --python 3.12 --extra asr --extra test
```

For development or CI without downloading the VibeVoice/Torch stack:

```bash
uv sync --python 3.12 --extra test
uv run pytest -q
```

## CLI

Basic transcription:

```bash
uv run transkriptor input.mp4 -o output.json
```

Optional chunk markers are absolute cut points in the source timeline. Supported formats include seconds, minutes, hours, and clock values:

```bash
uv run transkriptor input.mp4 -o output.json --chunks 100s,23m,59m
uv run transkriptor input.mp4 -o output.json --chunks 01:00,00:23:00
```

Hotwords can be passed directly or through a text file:

```bash
uv run transkriptor input.mp4 -o output.json \
  --hotword "Ada Lovelace" \
  --hotwords-file hotwords.txt
```

Device and model can be overridden:

```bash
uv run transkriptor input.mp4 -o output.json \
  --model-path microsoft/VibeVoice-ASR \
  --device cuda
```

If no chunk markers are provided, the file is processed as one VibeVoice-ASR input. Files longer than 60 minutes produce a warning in the JSON metadata. When chunking is enabled, speaker IDs are namespaced per chunk, for example `chunk2:SPEAKER_00`, because speaker identity is not stitched across chunks.

## API

Run the app:

```bash
uv run fastapi dev --host 127.0.0.1 --port 8000
```

Create a job:

```bash
curl -F "file=@input.mp4" \
  -F "chunk_markers=100s,23m,59m" \
  -F "hotwords=Ada Lovelace" \
  http://127.0.0.1:8000/jobs
```

Poll and fetch results:

```bash
curl http://127.0.0.1:8000/jobs/{job_id}
curl http://127.0.0.1:8000/jobs/{job_id}/result
```

Job state and uploaded files are stored under `.transkriptor_jobs/`. This is local development storage, not a production job database.

## JSON output

```json
{
  "metadata": {
    "source_file": "input.mp4",
    "model": "microsoft/VibeVoice-ASR",
    "duration_seconds": 12.3,
    "chunking": {
      "enabled": true,
      "markers_seconds": [100.0, 1380.0, 3540.0]
    },
    "hotwords": ["Ada Lovelace"],
    "warnings": []
  },
  "segments": [
    {
      "id": 1,
      "speaker": "chunk0:SPEAKER_00",
      "speaker_local": "SPEAKER_00",
      "chunk_index": 0,
      "start": 0.0,
      "end": 4.2,
      "text": "..."
    }
  ]
}
```

## Development

Run the test suite:

```bash
uv run --extra test pytest -q
```

The tests use a fake ASR backend and do not download or run the VibeVoice model.

## Runtime notes

- `ffprobe` is used to read source duration.
- `ffmpeg` is used to extract audio chunks when `--chunks` or `chunk_markers` is provided.
- The VibeVoice-ASR backend follows Microsoft demo usage: processor input, model generation, decode, then `post_process_transcription`.
- The API uses one background worker by default so a single model instance does not receive concurrent long-running jobs in one process.
