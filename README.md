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

The CLI prints progress for each chunk to stderr:

```text
Transcribing chunk 1/4: 00:00:00-00:26:14
Finished chunk 1/4; wrote partial result to output.json.partial
```

After every completed chunk, a partial JSON result is written next to the final output as `output.json.partial`. The final `output.json` is written only after the full run succeeds, and the partial file is removed at the end. If the process is interrupted or runs out of memory, keep the `.partial` file as the latest completed chunk result.

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

Longer video metadata can be passed as ASR context. This is useful for names, roles, show metadata, and expected topics:

```bash
uv run transkriptor input.mp4 -o output.json \
  --context-file video_context.txt
```

Example `video_context.txt`:

```text
Moderation: Markus Lanz.

Teilnehmer:
Karl Lauterbach, SPD-Politiker
Marie-Agnes Strack-Zimmermann, FDP-Politikerin
Robin Alexander, Journalist
Bernd Raffelhueschen, Oekonom

Themen:
Sozialpolitik, Gesundheitspolitik, Rentenpolitik, AfD-Abgrenzungsdebatte,
FDP-Spitze, Kanzlertausch-Debatte.
```

Device and model can be overridden:

```bash
uv run transkriptor input.mp4 -o output.json \
  --model-path microsoft/VibeVoice-ASR \
  --device cuda
```

Quantization can reduce VRAM usage. Supported values are:

- `none`: default bf16/float32 loading
- `8bit`: BitsAndBytes 8-bit loading
- `4bit`: alias for `4bit-nf4`
- `4bit-nf4`: recommended 4-bit mode for 16 GB VRAM
- `4bit-fp4`: alternative 4-bit mode

For a 16 GB GPU, start with:

```bash
uv run transkriptor input.mp4 -o output.json \
  --device cuda \
  --quantization 4bit \
  --chunks 00:26:14,00:45:33,01:06:50 \
  --context-file video_context.txt
```

## Model cache

VibeVoice-ASR is downloaded through Hugging Face Hub. By default, model files are cached under:

```text
~/.cache/huggingface/hub
```

For the default model, expect files below:

```text
~/.cache/huggingface/hub/models--microsoft--VibeVoice-ASR
```

Check cache size:

```bash
du -sh ~/.cache/huggingface/hub/models--microsoft--VibeVoice-ASR
```

Move the Hugging Face cache by setting `HF_HOME`:

```bash
HF_HOME=/data/hf-cache uv run transkriptor input.mp4 \
  -o output.json \
  --quantization 4bit
```

Or set only the Hub cache:

```bash
HF_HUB_CACHE=/data/hf-cache/hub uv run transkriptor input.mp4 \
  -o output.json \
  --quantization 4bit
```

`--quantization 4bit` reduces GPU memory during model loading/inference, but it still downloads the original model safetensors first. Quantization happens locally while loading the model.

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
  -F "context=Moderation: Markus Lanz. Teilnehmer: Karl Lauterbach..." \
  -F "quantization=4bit" \
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
    "context_provided": true,
    "quantization": "4bit",
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
- `--context`, `--context-file`, and API `context` are passed to VibeVoice-ASR as `context_info` together with hotwords.
- `--quantization 4bit` uses Transformers `BitsAndBytesConfig` with NF4 and double quantization.
- The API uses one background worker by default so a single model instance does not receive concurrent long-running jobs in one process.
