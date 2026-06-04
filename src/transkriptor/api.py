from __future__ import annotations

import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .asr import DEFAULT_MODEL_PATH
from .service import TranscriptionOptions, TranscriptionService


JOB_ROOT = Path(".transkriptor_jobs")
UPLOAD_ROOT = JOB_ROOT / "uploads"
RESULT_ROOT = JOB_ROOT / "results"
MAX_WORKERS = 1

app = FastAPI(title="transkriptor")
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    created_at: datetime
    updated_at: datetime
    source_file: str
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class CreateJobResponse(BaseModel):
    job_id: str
    status: Literal["queued"]
    status_url: str
    result_url: str


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _status_path(job_id: str) -> Path:
    return JOB_ROOT / f"{job_id}.json"


def _result_path(job_id: str) -> Path:
    return RESULT_ROOT / f"{job_id}.json"


def _read_status(job_id: str) -> JobStatus:
    path = _status_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatus.model_validate_json(path.read_text(encoding="utf-8"))


def _write_status(status: JobStatus) -> None:
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    _status_path(status.job_id).write_text(
        status.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _update_status(job_id: str, **changes: object) -> JobStatus:
    current = _read_status(job_id)
    updated = current.model_copy(update={**changes, "updated_at": _utcnow()})
    _write_status(updated)
    return updated


def _parse_hotwords(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _run_job(
    *,
    job_id: str,
    source: Path,
    options: TranscriptionOptions,
) -> None:
    try:
        _update_status(job_id, status="running")
        result = TranscriptionService().transcribe_file(source, options)
        RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        _result_path(job_id).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        warnings = result.get("metadata", {}).get("warnings", [])
        _update_status(job_id, status="succeeded", warnings=warnings)
    except Exception as exc:
        _update_status(job_id, status="failed", error=str(exc))


def schedule_job(*, job_id: str, source: Path, options: TranscriptionOptions) -> None:
    executor.submit(_run_job, job_id=job_id, source=source, options=options)


@app.post("/jobs", response_model=CreateJobResponse, status_code=202)
async def create_job(
    file: Annotated[UploadFile, File()],
    chunk_markers: Annotated[str | None, Form()] = None,
    hotwords: Annotated[str | None, Form()] = None,
    model_path: Annotated[str, Form()] = DEFAULT_MODEL_PATH,
    device: Annotated[str, Form()] = "auto",
) -> CreateJobResponse:
    if file.filename is None or not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="only .mp4 uploads are supported")

    job_id = uuid.uuid4().hex
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_ROOT / f"{job_id}.mp4"
    with destination.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    status = JobStatus(
        job_id=job_id,
        status="queued",
        created_at=_utcnow(),
        updated_at=_utcnow(),
        source_file=file.filename,
    )
    _write_status(status)

    options = TranscriptionOptions(
        model_path=model_path,
        device=device,
        chunk_markers=chunk_markers,
        hotwords=_parse_hotwords(hotwords),
    )
    schedule_job(
        job_id=job_id,
        source=destination,
        options=options,
    )
    return CreateJobResponse(
        job_id=job_id,
        status="queued",
        status_url=f"/jobs/{job_id}",
        result_url=f"/jobs/{job_id}/result",
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    return _read_status(job_id)


@app.get("/jobs/{job_id}/result")
async def get_result(job_id: str) -> JSONResponse:
    status = _read_status(job_id)
    if status.status != "succeeded":
        raise HTTPException(status_code=409, detail=f"job status is {status.status}")

    path = _result_path(job_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="result not found")
    return JSONResponse(content=json.loads(path.read_text(encoding="utf-8")))
