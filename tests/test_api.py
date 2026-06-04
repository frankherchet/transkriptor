from __future__ import annotations

import json
from pathlib import Path

import httpx

import transkriptor.api as api_module


def test_create_job_and_fetch_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(api_module, "JOB_ROOT", tmp_path / "jobs")
    monkeypatch.setattr(api_module, "UPLOAD_ROOT", tmp_path / "jobs" / "uploads")
    monkeypatch.setattr(api_module, "RESULT_ROOT", tmp_path / "jobs" / "results")

    def fake_schedule_job(*, job_id, source, options) -> None:
        api_module.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        result = {
            "metadata": {
                "warnings": [],
                "source_file": source.name,
                "model": options.model_path,
                "chunking": {"enabled": False, "markers_seconds": []},
                "hotwords": options.hotwords,
            },
            "segments": [],
        }
        api_module._result_path(job_id).write_text(json.dumps(result), encoding="utf-8")
        api_module._update_status(job_id, status="succeeded", warnings=[])

    monkeypatch.setattr(api_module, "schedule_job", fake_schedule_job)

    transport = httpx.ASGITransport(app=api_module.app)

    async def run_requests() -> None:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/jobs",
                files={"file": ("input.mp4", b"fake", "video/mp4")},
                data={"hotwords": "Ada,Grace"},
            )

            assert response.status_code == 202
            job_id = response.json()["job_id"]
            status = await client.get(f"/jobs/{job_id}")
            result = await client.get(f"/jobs/{job_id}/result")

        assert status.json()["status"] == "succeeded"
        assert result.json()["metadata"]["hotwords"] == [
            "Ada",
            "Grace",
        ]

    import asyncio

    asyncio.run(run_requests())
