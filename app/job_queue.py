"""Async job queue for long-running requests in Nvidia Model Bridge."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Represents an async job."""
    id: str
    status: JobStatus
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    request_data: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    progress: str = ""


class JobQueue:
    """In-memory job queue for async request processing."""

    def __init__(self, max_jobs: int = 1000, job_ttl: int = 3600) -> None:
        self._jobs: dict[str, Job] = {}
        self._max_jobs = max_jobs
        self._job_ttl = job_ttl
        self._lock = Lock()

    def create_job(self, request_data: dict[str, Any]) -> Job:
        """Create a new pending job."""
        self._cleanup_expired()
        job = Job(
            id=str(uuid.uuid4()),
            status=JobStatus.PENDING,
            created_at=time.time(),
            request_data=request_data,
        )
        with self._lock:
            if len(self._jobs) >= self._max_jobs:
                self._evict_oldest()
            self._jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        progress: str | None = None,
    ) -> Job | None:
        """Update a job's status and result."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if status is not None:
                job.status = status
                if status == JobStatus.RUNNING:
                    job.started_at = time.time()
                elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    job.completed_at = time.time()
            if result is not None:
                job.result = result
            if error is not None:
                job.error = error
            if progress is not None:
                job.progress = progress
            return job

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List recent jobs."""
        with self._lock:
            jobs = sorted(
                self._jobs.values(), key=lambda j: j.created_at, reverse=True
            )[:limit]
        return [self._job_to_dict(job) for job in jobs]

    def _job_to_dict(self, job: Job) -> dict[str, Any]:
        elapsed = None
        if job.started_at:
            end = job.completed_at or time.time()
            elapsed = round(end - job.started_at, 4)
        return {
            "id": job.id,
            "status": job.status.value,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "elapsed_seconds": elapsed,
            "progress": job.progress,
            "has_result": job.result is not None,
            "error": job.error,
        }

    def _cleanup_expired(self) -> None:
        now = time.time()
        with self._lock:
            expired = [
                job_id
                for job_id, job in self._jobs.items()
                if now - job.created_at > self._job_ttl
                and job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
            ]
            for job_id in expired:
                del self._jobs[job_id]

    def _evict_oldest(self) -> None:
        completed = [
            (job_id, job)
            for job_id, job in self._jobs.items()
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
        ]
        if completed:
            oldest_id = min(completed, key=lambda x: x[1].created_at)[0]
            del self._jobs[oldest_id]


# Module-level singleton
_job_queue: JobQueue | None = None


def get_job_queue(max_jobs: int = 1000, job_ttl: int = 3600) -> JobQueue:
    """Get or create the job queue singleton."""
    global _job_queue
    if _job_queue is None:
        _job_queue = JobQueue(max_jobs=max_jobs, job_ttl=job_ttl)
    return _job_queue
