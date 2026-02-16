"""
Jobs Router
Job status, download, and management endpoints
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from fastapi.responses import FileResponse

from api.models.responses import JobStatusResponse, ErrorResponse
from api.models.enums import JobStatus, JobType
from api.services.job_store import JobStore, get_job_store
from api.services.storage import StorageService, get_storage_service
from api.middleware.auth import get_api_key
from api.middleware.rate_limit import limiter
from api.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status",
    description="Get the current status of a job including progress and download URLs when complete.",
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
@limiter.limit(settings.rate_limit_default)
async def get_job_status(
    request: Request,
    job_id: str = Path(..., description="Job ID"),
    api_key: Dict[str, Any] = Depends(get_api_key),
    job_store: JobStore = Depends(get_job_store),
) -> JobStatusResponse:
    """
    Get the current status of a processing job.

    Returns:
    - `pending`: Job is queued
    - `processing`: Job is currently being processed
    - `completed`: Job finished successfully, download_urls available
    - `failed`: Job failed, error message available
    - `cancelled`: Job was cancelled
    """
    job_data = job_store.get_job(job_id)

    if not job_data:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found"
        )

    return JobStatusResponse(
        job_id=job_data["job_id"],
        status=JobStatus(job_data["status"]),
        job_type=JobType(job_data["job_type"]),
        created_at=datetime.fromisoformat(job_data["created_at"]),
        updated_at=datetime.fromisoformat(job_data["updated_at"]) if job_data.get("updated_at") else None,
        completed_at=datetime.fromisoformat(job_data["completed_at"]) if job_data.get("completed_at") else None,
        progress=job_data.get("progress"),
        message=job_data.get("message"),
        error=job_data.get("error"),
        download_urls=job_data.get("download_urls"),
        input_count=job_data.get("input_count"),
        output_count=job_data.get("output_count"),
    )


@router.get(
    "/{job_id}/download/{filename}",
    summary="Download job output",
    description="Download a specific output file from a completed job.",
    responses={
        404: {"model": ErrorResponse, "description": "Job or file not found"},
        400: {"model": ErrorResponse, "description": "Job not completed"},
    },
)
@limiter.limit(settings.rate_limit_default)
async def download_file(
    request: Request,
    job_id: str = Path(..., description="Job ID"),
    filename: str = Path(..., description="Output filename"),
    api_key: Dict[str, Any] = Depends(get_api_key),
    job_store: JobStore = Depends(get_job_store),
    storage: StorageService = Depends(get_storage_service),
) -> FileResponse:
    """
    Download an output file from a completed job.

    The filename must match one of the files listed in the job's download_urls.
    """
    # Check job exists
    job_data = job_store.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Check job is completed
    if job_data["status"] != JobStatus.COMPLETED.value:
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed (status: {job_data['status']})"
        )

    # Get file path
    file_path = storage.get_file_path(job_id, filename, is_output=True)
    if not file_path:
        raise HTTPException(
            status_code=404,
            detail=f"File {filename} not found for job {job_id}"
        )

    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".obj": "model/obj",
        ".ply": "application/x-ply",
    }
    media_type = media_types.get(suffix, "application/octet-stream")

    logger.info(f"Serving file {filename} for job {job_id}")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type,
    )


@router.delete(
    "/{job_id}",
    summary="Cancel and delete job",
    description="Cancel a pending/processing job and delete all associated files.",
    responses={
        404: {"model": ErrorResponse, "description": "Job not found"},
    },
)
@limiter.limit(settings.rate_limit_default)
async def delete_job(
    request: Request,
    job_id: str = Path(..., description="Job ID"),
    api_key: Dict[str, Any] = Depends(get_api_key),
    job_store: JobStore = Depends(get_job_store),
    storage: StorageService = Depends(get_storage_service),
) -> Dict[str, str]:
    """
    Cancel a job and delete all associated files.

    - If job is pending or processing, it will be cancelled
    - All uploaded and output files will be deleted
    - The job record will be removed
    """
    # Check job exists
    job_data = job_store.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Try to revoke Celery task if still running
    if job_data.get("celery_task_id"):
        try:
            from api.tasks.celery_app import celery_app
            celery_app.control.revoke(job_data["celery_task_id"], terminate=True)
            logger.info(f"Revoked Celery task {job_data['celery_task_id']} for job {job_id}")
        except Exception as e:
            logger.warning(f"Failed to revoke Celery task: {e}")

    # Mark as cancelled (if not already completed/failed)
    if job_data["status"] in (JobStatus.PENDING.value, JobStatus.PROCESSING.value):
        job_store.set_cancelled(job_id)

    # Clean up files
    storage.cleanup_job(job_id)

    # Delete job record
    job_store.delete_job(job_id)

    logger.info(f"Deleted job {job_id}")

    return {"message": f"Job {job_id} cancelled and deleted"}
