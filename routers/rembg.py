"""
RemBG Router
Background removal endpoint
"""

import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, File, UploadFile, Depends, Form, HTTPException, Request

from api.models.responses import JobResponse
from api.models.enums import JobStatus, JobType
from api.services.job_store import JobStore, get_job_store
from api.services.storage import StorageService, get_storage_service
from api.middleware.auth import get_api_key
from api.middleware.rate_limit import limiter
from api.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rembg", tags=["Background Removal"])


@router.post(
    "",
    response_model=JobResponse,
    summary="Remove background from images",
    description="Upload 1-10 images to have their backgrounds removed. Returns a job ID for tracking.",
)
@limiter.limit(settings.rate_limit_rembg)
async def remove_background(
    request: Request,
    files: List[UploadFile] = File(..., description="Image files (1-10)"),
    model: str = Form(default="u2net", description="RemBG model (u2net, u2netp, isnet-general-use)"),
    alpha_matting: bool = Form(default=False, description="Enable alpha matting for better edges"),
    alpha_matting_foreground_threshold: int = Form(default=240, ge=0, le=255),
    alpha_matting_background_threshold: int = Form(default=10, ge=0, le=255),
    api_key: Dict[str, Any] = Depends(get_api_key),
    job_store: JobStore = Depends(get_job_store),
    storage: StorageService = Depends(get_storage_service),
) -> JobResponse:
    """
    Remove backgrounds from uploaded images.

    - Upload 1-10 image files (jpg, png, webp, etc.)
    - Returns a job ID immediately
    - Poll /api/v1/jobs/{job_id} for status
    - Download results when complete
    """
    # Validate file count
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one file is required")

    if len(files) > settings.max_files_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_files_per_request} files per request"
        )

    # Generate job ID
    job_id = str(uuid.uuid4())

    try:
        # Save uploaded files
        input_paths, filenames = await storage.save_uploads(files, job_id)

        # Create job record
        job_data = job_store.create_job(
            job_id=job_id,
            job_type=JobType.REMBG,
            input_count=len(files),
            filenames=filenames,
            metadata={
                "model": model,
                "alpha_matting": alpha_matting,
            },
        )

        # Queue the task
        from api.tasks.rembg_tasks import process_rembg

        output_dir = str(storage.get_job_output_dir(job_id))

        task = process_rembg.delay(
            job_id=job_id,
            input_paths=[str(p) for p in input_paths],
            output_dir=output_dir,
            model=model,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
        )

        # Update job with Celery task ID
        job_store.update_job(job_id, celery_task_id=task.id)

        logger.info(f"Created RemBG job {job_id} with {len(files)} files")

        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            job_type=JobType.REMBG,
            created_at=datetime.fromisoformat(job_data["created_at"]),
            message=f"Processing {len(files)} image(s)",
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Clean up on error
        storage.cleanup_job(job_id)
        logger.error(f"Failed to create RemBG job: {e}")
        raise HTTPException(status_code=500, detail=str(e))
