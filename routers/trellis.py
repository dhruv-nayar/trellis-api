"""
TRELLIS Router
Image to 3D conversion endpoint
"""

import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any
from fastapi import APIRouter, File, UploadFile, Depends, Form, HTTPException, Request

from api.models.responses import JobResponse
from api.models.enums import JobStatus, JobType, TrellisBackend
from api.services.job_store import JobStore, get_job_store
from api.services.storage import StorageService, get_storage_service
from api.middleware.auth import get_api_key
from api.middleware.rate_limit import limiter
from api.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trellis", tags=["Image to 3D"])


@router.post(
    "",
    response_model=JobResponse,
    summary="Convert images to 3D model",
    description="Upload 1+ images to generate a 3D GLB model. Multiple images enable multi-view reconstruction.",
)
@limiter.limit(settings.rate_limit_trellis)
async def convert_to_3d(
    request: Request,
    files: List[UploadFile] = File(..., description="Image files (1 or more for multi-view)"),
    seed: int = Form(default=1, ge=0, description="Random seed for reproducibility"),
    texture_size: int = Form(default=2048, ge=512, le=4096, description="Texture resolution"),
    optimize: bool = Form(default=True, description="Optimize/simplify the mesh"),
    backend: str = Form(default="huggingface", description="Backend: huggingface or runpod"),
    api_key: Dict[str, Any] = Depends(get_api_key),
    job_store: JobStore = Depends(get_job_store),
    storage: StorageService = Depends(get_storage_service),
) -> JobResponse:
    """
    Convert images to a 3D GLB model.

    - Single image: Standard image-to-3D conversion
    - Multiple images: Multi-view reconstruction (experimental, best with similar compositions)

    Returns a job ID immediately. Poll /api/v1/jobs/{job_id} for status.
    """
    # Validate file count
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one file is required")

    if len(files) > settings.max_files_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_files_per_request} files per request"
        )

    # Validate backend
    try:
        trellis_backend = TrellisBackend(backend)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid backend: {backend}. Use 'huggingface' or 'runpod'"
        )

    # Check RunPod availability if requested
    if trellis_backend == TrellisBackend.RUNPOD and not settings.runpod_endpoint:
        raise HTTPException(
            status_code=400,
            detail="RunPod backend not configured. Use 'huggingface' backend."
        )

    # Generate job ID
    job_id = str(uuid.uuid4())

    try:
        # Save uploaded files
        input_paths, filenames = await storage.save_uploads(files, job_id)

        # Create job record
        job_data = job_store.create_job(
            job_id=job_id,
            job_type=JobType.TRELLIS,
            input_count=len(files),
            filenames=filenames,
            metadata={
                "backend": backend,
                "seed": seed,
                "texture_size": texture_size,
                "optimize": optimize,
                "multi_view": len(files) > 1,
            },
        )

        # Queue the task
        from api.tasks.trellis_tasks import process_trellis

        output_dir = storage.get_job_output_dir(job_id)
        output_filename = f"{job_id}.glb"
        output_path = str(output_dir / output_filename)

        task = process_trellis.delay(
            job_id=job_id,
            input_paths=[str(p) for p in input_paths],
            output_path=output_path,
            backend=backend,
            seed=seed,
            texture_size=texture_size,
            optimize=optimize,
        )

        # Update job with Celery task ID
        job_store.update_job(job_id, celery_task_id=task.id)

        mode_str = "multi-view" if len(files) > 1 else "single image"
        logger.info(f"Created TRELLIS job {job_id} ({mode_str}, {len(files)} files)")

        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            job_type=JobType.TRELLIS,
            created_at=datetime.fromisoformat(job_data["created_at"]),
            message=f"Queued {mode_str} conversion ({len(files)} image(s))",
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Clean up on error
        storage.cleanup_job(job_id)
        logger.error(f"Failed to create TRELLIS job: {e}")
        raise HTTPException(status_code=500, detail=str(e))
