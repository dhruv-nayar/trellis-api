"""
Celery tasks for TRELLIS image-to-3D conversion
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from celery import current_task
from api.tasks.celery_app import celery_app
from api.services.trellis_service import get_trellis_service
from api.services.job_store import get_job_store
from api.models.enums import JobStatus, TrellisBackend
from api.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="api.tasks.trellis_tasks.process_trellis",
    max_retries=settings.trellis_max_retries,
    default_retry_delay=60,
    soft_time_limit=settings.trellis_task_timeout,
    time_limit=settings.trellis_task_timeout + 120,
    acks_late=True,
)
def process_trellis(
    self,
    job_id: str,
    input_paths: List[str],
    output_path: str,
    backend: str = "huggingface",
    seed: int = 1,
    texture_size: int = 2048,
    optimize: bool = True,
) -> Dict[str, Any]:
    """
    Celery task to convert images to 3D GLB using TRELLIS.

    Args:
        job_id: Unique job identifier
        input_paths: List of input file paths (as strings)
        output_path: Output GLB file path (as string)
        backend: TRELLIS backend ("huggingface" or "runpod")
        seed: Random seed for reproducibility
        texture_size: Texture resolution
        optimize: Whether to optimize mesh

    Returns:
        Dict with task results
    """
    logger.info(f"Starting TRELLIS task for job {job_id} with {len(input_paths)} images")

    job_store = get_job_store()
    celery_task_id = current_task.request.id

    # Mark job as processing
    job_store.set_processing(job_id, celery_task_id)
    job_store.update_job(
        job_id,
        message="Connecting to TRELLIS model...",
        progress=10,
    )

    try:
        # Get TRELLIS service
        trellis_service = get_trellis_service()
        trellis_backend = TrellisBackend(backend)

        # Convert string paths to Path objects
        input_path_objects = [Path(p) for p in input_paths]
        output_path_obj = Path(output_path)

        # Update progress
        job_store.update_job(
            job_id,
            message=f"Processing {len(input_paths)} image(s) with TRELLIS...",
            progress=30,
        )
        self.update_state(
            state="PROCESSING",
            meta={"progress": 30, "message": "Running inference"},
        )

        # Process images
        result_path = trellis_service.process(
            image_paths=input_path_objects,
            output_path=output_path_obj,
            backend=trellis_backend,
            seed=seed,
            texture_size=texture_size,
            optimize=optimize,
        )

        # Generate download URL
        download_url = f"/api/v1/jobs/{job_id}/download/{result_path.name}"

        # Mark job as completed
        job_store.set_completed(
            job_id=job_id,
            output_count=1,
            download_urls=[download_url],
            message="Successfully generated 3D model",
        )

        result = {
            "job_id": job_id,
            "status": "completed",
            "output_path": str(result_path),
            "download_url": download_url,
        }

        logger.info(f"TRELLIS task completed for job {job_id}")
        return result

    except Exception as e:
        error_msg = str(e)
        logger.error(f"TRELLIS task failed for job {job_id}: {error_msg}")

        # Check if we should retry
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying TRELLIS task for job {job_id} (attempt {self.request.retries + 1})")
            job_store.update_job(
                job_id,
                message=f"Retrying... (attempt {self.request.retries + 2}/{self.max_retries + 1})",
            )
            raise self.retry(exc=e)

        # Mark job as failed
        job_store.set_failed(job_id, error_msg)

        return {
            "job_id": job_id,
            "status": "failed",
            "error": error_msg,
        }
