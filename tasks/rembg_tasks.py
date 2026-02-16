"""
Celery tasks for RemBG background removal
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from celery import current_task
from api.tasks.celery_app import celery_app
from api.services.rembg_service import get_rembg_service
from api.services.job_store import get_job_store
from api.models.enums import JobStatus
from api.config import settings

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    name="api.tasks.rembg_tasks.process_rembg",
    max_retries=settings.rembg_max_retries,
    default_retry_delay=30,
    soft_time_limit=settings.rembg_task_timeout,
    time_limit=settings.rembg_task_timeout + 60,
    acks_late=True,
)
def process_rembg(
    self,
    job_id: str,
    input_paths: List[str],
    output_dir: str,
    model: str = "u2net",
    alpha_matting: bool = False,
    alpha_matting_foreground_threshold: int = 240,
    alpha_matting_background_threshold: int = 10,
) -> Dict[str, Any]:
    """
    Celery task to remove backgrounds from images.

    Args:
        job_id: Unique job identifier
        input_paths: List of input file paths (as strings)
        output_dir: Output directory path (as string)
        model: RemBG model name
        alpha_matting: Enable alpha matting
        alpha_matting_foreground_threshold: Foreground threshold
        alpha_matting_background_threshold: Background threshold

    Returns:
        Dict with task results
    """
    logger.info(f"Starting RemBG task for job {job_id} with {len(input_paths)} images")

    job_store = get_job_store()
    celery_task_id = current_task.request.id

    # Mark job as processing
    job_store.set_processing(job_id, celery_task_id)

    try:
        # Get RemBG service
        rembg_service = get_rembg_service(model_name=model)

        # Convert string paths to Path objects
        input_path_objects = [Path(p) for p in input_paths]
        output_dir_path = Path(output_dir)

        # Progress callback to update job status
        def update_progress(current: int, total: int):
            progress = int((current / total) * 100)
            job_store.update_job(
                job_id,
                progress=progress,
                message=f"Processing image {current}/{total}",
            )
            # Also update Celery task state
            self.update_state(
                state="PROCESSING",
                meta={"progress": progress, "current": current, "total": total},
            )

        # Process images
        output_paths = rembg_service.process_batch(
            input_paths=input_path_objects,
            output_dir=output_dir_path,
            alpha_matting=alpha_matting,
            alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
            alpha_matting_background_threshold=alpha_matting_background_threshold,
            progress_callback=update_progress,
        )

        # Generate download URLs
        download_urls = [
            f"/api/v1/jobs/{job_id}/download/{path.name}"
            for path in output_paths
        ]

        # Mark job as completed
        job_store.set_completed(
            job_id=job_id,
            output_count=len(output_paths),
            download_urls=download_urls,
            message=f"Successfully removed backgrounds from {len(output_paths)} image(s)",
        )

        result = {
            "job_id": job_id,
            "status": "completed",
            "output_count": len(output_paths),
            "output_paths": [str(p) for p in output_paths],
            "download_urls": download_urls,
        }

        logger.info(f"RemBG task completed for job {job_id}: {len(output_paths)} outputs")
        return result

    except Exception as e:
        error_msg = str(e)
        logger.error(f"RemBG task failed for job {job_id}: {error_msg}")

        # Check if we should retry
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying RemBG task for job {job_id} (attempt {self.request.retries + 1})")
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
