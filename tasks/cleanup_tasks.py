"""
Celery tasks for cleanup and maintenance
"""

import logging
from api.tasks.celery_app import celery_app
from api.services.storage import get_storage_service

logger = logging.getLogger(__name__)


@celery_app.task(
    name="api.tasks.cleanup_tasks.cleanup_expired_jobs",
    ignore_result=True,
)
def cleanup_expired_jobs():
    """
    Periodic task to clean up expired job files.

    This task runs hourly (configured in celery_app.py beat_schedule)
    and removes job directories older than cleanup_after_hours.
    """
    logger.info("Starting expired jobs cleanup")

    try:
        storage = get_storage_service()
        cleaned = storage.cleanup_old_jobs()

        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired job directories")
        else:
            logger.debug("No expired jobs to clean up")

        return {"cleaned": cleaned}

    except Exception as e:
        logger.error(f"Cleanup task failed: {e}")
        raise


@celery_app.task(
    name="api.tasks.cleanup_tasks.cleanup_specific_job",
    ignore_result=True,
)
def cleanup_specific_job(job_id: str):
    """
    Task to clean up a specific job's files.

    Can be used to schedule cleanup after a job completes or
    when a job is cancelled.

    Args:
        job_id: The job ID to clean up
    """
    logger.info(f"Cleaning up job {job_id}")

    try:
        storage = get_storage_service()
        cleaned = storage.cleanup_job(job_id)

        if cleaned:
            logger.info(f"Cleaned up job {job_id}")
        else:
            logger.debug(f"No files to clean up for job {job_id}")

        return {"job_id": job_id, "cleaned": cleaned}

    except Exception as e:
        logger.error(f"Failed to clean up job {job_id}: {e}")
        raise
