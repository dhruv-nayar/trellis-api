"""
Redis-based job state management
"""

import json
import redis
from datetime import datetime
from typing import Optional, Dict, Any, List
from api.models.enums import JobStatus, JobType
from api.config import settings
import logging

logger = logging.getLogger(__name__)


class JobStore:
    """Manages job state in Redis"""

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.redis_url
        self._redis: Optional[redis.Redis] = None
        self.prefix = "job:"
        self.ttl = settings.cleanup_after_hours * 3600  # Convert hours to seconds

    @property
    def redis(self) -> redis.Redis:
        """Lazy Redis connection"""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=True)
        return self._redis

    def _key(self, job_id: str) -> str:
        """Generate Redis key for job"""
        return f"{self.prefix}{job_id}"

    def create_job(
        self,
        job_id: str,
        job_type: JobType,
        input_count: int,
        filenames: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a new job record"""
        now = datetime.utcnow().isoformat()
        job_data = {
            "job_id": job_id,
            "job_type": job_type.value,
            "status": JobStatus.PENDING.value,
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
            "progress": 0,
            "message": f"Queued for processing ({input_count} file(s))",
            "error": None,
            "input_count": input_count,
            "output_count": 0,
            "filenames": filenames,
            "download_urls": [],
            "celery_task_id": None,
            **(metadata or {}),
        }

        key = self._key(job_id)
        self.redis.setex(key, self.ttl, json.dumps(job_data))
        logger.info(f"Created job {job_id} of type {job_type.value}")

        return job_data

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job data by ID"""
        key = self._key(job_id)
        data = self.redis.get(key)
        if data:
            return json.loads(data)
        return None

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[int] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        output_count: Optional[int] = None,
        download_urls: Optional[List[str]] = None,
        celery_task_id: Optional[str] = None,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """Update job with new data"""
        job_data = self.get_job(job_id)
        if not job_data:
            logger.warning(f"Attempted to update non-existent job {job_id}")
            return None

        now = datetime.utcnow().isoformat()
        job_data["updated_at"] = now

        if status is not None:
            job_data["status"] = status.value
            if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                job_data["completed_at"] = now

        if progress is not None:
            job_data["progress"] = progress

        if message is not None:
            job_data["message"] = message

        if error is not None:
            job_data["error"] = error

        if output_count is not None:
            job_data["output_count"] = output_count

        if download_urls is not None:
            job_data["download_urls"] = download_urls

        if celery_task_id is not None:
            job_data["celery_task_id"] = celery_task_id

        # Apply any additional kwargs
        for key, value in kwargs.items():
            job_data[key] = value

        # Save updated job
        redis_key = self._key(job_id)
        self.redis.setex(redis_key, self.ttl, json.dumps(job_data))

        logger.debug(f"Updated job {job_id}: status={status}, progress={progress}")
        return job_data

    def set_processing(self, job_id: str, celery_task_id: str) -> Optional[Dict[str, Any]]:
        """Mark job as processing"""
        return self.update_job(
            job_id,
            status=JobStatus.PROCESSING,
            celery_task_id=celery_task_id,
            message="Processing started",
        )

    def set_completed(
        self,
        job_id: str,
        output_count: int,
        download_urls: List[str],
        message: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Mark job as completed"""
        return self.update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            output_count=output_count,
            download_urls=download_urls,
            message=message or f"Successfully processed {output_count} file(s)",
        )

    def set_failed(self, job_id: str, error: str) -> Optional[Dict[str, Any]]:
        """Mark job as failed"""
        return self.update_job(
            job_id,
            status=JobStatus.FAILED,
            error=error,
            message="Processing failed",
        )

    def set_cancelled(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Mark job as cancelled"""
        return self.update_job(
            job_id,
            status=JobStatus.CANCELLED,
            message="Job cancelled by user",
        )

    def delete_job(self, job_id: str) -> bool:
        """Delete job record"""
        key = self._key(job_id)
        result = self.redis.delete(key)
        if result:
            logger.info(f"Deleted job {job_id}")
        return result > 0

    def get_all_jobs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all jobs (for admin/debugging)"""
        pattern = f"{self.prefix}*"
        keys = self.redis.keys(pattern)[:limit]
        jobs = []
        for key in keys:
            data = self.redis.get(key)
            if data:
                jobs.append(json.loads(data))
        return jobs

    def health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            return self.redis.ping()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False


# Global instance for dependency injection
_job_store: Optional[JobStore] = None


def get_job_store() -> JobStore:
    """Get or create JobStore instance"""
    global _job_store
    if _job_store is None:
        _job_store = JobStore()
    return _job_store
