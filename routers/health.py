"""
Health Router
Health check and status endpoints
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends

from api.models.responses import HealthResponse
from api.services.job_store import JobStore, get_job_store
from api.services.storage import StorageService, get_storage_service
from api.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check API health including Redis and Celery status.",
)
async def health_check(
    job_store: JobStore = Depends(get_job_store),
) -> HealthResponse:
    """
    Health check endpoint.

    Returns:
    - API status
    - Redis connection status
    - Number of active Celery workers
    """
    # Check Redis connection
    redis_connected = job_store.health_check()

    # Check Celery workers
    celery_workers = 0
    try:
        from api.tasks.celery_app import celery_app
        inspect = celery_app.control.inspect()
        active = inspect.active()
        if active:
            celery_workers = len(active)
    except Exception as e:
        logger.warning(f"Could not inspect Celery workers: {e}")

    return HealthResponse(
        status="healthy" if redis_connected else "degraded",
        version=settings.app_version,
        redis_connected=redis_connected,
        celery_workers=celery_workers,
    )


@router.get(
    "/",
    summary="Root endpoint",
    description="API information and available endpoints.",
)
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "rembg": "/api/v1/rembg",
            "trellis": "/api/v1/trellis",
            "jobs": "/api/v1/jobs/{job_id}",
        },
        "documentation": "/docs",
    }


@router.get(
    "/stats",
    summary="Storage statistics",
    description="Get storage usage statistics.",
)
async def storage_stats(
    storage: StorageService = Depends(get_storage_service),
) -> Dict[str, Any]:
    """Get storage usage statistics."""
    return {
        "disk_usage": storage.get_disk_usage(),
        "cleanup_after_hours": settings.cleanup_after_hours,
    }
