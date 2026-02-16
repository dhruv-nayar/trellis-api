"""
FastAPI Dependencies
"""

from typing import Dict, Any
from fastapi import Depends
from api.middleware.auth import get_api_key
from api.services.job_store import JobStore, get_job_store
from api.services.storage import StorageService, get_storage_service

__all__ = [
    "get_api_key",
    "get_job_store",
    "get_storage_service",
    "get_authenticated_job_store",
]


async def get_authenticated_job_store(
    api_key: Dict[str, Any] = Depends(get_api_key),
    job_store: JobStore = Depends(get_job_store),
) -> JobStore:
    """
    Dependency that requires authentication and provides job store.
    Use this for endpoints that need both auth and job store.
    """
    return job_store
