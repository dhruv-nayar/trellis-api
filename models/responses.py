"""
Response models for API endpoints
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from api.models.enums import JobStatus, JobType


class JobResponse(BaseModel):
    """Response when a job is created"""
    job_id: str = Field(..., description="Unique identifier for the job")
    status: JobStatus = Field(..., description="Current status of the job")
    job_type: JobType = Field(..., description="Type of job (rembg or trellis)")
    created_at: datetime = Field(..., description="Timestamp when job was created")
    message: Optional[str] = Field(None, description="Human-readable status message")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "job_type": "rembg",
                "created_at": "2024-02-15T10:30:00Z",
                "message": "Processing 3 image(s)"
            }
        }


class JobStatusResponse(BaseModel):
    """Detailed job status response"""
    job_id: str = Field(..., description="Unique identifier for the job")
    status: JobStatus = Field(..., description="Current status of the job")
    job_type: JobType = Field(..., description="Type of job (rembg or trellis)")
    created_at: datetime = Field(..., description="Timestamp when job was created")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    progress: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="Progress percentage (0-100)"
    )
    message: Optional[str] = Field(None, description="Human-readable status message")
    error: Optional[str] = Field(None, description="Error message if job failed")
    download_urls: Optional[List[str]] = Field(
        None,
        description="URLs to download results when completed"
    )
    input_count: Optional[int] = Field(None, description="Number of input files")
    output_count: Optional[int] = Field(None, description="Number of output files")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "job_type": "rembg",
                "created_at": "2024-02-15T10:30:00Z",
                "updated_at": "2024-02-15T10:30:45Z",
                "completed_at": "2024-02-15T10:30:45Z",
                "progress": 100,
                "message": "Successfully processed 3 images",
                "download_urls": [
                    "/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/download/image1_nobg.png",
                    "/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/download/image2_nobg.png",
                    "/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/download/image3_nobg.png"
                ],
                "input_count": 3,
                "output_count": 3
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    redis_connected: bool = Field(..., description="Redis connection status")
    celery_workers: int = Field(..., description="Number of active Celery workers")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "2.0.0",
                "redis_connected": True,
                "celery_workers": 2
            }
        }


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid file type",
                "detail": "Only image files (jpg, png, webp) are accepted"
            }
        }
