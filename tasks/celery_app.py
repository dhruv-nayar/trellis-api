"""
Celery application configuration
"""

from celery import Celery
import os

# Get broker and backend URLs from environment
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

# Create Celery app
celery_app = Celery(
    "trellis_api",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "api.tasks.rembg_tasks",
        "api.tasks.trellis_tasks",
        "api.tasks.cleanup_tasks",
    ]
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Task routing - separate queues for different task types
    task_routes={
        "api.tasks.rembg_tasks.*": {"queue": "rembg"},
        "api.tasks.trellis_tasks.*": {"queue": "trellis"},
        "api.tasks.cleanup_tasks.*": {"queue": "cleanup"},
    },

    # Default queue for unrouted tasks
    task_default_queue="default",

    # Acknowledgement settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Result expiration (24 hours)
    result_expires=86400,

    # Worker settings
    worker_prefetch_multiplier=1,  # One task at a time for GPU-bound tasks

    # Task tracking
    task_track_started=True,

    # Beat scheduler for periodic tasks
    beat_schedule={
        "cleanup-expired-jobs": {
            "task": "api.tasks.cleanup_tasks.cleanup_expired_jobs",
            "schedule": 3600.0,  # Every hour
        },
    },
)


# Task state custom states
class TaskState:
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
