"""
Tests for JobStore service
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch

from api.models.enums import JobStatus, JobType


class TestJobStoreCreate:
    """Tests for job creation"""

    def test_create_job_success(self, mock_job_store):
        """Test creating a job with valid params"""
        job_data = mock_job_store.create_job(
            job_id="test-123",
            job_type=JobType.REMBG,
            input_count=2,
            filenames=["image1.jpg", "image2.png"],
        )

        assert job_data["job_id"] == "test-123"
        assert job_data["job_type"] == "rembg"
        assert job_data["status"] == "pending"
        assert job_data["input_count"] == 2
        assert job_data["output_count"] == 0
        assert job_data["filenames"] == ["image1.jpg", "image2.png"]
        assert job_data["download_urls"] == []
        assert job_data["error"] is None

    def test_create_job_sets_timestamps(self, mock_job_store):
        """Test that job creation sets timestamps"""
        job_data = mock_job_store.create_job(
            job_id="test-456",
            job_type=JobType.TRELLIS,
            input_count=1,
            filenames=["model.jpg"],
        )

        assert "created_at" in job_data
        assert "updated_at" in job_data
        assert job_data["completed_at"] is None
        # Timestamps should be ISO format
        datetime.fromisoformat(job_data["created_at"])
        datetime.fromisoformat(job_data["updated_at"])

    def test_create_job_with_metadata(self, mock_job_store):
        """Test creating job with additional metadata"""
        job_data = mock_job_store.create_job(
            job_id="test-789",
            job_type=JobType.TRELLIS,
            input_count=1,
            filenames=["test.jpg"],
            metadata={"seed": 42, "backend": "huggingface"},
        )

        assert job_data["seed"] == 42
        assert job_data["backend"] == "huggingface"

    def test_create_job_stores_in_redis(self, mock_job_store, mock_redis_connection):
        """Test that job is stored in Redis"""
        mock_job_store.create_job(
            job_id="test-store",
            job_type=JobType.REMBG,
            input_count=1,
            filenames=["test.jpg"],
        )

        # Verify job exists in Redis
        stored = mock_redis_connection.get("job:test-store")
        assert stored is not None
        data = json.loads(stored)
        assert data["job_id"] == "test-store"


class TestJobStoreGet:
    """Tests for job retrieval"""

    def test_get_job_exists(self, mock_job_store):
        """Test retrieving an existing job"""
        # Create job first
        mock_job_store.create_job(
            job_id="get-test",
            job_type=JobType.REMBG,
            input_count=1,
            filenames=["test.jpg"],
        )

        # Retrieve it
        job = mock_job_store.get_job("get-test")
        assert job is not None
        assert job["job_id"] == "get-test"
        assert job["job_type"] == "rembg"

    def test_get_job_not_found(self, mock_job_store):
        """Test retrieving non-existent job returns None"""
        job = mock_job_store.get_job("non-existent-job")
        assert job is None


class TestJobStoreUpdate:
    """Tests for job updates"""

    def test_update_job_status(self, mock_job_store):
        """Test updating job status"""
        mock_job_store.create_job(
            job_id="update-status",
            job_type=JobType.REMBG,
            input_count=1,
            filenames=["test.jpg"],
        )

        updated = mock_job_store.update_job(
            "update-status",
            status=JobStatus.PROCESSING,
        )

        assert updated["status"] == "processing"

    def test_update_job_progress(self, mock_job_store):
        """Test updating job progress"""
        mock_job_store.create_job(
            job_id="update-progress",
            job_type=JobType.REMBG,
            input_count=2,
            filenames=["test1.jpg", "test2.jpg"],
        )

        updated = mock_job_store.update_job(
            "update-progress",
            progress=50,
            message="Processing 1 of 2",
        )

        assert updated["progress"] == 50
        assert updated["message"] == "Processing 1 of 2"

    def test_update_job_updates_timestamp(self, mock_job_store):
        """Test that update modifies updated_at timestamp"""
        mock_job_store.create_job(
            job_id="update-time",
            job_type=JobType.REMBG,
            input_count=1,
            filenames=["test.jpg"],
        )

        job_before = mock_job_store.get_job("update-time")
        original_updated = job_before["updated_at"]

        # Update the job
        mock_job_store.update_job("update-time", progress=50)
        job_after = mock_job_store.get_job("update-time")

        # updated_at should change (or be same if too fast)
        assert job_after["updated_at"] >= original_updated

    def test_update_nonexistent_job(self, mock_job_store):
        """Test updating non-existent job returns None"""
        result = mock_job_store.update_job(
            "does-not-exist",
            status=JobStatus.PROCESSING,
        )
        assert result is None

    def test_update_job_with_kwargs(self, mock_job_store):
        """Test updating job with additional kwargs"""
        mock_job_store.create_job(
            job_id="update-kwargs",
            job_type=JobType.REMBG,
            input_count=1,
            filenames=["test.jpg"],
        )

        updated = mock_job_store.update_job(
            "update-kwargs",
            custom_field="custom_value",
        )

        assert updated["custom_field"] == "custom_value"


class TestJobStoreStatusTransitions:
    """Tests for status transition helper methods"""

    def test_set_processing(self, mock_job_store):
        """Test set_processing sets correct status and celery_task_id"""
        mock_job_store.create_job(
            job_id="proc-test",
            job_type=JobType.REMBG,
            input_count=1,
            filenames=["test.jpg"],
        )

        result = mock_job_store.set_processing("proc-test", "celery-123")

        assert result["status"] == "processing"
        assert result["celery_task_id"] == "celery-123"
        assert result["message"] == "Processing started"

    def test_set_completed_adds_download_urls(self, mock_job_store):
        """Test set_completed adds download URLs and sets progress to 100"""
        mock_job_store.create_job(
            job_id="complete-test",
            job_type=JobType.REMBG,
            input_count=2,
            filenames=["test1.jpg", "test2.jpg"],
        )

        result = mock_job_store.set_completed(
            "complete-test",
            output_count=2,
            download_urls=[
                "/api/v1/jobs/complete-test/download/test1_nobg.png",
                "/api/v1/jobs/complete-test/download/test2_nobg.png",
            ],
        )

        assert result["status"] == "completed"
        assert result["progress"] == 100
        assert result["output_count"] == 2
        assert len(result["download_urls"]) == 2
        assert result["completed_at"] is not None

    def test_set_completed_with_custom_message(self, mock_job_store):
        """Test set_completed with custom message"""
        mock_job_store.create_job(
            job_id="complete-msg",
            job_type=JobType.TRELLIS,
            input_count=1,
            filenames=["test.jpg"],
        )

        result = mock_job_store.set_completed(
            "complete-msg",
            output_count=1,
            download_urls=["/download/output.glb"],
            message="3D model generated successfully",
        )

        assert result["message"] == "3D model generated successfully"

    def test_set_failed_records_error(self, mock_job_store):
        """Test set_failed records error message"""
        mock_job_store.create_job(
            job_id="fail-test",
            job_type=JobType.REMBG,
            input_count=1,
            filenames=["test.jpg"],
        )

        result = mock_job_store.set_failed(
            "fail-test",
            error="Failed to process image: unsupported format",
        )

        assert result["status"] == "failed"
        assert result["error"] == "Failed to process image: unsupported format"
        assert result["message"] == "Processing failed"
        assert result["completed_at"] is not None

    def test_set_cancelled(self, mock_job_store):
        """Test set_cancelled sets correct status"""
        mock_job_store.create_job(
            job_id="cancel-test",
            job_type=JobType.REMBG,
            input_count=1,
            filenames=["test.jpg"],
        )

        result = mock_job_store.set_cancelled("cancel-test")

        assert result["status"] == "cancelled"
        assert result["message"] == "Job cancelled by user"
        assert result["completed_at"] is not None


class TestJobStoreDelete:
    """Tests for job deletion"""

    def test_delete_job_removes_from_redis(self, mock_job_store, mock_redis_connection):
        """Test that deletion removes job from Redis"""
        mock_job_store.create_job(
            job_id="delete-test",
            job_type=JobType.REMBG,
            input_count=1,
            filenames=["test.jpg"],
        )

        # Verify exists
        assert mock_redis_connection.get("job:delete-test") is not None

        # Delete
        result = mock_job_store.delete_job("delete-test")
        assert result is True

        # Verify gone
        assert mock_redis_connection.get("job:delete-test") is None

    def test_delete_nonexistent_job(self, mock_job_store):
        """Test deleting non-existent job returns False"""
        result = mock_job_store.delete_job("never-existed")
        assert result is False


class TestJobStoreGetAll:
    """Tests for get_all_jobs"""

    def test_get_all_jobs(self, mock_job_store):
        """Test retrieving all jobs"""
        # Create multiple jobs
        for i in range(3):
            mock_job_store.create_job(
                job_id=f"all-jobs-{i}",
                job_type=JobType.REMBG,
                input_count=1,
                filenames=[f"test{i}.jpg"],
            )

        jobs = mock_job_store.get_all_jobs()
        assert len(jobs) >= 3

    def test_get_all_jobs_with_limit(self, mock_job_store):
        """Test get_all_jobs respects limit"""
        # Create 5 jobs
        for i in range(5):
            mock_job_store.create_job(
                job_id=f"limited-{i}",
                job_type=JobType.REMBG,
                input_count=1,
                filenames=[f"test{i}.jpg"],
            )

        jobs = mock_job_store.get_all_jobs(limit=3)
        assert len(jobs) <= 3


class TestJobStoreHealth:
    """Tests for health check"""

    def test_health_check_redis_connected(self, mock_job_store):
        """Test health check returns True when connected"""
        result = mock_job_store.health_check()
        assert result is True

    def test_health_check_redis_down(self, mock_job_store, monkeypatch):
        """Test health check returns False when Redis is down"""
        def mock_ping():
            raise ConnectionError("Redis unavailable")

        monkeypatch.setattr(mock_job_store.redis, "ping", mock_ping)

        result = mock_job_store.health_check()
        assert result is False


class TestGetJobStore:
    """Tests for get_job_store dependency"""

    def test_get_job_store_returns_instance(self, mock_redis_connection):
        """Test get_job_store returns JobStore instance"""
        from api.services.job_store import get_job_store, _job_store
        import api.services.job_store as job_store_module

        # Reset global
        job_store_module._job_store = None

        store = get_job_store()
        assert store is not None

        # Should return same instance
        store2 = get_job_store()
        assert store is store2

        # Reset for other tests
        job_store_module._job_store = None
